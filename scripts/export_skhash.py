#!/usr/bin/env python
"""Export btpc polarity predictions into a ready-to-run SKHASH directory.

Reads ``predictions_anchor_mapped.csv`` from ``btpc-predict``, applies the
paper's MC/MCM rejection (reject when mean confidence < mc-threshold AND mean
center margin < mcm-threshold, i.e. ``--reject-strategy and``), and writes the
SKHASH first-motion inputs:

- ``input/sk_sta.csv``      network,station,channel,latitude,longitude,elevation
- ``input/sk_ctlg.csv``     time,latitude,longitude,depth,mag,event_id
- ``input/polarities.csv``  event_id,network,station,channel,p_polarity (+1 up / -1 down)
- ``control_auto.txt``      SKHASH control file with the paper's settings
- ``input/predictions_for_skhash.csv`` / ``input/rejected_predictions.csv``
- ``export_summary.json``

Event coordinates come from an events CSV, station coordinates from a
stations CSV, and the record-to-event/station mapping from the phasenet group
of the waveform HDF5 built by ``make_pwave_dataset.py`` (so the ``event_id``
and ``station`` columns of the pick table are carried through). When one
event was picked on one station more than once, the pick with the highest
mean confidence is kept.

Example:

    python scripts/export_skhash.py \
        --predictions runs/mine/stage2/predict_out/predictions_anchor_mapped.csv \
        --data-path my_data.h5 --events events.csv --stations stations.csv \
        --vmodel vel.txt --output-dir skhash_run

    SKHASH skhash_run/control_auto.txt        # then run SKHASH itself

Events CSV columns: ``event_id,lat,lon,dep,mag,time`` (ISO time; ``depth``
is accepted for ``dep``, ``ot`` for ``time``). Stations CSV columns:
``station`` (``NET.STA`` or bare code; ``network`` column optional),
``lat,lon,ele`` (``elevation`` accepted). The velocity model is the SKHASH
1-D format ``depth_km,Vp_km_s`` (see docs/skhash.md).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

PRED_REQUIRED = ("record_id", "snr", "pred_polarity", "mean_confidence", "mean_center_margin")

EVENT_DEPTH_ALIASES = ("dep", "depth")
EVENT_TIME_ALIASES = ("time", "ot")
STATION_ELE_ALIASES = ("ele", "elevation")


def _pick_column(fieldnames, candidates, what):
    for name in candidates:
        if name in fieldnames:
            return name
    raise SystemExit(f"CSV needs a {what} column; one of {list(candidates)}")


def _column(row: dict, names, what: str):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    raise SystemExit(f"Missing {what} column (looked for {list(names)})")


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def parse_iso_time(value: str) -> str:
    """Normalize an ISO timestamp to SKHASH's %Y-%m-%dT%H:%M:%S."""
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise SystemExit(f"Cannot parse event time {value!r}; use ISO format, e.g. 2019-07-04T17:35:01")


def read_predictions(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in PRED_REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} missing columns: {missing}")
        for row in reader:
            try:
                rows.append({
                    "record_id": int(float(row["record_id"])),
                    "snr": float(row["snr"]),
                    "pred": int(float(row["pred_polarity"])),
                    "mc": float(row["mean_confidence"]),
                    "mcm": float(row["mean_center_margin"]),
                })
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"No usable rows in {csv_path}")
    return rows


def read_record_mapping(data_path: str):
    """record_id -> (event_id, station) from the phasenet group."""
    with h5py.File(data_path, "r") as handle:
        if "phasenet" not in handle:
            raise SystemExit(f"{data_path} has no 'phasenet' group (build it with make_pwave_dataset.py)")
        group = handle["phasenet"]
        for key in ("record_id", "event_id", "station"):
            if key not in group:
                raise SystemExit(f"{data_path} is missing phasenet/{key}")
        record_ids = np.asarray(group["record_id"][:]).reshape(-1)
        event_ids = np.asarray(group["event_id"][:]).reshape(-1)
        stations = [_decode(v) for v in np.asarray(group["station"][:]).reshape(-1)]
    return {int(rid): (int(eid), sta) for rid, eid, sta in zip(record_ids, event_ids, stations)}


def read_events_csv(csv_path: str):
    events = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _pick_column(reader.fieldnames or [], ("event_id",), "event_id")
        for row in reader:
            eid = str(_column(row, ("event_id",), "event_id")).strip()
            events[eid] = {
                "latitude": float(_column(row, ("lat", "latitude"), "latitude")),
                "longitude": float(_column(row, ("lon", "longitude"), "longitude")),
                "depth": float(_column(row, EVENT_DEPTH_ALIASES, "depth")),
                "mag": float(row.get("mag") or 0.0),
                "time": parse_iso_time(_column(row, EVENT_TIME_ALIASES, "event time")),
            }
    if not events:
        raise SystemExit(f"No events in {csv_path}")
    return events


def read_stations_csv(csv_path: str):
    stations = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _pick_column(reader.fieldnames or [], ("station", "sta"), "station")
        for row in reader:
            code = str(_column(row, ("station", "sta"), "station")).strip()
            network = str(row.get("network") or "").strip()
            parts = [p for p in code.split(".") if p]
            if len(parts) >= 2:
                network, station = parts[0], parts[1]
            else:
                station = parts[0] if parts else code
                network = network or "--"
            stations[f"{network}.{station}"] = {
                "network": network,
                "station": station,
                "latitude": float(_column(row, ("lat", "latitude"), "latitude")),
                "longitude": float(_column(row, ("lon", "longitude"), "longitude")),
                "elevation": float(_column(row, STATION_ELE_ALIASES, "elevation")),
            }
    if not stations:
        raise SystemExit(f"No stations in {csv_path}")
    return stations


def write_csv(path: Path, header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def write_control_file(path: Path, args, input_dir: Path, output_dir: Path):
    text = f"""## Control file

$use_fortran
False

$num_cpus
{args.num_cpus}

$require_location_match
False

$stfile        # station list filepath
{(input_dir / "sk_sta.csv").as_posix()}

$catfile       # earthquake catalog filepath
{(input_dir / "sk_ctlg.csv").as_posix()}

$fpfile        # P-polarity input filepath
{(input_dir / "polarities.csv").as_posix()}

$outfile1      # focal mechanisms output filepath
{(output_dir / "out.csv").as_posix()}

$outfile_pol_agree  # record of polarity (dis)agreeement output filepath
{(output_dir / "polagree.csv").as_posix()}

$outfile_pol_info
{(output_dir / "polinfo.csv").as_posix()}

$outfolder_plots
{(output_dir / "figure").as_posix()}

$npolmin       # mininum number of polarity data (e.g., 8)
{args.npolmin}

$min_polarity_weight  # Any polarities with a abs(weight) < min_polarity_weight will be ignored
{args.min_polarity_weight}

$nmc           # number of trials (e.g., 30)
{args.nmc}

$maxout        # max num of acceptable focal mech. outputs (e.g., 500)
{args.maxout}

$ratmin        # minimum allowed signal to noise ratio
{args.ratmin}

$badfrac       # fraction polarities assumed bad
{args.badfrac}

$qbadfrac      # assumed noise in amplitude ratios, log10 (e.g. 0.3 for a factor of 2)
{args.qbadfrac}

$delmax        # maximum allowed source-receiver distance in km.
{args.delmax}

$prob_max      # probability threshold for multiples (e.g., 0.1)
{args.prob_max}

$output_angle_precision
4

$vmodel_paths
{Path(args.vmodel).resolve().as_posix()}
"""
    path.write_text(text, encoding="utf-8")


def select_picks(predictions, reject_mask, mapping, events, stations, args, dropped):
    """Keep accepted binary picks with resolvable event/station; best pick per event-station."""
    best_pick = {}
    for pred, is_reject in zip(predictions, reject_mask):
        record_id = pred["record_id"]
        if is_reject:
            continue
        if pred["pred"] not in (0, 1):
            dropped["uncertain_polarity"] += 1
            continue
        if pred["snr"] < args.min_snr:
            dropped["below_min_snr"] += 1
            continue
        entry = mapping.get(record_id)
        if entry is None:
            dropped["unknown_record"] += 1
            continue
        event_id, station = entry
        if int(event_id) < 0:
            dropped["missing_event_id"] += 1
            continue
        event = events.get(str(event_id))
        if event is None:
            dropped["missing_event_row"] += 1
            continue
        sta = stations.get(station)
        if sta is None:
            bare = {k.split(".")[-1]: v for k, v in stations.items()}
            sta = bare.get(station.split(".")[-1])
        if sta is None:
            dropped["missing_station_row"] += 1
            continue
        key = (str(event_id), sta["network"], sta["station"])
        if key in best_pick:
            dropped["duplicate_station_pick"] += 1
            if pred["mc"] > best_pick[key]["row"]["mc"]:
                best_pick[key] = {"row": pred, "event": event, "sta": sta, "event_id": str(event_id)}
        else:
            best_pick[key] = {"row": pred, "event": event, "sta": sta, "event_id": str(event_id)}
    return sorted(best_pick.values(), key=lambda p: (p["event_id"], p["sta"]["network"], p["sta"]["station"]))


def main():
    parser = argparse.ArgumentParser(
        description="Export btpc predictions to SKHASH first-motion inputs."
    )
    parser.add_argument("--predictions", required=True, help="predictions_anchor_mapped.csv from btpc-predict")
    parser.add_argument("--data-path", required=True, help="waveform HDF5 (phasenet group) for record metadata")
    parser.add_argument("--events", required=True, help="events CSV: event_id,lat,lon,dep,mag,time")
    parser.add_argument("--stations", required=True, help="stations CSV: station(NET.STA),lat,lon,ele")
    parser.add_argument("--vmodel", required=True, help="1-D velocity model (depth_km,Vp_km_s)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mc-threshold", type=float, default=0.8, help="reject when mean_confidence below this")
    parser.add_argument("--mcm-threshold", type=float, default=0.05, help="reject when mean_center_margin below this")
    parser.add_argument("--reject-strategy", choices=["and", "or"], default="and",
                        help="and: reject only when both thresholds fail (paper default); or: either")
    parser.add_argument("--min-snr", type=float, default=0.0, help="drop accepted picks below this SNR")
    parser.add_argument("--channel", default="HHZ", help="channel code written to the SKHASH inputs")
    # SKHASH control-file settings (defaults = paper values)
    parser.add_argument("--npolmin", type=int, default=12)
    parser.add_argument("--min-polarity-weight", type=float, default=0.1)
    parser.add_argument("--nmc", type=int, default=30)
    parser.add_argument("--maxout", type=int, default=500)
    parser.add_argument("--ratmin", type=float, default=3.0)
    parser.add_argument("--badfrac", type=float, default=0.0)
    parser.add_argument("--qbadfrac", type=float, default=0.3)
    parser.add_argument("--delmax", type=float, default=120.0)
    parser.add_argument("--prob-max", type=float, default=0.1)
    parser.add_argument("--num-cpus", type=int, default=5)
    args = parser.parse_args()

    predictions = read_predictions(args.predictions)
    mapping = read_record_mapping(args.data_path)
    events = read_events_csv(args.events)
    stations = read_stations_csv(args.stations)

    mc_fail = np.array([p["mc"] < args.mc_threshold for p in predictions])
    mcm_fail = np.array([p["mcm"] < args.mcm_threshold for p in predictions])
    reject_mask = mc_fail & mcm_fail if args.reject_strategy == "and" else mc_fail | mcm_fail

    dropped = {
        "uncertain_polarity": 0, "unknown_record": 0, "missing_event_id": 0,
        "below_min_snr": 0, "missing_event_row": 0, "missing_station_row": 0,
        "duplicate_station_pick": 0,
    }
    picks = select_picks(predictions, reject_mask, mapping, events, stations, args, dropped)
    rejected_rows = [
        {"record_id": p["record_id"], "snr": p["snr"], "pred": p["pred"], "mc": p["mc"], "mcm": p["mcm"]}
        for p, is_reject in zip(predictions, reject_mask) if is_reject
    ]

    out_dir = Path(args.output_dir)
    input_dir = out_dir / "input"
    output_dir = out_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(input_dir / "predictions_for_skhash.csv",
              ["record_id", "snr", "pred", "mc", "mcm"],
              [[p["row"][k] for k in ("record_id", "snr", "pred", "mc", "mcm")] for p in picks])
    write_csv(input_dir / "rejected_predictions.csv",
              ["record_id", "snr", "pred", "mc", "mcm", "is_reject"],
              [[r[k] for k in ("record_id", "snr", "pred", "mc", "mcm")] + [1] for r in rejected_rows])

    used_stations = {f"{p['sta']['network']}.{p['sta']['station']}": p["sta"] for p in picks}
    sta_rows = sorted(used_stations.values(), key=lambda s: (s["network"], s["station"]))
    write_csv(input_dir / "sk_sta.csv",
              ["network", "station", "channel", "latitude", "longitude", "elevation"],
              [[s["network"], s["station"], args.channel, s["latitude"], s["longitude"], s["elevation"]] for s in sta_rows])

    used_events = {p["event_id"]: p["event"] for p in picks}
    evt_rows = sorted(used_events.items(), key=lambda kv: kv[0])
    write_csv(input_dir / "sk_ctlg.csv",
              ["time", "latitude", "longitude", "depth", "mag", "event_id"],
              [[e["time"], e["latitude"], e["longitude"], e["depth"], e["mag"], eid] for eid, e in evt_rows])

    write_csv(input_dir / "polarities.csv",
              ["event_id", "network", "station", "channel", "p_polarity"],
              [[p["event_id"], p["sta"]["network"], p["sta"]["station"], args.channel,
                1.0 if p["row"]["pred"] == 0 else -1.0] for p in picks])

    write_control_file(out_dir / "control_auto.txt", args, input_dir, output_dir)

    summary = {
        "mc_threshold": args.mc_threshold,
        "mcm_threshold": args.mcm_threshold,
        "reject_strategy": args.reject_strategy,
        "min_snr": args.min_snr,
        "total_predictions": len(predictions),
        "rejected": len(rejected_rows),
        "polarities_exported": len(picks),
        "events_exported": len(used_events),
        "stations_exported": len(used_stations),
        "dropped": dropped,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Exported SKHASH inputs to {out_dir}")
    print(f"polarities: {len(picks)} | events: {len(used_events)} | stations: {len(used_stations)} | rejected: {len(rejected_rows)}")
    if dropped["duplicate_station_pick"]:
        print(f"kept the higher-confidence pick for {dropped['duplicate_station_pick']} duplicate event-station pairs")
    for key, count in dropped.items():
        if count and key != "duplicate_station_pick":
            print(f"dropped {key}: {count}")
    print(f"next: SKHASH {out_dir / 'control_auto.txt'}")


if __name__ == "__main__":
    main()
