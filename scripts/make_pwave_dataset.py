#!/usr/bin/env python
"""Build a BTPC-ready HDF5 from P-wave picks and event waveforms.

Input is a pick table plus miniseed/SAC waveforms; output is the
``phasenet``-group layout that ``btpc-train --dataset-source unlabeled``
and ``btpc-predict --target-source ridge`` read directly:

- ``phasenet/waveforms``  (n, pre_sec + post_sec) * fs + 1) float32, Z component,
  bandpass-filtered, P arrival at sample index ``pre_sec * fs`` (= 1000 for the
  defaults)
- ``phasenet/snr``        SNR = max|P..P+0.5s| / max|P-0.5s..P| on the filtered
  waveform (the paper's definition)
- ``phasenet/record_id``  int32 ids
- ``phasenet/station``, ``phasenet/pick_time``, optional ``event_id`` metadata

Two pick sources:

1. ``--picks picks.csv`` with columns ``waveform_path,p_time`` (any time string
   UTCDateTime understands) plus optional ``record_id,event_id,station,channel``.
2. ``--sac-dir DIR --sac-pick-field a``: every ``*.sac`` file under DIR is read
   and its P pick taken from that SAC header field (relative seconds).

Example:

    python scripts/make_pwave_dataset.py --picks picks.csv --output my_data.h5

    python scripts/make_pwave_dataset.py --sac-dir sac/ --sac-pick-field a \
        --output my_data.h5

Then train and predict on your own region:

    btpc-train stage1 --dataset-source unlabeled --data-path my_data.h5 \
        --num-used 0 --save-path runs/mine/stage1
    btpc-train stage2 --stage1-dir runs/mine/stage1
    btpc-predict --stage2-checkpoint runs/mine/stage2/stage2_cluster_classifier.pth \
        --target-source ridge --data-path my_data.h5 \
        --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz

Requires obspy (``pip install obspy``); it is not a core btpc dependency.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt

try:
    from obspy import UTCDateTime, read
except ImportError:  # pragma: no cover
    sys.exit("obspy is required for waveform reading: pip install obspy")


PICKS_PATH_COLUMNS = ("waveform_path", "file", "path")
PICKS_TIME_COLUMNS = ("p_time", "pick_time", "phase_time", "time")


def _pick_column(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def read_picks_csv(csv_path: str):
    """Yield one dict per pick row with path / p_time / optional metadata."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        path_col = _pick_column(fieldnames, PICKS_PATH_COLUMNS)
        time_col = _pick_column(fieldnames, PICKS_TIME_COLUMNS)
        if not path_col or not time_col:
            raise SystemExit(
                f"{csv_path} needs a waveform-path column (one of "
                f"{PICKS_PATH_COLUMNS}) and a pick-time column (one of "
                f"{PICKS_TIME_COLUMNS}); found columns: {fieldnames}"
            )
        for row in reader:
            path = (row[path_col] or "").strip()
            time_str = (row[time_col] or "").strip()
            if not path or not time_str:
                continue
            yield {
                "waveform_path": path,
                "p_time": UTCDateTime(time_str),
                "record_id": row.get("record_id") or None,
                "event_id": row.get("event_id") or None,
                "station": (row.get("station") or "").strip() or None,
                "channel": (row.get("channel") or "").strip() or None,
            }


def iter_sac_picks(sac_dir: str, pick_field: str):
    """Yield picks read from SAC headers (pick = field value in seconds)."""
    for path in sorted(Path(sac_dir).rglob("*.sac")):
        try:
            st = read(str(path), headonly=True)
        except Exception:
            continue
        for tr in st:
            value = getattr(tr.stats.sac, pick_field, None)
            if value is None or not np.isfinite(value):
                continue
            yield {
                "waveform_path": str(path),
                "p_time": tr.stats.starttime + float(value),
                "record_id": None,
                "event_id": None,
                "station": f"{tr.stats.network}.{tr.stats.station}",
                "channel": None,
            }


def choose_z_trace(st, channel: str | None):
    """Pick the vertical trace: named channel, else HHZ > BHZ > EHZ > *Z."""

    def priority(code: str) -> int:
        for rank, name in enumerate(("HHZ", "BHZ", "EHZ")):
            if code == name:
                return rank
        return len(("HHZ", "BHZ", "EHZ"))

    if channel:
        named = [tr for tr in st if str(tr.stats.channel) == channel]
        if named:
            return named[0]
    z_traces = [tr for tr in st if str(getattr(tr.stats, "channel", "")).endswith("Z")]
    if not z_traces:
        return None
    z_traces.sort(key=lambda tr: (priority(str(tr.stats.channel)), -tr.stats.npts))
    return z_traces[0]


def bandpass(data: np.ndarray, fs: float, freqmin: float, freqmax: float, order: int):
    nyq = 0.5 * fs
    low = freqmin / nyq
    high = min(freqmax / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def snr_around_p(data: np.ndarray, p_idx: int, fs: float, half_sec: float = 0.5):
    """SNR = max|P..P+half_sec| / max|P-half_sec..P| (paper definition)."""
    half = int(round(half_sec * fs))
    if half <= 0 or p_idx < half or p_idx + half > len(data):
        return float("nan")
    noise_max = float(np.max(np.abs(data[p_idx - half : p_idx])))
    signal_max = float(np.max(np.abs(data[p_idx : p_idx + half])))
    if noise_max <= 0:
        return float("nan")
    return signal_max / noise_max


def process_pick(pick, args, sample_len: int, p_idx: int):
    """Return (waveform, snr, station, None) or (None, None, None, failure_reason)."""
    try:
        st = read(pick["waveform_path"])
    except Exception:
        return None, None, None, "read_error"

    start = pick["p_time"] - args.pre_sec
    end = pick["p_time"] + args.post_sec
    st_cut = st.copy().trim(starttime=start, endtime=end, pad=True, fill_value=0)
    tr = choose_z_trace(st_cut, pick["channel"])
    if tr is None:
        return None, None, None, "no_z"

    if abs(float(tr.stats.sampling_rate) - args.sampling_rate) > 1e-6:
        if args.no_resample:
            return None, None, None, "bad_sr"
        tr = tr.resample(args.sampling_rate)

    tr = tr.copy().trim(starttime=start, endtime=end, pad=True, fill_value=0)
    data = np.asarray(tr.data, dtype=np.float64)
    if data.shape[0] < sample_len:
        data = np.pad(data, (0, sample_len - data.shape[0]), mode="constant")
    else:
        data = data[:sample_len]

    if not args.no_filter:
        data = bandpass(data, args.sampling_rate, args.freqmin, args.freqmax, args.filter_order)

    snr = snr_around_p(data, p_idx, args.sampling_rate)
    station = pick["station"] or f"{tr.stats.network}.{tr.stats.station}"
    return np.asarray(data, dtype=np.float32), float(snr), station, None


def main():
    parser = argparse.ArgumentParser(
        description="Cut P-wave windows from picked waveforms into a BTPC HDF5."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--picks", help="CSV with waveform_path and p_time columns")
    source.add_argument("--sac-dir", help="directory of SAC files with P picks in the header")
    parser.add_argument("--sac-pick-field", default="a", help="SAC header field holding the P pick (default: a)")
    parser.add_argument("--output", required=True, help="output HDF5 path")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--pre-sec", type=float, default=10.0, help="seconds before the P pick (sets its sample index)")
    parser.add_argument("--post-sec", type=float, default=30.0, help="seconds after the P pick")
    parser.add_argument("--freqmin", type=float, default=1.0)
    parser.add_argument("--freqmax", type=float, default=20.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--no-filter", action="store_true", help="skip the bandpass filter")
    parser.add_argument("--no-resample", action="store_true", help="skip records not at --sampling-rate instead of resampling")
    args = parser.parse_args()

    if args.freqmin >= args.freqmax:
        parser.error("--freqmin must be below --freqmax")

    fs = args.sampling_rate
    p_idx = int(round(args.pre_sec * fs))
    sample_len = int(round((args.pre_sec + args.post_sec) * fs)) + 1

    picks = read_picks_csv(args.picks) if args.picks else iter_sac_picks(args.sac_dir, args.sac_pick_field)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    stats = {"written": 0, "read_error": 0, "no_z": 0, "bad_sr": 0}

    with h5py.File(args.output, "w") as hf:
        grp = hf.create_group("phasenet")
        wave_ds = grp.create_dataset(
            "waveforms", shape=(0, sample_len), maxshape=(None, sample_len),
            dtype="float32", compression="gzip",
        )
        snr_ds = grp.create_dataset("snr", shape=(0,), maxshape=(None,), dtype="float32")
        rid_ds = grp.create_dataset("record_id", shape=(0,), maxshape=(None,), dtype="int32")
        eid_ds = grp.create_dataset("event_id", shape=(0,), maxshape=(None,), dtype="int32")
        sta_ds = grp.create_dataset(
            "station", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8")
        )
        pick_ds = grp.create_dataset(
            "pick_time", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8")
        )
        grp.attrs["p_arrival_index"] = p_idx
        grp.attrs["sampling_rate_hz"] = fs
        grp.attrs["freqmin_hz"] = args.freqmin
        grp.attrs["freqmax_hz"] = args.freqmax
        grp.attrs["filter_order"] = args.filter_order
        grp.attrs["filtered"] = not args.no_filter

        for auto_id, pick in enumerate(picks):
            data, snr, station, failure = process_pick(pick, args, sample_len, p_idx)
            if failure is not None:
                stats[failure] += 1
                continue

            try:
                record_id = int(pick["record_id"]) if pick["record_id"] else auto_id
            except ValueError:
                record_id = auto_id
            try:
                event_id = int(pick["event_id"]) if pick["event_id"] else -1
            except ValueError:
                event_id = -1

            n = wave_ds.shape[0]
            for ds in (wave_ds, snr_ds, rid_ds, eid_ds, sta_ds, pick_ds):
                ds.resize((n + 1,) + ds.shape[1:])
            wave_ds[n] = data
            snr_ds[n] = np.float32(snr)
            rid_ds[n] = np.int32(record_id)
            eid_ds[n] = np.int32(event_id)
            sta_ds[n] = station
            pick_ds[n] = str(pick["p_time"])
            stats["written"] += 1

    print(f"Saved {args.output}")
    print(f"records written: {stats['written']}")
    print(f"unreadable waveform files: {stats['read_error']}")
    print(f"records without a Z component: {stats['no_z']}")
    if args.no_resample:
        print(f"records not at {fs} Hz (skipped): {stats['bad_sr']}")
    print(f"P arrival at sample index {p_idx}; window {sample_len} samples at {fs} Hz")
    print(f"filter: {'none' if args.no_filter else f'{args.freqmin}-{args.freqmax} Hz order {args.filter_order}'}")


if __name__ == "__main__":
    main()
