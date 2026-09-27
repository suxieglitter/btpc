# Feeding BTPC polarities to SKHASH

`scripts/export_skhash.py` turns a `btpc-predict` output into a ready-to-run
[SKHASH](https://github.com/skhash/skhash) first-motion directory. SKHASH
itself is not a btpc dependency — install it separately (`pip install skhash`;
the results here were produced with version 1.1.2).

## What the exporter does

1. Reads `predictions_anchor_mapped.csv` (columns `record_id`, `snr`,
   `pred_polarity`, `mean_confidence`, `mean_center_margin`).
2. Applies the paper's rejection: a pick is rejected when
   `mean_confidence < --mc-threshold` **and** `mean_center_margin <
   --mcm-threshold` (`--reject-strategy and`; `or` rejects on either). The
   paper's Ridgecrest settings are the defaults (0.8 / 0.05 / and).
3. Drops uncertain labels (only up/down are exported), picks below
   `--min-snr`, and records whose event or station cannot be resolved.
4. When one event was picked more than once on the same station, keeps the
   pick with the highest mean confidence.
5. Writes:

| File | Purpose |
|---|---|
| `input/sk_sta.csv` | station list (only stations actually used) |
| `input/sk_ctlg.csv` | event catalog (only events with at least one pick) |
| `input/polarities.csv` | `event_id,network,station,channel,p_polarity` (+1 up, −1 down) |
| `control_auto.txt` | SKHASH control file with the paper's settings |
| `input/predictions_for_skhash.csv` | accepted picks with their mc/mcm |
| `input/rejected_predictions.csv` | rejected picks |
| `export_summary.json` | counts and thresholds |

The export was checked against the paper's preparation pipeline on the
Ridgecrest predictions: identical polarity rows (25,468), identical event
catalog, and the station list is the used-station subset.

## Input CSVs

The record → event/station mapping comes from the waveform HDF5 built by
`make_pwave_dataset.py`, so give the exporter the same `--data-path`. That
means the optional `event_id` and `station` columns of your pick table are
carried through to SKHASH.

Events CSV (`--events`): `event_id,lat,lon,dep,mag,time` — `depth` and `ot`
are accepted for `dep` and `time`; the time is ISO format
(`2019-07-04T17:35:01`).

Stations CSV (`--stations`): `station,lat,lon,ele` — station is `NET.STA`
(or a bare code with an optional `network` column); `elevation` is accepted
for `ele`.

Velocity model (`--vmodel`): SKHASH 1-D format, one layer per line as
`depth_km,Vp_km_s`:

```
# Depth (km), Vp (km/s)
0.0, 1.8200
0.52, 5.0000
15.0, 7.2000
```

## Running

```bash
python scripts/export_skhash.py \
    --predictions runs/mine/stage2/predict_out/predictions_anchor_mapped.csv \
    --data-path my_data.h5 \
    --events events.csv --stations stations.csv \
    --vmodel vel.txt \
    --output-dir skhash_run

SKHASH skhash_run/control_auto.txt
```

The control-file parameters (`--npolmin 12`, `--nmc 30`, `--maxout 500`,
`--ratmin 3`, `--delmax 120`, …) default to the paper's Ridgecrest values;
tune them to your network (in particular `--npolmin`, the minimum number of
polarities per event, and `--delmax`, the maximum event-station distance in
km). SKHASH writes focal mechanisms to `skhash_run/output/out.csv`.

To compare the resulting mechanisms against an independent reference (as the
paper does with Kagan angles against the SCSN consensus catalog), compute the
Kagan angle between the mechanism pairs with any standard implementation of
the Kagan (2007) double-couple rotation algorithm — that evaluation step is
outside the scope of this package.
