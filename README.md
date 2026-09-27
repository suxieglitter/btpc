# BTPC — Barlow Twins P-wave polarity classification

BTPC classifies the first-motion polarity of seismic P waves (up vs. down)
**without any manually labelled training examples**. You point it at P-wave
windows from your own region, it trains itself, and the accepted polarities
feed directly into SKHASH first-motion focal-mechanism inversion.

The pipeline has three stages:

1. **Self-supervised encoder (Stage 1).** A 1-D residual encoder with
   attention is trained with the [Barlow Twins](https://arxiv.org/abs/2103.03230)
   loss on two augmented views of each 0.32 s P-wave window. Every
   `filter_interval` epochs, an unsupervised reliability score (waveform
   quality, augmentation consistency, kNN neighbor agreement, epoch stability,
   cluster margin) down-weights or rejects unreliable samples.
2. **Pseudo-label classifier (Stage 2).** Encoder features are clustered into
   two classes (spectral clustering by default) and a classification head is
   trained on the retained samples with the encoder frozen.
3. **Anchored prediction.** At prediction time a small bank of labelled SCSN
   anchor waveforms (shipped in `anchors/`) fixes the cluster-to-polarity
   mapping, and test-time-augmentation voting plus a margin criterion reject
   unstable predictions (label 2 = uncertain).

The full workflow on your own data, start to finish:

```
picks (CSV or SAC)  ──make_pwave_dataset.py──►  my_data.h5
my_data.h5          ──btpc-train stage1+stage2──►  model
my_data.h5 + model  ──btpc-predict──►  polarities CSV (up/down/uncertain)
polarities CSV      ──export_skhash.py──►  SKHASH inputs ──SKHASH──►  out.csv
```

Each arrow is one command; see [Using BTPC on your own region](#using-btpc-on-your-own-region).

The package is a cleaned-up release of the code used for the paper model
(experiment `mdl_d20260518`, seed-42 run; see `configs/btpc_0518.yaml`), and
the paper results are reproducible from it (see
[Reproducing the paper](#reproducing-the-paper)).

## Installation

```bash
git clone <repository-url>
cd btpc
pip install -e .
```

Python 3.10+ and PyTorch 2.0+ are required; see `requirements.txt` for the
full dependency list. Two optional dependencies are only needed for specific
steps: **obspy** for `make_pwave_dataset.py` (reading SAC/mSEED waveforms)
and **SKHASH** (`pip install skhash`) for the focal-mechanism inversion.

Tip: on Linux a plain `pip install torch` pulls the full CUDA stack
(2–3 GB of downloads). For CPU-only training use the CPU wheel instead
(~200 MB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start (no data needed)

`scripts/make_toy_data.py` generates a small synthetic dataset and a matching
anchor bank, so the whole pipeline runs in under a minute on CPU. Use it to
check the installation before touching real data:

```bash
python scripts/make_toy_data.py --output-dir runs/toy

# Stage 1 (tiny settings for the smoke test)
btpc-train stage1 --data-path runs/toy/toy_scsn.hdf5 \
    --num-used 512 --batch-size 64 --epochs 6 \
    --warmup-epochs 2 --filter-interval 2 --n-tta-views 3 \
    --save-path runs/toy/stage1

# Stage 2 (pseudo-label classifier; output lands in runs/toy/stage2)
btpc-train stage2 --stage1-dir runs/toy/stage1 --stage2-epochs 2 --batch-size 64

# Validation on the nontrain split of the training file
btpc-valid --stage2-checkpoint runs/toy/stage2/stage2_cluster_classifier.pth \
    --n-tta-views 2 --chunk-size 1024

# Prediction on unused rows of the training file
btpc-predict --stage2-checkpoint runs/toy/stage2/stage2_cluster_classifier.pth \
    --target-source train_unused \
    --anchor-bank-path runs/toy/toy_anchor_bank.npz \
    --max-samples 256 --n-tta-views 2
```

## Using BTPC on your own region

This is the intended everyday use: retrain on unlabelled waveforms from a
region of interest and invert the resulting polarities. You need four inputs:

- **P picks you already have** — either a pick-table CSV or SAC files with
  picks in a header field;
- **an event catalog** — one row per event with location and magnitude;
- **a station list** — one row per station with coordinates;
- **a 1-D velocity model** for your region.

### 1. Prepare the pick table

A CSV with one row per P pick. `waveform_path` and `p_time` are required;
`record_id`, `event_id`, `station` and `channel` are optional but
**`event_id` and `station` are needed later for SKHASH**, so include them:

```csv
waveform_path,p_time,record_id,event_id,station
data/2019-07-04/CI.CCA.2019-07-04.mseed,2019-07-04T17:35:13.078,0,0,CI.CCA
data/2019-07-04/CI.CCC.2019-07-04.mseed,2019-07-04T17:35:06.138,1,0,CI.CCC
```

`waveform_path` points to any file obspy can read (mSEED, SAC, …) and may
repeat across rows; `p_time` is any ISO timestamp. If your picks live in SAC
headers instead, skip the CSV and use `--sac-dir` + `--sac-pick-field` in
step 2 — that mode fills in station names automatically but leaves
`event_id` unset, which is fine for training and prediction (steps 3–4) but
not for the SKHASH export (step 5), so prefer the CSV table when you are
going all the way to mechanisms.

### 2. Cut the waveform windows

```bash
python scripts/make_pwave_dataset.py --picks picks.csv --output my_data.h5
# or: --sac-dir sac/ --sac-pick-field a
```

This applies the paper's processing — vertical component (HHZ > BHZ > EHZ >
any `*Z`, or the `channel` column), resampling to 100 Hz, a 1–20 Hz 4th-order
bandpass, and SNR = max|P..P+0.5 s| / max|P−0.5 s..P| — and writes
`my_data.h5` with the P arrival at sample index 1000. Window length, sampling
rate and filter corners are adjustable (`--pre-sec`, `--post-sec`,
`--freqmin/--freqmax`, `--no-filter`); keep the defaults to stay on the
paper's processing. Details: `docs/data.md` → *Preparing your own dataset*.

### 3. Train without labels

```bash
btpc-train stage1 --dataset-source unlabeled --data-path my_data.h5 \
    --num-used 0 --epochs 120 --n-tta-views 4 --eval-interval 0 \
    --save-path runs/mine/stage1
btpc-train stage2 --stage1-dir runs/mine/stage1
```

`--num-used 0` uses every record in the file; no polarity labels are read at
any point. The model lands in `runs/mine/stage2/stage2_cluster_classifier.pth`.

### 4. Predict polarities

```bash
btpc-predict --stage2-checkpoint runs/mine/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

The anchor bank ships with the repository (10,000 labelled SCSN waveforms;
prediction subsamples 100 per polarity from it). The paper's Ridgecrest
application used this same bank across regions. `--target-source ridge` is
the mode that reads the `phasenet`-group layout (the name is inherited from
the paper's Ridgecrest data; it works on any file from step 2). Predictions,
confidences and margins are written to
`runs/mine/stage2/predict_out/predictions_anchor_mapped.csv`.

### 5. Invert with SKHASH

Prepare the three region inputs (column aliases in `docs/skhash.md`):

```csv
# events.csv: event_id,lat,lon,dep,mag,time
0,35.6434,-117.566081,6.6,4.28,2019-07-04T17:35:01

# stations.csv: station,lat,lon,ele        (station is NET.STA or bare code)
CI.APL,35.34149,-116.87464,959.0

# vel.txt: depth_km,Vp_km_s
0.0,1.82
0.52,5.00
15.0,7.20
```

Then export and run:

```bash
python scripts/export_skhash.py \
    --predictions runs/mine/stage2/predict_out/predictions_anchor_mapped.csv \
    --data-path my_data.h5 \
    --events events.csv --stations stations.csv --vmodel vel.txt \
    --output-dir skhash_run
SKHASH skhash_run/control_auto.txt
```

The exporter applies the paper's rejection (mean confidence < 0.8 AND mean
center margin < 0.05; both tunable), keeps the higher-confidence pick when
one event was picked twice on one station, and writes the station/event/
polarity files plus a control file with the paper's SKHASH settings. The
focal mechanisms land in `skhash_run/output/out.csv`. To compare them against
an independent catalog, compute Kagan angles with any standard Kagan (2007)
implementation — that evaluation is outside this package. Full details and
tunable control-file parameters: `docs/skhash.md`.

## Predicting with the pretrained paper model

`models/0518_42/` ships the paper's SCSN-trained model (refined SCSN dataset,
seed 42, 80 epochs — the `configs/btpc_0518.yaml` run).
`stage2_cluster_classifier.pth` is the prediction checkpoint (encoder + head
+ cluster centers); `stage1_final_pwave_model.pth` is the bare Stage 1
encoder, kept for analysis. It reproduces the paper's validation numbers
through this package (acc 0.9630 before rejection, accept rate 0.9528, acc
0.9793 on accepted picks; see `docs/reproduction.md`).

Use it to predict directly, without retraining — e.g. on a `phasenet`-layout
file from step 2:

```bash
btpc-predict --stage2-checkpoint models/0518_42/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

Caveat: this model was trained and validated on SCSN data; how well it
transfers to another region without retraining was not part of the paper.
The recommended workflow for a new region remains the self-training path
above — the pretrained model is there for SCSN-style data, quick trials, and
as a baseline to compare your retrained model against.

## Reproducing the paper

With the real datasets in place (see [docs/data.md](docs/data.md)), the
reference configuration reproduces the paper model. A full from-scratch
reproduction of the paper results (SCSN models, the confusion-matrix figure,
Ridgecrest predictions and the Kagan-angle comparison) is documented in
[docs/reproduction.md](docs/reproduction.md).

```bash
btpc-train stage1 --config configs/btpc_0518.yaml --save-path runs/0518_42/stage1
btpc-train stage2 --stage1-dir runs/0518_42/stage1
btpc-valid --stage2-checkpoint runs/0518_42/stage2/stage2_cluster_classifier.pth
```

Any value in the YAML file can be overridden on the command line, e.g.
`--epochs 40 --batch-size 128`. To reproduce the paper's in-sample validation
numbers exactly, evaluate on the reconstructed Stage 1 training rows with the
paper's TTA and reject settings (reject only when mean confidence < 0.8 AND
mean center margin < 0.05), and compare with
`scripts/compare_paper_valid.py`:

```bash
btpc-valid --stage2-checkpoint runs/0518_42/stage2/stage2_cluster_classifier.pth \
    --target-source train --n-tta-views 30 --tta-max-shift 0 --tta-noise-std 0.015 \
    --reject-rule confidence_center_margin --reject-strategy all
```

### The paper's Ridgecrest application

The paper self-trains on the unlabelled `phasenet` group of the Ridgecrest
consensus file (27,698 waveforms, P arrival at sample index 1000), predicts
the same rows and feeds the accepted polarities to SKHASH — the same
own-region workflow as above, on the paper's data:

```bash
btpc-train stage1 --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --dataset-source ridgecrest_unlabeled --no-bino --num-used 0 \
    --epochs 120 --n-tta-views 4 --eval-interval 0 --save-path runs/ridge/stage1
btpc-train stage2 --stage1-dir runs/ridge/stage1
btpc-predict --stage2-checkpoint runs/ridge/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

`--dataset-source ridgecrest_unlabeled` and `unlabeled` are aliases: both
read the `phasenet`-group layout that `make_pwave_dataset.py` writes.

## Reference hyperparameters

Stage 1 defaults (also in `configs/btpc_0518.yaml`):

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| `snr_range` | [0, 1000) | | `aug_shift` | 1 |
| `num_used` | 10000 | | `aug_noise_std_range` | [0.05, 0.2] |
| `resize` | 32 | | `aug_scale_range` | [0.8, 1.2] |
| `shift` | 0 | | `enable_periodic_filtering` | true |
| `bino` | true | | `warmup_epochs` | 20 |
| `base_cha` | 16 | | `filter_interval` | 10 |
| `projector_dims` | 512 | | `n_tta_views` | 30 |
| `batch_size` | 256 | | `tta_max_shift` | 2 |
| `lr` | 0.001 | | `knn_k` | 10 |
| `lambda_param` | 0.005 | | `low/high_score_threshold` | 0.45 / 0.7 |
| `epochs` | 80 | | `low_score_patience` | 2 |
| `cluster_feature` | encoder | | `w_q/w_aug/w_knn/w_stab/w_margin` | 0.15/0.2/0.2/0.25/0.2 |
| `norm_mod` | max | | `downweight_loss_scale` | 0.35 |
| `eval_source` | clean | | `stability_ema_alpha` | 0.5 |
| `stage1_cluster_method` | spectral | | `seed` | 42 |

Stage 2 defaults: `stage2_cluster_method` spectral, `stage2_epochs` 30,
`stage2_lr` 0.001, `stage2_weight_decay` 1e-4, down-weighted samples included,
encoder frozen, linear head. Validation/prediction defaults: `n_tta_views` 4,
`tta_max_shift` 2, vote threshold 0.8, margin threshold 0.15.

## Outputs

- `stage1/`: `final_pwave_model.pth`, `stage1_config.json`, `stage1_summary.json`,
  `filtering/` (per-epoch filter CSV/NPZ + timeline), `cluster_results/`
  (CSV, t-SNE/PCA plots, confusion matrices, example waveforms), `train_out/`.
- `stage2/`: `stage2_cluster_classifier.pth` (encoder + head + cluster centers
  + feature normalization stats), `pseudo_labels.csv`, `stage2_summary.json`,
  `cluster_results/`.
- `valid_out/`: `valid_predictions.csv` (per-sample metrics + final labels),
  `valid_summary.json`, confusion matrices before/after rejection, metric
  histograms.
- `predict_*_out/`: `predictions_anchor_mapped.csv` (per-record polarities,
  confidences, margins), `predict_summary.json`, t-SNE plots of
  cluster/polarity assignments.
- SKHASH run directory (from `export_skhash.py`): `input/sk_sta.csv`,
  `input/sk_ctlg.csv`, `input/polarities.csv`, `control_auto.txt`,
  `input/predictions_for_skhash.csv`, `input/rejected_predictions.csv`,
  `export_summary.json`, and SKHASH's `output/out.csv`.

## License

MIT — see [LICENSE](LICENSE).
