# Datasets

BTPC is trained and evaluated on three HDF5 files. None of them ship with the
repository: download (or prepare) them locally and pass their paths on the
command line. The paths below are the local copies used to produce the paper
results; they will be replaced by public download links once the datasets are
hosted.

## 1. SCSN consensus (base version)

Used for the full-scale evaluation of the 0518 model family and for early
training experiments.

- Local path: `/home/glitter/software/a_dl_cluster/data/scsn_consensus_fm_CFM_EQ.hdf5`
- Size: 3.0 GB; 1,242,183 samples with SNR in [0, 1000)
- Layout (root level): `X` (waveforms, float32), `Y` (labels: 0 = up,
  1 = down, 2 = uncertain), `snr` (signal-to-noise ratio)

## 2. SCSN consensus (refined version) — paper training set

Used to train the paper model (see `configs/btpc_0518.yaml`). It is the base
version after proportion balancing, a `goodmin5` quality filter, and the
addition of real noise.

- Local path: `/home/glitter/software/a_dl_cluster/data/scsn_consensus_fm_CFM_EQ_proportion_binary_snr_100k_goodmin5_realnoise_combined.hdf5`
- Size: 682 MB
- Layout: same keys (`X` / `Y` / `snr`)

## 3. Ridgecrest waveforms (1-20 Hz bandpassed)

Used with `btpc-predict --target-source ridge`.

- Local path: `/home/glitter/software/Barlow_Twins/Ridgecrest_Test/data/consensus_waveforms_bp1_20.h5`
- Size: 932 MB
- Layout (group `phasenet`): `waveforms` (float32), `snr`, `record_id`
- The P arrival sits at sample index 1000 (100 Hz sampling).

## Common conventions

- Waveform length: 600 samples at 100 Hz; the P arrival is at index 300 in
  the SCSN files.
- Training windows are 32 samples (0.32 s) centered on the P arrival
  (`resize: 32`, `shift: 0` in the config).

## Preparing your own dataset

BTPC is label-free at training time, so you can train on waveforms from your
own region: cut P-wave windows around already-picked P arrivals with
`scripts/make_pwave_dataset.py` and train with `--dataset-source unlabeled`.

The script applies the same processing as the paper's Ridgecrest file:
vertical component (named channel, else HHZ > BHZ > EHZ > any `*Z`), resample
to 100 Hz, a 1–20 Hz 4th-order Butterworth bandpass, and
SNR = max|P..P+0.5 s| / max|P−0.5 s..P| computed on the filtered waveform.
The output places the P arrival at sample index 1000 (10 s after the window
start). It was checked to reproduce the paper's
`consensus_waveforms_bp1_20.h5` sample-for-sample.

Two pick sources are supported. A CSV pick table (export one from any picker;
`waveform_path` and `p_time` are the required columns, `record_id`, `event_id`,
`station` and `channel` are optional):

```csv
waveform_path,p_time,record_id,station
data/2019-07-04/CI.CCA.2019-07-04.mseed,2019-07-04T17:35:13.078,0,CI.CCA
```

```bash
python scripts/make_pwave_dataset.py --picks picks.csv --output my_data.h5
```

Or a directory of SAC files whose P pick lives in a SAC header field
(`a` by default, `t0`…`t9` also work):

```bash
python scripts/make_pwave_dataset.py --sac-dir sac/ --sac-pick-field a --output my_data.h5
```

Window length, sampling rate and filter corners are adjustable
(`--pre-sec`, `--post-sec`, `--freqmin/--freqmax`, `--no-filter`); keep the
defaults if you want to stay on the paper's processing. Rows whose noise
window is flat zero get `snr = nan` and are dropped by the SNR range filter at
training/prediction time. The script needs obspy (`pip install obspy`), which
is not a core btpc dependency.

Then train on your region without any labels and predict with the shipped
SCSN anchor bank:

```bash
btpc-train stage1 --dataset-source unlabeled --data-path my_data.h5 \
    --num-used 0 --epochs 120 --n-tta-views 4 --eval-interval 0 \
    --save-path runs/mine/stage1
btpc-train stage2 --stage1-dir runs/mine/stage1
btpc-predict --stage2-checkpoint runs/mine/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

`--dataset-source unlabeled` is an alias of `ridgecrest_unlabeled`: both read
the `phasenet` group produced by this script.

## Building the anchor bank

`btpc-predict` needs a small anchor bank of labelled SCSN waveforms to map
the A/B clusters onto up/down polarities. The repository ships the bank used
in the paper — `anchors/anchor_bank_scsn_50_1000.npz` (10,000 labelled SCSN
waveforms, 5,000 per polarity, SNR 50–1000, plus its CSV manifest; prediction
subsamples 100 per polarity from it) — so prediction works out of the box;
the paper's Ridgecrest application used exactly this bank across regions.

To build a bank from your own labelled waveforms instead (any SCSN file with
`Y`/`snr` works):

```python
from btpc.anchor_utils import build_anchor_bank

build_anchor_bank(
    bank_path="anchor_bank_scsn_50_1000.npz",
    data_path="/path/to/scsn_consensus_fm_CFM_EQ.hdf5",
    snr_range=(50.0, 1000.0),
    pool_per_label=5000,
    seed=42,
)
```

Then pass `--anchor-bank-path anchor_bank_scsn_50_1000.npz` to `btpc-predict`.

## Smoke test without any real data

`scripts/make_toy_data.py` generates a tiny synthetic dataset with the same
layout (plus a matching toy anchor bank) so the whole pipeline can be run
end-to-end before downloading anything:

```bash
python scripts/make_toy_data.py --output-dir runs/toy
```
