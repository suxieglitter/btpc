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

## Building the anchor bank

`btpc-predict` needs a small anchor bank of labelled SCSN waveforms to map
the A/B clusters onto up/down polarities. Build it once from dataset 1
(any SCSN file with `Y`/`snr` works):

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
