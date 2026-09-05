# BTPC — Barlow Twins P-wave polarity classification

BTPC is a label-free pipeline that classifies the first-motion polarity of
seismic P waves (up vs. down) without manually labelled training examples.

1. **Stage 1 — self-supervised encoder.** A 1-D residual encoder with
   attention is trained with the [Barlow Twins](https://arxiv.org/abs/2103.03230)
   loss on two augmented views of each 0.32 s P-wave window. Every
   `filter_interval` epochs, an unsupervised reliability score (waveform
   quality, augmentation consistency, kNN neighbor agreement, epoch stability,
   cluster margin) down-weights or rejects unreliable samples.
2. **Stage 2 — pseudo-label classifier.** Encoder features are clustered into
   two classes (spectral clustering by default) and a classification head is
   trained on the retained samples with the encoder frozen.
3. **Anchored prediction.** At validation/prediction time a small bank of
   labelled SCSN anchor waveforms fixes the cluster-to-polarity mapping, and
   test-time-augmentation voting plus a margin criterion reject unstable
   predictions (label 2 = uncertain).

The package is a cleaned-up release of the code used for the paper model
(experiment `mdl_d20260518`, seed-42 run; see `configs/btpc_0518.yaml`).

## Install

```bash
git clone <repository-url>
cd btpc
pip install -e .
```

Python 3.10+ and PyTorch 2.0+ are required; see `requirements.txt` for the
full dependency list.

## Quick start (no data needed)

`scripts/make_toy_data.py` generates a small synthetic SCSN-style dataset and
a matching anchor bank, so the whole pipeline runs in under a minute on CPU:

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

Each step writes its artifacts under the given `--save-path` (or next to the
checkpoint): checkpoints, config snapshots, CSV predictions, confusion
matrices, clustering visualizations and filtering timelines.

## Training the paper model

With the real dataset in place (see [docs/data.md](docs/data.md)), the
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

Prediction on external event waveforms:

```bash
btpc-predict --stage2-checkpoint runs/0518_42/stage2/stage2_cluster_classifier.pth \
    --target-source ridge \
    --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --anchor-bank-path anchor_bank_scsn_50_1000.npz
```

### Training on unlabeled Ridgecrest waveforms

The paper's Ridgecrest application self-trains on the unlabelled `phasenet`
group of the Ridgecrest consensus file (P arrival at sample index 1000), then
predicts the same rows and feeds the accepted polarities to SKHASH:

```bash
btpc-train stage1 --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --dataset-source ridgecrest_unlabeled --no-bino --num-used 0 \
    --epochs 120 --n-tta-views 4 --eval-interval 0 --save-path runs/ridge/stage1
btpc-train stage2 --stage1-dir runs/ridge/stage1
btpc-predict --stage2-checkpoint runs/ridge/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --anchor-bank-path anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

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
- `predict_*_out/`: `predictions_anchor_mapped.csv`, `predict_summary.json`,
  t-SNE plots of cluster/polarity assignments.

## License

MIT — see [LICENSE](LICENSE).
