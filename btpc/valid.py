"""Validation of Stage 2 polarity predictions on the non-overlapping portion
of the training source file.

The validation split is reconstructed from the training HDF5: rows eligible in
the Stage 1 SNR range are split into the (reproducible) training candidate
pool and the remaining ``nontrain`` rows. The Stage 2 classifier predicts the
nontrain rows; cluster IDs are mapped to physical up/down polarities with the
training split, and unstable predictions (low vote consistency or low margin)
are rejected (final label 2).
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .anchor_utils import apply_cluster_mapping, fit_anchor_cluster_mapping, polarity_name
from .data import (
    DEFAULT_LABEL_KEY,
    DEFAULT_SNR_KEY,
    DEFAULT_WAVEFORM_KEY,
    read_scsn_rows,
)
from .predict import compute_prediction_metrics, load_stage2_bundle
from .utils import (
    checked_path,
    maybe_mkdir,
    resolve_data_path,
    save_confusion_matrix_percent,
    save_json,
)

VALID_HIST_METRICS = [
    ("snr", "SNR"),
    ("embedding_drift", "Embedding Drift"),
    ("vote_consistency", "Vote Consistency"),
    ("mean_confidence", "Mean Confidence"),
    ("mean_margin", "Mean Margin"),
    ("mean_center_margin", "Mean Center Margin"),
]


def save_valid_summary(save_dir: str, payload: Dict):
    save_json(os.path.join(save_dir, "valid_summary.json"), payload)


def iter_chunks(indices: np.ndarray, chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, len(indices), chunk_size):
        yield np.asarray(indices[start : start + chunk_size], dtype=np.int64)


def decode_object_array(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in np.asarray(values):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        elif isinstance(value, np.generic):
            item = value.item()
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8", errors="ignore"))
            else:
                decoded.append(str(item))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def read_source_slices(
    data_path: str,
    source_indices: np.ndarray,
    waveform_key: str = DEFAULT_WAVEFORM_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    snr_key: str = DEFAULT_SNR_KEY,
):
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.size == 0:
        return (
            np.empty((0, 600), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )

    waveforms, labels, snrs, extras = read_scsn_rows(
        data_path,
        source_indices,
        waveform_key=waveform_key,
        label_key=label_key,
        snr_key=snr_key,
        extra_keys=("evids", "sncls"),
    )
    labels = labels.astype(np.int64, copy=False)
    snrs = snrs.astype(np.float32, copy=False)

    evids_raw = extras.get("evids")
    if evids_raw is not None:
        evids = evids_raw.astype(np.int64, copy=False)
    else:
        evids = np.full(source_indices.shape, -1, dtype=np.int64)

    sncls_raw = extras.get("sncls")
    if sncls_raw is not None:
        sncls = decode_object_array(sncls_raw)
    else:
        sncls = np.full(source_indices.shape, "", dtype=object)

    return waveforms, labels, snrs, evids, sncls


def preprocess_clean_waveforms(
    raw_waveforms: np.ndarray,
    resize: int,
    shift: int,
    norm_mod: str,
    sampling_rate: float = 100.0,
    p_arrival_index: int = 300,
) -> np.ndarray:
    waveforms = np.asarray(raw_waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        raise ValueError(
            f"raw_waveforms must have shape (n_samples, n_points); got {waveforms.shape}."
        )

    detrended = signal.detrend(waveforms, axis=1).astype(np.float32, copy=False)
    half_win = int(resize) // 2
    start_idx = int(p_arrival_index) - half_win + int(shift)
    end_idx = int(p_arrival_index) + half_win + int(shift)
    if start_idx < 0 or end_idx > detrended.shape[1]:
        raise ValueError(
            f"Invalid crop window [{start_idx}, {end_idx}) for waveform length {detrended.shape[1]}."
        )

    p_wave = detrended[:, start_idx:end_idx].astype(np.float32, copy=False)
    if norm_mod == "rms":
        noise_win_len = int(0.5 * float(sampling_rate))
        noise_start = max(0, int(p_arrival_index) - 10 - noise_win_len)
        noise_end = max(noise_start + 1, int(p_arrival_index) - 10)
        noise_template = detrended[:, noise_start:noise_end]
        rms = np.sqrt(np.mean(noise_template ** 2, axis=1, keepdims=True))
        rms = np.where(rms > 1e-6, rms, 1.0)
        p_wave = p_wave / rms
    elif norm_mod == "max":
        max_vals = np.max(np.abs(p_wave), axis=1, keepdims=True)
        max_vals = np.where(max_vals > 1e-6, max_vals, 1.0)
        p_wave = p_wave / max_vals
    elif norm_mod == "bn":
        mean = np.mean(p_wave, axis=1, keepdims=True)
        std = np.std(p_wave, axis=1, keepdims=True)
        p_wave = (p_wave - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unsupported norm_mod: {norm_mod}")
    return np.asarray(p_wave, dtype=np.float32)


def build_split_from_training_source(
    data_path: str,
    snr_min: float,
    snr_max: float,
    num_used: int,
    bino: bool,
    include_label2: bool,
    exclude_initial_selected_pool: bool,
    max_valid_samples: int,
    label_key: str = DEFAULT_LABEL_KEY,
    snr_key: str = DEFAULT_SNR_KEY,
):
    with h5py.File(data_path, "r") as handle:
        missing_keys = [key for key in (label_key, snr_key) if key not in handle]
        if missing_keys:
            available_keys = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Missing dataset key(s) {missing_keys} in {data_path}. "
                f"Available keys: {available_keys}"
            )
        snr_all = np.asarray(handle[snr_key][:], dtype=np.float32)
        label_all = np.asarray(handle[label_key][:], dtype=np.int64)

    eligible = np.where((snr_all >= float(snr_min)) & (snr_all < float(snr_max)))[0]
    if eligible.size == 0:
        raise ValueError(f"No source samples found in SNR range [{snr_min}, {snr_max}).")

    if int(num_used) > 0 and eligible.size > int(num_used):
        train_candidate_indices = np.sort(eligible[: int(num_used)])
    else:
        train_candidate_indices = np.sort(eligible)

    candidate_labels = label_all[train_candidate_indices]
    if bool(bino):
        actual_train_mask = candidate_labels != 2
    else:
        actual_train_mask = np.ones(train_candidate_indices.shape, dtype=bool)

    train_source_indices = train_candidate_indices[actual_train_mask]
    train_labels = candidate_labels[actual_train_mask]

    excluded_for_valid = (
        train_candidate_indices if exclude_initial_selected_pool else train_source_indices
    )
    valid_source_indices = np.setdiff1d(eligible, excluded_for_valid, assume_unique=True)
    valid_labels = label_all[valid_source_indices]

    if not include_label2:
        keep_mask = valid_labels != 2
        valid_source_indices = valid_source_indices[keep_mask]
        valid_labels = valid_labels[keep_mask]

    if int(max_valid_samples) > 0 and valid_source_indices.size > int(max_valid_samples):
        valid_source_indices = valid_source_indices[: int(max_valid_samples)]
        valid_labels = valid_labels[: int(max_valid_samples)]

    if valid_source_indices.size == 0:
        raise ValueError("Validation split is empty after applying the requested filters.")

    return {
        "eligible_indices": np.asarray(eligible, dtype=np.int64),
        "train_candidate_indices": np.asarray(train_candidate_indices, dtype=np.int64),
        "train_source_indices": np.asarray(train_source_indices, dtype=np.int64),
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "valid_source_indices": np.asarray(valid_source_indices, dtype=np.int64),
        "valid_labels": np.asarray(valid_labels, dtype=np.int64),
    }


def compute_label_counts(labels: np.ndarray) -> Dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        "up": int(np.sum(labels == 0)),
        "down": int(np.sum(labels == 1)),
        "uncertain": int(np.sum(labels == 2)),
    }


def compute_precision_recall_acc(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_labels,
) -> Dict:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    class_labels = [int(label) for label in class_labels]
    total = int(len(true_labels))

    per_class = {}
    for label in class_labels:
        true_mask = true_labels == label
        pred_mask = pred_labels == label
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum((~true_mask) & pred_mask))
        fn = int(np.sum(true_mask & (~pred_mask)))
        tn = int(np.sum((~true_mask) & (~pred_mask)))

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        acc = float((tp + tn) / max(total, 1))

        per_class[polarity_name(label)] = {
            "label": int(label),
            "support": int(np.sum(true_mask)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "acc_one_vs_rest": acc,
        }

    return {
        "sample_count": total,
        "overall_accuracy": float(np.mean(true_labels == pred_labels)) if total > 0 else float("nan"),
        "per_class": per_class,
    }


def load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prediction_payload_from_csv(csv_path: str) -> Dict:
    true_labels = []
    pred_labels = []
    final_labels = []
    metric_values = {key: [] for key, _ in VALID_HIST_METRICS}

    text = Path(csv_path).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = set(reader.fieldnames or [])
    available_metric_keys = [key for key, _ in VALID_HIST_METRICS if key in fieldnames]

    for row in reader:
        true_labels.append(int(row["true_label"]))
        pred_labels.append(int(row["pred_polarity"]))
        final_labels.append(int(row["final_label"]))
        for key in available_metric_keys:
            metric_values[key].append(float(row[key]))

    return {
        "true_labels": np.asarray(true_labels, dtype=np.int64),
        "pred_labels": np.asarray(pred_labels, dtype=np.int64),
        "final_labels": np.asarray(final_labels, dtype=np.int64),
        "metrics": {
            key: np.asarray(values, dtype=np.float32)
            for key, values in metric_values.items()
            if len(values) > 0
        },
    }


def build_metrics_payload(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    final_labels: np.ndarray,
) -> Dict:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    final_labels = np.asarray(final_labels, dtype=np.int64)

    accepted_mask = final_labels != 2
    metrics_before_reject = compute_precision_recall_acc(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_labels=[0, 1],
    )
    metrics_after_reject = compute_precision_recall_acc(
        true_labels=true_labels,
        pred_labels=final_labels,
        class_labels=[0, 1],
    )
    metrics_accepted_only = compute_precision_recall_acc(
        true_labels=true_labels[accepted_mask],
        pred_labels=pred_labels[accepted_mask],
        class_labels=[0, 1],
    )

    acc_updown_before_reject = (
        float(np.mean(true_labels == pred_labels)) if true_labels.size > 0 else float("nan")
    )

    return {
        "acc_updown_before_reject": acc_updown_before_reject,
        "acc_with_reject_as_wrong": (
            float(np.mean(true_labels == final_labels)) if true_labels.size > 0 else float("nan")
        ),
        "acc_on_accepted_only": (
            float(np.mean(pred_labels[accepted_mask] == true_labels[accepted_mask]))
            if np.any(accepted_mask)
            else float("nan")
        ),
        "accuracy_before_reject": acc_updown_before_reject,
        "accuracy_after_reject_reject_as_wrong": (
            float(np.mean(true_labels == final_labels)) if true_labels.size > 0 else float("nan")
        ),
        "accept_rate": float(np.mean(accepted_mask)) if true_labels.size > 0 else float("nan"),
        "accuracy_on_accepted_only": (
            float(np.mean(pred_labels[accepted_mask] == true_labels[accepted_mask]))
            if np.any(accepted_mask)
            else float("nan")
        ),
        "before_reject_binary": metrics_before_reject,
        "after_reject_binary": metrics_after_reject,
        "accepted_only_binary": metrics_accepted_only,
    }


def _build_histogram_bins(values: np.ndarray, n_bins: int = 40):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-3)
        return np.linspace(vmin - pad, vmax + pad, 11)
    return np.linspace(vmin, vmax, int(n_bins) + 1)


def _build_log_histogram_bins(values: np.ndarray, n_bins: int = 40):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.geomspace(vmin / 1.1, vmax * 1.1, 11)
    return np.geomspace(vmin, vmax, int(n_bins) + 1)


def save_valid_metric_histograms(
    save_dir: str,
    metric_arrays: Dict[str, np.ndarray],
    final_labels: np.ndarray,
):
    final_labels = np.asarray(final_labels, dtype=np.int64)
    reject_mask = final_labels == 2
    if final_labels.size == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()

    for ax_idx, (metric_key, metric_title) in enumerate(VALID_HIST_METRICS):
        ax = axes[ax_idx]
        values = metric_arrays.get(metric_key)
        if values is None:
            ax.axis("off")
            continue

        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != final_labels.shape[0]:
            raise ValueError(
                f"Metric '{metric_key}' length {values.shape[0]} does not match "
                f"final_labels length {final_labels.shape[0]}."
            )

        finite_mask = np.isfinite(values)
        if metric_key == "snr":
            positive_mask = finite_mask & (values > 0)
            all_values = values[positive_mask]
            reject_values = values[reject_mask & positive_mask]
            bins = _build_log_histogram_bins(all_values)
        else:
            all_values = values[finite_mask]
            reject_values = values[reject_mask & finite_mask]
            bins = _build_histogram_bins(all_values)

        if bins is None:
            ax.text(0.5, 0.5, f"No finite values for {metric_key}", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.hist(
            all_values,
            bins=bins,
            color="#4C78A8",
            alpha=0.58,
            label=f"all (n={all_values.size})",
        )
        if reject_values.size > 0:
            ax.hist(
                reject_values,
                bins=bins,
                color="#E45756",
                alpha=0.58,
                label=f"reject (n={reject_values.size})",
            )

        ax.set_title(metric_title)
        ax.set_xlabel(metric_key)
        ax.set_ylabel("Count")
        if metric_key == "snr":
            ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=9)

        stats_lines = [f"all mean={np.mean(all_values):.4f}", f"all median={np.median(all_values):.4f}"]
        if reject_values.size > 0:
            stats_lines.extend(
                [
                    f"reject mean={np.mean(reject_values):.4f}",
                    f"reject median={np.median(reject_values):.4f}",
                ]
            )
        ax.text(
            0.98,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
        )

    for ax in axes[len(VALID_HIST_METRICS) :]:
        ax.axis("off")

    fig.suptitle("Validation Metrics Histograms | All vs Reject", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(save_dir, "valid_metric_histograms_all_vs_reject.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def print_binary_metrics(metrics: Dict):
    before_up = metrics["before_reject_binary"]["per_class"]["up"]
    before_down = metrics["before_reject_binary"]["per_class"]["down"]
    after_up = metrics["after_reject_binary"]["per_class"]["up"]
    after_down = metrics["after_reject_binary"]["per_class"]["down"]
    accepted_up = metrics["accepted_only_binary"]["per_class"]["up"]
    accepted_down = metrics["accepted_only_binary"]["per_class"]["down"]

    print(
        f"Validation metrics | acc_updown_before_reject={metrics['acc_updown_before_reject']:.4f} | "
        f"acc_with_reject_as_wrong={metrics['acc_with_reject_as_wrong']:.4f} | "
        f"accept_rate={metrics['accept_rate']:.4f} | "
        f"acc_on_accepted_only={metrics['acc_on_accepted_only']:.4f}"
    )
    print(
        f"Before reject | "
        f"up: precision={before_up['precision']:.4f}, recall={before_up['recall']:.4f}, acc={before_up['acc_one_vs_rest']:.4f} | "
        f"down: precision={before_down['precision']:.4f}, recall={before_down['recall']:.4f}, acc={before_down['acc_one_vs_rest']:.4f}"
    )
    print(
        f"After reject | "
        f"up: precision={after_up['precision']:.4f}, recall={after_up['recall']:.4f}, acc={after_up['acc_one_vs_rest']:.4f} | "
        f"down: precision={after_down['precision']:.4f}, recall={after_down['recall']:.4f}, acc={after_down['acc_one_vs_rest']:.4f}"
    )
    print(
        f"Accepted only | "
        f"up: precision={accepted_up['precision']:.4f}, recall={accepted_up['recall']:.4f}, acc={accepted_up['acc_one_vs_rest']:.4f} | "
        f"down: precision={accepted_down['precision']:.4f}, recall={accepted_down['recall']:.4f}, acc={accepted_down['acc_one_vs_rest']:.4f}"
    )


def finalize_outputs_from_predictions(
    save_dir: str,
    csv_path: str,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    final_labels: np.ndarray,
    metric_arrays: Dict[str, np.ndarray] | None = None,
    base_summary: Dict | None = None,
):
    save_valid_confusions(
        true_labels=true_labels,
        pred_labels=pred_labels,
        final_labels=final_labels,
        save_dir=save_dir,
    )
    if metric_arrays:
        save_valid_metric_histograms(
            save_dir=save_dir,
            metric_arrays=metric_arrays,
            final_labels=final_labels,
        )

    metrics_payload = build_metrics_payload(
        true_labels=true_labels,
        pred_labels=pred_labels,
        final_labels=final_labels,
    )

    summary = dict(base_summary or {})
    summary["save_dir"] = save_dir
    summary["csv_path"] = csv_path
    summary["label_counts"] = {
        "true": compute_label_counts(true_labels),
        "pred_before_reject": compute_label_counts(pred_labels),
        "final_after_reject": compute_label_counts(final_labels),
    }
    summary["metrics"] = metrics_payload
    save_valid_summary(save_dir, summary)

    print(f"Saved validation csv to: {csv_path}")
    print_binary_metrics(metrics_payload)


def fit_mapping_from_train_split(
    model,
    bundle,
    train_data_path: str,
    train_source_indices: np.ndarray,
    train_labels: np.ndarray,
    n_tta_views: int,
    tta_max_shift: int,
    batch_size: int,
    waveform_key: str = DEFAULT_WAVEFORM_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    snr_key: str = DEFAULT_SNR_KEY,
):
    stage1_config = bundle["stage1_config"]
    raw_waveforms, _, _, _, _ = read_source_slices(
        train_data_path,
        train_source_indices,
        waveform_key=waveform_key,
        label_key=label_key,
        snr_key=snr_key,
    )
    clean_waveforms = preprocess_clean_waveforms(
        raw_waveforms,
        resize=int(stage1_config["resize"]),
        shift=int(stage1_config["shift"]),
        norm_mod=str(stage1_config["norm_mod"]),
    )

    cluster_centers = np.asarray(bundle["cluster_centers"], dtype=np.float32)
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)

    train_metrics = compute_prediction_metrics(
        model=model,
        clean_waveforms=clean_waveforms,
        cluster_centers=cluster_centers,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_tta_views=max(int(n_tta_views), 1),
        tta_max_shift=int(tta_max_shift),
        batch_size=int(batch_size),
    )
    mapping, cluster_stats = fit_anchor_cluster_mapping(
        cluster_labels=train_metrics["modal_class"],
        is_anchor=np.ones(train_metrics["modal_class"].shape, dtype=np.int64),
        known_labels=np.asarray(train_labels, dtype=np.int64),
    )
    mapped_train = apply_cluster_mapping(train_metrics["modal_class"], mapping)
    mapping_accuracy = float(np.mean(mapped_train == np.asarray(train_labels, dtype=np.int64)))
    return mapping, cluster_stats, mapping_accuracy


def save_valid_confusions(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    final_labels: np.ndarray,
    save_dir: str,
):
    has_label2 = np.any(np.asarray(true_labels, dtype=np.int64) == 2)
    before_classes = [0, 1, 2] if has_label2 else [0, 1]
    before_prefix = (
        "confusion_matrix_percent_all012_before_reject"
        if has_label2
        else "confusion_matrix_percent_binary01_before_reject"
    )
    save_confusion_matrix_percent(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_labels=before_classes,
        out_dir=save_dir,
        filename_prefix=before_prefix,
        title="Validation Confusion Matrix (%) | Before reject",
    )
    save_confusion_matrix_percent(
        true_labels=true_labels,
        pred_labels=final_labels,
        class_labels=[0, 1, 2],
        out_dir=save_dir,
        filename_prefix="confusion_matrix_percent_binary012_after_reject",
        title="Validation Confusion Matrix (%) | After reject",
    )


def main_valid(args):
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(str(args.stage2_checkpoint)), "valid_out"
    )
    maybe_mkdir(save_dir)
    csv_path = os.path.join(save_dir, "valid_predictions.csv")
    summary_path = os.path.join(save_dir, "valid_summary.json")

    if os.path.exists(csv_path) and not bool(args.force_rerun_predict):
        print(f"Reusing existing validation predictions: {csv_path}")
        csv_payload = load_prediction_payload_from_csv(csv_path)
        true_labels_all = np.asarray(csv_payload["true_labels"], dtype=np.int64)
        pred_labels_all = np.asarray(csv_payload["pred_labels"], dtype=np.int64)
        final_labels_all = np.asarray(csv_payload["final_labels"], dtype=np.int64)
        existing_summary = load_json_if_exists(summary_path)
        if existing_summary is None:
            existing_summary = {
                "stage2_checkpoint": args.stage2_checkpoint,
                "note": "Metrics were rebuilt from an existing valid_predictions.csv without rerunning model inference.",
            }
        else:
            existing_summary["note"] = (
                "Metrics were rebuilt from an existing valid_predictions.csv without rerunning model inference."
            )
        finalize_outputs_from_predictions(
            save_dir=save_dir,
            csv_path=csv_path,
            true_labels=true_labels_all,
            pred_labels=pred_labels_all,
            final_labels=final_labels_all,
            metric_arrays=csv_payload["metrics"],
            base_summary=existing_summary,
        )
        return

    model, bundle = load_stage2_bundle(args.stage2_checkpoint)
    stage1_config = bundle["stage1_config"]

    data_path = args.data_path
    if data_path is None:
        data_path = stage1_config.get("train_data_path", stage1_config.get("data_path"))
    data_path = resolve_data_path(data_path)
    waveform_key = args.waveform_key or stage1_config.get("data_waveform_key", DEFAULT_WAVEFORM_KEY)
    label_key = args.label_key or stage1_config.get("data_label_key", DEFAULT_LABEL_KEY)
    snr_key = args.snr_key or stage1_config.get("data_snr_key", DEFAULT_SNR_KEY)

    resize = int(stage1_config["resize"])
    shift = int(stage1_config["shift"])
    batch_size = int(stage1_config["batch_size"])
    norm_mod = str(stage1_config["norm_mod"])
    snr_min = float(stage1_config["snr_range"][0])
    snr_max = float(stage1_config["snr_range"][1])
    num_used = int(stage1_config["num_used"])
    bino = bool(stage1_config["bino"])

    split_info = build_split_from_training_source(
        data_path=data_path,
        snr_min=snr_min,
        snr_max=snr_max,
        num_used=num_used,
        bino=bino,
        include_label2=bool(args.include_label2),
        exclude_initial_selected_pool=bool(args.exclude_initial_selected_pool),
        max_valid_samples=int(args.max_valid_samples),
        label_key=label_key,
        snr_key=snr_key,
    )

    mapping, cluster_stats, mapping_accuracy = fit_mapping_from_train_split(
        model=model,
        bundle=bundle,
        train_data_path=data_path,
        train_source_indices=split_info["train_source_indices"],
        train_labels=split_info["train_labels"],
        n_tta_views=args.n_tta_views,
        tta_max_shift=args.tta_max_shift,
        batch_size=batch_size,
        waveform_key=waveform_key,
        label_key=label_key,
        snr_key=snr_key,
    )

    print("Validation-time train-split mapping:")
    for cluster_id in sorted(cluster_stats):
        stat = cluster_stats[cluster_id]
        print(
            f"  class {cluster_id} -> {polarity_name(stat['mapped_label'])} | "
            f"up={stat['up_count']} | down={stat['down_count']} | purity={stat['purity']:.4f}"
        )
    print(f"Mapping accuracy on train split: {mapping_accuracy:.4f}")

    cluster_centers = np.asarray(bundle["cluster_centers"], dtype=np.float32)
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)

    all_true_labels = []
    all_pred_labels = []
    all_final_labels = []
    metric_buffers = {key: [] for key, _ in VALID_HIST_METRICS}
    valid_indices = split_info["valid_source_indices"]
    n_chunks = (len(valid_indices) + max(int(args.chunk_size), 1) - 1) // max(int(args.chunk_size), 1)

    header = [
        "source_index",
        "evid",
        "sncl",
        "snr",
        "true_label",
        "true_label_name",
        "pred_cluster_class",
        "pred_polarity",
        "pred_polarity_name",
        "vote_consistency",
        "mean_confidence",
        "mean_margin",
        "mean_center_margin",
        "embedding_drift",
        "is_reject",
        "final_label",
        "final_label_name",
    ]
    csv_chunks = []

    for chunk_idx, source_chunk in enumerate(iter_chunks(valid_indices, args.chunk_size), start=1):
        raw_waveforms, true_labels, snrs, evids, sncls = read_source_slices(
            data_path,
            source_chunk,
            waveform_key=waveform_key,
            label_key=label_key,
            snr_key=snr_key,
        )
        clean_waveforms = preprocess_clean_waveforms(
            raw_waveforms,
            resize=resize,
            shift=shift,
            norm_mod=norm_mod,
        )

        metrics = compute_prediction_metrics(
            model=model,
            clean_waveforms=clean_waveforms,
            cluster_centers=cluster_centers,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            n_tta_views=max(int(args.n_tta_views), 1),
            tta_max_shift=int(args.tta_max_shift),
            batch_size=batch_size,
        )
        pred_polarity = apply_cluster_mapping(metrics["modal_class"], mapping)
        reject_mask = (
            (metrics["vote_consistency"] < float(args.valid_vote_threshold))
            | (metrics["mean_margin"] < float(args.valid_margin_threshold))
        )
        final_label = pred_polarity.copy()
        final_label[reject_mask] = 2

        all_true_labels.append(np.asarray(true_labels, dtype=np.int64))
        all_pred_labels.append(np.asarray(pred_polarity, dtype=np.int64))
        all_final_labels.append(np.asarray(final_label, dtype=np.int64))
        metric_buffers["snr"].append(np.asarray(snrs, dtype=np.float32))
        for metric_key in metric_buffers:
            if metric_key == "snr":
                continue
            metric_buffers[metric_key].append(np.asarray(metrics[metric_key], dtype=np.float32))

        rows = []
        for row_idx in range(len(source_chunk)):
            rows.append(
                [
                    int(source_chunk[row_idx]),
                    int(evids[row_idx]) if row_idx < len(evids) else -1,
                    str(sncls[row_idx]) if row_idx < len(sncls) else "",
                    float(snrs[row_idx]),
                    int(true_labels[row_idx]),
                    polarity_name(true_labels[row_idx]),
                    int(metrics["modal_class"][row_idx]),
                    int(pred_polarity[row_idx]),
                    polarity_name(pred_polarity[row_idx]),
                    float(metrics["vote_consistency"][row_idx]),
                    float(metrics["mean_confidence"][row_idx]),
                    float(metrics["mean_margin"][row_idx]),
                    float(metrics["mean_center_margin"][row_idx]),
                    float(metrics["embedding_drift"][row_idx]),
                    int(bool(reject_mask[row_idx])),
                    int(final_label[row_idx]),
                    polarity_name(final_label[row_idx]),
                ]
            )
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerows(rows)
        csv_chunks.append(buffer.getvalue())

        if chunk_idx == 1 or chunk_idx == n_chunks or chunk_idx % 10 == 0:
            total_count = int(sum(len(x) for x in all_true_labels))
            accepted_count = int(sum(np.sum(x != 2) for x in all_final_labels))
            print(
                f"Processed validation chunk {chunk_idx}/{n_chunks} | "
                f"samples={total_count} | accepted={accepted_count}"
            )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    buffer.write("".join(csv_chunks))
    Path(checked_path(csv_path)).write_text(buffer.getvalue(), encoding="utf-8")

    true_labels_all = np.concatenate(all_true_labels, axis=0)
    pred_labels_all = np.concatenate(all_pred_labels, axis=0)
    final_labels_all = np.concatenate(all_final_labels, axis=0)
    metric_arrays_all = {
        key: np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.empty((0,), dtype=np.float32)
        for key, chunks in metric_buffers.items()
    }

    summary = {
        "stage2_checkpoint": args.stage2_checkpoint,
        "source_data_path": data_path,
        "source_waveform_key": waveform_key,
        "source_label_key": label_key,
        "source_snr_key": snr_key,
        "save_dir": save_dir,
        "csv_path": csv_path,
        "split_definition": {
            "snr_range": [snr_min, snr_max],
            "num_used": num_used,
            "bino": bino,
            "include_label2": bool(args.include_label2),
            "exclude_initial_selected_pool": bool(args.exclude_initial_selected_pool),
            "note": (
                "valid split is built from the original training-source file after removing "
                "the reconstructed Stage1 training rows."
            ),
        },
        "split_sizes": {
            "eligible_source_count": int(len(split_info["eligible_indices"])),
            "train_candidate_count_pre_bino": int(len(split_info["train_candidate_indices"])),
            "train_actual_count_used_for_mapping": int(len(split_info["train_source_indices"])),
            "valid_count": int(len(valid_indices)),
        },
        "thresholds": {
            "valid_vote_threshold": float(args.valid_vote_threshold),
            "valid_margin_threshold": float(args.valid_margin_threshold),
        },
        "cluster_mapping": {int(k): int(v) for k, v in mapping.items()},
        "cluster_stats": cluster_stats,
        "mapping_accuracy_on_train_split": float(mapping_accuracy),
    }
    finalize_outputs_from_predictions(
        save_dir=save_dir,
        csv_path=csv_path,
        true_labels=true_labels_all,
        pred_labels=pred_labels_all,
        final_labels=final_labels_all,
        metric_arrays=metric_arrays_all,
        base_summary=summary,
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        prog="btpc-valid",
        description=(
            "Validate Stage 2 polarity predictions on the non-overlapping "
            "(nontrain) portion of the training source file."
        ),
    )
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Training-source HDF5 (default: the path recorded in stage1_config.json).",
    )
    parser.add_argument("--waveform-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--snr-key", default=None)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--include-label2", action="store_true", default=False)
    parser.add_argument(
        "--exclude-initial-selected-pool",
        action="store_true",
        default=False,
        help="Exclude the full pre-bino selected pool instead of only the rows actually used by training.",
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--n-tta-views", type=int, default=4)
    parser.add_argument("--tta-max-shift", type=int, default=2)
    parser.add_argument("--valid-vote-threshold", type=float, default=0.8)
    parser.add_argument("--valid-margin-threshold", type=float, default=0.15)
    parser.add_argument("--force-rerun-predict", action="store_true", default=False)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    main_valid(args)


if __name__ == "__main__":
    main()
