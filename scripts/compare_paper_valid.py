#!/usr/bin/env python3
"""Compare reproduced paper-validation outputs against the original ones.

Each ``--pair`` argument has the form ``reproduced_dir=reference_dir`` where both
directories contain a ``valid_predictions.csv`` (and optionally a
``valid_summary.json``). For every pair the script re-derives the paper's
in-sample validation numbers from the prediction CSVs with the paper reject
rule (a sample is rejected when all active criteria are violated, i.e. mean
confidence < threshold AND mean center margin < threshold) and prints a
side-by-side table.

Example:
    python scripts/compare_paper_valid.py \
        --pair runs/repro_v2_s42/paper_valid=/path/to/original/predict_valid-dir
"""

import argparse
import csv
import json
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.8
CENTER_MARGIN_THRESHOLD = 0.05


def read_predictions(csv_path: str):
    true_labels = []
    pred_labels = []
    confidences = []
    center_margins = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            true_labels.append(int(row["true_label"]))
            pred_labels.append(int(row["pred_polarity"]))
            confidences.append(float(row["mean_confidence"]))
            center_margins.append(float(row["mean_center_margin"]))
    return {
        "true": true_labels,
        "pred": pred_labels,
        "confidence": confidences,
        "center_margin": center_margins,
    }


def paper_metrics(predictions, reject_strategy: str = "all"):
    import numpy as np

    true = np.asarray(predictions["true"], dtype=np.int64)
    pred = np.asarray(predictions["pred"], dtype=np.int64)
    confidence = np.asarray(predictions["confidence"], dtype=np.float64)
    center_margin = np.asarray(predictions["center_margin"], dtype=np.float64)

    binary = (true == 0) | (true == 1)
    confidence_fail = confidence < CONFIDENCE_THRESHOLD
    center_margin_fail = center_margin < CENTER_MARGIN_THRESHOLD
    if reject_strategy == "all":
        rejected = confidence_fail & center_margin_fail
    else:
        rejected = confidence_fail | center_margin_fail
    accepted = binary & ~rejected

    def confusion(rows_true, rows_pred, classes):
        return {
            f"{t}->{p}": int(np.sum((rows_true == t) & (rows_pred == p)))
            for t in classes
            for p in classes
        }

    return {
        "n": int(binary.sum()),
        "acc_before_reject": float(np.mean(pred[binary] == true[binary])),
        "accept_rate": float(np.mean(accepted)),
        "acc_on_accepted_only": (
            float(np.mean(pred[accepted] == true[accepted])) if np.any(accepted) else float("nan")
        ),
        "reject_count": int(np.sum(binary & ~accepted)),
        "confusion_before_reject": confusion(true[binary], pred[binary], [0, 1]),
        "confusion_final_012": confusion(true[binary], np.where(accepted, pred[binary], 2), [0, 1, 2]),
    }


def summary_mapping(summary_path: str):
    path = Path(summary_path)
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mapping_accuracy": summary.get("mapping_accuracy_on_train_split"),
        "cluster_mapping": summary.get("cluster_mapping"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="REPRODUCED_DIR=REFERENCE_DIR",
        help="reproduced predict dir vs original predict dir (valid_predictions.csv in each)",
    )
    parser.add_argument(
        "--reject-strategy",
        choices=["all", "any"],
        default="all",
        help="all: reject only when every criterion is violated (paper); any: reject on any violation.",
    )
    args = parser.parse_args()

    for pair in args.pair:
        reproduced_dir, reference_dir = pair.split("=", 1)
        reproduced_csv = Path(reproduced_dir) / "valid_predictions.csv"
        reference_csv = Path(reference_dir) / "valid_predictions.csv"
        reproduced = paper_metrics(read_predictions(str(reproduced_csv)), args.reject_strategy)
        reference = paper_metrics(read_predictions(str(reference_csv)), args.reject_strategy)
        reproduced_map = summary_mapping(str(Path(reproduced_dir) / "valid_summary.json"))
        reference_map = summary_mapping(str(Path(reference_dir) / "valid_summary.json"))

        name = Path(reproduced_dir).parent.name
        print(f"\n## {name}")
        print(f"reproduced: {reproduced_dir}")
        print(f"reference : {reference_dir}")
        header = f"{'metric':<28}{'reference':>12}{'reproduced':>12}{'delta':>10}"
        print(header)
        print("-" * len(header))
        rows = [
            ("acc_before_reject", reference["acc_before_reject"], reproduced["acc_before_reject"]),
            ("accept_rate", reference["accept_rate"], reproduced["accept_rate"]),
            ("acc_on_accepted_only", reference["acc_on_accepted_only"], reproduced["acc_on_accepted_only"]),
            ("reject_count", reference["reject_count"], reproduced["reject_count"]),
        ]
        if reference_map and reproduced_map:
            if reference_map["mapping_accuracy"] is not None and reproduced_map["mapping_accuracy"] is not None:
                rows.append(
                    (
                        "mapping_accuracy",
                        reference_map["mapping_accuracy"],
                        reproduced_map["mapping_accuracy"],
                    )
                )
        for metric, ref_value, new_value in rows:
            delta = new_value - ref_value
            print(f"{metric:<28}{ref_value:>12.4f}{new_value:>12.4f}{delta:>+10.4f}")
        print(f"reference cluster mapping: {reference_map and reference_map['cluster_mapping']}")
        print(f"reproduced cluster mapping: {reproduced_map and reproduced_map['cluster_mapping']}")
        print(f"reference confusion before reject: {reference['confusion_before_reject']}")
        print(f"reproduced confusion before reject: {reproduced['confusion_before_reject']}")


if __name__ == "__main__":
    main()
