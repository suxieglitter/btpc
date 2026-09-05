#!/usr/bin/env python3
"""Compare reproduced Ridgecrest predictions against the original ones.

Joins the reproduced ``predictions_anchor_mapped.csv`` (btpc-predict) with the
original ``valid_predictions.csv`` on ``source_index`` (both use the same
SNR-filtered, sorted row order of the phasenet group) and reports cluster /
polarity agreement, TTA-metric differences, and reject-set agreement under the
paper's mc x mcm rule.

Example:
    python scripts/compare_ridge_predictions.py \
        --reproduced runs/repro_ridge_unlabeled/ridge_predict/predictions_anchor_mapped.csv \
        --reference /path/to/original/valid_predictions.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

METRIC_COLUMNS = ("vote_consistency", "mean_confidence", "mean_margin", "mean_center_margin")


def read_predictions(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["source_index"])] = row
    return rows


def reject_mask(confidence, center_margin, mc_threshold, mcm_threshold):
    return (confidence < mc_threshold) & (center_margin < mcm_threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduced", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--mc-threshold", type=float, default=0.8)
    parser.add_argument("--mcm-threshold", type=float, default=0.05)
    args = parser.parse_args()

    new_rows = read_predictions(args.reproduced)
    ref_rows = read_predictions(args.reference)
    shared = sorted(set(new_rows) & set(ref_rows))
    print(f"reference rows: {len(ref_rows)} | reproduced rows: {len(new_rows)} | joined: {len(shared)}")
    if len(ref_rows) != len(shared) or len(new_rows) != len(shared):
        missing_ref = sorted(set(ref_rows) - set(new_rows))[:10]
        missing_new = sorted(set(new_rows) - set(ref_rows))[:10]
        print(f"  only in reference: {len(set(ref_rows) - set(new_rows))} e.g. {missing_ref}")
        print(f"  only in reproduced: {len(set(new_rows) - set(ref_rows))} e.g. {missing_new}")

    ref_cluster = np.array([int(ref_rows[i]["pred_cluster_class"]) for i in shared])
    new_cluster = np.array([int(new_rows[i]["pred_cluster_class"]) for i in shared])
    ref_polarity = np.array([int(ref_rows[i]["pred_polarity"]) for i in shared])
    new_polarity = np.array([int(new_rows[i]["pred_polarity"]) for i in shared])

    cluster_agree = float(np.mean(ref_cluster == new_cluster))
    same_direction_agree = float(np.mean(ref_polarity == new_polarity))
    flipped_agree = float(np.mean(ref_polarity == (1 - new_polarity)))
    print(f"\npred_cluster_class agreement (mapping-independent): {cluster_agree:.4f}")
    print(f"pred_polarity agreement as-written: {same_direction_agree:.4f}")
    print(f"pred_polarity agreement with flipped mapping: {flipped_agree:.4f}")

    print("\nTTA metric differences (reference vs reproduced):")
    header = f"{'metric':<20}{'mean|diff|':>12}{'max|diff|':>12}{'pearson r':>12}"
    print(header)
    print("-" * len(header))
    for column in METRIC_COLUMNS:
        ref_values = np.array([float(ref_rows[i][column]) for i in shared])
        new_values = np.array([float(new_rows[i][column]) for i in shared])
        diff = np.abs(ref_values - new_values)
        pearson = float(np.corrcoef(ref_values, new_values)[0, 1])
        print(f"{column:<20}{diff.mean():>12.5f}{diff.max():>12.5f}{pearson:>12.4f}")

    ref_conf = np.array([float(ref_rows[i]["mean_confidence"]) for i in shared])
    ref_mcm = np.array([float(ref_rows[i]["mean_center_margin"]) for i in shared])
    new_conf = np.array([float(new_rows[i]["mean_confidence"]) for i in shared])
    new_mcm = np.array([float(new_rows[i]["mean_center_margin"]) for i in shared])
    ref_reject = reject_mask(ref_conf, ref_mcm, args.mc_threshold, args.mcm_threshold)
    new_reject = reject_mask(new_conf, new_mcm, args.mc_threshold, args.mcm_threshold)
    both = (ref_reject & new_reject).sum()
    either = (ref_reject | new_reject).sum()
    jaccard = float(both / either) if either else 1.0
    print(
        f"\nreject set @ mc<{args.mc_threshold} & mcm<{args.mcm_threshold}: "
        f"reference {int(ref_reject.sum())} | reproduced {int(new_reject.sum())} | "
        f"overlap {int(both)} | Jaccard {jaccard:.4f}"
    )

    ref_final = np.array([int(ref_rows[i]["final_label"]) for i in shared])
    new_final = np.array([int(new_rows[i]["final_label"]) for i in shared])
    print(f"up/down counts: reference {np.bincount(ref_polarity, minlength=2)} vs "
          f"reproduced {np.bincount(new_polarity, minlength=2)}")
    print(f"stored final_label (reference reject rule) agreement: {float(np.mean(ref_final == new_final)):.4f}")


if __name__ == "__main__":
    main()
