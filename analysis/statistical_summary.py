"""
Statistical Summary
====================
Computes mean ± std summary statistics across all simulation runs and
saves a LaTeX-ready table to analysis/stats/summary_table.csv.

Usage:
    python analysis/statistical_summary.py \
        --input simulation/outputs/results.csv \
        --output analysis/stats/summary_table.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


METRICS = [
    ("Detection Accuracy — Proposed (%)",  "accuracy_proposed"),
    ("Detection Accuracy — Bayesian (%)",  "accuracy_bayesian"),
    ("Detection Accuracy — Static (%)",    "accuracy_static"),
    ("Precision — Proposed (%)",           "precision_proposed"),
    ("Precision — Bayesian (%)",           "precision_bayesian"),
    ("Precision — Static (%)",             "precision_static"),
    ("Recall — Proposed (%)",              "recall_proposed"),
    ("Recall — Bayesian (%)",              "recall_bayesian"),
    ("Recall — Static (%)",                "recall_static"),
    ("F1 Score — Proposed (%)",            "f1_proposed"),
    ("F1 Score — Bayesian (%)",            "f1_bayesian"),
    ("F1 Score — Static (%)",              "f1_static"),
    ("Packet Delivery Ratio — Proposed (%)", "pdr_proposed"),
    ("Packet Delivery Ratio — Bayesian (%)", "pdr_bayesian"),
    ("Packet Delivery Ratio — Static (%)",   "pdr_static"),
    ("Residual Energy — Proposed (%)",     "energy_proposed"),
    ("Residual Energy — Bayesian (%)",     "energy_bayesian"),
    ("Residual Energy — Static (%)",       "energy_static"),
]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, min, max across all intervals for each metric."""
    rows = []
    for label, col_prefix in METRICS:
        mean_col = f"{col_prefix}_mean"
        std_col  = f"{col_prefix}_std"
        if mean_col not in df.columns:
            continue
        rows.append({
            "Metric": label,
            "Mean":   round(df[mean_col].mean(), 2),
            "Std":    round(df[std_col].mean(), 2),
            "Min":    round(df[mean_col].min(), 2),
            "Max":    round(df[mean_col].max(), 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="simulation/outputs/results.csv")
    parser.add_argument("--output", default="analysis/stats/summary_table.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Results CSV not found: {args.input}. "
            "Run the simulation pipeline first: python scripts/run_full_pipeline.py"
        )

    df = pd.read_csv(args.input)
    summary = compute_summary(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("\n=== Performance Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
