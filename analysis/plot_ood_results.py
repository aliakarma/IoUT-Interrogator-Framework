"""
Plot OOD robustness under distribution shift.

Reads long-format OOD metric rows from:
    dataset_name, metric_name, value

and creates publication-ready robustness plots against:
  - adversarial fraction
  - noise level

Output:
  analysis/plots/ood_robustness.png
"""

import argparse
import os
import re
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_adv_noise(dataset_name: str) -> Tuple[float, float]:
    m = re.search(r"adv([0-9]+(?:\.[0-9]+)?)_noise([0-9]+(?:\.[0-9]+)?)", dataset_name)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def _ci95(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))


def _aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    grouped = (
        df.groupby(by)["value"]
        .apply(lambda x: pd.Series({"mean": float(x.mean()), "ci": _ci95(x.to_numpy(dtype=np.float64)), "n": int(len(x))}))
        .unstack()
        .reset_index()
        .sort_values(by=by)
    )
    return grouped


def plot_ood_results(input_path: str, output_path: str, metric: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"OOD results not found: {input_path}. Run scripts/run_ood_evaluation.py first."
        )

    df = pd.read_csv(input_path)
    required = {"dataset_name", "metric_name", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {sorted(required)}")

    metric_df = df[df["metric_name"] == metric].copy()
    if metric_df.empty:
        available = sorted(df["metric_name"].dropna().unique().tolist())
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {available}")

    parsed = metric_df["dataset_name"].apply(_parse_adv_noise)
    metric_df["adversarial_fraction"] = parsed.apply(lambda t: t[0])
    metric_df["noise_level"] = parsed.apply(lambda t: t[1])
    metric_df = metric_df.dropna(subset=["adversarial_fraction", "noise_level"])

    if metric_df.empty:
        raise ValueError(
            "Could not parse adversarial fraction/noise level from dataset_name. "
            "Expected pattern like data_adv0.15_noise0.1"
        )

    adv_stats = _aggregate(metric_df, by="adversarial_fraction")
    noise_stats = _aggregate(metric_df, by="noise_level")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    ax0 = axes[0]
    ax0.errorbar(
        adv_stats["adversarial_fraction"].to_numpy(dtype=np.float64),
        adv_stats["mean"].to_numpy(dtype=np.float64),
        yerr=adv_stats["ci"].to_numpy(dtype=np.float64),
        fmt="-o",
        linewidth=2.2,
        markersize=5,
        capsize=4,
        color="#1f77b4",
        ecolor="#6baed6",
    )
    ax0.set_title(f"{metric} vs Adversarial Fraction", fontsize=12.5, fontweight="bold")
    ax0.set_xlabel("Adversarial Fraction", fontsize=11)
    ax0.set_ylabel(f"{metric} (Mean +/- 95% CI)", fontsize=11)

    ax1 = axes[1]
    ax1.errorbar(
        noise_stats["noise_level"].to_numpy(dtype=np.float64),
        noise_stats["mean"].to_numpy(dtype=np.float64),
        yerr=noise_stats["ci"].to_numpy(dtype=np.float64),
        fmt="-o",
        linewidth=2.2,
        markersize=5,
        capsize=4,
        color="#2ca02c",
        ecolor="#74c476",
    )
    ax1.set_title(f"{metric} vs Noise Level", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Noise Level", fontsize=11)
    ax1.set_ylabel(f"{metric} (Mean +/- 95% CI)", fontsize=11)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("OOD Robustness Under Distribution Shift", fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OOD robustness from ood_results.csv")
    parser.add_argument("--input", default="analysis/stats/ood_results.csv")
    parser.add_argument("--output", default="analysis/plots/ood_robustness.png")
    parser.add_argument("--metric", default="C", help="Metric to visualize (e.g., C, RP, P, A, TI, ECE, Brier)")
    args = parser.parse_args()

    plot_ood_results(args.input, args.output, args.metric)


if __name__ == "__main__":
    main()
