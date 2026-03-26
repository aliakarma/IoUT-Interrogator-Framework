"""
Plot: Ablation Study (Table 3 / Section VI.E)
===============================================
Visualises ablation-style comparison as grouped bars from simulation output.

Usage:
    python analysis/plot_ablation.py \
        --output analysis/plots/ablation_study.png
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ablation_data(input_path: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Simulation output not found: {input_path}. "
            "Run simulation/scripts/run_simulation.py first."
        )

    df = pd.read_csv(input_path)
    required = [
        "accuracy_static_mean", "accuracy_bayesian_mean", "accuracy_proposed_mean",
        "pdr_static_mean", "pdr_bayesian_mean", "pdr_proposed_mean",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {input_path}: {col}")

    return {
        "configs": [
            "A\n(Static Auth)",
            "B\n(+ Bayesian Trust)",
            "C\n(+ Transformer\n+ Blockchain)",
        ],
        "accuracy": [
            float(df["accuracy_static_mean"].iloc[-1]),
            float(df["accuracy_bayesian_mean"].iloc[-1]),
            float(df["accuracy_proposed_mean"].iloc[-1]),
        ],
        "pdr": [
            float(df["pdr_static_mean"].iloc[-1]),
            float(df["pdr_bayesian_mean"].iloc[-1]),
            float(df["pdr_proposed_mean"].iloc[-1]),
        ],
    }


def plot_ablation(input_path: str, output_path: str):
    ablation_data = load_ablation_data(input_path)
    x = np.arange(len(ablation_data["configs"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_acc = ax.bar(x - width / 2, ablation_data["accuracy"], width,
                      label="Detection Accuracy (%)",
                      color=["#aec7e8", "#6baed6", "#2171b5"],
                      edgecolor="white", linewidth=0.8, zorder=3)
    bars_pdr = ax.bar(x + width / 2, ablation_data["pdr"], width,
                      label="Packet Delivery Ratio (%)",
                      color=["#a1d99b", "#41ab5d", "#006d2c"],
                      edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")

    for bar in bars_pdr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")

    ax.set_xlabel("Ablation Configuration", fontsize=12)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("Ablation Study — Component-Wise Performance Attribution",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_data["configs"], fontsize=10)
    ax.set_ylim(65, 102)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="simulation/outputs/results.csv")
    parser.add_argument("--output", default="analysis/plots/ablation_study.png")
    args = parser.parse_args()
    plot_ablation(args.input, args.output)


if __name__ == "__main__":
    main()
