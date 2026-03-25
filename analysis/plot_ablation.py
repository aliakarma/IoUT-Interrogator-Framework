"""
Plot: Ablation Study (Table 3 / Section VI.E)
===============================================
Visualises the three-configuration ablation results as grouped bar charts.
Values are taken directly from Table 3 of the paper (no fabricated data).

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


# Values from Table 3 of the paper — do not modify
ABLATION_DATA = {
    "configs":  ["A\n(Static Auth)", "B\n(+ Bayesian Trust)", "C\n(+ Transformer\n+ Blockchain)"],
    "accuracy": [72.5, 86.1, 94.2],
    "pdr":      [79.4, 86.7, 91.6],
}


def plot_ablation(output_path: str):
    x = np.arange(len(ABLATION_DATA["configs"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_acc = ax.bar(x - width / 2, ABLATION_DATA["accuracy"], width,
                      label="Detection Accuracy (%)",
                      color=["#aec7e8", "#6baed6", "#2171b5"],
                      edgecolor="white", linewidth=0.8, zorder=3)
    bars_pdr = ax.bar(x + width / 2, ABLATION_DATA["pdr"], width,
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

    # Marginal gain annotations
    ax.annotate("", xy=(x[1] - width / 2, 87.5), xytext=(x[0] - width / 2, 87.5),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    ax.text((x[0] + x[1]) / 2 - width / 2, 88.5, "+13.6%",
            ha="center", fontsize=8.5, color="#333333")

    ax.annotate("", xy=(x[2] - width / 2, 95.5), xytext=(x[1] - width / 2, 95.5),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    ax.text((x[1] + x[2]) / 2 - width / 2, 96.5, "+8.1%",
            ha="center", fontsize=8.5, color="#333333")

    ax.set_xlabel("Ablation Configuration", fontsize=12)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("Ablation Study — Component-Wise Performance Attribution",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_DATA["configs"], fontsize=10)
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
    parser.add_argument("--output", default="analysis/plots/ablation_study.png")
    args = parser.parse_args()
    plot_ablation(args.output)


if __name__ == "__main__":
    main()
