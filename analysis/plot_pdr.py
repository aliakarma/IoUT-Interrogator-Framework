"""
Plot: Packet Delivery Ratio (Figure 6)
========================================
Reproduces the PDR comparison figure from the paper.

Usage:
    python analysis/plot_pdr.py \
        --input simulation/outputs/results.csv \
        --output analysis/plots/pdr_comparison.png
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_or_generate_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Simulation output not found: {input_path}. "
            "Run simulation/scripts/run_simulation.py first."
        )
    print(f"Loading simulation data from: {input_path}")
    return pd.read_csv(input_path)


def plot_pdr(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    intervals = df["interval"].values

    ax.plot(intervals, df["pdr_proposed_mean"], color="#2ca02c",
            linewidth=2.5, label="Proposed Interrogator Framework", zorder=3)
    ax.fill_between(intervals,
                    df["pdr_proposed_mean"] - df["pdr_proposed_std"],
                    df["pdr_proposed_mean"] + df["pdr_proposed_std"],
                    alpha=0.15, color="#2ca02c")

    ax.plot(intervals, df["pdr_bayesian_mean"], color="#ff7f0e",
            linewidth=2.0, linestyle="-.", label="Bayesian Trust", zorder=2)
    ax.fill_between(intervals,
                    df["pdr_bayesian_mean"] - df["pdr_bayesian_std"],
                    df["pdr_bayesian_mean"] + df["pdr_bayesian_std"],
                    alpha=0.12, color="#ff7f0e")

    ax.plot(intervals, df["pdr_static_mean"], color="#1f77b4",
            linewidth=2.0, linestyle="--", label="Static Trust", zorder=1)
    ax.fill_between(intervals,
                    df["pdr_static_mean"] - df["pdr_static_std"],
                    df["pdr_static_mean"] + df["pdr_static_std"],
                    alpha=0.12, color="#1f77b4")

    ax.set_xlabel("Monitoring Intervals", fontsize=12)
    ax.set_ylabel("Packet Delivery Ratio (%)", fontsize=12)
    ax.set_title("Packet Delivery Ratio — Trust Architecture Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(intervals[0], intervals[-1])
    ax.set_ylim(74, 95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="simulation/outputs/results.csv")
    parser.add_argument("--output", default="analysis/plots/pdr_comparison.png")
    args = parser.parse_args()
    df = load_or_generate_data(args.input)
    plot_pdr(df, args.output)


if __name__ == "__main__":
    main()
