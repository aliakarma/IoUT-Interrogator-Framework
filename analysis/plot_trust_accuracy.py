"""
Plot: Anomaly Detection Accuracy (Figure 4)
=============================================
Reproduces the trust accuracy comparison figure from the paper.
Reads strictly from results/simulation/results.csv.

Usage:
    python analysis/plot_trust_accuracy.py \
        --input results/simulation/results.csv \
        --output results/plots/trust_accuracy.png
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_or_generate_data(input_path: str) -> pd.DataFrame:
    """Load simulation results CSV; fail fast if it is missing."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Simulation output not found: {input_path}. "
            "Run simulation/scripts/run_simulation.py first."
        )
    print(f"Loading simulation data from: {input_path}")
    return pd.read_csv(input_path)


def plot_trust_accuracy(df: pd.DataFrame, output_path: str):
    """Generate the anomaly detection accuracy comparison figure."""
    fig, ax = plt.subplots(figsize=(9, 5))

    intervals = df["interval"].values

    # Proposed
    ax.plot(intervals, df["accuracy_proposed_mean"], color="#2ca02c",
            linewidth=2.5, label="Proposed Interrogator Framework", zorder=3)
    ax.fill_between(intervals,
                    df["accuracy_proposed_mean"] - df["accuracy_proposed_std"],
                    df["accuracy_proposed_mean"] + df["accuracy_proposed_std"],
                    alpha=0.15, color="#2ca02c")

    # Bayesian
    ax.plot(intervals, df["accuracy_bayesian_mean"], color="#ff7f0e",
            linewidth=2.0, linestyle="-.", label="Bayesian Trust", zorder=2)
    ax.fill_between(intervals,
                    df["accuracy_bayesian_mean"] - df["accuracy_bayesian_std"],
                    df["accuracy_bayesian_mean"] + df["accuracy_bayesian_std"],
                    alpha=0.12, color="#ff7f0e")

    # Static
    ax.plot(intervals, df["accuracy_static_mean"], color="#1f77b4",
            linewidth=2.0, linestyle="--", label="Static Trust", zorder=1)
    ax.fill_between(intervals,
                    df["accuracy_static_mean"] - df["accuracy_static_std"],
                    df["accuracy_static_mean"] + df["accuracy_static_std"],
                    alpha=0.12, color="#1f77b4")

    ax.set_xlabel("Monitoring Intervals", fontsize=12)
    ax.set_ylabel("Anomaly Detection Accuracy (%)", fontsize=12)
    ax.set_title("Anomaly Detection Accuracy — Trust Architecture Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(intervals[0], intervals[-1])
    ax.set_ylim(65, 100)
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
    parser.add_argument("--input",  default="results/simulation/results.csv")
    parser.add_argument("--output", default="results/plots/trust_accuracy.png")
    args = parser.parse_args()

    df = load_or_generate_data(args.input)
    plot_trust_accuracy(df, args.output)


if __name__ == "__main__":
    main()
