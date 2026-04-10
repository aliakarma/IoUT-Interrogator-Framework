"""
Plot: Residual Energy Comparison (Figure 5)
=============================================
Reproduces the energy consumption comparison figure from the paper.

Usage:
    python analysis/plot_energy.py \
        --input results/simulation/results.csv \
        --output results/plots/energy_comparison.png
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


def plot_energy(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    intervals = df["interval"].values

    ax.plot(intervals, df["energy_static_mean"],   color="#1f77b4",
            linewidth=2.5, label="Static Trust", zorder=3)
    ax.fill_between(intervals,
                    df["energy_static_mean"] - df["energy_static_std"],
                    df["energy_static_mean"] + df["energy_static_std"],
                    alpha=0.12, color="#1f77b4")

    ax.plot(intervals, df["energy_bayesian_mean"], color="#ff7f0e",
            linewidth=2.0, linestyle="--", label="Bayesian Trust", zorder=2)
    ax.fill_between(intervals,
                    df["energy_bayesian_mean"] - df["energy_bayesian_std"],
                    df["energy_bayesian_mean"] + df["energy_bayesian_std"],
                    alpha=0.12, color="#ff7f0e")

    ax.plot(intervals, df["energy_proposed_mean"], color="#2ca02c",
            linewidth=2.0, linestyle="-", label="Proposed Interrogator Framework",
            zorder=1)
    ax.fill_between(intervals,
                    df["energy_proposed_mean"] - df["energy_proposed_std"],
                    df["energy_proposed_mean"] + df["energy_proposed_std"],
                    alpha=0.12, color="#2ca02c")

    # Overhead annotation
    static_final   = df["energy_static_mean"].iloc[-1]
    proposed_final = df["energy_proposed_mean"].iloc[-1]
    overhead_pct = abs(proposed_final - static_final) / static_final * 100
    ax.annotate(
        f"Overhead ≈ {overhead_pct:.1f}%",
        xy=(intervals[-1], (static_final + proposed_final) / 2),
        xytext=(-80, 0), textcoords="offset points",
        fontsize=9, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )

    ax.set_xlabel("Time (Monitoring Intervals)", fontsize=12)
    ax.set_ylabel("Residual Energy (%)", fontsize=12)
    ax.set_title("Residual Energy Comparison — Trust Architecture Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(intervals[0], intervals[-1])
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/simulation/results.csv")
    parser.add_argument("--output", default="results/plots/energy_comparison.png")
    args = parser.parse_args()
    df = load_or_generate_data(args.input)
    plot_energy(df, args.output)


if __name__ == "__main__":
    main()
