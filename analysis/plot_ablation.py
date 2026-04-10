"""
Plot: Ablation Study (Table 3 / Section VI.E)
===============================================
Visualises ablation-style comparison using full-interval statistics rather
than only the last interval, and includes interval trajectories.

Usage:
    python analysis/plot_ablation.py \
        --output results/plots/ablation_study.png
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ci95(values: np.ndarray) -> float:
    """Approximate 95% confidence interval half-width over intervals."""
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


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

    # Full-trajectory vectors
    acc_static = df["accuracy_static_mean"].to_numpy(dtype=np.float64)
    acc_bayes = df["accuracy_bayesian_mean"].to_numpy(dtype=np.float64)
    acc_prop = df["accuracy_proposed_mean"].to_numpy(dtype=np.float64)

    pdr_static = df["pdr_static_mean"].to_numpy(dtype=np.float64)
    pdr_bayes = df["pdr_bayesian_mean"].to_numpy(dtype=np.float64)
    pdr_prop = df["pdr_proposed_mean"].to_numpy(dtype=np.float64)

    return {
        "intervals": df["interval"].to_numpy(dtype=np.int32),
        "configs": [
            "A\n(Static Auth)",
            "B\n(+ Bayesian Trust)",
            "C\n(+ Transformer\n+ Blockchain)",
        ],
        "accuracy_traj": {
            "A": acc_static,
            "B": acc_bayes,
            "C": acc_prop,
        },
        "pdr_traj": {
            "A": pdr_static,
            "B": pdr_bayes,
            "C": pdr_prop,
        },
        "accuracy": [
            float(acc_static.mean()),
            float(acc_bayes.mean()),
            float(acc_prop.mean()),
        ],
        "accuracy_ci": [
            _ci95(acc_static),
            _ci95(acc_bayes),
            _ci95(acc_prop),
        ],
        "pdr": [
            float(pdr_static.mean()),
            float(pdr_bayes.mean()),
            float(pdr_prop.mean()),
        ],
        "pdr_ci": [
            _ci95(pdr_static),
            _ci95(pdr_bayes),
            _ci95(pdr_prop),
        ],
    }


def plot_ablation(input_path: str, output_path: str):
    ablation_data = load_ablation_data(input_path)
    x = np.arange(len(ablation_data["configs"]))
    width = 0.35

    fig, (ax_traj, ax_bar) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # ── Trajectory panel (full experiment) ─────────────────────────────
    intervals = ablation_data["intervals"]
    ax_traj.plot(intervals, ablation_data["accuracy_traj"]["A"], color="#1f77b4", linestyle="--", linewidth=2.0, label="Accuracy A")
    ax_traj.plot(intervals, ablation_data["accuracy_traj"]["B"], color="#ff7f0e", linestyle="-.", linewidth=2.0, label="Accuracy B")
    ax_traj.plot(intervals, ablation_data["accuracy_traj"]["C"], color="#2ca02c", linestyle="-", linewidth=2.4, label="Accuracy C")

    ax_traj.plot(intervals, ablation_data["pdr_traj"]["A"], color="#1f77b4", linestyle=":", linewidth=1.8, alpha=0.85, label="PDR A")
    ax_traj.plot(intervals, ablation_data["pdr_traj"]["B"], color="#ff7f0e", linestyle=":", linewidth=1.8, alpha=0.85, label="PDR B")
    ax_traj.plot(intervals, ablation_data["pdr_traj"]["C"], color="#2ca02c", linestyle=":", linewidth=1.8, alpha=0.85, label="PDR C")

    ax_traj.set_title("Ablation Trajectories Across Monitoring Intervals", fontsize=12.5, fontweight="bold")
    ax_traj.set_xlabel("Monitoring Interval", fontsize=11)
    ax_traj.set_ylabel("Performance (%)", fontsize=11)
    ax_traj.grid(True, alpha=0.3, linestyle="--")
    ax_traj.legend(fontsize=9.5, ncol=3, framealpha=0.9)

    # ── Interval-averaged bars with CI ─────────────────────────────────
    bars_acc = ax_bar.bar(
        x - width / 2,
        ablation_data["accuracy"],
        width,
        yerr=ablation_data["accuracy_ci"],
        capsize=5,
        label="Detection Accuracy Mean ± 95% CI",
        color=["#aec7e8", "#6baed6", "#2171b5"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_pdr = ax_bar.bar(
        x + width / 2,
        ablation_data["pdr"],
        width,
        yerr=ablation_data["pdr_ci"],
        capsize=5,
        label="Packet Delivery Ratio Mean ± 95% CI",
        color=["#a1d99b", "#41ab5d", "#006d2c"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Value labels on bars
    for bar in bars_acc:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")

    for bar in bars_pdr:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")

    ax_bar.set_xlabel("Ablation Configuration", fontsize=12)
    ax_bar.set_ylabel("Performance (%)", fontsize=12)
    ax_bar.set_title(
        "Ablation Study — Mean Performance Across All Intervals (with 95% CI)",
        fontsize=13,
        fontweight="bold",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ablation_data["configs"], fontsize=10)
    ax_bar.set_ylim(65, 102)
    ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax_bar.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/simulation/results.csv")
    parser.add_argument("--output", default="results/plots/ablation_study.png")
    args = parser.parse_args()
    plot_ablation(args.input, args.output)


if __name__ == "__main__":
    main()
