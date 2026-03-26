"""
Master Plot Script
==================
Generates all paper figures in one command.

Usage:
    python analysis/plot_all.py [--input simulation/outputs/results.csv]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--input",  default="simulation/outputs/results.csv")
    parser.add_argument("--outdir", default="analysis/plots/")
    parser.add_argument("--trust-csv", default="data/processed/trust_scores.csv")
    parser.add_argument("--tau-min", type=float, default=0.65)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=== Generating all figures ===\n")

    # Figure 4 — Trust Accuracy
    from analysis.plot_trust_accuracy import load_or_generate_data, plot_trust_accuracy
    df = load_or_generate_data(args.input)
    plot_trust_accuracy(df, os.path.join(args.outdir, "trust_accuracy.png"))

    # Figure 5 — Energy
    from analysis.plot_energy import load_or_generate_data as load_energy, plot_energy
    df_e = load_energy(args.input)
    plot_energy(df_e, os.path.join(args.outdir, "energy_comparison.png"))

    # Figure 6 — PDR
    from analysis.plot_pdr import load_or_generate_data as load_pdr, plot_pdr
    df_p = load_pdr(args.input)
    plot_pdr(df_p, os.path.join(args.outdir, "pdr_comparison.png"))

    # Ablation figure
    from analysis.plot_ablation import plot_ablation
    plot_ablation(os.path.join(args.outdir, "ablation_study.png"))

    # Calibration diagnostics (optional; requires inference output)
    if os.path.exists(args.trust_csv):
        from analysis.plot_calibration import (
            plot_confidence_distribution,
            plot_trust_histogram,
            plot_reliability_curve,
            save_confidence_summary,
        )
        import pandas as pd

        df_t = pd.read_csv(args.trust_csv)
        plot_trust_histogram(df_t, args.outdir)
        plot_confidence_distribution(df_t, args.outdir)
        save_confidence_summary(df_t)
        plot_reliability_curve(df_t, args.outdir, tau_min=args.tau_min)
    else:
        print(f"Skipping calibration plots (not found): {args.trust_csv}")

    print("\n=== All figures saved to:", args.outdir, "===")


if __name__ == "__main__":
    main()
