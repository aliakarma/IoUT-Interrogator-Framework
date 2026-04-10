"""
Calibration Diagnostics for Trust Scores
========================================
Generates calibration artifacts from inference output CSV:
  1) Trust score histogram
  2) Confidence distribution summary CSV
  3) Optional reliability curve (if true labels are available)

Usage:
    python analysis/plot_calibration.py \
        --input results/processed/trust_scores.csv \
        --outdir results/plots \
        --tau-min 0.65
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_output_dirs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("results/stats", exist_ok=True)


def _confidence_from_trust(trust_scores: np.ndarray) -> np.ndarray:
    p_adv = 1.0 - trust_scores
    return np.maximum(trust_scores, p_adv)


def plot_trust_histogram(df: pd.DataFrame, outdir: str):
    trust = df["trust_score"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(8, 4.8))
    if "true_label" in df.columns and (df["true_label"] != -1).any():
        legit = df[df["true_label"] == 0]["trust_score"].to_numpy(dtype=np.float64)
        adv = df[df["true_label"] == 1]["trust_score"].to_numpy(dtype=np.float64)
        if len(legit) > 0:
            plt.hist(
                legit,
                bins=20,
                range=(0.0, 1.0),
                alpha=0.65,
                edgecolor="black",
                label="Legitimate",
            )
        if len(adv) > 0:
            plt.hist(
                adv,
                bins=20,
                range=(0.0, 1.0),
                alpha=0.65,
                edgecolor="black",
                label="Adversarial",
            )
        if len(legit) > 0 or len(adv) > 0:
            plt.legend()
    else:
        plt.hist(trust, bins=20, range=(0.0, 1.0), alpha=0.85, edgecolor="black")
    plt.xlabel("Trust score")
    plt.ylabel("Count")
    plt.title("Trust Score Distribution")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    out_path = os.path.join(outdir, "trust_score_histogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_confidence_distribution(df: pd.DataFrame, outdir: str):
    trust = df["trust_score"].to_numpy(dtype=np.float64)
    confidence = _confidence_from_trust(trust)

    plt.figure(figsize=(8, 4.8))
    if "true_label" in df.columns and (df["true_label"] != -1).any():
        legit_conf = _confidence_from_trust(
            df[df["true_label"] == 0]["trust_score"].to_numpy(dtype=np.float64)
        )
        adv_conf = _confidence_from_trust(
            df[df["true_label"] == 1]["trust_score"].to_numpy(dtype=np.float64)
        )
        if len(legit_conf) > 0:
            plt.hist(
                legit_conf,
                bins=20,
                range=(0.0, 1.0),
                alpha=0.65,
                edgecolor="black",
                label="Legitimate",
            )
        if len(adv_conf) > 0:
            plt.hist(
                adv_conf,
                bins=20,
                range=(0.0, 1.0),
                alpha=0.65,
                edgecolor="black",
                label="Adversarial",
            )
        if len(legit_conf) > 0 or len(adv_conf) > 0:
            plt.legend()
    else:
        plt.hist(confidence, bins=20, range=(0.0, 1.0), alpha=0.85, edgecolor="black")

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    out_path = os.path.join(outdir, "confidence_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_confidence_summary(df: pd.DataFrame):
    trust = df["trust_score"].to_numpy(dtype=np.float64)
    confidence = _confidence_from_trust(trust)

    summary = {
        "n": len(confidence),
        "trust_mean": float(np.mean(trust)),
        "trust_std": float(np.std(trust)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
        "confidence_min": float(np.min(confidence)),
        "confidence_q10": float(np.quantile(confidence, 0.10)),
        "confidence_median": float(np.quantile(confidence, 0.50)),
        "confidence_q90": float(np.quantile(confidence, 0.90)),
        "confidence_max": float(np.max(confidence)),
        "extreme_fraction": float(np.mean((trust <= 0.1) | (trust >= 0.9))),
    }

    if "true_label" in df.columns and (df["true_label"] != -1).any():
        legit = df[df["true_label"] == 0]["trust_score"].to_numpy(dtype=np.float64)
        adv = df[df["true_label"] == 1]["trust_score"].to_numpy(dtype=np.float64)
        if len(legit) > 0:
            summary["legit_count"] = int(len(legit))
            summary["legit_trust_mean"] = float(np.mean(legit))
            summary["legit_trust_std"] = float(np.std(legit))
        if len(adv) > 0:
            summary["adv_count"] = int(len(adv))
            summary["adv_trust_mean"] = float(np.mean(adv))
            summary["adv_trust_std"] = float(np.std(adv))

    out_csv = "results/stats/confidence_distribution_summary.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def plot_reliability_curve(df: pd.DataFrame, outdir: str, tau_min: float):
    if "true_label" not in df.columns:
        print("Skipping reliability curve: true_label column not found.")
        return

    labeled = df[df["true_label"] != -1].copy()
    if labeled.empty:
        print("Skipping reliability curve: no valid labels in input.")
        return

    trust = labeled["trust_score"].to_numpy(dtype=np.float64)
    p_adv = 1.0 - trust
    threshold = 1.0 - tau_min

    pred_adv = (p_adv > threshold).astype(int)
    true_adv = labeled["true_label"].to_numpy(dtype=np.int64)
    correctness = (pred_adv == true_adv).astype(np.float64)

    confidence = np.maximum(p_adv, 1.0 - p_adv)
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidence, bins) - 1

    x_conf = []
    y_acc = []
    empty_bins = 0
    for b in range(10):
        mask = bin_ids == b
        if np.any(mask):
            x_conf.append(float(np.mean(confidence[mask])))
            y_acc.append(float(np.mean(correctness[mask])))
        else:
            empty_bins += 1

    if not x_conf:
        print("Skipping reliability curve: no populated confidence bins.")
        return

    if empty_bins > 0:
        print(
            f"Warning: {empty_bins}/10 reliability bins are empty; "
            "plotting populated bins only."
        )

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    plt.plot(x_conf, y_acc, marker="o", linewidth=2.0, label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, "reliability_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate trust calibration diagnostics")
    parser.add_argument("--input", default="results/processed/trust_scores.csv")
    parser.add_argument("--outdir", default="results/plots")
    parser.add_argument("--tau-min", type=float, default=0.65)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    _ensure_output_dirs(args.outdir)
    df = pd.read_csv(args.input)

    if "trust_score" not in df.columns:
        raise ValueError("Input CSV must contain 'trust_score' column.")

    plot_trust_histogram(df, args.outdir)
    plot_confidence_distribution(df, args.outdir)
    save_confidence_summary(df)
    plot_reliability_curve(df, args.outdir, tau_min=args.tau_min)


if __name__ == "__main__":
    main()
