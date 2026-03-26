"""
Trust Inference Script  [FIXED v2]
=====================================
Runs the trained transformer on a behavioral sequence file and outputs
per-agent trust scores with anomaly decisions.

Changes vs v1:
  - Uses compute_trust_score() which returns 1 - sigmoid(logit) (correct).
  - v1 returned sigmoid(logit) = P(adversarial), causing every legitimate
    agent to appear anomalous (RC-5 fix).
  - Anomaly flag: trust_score < tau_min  (unchanged logic, now correct).
  - Added score distribution summary to detect single-class collapse.
  - Added --threshold override for flexible testing (0.5 and 0.7 testable).

Usage:
    python model/inference/infer.py \
        --checkpoint model/checkpoints/best_model.pt \
        --config     model/configs/transformer_config.json \
        --sequences  data/sample/sample_sequences.json \
        --output     data/processed/trust_scores.csv \
        --tau-min    0.65

    # Test with alternative threshold:
    python model/inference/infer.py --tau-min 0.5
    python model/inference/infer.py --tau-min 0.7
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.transformer_model import (
    TrustTransformer,
    compute_trust_score,
    set_seeds,
)


def run_inference(
    checkpoint_path: str,
    config_path: str,
    sequences_path: str,
    output_path: str,
    tau_min: float = 0.65,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run inference on a JSON file of behavioral sequences.

    Columns in output DataFrame:
        agent_id      — agent identifier
        trust_score   — float in [0,1]; HIGH = trustworthy (1 - sigmoid(logit))
        is_anomalous  — 1 if trust_score < tau_min, else 0
        true_label    — ground-truth label (0=legit, 1=adv); -1 if unavailable
        attack_type   — attack type string or 'unknown'

    FIX vs v1:
        v1 called model(x) and returned sigmoid(logit) = P(adversarial).
        This caused all legitimate agents (low P(adv) → low score) to be
        flagged as anomalous when compared against tau_min=0.65.
        Fixed via compute_trust_score() → 1 - sigmoid(logit).
    """
    set_seeds(seed)

    with open(config_path) as f:
        config = json.load(f)
    with open(sequences_path) as f:
        sequences = json.load(f)

    seq_len = config["architecture"]["seq_len"]

    # Load model
    model = TrustTransformer(config)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            f"WARNING: Checkpoint not found ({checkpoint_path}). "
            "Using untrained model — scores will be near-random."
        )
    model.eval()

    results = []
    for item in sequences:
        seq = np.array(item["sequence"], dtype=np.float32)
        # Pad / truncate to seq_len
        if seq.shape[0] < seq_len:
            pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:seq_len]

        # compute_trust_score returns 1 - sigmoid(logit)  [RC-5 fix]
        trust_score  = compute_trust_score(model, seq)
        is_anomalous = int(trust_score < tau_min)

        results.append({
            "agent_id":    item.get("agent_id", -1),
            "trust_score": round(trust_score, 4),
            "is_anomalous": is_anomalous,
            "true_label":  item.get("label", -1),
            "attack_type": item.get("attack_type", "unknown"),
        })

    df = pd.DataFrame(results)

    # ── Score distribution check — detects single-class collapse ──────────
    n_flagged  = df["is_anomalous"].sum()
    score_min  = df["trust_score"].min()
    score_max  = df["trust_score"].max()
    score_mean = df["trust_score"].mean()
    score_std  = df["trust_score"].std()

    print(f"\nInference Results  (tau_min={tau_min})")
    print(f"  Sequences evaluated : {len(df)}")
    print(f"  Flagged as anomalous: {n_flagged} ({100*n_flagged/len(df):.1f}%)")
    print(f"  Trust score stats   : "
          f"min={score_min:.3f}  max={score_max:.3f}  "
          f"mean={score_mean:.3f}  std={score_std:.3f}")

    # Collapse warning
    if score_std < 0.01:
        print(
            "  [WARNING] All trust scores are nearly identical "
            f"(std={score_std:.4f}). "
            "Model may not be trained. Run train.py first."
        )
    elif n_flagged == len(df):
        print(
            "  [WARNING] All agents flagged as anomalous. "
            "Consider lowering tau_min or checking model checkpoint."
        )
    elif n_flagged == 0:
        print(
            "  [WARNING] No agents flagged as anomalous. "
            "Consider raising tau_min or checking model checkpoint."
        )

    # ── Accuracy if ground-truth labels available ─────────────────────────
    if (df["true_label"] != -1).all():
        n_true_adv = (df["true_label"] == 1).sum()
        tp = ((df["is_anomalous"] == 1) & (df["true_label"] == 1)).sum()
        tn = ((df["is_anomalous"] == 0) & (df["true_label"] == 0)).sum()
        fp = ((df["is_anomalous"] == 1) & (df["true_label"] == 0)).sum()
        fn = ((df["is_anomalous"] == 0) & (df["true_label"] == 1)).sum()
        accuracy  = (tp + tn) / len(df)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"  True adversarial    : {n_true_adv}")
        print(f"  Accuracy            : {accuracy:.4f}  ({accuracy*100:.1f}%)")
        print(f"  Precision           : {precision:.4f}")
        print(f"  Recall              : {recall:.4f}")
        print(f"  F1 Score            : {f1:.4f}")
        print(f"  Confusion (TP/TN/FP/FN): {tp}/{tn}/{fp}/{fn}")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run behavioral trust inference [FIXED v2]"
    )
    parser.add_argument("--checkpoint",
                        default="model/checkpoints/best_model.pt")
    parser.add_argument("--config",
                        default="model/configs/transformer_config.json")
    parser.add_argument("--sequences",
                        default="data/sample/sample_sequences.json")
    parser.add_argument("--output",
                        default="data/processed/trust_scores.csv")
    parser.add_argument("--tau-min",   type=float, default=0.65,
                        help="Trust threshold (paper default=0.65). "
                             "Test with 0.5 and 0.7 to verify non-collapse.")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    print("=== Trust Inference [v2 — FIXED] ===")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Sequences  : {args.sequences}")
    print(f"  tau_min    : {args.tau_min}")
    print(f"  P(adv) thr : {1 - args.tau_min:.2f}  "
          f"(flag when P(adversarial) > {1-args.tau_min:.2f})")

    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        sequences_path=args.sequences,
        output_path=args.output,
        tau_min=args.tau_min,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
