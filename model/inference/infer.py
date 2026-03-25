"""
Trust Inference Script
=======================
Runs the trained transformer model on a behavioral sequence file and
outputs per-agent trust scores with anomaly decisions.

Usage:
    python model/inference/infer.py \
        --checkpoint model/checkpoints/best_model.pt \
        --config model/configs/transformer_config.json \
        --sequences data/sample/sample_sequences.json \
        --output data/processed/trust_scores.csv \
        --tau-min 0.65
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
    TrustTransformer, compute_trust_score
)


def run_inference(checkpoint_path: str, config_path: str,
                  sequences_path: str, output_path: str,
                  tau_min: float = 0.65) -> pd.DataFrame:
    """
    Run inference on a JSON file of behavioral sequences.

    Returns a DataFrame with columns:
        agent_id, trust_score, is_anomalous, true_label, attack_type
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    with open(sequences_path, "r") as f:
        sequences = json.load(f)

    seq_len = config["architecture"]["seq_len"]

    # Load model
    model = TrustTransformer(config)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True,
                                          map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found ({checkpoint_path}). "
              f"Using untrained model — scores will be random.")
    model.eval()

    results = []
    for item in sequences:
        seq = np.array(item["sequence"], dtype=np.float32)
        if seq.shape[0] < seq_len:
            pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:seq_len]

        score = compute_trust_score(model, seq)
        is_anomalous = score < tau_min

        results.append({
            "agent_id":    item.get("agent_id", -1),
            "trust_score": round(score, 4),
            "is_anomalous": int(is_anomalous),
            "true_label":  item.get("label", -1),
            "attack_type": item.get("attack_type", "unknown"),
        })

    df = pd.DataFrame(results)

    # Compute accuracy if labels are available
    if "true_label" in df.columns and (df["true_label"] != -1).all():
        accuracy = (df["is_anomalous"] == df["true_label"]).mean()
        n_detected = (df["is_anomalous"] == 1).sum()
        n_actual_adv = (df["true_label"] == 1).sum()
        print(f"\nInference Results:")
        print(f"  Sequences evaluated: {len(df)}")
        print(f"  Anomalies detected:  {n_detected}")
        print(f"  True adversarial:    {n_actual_adv}")
        print(f"  Accuracy:            {accuracy:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved results to:    {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run behavioral trust inference on agent sequences"
    )
    parser.add_argument("--checkpoint",
                        default="model/checkpoints/best_model.pt")
    parser.add_argument("--config",
                        default="model/configs/transformer_config.json")
    parser.add_argument("--sequences",
                        default="data/sample/sample_sequences.json")
    parser.add_argument("--output",
                        default="data/processed/trust_scores.csv")
    parser.add_argument("--tau-min", type=float, default=0.65)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        sequences_path=args.sequences,
        output_path=args.output,
        tau_min=args.tau_min,
    )


if __name__ == "__main__":
    main()
