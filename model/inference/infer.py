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
        --checkpoint results/checkpoints/best_model.pt \
        --config     model/configs/transformer_config.json \
        --sequences  data/sample/sample_sequences.json \
        --output     results/processed/trust_scores.csv \
        --tau-min    0.65

    # Test with alternative threshold:
    python model/inference/infer.py --tau-min 0.5
    python model/inference/infer.py --tau-min 0.7
"""

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from compat import ensure_supported_python
from model.inference.transformer_model import (
    TrustTransformer,
    load_model,
    normalize_sequence_features,
    _prepare_sequence_array,
    set_seeds,
)


def _pad_or_truncate_sequence(seq_list, seq_len: int) -> np.ndarray:
    seq = np.array(seq_list, dtype=np.float32)
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:seq_len]
    return seq


def _load_json_sequences(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def _stratified_calibration_set(sequences: list, min_per_class: int = 20) -> Tuple[list, dict]:
    legit = [item for item in sequences if item.get("label", -1) == 0]
    adv = [item for item in sequences if item.get("label", -1) == 1]

    if len(adv) == 0:
        raise ValueError("Calibration set contains zero adversarial samples.")
    if len(legit) == 0:
        raise ValueError("Calibration set contains zero legitimate samples.")

    if len(legit) < min_per_class or len(adv) < min_per_class:
        return sequences, {
            "mode": "full_set",
            "source_legit": len(legit),
            "source_adv": len(adv),
            "eval_legit": len(legit),
            "eval_adv": len(adv),
        }

    n_each = min(len(legit), len(adv))
    return legit[:n_each] + adv[:n_each], {
        "mode": "stratified_balanced",
        "source_legit": len(legit),
        "source_adv": len(adv),
        "eval_legit": n_each,
        "eval_adv": n_each,
    }


def _compute_stratified_calibration_stats(model, sequences: list, seq_len: int, temperature: float) -> dict:
    trust_legit = []
    trust_adv = []

    model.eval()
    with torch.no_grad():
        for item in sequences:
            seq = _pad_or_truncate_sequence(item["sequence"], seq_len)
            logit = float(model(torch.from_numpy(seq).unsqueeze(0)).squeeze())
            p_adv = 1.0 / (1.0 + np.exp(-(logit / temperature)))
            trust = 1.0 - p_adv
            if item.get("label", -1) == 0:
                trust_legit.append(trust)
            elif item.get("label", -1) == 1:
                trust_adv.append(trust)

    if len(trust_adv) == 0:
        raise ValueError("Stratified calibration stats failed: adversarial count is zero.")

    legit_arr = np.array(trust_legit, dtype=np.float64)
    adv_arr = np.array(trust_adv, dtype=np.float64)
    combined = np.array(trust_legit + trust_adv, dtype=np.float64)

    return {
        "legit_count": int(len(legit_arr)),
        "adv_count": int(len(adv_arr)),
        "legit_mean": float(np.mean(legit_arr)),
        "legit_std": float(np.std(legit_arr)),
        "adv_mean": float(np.mean(adv_arr)),
        "adv_std": float(np.std(adv_arr)),
        "combined_mean": float(np.mean(combined)),
        "combined_std": float(np.std(combined)),
    }


def _choose_calibration_sequences(primary_sequences: list, fallback_path: str) -> Tuple[list, str]:
    has_labels = all("label" in item for item in primary_sequences)
    legit = sum(1 for item in primary_sequences if item.get("label", -1) == 0)
    adv = sum(1 for item in primary_sequences if item.get("label", -1) == 1)

    if has_labels and legit > 0 and adv > 0:
        return primary_sequences, "primary"

    if os.path.exists(fallback_path):
        fallback_sequences = _load_json_sequences(fallback_path)
        return fallback_sequences, "fallback"

    return primary_sequences, "primary"


def run_inference(
    checkpoint_path: str,
    config_path: str,
    sequences_path: str,
    output_path: str,
    tau_min: float = 0.65,
    temperature: float = 1.5,
    calibration_sequences_path: str = "data/raw/behavioral_sequences.json",
    calibration_min_per_class: int = 20,
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
        temperature   — temperature used in sigmoid(logit / T)

    FIX vs v1:
        v1 called model(x) and returned sigmoid(logit) = P(adversarial).
        This caused all legitimate agents (low P(adv) → low score) to be
        flagged as anomalous when compared against tau_min=0.65.
        Fixed via compute_trust_score() → 1 - sigmoid(logit).
    """
    set_seeds(seed)

    with open(config_path) as f:
        config = json.load(f)
    sequences = _load_json_sequences(sequences_path)

    seq_len = config["architecture"]["seq_len"]

    # Load model
    if os.path.exists(checkpoint_path):
        model = load_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            quantized=False,
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            f"WARNING: Checkpoint not found ({checkpoint_path}). "
            "Using untrained model — scores will be near-random."
        )
        model = TrustTransformer(config)
    model.eval()

    results = []
    for item in sequences:
        seq = _pad_or_truncate_sequence(item["sequence"], seq_len)
        seq = _prepare_sequence_array(seq, expected_dim=int(config["architecture"]["input_dim"]))
        seq = normalize_sequence_features(
            seq,
            getattr(model, "feature_mean", None),
            getattr(model, "feature_std", None),
        )

        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0)
            logit = float(model(x).squeeze())

        # Calibrated trust: 1 - sigmoid(logit / T)
        trust_score  = 1.0 - (1.0 / (1.0 + np.exp(-(logit / temperature))))
        is_anomalous = int(trust_score < tau_min)

        results.append({
            "agent_id":    item.get("agent_id", -1),
            "trust_score": round(trust_score, 4),
            "is_anomalous": is_anomalous,
            "true_label":  item.get("label", -1),
            "attack_type": item.get("attack_type", "unknown"),
            "temperature": temperature,
        })

    df = pd.DataFrame(results)

    # ── Primary output summary ─────────────────────────────────────────────
    n_flagged  = int(df["is_anomalous"].sum())
    score_min  = float(df["trust_score"].min())
    score_max  = float(df["trust_score"].max())
    score_mean = float(df["trust_score"].mean())
    score_std  = float(df["trust_score"].std())

    print(f"\nInference Results  (tau_min={tau_min})")
    print(f"  Sequences evaluated : {len(df)}")
    print(f"  Flagged as anomalous: {n_flagged} ({100*n_flagged/len(df):.1f}%)")
    print(f"  Temperature         : {temperature:.2f}")
    print(f"  Trust score stats   : "
          f"min={score_min:.3f}  max={score_max:.3f}  "
          f"mean={score_mean:.3f}  std={score_std:.3f}")

    # ── Stratified calibration validation (mixed set) ─────────────────────
    calibration_sequences, calibration_source = _choose_calibration_sequences(
        primary_sequences=sequences,
        fallback_path=calibration_sequences_path,
    )

    calibration_subset, calibration_meta = _stratified_calibration_set(
        calibration_sequences,
        min_per_class=calibration_min_per_class,
    )
    calibration_stats = _compute_stratified_calibration_stats(
        model,
        calibration_subset,
        seq_len=seq_len,
        temperature=temperature,
    )

    print("\nCalibration Validation")
    print(f"  Calibration source  : {calibration_source}")
    print(f"  Calibration mode    : {calibration_meta['mode']}")
    print(f"  Source counts       : legit={calibration_meta['source_legit']} adv={calibration_meta['source_adv']}")
    print(f"  Eval counts         : legit={calibration_stats['legit_count']} adv={calibration_stats['adv_count']}")
    print(f"  Legit trust         : mean={calibration_stats['legit_mean']:.4f} std={calibration_stats['legit_std']:.4f}")
    print(f"  Adv trust           : mean={calibration_stats['adv_mean']:.4f} std={calibration_stats['adv_std']:.4f}")
    print(f"  Combined trust      : mean={calibration_stats['combined_mean']:.4f} std={calibration_stats['combined_std']:.4f}")

    # Correct warning logic (stratified combined std only)
    if calibration_stats["combined_std"] < 0.05:
        print(
            "  [WARNING] Stratified trust spread is too narrow "
            f"(combined std={calibration_stats['combined_std']:.4f} < 0.05)."
        )

    # ── Accuracy if ground-truth labels available ─────────────────────────
    if (df["true_label"] != -1).all():
        n_true_adv = (df["true_label"] == 1).sum()
        n_true_legit = (df["true_label"] == 0).sum()
        tp = ((df["is_anomalous"] == 1) & (df["true_label"] == 1)).sum()
        tn = ((df["is_anomalous"] == 0) & (df["true_label"] == 0)).sum()
        fp = ((df["is_anomalous"] == 1) & (df["true_label"] == 0)).sum()
        fn = ((df["is_anomalous"] == 0) & (df["true_label"] == 1)).sum()
        accuracy  = (tp + tn) / len(df)

        print(f"  True legitimate     : {n_true_legit}")
        print(f"  True adversarial    : {n_true_adv}")
        print(f"  Accuracy            : {accuracy:.4f}  ({accuracy*100:.1f}%)")
        if n_true_adv > 0 and n_true_legit > 0:
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            print(f"  Precision           : {precision:.4f}")
            print(f"  Recall              : {recall:.4f}")
            print(f"  F1 Score            : {f1:.4f}")
            print(f"  Confusion (TP/TN/FP/FN): {tp}/{tn}/{fp}/{fn}")
        else:
            print("  Precision/Recall/F1 : N/A (single-class labeled inference set)")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    return df


def main():
    ensure_supported_python()
    parser = argparse.ArgumentParser(
        description="Run behavioral trust inference [FIXED v2]"
    )
    parser.add_argument("--checkpoint",
                        default="results/checkpoints/best_model.pt")
    parser.add_argument("--config",
                        default="model/configs/transformer_config.json")
    parser.add_argument("--sequences",
                        default="data/sample/sample_sequences.json")
    parser.add_argument("--output",
                        default="results/processed/trust_scores.csv")
    parser.add_argument("--tau-min",   type=float, default=0.65,
                        help="Trust threshold (paper default=0.65). "
                             "Test with 0.5 and 0.7 to verify non-collapse.")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Temperature for calibrated trust score: trust=1-sigmoid(logit/T).")
    parser.add_argument("--calibration-sequences",
                        default="data/raw/behavioral_sequences.json",
                        help="Fallback mixed-label dataset for calibration validation.")
    parser.add_argument("--calibration-min-per-class", type=int, default=20,
                        help="Minimum samples per class for stratified calibration checks.")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    print("=== Trust Inference [v2 — FIXED] ===")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Sequences  : {args.sequences}")
    print(f"  tau_min    : {args.tau_min}")
    print(f"  temperature: {args.temperature}")
    print(f"  P(adv) thr : {1 - args.tau_min:.2f}  "
          f"(flag when P(adversarial) > {1-args.tau_min:.2f})")

    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        sequences_path=args.sequences,
        output_path=args.output,
        tau_min=args.tau_min,
        temperature=args.temperature,
        calibration_sequences_path=args.calibration_sequences,
        calibration_min_per_class=args.calibration_min_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
