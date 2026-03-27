"""
Run OOD evaluation for trained transformer checkpoints.

Evaluates a trained model under distribution shifts across datasets that vary
adversarial fraction and noise level, and saves metrics in long format:

    dataset_name,metric_name,value

Metrics:
  C, RP, P, A, TI, ECE, Brier

Examples:
  python scripts/run_ood_evaluation.py \
      --checkpoint model/checkpoints/best_model.pt

  python scripts/run_ood_evaluation.py \
      --checkpoint model/checkpoints/best_model.pt \
      --generate-datasets \
      --adversarial-fractions 0.05,0.15,0.30 \
      --noise-levels 0.04,0.08,0.12
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python
from model.inference.transformer_model import (
    TrustDataset,
    TrustTransformer,
    _adv_threshold,
    set_seeds,
    temperature_scale_logits,
)
from simulation.scripts.generate_behavioral_data import (
    _format_float_token,
    _parse_attack_mix_spec,
    _parse_float_list,
    generate_dataset,
)


def _extract_dataset_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _is_sequence_dataset(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith(".json") and ("_labels" not in base)


def _evaluate_dataset(
    model: TrustTransformer,
    sequences: list,
    seq_len: int,
    batch_size: int,
    tau_min: float,
    temperature: float,
) -> Dict[str, float]:
    ds = TrustDataset(sequences, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    labels = [item.get("label", 0) for item in sequences]
    n_adv = int(sum(int(l) for l in labels))
    n_legit = max(len(labels) - n_adv, 0)
    pos_weight = (n_legit / max(n_adv, 1)) if len(labels) else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )

    model.eval()
    threshold = _adv_threshold(tau_min)
    total_loss = 0.0
    all_preds: List[float] = []
    all_labels: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch_sequences, batch_labels in loader:
            logits = model(batch_sequences)
            loss = criterion(logits, batch_labels)
            total_loss += loss.item() * batch_sequences.size(0)

            probs = torch.sigmoid(temperature_scale_logits(logits, temperature))
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(batch_labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    y_pred = np.array(all_preds, dtype=np.float32)
    y_true = np.array(all_labels, dtype=np.float32)
    p_adv = np.array(all_probs, dtype=np.float32)

    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)

    brier = float(np.mean((p_adv - y_true) ** 2)) if len(y_true) else 0.0

    n_bins = 10
    ece = 0.0
    if len(y_true) > 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            lo = edges[i]
            hi = edges[i + 1]
            if i == n_bins - 1:
                in_bin = (p_adv >= lo) & (p_adv <= hi)
            else:
                in_bin = (p_adv >= lo) & (p_adv < hi)

            if not np.any(in_bin):
                continue

            p_mean = float(p_adv[in_bin].mean())
            y_mean = float(y_true[in_bin].mean())
            weight = float(in_bin.mean())
            ece += weight * abs(p_mean - y_mean)

    trust_scores = 1.0 - p_adv
    trust_index = float(trust_scores.mean()) if len(trust_scores) else 0.0

    return {
        "C": float(accuracy),
        "RP": float(recall),
        "P": float(precision),
        "A": float(f1),
        "TI": float(trust_index),
        "ECE": float(ece),
        "Brier": float(brier),
        "loss": float(total_loss / max(len(y_true), 1)),
    }


def _maybe_generate_ood_datasets(
    out_dir: str,
    num_sequences: int,
    seq_len: int,
    seed: int,
    adversarial_fractions: List[float],
    noise_levels: List[float],
    attack_mix_spec: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    generated_paths: List[str] = []

    attack_mix = _parse_attack_mix_spec(attack_mix_spec)

    idx = 0
    for adv in adversarial_fractions:
        for noise in noise_levels:
            cfg_seed = seed + idx * 10000
            idx += 1

            sequences, _ = generate_dataset(
                num_sequences=num_sequences,
                K=seq_len,
                adv_fraction=adv,
                seed=cfg_seed,
                noise_sigma=noise,
                attack_mix=attack_mix,
            )

            name = f"data_adv{_format_float_token(adv)}_noise{_format_float_token(noise)}"
            if attack_mix is not None:
                name = f"{name}_custommix"
            path = os.path.join(out_dir, f"{name}.json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(sequences, f, indent=2)

            generated_paths.append(path)

    return generated_paths


def _parse_adv_noise_from_name(name: str) -> Tuple[float, float]:
    m = re.search(r"adv([0-9]+(?:\.[0-9]+)?)_noise([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    ensure_supported_python()

    parser = argparse.ArgumentParser(description="Run OOD robustness evaluation across shifted datasets")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", default="model/configs/transformer_config.json")
    parser.add_argument("--dataset-pattern", default="data/raw/data_adv*_noise*.json")
    parser.add_argument("--output", default="analysis/stats/ood_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tau-min", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--generate-datasets", action="store_true")
    parser.add_argument("--ood-out-dir", default="data/raw")
    parser.add_argument("--num-sequences", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--adversarial-fractions", default="0.05,0.15,0.3,0.45")
    parser.add_argument("--noise-levels", default="0.04,0.08,0.12")
    parser.add_argument("--attack-mix", default="uniform")

    args = parser.parse_args()

    set_seeds(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seq_len = int(args.seq_len or config["architecture"]["seq_len"])
    batch_size = int(args.batch_size or config["training"].get("batch_size", 64))
    tau_min = float(args.tau_min if args.tau_min is not None else config["trust_scoring"]["threshold_tau_min"])
    temperature = float(
        args.temperature
        if args.temperature is not None
        else config.get("trust_scoring", {}).get("inference_temperature", 1.5)
    )

    model = TrustTransformer(config)
    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    )
    model.eval()

    generated_paths: List[str] = []
    if args.generate_datasets:
        adv_values = _parse_float_list(args.adversarial_fractions)
        noise_values = _parse_float_list(args.noise_levels)
        generated_paths = _maybe_generate_ood_datasets(
            out_dir=args.ood_out_dir,
            num_sequences=args.num_sequences,
            seq_len=seq_len,
            seed=args.seed,
            adversarial_fractions=adv_values,
            noise_levels=noise_values,
            attack_mix_spec=args.attack_mix,
        )

    dataset_paths = sorted(
        [p for p in glob.glob(args.dataset_pattern) if _is_sequence_dataset(p)]
    )
    if generated_paths:
        generated_set = set(os.path.abspath(p) for p in generated_paths)
        dataset_paths = [p for p in dataset_paths if os.path.abspath(p) in generated_set]

    if not dataset_paths:
        raise FileNotFoundError(
            "No OOD dataset files matched. "
            f"Pattern used: {args.dataset_pattern}. "
            "Generate datasets first or adjust --dataset-pattern."
        )

    rows: List[Dict[str, object]] = []
    for path in dataset_paths:
        with open(path, "r", encoding="utf-8") as f:
            sequences = json.load(f)

        dataset_name = _extract_dataset_name(path)
        adv_fraction, noise_level = _parse_adv_noise_from_name(dataset_name)

        metrics = _evaluate_dataset(
            model=model,
            sequences=sequences,
            seq_len=seq_len,
            batch_size=batch_size,
            tau_min=tau_min,
            temperature=temperature,
        )

        for metric_name in ["C", "RP", "P", "A", "TI", "ECE", "Brier"]:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "metric_name": metric_name,
                    "value": float(metrics[metric_name]),
                    "adversarial_fraction": adv_fraction,
                    "noise_level": noise_level,
                }
            )

        print(
            f"Evaluated {dataset_name}: "
            f"C={metrics['C']:.4f} RP={metrics['RP']:.4f} P={metrics['P']:.4f} "
            f"A={metrics['A']:.4f} TI={metrics['TI']:.4f} "
            f"ECE={metrics['ECE']:.4f} Brier={metrics['Brier']:.4f}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df = pd.DataFrame(rows)[["dataset_name", "metric_name", "value"]]
    out_df.to_csv(args.output, index=False)
    print(f"Saved OOD evaluation results: {args.output}")


if __name__ == "__main__":
    main()
