"""
LSTM Baseline for Behavioral Trust Inference
============================================
Trains an LSTM-based baseline on the same sequence inputs and identical
stratified train/val/test split used by the transformer model.

Outputs per-split metrics in the same schema used by other baselines so
comparison is straightforward.

Usage:
  python model/inference/lstm_baseline.py \
      --config model/configs/transformer_config.json \
      --data data/raw/behavioral_sequences.json \
      --output analysis/stats/lstm_baseline_results.csv
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.transformer_model import (  # pylint: disable=wrong-import-position
    TrustDataset,
    TrustTransformer,
    _adv_threshold,
    set_seeds,
    stratified_split,
    temperature_scale_logits,
)


class LSTMBaseline(nn.Module):
    """Sequence classifier baseline that outputs raw logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        return self.head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _compute_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    tau_min: float,
    temperature: float,
) -> Dict[str, float]:
    p_adv = 1.0 / (1.0 + np.exp(-(logits / max(temperature, 1e-6))))
    threshold = _adv_threshold(tau_min)
    preds = (p_adv > threshold).astype(np.int32)
    labels_i = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels_i == 1)).sum())
    tn = int(((preds == 0) & (labels_i == 0)).sum())
    fp = int(((preds == 1) & (labels_i == 0)).sum())
    fn = int(((preds == 0) & (labels_i == 1)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    trust_scores = 1.0 - p_adv
    ti = float(trust_scores.mean()) if trust_scores.size else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "C": float(accuracy),
        "RP": float(recall),
        "P": float(precision),
        "A": float(f1),
        "TI": float(ti),
        "trust_mean": float(ti),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for sequences, labels in loader:
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * sequences.size(0)
        total += int(sequences.size(0))
    return total_loss / max(total, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    tau_min: float,
    temperature: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * sequences.size(0)
            total += int(sequences.size(0))

            scaled_logits = temperature_scale_logits(logits, temperature)
            all_logits.extend(scaled_logits.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())

    mean_loss = total_loss / max(total, 1)
    metrics = _compute_metrics_from_logits(
        logits=np.array(all_logits, dtype=np.float64),
        labels=np.array(all_labels, dtype=np.float64),
        tau_min=tau_min,
        temperature=1.0,
    )
    return mean_loss, metrics


def run_lstm_baseline(
    config_path: str,
    data_path: str,
    output_path: str,
    checkpoint_dir: str,
    hidden_dim: int,
    num_layers: int,
    bidirectional: bool,
) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    seed = int(config["training"]["random_seed"])
    set_seeds(seed)

    train_data, val_data, test_data = stratified_split(
        sequences,
        train_frac=float(config["training"]["train_split"]),
        val_frac=float(config["training"]["val_split"]),
        seed=seed,
    )

    seq_len = int(config["architecture"]["seq_len"])
    input_dim = int(config["architecture"]["input_dim"])
    tau_min = float(config["trust_scoring"]["threshold_tau_min"])
    temperature = float(config.get("trust_scoring", {}).get("temperature", 1.5))

    train_ds = TrustDataset(train_data, seq_len=seq_len)
    val_ds = TrustDataset(val_data, seq_len=seq_len)
    test_ds = TrustDataset(test_data, seq_len=seq_len)

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_legit, n_adv, pos_weight = train_ds.class_weights()

    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=float(config["architecture"].get("dropout", 0.2)),
        bidirectional=bidirectional,
    )

    transformer_ref = TrustTransformer(config)
    lstm_params = model.count_parameters()
    transformer_params = transformer_ref.count_parameters()

    print("=== LSTM Baseline ===")
    print(f"  LSTM params       : {lstm_params:,}")
    print(f"  Transformer params: {transformer_params:,}")
    if lstm_params > (transformer_params * 1.25):
        print("  [WARNING] LSTM parameter count is substantially larger than transformer.")
    else:
        print("  [OK] LSTM parameter scale is comparable.")
    print(f"  Train class ratio : legit={n_legit}, adv={n_adv}, pos_weight={pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=max(float(config["training"].get("weight_decay", 0.0)), 1e-3),
    )

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    best_val_acc = -1.0
    patience_count = 0

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best_lstm_baseline.pt")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = _evaluate(model, val_loader, criterion, tau_min=tau_min, temperature=temperature)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_metrics['accuracy']:.4f} "
                f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f}"
            )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (best val acc={best_val_acc:.4f})")
                break

    model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location="cpu"))

    _, train_metrics = _evaluate(model, train_loader, criterion, tau_min=tau_min, temperature=temperature)
    _, val_metrics = _evaluate(model, val_loader, criterion, tau_min=tau_min, temperature=temperature)
    _, test_metrics = _evaluate(model, test_loader, criterion, tau_min=tau_min, temperature=temperature)

    rows = []
    for split_name, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        rows.append({"model": "lstm_baseline", "split": split_name, **metrics})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM baseline")
    parser.add_argument("--config", default="model/configs/transformer_config.json")
    parser.add_argument("--data", default="data/raw/behavioral_sequences.json")
    parser.add_argument("--output", default="analysis/stats/lstm_baseline_results.csv")
    parser.add_argument("--checkpoint-dir", default="model/checkpoints/")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_true",
                        help="Enable bidirectional LSTM (default: enabled).")
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false",
                        help="Disable bidirectional LSTM.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Data not found: {args.data}. Generate it first with simulation/scripts/generate_behavioral_data.py"
        )

    results = run_lstm_baseline(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
    )

    print("\n=== LSTM Baseline Results ===")
    print(results.to_string(index=False))
    print(f"\nSaved LSTM baseline metrics to: {args.output}")


if __name__ == "__main__":
    main()
