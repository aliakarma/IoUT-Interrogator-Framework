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
    --output results/stats/lstm_baseline_results.csv
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.data_hardening import harden_training_sequences
from model.inference.transformer_model import (  # pylint: disable=wrong-import-position
    TemperatureScaler,
    TrustDataset,
    TrustTransformer,
    load_preprocessing_stats,
    normalize_sequence_features,
    set_seeds,
    stratified_split,
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
    threshold: float,
) -> Dict[str, float]:
    p_adv = 1.0 / (1.0 + np.exp(-logits))
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
    roc_auc = float(roc_auc_score(labels_i, p_adv)) if len(np.unique(labels_i)) > 1 else float("nan")
    pr_auc = float(average_precision_score(labels_i, p_adv)) if len(np.unique(labels_i)) > 1 else float("nan")
    brier = float(np.mean((p_adv - labels_i) ** 2))
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for idx in range(10):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (p_adv >= lo) & (p_adv <= hi) if idx == 9 else (p_adv >= lo) & (p_adv < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(p_adv[mask].mean()) - float(labels_i[mask].mean()))

    trust_scores = 1.0 - p_adv
    ti = float(trust_scores.mean()) if trust_scores.size else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": float(ece),
        "brier": float(brier),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold_used": float(threshold),
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
    threshold: float,
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

            scaled_logits = logits / max(temperature, 1e-6)
            all_logits.extend(scaled_logits.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())

    mean_loss = total_loss / max(total, 1)
    metrics = _compute_metrics_from_logits(
        logits=np.array(all_logits, dtype=np.float64),
        labels=np.array(all_labels, dtype=np.float64),
        threshold=threshold,
    )
    return mean_loss, metrics


def _tune_threshold(logits: np.ndarray, labels: np.ndarray, target_recall: float = 0.75) -> Dict[str, float]:
    rows = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        metrics = _compute_metrics_from_logits(logits, labels, threshold=float(threshold))
        rows.append(metrics)
    eligible = [row for row in rows if row["recall"] >= target_recall]
    if eligible:
        best = max(eligible, key=lambda row: (row["f1"], row["precision"], -row["threshold_used"]))
        best["selection_policy"] = "max_f1_subject_to_recall>=0.75"
    else:
        best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
        best["selection_policy"] = "max_f1_fallback"
    return best


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
    hardening_cfg = config.get("training", {}).get("data_hardening", {})
    train_data, hardening_report = harden_training_sequences(
        train_data,
        seed=seed,
        enabled=bool(hardening_cfg.get("enabled", True)),
        gaussian_sigma_frac=float(hardening_cfg.get("gaussian_sigma_frac", 0.10)),
        overlap_alpha_low=float(hardening_cfg.get("overlap_alpha_low", 0.60)),
        overlap_alpha_high=float(hardening_cfg.get("overlap_alpha_high", 0.90)),
        label_flip_fraction=float(hardening_cfg.get("label_flip_fraction", 0.03)),
        hard_duplicate_fraction=float(hardening_cfg.get("hard_duplicate_fraction", 0.08)),
        feature_dropout_rate=float(hardening_cfg.get("feature_dropout_rate", 0.08)),
        top_feature_fraction=float(hardening_cfg.get("top_feature_fraction", 0.08)),
    )

    seq_len = int(config["architecture"]["seq_len"])
    input_dim = int(config["architecture"]["input_dim"])
    temperature = float(config.get("trust_scoring", {}).get("temperature", 1.5))

    train_ds = TrustDataset(train_data, seq_len=seq_len)
    val_ds = TrustDataset(val_data, seq_len=seq_len)
    test_ds = TrustDataset(test_data, seq_len=seq_len)

    feature_mean, feature_std = load_preprocessing_stats(checkpoint_dir)
    if feature_mean is None or feature_std is None:
        stats_mean = np.vstack([seq for seq, _ in train_ds.data]).mean(axis=0, keepdims=True)
        stats_std = np.vstack([seq for seq, _ in train_ds.data]).std(axis=0, keepdims=True)
        feature_mean, feature_std = stats_mean.astype(np.float32), np.where(stats_std < 1e-6, 1.0, stats_std).astype(np.float32)
    for dataset in [train_ds, val_ds, test_ds]:
        for idx in range(len(dataset.data)):
            seq, label = dataset.data[idx]
            dataset.data[idx] = (normalize_sequence_features(seq, feature_mean, feature_std), label)

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
    print(
        f"  Hardened train    : {hardening_report.n_original} -> {hardening_report.n_augmented} "
        f"(label_flips={hardening_report.label_flip_count}, "
        f"hard_duplicates={hardening_report.hard_sample_duplicates})"
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=max(float(config["training"].get("weight_decay", 0.0)), 1e-3),
    )

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    best_val_score = -1.0
    patience_count = 0

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best_lstm_baseline.pt")

    selected_threshold = 0.35
    selected_temperature = temperature
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion)
        val_logits = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for sequences_batch, labels_batch in val_loader:
                val_logits.append(model(sequences_batch))
                val_labels.append(labels_batch)
        val_logits_tensor = torch.cat(val_logits, dim=0)
        val_labels_tensor = torch.cat(val_labels, dim=0)
        scaler = TemperatureScaler(init_temperature=temperature)
        optimizer_temp = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=20)
        criterion_temp = nn.BCEWithLogitsLoss()

        def _closure():
            optimizer_temp.zero_grad()
            loss = criterion_temp(val_logits_tensor / scaler.temperature, val_labels_tensor)
            loss.backward()
            return loss

        optimizer_temp.step(_closure)
        with torch.no_grad():
            scaler.temperature.clamp_(min=0.25, max=10.0)
        val_logits_np = (val_logits_tensor / scaler.temperature).detach().squeeze(-1).cpu().numpy()
        val_labels_np = val_labels_tensor.squeeze(-1).cpu().numpy()
        threshold_info = _tune_threshold(val_logits_np, val_labels_np)
        current_threshold = float(threshold_info["threshold_used"])
        current_temperature = float(scaler.temperature.item())
        val_loss, val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            threshold=current_threshold,
            temperature=current_temperature,
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_metrics['accuracy']:.4f} "
                f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f} "
                f"F1={val_metrics['f1']:.4f} thr={current_threshold:.3f} T={current_temperature:.3f}"
            )

        current_score = float(val_metrics["f1"] + 0.05 * val_metrics["recall"])
        if current_score > best_val_score:
            best_val_score = current_score
            patience_count = 0
            selected_threshold = current_threshold
            selected_temperature = current_temperature
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (best val score={best_val_score:.4f})")
                break

    model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location="cpu"))

    _, train_metrics = _evaluate(model, train_loader, criterion, threshold=selected_threshold, temperature=selected_temperature)
    _, val_metrics = _evaluate(model, val_loader, criterion, threshold=selected_threshold, temperature=selected_temperature)
    _, test_metrics = _evaluate(model, test_loader, criterion, threshold=selected_threshold, temperature=selected_temperature)

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
    parser.add_argument("--output", default="results/stats/lstm_baseline_results.csv")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/")
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
