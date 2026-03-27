"""
Baseline Models for Trust Inference
===================================
Train and evaluate classical baselines on the same dataset and stratified
train/val/test split used by the transformer pipeline.

Models:
  - Logistic Regression
  - Random Forest

Output:
  CSV summary with one row per (model, split) including metrics compatible
  with the main model reporting (accuracy, precision, recall, F1, confusion
  counts) plus alias columns requested for downstream tables.

Usage:
  python model/inference/baseline_models.py \
      --config model/configs/transformer_config.json \
      --data data/raw/behavioral_sequences.json \
      --output analysis/stats/baseline_results.csv
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.transformer_model import _adv_threshold, stratified_split


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _pad_or_truncate_sequence(seq_list: List[List[float]], seq_len: int) -> np.ndarray:
    seq = np.array(seq_list, dtype=np.float32)
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:seq_len]
    return seq


def prepare_xy(sequences: List[dict], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sequence samples into fixed-size flattened feature vectors."""
    x_rows = []
    y_rows = []
    for item in sequences:
        seq = _pad_or_truncate_sequence(item["sequence"], seq_len)
        x_rows.append(seq.reshape(-1))
        y_rows.append(int(item["label"]))
    return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray


def build_splits(config: dict, sequences: List[dict]) -> Dict[str, SplitData]:
    train_data, val_data, test_data = stratified_split(
        sequences,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
        seed=config["training"]["random_seed"],
    )
    seq_len = int(config["architecture"]["seq_len"])

    x_train, y_train = prepare_xy(train_data, seq_len)
    x_val, y_val = prepare_xy(val_data, seq_len)
    x_test, y_test = prepare_xy(test_data, seq_len)

    return {
        "train": SplitData(x_train, y_train),
        "val": SplitData(x_val, y_val),
        "test": SplitData(x_test, y_test),
    }


def compute_metrics_from_proba(y_true: np.ndarray, p_adv: np.ndarray, tau_min: float) -> Dict[str, float]:
    """
    Compute metrics using the same threshold semantics as transformer pipeline.
    trust = 1 - p_adv, predicted adversarial if p_adv > (1 - tau_min).
    """
    threshold = _adv_threshold(tau_min)
    y_pred = (p_adv > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    trust_scores = 1.0 - p_adv
    ti = float(np.mean(trust_scores)) if trust_scores.size else 0.0

    # Alias columns requested by user/paper shorthand.
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "C": float(accuracy),
        "RP": float(recall),
        "P": float(precision),
        "A": float(f1),
        "TI": float(ti),
        "trust_mean": float(ti),
    }


def build_logistic_candidates(seed: int) -> List[Pipeline]:
    c_values = [0.5, 1.0, 2.0]
    models = []
    for c in c_values:
        models.append(
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c,
                            solver="liblinear",
                            class_weight="balanced",
                            random_state=seed,
                            max_iter=1000,
                        ),
                    ),
                ]
            )
        )
    return models


def build_rf_candidates(seed: int) -> List[RandomForestClassifier]:
    params = [
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 300, "max_depth": 16},
        {"n_estimators": 500, "max_depth": None},
    ]
    models = []
    for p in params:
        models.append(
            RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        )
    return models


def select_best_model(
    candidates: List,
    train: SplitData,
    val: SplitData,
    tau_min: float,
) -> object:
    """Model selection by validation accuracy under shared threshold semantics."""
    best_model = None
    best_val_acc = -1.0

    for model in candidates:
        model.fit(train.x, train.y)
        p_adv_val = model.predict_proba(val.x)[:, 1]
        val_metrics = compute_metrics_from_proba(val.y, p_adv_val, tau_min)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model = model

    return best_model


def evaluate_model(model, splits: Dict[str, SplitData], tau_min: float, model_name: str) -> pd.DataFrame:
    rows = []
    for split_name, split_data in splits.items():
        p_adv = model.predict_proba(split_data.x)[:, 1]
        metrics = compute_metrics_from_proba(split_data.y, p_adv, tau_min)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_baselines(config_path: str, data_path: str, output_path: str) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    seed = int(config["training"]["random_seed"])
    tau_min = float(config["trust_scoring"]["threshold_tau_min"])
    set_seeds(seed)

    splits = build_splits(config, sequences)

    logreg = select_best_model(
        build_logistic_candidates(seed),
        train=splits["train"],
        val=splits["val"],
        tau_min=tau_min,
    )

    rf = select_best_model(
        build_rf_candidates(seed),
        train=splits["train"],
        val=splits["val"],
        tau_min=tau_min,
    )

    results = pd.concat(
        [
            evaluate_model(logreg, splits, tau_min, "logistic_regression"),
            evaluate_model(rf, splits, tau_min, "random_forest"),
        ],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline trust models")
    parser.add_argument("--config", default="model/configs/transformer_config.json")
    parser.add_argument("--data", default="data/raw/behavioral_sequences.json")
    parser.add_argument("--output", default="analysis/stats/baseline_results.csv")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Data not found: {args.data}. Generate it first with simulation/scripts/generate_behavioral_data.py"
        )

    df = run_baselines(args.config, args.data, args.output)
    print("\n=== Baseline Model Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved baseline metrics to: {args.output}")


if __name__ == "__main__":
    main()
