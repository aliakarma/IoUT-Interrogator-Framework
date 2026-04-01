"""
Leakage-safe classical baselines for behavioral trust inference.

This module rebuilds the sklearn baselines with:
- split-before-transform discipline
- train-only preprocessing via sklearn Pipeline
- validation-only Platt scaling
- validation-only threshold tuning with recall-aware objective
- richer evaluation metrics for auditability
"""

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.data_hardening import harden_training_sequences
from model.inference.transformer_model import stratified_split


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
    x_rows = []
    y_rows = []
    for item in sequences:
        seq = _pad_or_truncate_sequence(item["sequence"], seq_len)
        x_rows.append(seq.reshape(-1))
        y_rows.append(int(item["label"]))
    return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def _hash_rows(x: np.ndarray) -> List[str]:
    hashes = []
    for row in x:
        hashes.append(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest())
    return hashes


def expected_calibration_error(y_true: np.ndarray, p_adv: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p_adv >= lo) & (p_adv <= hi)
        else:
            mask = (p_adv >= lo) & (p_adv < hi)
        if not np.any(mask):
            continue
        conf = float(p_adv[mask].mean())
        acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return float(ece)


def tune_threshold(
    y_true: np.ndarray,
    p_adv: np.ndarray,
    thresholds: np.ndarray | None = None,
    target_recall: float = 0.75,
) -> Dict[str, float]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05, dtype=np.float64)

    rows = []
    for threshold in thresholds:
        y_pred = (p_adv > threshold).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        accuracy = float((y_pred == y_true).mean())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    eligible = [row for row in rows if row["recall"] >= target_recall]
    if eligible:
        best = max(eligible, key=lambda row: (row["f1"], row["precision"], -row["threshold"]))
        policy = "max_f1_subject_to_recall>=0.75"
    else:
        best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
        policy = "max_f1_fallback"
    best["selection_policy"] = policy
    best["target_recall"] = float(target_recall)
    return best


def compute_metrics(
    y_true: np.ndarray,
    p_adv: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (p_adv > threshold).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    accuracy = float((y_pred == y_true).mean())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    roc_auc = float(roc_auc_score(y_true, p_adv)) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, p_adv)) if len(np.unique(y_true)) > 1 else float("nan")
    brier = float(np.mean((p_adv - y_true) ** 2))
    ece = expected_calibration_error(y_true, p_adv)
    trust_scores = 1.0 - p_adv
    ti = float(trust_scores.mean()) if trust_scores.size else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "brier": brier,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold_used": float(threshold),
        "C": accuracy,
        "RP": float(recall),
        "P": float(precision),
        "A": float(f1),
        "TI": float(ti),
        "trust_mean": float(ti),
    }


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    hashes: List[str]


def build_splits(config: dict, sequences: List[dict]) -> Tuple[Dict[str, SplitData], Dict[str, object]]:
    train_data, val_data, test_data = stratified_split(
        sequences,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
        seed=config["training"]["random_seed"],
    )
    hardening_cfg = config.get("training", {}).get("data_hardening", {})
    train_data, hardening_report = harden_training_sequences(
        train_data,
        seed=int(config["training"]["random_seed"]),
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

    x_train, y_train = prepare_xy(train_data, seq_len)
    x_val, y_val = prepare_xy(val_data, seq_len)
    x_test, y_test = prepare_xy(test_data, seq_len)

    splits = {
        "train": SplitData(x_train, y_train, _hash_rows(x_train)),
        "val": SplitData(x_val, y_val, _hash_rows(x_val)),
        "test": SplitData(x_test, y_test, _hash_rows(x_test)),
    }

    duplicates = {
        "train_val": len(set(splits["train"].hashes) & set(splits["val"].hashes)),
        "train_test": len(set(splits["train"].hashes) & set(splits["test"].hashes)),
        "val_test": len(set(splits["val"].hashes) & set(splits["test"].hashes)),
    }

    audit = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_train_original": int(hardening_report.n_original),
        "n_train_augmented": int(hardening_report.n_augmented),
        "class_ratio_train": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "class_ratio_val": float((y_val == 0).sum() / max((y_val == 1).sum(), 1)),
        "class_ratio_test": float((y_test == 0).sum() / max((y_test == 1).sum(), 1)),
        "duplicate_hashes": duplicates,
        "imbalance_gt_5_to_1": bool(((y_train == 0).sum() / max((y_train == 1).sum(), 1)) > 5.0),
        "hardening": {
            "label_flip_count": int(hardening_report.label_flip_count),
            "hard_sample_duplicates": int(hardening_report.hard_sample_duplicates),
            "gaussian_noise_sigma_frac": float(hardening_report.gaussian_noise_sigma_frac),
            "feature_dropout_rate": float(hardening_report.feature_dropout_rate),
        },
    }
    return splits, audit


def build_logistic_candidates(seed: int) -> List[Pipeline]:
    c_values = [0.25, 0.5, 1.0, 2.0]
    return [
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=1500,
                    ),
                ),
            ]
        )
        for c in c_values
    ]


def build_rf_candidates(seed: int) -> List[Pipeline]:
    params = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
    ]
    models: List[Pipeline] = []
    for p in params:
        models.append(
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=p["n_estimators"],
                            max_depth=p["max_depth"],
                            min_samples_leaf=p["min_samples_leaf"],
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
        )
    return models


def fit_calibrated_model(model, train: SplitData, val: SplitData):
    model.fit(train.x, train.y)
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    calibrated.fit(val.x, val.y)
    return calibrated


def select_best_model(
    candidates: List,
    train: SplitData,
    val: SplitData,
) -> Tuple[object, Dict[str, float]]:
    best_model = None
    best_selection = None
    best_score = -1.0

    for model in candidates:
        calibrated = fit_calibrated_model(model, train=train, val=val)
        p_adv_val = calibrated.predict_proba(val.x)[:, 1]
        threshold_info = tune_threshold(val.y, p_adv_val)
        score = float(threshold_info["f1"] + 0.05 * threshold_info["recall"])
        if score > best_score:
            best_score = score
            best_model = calibrated
            best_selection = threshold_info

    if best_model is None or best_selection is None:
        raise RuntimeError("Failed to select a calibrated baseline model.")
    return best_model, best_selection


def evaluate_model(
    model,
    splits: Dict[str, SplitData],
    selected_threshold: float,
    model_name: str,
    audit: Dict[str, object],
) -> pd.DataFrame:
    rows = []
    for split_name, split_data in splits.items():
        p_adv = model.predict_proba(split_data.x)[:, 1]
        metrics = compute_metrics(split_data.y, p_adv, threshold=selected_threshold)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                **metrics,
                "class_imbalance_gt_5_to_1": audit["imbalance_gt_5_to_1"],
                "duplicate_train_val": audit["duplicate_hashes"]["train_val"],
                "duplicate_train_test": audit["duplicate_hashes"]["train_test"],
                "duplicate_val_test": audit["duplicate_hashes"]["val_test"],
            }
        )
    return pd.DataFrame(rows)


def run_baselines(config_path: str, data_path: str, output_path: str) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    seed = int(config["training"]["random_seed"])
    set_seeds(seed)

    splits, audit = build_splits(config, sequences)
    print("=== Baseline Audit ===")
    print(
        f"  Class ratios (legit:adv) train/val/test = "
        f"{audit['class_ratio_train']:.2f} / {audit['class_ratio_val']:.2f} / {audit['class_ratio_test']:.2f}"
    )
    print(f"  Duplicate hashes across splits: {audit['duplicate_hashes']}")
    print(f"  Imbalance > 5:1: {audit['imbalance_gt_5_to_1']}")
    print(
        f"  Hardened train set: {audit['n_train_original']} -> {audit['n_train_augmented']} "
        f"(label_flips={audit['hardening']['label_flip_count']}, "
        f"hard_duplicates={audit['hardening']['hard_sample_duplicates']})"
    )

    logreg, logreg_selection = select_best_model(
        build_logistic_candidates(seed),
        train=splits["train"],
        val=splits["val"],
    )
    rf, rf_selection = select_best_model(
        build_rf_candidates(seed),
        train=splits["train"],
        val=splits["val"],
    )

    print(
        f"  Logistic selection: threshold={logreg_selection['threshold']:.3f} "
        f"prec={logreg_selection['precision']:.3f} rec={logreg_selection['recall']:.3f} "
        f"f1={logreg_selection['f1']:.3f} policy={logreg_selection['selection_policy']}"
    )
    print(
        f"  RF selection      : threshold={rf_selection['threshold']:.3f} "
        f"prec={rf_selection['precision']:.3f} rec={rf_selection['recall']:.3f} "
        f"f1={rf_selection['f1']:.3f} policy={rf_selection['selection_policy']}"
    )

    results = pd.concat(
        [
            evaluate_model(logreg, splits, logreg_selection["threshold"], "logistic_regression", audit),
            evaluate_model(rf, splits, rf_selection["threshold"], "random_forest", audit),
        ],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate leakage-safe baseline trust models")
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
