from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_class_distribution(y_true):
    return dict(Counter(int(value) for value in y_true))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float | Dict[str, float] | Dict[str, int] | Dict[str, Dict[str, float]]]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1) if y_prob is not None else None

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_prob_arr is not None and y_prob_arr.shape != y_true_arr.shape:
        raise ValueError("y_prob must have the same shape as y_true")

    accuracy = float((y_true_arr == y_pred_arr).mean())
    tp = float(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = float(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = float(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = float(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    precision_pos = _safe_divide(tp, tp + fp)
    recall_pos = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_divide(tn, tn + fn)
    recall_neg = _safe_divide(tn, tn + fp)
    balanced_accuracy = float((recall_pos + recall_neg) / 2.0)

    roc_auc = 0.0
    pr_auc = 0.0
    if y_prob_arr is not None and len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    class_distribution = compute_class_distribution(y_true_arr.tolist())

    confusion_matrix = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    row_neg = tn + fp
    row_pos = fn + tp
    confusion_matrix_normalized = {
        "tn": _safe_divide(tn, row_neg),
        "fp": _safe_divide(fp, row_neg),
        "fn": _safe_divide(fn, row_pos),
        "tp": _safe_divide(tp, row_pos),
    }

    per_class_precision_recall = {
        "class_0": {"precision": float(precision_neg), "recall": float(recall_neg)},
        "class_1": {"precision": float(precision_pos), "recall": float(recall_pos)},
    }

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision_pos),
        "recall": float(recall_pos),
        "recall_class_0": float(recall_neg),
        "recall_class_1": float(recall_pos),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "class_distribution": class_distribution,
        "per_class_precision_recall": per_class_precision_recall,
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_normalized": confusion_matrix_normalized,
    }
