from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    accuracy = float((y_true_arr == y_pred_arr).mean())
    tp = float(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    fp = float(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = float(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
