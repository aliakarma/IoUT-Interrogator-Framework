from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from evaluation.metrics import compute_metrics


def _predict_with_model(model, loader, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict") and hasattr(model, "predict_proba") and not isinstance(model, torch.nn.Module):
        y_prob = np.asarray(model.predict_proba(loader), dtype=np.float32)
        y_pred = (y_prob >= threshold).astype(np.int64)
        return y_pred, y_prob

    device = next(model.parameters()).device if isinstance(model, torch.nn.Module) else torch.device("cpu")
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(signal, lengths)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    y_prob = np.asarray(probabilities, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return y_pred, y_prob


def tune_threshold(
    model,
    val_loader,
    metric_name: str = "f1",
    candidate_thresholds: list[float] | None = None,
) -> float:
    thresholds = candidate_thresholds or [float(value) for value in np.linspace(0.2, 0.8, 13)]
    best_threshold = 0.5
    best_score = -1.0

    y_true = []
    for batch in val_loader:
        y_true.extend(batch["label"].cpu().numpy().tolist())
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    if y_true_arr.size == 0:
        return best_threshold

    _, y_prob = _predict_with_model(model, val_loader, threshold=0.5)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        metrics = compute_metrics(y_true_arr, y_pred, y_prob=y_prob)
        score = float(metrics.get(metric_name, 0.0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(
    model,
    test_loader,
    output_dir: Path,
    threshold: float = 0.5,
    save_confusion_matrix: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = []
    for batch in test_loader:
        y_true.extend(batch["label"].cpu().numpy().tolist())
    y_pred, y_prob = _predict_with_model(model, test_loader, threshold=threshold)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    pr_curve_payload = {"precision": [], "recall": [], "thresholds": []}
    if len(y_true) and len(np.unique(np.asarray(y_true))) > 1:
        precision, recall, thresholds = precision_recall_curve(np.asarray(y_true, dtype=np.int64), y_prob)
        pr_curve_payload = {
            "precision": [float(value) for value in precision.tolist()],
            "recall": [float(value) for value in recall.tolist()],
            "thresholds": [float(value) for value in thresholds.tolist()],
        }
    metrics.update(
        {
            "threshold": float(threshold),
            "n_samples": int(len(y_true)),
            "mean_probability": float(np.mean(y_prob)) if len(y_prob) else 0.0,
            "pr_curve": pr_curve_payload,
        }
    )

    if metrics["accuracy"] >= metrics["recall"] + 0.25:
        warnings.warn(
            (
                "Metrics sanity warning: accuracy is much higher than recall. "
                "This may indicate class imbalance or threshold bias."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if save_confusion_matrix:
        with (output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "confusion_matrix": metrics.get("confusion_matrix", {}),
                    "confusion_matrix_normalized": metrics.get("confusion_matrix_normalized", {}),
                },
                handle,
                indent=2,
            )

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "y_true": [int(value) for value in np.asarray(y_true, dtype=np.int64).tolist()],
                "y_pred": [int(value) for value in y_pred.tolist()],
                "y_prob": [float(value) for value in y_prob.tolist()],
            },
            handle,
            indent=2,
        )

    with (output_dir / "pr_curve.json").open("w", encoding="utf-8") as handle:
        json.dump(pr_curve_payload, handle, indent=2)

    return metrics
