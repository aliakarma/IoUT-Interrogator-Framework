from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

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

    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "threshold": float(threshold),
            "n_samples": int(len(y_true)),
            "mean_probability": float(np.mean(y_prob)) if len(y_prob) else 0.0,
        }
    )

    confusion = None
    if save_confusion_matrix:
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        confusion = {
            "tn": int(((y_true_arr == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true_arr == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true_arr == 1) & (y_pred == 0)).sum()),
            "tp": int(((y_true_arr == 1) & (y_pred == 1)).sum()),
        }
        metrics["confusion_matrix"] = confusion

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if confusion is not None:
        with (output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as handle:
            json.dump(confusion, handle, indent=2)

    return metrics
