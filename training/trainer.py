from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_eval(model: torch.nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)
            logits = model(signal, lengths)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return {"loss": total_loss / max(total_items, 1)}


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    pos_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    normalized = loss_type.lower().strip()
    if normalized == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
        pt = torch.exp(-bce)
        loss = ((1.0 - pt) ** focal_gamma) * bce
        return loss.mean()
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def fit_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    results_dir: Path,
    log_dir: Path,
    input_dim: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    logger = logging.getLogger("iout.training")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    training_cfg = config["training"]
    epochs = int(training_cfg.get("epochs", 8))
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 3))
    loss_type = str(training_cfg.get("loss_type", "bce"))
    focal_gamma = float(training_cfg.get("focal_gamma", 1.5))

    positive_count = 0
    negative_count = 0
    for batch in train_loader:
        labels = batch["label"].numpy()
        positive_count += int(labels.sum())
        negative_count += int(len(labels) - labels.sum())
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_path = results_dir / "checkpoints" / "best.pt"
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        item_count = 0
        for batch in train_loader:
            signal = batch["signal"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(signal, lengths)
            loss = _compute_loss(logits, labels, loss_type=loss_type, pos_weight=pos_weight, focal_gamma=focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += float(loss.item()) * batch_size
            item_count += batch_size

        train_loss = running_loss / max(item_count, 1)
        val_stats = _run_eval(model, val_loader, criterion, device)

        logger.info("Epoch %d/%d | train_loss=%.6f | val_loss=%.6f", epoch, epochs, train_loss, val_stats["loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_stats["loss"]})

        if val_stats["loss"] <= best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim}, best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d)",
                epoch,
                early_stopping_patience,
            )
            break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    summary = {
        "best_val_loss": best_val_loss,
        "epochs": len(history),
        "checkpoint_path": str(best_path),
        "history": history,
        "params": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "runtime_seconds": float(time.time() - start_time),
        "early_stopping_patience": early_stopping_patience,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
    }
    if history:
        final_gap = float(history[-1]["val_loss"] - history[-1]["train_loss"])
        summary["train_val_loss_gap"] = final_gap
        summary["overfitting_warning"] = bool(final_gap > 0.2)
        if summary["overfitting_warning"]:
            logger.warning(
                "Training sanity warning: potential overfitting detected (val_loss - train_loss = %.4f)",
                final_gap,
            )
    with (results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return model, summary
