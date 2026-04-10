#!/usr/bin/env python
"""Generate PR (precision-recall) curves and save data for model evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load y_true and y_proba from results directory."""
    predictions_json = results_dir / "predictions.json"
    
    if not predictions_json.exists():
        raise FileNotFoundError(f"predictions.json not found in {results_dir}")
    
    with predictions_json.open("r") as f:
        data = json.load(f)
    
    y_true = np.array(data["y_true"], dtype=np.int64)
    # Handle both 'y_proba' and 'y_prob' keys for compatibility
    y_proba_key = "y_proba" if "y_proba" in data else "y_prob"
    y_proba = np.array(data[y_proba_key], dtype=np.float64)
    
    return y_true, y_proba


def compute_pr_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Compute precision-recall curve and AUC."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "pr_auc": float(pr_auc),
    }


def plot_pr_curve(
    precision: list[float],
    recall: list[float],
    pr_auc: float,
    output_path: Path,
    model_name: str = "Model",
) -> None:
    """Plot and save PR curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        recall, precision,
        marker='o', linestyle='-', linewidth=2, markersize=4,
        label=f"{model_name} (AUC = {pr_auc:.4f})"
    )
    ax.fill_between(recall, precision, alpha=0.2)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"PR curve saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate PR curves from model predictions")
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        required=True,
        help="Path to results directory containing predictions.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and data. Defaults to results_dir"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Model",
        help="Model name for plot legend"
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix for output filenames (e.g., '_real' produces pr_curve_real.json)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    y_true, y_proba = load_predictions(results_dir)
    logger.info(f"Loaded {len(y_true)} predictions from {results_dir}")
    
    # Compute PR curve
    pr_data = compute_pr_curve(y_true, y_proba)
    pr_auc = pr_data["pr_auc"]
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    
    # Save PR curve data
    data_output = output_dir / f"pr_curve{args.name_suffix}.json"
    with data_output.open("w") as f:
        json.dump(pr_data, f, indent=2)
    logger.info(f"PR curve data saved to {data_output}")
    
    # Plot and save figure
    fig_output = output_dir / f"pr_curve{args.name_suffix}.png"
    plot_pr_curve(
        precision=pr_data["precision"],
        recall=pr_data["recall"],
        pr_auc=pr_auc,
        output_path=fig_output,
        model_name=args.model_name,
    )
    
    print(f"\n✓ PR curve generation complete")
    print(f"  Data: {data_output}")
    print(f"  Plot: {fig_output}")
    print(f"  PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
