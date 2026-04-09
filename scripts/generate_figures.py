from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_model_and_seed(results_dir: Path) -> Tuple[str, int, Dict[str, Any]]:
    aggregate_path = results_dir / "aggregate_metrics.json"
    aggregate = _read_json(aggregate_path)
    model_map = aggregate.get("models", {})
    if not model_map:
        raise ValueError("No models found in aggregate_metrics.json")

    best_model = max(model_map.keys(), key=lambda name: float(model_map[name]["aggregate"]["f1"]["mean"]))
    seeds = [int(seed) for seed in aggregate.get("seeds", [])]
    if not seeds:
        raise ValueError("No seeds found in aggregate_metrics.json")

    best_seed = seeds[0]
    best_seed_f1 = -1.0
    for seed in seeds:
        run_result_path = results_dir / best_model / f"seed_{seed}" / "run_result.json"
        if not run_result_path.exists():
            continue
        run_result = _read_json(run_result_path)
        f1_value = float(run_result.get("eval", {}).get("f1", 0.0))
        if f1_value > best_seed_f1:
            best_seed_f1 = f1_value
            best_seed = seed

    return best_model, best_seed, aggregate


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready figures from real experiment outputs")
    parser.add_argument("--results-dir", default="results/final_20seed")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--figures-data-dir", default="results/figures_data")
    parser.add_argument("--learning-curves-dir", default="results/learning_curves")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robustness-path", default="results/robustness_analysis.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_data_dir = Path(args.figures_data_dir)
    learning_curves_dir = Path(args.learning_curves_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures_data_dir.mkdir(parents=True, exist_ok=True)

    best_model, best_seed, aggregate = _best_model_and_seed(results_dir)
    seeds = [int(seed) for seed in aggregate.get("seeds", [])]

    learning_curve_path = learning_curves_dir / f"seed_{args.seed}.json"
    if not learning_curve_path.exists():
        fallback_path = learning_curves_dir / f"seed_{best_seed}.json"
        learning_curve_path = fallback_path
    learning_curve = _read_json(learning_curve_path)

    train_loss = [float(value) for value in learning_curve.get("train_loss", [])]
    val_loss = [float(value) for value in learning_curve.get("val_loss", [])]
    epochs = list(range(1, min(len(train_loss), len(val_loss)) + 1))
    learning_curve_payload = {
        "seed": int(args.seed) if (learning_curves_dir / f"seed_{args.seed}.json").exists() else int(best_seed),
        "epochs": epochs,
        "train_loss": train_loss[: len(epochs)],
        "val_loss": val_loss[: len(epochs)],
    }
    _write_json(figures_data_dir / "learning_curve.json", learning_curve_payload)

    plt.figure(figsize=(6, 4))
    plt.plot(learning_curve_payload["epochs"], learning_curve_payload["train_loss"], label="Train Loss")
    plt.plot(learning_curve_payload["epochs"], learning_curve_payload["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "learning_curve.png", dpi=300)
    plt.close()

    f1_values: List[float] = []
    for seed in seeds:
        run_result_path = results_dir / best_model / f"seed_{seed}" / "run_result.json"
        if run_result_path.exists():
            run_result = _read_json(run_result_path)
            f1_values.append(float(run_result.get("eval", {}).get("f1", 0.0)))

    f1_payload = {
        "model": best_model,
        "seeds": seeds,
        "f1": f1_values,
    }
    _write_json(figures_data_dir / "f1_distribution.json", f1_payload)

    plt.figure(figsize=(6, 4))
    plt.hist(f1_values, bins=10)
    plt.xlabel("F1 Score")
    plt.ylabel("Frequency")
    plt.title("F1 Distribution Across Seeds")
    plt.tight_layout()
    plt.savefig(figures_dir / "f1_histogram.png", dpi=300)
    plt.close()

    best_run_path = results_dir / best_model / f"seed_{best_seed}" / "pr_curve.json"
    if not best_run_path.exists():
        best_run_path = results_dir / best_model / f"seed_{best_seed}" / "metrics.json"
    pr_data = _read_json(best_run_path)
    if "pr_curve" in pr_data:
        pr_data = pr_data["pr_curve"]

    pr_payload = {
        "model": best_model,
        "seed": int(best_seed),
        "precision": [float(value) for value in pr_data.get("precision", [])],
        "recall": [float(value) for value in pr_data.get("recall", [])],
        "thresholds": [float(value) for value in pr_data.get("thresholds", [])],
    }
    _write_json(figures_data_dir / "pr_curve.json", pr_payload)

    plt.figure(figsize=(6, 4))
    plt.plot(pr_payload["recall"], pr_payload["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curve.png", dpi=300)
    plt.close()

    robustness_payload = _read_json(Path(args.robustness_path))
    noise_levels = [float(value) for value in robustness_payload.get("noise_levels", [])]
    accuracy = [float(value) for value in robustness_payload.get("accuracy", [])]
    robustness_export = {
        "noise_levels": noise_levels,
        "accuracy": accuracy,
        "degradation_rate": float(robustness_payload.get("degradation_rate", 0.0)),
    }
    _write_json(figures_data_dir / "robustness_curve.json", robustness_export)

    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, accuracy, marker="o")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_curve.png", dpi=300)
    plt.close()

    print(
        json.dumps(
            {
                "best_model": best_model,
                "best_seed_for_pr_curve": best_seed,
                "figures_dir": str(figures_dir),
                "figures_data_dir": str(figures_data_dir),
                "generated_figures": [
                    "learning_curve.png",
                    "f1_histogram.png",
                    "pr_curve.png",
                    "robustness_curve.png",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
