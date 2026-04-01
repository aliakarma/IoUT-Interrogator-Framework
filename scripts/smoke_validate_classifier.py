"""
Fast 3-seed smoke validation for the binary classification pipeline.

Outputs:
- analysis/smoke/smoke_metrics.csv
- analysis/smoke/before_after_comparison.csv
- analysis/smoke/confusion_matrices.csv
- analysis/smoke/diagnostic_report.md
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.inference.baseline_models import run_baselines
from model.inference.lstm_baseline import run_lstm_baseline
from model.inference.train import train_model
from simulation.scripts.generate_behavioral_data import generate_dataset


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_sequences(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_sequences(path: Path, sequences: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sequences, f)


def _temp_config(base_config_path: Path, seed: int, epochs: int) -> Path:
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["training"]["random_seed"] = int(seed)
    config["training"]["epochs"] = int(epochs)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"iout_smoke_seed_{seed}_"))
    out_path = temp_dir / "config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_path


def _snapshot_before() -> Dict[str, float]:
    eval_path = REPO_ROOT / "model" / "checkpoints" / "evaluation_metrics.json"
    if not eval_path.exists():
        return {}
    payload = _load_json(eval_path)
    test = payload.get("test", {})
    diagnostics = payload.get("test_diagnostics", {})
    return {
        "accuracy": float(test.get("accuracy", np.nan)),
        "precision": float(test.get("precision", np.nan)),
        "recall": float(test.get("recall", np.nan)),
        "f1": float(test.get("f1", np.nan)),
        "roc_auc": float(test.get("roc_auc", np.nan)),
        "pr_auc": float(test.get("pr_auc", np.nan)),
        "ece": float(test.get("ece", np.nan)),
        "brier": float(test.get("brier", np.nan)),
        "tp": int(diagnostics.get("tp", 0)),
        "tn": int(diagnostics.get("tn", 0)),
        "fp": int(diagnostics.get("fp", 0)),
        "fn": int(diagnostics.get("fn", 0)),
    }


def _base_value_matrix(sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not sequences:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)
    rows = []
    labels = []
    for item in sequences:
        seq = np.asarray(item["sequence"], dtype=np.float32)
        if seq.shape[1] == 30:
            seq = seq[:, [0, 6, 12, 18, 24]]
        rows.append(seq.mean(axis=0))
        labels.append(int(item["label"]))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _dataset_stats(sequences: List[Dict]) -> Dict[str, float]:
    x, y = _base_value_matrix(sequences)
    legit = x[y == 0]
    adv = x[y == 1]
    if len(legit) == 0 or len(adv) == 0:
        return {}
    legit_mean = legit.mean(axis=0)
    adv_mean = adv.mean(axis=0)
    legit_std = legit.std(axis=0)
    adv_std = adv.std(axis=0)
    pooled = np.maximum((legit_std + adv_std) / 2.0, 1e-6)
    separation = np.abs(legit_mean - adv_mean) / pooled
    return {
        "n_samples": int(len(sequences)),
        "n_legit": int((y == 0).sum()),
        "n_adv": int((y == 1).sum()),
        "adv_fraction": float((y == 1).mean()),
        "avg_base_feature_separation": float(separation.mean()),
        "max_base_feature_separation": float(separation.max()),
        "min_base_feature_separation": float(separation.min()),
    }


def _regenerate_harder_dataset(source_data_path: Path, config_path: Path, output_dir: Path) -> Tuple[Path, Dict[str, float], Dict[str, float], Dict[str, object]]:
    source_sequences = _read_sequences(source_data_path)
    before_stats = _dataset_stats(source_sequences)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    num_sequences = len(source_sequences)
    adv_fraction = before_stats.get("adv_fraction", 0.15)
    seq_len = int(config["architecture"]["seq_len"])
    dataset_seed = 314
    attack_mix = [
        ("low_and_slow", 0.40),
        ("coordinated_insider", 0.20),
        ("route_manipulation", 0.15),
        ("selective_packet_drop", 0.15),
        ("transmission_burst", 0.10),
    ]
    sequences, _ = generate_dataset(
        num_sequences=num_sequences,
        K=seq_len,
        adv_fraction=adv_fraction,
        seed=dataset_seed,
        noise_sigma=0.12,
        attack_mix=attack_mix,
        noise_distribution="mixture",
        burst_loss_prob=0.035,
        latency_spike_scale=0.06,
    )
    hardened_data_path = output_dir / "hardened_behavioral_sequences.json"
    _write_sequences(hardened_data_path, sequences)
    after_stats = _dataset_stats(sequences)
    generation_cfg = {
        "dataset_seed": dataset_seed,
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "adv_fraction": adv_fraction,
        "noise_sigma": 0.12,
        "noise_distribution": "mixture",
        "burst_loss_prob": 0.035,
        "latency_spike_scale": 0.06,
        "attack_mix": attack_mix,
    }
    return hardened_data_path, before_stats, after_stats, generation_cfg


def _aggregate_rows(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in {"model", "seed"}]
    summary_rows = []
    for model, sub in df.groupby("model"):
        row = {"model": model}
        for col in metric_cols:
            values = sub[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                low = float(np.percentile(values, 2.5))
                high = float(np.percentile(values, 97.5))
            else:
                low = high = float(values[0])
            row[f"{col}_ci95_low"] = low
            row[f"{col}_ci95_high"] = high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_smoke(config_path: Path, data_path: Path, seeds: List[int], epochs: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hardened_data_path, before_dataset_stats, after_dataset_stats, generation_cfg = _regenerate_harder_dataset(
        data_path,
        config_path,
        output_dir,
    )
    before = _snapshot_before()
    per_seed_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    audit_payload = {
        "seeds": seeds,
        "epochs": epochs,
        "dataset_generation": generation_cfg,
        "dataset_stats_before": before_dataset_stats,
        "dataset_stats_after": after_dataset_stats,
        "notes": [
            "Splits are stratified and created before preprocessing.",
            "Sklearn baselines use Pipeline(imputer, scaler, model) fit on train only.",
            "Calibration is selected on validation only.",
            "Threshold is tuned on validation only with recall-aware objective.",
            "Training split only is further hardened via noise, overlap injection, light label noise, hard-sample duplication, and feature dropout.",
            "Validation and test data are untouched after split except for train-fit normalization/scaling.",
        ],
    }

    for seed in seeds:
        temp_config = _temp_config(config_path, seed=seed, epochs=epochs)
        ckpt_dir = output_dir / f"seed_{seed}" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_model(
            config_path=str(temp_config),
            data_path=str(hardened_data_path),
            checkpoint_dir=str(ckpt_dir),
            verbose=False,
            run_sanity=False,
        )
        transformer_metrics = _load_json(ckpt_dir / "evaluation_metrics.json")
        t_test = transformer_metrics["test"]
        t_diag = transformer_metrics["test_diagnostics"]
        per_seed_rows.append(
            {
                "seed": seed,
                "model": "transformer",
                "accuracy": float(t_test["accuracy"]),
                "precision": float(t_test["precision"]),
                "recall": float(t_test["recall"]),
                "f1": float(t_test["f1"]),
                "roc_auc": float(t_test["roc_auc"]),
                "pr_auc": float(t_test["pr_auc"]),
                "ece": float(t_test["ece"]),
                "brier": float(t_test["brier"]),
                "fp": int(t_diag["fp"]),
                "tp": int(t_diag["tp"]),
                "tn": int(t_diag["tn"]),
                "fn": int(t_diag["fn"]),
            }
        )
        confusion_rows.append({"phase": "after", "model": "transformer", "seed": seed, **{k: int(t_diag[k]) for k in ["tp", "tn", "fp", "fn"]}})

        baseline_output = output_dir / f"seed_{seed}" / "baseline_results.csv"
        baseline_df = run_baselines(str(temp_config), str(hardened_data_path), str(baseline_output))
        for model_name in ["logistic_regression", "random_forest"]:
            row = baseline_df[(baseline_df["model"] == model_name) & (baseline_df["split"] == "test")].iloc[0]
            per_seed_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                    "ece": float(row["ece"]),
                    "brier": float(row["brier"]),
                    "fp": int(row["fp"]),
                    "tp": int(row["tp"]),
                    "tn": int(row["tn"]),
                    "fn": int(row["fn"]),
                }
            )

        lstm_output = output_dir / f"seed_{seed}" / "lstm_results.csv"
        lstm_df = run_lstm_baseline(
            config_path=str(temp_config),
            data_path=str(hardened_data_path),
            output_path=str(lstm_output),
            checkpoint_dir=str(ckpt_dir),
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
        )
        lstm_row = lstm_df[(lstm_df["model"] == "lstm_baseline") & (lstm_df["split"] == "test")].iloc[0]
        per_seed_rows.append(
            {
                "seed": seed,
                "model": "lstm_baseline",
                "accuracy": float(lstm_row["accuracy"]),
                "precision": float(lstm_row["precision"]),
                "recall": float(lstm_row["recall"]),
                "f1": float(lstm_row["f1"]),
                "roc_auc": float(lstm_row["roc_auc"]),
                "pr_auc": float(lstm_row["pr_auc"]),
                "ece": float(lstm_row["ece"]),
                "brier": float(lstm_row["brier"]),
                "fp": int(lstm_row["fp"]),
                "tp": int(lstm_row["tp"]),
                "tn": int(lstm_row["tn"]),
                "fn": int(lstm_row["fn"]),
            }
        )

    smoke_df = pd.DataFrame(per_seed_rows)
    smoke_df.to_csv(output_dir / "smoke_metrics.csv", index=False)
    summary_df = _aggregate_rows(per_seed_rows)
    summary_df.to_csv(output_dir / "smoke_summary.csv", index=False)
    pd.DataFrame(
        [
            {"phase": "before", **before_dataset_stats},
            {"phase": "after", **after_dataset_stats},
        ]
    ).to_csv(output_dir / "dataset_stats_comparison.csv", index=False)

    if before:
        confusion_rows.append(
            {
                "phase": "before",
                "model": "transformer",
                "seed": -1,
                "tp": before.get("tp", 0),
                "tn": before.get("tn", 0),
                "fp": before.get("fp", 0),
                "fn": before.get("fn", 0),
            }
        )
        after_transformer = summary_df[summary_df["model"] == "transformer"].iloc[0]
        comparison = pd.DataFrame(
            [
                {
                    "metric": "accuracy",
                    "before": before.get("accuracy", np.nan),
                    "after_mean": after_transformer["accuracy_mean"],
                },
                {
                    "metric": "precision",
                    "before": before.get("precision", np.nan),
                    "after_mean": after_transformer["precision_mean"],
                },
                {
                    "metric": "recall",
                    "before": before.get("recall", np.nan),
                    "after_mean": after_transformer["recall_mean"],
                },
                {
                    "metric": "f1",
                    "before": before.get("f1", np.nan),
                    "after_mean": after_transformer["f1_mean"],
                },
                {
                    "metric": "ece",
                    "before": before.get("ece", np.nan),
                    "after_mean": after_transformer["ece_mean"],
                },
                {
                    "metric": "brier",
                    "before": before.get("brier", np.nan),
                    "after_mean": after_transformer["brier_mean"],
                },
                {
                    "metric": "fp",
                    "before": before.get("fp", np.nan),
                    "after_mean": after_transformer["fp_mean"],
                },
            ]
        )
        comparison.to_csv(output_dir / "before_after_comparison.csv", index=False)

    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    _write_json(output_dir / "audit.json", audit_payload)

    transformer_summary = summary_df[summary_df["model"] == "transformer"].iloc[0]
    logistic_summary = summary_df[summary_df["model"] == "logistic_regression"].iloc[0]
    with open(output_dir / "diagnostic_report.md", "w", encoding="utf-8") as f:
        f.write("# Diagnostic Report\n\n")
        f.write("## Audit\n")
        f.write("- Train/validation/test split occurs before model fitting and preprocessing in the corrected baselines.\n")
        f.write("- Sklearn preprocessing is encapsulated in train-only pipelines.\n")
        f.write("- Threshold and calibration are selected on validation only.\n")
        f.write(
            f"- Dataset separability changed from {before_dataset_stats['avg_base_feature_separation']:.3f}x "
            f"to {after_dataset_stats['avg_base_feature_separation']:.3f}x average standardized separation "
            f"across base features.\n"
        )
        f.write(
            f"- Class distribution remains {after_dataset_stats['n_legit']} legitimate / "
            f"{after_dataset_stats['n_adv']} adversarial "
            f"(adv_fraction={after_dataset_stats['adv_fraction']:.3f}).\n"
        )
        f.write("- Harder generation uses heavier-tailed noise, mild burst loss, latency spikes, and a low-and-slow-heavy attack mix.\n")
        f.write("- Training data only is additionally hardened with Gaussian feature noise, overlap injection, 3% label noise, hard-sample duplication, and feature dropout.\n")
        f.write("\n## Smoke Outcome\n")
        f.write(
            f"- Transformer recall mean: {transformer_summary['recall_mean']:.4f}, "
            f"FP mean: {transformer_summary['fp_mean']:.4f}, "
            f"ECE mean: {transformer_summary['ece_mean']:.4f}\n"
        )
        f.write(
            f"- Logistic regression recall mean: {logistic_summary['recall_mean']:.4f}, "
            f"ECE mean: {logistic_summary['ece_mean']:.4f}\n"
        )
        f.write("- If logistic regression remains competitive, the likely explanation is that the enriched temporal features still preserve strong linear structure rather than leakage.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-seed smoke validation for the classification pipeline")
    parser.add_argument("--config", default="model/configs/transformer_config.json")
    parser.add_argument("--data", default="data/raw/behavioral_sequences.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default="analysis/smoke")
    args = parser.parse_args()

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    run_smoke(
        config_path=REPO_ROOT / args.config,
        data_path=REPO_ROOT / args.data,
        seeds=seeds,
        epochs=args.epochs,
        output_dir=REPO_ROOT / args.output_dir,
    )
    print(f"Smoke validation artifacts written to: {REPO_ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
