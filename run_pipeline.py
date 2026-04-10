from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from data.data_loader import IoUTDataset, build_dataloaders, load_records
from evaluation.evaluate import evaluate_model, tune_threshold
from models.baselines import build_baseline, count_learned_parameters
from models.hybrid_temporal import HybridTemporalClassifier
from models.temporal_cnn import TemporalCNNClassifier
from models.transformer_light import LightweightTransformerClassifier
from models.sequence_model import GRUClassifier, LSTMClassifier
from data.unsw_loader import load_unsw_nb15_records
from training.trainer import fit_model, set_global_seed


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def load_data(config: Dict[str, Any]) -> Tuple[Dict[str, IoUTDataset], Dict[str, Any]]:
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "synthetic")).lower()
    seq_len = int(data_cfg.get("seq_len", 64))
    splits_path = str(data_cfg.get("splits_path", "splits/split_v1.json"))

    if dataset_name == "unsw_nb15":
        default_synthetic_path = "data/raw/behavioral_sequences.json"
        unsw_default_path = str(data_cfg.get("unsw_path", "data/raw/unsw_nb15"))
        source_path = str(data_cfg.get("path", default_synthetic_path))
        if source_path == default_synthetic_path:
            source_path = unsw_default_path
        if splits_path == "splits/split_v1.json":
            splits_path = "splits/unsw_nb15_split_v1.json"
        seq_len = 20
        records = load_unsw_nb15_records(
            source=source_path,
            seq_len=seq_len,
            max_rows=data_cfg.get("max_rows"),
        )
    else:
        records = load_records(data_cfg["path"], file_format=data_cfg.get("format", "auto"))

    loaders, metadata, _ = build_dataloaders(
        records=records,
        seq_len=seq_len,
        train_split=float(data_cfg.get("train_split", 0.7)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
        seed=int(config["training"].get("seed", 42)),
        strategy=str(data_cfg.get("split_strategy", "group")),
        use_fixed_splits=bool(data_cfg.get("use_fixed_splits", True)),
        splits_path=splits_path,
    )

    train_ids = set(metadata.get("train_sensor_ids", []))
    val_ids = set(metadata.get("val_sensor_ids", []))
    test_ids = set(metadata.get("test_sensor_ids", []))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Leakage check failed: overlapping sensor_id values across train/val/test splits.")

    stats = metadata.get("dataset_statistics", {})
    if stats:
        logging.info("Dataset statistics: %s", json.dumps(stats, indent=2))

    return loaders, metadata


def build_model(config: Dict[str, Any], input_dim: int):
    model_cfg = config["model"]
    model_type = str(model_cfg.get("type", "gru")).lower()

    if model_type == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_type == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_type in {"temporal_cnn", "tcnn", "cnn"}:
        return TemporalCNNClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            channels=int(model_cfg.get("channels", 96)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_type in {"hybrid_temporal", "hybrid_cnn", "hybrid"}:
        return HybridTemporalClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            channels=int(model_cfg.get("channels", 96)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_type in {"transformer_light", "light_transformer", "transformer"}:
        return LightweightTransformerClassifier(
            input_dim=input_dim,
            d_model=int(model_cfg.get("d_model", 96)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 192)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_type == "baseline":
        model_type = str(config.get("baseline", {}).get("name", "logistic_regression")).lower()

    return build_baseline(model_type=model_type, seed=int(config["training"].get("seed", 42)))


def train(model, loaders: Dict[str, Any], config: Dict[str, Any], metadata: Dict[str, Any]):
    training_cfg = config["training"]
    results_dir = Path(training_cfg.get("results_dir", "results/run_1"))
    log_dir = Path(training_cfg.get("log_dir", "logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "fit") and not isinstance(model, torch.nn.Module):
        start_time = time.time()
        model.fit(loaders["train"])
        summary = {
            "mode": "baseline",
            "results_dir": str(results_dir),
            "params": int(count_learned_parameters(model)),
            "runtime_seconds": float(time.time() - start_time),
        }
        with (results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return model, summary

    return fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        results_dir=results_dir,
        log_dir=log_dir,
        input_dim=int(metadata["input_dim"]),
    )


def evaluate(model, loaders: Dict[str, Any], config: Dict[str, Any], metadata: Dict[str, Any]):
    evaluation_cfg = config.get("evaluation", {})
    results_dir = Path(config["training"].get("results_dir", "results/run_1"))
    threshold = float(evaluation_cfg.get("threshold", 0.5))
    if bool(evaluation_cfg.get("tune_threshold", False)):
        threshold = tune_threshold(
            model=model,
            val_loader=loaders["val"],
            metric_name=str(evaluation_cfg.get("threshold_metric", "f1")),
        )
    return evaluate_model(
        model=model,
        test_loader=loaders["test"],
        output_dir=results_dir,
        threshold=threshold,
        save_confusion_matrix=bool(evaluation_cfg.get("save_confusion_matrix", True)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoUT signal ML pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to pipeline config")
    parser.add_argument("--dataset", default=None, choices=["synthetic", "unsw_nb15"], help="Dataset source to run")
    parser.add_argument("--data-path", default=None, help="Optional dataset file path override")
    parser.add_argument("--split-strategy", default=None, choices=["group", "time"], help="Override dataset split strategy")
    return parser


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataset:
        config.setdefault("data", {})["dataset"] = args.dataset
    if args.data_path:
        config.setdefault("data", {})["path"] = args.data_path
    if args.split_strategy:
        config.setdefault("data", {})["split_strategy"] = args.split_strategy

    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = Path(config.get("training", {}).get("log_dir", "logs"))
    _setup_logging(log_dir)

    loaders, metadata = load_data(config)
    model = build_model(config, input_dim=int(metadata["input_dim"]))
    trained_model, train_summary = train(model, loaders, config, metadata)
    eval_summary = evaluate(trained_model, loaders, config, metadata)

    combined_summary = {
        "model": str(config.get("model", {}).get("type", "unknown")),
        "metadata": metadata,
        "train": train_summary,
        "eval": eval_summary,
        "params": int(count_learned_parameters(trained_model)),
    }
    results_dir = Path(config.get("training", {}).get("results_dir", "results/run_1"))
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2)

    metrics_mirror_path = Path("results") / "metrics.json"
    metrics_mirror_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_mirror_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_summary, handle, indent=2)

    logging.info("Training summary: %s", json.dumps(train_summary, indent=2))
    logging.info("Evaluation summary: %s", json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
