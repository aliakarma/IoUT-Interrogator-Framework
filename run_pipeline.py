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
from evaluation.evaluate import evaluate_model
from models.baselines import build_baseline, count_learned_parameters
from models.sequence_model import GRUClassifier, LSTMClassifier
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
    records = load_records(data_cfg["path"], file_format=data_cfg.get("format", "auto"))
    loaders, metadata = build_dataloaders(
        records=records,
        seq_len=int(data_cfg.get("seq_len", 64)),
        train_split=float(data_cfg.get("train_split", 0.7)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
        seed=int(config["training"].get("seed", 42)),
        strategy=str(data_cfg.get("split_strategy", "group")),
    )
    return loaders, metadata


def build_model(config: Dict[str, Any], input_dim: int):
    model_cfg = config["model"]
    model_type = str(model_cfg.get("type", "gru")).lower()

    if model_type == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )

    if model_type == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
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
    return evaluate_model(
        model=model,
        test_loader=loaders["test"],
        output_dir=results_dir,
        threshold=float(evaluation_cfg.get("threshold", 0.5)),
        save_confusion_matrix=bool(evaluation_cfg.get("save_confusion_matrix", True)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoUT signal ML pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to pipeline config")
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
        "metadata": metadata,
        "train": train_summary,
        "eval": eval_summary,
        "params": int(count_learned_parameters(trained_model)),
    }
    results_dir = Path(config.get("training", {}).get("results_dir", "results/run_1"))
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2)

    logging.info("Training summary: %s", json.dumps(train_summary, indent=2))
    logging.info("Evaluation summary: %s", json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
