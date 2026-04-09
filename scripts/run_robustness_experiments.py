from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import IoUTDataset, build_dataloaders, drop_missing_segments, inject_gaussian_noise, load_records, simulate_sensor_failure
from evaluation.evaluate import evaluate_model
from models.baselines import build_baseline
from models.sequence_model import GRUClassifier, LSTMClassifier
from training.trainer import fit_model, set_global_seed

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _build_model(model_type: str, input_dim: int, model_cfg: Dict[str, Any], seed: int):
    normalized = model_type.lower().strip()
    if normalized == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if normalized == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    return build_baseline(normalized, seed=seed)


def _build_loader(samples, base_dataset: IoUTDataset, batch_size: int, seq_len: int) -> DataLoader:
    dataset = IoUTDataset(samples, seq_len=seq_len, normalizer=base_dataset.normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: {
        "sensor_id": [item["sensor_id"] for item in batch],
        "signal": torch.nn.utils.rnn.pad_sequence([item["signal"] for item in batch], batch_first=True),
        "lengths": torch.tensor([int(item["length"].item()) for item in batch], dtype=torch.long),
        "label": torch.tensor([float(item["label"].item()) for item in batch], dtype=torch.float32),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness experiments for the IoUT pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--missing-fraction", type=float, default=0.15)
    parser.add_argument("--failure-fraction", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.model:
        config.setdefault("model", {})["type"] = args.model
    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    data_cfg = config["data"]
    records = load_records(data_cfg["path"], file_format=data_cfg.get("format", "auto"))
    loaders, metadata, splits = build_dataloaders(
        records=records,
        seq_len=int(data_cfg.get("seq_len", 64)),
        train_split=float(data_cfg.get("train_split", 0.7)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
        seed=seed,
        strategy=str(data_cfg.get("split_strategy", "group")),
    )
    train_samples, val_samples, test_samples = splits

    model = _build_model(str(config.get("model", {}).get("type", "gru")), int(metadata["input_dim"]), config.get("model", {}), seed=seed)
    if hasattr(model, "fit") and not hasattr(model, "parameters"):
        model.fit(loaders["train"])
    else:
        model, _ = fit_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=config,
            results_dir=Path(args.output_dir),
            log_dir=Path(args.output_dir) / "logs",
            input_dim=int(metadata["input_dim"]),
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    perturbations = {
        "clean": test_samples,
        "gaussian_noise": inject_gaussian_noise(test_samples, sigma=args.sigma, seed=seed),
        "missing_segments": drop_missing_segments(test_samples, drop_fraction=args.missing_fraction, seed=seed),
        "sensor_failure": simulate_sensor_failure(test_samples, failure_fraction=args.failure_fraction, seed=seed),
    }

    rows: List[Dict[str, Any]] = []
    for name, samples in perturbations.items():
        test_loader = _build_loader(samples, loaders["train"].dataset, batch_size=int(data_cfg.get("batch_size", 32)), seq_len=int(data_cfg.get("seq_len", 64)))
        metrics = evaluate_model(model, test_loader, output_dir=output_root / name)
        rows.append({"scenario": name, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "robustness_summary.csv", index=False)
    with (output_root / "robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
