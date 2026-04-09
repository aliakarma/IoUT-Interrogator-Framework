from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from data.data_loader import build_dataloaders, load_records
from evaluation.evaluate import evaluate_model
from models.baselines import build_baseline, count_learned_parameters
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


def _run_variant(config: Dict[str, Any], variant: str, output_dir: Path) -> Dict[str, Any]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = copy.deepcopy(config.get("model", {}))
    baseline_cfg = copy.deepcopy(config.get("baseline", {}))

    records = load_records(data_cfg["path"], file_format=data_cfg.get("format", "auto"))
    normalize = bool(data_cfg.get("normalize", True))
    model_type = str(model_cfg.get("type", "gru")).lower()

    if variant == "no_preprocessing":
        normalize = False
    elif variant == "no_temporal":
        model_type = str(baseline_cfg.get("name", "fft"))
    elif variant == "replace_with_baseline":
        model_type = str(baseline_cfg.get("name", "majority"))
    else:
        raise ValueError(f"Unsupported ablation variant: {variant}")

    loaders, metadata, _ = build_dataloaders(
        records=records,
        seq_len=int(data_cfg.get("seq_len", 64)),
        train_split=float(data_cfg.get("train_split", 0.7)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=normalize,
        seed=int(training_cfg.get("seed", 42)),
        strategy=str(data_cfg.get("split_strategy", "group")),
    )

    model = _build_model(model_type, int(metadata["input_dim"]), model_cfg, seed=int(training_cfg.get("seed", 42)))
    if hasattr(model, "fit") and not hasattr(model, "parameters"):
        model.fit(loaders["train"])
        train_summary = {"mode": "baseline", "params": int(count_learned_parameters(model)), "runtime_seconds": 0.0}
    else:
        trained_model, train_summary = fit_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=config,
            results_dir=output_dir,
            log_dir=output_dir / "logs",
            input_dim=int(metadata["input_dim"]),
        )
        model = trained_model

    eval_summary = evaluate_model(model, loaders["test"], output_dir=output_dir)
    result = {
        "variant": variant,
        "model_type": model_type,
        "metadata": metadata,
        "train": train_summary,
        "eval": eval_summary,
        "params": int(count_learned_parameters(model)),
    }
    with (output_dir / "ablation_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IoUT ablation studies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", default="no_preprocessing,no_temporal,replace_with_baseline")
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    seed = int(config.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for variant in [token.strip() for token in args.variants.split(",") if token.strip()]:
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        result = _run_variant(config, variant, variant_dir)
        rows.append(
            {
                "variant": variant,
                "accuracy": float(result["eval"].get("accuracy", 0.0)),
                "f1": float(result["eval"].get("f1", 0.0)),
                "params": int(result["params"]),
                "runtime_seconds": float(result["train"].get("runtime_seconds", 0.0)),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "ablation_summary.csv", index=False)
    with (output_root / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
