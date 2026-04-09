from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import pandas as pd

from run_pipeline import build_model, evaluate, load_config, load_data, train
from training.trainer import set_global_seed


def _flatten_metrics(run_name: str, seed: int, train_summary: Dict[str, Any], eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run": run_name,
        "seed": seed,
        "accuracy": float(eval_summary.get("accuracy", 0.0)),
        "precision": float(eval_summary.get("precision", 0.0)),
        "recall": float(eval_summary.get("recall", 0.0)),
        "f1": float(eval_summary.get("f1", 0.0)),
        "params": int(train_summary.get("params", 0)),
        "runtime_seconds": float(train_summary.get("runtime_seconds", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed IoUT benchmark experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Override model type, e.g. gru, lstm, fft, majority")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", default="results/multi_seed")
    parser.add_argument("--aggregate-output", default="results/aggregate_metrics.json")
    parser.add_argument("--table-output", default="results/summary_table.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.setdefault("model", {})["type"] = args.model

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    if len(seeds) != 5:
        print(f"Running {len(seeds)} seeds; the default benchmark target is 5 seeds.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        run_config = copy.deepcopy(config)
        run_config.setdefault("training", {})["seed"] = seed
        run_config["training"]["results_dir"] = str(output_dir / f"seed_{seed}")
        set_global_seed(seed)

        loaders, metadata = load_data(run_config)
        model = build_model(run_config, input_dim=int(metadata["input_dim"]))
        trained_model, train_summary = train(model, loaders, run_config, metadata)
        eval_summary = evaluate(trained_model, loaders, run_config, metadata)

        run_row = _flatten_metrics(str(run_config.get("model", {}).get("type", "unknown")), seed, train_summary, eval_summary)
        rows.append(run_row)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "run_result.json").open("w", encoding="utf-8") as handle:
            json.dump({"train": train_summary, "eval": eval_summary, "metadata": metadata}, handle, indent=2)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "multi_seed_results.csv", index=False)

    aggregate = {
        "seeds": seeds,
        "accuracy": {"mean": mean(frame["accuracy"]), "std": pstdev(frame["accuracy"]) if len(frame) > 1 else 0.0},
        "precision": {"mean": mean(frame["precision"]), "std": pstdev(frame["precision"]) if len(frame) > 1 else 0.0},
        "recall": {"mean": mean(frame["recall"]), "std": pstdev(frame["recall"]) if len(frame) > 1 else 0.0},
        "f1": {"mean": mean(frame["f1"]), "std": pstdev(frame["f1"]) if len(frame) > 1 else 0.0},
        "params": int(frame["params"].iloc[0]) if len(frame) else 0,
        "runtime_seconds": {"mean": mean(frame["runtime_seconds"]), "std": pstdev(frame["runtime_seconds"]) if len(frame) > 1 else 0.0},
    }

    Path(args.aggregate_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.aggregate_output).open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    table = pd.DataFrame(
        [
            {
                "Model": str(config.get("model", {}).get("type", "unknown")),
                "Accuracy": aggregate["accuracy"]["mean"],
                "F1": aggregate["f1"]["mean"],
                "Params": aggregate["params"],
                "Runtime": aggregate["runtime_seconds"]["mean"],
            }
        ]
    )
    Path(args.table_output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_output, index=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
