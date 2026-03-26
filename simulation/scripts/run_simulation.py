"""
IoUT Simulation Runner
======================
Executes multiple simulation runs, aggregates results, and saves to CSV.

Usage:
    python simulation/scripts/run_simulation.py \
        --config simulation/configs/simulation_params.json \
        --runs 30 \
        --seed 42 \
        --output simulation/outputs/results.csv
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from simulation.scripts.environment import IoUTEnvironment
from model.inference.transformer_model import load_model, compute_trust_score


def run_single_simulation(config_path: str, seed: int,
                           num_intervals: int = 20,
                           use_transformer: bool = False,
                           checkpoint_path: str = "model/checkpoints/best_model.pt",
                           model_config_path: str = "model/configs/transformer_config.json",
                           temperature: float = 1.5,
                           quantized: bool = True) -> dict:
    """Run one simulation trial and return interval-level results."""
    env = IoUTEnvironment(config_path=config_path, seed=seed)

    transformer_trust_fn = None
    sequence_len = 64
    if use_transformer:
        model = load_model(
            checkpoint_path=checkpoint_path,
            config_path=model_config_path,
            quantized=quantized,
        )

        with open(model_config_path, "r") as cfg_f:
            model_cfg = json.load(cfg_f)
        sequence_len = int(model_cfg["architecture"]["seq_len"])

        def _trust_fn(sequence_window: np.ndarray) -> float:
            return compute_trust_score(
                model,
                sequence_window,
                temperature=temperature,
            )

        transformer_trust_fn = _trust_fn

    return env.run(
        num_intervals=num_intervals,
        transformer_trust_fn=transformer_trust_fn,
        sequence_len=sequence_len,
    )


def aggregate_runs(all_results: list) -> pd.DataFrame:
    """
    Aggregate results across multiple runs.
    Computes mean and standard deviation per metric per interval.
    """
    num_intervals = len(all_results[0]["interval"])
    metrics = [
        "accuracy_proposed", "accuracy_bayesian", "accuracy_static",
        "pdr_proposed", "pdr_bayesian", "pdr_static",
        "energy_proposed", "energy_bayesian", "energy_static",
    ]

    agg = {"interval": list(range(1, num_intervals + 1))}
    for metric in metrics:
        values_per_interval = np.array([r[metric] for r in all_results])
        agg[f"{metric}_mean"] = np.mean(values_per_interval, axis=0).tolist()
        agg[f"{metric}_std"] = np.std(values_per_interval, axis=0).tolist()

    return pd.DataFrame(agg)


def main():
    parser = argparse.ArgumentParser(
        description="Run IoUT interrogator framework simulation"
    )
    parser.add_argument(
        "--config",
        default="simulation/configs/simulation_params.json",
        help="Path to simulation config JSON"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent simulation runs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (each run uses seed + run_index)"
    )
    parser.add_argument(
        "--intervals", type=int, default=20,
        help="Number of monitoring intervals per run"
    )
    parser.add_argument(
        "--output",
        default="simulation/outputs/results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="Use trained transformer trust function instead of heuristic trust.",
    )
    parser.add_argument(
        "--checkpoint",
        default="model/checkpoints/best_model.pt",
        help="Transformer checkpoint path (used when --use-transformer is set).",
    )
    parser.add_argument(
        "--model-config",
        default="model/configs/transformer_config.json",
        help="Transformer config path (used when --use-transformer is set).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Temperature for trust score calibration when using transformer.",
    )
    parser.add_argument(
        "--no-quantized-transformer",
        action="store_true",
        help="Disable dynamic INT8 quantization in simulation inference path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"\n=== IoUT Interrogator Framework Simulation ===")
    print(f"  Config:    {args.config}")
    print(f"  Runs:      {args.runs}")
    print(f"  Intervals: {args.intervals}")
    print(f"  Base seed: {args.seed}")
    print(f"  Use transformer trust: {args.use_transformer}")
    print(f"  Output:    {args.output}")
    print()

    all_results = []
    for run_idx in tqdm(range(args.runs), desc="Simulation runs"):
        seed = args.seed + run_idx
        result = run_single_simulation(
            config_path=args.config,
            seed=seed,
            num_intervals=args.intervals,
            use_transformer=args.use_transformer,
            checkpoint_path=args.checkpoint,
            model_config_path=args.model_config,
            temperature=args.temperature,
            quantized=not args.no_quantized_transformer,
        )
        all_results.append(result)

    print("\nAggregating results...")
    agg_df = aggregate_runs(all_results)
    agg_df.to_csv(args.output, index=False)
    print(f"Saved aggregated results to: {args.output}")

    # Print summary table
    print("\n=== Summary (mean ± std across all intervals) ===")
    metrics_summary = [
        ("Accuracy — Proposed (%)", "accuracy_proposed"),
        ("Accuracy — Bayesian (%)", "accuracy_bayesian"),
        ("Accuracy — Static (%)",   "accuracy_static"),
        ("PDR — Proposed (%)",      "pdr_proposed"),
        ("PDR — Bayesian (%)",      "pdr_bayesian"),
        ("PDR — Static (%)",        "pdr_static"),
    ]
    for label, metric in metrics_summary:
        col_mean = agg_df[f"{metric}_mean"]
        col_std = agg_df[f"{metric}_std"]
        print(f"  {label:<35} {col_mean.mean():.1f} ± {col_std.mean():.1f}")

    return agg_df


if __name__ == "__main__":
    main()
