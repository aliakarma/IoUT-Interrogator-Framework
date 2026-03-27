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
from compat import ensure_supported_python
from simulation.scripts.environment import IoUTEnvironment
from model.inference.transformer_model import load_model, compute_trust_score


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_single_simulation(config_path: str, seed: int,
                           num_intervals: int = 20,
                           use_transformer: bool = False,
                           checkpoint_path: str = "model/checkpoints/best_model.pt",
                           model_config_path: str = "model/configs/transformer_config.json",
                           temperature: float = 1.5,
                           quantized: bool = True,
                           log_trust_stats: bool = False,
                           tau_min_override: float = None,
                           sequence_len_override: int = None) -> dict:
    """Run one simulation trial and return interval-level results."""
    env = IoUTEnvironment(config_path=config_path, seed=seed)
    if tau_min_override is not None:
        env.mon_cfg["trust_threshold_tau_min"] = float(tau_min_override)

    transformer_trust_fn = None
    sequence_len = 64
    if use_transformer:
        model = None

        def _load(quant_flag: bool):
            return load_model(
                checkpoint_path=checkpoint_path,
                config_path=model_config_path,
                quantized=quant_flag,
            )

        try:
            model = _load(quantized)
        except Exception:
            if quantized:
                print("Warning: quantized transformer load failed; retrying without quantization.")
                model = _load(False)
            else:
                raise

        with open(model_config_path, "r") as cfg_f:
            model_cfg = json.load(cfg_f)
        model_seq_len = int(model_cfg["architecture"]["seq_len"])
        sequence_len = model_seq_len
        if sequence_len_override is not None:
            if int(sequence_len_override) > model_seq_len:
                raise ValueError(
                    f"sequence_len override {sequence_len_override} exceeds model seq_len {model_seq_len}."
                )
            sequence_len = int(sequence_len_override)

        # Validate model forward path once and fallback if quantized runtime is incompatible.
        try:
            _dummy = np.zeros((sequence_len, 5), dtype=np.float32)
            _ = compute_trust_score(model, _dummy, temperature=temperature)
        except Exception:
            if quantized:
                print("Warning: quantized transformer runtime incompatible in simulation; using non-quantized model.")
                model = _load(False)
            else:
                raise

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
        log_trust_stats=log_trust_stats,
    )


def aggregate_runs(all_results: list) -> pd.DataFrame:
    """
    Aggregate results across multiple runs.
    Computes mean and standard deviation per metric per interval.
    """
    num_intervals = len(all_results[0]["interval"])
    metrics = [k for k in all_results[0].keys() if k != "interval"]

    agg = {"interval": list(range(1, num_intervals + 1))}
    for metric in metrics:
        values_per_interval = np.array([r[metric] for r in all_results])
        agg[f"{metric}_mean"] = np.mean(values_per_interval, axis=0).tolist()
        agg[f"{metric}_std"] = np.std(values_per_interval, axis=0).tolist()

    return pd.DataFrame(agg)


def build_raw_results_long(all_results: list, run_seeds: list) -> pd.DataFrame:
    """
    Flatten run-level simulation outputs into long format.

    Output schema:
      run_id, seed, interval, metric_name, value

    One row per (run, interval, metric).
    """
    rows = []

    for run_id, (result, seed) in enumerate(zip(all_results, run_seeds), start=1):
        intervals = result.get("interval", [])
        metric_names = [k for k in result.keys() if k != "interval"]

        for idx, interval in enumerate(intervals):
            for metric_name in metric_names:
                value = result[metric_name][idx]
                rows.append(
                    {
                        "run_id": run_id,
                        "seed": int(seed),
                        "interval": int(interval),
                        "metric_name": metric_name,
                        "value": float(value),
                    }
                )

    return pd.DataFrame(rows, columns=["run_id", "seed", "interval", "metric_name", "value"])


def main():
    ensure_supported_python()
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
        "--raw-output",
        default="simulation/outputs/raw_results.csv",
        help="Output CSV path for long-format raw run-level metrics.",
    )
    parser.add_argument(
        "--use-transformer",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Use trained transformer trust function instead of heuristic trust (default: True).",
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
    parser.add_argument(
        "--log-trust-stats",
        action="store_true",
        help="Print per-interval proposed trust distribution stats (mean/std/min/max).",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=None,
        help="Optional override for trust threshold tau_min.",
    )
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=None,
        help="Optional override for transformer sequence length K.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)

    if os.path.abspath(args.output) == os.path.abspath(args.raw_output):
        raise ValueError("--raw-output must be different from --output to avoid overwriting aggregated results.")

    print(f"\n=== IoUT Interrogator Framework Simulation ===")
    print(f"  Config:    {args.config}")
    print(f"  Runs:      {args.runs}")
    print(f"  Intervals: {args.intervals}")
    print(f"  Base seed: {args.seed}")
    print(f"  Use transformer trust: {args.use_transformer}")
    print(f"  Output:    {args.output}")
    print()

    all_results = []
    run_seeds = []
    for run_idx in tqdm(range(args.runs), desc="Simulation runs"):
        seed = args.seed + run_idx
        run_seeds.append(seed)
        result = run_single_simulation(
            config_path=args.config,
            seed=seed,
            num_intervals=args.intervals,
            use_transformer=args.use_transformer,
            checkpoint_path=args.checkpoint,
            model_config_path=args.model_config,
            temperature=args.temperature,
            quantized=not args.no_quantized_transformer,
            log_trust_stats=args.log_trust_stats,
            tau_min_override=args.tau_min,
            sequence_len_override=args.sequence_len,
        )
        all_results.append(result)

    print("\nAggregating results...")
    agg_df = aggregate_runs(all_results)
    agg_df.to_csv(args.output, index=False)
    print(f"Saved aggregated results to: {args.output}")

    raw_df = build_raw_results_long(all_results, run_seeds)
    raw_df.to_csv(args.raw_output, index=False)
    print(f"Saved raw run-level results to: {args.raw_output}")

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
