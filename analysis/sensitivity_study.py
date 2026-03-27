"""
Sensitivity Study Runner
========================
Runs threshold sensitivity and sequence-length ablation experiments using the
same simulation stack as the main pipeline.

Usage:
    python analysis/sensitivity_study.py \
        --runs 20 \
        --intervals 20 \
        --seed 42
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python
from simulation.scripts.run_simulation import run_single_simulation


def _parse_float_list(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _parse_int_list(values: str) -> List[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _summarize_trial_results(all_results: List[dict]) -> dict:
    metrics = [k for k in all_results[0].keys() if k != "interval"]
    summary = {}
    for metric in metrics:
        per_run_curve = np.array([run[metric] for run in all_results], dtype=np.float64)
        per_run_mean = per_run_curve.mean(axis=1)
        summary[f"{metric}_mean"] = float(per_run_mean.mean())
        summary[f"{metric}_std"] = float(per_run_mean.std())
    return summary


def run_threshold_sensitivity(args) -> pd.DataFrame:
    rows = []
    for tau in args.tau_values:
        all_results = []
        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            result = run_single_simulation(
                config_path=args.config,
                seed=run_seed,
                num_intervals=args.intervals,
                use_transformer=args.use_transformer,
                checkpoint_path=args.checkpoint,
                model_config_path=args.model_config,
                temperature=args.temperature,
                quantized=not args.no_quantized_transformer,
                tau_min_override=tau,
            )
            all_results.append(result)
        row = {"tau_min": tau}
        row.update(_summarize_trial_results(all_results))
        rows.append(row)
    return pd.DataFrame(rows)


def run_sequence_ablation(args) -> pd.DataFrame:
    rows = []
    for seq_len in args.sequence_lengths:
        all_results = []
        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            result = run_single_simulation(
                config_path=args.config,
                seed=run_seed,
                num_intervals=args.intervals,
                use_transformer=args.use_transformer,
                checkpoint_path=args.checkpoint,
                model_config_path=args.model_config,
                temperature=args.temperature,
                quantized=not args.no_quantized_transformer,
                sequence_len_override=seq_len,
            )
            all_results.append(result)
        row = {"sequence_len": seq_len}
        row.update(_summarize_trial_results(all_results))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ensure_supported_python()
    parser = argparse.ArgumentParser(
        description="Run threshold and sequence-length sensitivity studies",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="simulation/configs/simulation_params.json")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-transformer", type=_parse_bool, default=True)
    parser.add_argument("--checkpoint", default="model/checkpoints/best_model.pt")
    parser.add_argument("--model-config", default="model/configs/transformer_config.json")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--no-quantized-transformer", action="store_true")
    parser.add_argument("--tau-values", type=_parse_float_list, default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--sequence-lengths", type=_parse_int_list, default="16,32,64")
    parser.add_argument("--threshold-output", default="analysis/stats/threshold_sensitivity.csv")
    parser.add_argument("--sequence-output", default="analysis/stats/sequence_length_ablation.csv")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            f"Unrecognized arguments: {' '.join(unknown)}. "
            "Run with --help to see supported flags."
        )

    os.makedirs(os.path.dirname(args.threshold_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.sequence_output), exist_ok=True)

    threshold_df = run_threshold_sensitivity(args)
    threshold_df.to_csv(args.threshold_output, index=False)

    sequence_df = run_sequence_ablation(args)
    sequence_df.to_csv(args.sequence_output, index=False)

    print("Saved threshold sensitivity:", args.threshold_output)
    print("Saved sequence-length ablation:", args.sequence_output)


if __name__ == "__main__":
    main()
