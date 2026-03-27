"""
Run multi-seed simulation experiments and aggregate raw outputs.

This script executes simulation + evaluation for many seeds and writes one
combined long-format file:

    seed, run_id, interval, metric_name, value

Default behavior runs 20 seeds to satisfy robustness requirements.
"""

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python
from analysis.statistical_summary import summarize_raw_results, summarize_seed_level
from simulation.scripts.run_simulation import build_raw_results_long, run_single_simulation


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_seed_list(seed_list: Optional[str], start_seed: int, num_seeds: int) -> List[int]:
    if seed_list:
        seeds = []
        for token in seed_list.split(","):
            token = token.strip()
            if token:
                seeds.append(int(token))
        if not seeds:
            raise ValueError("--seed-list was provided but no valid seeds were parsed.")
        return seeds

    return [start_seed + i for i in range(num_seeds)]


def main() -> None:
    ensure_supported_python()

    parser = argparse.ArgumentParser(description="Run multi-seed IoUT simulation experiments")
    parser.add_argument("--config", default="simulation/configs/simulation_params.json")
    parser.add_argument("--output", default="simulation/outputs/multi_seed_raw_results.csv")
    parser.add_argument("--summary-output", default="analysis/stats/multi_seed_summary.csv")

    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--seed-list",
        default=None,
        help="Optional explicit comma-separated seed list. Overrides --start-seed/--num-seeds.",
    )

    parser.add_argument("--runs-per-seed", type=int, default=1)
    parser.add_argument("--intervals", type=int, default=20)

    parser.add_argument(
        "--use-transformer",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Use transformer trust scores inside simulation loop (default: True).",
    )
    parser.add_argument("--checkpoint", default="model/checkpoints/best_model.pt")
    parser.add_argument("--model-config", default="model/configs/transformer_config.json")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--no-quantized-transformer", action="store_true")
    parser.add_argument("--tau-min", type=float, default=None)
    parser.add_argument("--sequence-len", type=int, default=None)

    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=95.0)
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seed_list, args.start_seed, args.num_seeds)
    if len(seeds) < 20:
        raise ValueError(
            f"At least 20 seeds are required. Parsed {len(seeds)} seed(s). "
            "Use --num-seeds >= 20 or provide a longer --seed-list."
        )
    if args.runs_per_seed < 1:
        raise ValueError("--runs-per-seed must be >= 1")

    print("\n=== Multi-Seed Experiment Runner ===")
    print(f"  Seeds:          {len(seeds)}")
    print(f"  Runs per seed:  {args.runs_per_seed}")
    print(f"  Intervals:      {args.intervals}")
    print(f"  Config:         {args.config}")
    print(f"  Use transformer:{args.use_transformer}")
    print(f"  Output raw:     {args.output}")
    print(f"  Output summary: {args.summary_output}")

    all_raw_frames = []

    for base_seed in tqdm(seeds, desc="Seeds"):
        seed_results = []

        for run_idx in range(args.runs_per_seed):
            simulation_seed = base_seed + run_idx
            result = run_single_simulation(
                config_path=args.config,
                seed=simulation_seed,
                num_intervals=args.intervals,
                use_transformer=args.use_transformer,
                checkpoint_path=args.checkpoint,
                model_config_path=args.model_config,
                temperature=args.temperature,
                quantized=not args.no_quantized_transformer,
                tau_min_override=args.tau_min,
                sequence_len_override=args.sequence_len,
            )
            seed_results.append(result)

        # Keep seed column as the experiment seed; run_id distinguishes repeats.
        seed_raw = build_raw_results_long(seed_results, [base_seed] * args.runs_per_seed)
        all_raw_frames.append(seed_raw)

        seed_eval = summarize_raw_results(
            seed_raw,
            n_bootstrap=max(args.bootstrap_samples, 1000),
            ci_level=args.ci_level,
            seed=base_seed,
        )

        key_metrics = seed_eval[seed_eval["metric_name"].isin(["accuracy_proposed", "accuracy_bayesian", "accuracy_static"])]
        if not key_metrics.empty:
            compact = ", ".join(
                [f"{r.metric_name}={r.mean:.3f}" for r in key_metrics.itertuples(index=False)]
            )
            print(f"Seed {base_seed}: {compact}")

    combined = pd.concat(all_raw_frames, ignore_index=True)
    combined = combined[["seed", "run_id", "interval", "metric_name", "value"]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved multi-seed raw results: {args.output}")

    multi_seed_summary = summarize_seed_level(
        combined,
        n_bootstrap=max(args.bootstrap_samples, 1000),
        ci_level=args.ci_level,
        seed=args.start_seed,
    )
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    multi_seed_summary.to_csv(args.summary_output, index=False)
    print(f"Saved multi-seed summary: {args.summary_output}")


if __name__ == "__main__":
    main()
