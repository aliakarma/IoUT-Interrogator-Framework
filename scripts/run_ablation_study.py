"""
Run component ablation experiments and aggregate results.

Configurations:
  - full_model
  - energy_disabled
  - routing_disabled
  - blockchain_disabled

Output schema:
  configuration, metric_name, value
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python
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


def _run_configuration(
    config_name: str,
    sim_config_path: str,
    runs: int,
    seed_base: int,
    intervals: int,
    use_transformer: bool,
    checkpoint: str,
    model_config: str,
    temperature: float,
    no_quantized_transformer: bool,
    disable_energy_model: bool,
    disable_routing_model: bool,
    disable_blockchain: bool,
) -> pd.DataFrame:
    all_results = []
    run_seeds = []

    for run_idx in tqdm(range(runs), desc=f"{config_name} runs", leave=False):
        seed = seed_base + run_idx
        run_seeds.append(seed)
        result = run_single_simulation(
            config_path=sim_config_path,
            seed=seed,
            num_intervals=intervals,
            use_transformer=use_transformer,
            checkpoint_path=checkpoint,
            model_config_path=model_config,
            temperature=temperature,
            quantized=not no_quantized_transformer,
            disable_energy_model=disable_energy_model,
            disable_routing_model=disable_routing_model,
            disable_blockchain=disable_blockchain,
        )
        all_results.append(result)

    raw_df = build_raw_results_long(all_results, run_seeds)

    summary = (
        raw_df.groupby("metric_name", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "value"})
    )
    summary.insert(0, "configuration", config_name)
    return summary[["configuration", "metric_name", "value"]]


def main() -> None:
    ensure_supported_python()

    parser = argparse.ArgumentParser(description="Run simulation component ablation study")
    parser.add_argument("--config", default="simulation/configs/simulation_params.json")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--output", default="analysis/stats/ablation_results.csv")

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

    args = parser.parse_args()

    configurations: List[Dict[str, object]] = [
        {
            "name": "full_model",
            "disable_energy_model": False,
            "disable_routing_model": False,
            "disable_blockchain": False,
        },
        {
            "name": "energy_disabled",
            "disable_energy_model": True,
            "disable_routing_model": False,
            "disable_blockchain": False,
        },
        {
            "name": "routing_disabled",
            "disable_energy_model": False,
            "disable_routing_model": True,
            "disable_blockchain": False,
        },
        {
            "name": "blockchain_disabled",
            "disable_energy_model": False,
            "disable_routing_model": False,
            "disable_blockchain": True,
        },
    ]

    all_rows = []
    for cfg in configurations:
        print(
            f"Running {cfg['name']} "
            f"(energy_off={cfg['disable_energy_model']}, "
            f"routing_off={cfg['disable_routing_model']}, "
            f"blockchain_off={cfg['disable_blockchain']})"
        )
        cfg_df = _run_configuration(
            config_name=str(cfg["name"]),
            sim_config_path=args.config,
            runs=args.runs,
            seed_base=args.seed,
            intervals=args.intervals,
            use_transformer=args.use_transformer,
            checkpoint=args.checkpoint,
            model_config=args.model_config,
            temperature=args.temperature,
            no_quantized_transformer=args.no_quantized_transformer,
            disable_energy_model=bool(cfg["disable_energy_model"]),
            disable_routing_model=bool(cfg["disable_routing_model"]),
            disable_blockchain=bool(cfg["disable_blockchain"]),
        )
        all_rows.append(cfg_df)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df["value"] = out_df["value"].astype(float).round(6)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved ablation results: {args.output}")


if __name__ == "__main__":
    main()
