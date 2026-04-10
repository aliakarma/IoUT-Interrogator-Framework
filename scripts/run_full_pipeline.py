"""
Full Reproducibility Pipeline
==============================
LEGACY OPTIONAL PIPELINE: kept for compatibility; primary entry points are
run_pipeline.py and run_unsw_publication_pipeline.py.
Runs the complete experiment pipeline in one command:
  1. Generate synthetic behavioral data
    2. Train the transformer trust model
    3. Run multi-run simulation
  4. Run inference on sample sequences
  5. Generate all figures and statistical summary

Usage:
    python scripts/run_full_pipeline.py --seed 42 --runs 5
    python scripts/run_full_pipeline.py --seed 42 --runs 30  # full paper results
    python scripts/run_full_pipeline.py --quick
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python


def step(label: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full IoUT reproducibility pipeline",
        allow_abbrev=False,
    )
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--runs",     type=int, default=30,
                        help="Number of simulation runs (30 for paper results)")
    parser.add_argument("--intervals",type=int, default=20)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode: use 2 simulation seeds/runs, 2 training epochs, and 5 monitoring intervals."
        ),
    )
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (useful if checkpoint exists)")
    parser.add_argument(
        "--use-transformer",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable transformer trust scores inside simulation loop (default: True).",
    )
    parser.add_argument(
        "--output-root",
        default="results/full_pipeline",
        help="Root directory for all generated artifacts (default: results/full_pipeline).",
    )
    return parser


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            f"Unrecognized arguments: {' '.join(unknown)}. "
            "Run with --help to see supported flags."
        )
    return args


def _build_quick_train_config(base_config_path: str) -> str:
    """Create a temporary training config that caps epochs to 2 for quick mode."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("training", {})["epochs"] = 2

    fd, temp_path = tempfile.mkstemp(prefix="transformer_quick_", suffix=".json")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return temp_path


def main():
    ensure_supported_python()
    args = _parse_args()
    print("[legacy optional] scripts/run_full_pipeline.py writes artifacts under results/ by default.")

    output_root = os.path.abspath(args.output_root)
    data_out_dir = os.path.join(output_root, "data")
    checkpoints_dir = os.path.join(output_root, "checkpoints")
    sim_out_dir = os.path.join(output_root, "simulation")
    processed_dir = os.path.join(output_root, "processed")
    plots_dir = os.path.join(output_root, "plots")
    stats_dir = os.path.join(output_root, "stats")
    export_dir = os.path.join(output_root, "exports")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(sim_out_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    generated_data_path = os.path.join(data_out_dir, "raw", "behavioral_sequences.json")
    sim_results_path = os.path.join(sim_out_dir, "results.csv")
    sim_raw_results_path = os.path.join(sim_out_dir, "raw_results.csv")
    checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
    preprocess_stats_path = os.path.join(checkpoints_dir, "preprocessing_stats.json")
    eval_metrics_path = os.path.join(checkpoints_dir, "evaluation_metrics.json")
    trust_scores_path = os.path.join(processed_dir, "trust_scores.csv")
    stats_summary_path = os.path.join(stats_dir, "summary_table.csv")
    provenance_path = os.path.join(export_dir, "provenance_manifest.json")

    effective_runs = args.runs
    effective_intervals = args.intervals
    effective_seed_count = 1

    if args.quick:
        effective_runs = 2
        effective_intervals = 5
        effective_seed_count = 2
        print("[quick mode] Overrides applied:")
        print("  simulation seed count (runs): 2")
        print("  training epochs: 2")
        print("  monitoring intervals: 5")
        print()

    t0 = time.time()

    # ── Step 1: Generate data ─────────────────────────────────────────────
    step("1/5 — Generate synthetic behavioral data")
    from simulation.scripts.generate_behavioral_data import main as gen_main
    import sys as _sys
    _sys.argv = [
        "generate_behavioral_data.py",
        "--num-sequences", "500",
        "--seq-len", "64",
        "--adv-fraction", "0.15",
        "--seed", str(args.seed),
        "--out-dir", data_out_dir,
    ]
    gen_main()

    # ── Step 2: Train model ───────────────────────────────────────────────
    quick_train_config = None
    if not args.skip_training:
        step("2/5 — Train transformer trust model")
        from model.inference.train import main as train_main

        train_config_path = "model/configs/transformer_config.json"
        if args.quick:
            quick_train_config = _build_quick_train_config(train_config_path)
            train_config_path = quick_train_config

        _sys.argv = [
            "train.py",
            "--config",         train_config_path,
            "--data",           generated_data_path,
            "--checkpoint-dir", checkpoints_dir,
            "--seed",           str(args.seed),
        ]
        try:
            train_main()
        finally:
            if quick_train_config and os.path.exists(quick_train_config):
                os.remove(quick_train_config)
    else:
        step("2/5 — Skipping model training (--skip-training flag set)")

    # ── Step 3: Run simulation ────────────────────────────────────────────
    step("3/5 — Run multi-agent IoUT simulation")
    from simulation.scripts.run_simulation import main as sim_main
    _sys.argv = [
        "run_simulation.py",
        "--config", "simulation/configs/simulation_params.json",
        "--runs",   str(effective_runs),
        "--seed",   str(args.seed),
        "--intervals", str(effective_intervals),
        "--output", sim_results_path,
        "--raw-output", sim_raw_results_path,
    ]
    if args.use_transformer:
        _sys.argv += [
            "--use-transformer",
            "True",
            "--no-quantized-transformer",
            "--checkpoint", checkpoint_path,
            "--model-config", "model/configs/transformer_config.json",
        ]
    sim_main()

    required_outputs = [
        sim_results_path,
        sim_raw_results_path,
    ]
    for output_path in required_outputs:
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected pipeline output was not created: {output_path}")

    # ── Step 4: Run inference ─────────────────────────────────────────────
    step("4/5 — Run trust inference on sample sequences")
    from model.inference.infer import run_inference
    run_inference(
        checkpoint_path=checkpoint_path,
        config_path="model/configs/transformer_config.json",
        sequences_path="data/sample/sample_sequences.json",
        output_path=trust_scores_path,
        tau_min=0.65,
    )

    # ── Step 5: Generate all figures ──────────────────────────────────────
    step("5/5 — Generate all figures and statistics")

    from analysis.plot_trust_accuracy import load_or_generate_data, plot_trust_accuracy
    df = load_or_generate_data(sim_results_path)
    plot_trust_accuracy(df, os.path.join(plots_dir, "trust_accuracy.png"))

    from analysis.plot_energy import load_or_generate_data as le, plot_energy
    plot_energy(le(sim_results_path), os.path.join(plots_dir, "energy_comparison.png"))

    from analysis.plot_pdr import load_or_generate_data as lp, plot_pdr
    plot_pdr(lp(sim_results_path), os.path.join(plots_dir, "pdr_comparison.png"))

    from analysis.plot_ablation import plot_ablation
    plot_ablation(
        sim_results_path,
        os.path.join(plots_dir, "ablation_study.png"),
    )

    from analysis.statistical_summary import main as stats_main
    _sys.argv = [
        "statistical_summary.py",
        "--input",  sim_raw_results_path,
        "--output", stats_summary_path,
        "--aggregate-level", "seed",
        "--bootstrap-samples", "2000",
        "--ci-level", "95",
        "--seed", str(args.seed),
    ]
    stats_main()

    from scripts.export_all_results import export_results
    export_results(
        output_dir=export_dir,
        sim_outputs_dir=sim_out_dir,
        analysis_stats_dir=stats_dir,
        checkpoint_dir=checkpoints_dir,
    )

    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    provenance = {
        "seed": args.seed,
        "runs": effective_runs,
        "intervals": effective_intervals,
        "quick": bool(args.quick),
        "use_transformer": bool(args.use_transformer),
        "artifacts": {},
    }
    artifact_paths = [
        checkpoint_path,
        preprocess_stats_path,
        eval_metrics_path,
        sim_results_path,
        sim_raw_results_path,
        stats_summary_path,
        os.path.join(export_dir, "main_metrics.csv"),
    ]
    for artifact_path in artifact_paths:
        if os.path.exists(artifact_path):
            provenance["artifacts"][artifact_path] = {
                "sha256": _sha256(artifact_path),
                "size_bytes": os.path.getsize(artifact_path),
            }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Figures:    {plots_dir}")
    print(f"  Statistics: {stats_summary_path}")
    print(f"  Results:    {sim_results_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Seed count: {effective_seed_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
