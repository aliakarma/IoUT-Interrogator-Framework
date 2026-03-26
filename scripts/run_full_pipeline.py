"""
Full Reproducibility Pipeline
==============================
Runs the complete experiment pipeline in one command:
  1. Generate synthetic behavioral data
    2. Train the transformer trust model
    3. Run multi-run simulation
  4. Run inference on sample sequences
  5. Generate all figures and statistical summary

Usage:
    python scripts/run_full_pipeline.py --seed 42 --runs 5
    python scripts/run_full_pipeline.py --seed 42 --runs 30  # full paper results
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


def main():
    parser = argparse.ArgumentParser(description="Full IoUT reproducibility pipeline")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--runs",     type=int, default=30,
                        help="Number of simulation runs (30 for paper results)")
    parser.add_argument("--intervals",type=int, default=20)
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
        "--simulation-use-transformer",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=None,
        help="Deprecated alias for --use-transformer.",
    )
    args = parser.parse_args()

    if args.simulation_use_transformer is not None:
        args.use_transformer = args.simulation_use_transformer

    t0 = time.time()

    # ── Step 1: Generate data ─────────────────────────────────────────────
    step("1/5 — Generate synthetic behavioral data")
    from simulation.scripts.generate_behavioral_data import generate_dataset, main as gen_main
    import sys as _sys
    _sys.argv = [
        "generate_behavioral_data.py",
        "--num-sequences", "500",
        "--seq-len", "64",
        "--adv-fraction", "0.15",
        "--seed", str(args.seed),
        "--out-dir", "data/",
    ]
    gen_main()

    # ── Step 2: Train model ───────────────────────────────────────────────
    if not args.skip_training:
        step("2/5 — Train transformer trust model")
        from model.inference.train import main as train_main
        _sys.argv = [
            "train.py",
            "--config",         "model/configs/transformer_config.json",
            "--data",           "data/raw/behavioral_sequences.json",
            "--checkpoint-dir", "model/checkpoints/",
            "--seed",           str(args.seed),
        ]
        train_main()
    else:
        step("2/5 — Skipping model training (--skip-training flag set)")

    # ── Step 3: Run simulation ────────────────────────────────────────────
    step("3/5 — Run multi-agent IoUT simulation")
    from simulation.scripts.run_simulation import main as sim_main
    _sys.argv = [
        "run_simulation.py",
        "--config", "simulation/configs/simulation_params.json",
        "--runs",   str(args.runs),
        "--seed",   str(args.seed),
        "--intervals", str(args.intervals),
        "--output", "simulation/outputs/results.csv",
    ]
    if args.use_transformer:
        _sys.argv += [
            "--use-transformer",
            "True",
            "--no-quantized-transformer",
            "--checkpoint", "model/checkpoints/best_model.pt",
            "--model-config", "model/configs/transformer_config.json",
        ]
    sim_main()

    # ── Step 4: Run inference ─────────────────────────────────────────────
    step("4/5 — Run trust inference on sample sequences")
    from model.inference.infer import run_inference
    run_inference(
        checkpoint_path="model/checkpoints/best_model.pt",
        config_path="model/configs/transformer_config.json",
        sequences_path="data/sample/sample_sequences.json",
        output_path="data/processed/trust_scores.csv",
        tau_min=0.65,
    )

    # ── Step 5: Generate all figures ──────────────────────────────────────
    step("5/5 — Generate all figures and statistics")

    from analysis.plot_trust_accuracy import load_or_generate_data, plot_trust_accuracy
    df = load_or_generate_data("simulation/outputs/results.csv")
    plot_trust_accuracy(df, "analysis/plots/trust_accuracy.png")

    from analysis.plot_energy import load_or_generate_data as le, plot_energy
    plot_energy(le("simulation/outputs/results.csv"), "analysis/plots/energy_comparison.png")

    from analysis.plot_pdr import load_or_generate_data as lp, plot_pdr
    plot_pdr(lp("simulation/outputs/results.csv"), "analysis/plots/pdr_comparison.png")

    from analysis.plot_ablation import plot_ablation
    plot_ablation(
        "simulation/outputs/results.csv",
        "analysis/plots/ablation_study.png",
    )

    from analysis.statistical_summary import main as stats_main
    _sys.argv = [
        "statistical_summary.py",
        "--input",  "simulation/outputs/results.csv",
        "--output", "analysis/stats/summary_table.csv",
    ]
    stats_main()

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Figures:    analysis/plots/")
    print(f"  Statistics: analysis/stats/summary_table.csv")
    print(f"  Results:    simulation/outputs/results.csv")
    print(f"  Checkpoint: model/checkpoints/best_model.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
