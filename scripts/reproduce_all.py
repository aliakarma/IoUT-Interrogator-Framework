"""
Artifact Reproduction Script
============================
LEGACY ENTRY POINT (optional): maintained for backward compatibility.
Reproduces packaged Python artifact outputs (30 runs, seed=42).

Usage:
    python scripts/reproduce_all.py
    python scripts/reproduce_all.py --seed 42 --runs 30
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python


def main():
    ensure_supported_python()
    print("[legacy] scripts/reproduce_all.py is an optional backward-compatible entry point.")
    parser = argparse.ArgumentParser(
        description="Reproduce Python artifact outputs",
        allow_abbrev=False,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--quick", action="store_true",
                        help="Delegate to quick pipeline mode (2 runs, 2 epochs, 5 intervals).")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training and reuse existing checkpoint.")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            f"Unrecognized arguments: {' '.join(unknown)}. "
            "Run with --help to see supported flags."
        )

    print("=== Full Artifact Reproduction ===")
    print(f"  Seed: {args.seed}, Runs: {args.runs}, Intervals: {args.intervals}")
    print("  Expected runtime: ~10-20 minutes on CPU\n")

    # Delegate to full pipeline with paper-matching parameters
    cmd = [
        sys.executable,
        "scripts/run_full_pipeline.py",
        "--seed", str(args.seed),
        "--runs", str(args.runs),
        "--intervals", str(args.intervals),
        "--use-transformer", "True",
    ]
    if args.quick:
        cmd.append("--quick")
    if args.skip_training:
        cmd.append("--skip-training")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
