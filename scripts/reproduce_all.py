"""
Artifact Reproduction Script
============================
Reproduces packaged Python artifact outputs (30 runs, seed=42).

Usage:
    python scripts/reproduce_all.py
    python scripts/reproduce_all.py --seed 42 --runs 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Reproduce Python artifact outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    print("=== Full Artifact Reproduction ===")
    print(f"  Seed: {args.seed}, Runs: {args.runs}")
    print("  Expected runtime: ~10-20 minutes on CPU\n")

    # Delegate to full pipeline with paper-matching parameters
    os.system(
        f"python scripts/run_full_pipeline.py "
        f"--seed {args.seed} --runs {args.runs} --intervals 20 --use-transformer True"
    )


if __name__ == "__main__":
    main()
