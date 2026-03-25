"""
Trust Transformer Training Script
===================================
Trains the behavioral trust transformer model from behavioral sequence data.

Usage:
    python model/inference/train.py \
        --config model/configs/transformer_config.json \
        --data data/raw/behavioral_sequences.json \
        --checkpoint-dir model/checkpoints/ \
        --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.inference.transformer_model import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the IoUT behavioral trust transformer"
    )
    parser.add_argument(
        "--config",
        default="model/configs/transformer_config.json",
        help="Path to transformer config JSON"
    )
    parser.add_argument(
        "--data",
        default="data/raw/behavioral_sequences.json",
        help="Path to behavioral sequences JSON"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="model/checkpoints/",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("=== IoUT Trust Transformer Training ===")
    print(f"  Config:          {args.config}")
    print(f"  Data:            {args.data}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("Run simulation/scripts/generate_behavioral_data.py first.")
        sys.exit(1)

    model = train_model(
        config_path=args.config,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
    )

    print(f"\nTraining complete. Best model saved to: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
