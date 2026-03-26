"""
Behavioral Sequence Data Generator
====================================
Generates synthetic behavioral feature sequences for IoUT agents and saves
them as JSON (sequences) and CSV (labels) for model training and inference.

Each sequence has shape (K, 5) where:
  K = sequence length (default 64)
  5 features: timing_var, retx_freq, routing_stab, neighbor_churn, protocol_comp

Usage:
    python simulation/scripts/generate_behavioral_data.py \
        --agents 50 --seq-len 64 --num-sequences 500 \
        --adv-fraction 0.15 --seed 42 \
        --out-dir data/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import List, Tuple


ATTACK_TYPES = [
    "selective_packet_drop",
    "route_manipulation",
    "transmission_burst",
    "coordinated_insider",
]


def generate_legitimate_sequence(K: int, rng: np.random.Generator) -> np.ndarray:
    """Generate K-step behavioral sequence for a legitimate agent."""
    seq = np.zeros((K, 5), dtype=np.float32)
    for k in range(K):
        seq[k] = [
            rng.beta(2, 8),    # timing variance: low
            rng.beta(1, 9),    # retransmission frequency: low
            rng.beta(8, 2),    # routing stability: high
            rng.beta(1, 9),    # neighbor churn: low
            rng.beta(9, 1),    # protocol compliance: high
        ]
    return seq


def generate_adversarial_sequence(K: int, attack_type: str,
                                   rng: np.random.Generator) -> np.ndarray:
    """Generate K-step behavioral sequence for an adversarial agent."""
    seq = np.zeros((K, 5), dtype=np.float32)
    for k in range(K):
        if attack_type == "selective_packet_drop":
            row = [rng.beta(4, 4), rng.beta(7, 3), rng.beta(4, 6),
                   rng.beta(2, 8), rng.beta(6, 4)]
        elif attack_type == "route_manipulation":
            row = [rng.beta(3, 7), rng.beta(2, 8), rng.beta(2, 8),
                   rng.beta(7, 3), rng.beta(4, 6)]
        elif attack_type == "transmission_burst":
            row = [rng.beta(8, 2), rng.beta(8, 2), rng.beta(5, 5),
                   rng.beta(3, 7), rng.beta(5, 5)]
        else:  # coordinated_insider
            row = [rng.beta(4, 6), rng.beta(5, 5), rng.beta(5, 5),
                   rng.beta(4, 6), rng.beta(4, 6)]
        seq[k] = row
    return seq


def generate_dataset(num_sequences: int, K: int, adv_fraction: float,
                      seed: int) -> Tuple[List[dict], pd.DataFrame]:
    """
    Generate a full dataset of behavioral sequences with labels.

    Returns:
        sequences: list of dicts {agent_id, sequence, label, attack_type}
        label_df:  DataFrame with agent_id, label (0=legitimate, 1=adversarial),
                   attack_type
    """
    rng = np.random.default_rng(seed)
    n_adv = int(num_sequences * adv_fraction)
    n_legit = num_sequences - n_adv

    sequences = []
    rows = []

    # Legitimate agents
    for i in range(n_legit):
        seq = generate_legitimate_sequence(K, rng)
        sequences.append({
            "agent_id": i,
            "label": 0,
            "attack_type": "none",
            "sequence": seq.tolist(),
        })
        rows.append({"agent_id": i, "label": 0, "attack_type": "none"})

    # Adversarial agents
    rng_py = np.random.default_rng(seed + 1)
    attack_pool = ATTACK_TYPES
    for j in range(n_adv):
        agent_id = n_legit + j
        atk = attack_pool[int(rng_py.integers(0, len(attack_pool)))]
        seq = generate_adversarial_sequence(K, atk, rng)
        sequences.append({
            "agent_id": agent_id,
            "label": 1,
            "attack_type": atk,
            "sequence": seq.tolist(),
        })
        rows.append({"agent_id": agent_id, "label": 1, "attack_type": atk})

    label_df = pd.DataFrame(rows)
    return sequences, label_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic IoUT behavioral sequence dataset"
    )
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-sequences", type=int, default=500)
    parser.add_argument("--adv-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "sample"), exist_ok=True)

    print(f"Generating {args.num_sequences} behavioral sequences "
          f"(K={args.seq_len}, adv_fraction={args.adv_fraction}, seed={args.seed})...")

    sequences, label_df = generate_dataset(
        num_sequences=args.num_sequences,
        K=args.seq_len,
        adv_fraction=args.adv_fraction,
        seed=args.seed,
    )

    # Save full dataset
    seq_path = os.path.join(args.out_dir, "raw", "behavioral_sequences.json")
    with open(seq_path, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved sequences: {seq_path}  ({len(sequences)} records)")

    label_path = os.path.join(args.out_dir, "raw", "labels.csv")
    label_df.to_csv(label_path, index=False)
    print(f"Saved labels:    {label_path}")

    # Save small sample (10 sequences) for quick demos
    sample_seqs = sequences[:10]
    sample_path = os.path.join(args.out_dir, "sample", "sample_sequences.json")
    with open(sample_path, "w") as f:
        json.dump(sample_seqs, f, indent=2)
    print(f"Saved sample:    {sample_path}  (10 records)")

    # Dataset statistics
    n_adv = label_df["label"].sum()
    n_legit = len(label_df) - n_adv
    print(f"\nDataset statistics:")
    print(f"  Total sequences:  {len(sequences)}")
    print(f"  Legitimate:       {n_legit} ({100*n_legit/len(sequences):.1f}%)")
    print(f"  Adversarial:      {n_adv} ({100*n_adv/len(sequences):.1f}%)")
    print(f"  Attack breakdown: {label_df[label_df.label==1].attack_type.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
