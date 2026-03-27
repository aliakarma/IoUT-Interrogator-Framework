"""
Behavioral Sequence Data Generator  [FIXED v2]
================================================
Generates synthetic behavioral feature sequences for IoUT agents with
realistic class overlap, environmental noise, adversarial mimicry, and
temporal correlation — ensuring the dataset is non-trivially learnable.

ROOT CAUSES FIXED vs v1:
  RC-D1  Trivially separable distributions:
           v1 used Beta distributions with 3-8 std separation between classes.
           A 4-layer transformer fits this perfectly in 5 epochs → loss=0, acc=100%.
           FIX: Tighten Beta parameters so feature means are within 1.0-2.0 std.
                Add per-step Gaussian noise σ=0.08 to simulate channel variation.
                Add temporal drift to make features autocorrelated over time.
  RC-D2  Missing adversarial mimicry:
           v1 had no "smart" adversary that tries to mimic legitimate patterns.
           FIX: Add "low_and_slow" attack type — subtle deviations that partially
                overlap with legitimate behavior, creating irreducible Bayes error.
  RC-D3  Sample file had only legitimate agents:
           v1 always saved the first 10 records which were all legitimate (label=0).
           Inference test: 10/10 flagged=0, true_adv=0, misleading accuracy=100%.
           FIX: Stratified sample — always saves 7 legitimate + 3 adversarial.
  RC-D4  No temporal correlation:
         v1 drew each time step independently from same distribution.
         FIX: Add exponential-smoothing temporal correlation — each step
             partially inherits the previous state while reverting toward
             a behavioral mean, creating realistic persistence.

Usage:
    python simulation/scripts/generate_behavioral_data.py \
        --num-sequences 500 --seq-len 64 \
        --adv-fraction 0.15 --seed 42 \
        --out-dir data/
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Attack type registry
# ---------------------------------------------------------------------------

ATTACK_TYPES = [
    "selective_packet_drop",
    "route_manipulation",
    "transmission_burst",
    "coordinated_insider",
    "low_and_slow",          # NEW: mimicry attack — subtle, hard to detect
]

# ---------------------------------------------------------------------------
# Feature generation helpers
# ---------------------------------------------------------------------------

# Global default noise level: sigma = 0.08 simulates acoustic channel variation
# and measurement uncertainty in the interrogator's metadata collection.
DEFAULT_NOISE_SIGMA = 0.08

# Exponential-smoothing persistence coefficient: φ = 0.4
# Each time step partially inherits the previous step's state.
AR_PHI = 0.4


def _ar1_sequence(mean_vec: np.ndarray, K: int,
                  rng: np.random.Generator,
                  phi: float = AR_PHI,
                  noise_sigma: float = DEFAULT_NOISE_SIGMA) -> np.ndarray:
    """
    Generate a K-step temporally correlated sequence around a mean feature vector.

    Each step: x_t = phi * x_{t-1} + (1-phi) * mean + eps_t
    This is an exponential-smoothing state process (mean-reverting with noise),
    not strict AR(1) around zero.
    """
    seq = np.zeros((K, len(mean_vec)), dtype=np.float32)
    x_prev = mean_vec.copy()
    for k in range(K):
        eps      = rng.normal(0.0, noise_sigma, size=mean_vec.shape).astype(np.float32)
        x_curr   = phi * x_prev + (1.0 - phi) * mean_vec + eps
        seq[k]   = np.clip(x_curr, 0.0, 1.0)
        x_prev   = seq[k]
    return seq


# ---------------------------------------------------------------------------
# Legitimate agent sequences
# ---------------------------------------------------------------------------

# v1 used Beta(2,8), Beta(1,9), Beta(8,2), Beta(1,9), Beta(9,1)
# Those have means [0.20, 0.10, 0.80, 0.10, 0.90] — very extreme.
# v2 uses means closer to centre so classes overlap more realistically.
_LEGIT_MEAN = np.array([0.22, 0.12, 0.78, 0.13, 0.85], dtype=np.float32)
# Feature σ ≈ 0.10-0.12 after noise → target separation ≤ 2.0x with adversarial


def generate_legitimate_sequence(
    K: int,
    rng: np.random.Generator,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    phi: float = AR_PHI,
) -> np.ndarray:
    """
    Temporally correlated sequence centered on legitimate behavioral means.
    Legitimate agents: low timing variance, low retransmission, high routing
    stability, low neighbor churn, high protocol compliance.
    """
    # Small per-agent jitter: each legitimate agent has slightly different baseline
    jitter = rng.normal(0.0, 0.04, size=_LEGIT_MEAN.shape).astype(np.float32)
    agent_mean = np.clip(_LEGIT_MEAN + jitter, 0.05, 0.95)
    return _ar1_sequence(agent_mean, K, rng, phi=phi, noise_sigma=noise_sigma)


# ---------------------------------------------------------------------------
# Adversarial agent sequences
# ---------------------------------------------------------------------------

# Tightened adversarial means — still deviating from legitimate, but closer
# to legitimate means than v1, creating non-trivial classification difficulty.
_ADV_MEANS = {
    # v1: retx ~Beta(7,3)→0.70. v2: 0.52 — less extreme, more overlap
    "selective_packet_drop": np.array([0.38, 0.52, 0.48, 0.22, 0.62], dtype=np.float32),
    # v1: routing_stab ~Beta(2,8)→0.20. v2: 0.38 — less extreme
    "route_manipulation":    np.array([0.28, 0.22, 0.38, 0.58, 0.52], dtype=np.float32),
    # v1: timing_var ~Beta(8,2)→0.80. v2: 0.65 — closer to legitimate
    "transmission_burst":    np.array([0.65, 0.62, 0.52, 0.38, 0.55], dtype=np.float32),
    # v1: coord_insider was already relatively subtle
    "coordinated_insider":   np.array([0.35, 0.42, 0.55, 0.38, 0.48], dtype=np.float32),
    # NEW: low_and_slow — barely deviates, partial overlap with legitimate
    # This is the hardest attack to detect (mimicry strategy)
    "low_and_slow":          np.array([0.28, 0.20, 0.68, 0.20, 0.72], dtype=np.float32),
}


def generate_adversarial_sequence(
    K: int,
    attack_type: str,
    rng: np.random.Generator,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    phi: float = AR_PHI,
) -> np.ndarray:
    """
    Temporally correlated sequence centered on adversarial behavioral means.
    The tightened means create realistic class overlap vs legitimate agents.
    'low_and_slow' attack deliberately mimics legitimate patterns with subtle
    deviations — creating irreducible Bayes error in the classification task.
    """
    mean = _ADV_MEANS.get(attack_type, _ADV_MEANS["coordinated_insider"]).copy()

    # Per-agent jitter (larger for adversarial to simulate inconsistent behavior)
    jitter = rng.normal(0.0, 0.05, size=mean.shape).astype(np.float32)
    agent_mean = np.clip(mean + jitter, 0.05, 0.95)

    # Adversarial agents have higher temporal variance (less stable behavior)
    return _ar1_sequence(agent_mean, K, rng, phi=phi * 0.7, noise_sigma=noise_sigma * 1.3)


def _parse_float_list(value: str) -> List[float]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"Expected at least one numeric value in '{value}'.")
    return values


def _format_float_token(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _parse_attack_mix_spec(spec: str) -> Optional[List[Tuple[str, float]]]:
    """
    Parse one attack mix specification.

    Supported formats:
      - "uniform" (default round-robin distribution, preserves legacy behavior)
      - "name1:0.4,name2:0.6"
    """
    spec = spec.strip()
    if not spec or spec.lower() == "uniform":
        return None

    pairs: List[Tuple[str, float]] = []
    total = 0.0
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid attack mix token '{chunk}'. Use format type:weight."
            )
        name, weight_s = chunk.split(":", 1)
        name = name.strip()
        if name not in ATTACK_TYPES:
            raise ValueError(
                f"Unknown attack type '{name}'. Allowed: {', '.join(ATTACK_TYPES)}"
            )
        weight = float(weight_s.strip())
        if weight <= 0:
            raise ValueError("Attack mix weights must be positive.")
        pairs.append((name, weight))
        total += weight

    if not pairs:
        raise ValueError("Attack mix specification is empty.")

    return [(name, weight / total) for name, weight in pairs]


def _sample_attack_type(
    rng: np.random.Generator,
    attack_mix: Optional[List[Tuple[str, float]]],
    index: int,
) -> str:
    if attack_mix is None:
        # Legacy behavior: deterministic round-robin allocation.
        return ATTACK_TYPES[index % len(ATTACK_TYPES)]

    names = [name for name, _ in attack_mix]
    probs = np.array([weight for _, weight in attack_mix], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_sequences: int,
    K: int,
    adv_fraction: float,
    seed: int,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    attack_mix: Optional[List[Tuple[str, float]]] = None,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Generate a full dataset of behavioral sequences with labels.

    Changes vs v1:
            - Exponential-smoothing temporal correlation on all sequences.
      - Tightened class means for realistic overlap.
      - Gaussian noise on every feature step.
      - 'low_and_slow' mimicry attack type added.
      - Uses separate RNG streams for legit vs adversarial.

    Returns:
        sequences: list of dicts {agent_id, sequence, label, attack_type}
        label_df:  DataFrame with agent_id, label, attack_type
    """
    rng_legit = np.random.default_rng(seed)
    rng_adv   = np.random.default_rng(seed + 1000)   # independent stream

    n_adv   = int(num_sequences * adv_fraction)
    n_legit = num_sequences - n_adv

    sequences: List[dict] = []
    rows: List[dict]      = []

    # ── Legitimate agents ─────────────────────────────────────────────────
    for i in range(n_legit):
        seq = generate_legitimate_sequence(K, rng_legit, noise_sigma=noise_sigma)
        sequences.append({
            "agent_id":   i,
            "label":      0,
            "attack_type":"none",
            "sequence":   seq.tolist(),
        })
        rows.append({"agent_id": i, "label": 0, "attack_type": "none"})

    # ── Adversarial agents ────────────────────────────────────────────────
    # Distribute attack types evenly across adversarial agents
    for j in range(n_adv):
        agent_id  = n_legit + j
        atk = _sample_attack_type(rng_adv, attack_mix=attack_mix, index=j)
        seq = generate_adversarial_sequence(K, atk, rng_adv, noise_sigma=noise_sigma)
        sequences.append({
            "agent_id":   agent_id,
            "label":      1,
            "attack_type": atk,
            "sequence":   seq.tolist(),
        })
        rows.append({"agent_id": agent_id, "label": 1, "attack_type": atk})

    label_df = pd.DataFrame(rows)
    return sequences, label_df


def generate_stratified_sample(
    sequences: List[dict],
    n_legit_sample: int = 7,
    n_adv_sample:   int = 3,
    seed: int = 99,
) -> List[dict]:
    """
    Return a stratified sample with guaranteed representation of both classes.

    FIX vs v1: v1 returned sequences[:10] which were always all legitimate.
    This caused inference tests to show 0/10 adversarial detected (no adversarial
    in the test set), misleadingly appearing as correct behaviour but masking
    any model failures on adversarial inputs.
    """
    rng = np.random.default_rng(seed)

    legit_pool = [s for s in sequences if s["label"] == 0]
    adv_pool   = [s for s in sequences if s["label"] == 1]

    n_l = min(n_legit_sample, len(legit_pool))
    n_a = min(n_adv_sample,   len(adv_pool))

    legit_idx = rng.choice(len(legit_pool), size=n_l, replace=False).tolist()
    adv_idx   = rng.choice(len(adv_pool),   size=n_a, replace=False).tolist()

    sample = [legit_pool[i] for i in legit_idx] + \
             [adv_pool[i]   for i in adv_idx]

    # Shuffle so adversarial agents aren't always at the end
    indices = rng.permutation(len(sample)).tolist()
    return [sample[i] for i in indices]


# ---------------------------------------------------------------------------
# Dataset statistics helper
# ---------------------------------------------------------------------------

def print_feature_stats(sequences: List[dict], n_show: int = 3):
    """Print per-class feature mean/std to verify overlap is realistic."""
    import numpy as np
    feat_names = ["timing_var", "retx_freq", "routing_stab",
                  "neighbor_churn", "protocol_comp"]

    legit_seqs = [np.array(s["sequence"]) for s in sequences if s["label"] == 0]
    adv_seqs   = [np.array(s["sequence"]) for s in sequences if s["label"] == 1]

    if not legit_seqs or not adv_seqs:
        return

    # Mean over all time steps and agents
    legit_flat = np.concatenate(legit_seqs, axis=0)   # (N_l * K, 5)
    adv_flat   = np.concatenate(adv_seqs,   axis=0)   # (N_a * K, 5)

    print(f"\n  {'Feature':<20} {'Legit μ±σ':<16} {'Adv μ±σ':<16} {'Sep (x std)'}")
    print(f"  {'-'*65}")
    for i, fname in enumerate(feat_names):
        lm, ls = legit_flat[:, i].mean(), legit_flat[:, i].std()
        am, as_ = adv_flat[:, i].mean(),   adv_flat[:, i].std()
        sep = abs(lm - am) / ((ls + as_) / 2 + 1e-8)
        print(f"  {fname:<20} {lm:.2f}±{ls:.2f}       {am:.2f}±{as_:.2f}       {sep:.1f}x")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic IoUT behavioral sequence dataset [v2 — FIXED]"
    )
    parser.add_argument("--agents",         type=int,   default=50)
    parser.add_argument("--seq-len",        type=int,   default=64)
    parser.add_argument("--num-sequences",  type=int,   default=500)
    parser.add_argument("--adv-fraction",   type=float, default=0.15)
    parser.add_argument(
        "--adversarial-fraction",
        default=None,
        help=(
            "Comma-separated adversarial fractions for OOD generation, "
            "e.g. 0.1,0.15,0.3. If omitted, uses --adv-fraction."
        ),
    )
    parser.add_argument(
        "--noise-level",
        default=None,
        help=(
            "Comma-separated Gaussian noise sigmas for OOD generation, "
            "e.g. 0.05,0.08,0.12. If omitted, uses default 0.08."
        ),
    )
    parser.add_argument(
        "--attack-mix",
        default="uniform",
        help=(
            "Attack mix spec for adversarial samples. Use 'uniform' for default "
            "round-robin behavior, or weighted format like "
            "selective_packet_drop:0.4,low_and_slow:0.6. "
            "For multiple configs, separate specs with ';'."
        ),
    )
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--out-dir",        default="data/")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "raw"),       exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "sample"),    exist_ok=True)

    adv_values = (
        _parse_float_list(args.adversarial_fraction)
        if args.adversarial_fraction is not None
        else [args.adv_fraction]
    )
    noise_values = (
        _parse_float_list(args.noise_level)
        if args.noise_level is not None
        else [DEFAULT_NOISE_SIGMA]
    )
    attack_mix_specs = [s.strip() for s in args.attack_mix.split(";") if s.strip()]
    if not attack_mix_specs:
        attack_mix_specs = ["uniform"]

    configs: List[Dict[str, object]] = []
    for adv in adv_values:
        if not (0.0 <= adv <= 1.0):
            raise ValueError(f"adversarial fraction must be in [0,1], got {adv}")
        for noise in noise_values:
            if noise < 0:
                raise ValueError(f"noise level must be >= 0, got {noise}")
            for mix_spec in attack_mix_specs:
                configs.append(
                    {
                        "adv_fraction": float(adv),
                        "noise_sigma": float(noise),
                        "attack_mix_spec": mix_spec,
                        "attack_mix": _parse_attack_mix_spec(mix_spec),
                    }
                )

    print(
        f"Generating {len(configs)} dataset configuration(s) "
        f"(num_sequences={args.num_sequences}, K={args.seq_len}, seed={args.seed}, ar_phi={AR_PHI})"
    )

    raw_dir = os.path.join(args.out_dir, "raw")
    used_names: Dict[str, int] = {}

    default_single_config = (
        len(configs) == 1
        and configs[0]["adv_fraction"] == args.adv_fraction
        and configs[0]["noise_sigma"] == DEFAULT_NOISE_SIGMA
        and configs[0]["attack_mix"] is None
    )

    for idx, cfg in enumerate(configs):
        adv_fraction = float(cfg["adv_fraction"])
        noise_sigma = float(cfg["noise_sigma"])
        attack_mix = cfg["attack_mix"]
        attack_mix_spec = str(cfg["attack_mix_spec"])
        cfg_seed = args.seed + idx * 10000

        sequences, label_df = generate_dataset(
            num_sequences=args.num_sequences,
            K=args.seq_len,
            adv_fraction=adv_fraction,
            seed=cfg_seed,
            noise_sigma=noise_sigma,
            attack_mix=attack_mix,
        )

        base_name = (
            f"data_adv{_format_float_token(adv_fraction)}"
            f"_noise{_format_float_token(noise_sigma)}"
        )
        if attack_mix is not None:
            base_name = f"{base_name}_custommix"

        duplicate_count = used_names.get(base_name, 0)
        used_names[base_name] = duplicate_count + 1
        if duplicate_count > 0:
            base_name = f"{base_name}_{duplicate_count + 1}"

        seq_path_named = os.path.join(raw_dir, f"{base_name}.json")
        labels_path_named = os.path.join(raw_dir, f"{base_name}_labels.csv")

        with open(seq_path_named, "w") as f:
            json.dump(sequences, f, indent=2)
        label_df.to_csv(labels_path_named, index=False)

        print(
            f"[{idx+1}/{len(configs)}] adv={adv_fraction:.3f}, noise={noise_sigma:.3f}, "
            f"mix={attack_mix_spec}, seed={cfg_seed}"
        )
        print(f"  Saved sequences: {seq_path_named}")
        print(f"  Saved labels:    {labels_path_named}")

        if default_single_config:
            # Preserve legacy default outputs exactly for existing workflows.
            seq_path = os.path.join(raw_dir, "behavioral_sequences.json")
            label_path = os.path.join(raw_dir, "labels.csv")

            with open(seq_path, "w") as f:
                json.dump(sequences, f, indent=2)
            label_df.to_csv(label_path, index=False)
            print(f"  Saved legacy sequences: {seq_path}  ({len(sequences)} records)")
            print(f"  Saved legacy labels:    {label_path}")

            sample_seqs = generate_stratified_sample(
                sequences, n_legit_sample=7, n_adv_sample=3, seed=args.seed + 99
            )
            sample_path = os.path.join(args.out_dir, "sample", "sample_sequences.json")
            with open(sample_path, "w") as f:
                json.dump(sample_seqs, f, indent=2)

            n_adv_sample = sum(1 for s in sample_seqs if s["label"] == 1)
            n_legit_sample = len(sample_seqs) - n_adv_sample
            print(
                f"  Saved legacy sample:    {sample_path} "
                f"({len(sample_seqs)} records: {n_legit_sample} legit, {n_adv_sample} adversarial)"
            )

            n_adv = int(label_df["label"].sum())
            n_legit = len(label_df) - n_adv
            print("\nDataset statistics:")
            print(f"  Total sequences : {len(sequences)}")
            print(f"  Legitimate      : {n_legit} ({100*n_legit/len(sequences):.1f}%)")
            print(f"  Adversarial     : {n_adv} ({100*n_adv/len(sequences):.1f}%)")
            print(
                f"  Attack breakdown: "
                f"{label_df[label_df.label==1].attack_type.value_counts().to_dict()}"
            )

            print("\nFeature overlap analysis (target: separation < 2.5x):")
            print_feature_stats(sequences)
            print()
            print("NOTE: Low-and-slow mimicry attack intentionally overlaps with legitimate")
            print("      distributions - this creates irreducible Bayes error (~5-10%)")
            print("      so the model cannot achieve perfect accuracy.")


if __name__ == "__main__":
    main()
