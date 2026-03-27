# Reproducibility Guide

This document provides step-by-step instructions to fully reproduce all results reported in the paper.

> **Scope note:** This repository is a Python artifact reimplementation. The
> paper prototype references NS-3 Aqua-Sim + Hyperledger Fabric; those original
> scripts are not bundled here. Treat this guide as reproducibility for the
> packaged Python artifact outputs.

---

## Environment Requirements

| Component | Version |
|---|---|
| Python | 3.9-3.11 |
| PyTorch | 2.1.x (CPU) |
| NumPy | 1.24.x |
| Pandas | 2.0.x |
| Matplotlib | 3.7.x |
| OS | Linux / macOS / Windows |
| RAM | ≥ 4 GB |
| GPU | Not required (CPU-only) |

Estimated runtime (CPU, 30 runs): **15–25 minutes**.

---

## Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate         # Linux/macOS
# venv\Scripts\activate          # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The CLI entry points enforce the supported interpreter window at runtime.
If you run under an unsupported Python version, scripts fail fast with a clear
version error message.

Alternatively, use Docker:
```bash
docker build -t ioUt-framework .
docker run --rm -v $(pwd)/results:/workspace/results ioUt-framework
```

---

## Step 2: Generate Synthetic Data

```bash
python simulation/scripts/generate_behavioral_data.py \
    --num-sequences 500 \
    --seq-len 64 \
    --adv-fraction 0.15 \
    --seed 42 \
    --out-dir data/
```

This generates:
- `data/raw/behavioral_sequences.json` — 500 sequences (K=64, 5 features)
- `data/raw/labels.csv` — agent labels and attack types
- `data/sample/sample_sequences.json` — 10 sample sequences for quick demos

---

## Step 3: Run Simulation (30 Runs)

```bash
python simulation/scripts/run_simulation.py \
    --config simulation/configs/simulation_params.json \
    --runs 30 \
    --seed 42 \
    --intervals 20 \
    --output simulation/outputs/results.csv
```

Expected output: `simulation/outputs/results.csv` with 20 rows and aggregated
mean/std columns for accuracy, precision, recall, F1, confusion matrix counts,
PDR, energy, and trust-distribution diagnostics.

---

## Step 3b: Sensitivity Studies

```bash
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42
```

Expected outputs:
- `analysis/stats/threshold_sensitivity.csv`
- `analysis/stats/sequence_length_ablation.csv`

---

## Step 4: Train the Transformer Trust Model

```bash
python model/inference/train.py \
    --config model/configs/transformer_config.json \
    --data data/raw/behavioral_sequences.json \
    --checkpoint-dir model/checkpoints/ \
    --seed 42
```

Expected: `model/checkpoints/best_model.pt` (~5–7 MB)

---

## Step 5: Run Inference

```bash
python model/inference/infer.py \
    --checkpoint model/checkpoints/best_model.pt \
    --config model/configs/transformer_config.json \
    --sequences data/sample/sample_sequences.json \
    --output data/processed/trust_scores.csv
```

---

## Step 6: Generate All Figures

```bash
python analysis/plot_all.py --input simulation/outputs/results.csv
```

Figures saved to `analysis/plots/`:

| File | Paper Figure |
|---|---|
| `trust_accuracy.png` | Figure 4 |
| `energy_comparison.png` | Figure 5 |
| `pdr_comparison.png` | Figure 6 |
| `ablation_study.png` | Table 3 visualisation |

---

## One-Command Reproduction

```bash
python scripts/reproduce_all.py --seed 42 --runs 30
```

---

## Fixed Random Seeds

All random components use seed 42 (or `42 + run_index` for multi-run experiments):

| Component | Seed |
|---|---|
| Agent deployment | `seed + 0` |
| Adversarial assignment | `seed + 0` |
| Behavioral features | `seed` (numpy rng) |
| Model training | `seed` (torch + numpy) |
| Simulation run $i$ | `seed + i` |

---

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run a specific test file
pytest tests/test_environment.py -v
```

---

## Expected Results

After 30 runs with seed 42:

| Metric | Expected Value | Notes |
|---|---|---|
| Accuracy — Proposed (simulation) | run-dependent | Use `results.csv` from your run |
| Accuracy — Bayesian (simulation) | run-dependent | Stateful Bayesian update |
| Accuracy — Static (simulation) | ~86% | 86% = legitimate fraction |
| Model Test Accuracy (transformer) | run-dependent | Dataset and seed sensitive |
| PDR — Proposed | run-dependent | Run-to-run variation |
| Energy Overhead (proposed vs static) | run-dependent | Derived from simulation model |
| Energy Residual at interval 20 | run-dependent | See simulation output |
| Trust Score Std (mixed sample) | >0.15 | <0.05 indicates constant output |

> **v3 note:** The transformer model intentionally does NOT achieve 100% accuracy
> on the training dataset. The `low_and_slow` mimicry attack creates irreducible
> Bayes error (~5-10%), ensuring the model generalizes rather than memorizes.
>
> Transformer-backed simulation path is available via
> `simulation/scripts/run_simulation.py --use-transformer`.

---

## Troubleshooting

**`ModuleNotFoundError`:** Ensure you are running from the repository root and
that `PYTHONPATH` includes the repository root:
```bash
export PYTHONPATH=$(pwd)
```

**Slow training:** The model trains on CPU in ~3–8 minutes for 50 epochs on 500 sequences.
Reduce `--epochs` in `model/configs/transformer_config.json` for faster demos.

**Missing config file:** All configs are included in the repository under
`simulation/configs/` and `model/configs/`.
