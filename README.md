# Agents for Agents: An Interrogator-Based Secure Framework for Autonomous IoUT

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: Under Review](https://img.shields.io/badge/Paper-Under%20Review-orange)]()
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb)

> **Paper under review.** Code and data are provided for reproducibility purposes.

---

## Overview

This repository contains the full implementation of the **interrogator-based behavioral trust framework** for autonomous Internet of Underwater Things (IoUT) deployments, as described in the paper:

> **Agents for Agents: An Interrogator-Based Secure Framework for Autonomous Internet of Underwater Things**
> Ali Akarma, Toqeer Ali Syed, Abdul Khadar Jilani, Salman Jan, Hammad Muneer, Muazzam A. Khan, Changli Yu
> *[Journal Name], 2026*

The framework combines:
- A **lightweight quantized transformer** for per-agent behavioral trust inference
- A **permissioned blockchain consortium** (PBFT, 9 validators) for tamper-resistant trust governance
- A **three-tier proportional enforcement** strategy for scalable anomaly containment
- A **multi-agent IoUT simulation** environment with realistic acoustic channel constraints

---

## Key Results

| Metric | Static Trust | Bayesian Trust | **Proposed** |
|---|---|---|---|
| Anomaly Detection Accuracy | 72.5% | 86.1% | **94.2%** |
| Packet Delivery Ratio | 79.4% | 86.7% | **91.6%** |
| Energy Overhead vs. Static | — | ~5.8% | **5.8%** |
| Early Detection Latency Reduction | — | — | **−27%** |

*Results averaged over 30 independent simulation runs, 50 agents, 15% adversarial fraction.*

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework
pip install -r requirements.txt

# 2. Generate synthetic data and run simulation
python scripts/run_full_pipeline.py

# 3. Reproduce all figures
python analysis/plot_all.py
```

---

## Repository Structure

```
IoUT-Interrogator-Framework/
├── data/                     # Synthetic datasets (generated, not collected)
│   ├── sample/               # Pre-generated sample sequences
│   ├── raw/                  # Raw simulation outputs
│   └── processed/            # Processed features for model input
├── simulation/               # IoUT environment simulator
│   ├── configs/              # Simulation parameter configs (JSON)
│   ├── scripts/              # Simulation runner scripts
│   └── outputs/              # Simulation result CSVs
├── model/                    # Transformer trust inference engine
│   ├── configs/              # Model hyperparameter configs
│   ├── inference/            # Training and inference scripts
│   └── checkpoints/          # Saved model weights
├── blockchain/               # PBFT consortium governance layer
│   ├── configs/              # Hyperledger Fabric config templates
│   ├── chaincode/            # Trust ledger smart contract skeleton
│   └── scripts/              # Ledger interaction scripts
├── analysis/                 # Result analysis and plotting
│   ├── plots/                # Generated figure outputs
│   └── stats/                # Statistical summary CSVs
├── notebooks/                # Google Colab-ready notebooks
├── scripts/                  # End-to-end pipeline scripts
├── tests/                    # Unit and integration tests
├── docs/                     # Extended documentation
└── .github/workflows/        # CI/CD pipeline (GitHub Actions)
```

---

## Colab Notebooks

| Notebook | Description | Open |
|---|---|---|
| `01_trust_inference_demo.ipynb` | Train and run the transformer trust model | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb) |
| `02_simulation_analysis.ipynb` | Analyse simulation outputs and compare baselines | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/02_simulation_analysis.ipynb) |
| `03_blockchain_demo.ipynb` | Mock PBFT blockchain trust ledger demo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/03_blockchain_demo.ipynb) |
| `04_ablation_study.ipynb` | Reproduce the ablation study figures | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/04_ablation_study.ipynb) |

---

## Reproducibility

All experiments use fixed random seeds for reproducibility. To fully reproduce all paper results:

```bash
python scripts/reproduce_all.py --seed 42 --runs 30
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for detailed step-by-step instructions, hardware requirements, and expected runtimes.

---

## Environment

- Python 3.9+
- PyTorch 2.0+ (CPU-only; no GPU required)
- See `requirements.txt` for full dependency list
- Docker image available: `docker build -t ioUt-framework .`

---

## Citation

If you use this code, please cite:

```bibtex
@article{akarma2026agents,
  author  = {Akarma, Ali and Syed, Toqeer Ali and Jilani, Abdul Khadar and
             Jan, Salman and Muneer, Hammad and Khan, Muazzam A. and Yu, Changli},
  title   = {Agents for Agents: An Interrogator-Based Secure Framework for
             Autonomous Internet of Underwater Things},
  journal = {[Journal Name]},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
