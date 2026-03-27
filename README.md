# Interrogator-Based Behavioral Trust Inference for IoUT

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)]()
[![Colab Notebooks](https://colab.research.google.com/assets/colab-badge.svg)](#interactive-notebooks)
[![DOI](https://img.shields.io/badge/DOI-Dataset-blue)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

---

## � Table of Contents

- [🔬 Overview](#overview)
- [⚠️ Scope and Research Positioning](#scope-and-research-positioning)
- [✨ Key Features](#key-features)
- [📈 Reproducible Results](#reproducible-results)
- [🚀 Getting Started in 3 Steps](#getting-started-in-3-steps)
- [📚 Interactive Notebooks](#interactive-notebooks)
- [🗂️ Repository Structure](#repository-structure)
- [🖥️ Environment Setup](#environment-setup)
- [📋 Supplementary Materials](#supplementary-materials)
- [📚 Artifact Availability](#artifact-availability)
- [📖 Citation](#citation)
- [📄 License](#license)
- [🤝 Reproducibility Commitment](#reproducibility-commitment)

---

## 🔬 Overview {#overview}

This repository provides a **research-grade, reproducible implementation** of an interrogator-based behavioral trust framework for autonomous Internet of Underwater Things (IoUT) environments.

The framework introduces a **continuous behavioral trust inference mechanism** based on temporal sequence modeling, enabling detection of adversarial or faulty agents using communication-level metadata without accessing payload data.

The implementation is designed for:

* controlled experimentation,
* reproducible evaluation, and
* analysis of trust inference mechanisms under stochastic behavioral conditions.

---

## ⚠️ Scope and Research Positioning {#scope-and-research-positioning}

This repository corresponds to a **behavioral-level simulation framework**, not a full network or acoustic simulation system.

**Important clarifications:**

* The simulation operates at the **communication-metadata abstraction level**.
* It does **not model physical-layer acoustic effects** (e.g., propagation delay, multipath, Doppler).
* The blockchain governance layer is included as a **conceptual module only** and is **not part of the evaluated results**.
* Results should be interpreted as validating **behavioral trust inference**, not full IoUT deployment performance.

For full details, see:

* [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

---

## ✨ Key Features {#key-features}

* 🤖 **Transformer-Based Trust Inference**
  Temporal modeling of agent behavior using sequence-based learning.

* 📡 **Metadata-Driven Monitoring**
  Trust estimation based solely on communication-level features.

* 🔁 **Dynamic Trust Updating**
  Continuous trust scoring with exponential smoothing.

* 🚫 **Trust-Based Agent Regulation**
  Threshold-based gating mechanism for anomaly mitigation.

* 📊 **Reproducible Evaluation Pipeline**
  End-to-end pipeline for simulation, training, and analysis.

* 🧪 **Controlled Behavioral Simulation**
  Synthetic AR(1)-based behavioral modeling with adversarial scenarios.

---

## 📈 Reproducible Results {#reproducible-results}

Evaluation is conducted over:

* 30 independent runs
* 20 monitoring intervals per run
* 50 agents
* 15% adversarial fraction

Source files:

* `simulation/outputs/results.csv`
* `analysis/stats/summary_table.csv`

| Metric                    |           Static |      Bayesian |          Proposed |
| ------------------------- | ---------------: | ------------: | ----------------: |
| Detection Accuracy (%)    |     86.00 ± 0.00 |  78.71 ± 8.19 |  **91.87 ± 4.96** |
| Precision (%)             |      0.00 ± 0.00 | 44.99 ± 15.62 | **72.39 ± 18.34** |
| Recall (%)                |      0.00 ± 0.00 | 58.52 ± 20.26 | **76.69 ± 12.50** |
| F1 Score (%)              |      0.00 ± 0.00 | 46.61 ± 13.15 | **70.32 ± 13.14** |
| Packet Delivery Ratio (%) |     91.59 ± 3.85 |  92.31 ± 3.71 |  **92.39 ± 3.75** |
| Residual Energy (%)       | **75.33 ± 0.05** |  73.95 ± 0.06 |      73.87 ± 0.05 |

**Key observations:**

* Static trust fails completely in adversarial detection (F1 = 0).
* Temporal modeling significantly improves detection performance.
* Communication reliability (PDR) remains statistically comparable across methods.
* Energy overhead is modest relative to baseline approaches.

---

## 🚀 Getting Started in 3 Steps {#getting-started-in-3-steps}

### Step 1: Clone & Install

```bash
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework
pip install -r requirements.txt
```

### Step 2: Quick Demo (5 minutes)

```bash
# Run a minimal simulation with sample data
python scripts/run_full_pipeline.py --quick
```

This generates:
- Sample trust inference results
- Basic evaluation plots
- Verification of environment setup

### Step 3: Full Reproducible Evaluation

```bash
# Reproduce all main results (30 runs, ~2-4 hours depending on hardware)
python scripts/reproduce_all.py --seed 42 --runs 30

# Generate sensitivity analysis
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42
```

---

## 📚 Interactive Notebooks {#interactive-notebooks}

All notebooks are fully compatible with Google Colab for cloud-based execution:

| Notebook | Description | Colab Link |
| -------- | ----------- | ---------- |
| 01_trust_inference_demo.ipynb | Model training, inference, and trust scoring visualization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb) |
| 02_simulation_analysis.ipynb | Performance metrics, comparisons with baselines, statistical analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/02_simulation_analysis.ipynb) |
| 03_blockchain_demo.ipynb | Conceptual trust governance and blockchain integration flow | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/03_blockchain_demo.ipynb) |
| 04_ablation_study.ipynb | Component sensitivity, parameter tuning, and ablation analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/04_ablation_study.ipynb) |

---

## 🗂️ Repository Structure {#repository-structure}

```text
IoUT-Interrogator-Framework/
│
├── 📂 simulation/                    # Behavioral Simulation Environment
│   ├── configs/
│   │   └── simulation_params.json       # Simulation hyperparameters (agents, runs, intervals)
│   ├── scripts/
│   │   ├── environment.py               # IoUT environment implementation
│   │   ├── generate_behavioral_data.py  # Synthetic AR(1) behavioral sequence generation
│   │   └── run_simulation.py            # Simulation executor
│   └── outputs/
│       ├── results.csv                  # Aggregated performance metrics
│       └── results_*.csv                # Per-run detailed results
│
├── 📂 model/                         # Transformer-Based Trust Model
│   ├── configs/
│   │   └── transformer_config.json      # Model architecture and hyperparameters
│   ├── checkpoints/
│   │   └── best_model.pt                # Trained transformer model weights
│   └── inference/
│       ├── transformer_model.py         # Transformer architecture definition
│       ├── train.py                     # Model training pipeline
│       ├── infer.py                     # Inference and trust scoring
│       └── __init__.py
│
├── 📂 analysis/                      # Evaluation & Visualization
│   ├── plots/                           # Generated evaluation figures
│   ├── stats/
│   │   ├── summary_table.csv            # Aggregated results across runs
│   │   ├── threshold_sensitivity.csv    # Sensitivity analysis outputs
│   │   └── sequence_length_ablation.csv # Ablation study results
│   ├── plot_*.py                        # Individual plot generation scripts
│   ├── sensitivity_study.py             # Threshold and ablation analysis
│   ├── statistical_summary.py           # Statistical computation
│   └── __init__.py
│
├── 📂 blockchain/                    # Conceptual Governance Module
│   ├── configs/
│   │   └── consortium_config.json       # Blockchain network configuration
│   ├── scripts/
│   │   └── mock_ledger.py               # Simulated blockchain ledger
│   ├── chaincode/
│   │   └── trust_ledger.go              # Smart contract template (Hyperledger Fabric)
│   └── __init__.py
│
├── 📂 data/                          # Datasets
│   ├── raw/
│   │   ├── behavioral_sequences.json    # Raw behavioral data
│   │   └── labels.csv                   # Ground truth agent labels
│   ├── processed/
│   │   ├── trust_scores.csv             # Processed trust inference outputs
│   │   └── trust_scores_only_legit.csv  # Filtered trust scores (legitimate agents)
│   └── sample/
│       └── sample_sequences.json        # Sample data for quick testing
│
├── 📂 scripts/                       # Execution Pipelines
│   ├── run_full_pipeline.py             # End-to-end: simulation → training → evaluation
│   ├── reproduce_all.py                 # Reproducible evaluation with seed management
│   └── __init__.py
│
├── 📂 tests/                         # Unit & Integration Tests
│   ├── test_environment.py              # Simulation environment tests
│   ├── test_model.py                    # Model inference tests
│   ├── test_blockchain.py               # Blockchain module tests
│   └── __init__.py
│
├── 📂 notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01_trust_inference_demo.ipynb    # Model demo & training walkthrough
│   ├── 02_simulation_analysis.ipynb     # Results analysis & comparison
│   ├── 03_blockchain_demo.ipynb         # Governance flow demonstration
│   └── 04_ablation_study.ipynb          # Parameter sensitivity & ablation
│
├── 📂 docs/                          # Documentation
│   ├── REPRODUCIBILITY.md               # Full reproducibility guidelines
│   ├── STRUCTURE.md                     # Detailed architecture documentation
│   ├── CHANGELOG.md                     # Version history & updates
│   └── README.md                        # This file (main documentation)
│
├── 📄 requirements.txt                # Python dependencies (PyTorch, NumPy, Pandas, etc.)
├── 📄 Dockerfile                      # Docker containerization
├── 📄 LICENSE                         # MIT License
├── 📄 CITATION.cff                    # Citation metadata (BibTeX & APA formats)
└── 📄 README.md                       # Project overview & quick start

```

### 📋 Key Files

| File | Purpose |
|------|----------|
| `requirements.txt` | All Python package dependencies |
| `Dockerfile` | Containerized environment for reproducible execution |
| `CITATION.cff` | Automated citation generation (GitHub & Zenodo) |
| `docs/REPRODUCIBILITY.md` | Complete reproduction instructions and parameter specifications |
| `simulation/outputs/results.csv` | Main results file with all metrics |
| `analysis/stats/summary_table.csv` | Aggregated statistics across all runs |

---

## 🖥️ Environment Setup {#environment-setup}

### System Requirements

* **Python**: 3.9 or higher
* **PyTorch**: 2.x (CPU/GPU compatible)
* **Memory**: ≥4 GB RAM (8+ GB recommended for full evaluation)
* **Storage**: ≥2 GB for models, data, and results

### Dependencies

```bash
numpy>=1.21.0
pandas>=1.3.0
pytorch>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.60.0
```

See [requirements.txt](requirements.txt) for full specification.

### Docker Setup (Recommended)

```bash
# Build Docker image
docker build -t iout-interrogator:latest .

# Run container with GPU support (if available)
docker run --gpus all -it iout-interrogator:latest

# Run with volume mounts for results persistence
docker run --gpus all -v $(pwd)/results:/app/results iout-interrogator:latest
```

---

## 📋 Supplementary Materials {#supplementary-materials}

Additional resources are available in the `docs/` directory:

| Document | Content |
|----------|---------|
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Detailed reproduction guide, parameter specifications, and result validation |
| [STRUCTURE.md](docs/STRUCTURE.md) | In-depth architecture documentation and module descriptions |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history, updates, and breaking changes |

---

## 📚 Artifact Availability {#artifact-availability}

This repository is designed for **full reproducibility and transparency**:

✅ **Fully Open Source** — MIT License  
✅ **Code + Data** — All research artifacts included  
✅ **Reproducible** — Fixed seeds, documented parameters  
✅ **Containerized** — Docker support for consistent environments  
✅ **Cloud-Friendly** — All notebooks run in Google Colab  
✅ **Well-Documented** — Comprehensive docs and inline comments  

### Paper Availability

- **Paper Status**: Under review
- **Repository**: [IoUT-Interrogator-Framework](https://github.com/aliakarma/IoUT-Interrogator-Framework)
- **License**: MIT (Open Source)
- **Permanence**: GitHub + Zenodo archival

---

## 📖 Citation {#citation}

### BibTeX Format

```bibtex
@article{akarma2026interrogator,
  author  = {Akarma, Ali and Syed, Toqeer Ali and Jilani, Abdul Khadar and
             Jan, Salman and Muneer, Hammad and Khan, Muazzam A. and Yu, Changli},
  title   = {Interrogator-Based Behavioral Trust Inference for the Internet of Underwater Things Using Transformer-Based Temporal Modeling},
  journal = {[Under Review]},
  year    = {2026}
}
```

### APA Format

Akarma, A., Syed, T. A., Jilani, A. K., Jan, S., Muneer, H., Khan, M. A., & Yu, C. (2026). Interrogator-based behavioral trust inference for the internet of underwater things using transformer-based temporal modeling. *[Journal Name]*, [Volume(Issue)], pages. (Under Review)

---

## 📄 License {#license}

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Reproducibility Commitment {#reproducibility-commitment}

All code, simulation parameters, and evaluation procedures are provided to ensure **full transparency and reproducibility**.

✨ **Artifacts will be maintained and updated** in alignment with the final published version of the paper.

### Reproduction Checklist

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Run `python scripts/reproduce_all.py --seed 42 --runs 30`
- [ ] Verify results match [analysis/stats/summary_table.csv](analysis/stats/summary_table.csv)
- [ ] Review plots in [analysis/plots/](analysis/plots/)
