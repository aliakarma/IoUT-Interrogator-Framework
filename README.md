<div align="center">

# 🌊 IoUT Interrogator Framework

### Interrogator-Based Behavioral Trust Inference for the Internet of Underwater Things

*Transformer-Based Temporal Modeling for Autonomous Underwater Networks*

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge&logo=opensourceinitiative&logoColor=black)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-FF6B35?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![DOI](https://img.shields.io/badge/DOI-Pending-blue?style=for-the-badge&logo=doi&logoColor=white)](#)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](#-notebooks)

---

**[Overview](#-overview) · [Features](#-key-features) · [Results](#-results) · [Quick Start](#-quick-start) · [Notebooks](#-notebooks) · [Citation](#-citation)**

</div>

---

## 📋 Overview

This framework introduces a **continuous behavioral trust inference mechanism** for autonomous underwater networks, combining temporal sequence modeling with communication-level metadata analysis. It enables detection of adversarial or faulty agents **without accessing payload data**, preserving privacy while maintaining robust network security.

```
Agent Behavior ──► Metadata Collection ──► Transformer Model ──► Trust Scoring
                                                                        │
               Trust-Based Regulation ◄── Anomaly Detection ◄──────────┘
```

### Designed For

| Use Case | Description |
|----------|-------------|
| **Controlled Experimentation** | Isolate variables for rigorous behavioral analysis |
| **Reproducible Evaluation** | Fixed seeds, documented parameters, containerized execution |
| **Trust Mechanism Analysis** | Study inference under stochastic behavioral conditions |
| **Adversarial Testing** | Evaluate robustness against simulated attack scenarios |

---

## ⚠️ Scope

> **Important:** This framework operates at the **behavioral abstraction layer**. It is not a full network stack or physical-layer simulator.

<details>
<summary><strong>What this framework includes and excludes</strong></summary>

<br>

**✅ Included**

- Communication-metadata-level simulation abstraction
- Transformer-based temporal modeling for trust inference
- Dynamic trust updating with exponential smoothing
- Threshold-based gating for anomaly mitigation
- Statistical evaluation pipeline with 30 independent runs

**❌ Not Included**

- Physical-layer acoustic effects (propagation delay, multipath, Doppler)
- Full underwater network protocol or deployment simulation
- Production-ready blockchain integration (conceptual module only)
- Real-world hardware deployment or embedded system support

</details>

---

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Transformer-Based Trust Inference** | Temporal modeling of agent behavior via sequence learning | Captures long-range behavioral dependencies |
| **Metadata-Driven Monitoring** | Trust estimation using communication-level features only | Privacy-preserving and payload-agnostic |
| **Dynamic Trust Updating** | Continuous scoring with exponential smoothing | Adapts to evolving agent behavior in real time |
| **Trust-Based Agent Regulation** | Threshold-based gating for anomaly mitigation | Proactive defense against adversarial agents |
| **Reproducible Evaluation Pipeline** | End-to-end simulation, training, and analysis workflow | Ensures scientific rigor and repeatability |
| **Controlled Behavioral Simulation** | Synthetic AR(1)-based behavioral modeling | Enables systematic testing under controlled conditions |

---

## 📊 Results

### Experimental Configuration

```yaml
evaluation_setup:
  independent_runs:      30
  monitoring_intervals:  20
  total_agents:          50
  adversarial_fraction:  0.15
  random_seed:           42
```

### Performance Comparison

| Metric | Static Trust | Bayesian Trust | **Proposed (Ours)** |
|--------|:-----------:|:--------------:|:-------------------:|
| Detection Accuracy (%) | 86.00 ± 0.00 | 78.71 ± 8.19 | **91.87 ± 4.96** |
| Precision (%) | 0.00 ± 0.00 | 44.99 ± 15.62 | **72.39 ± 18.34** |
| Recall (%) | 0.00 ± 0.00 | 58.52 ± 20.26 | **76.69 ± 12.50** |
| F1 Score (%) | 0.00 ± 0.00 | 46.61 ± 13.15 | **70.32 ± 13.14** |
| Packet Delivery Ratio (%) | 91.59 ± 3.85 | 92.31 ± 3.71 | **92.39 ± 3.75** |
| Residual Energy (%) | **75.33 ± 0.05** | 73.95 ± 0.06 | 73.87 ± 0.05 |

<details>
<summary><strong>Key findings</strong></summary>

<br>

1. **Static trust fails entirely** at adversarial detection (F1 = 0.00), confirming that fixed thresholds cannot generalize.
2. **Temporal modeling significantly improves performance** across all detection metrics compared to both baselines.
3. **Communication reliability (PDR)** remains comparable across methods, confirming low interference from the trust layer.
4. **Energy overhead is modest** relative to baseline approaches (~1.5% additional consumption).
5. **The proposed method achieves the best trade-off** between detection accuracy and system efficiency.

**Source files:**
- `simulation/outputs/results.csv` — Raw per-run outputs
- `analysis/stats/summary_table.csv` — Aggregated statistics
- `analysis/plot_*.py` — Visualization scripts

</details>

---

## 🚀 Quick Start

### 1 · Clone and Install

```bash
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework
pip install -r requirements.txt
python -c "import torch; print('PyTorch', torch.__version__, '— Ready')"
```

### 2 · Run the Quick Demo (~5 minutes)

```bash
python scripts/run_full_pipeline.py --quick
```

Expected outputs:

- Sample trust inference results
- Evaluation plots → `analysis/plots/`
- Environment validation log

### 3 · Full Reproducible Evaluation

```bash
# Run 30 independent experiments
python scripts/reproduce_all.py --seed 42 --runs 30

# Run sensitivity analysis
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42

# View aggregated results
cat analysis/stats/summary_table.csv
```

<details>
<summary><strong>Advanced configuration options</strong></summary>

<br>

```bash
# Full evaluation with custom parameters
python scripts/reproduce_all.py \
  --seed 42 \
  --runs 30 \
  --agents 50 \
  --intervals 20 \
  --adversarial-fraction 0.15 \
  --output-dir ./custom_results

# Component ablation study
python analysis/sensitivity_study.py \
  --study-type ablation \
  --components transformer,smoothing,threshold \
  --seed 42
```

</details>

<details>
<summary><strong>Docker (recommended for full reproducibility)</strong></summary>

<br>

```bash
# Build image
docker build -t iout-interrogator:latest .

# Run with GPU support and result persistence
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/notebooks:/app/notebooks \
  iout-interrogator:latest

# Quick demo inside container
docker run --rm iout-interrogator:latest \
  python scripts/run_full_pipeline.py --quick
```

</details>

---

## 📓 Notebooks

All notebooks are fully compatible with **Google Colab** — no local setup required.

| # | Notebook | Description | Launch |
|---|----------|-------------|--------|
| 01 | `trust_inference_demo` | Model training, inference, and trust score visualization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb) |
| 02 | `simulation_analysis` | Performance metrics, baseline comparisons, statistical tests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/02_simulation_analysis.ipynb) |
| 03 | `blockchain_demo` | Conceptual trust governance and blockchain integration flow | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/03_blockchain_demo.ipynb) |
| 04 | `ablation_study` | Component sensitivity, parameter tuning, ablation analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/04_ablation_study.ipynb) |

<details>
<summary><strong>Local Jupyter setup</strong></summary>

<br>

```bash
pip install jupyter notebook ipywidgets
jupyter notebook notebooks/
# Access at: http://localhost:8888
```

</details>

---

## 🗂️ Repository Structure

<details open>
<summary><strong>Full directory tree</strong></summary>

<br>

```
IoUT-Interrogator-Framework/
│
├── simulation/                        # Behavioral Simulation Environment
│   ├── configs/
│   │   └── simulation_params.json     # Simulation hyperparameters
│   ├── scripts/
│   │   ├── environment.py             # IoUT environment implementation
│   │   ├── generate_behavioral_data.py  # Synthetic AR(1) data generation
│   │   └── run_simulation.py          # Simulation executor
│   └── outputs/
│       ├── results.csv                # Aggregated performance metrics
│       └── results_*.csv              # Per-run detailed results
│
├── model/                             # Transformer-Based Trust Model
│   ├── configs/
│   │   └── transformer_config.json    # Model architecture & hyperparameters
│   ├── checkpoints/
│   │   └── best_model.pt              # Trained model weights
│   └── inference/
│       ├── transformer_model.py       # Transformer architecture definition
│       ├── train.py                   # Model training pipeline
│       ├── infer.py                   # Inference and trust scoring
│       └── __init__.py
│
├── analysis/                          # Evaluation & Visualization
│   ├── plots/                         # Generated evaluation figures
│   ├── stats/
│   │   ├── summary_table.csv          # Aggregated results across runs
│   │   ├── threshold_sensitivity.csv  # Sensitivity analysis outputs
│   │   └── sequence_length_ablation.csv
│   ├── plot_*.py                      # Individual plot generation scripts
│   ├── sensitivity_study.py           # Threshold and ablation analysis
│   ├── statistical_summary.py         # Statistical computation
│   └── __init__.py
│
├── blockchain/                        # Conceptual Governance Module
│   ├── configs/
│   │   └── consortium_config.json     # Blockchain network configuration
│   ├── scripts/
│   │   └── mock_ledger.py             # Simulated blockchain ledger
│   ├── chaincode/
│   │   └── trust_ledger.go            # Smart contract template (Hyperledger)
│   └── __init__.py
│
├── data/                              # Datasets
│   ├── raw/
│   │   ├── behavioral_sequences.json  # Raw behavioral data
│   │   └── labels.csv                 # Ground truth agent labels
│   ├── processed/
│   │   ├── trust_scores.csv           # Processed trust inference outputs
│   │   └── trust_scores_only_legit.csv
│   └── sample/
│       └── sample_sequences.json      # Sample data for quick testing
│
├── scripts/                           # Execution Pipelines
│   ├── run_full_pipeline.py           # End-to-end: simulation → training → eval
│   ├── reproduce_all.py               # Reproducible evaluation with seed management
│   └── __init__.py
│
├── tests/                             # Unit & Integration Tests
│   ├── test_environment.py
│   ├── test_model.py
│   ├── test_blockchain.py
│   └── __init__.py
│
├── notebooks/                         # Interactive Jupyter Notebooks
│   ├── 01_trust_inference_demo.ipynb
│   ├── 02_simulation_analysis.ipynb
│   ├── 03_blockchain_demo.ipynb
│   └── 04_ablation_study.ipynb
│
├── docs/                              # Documentation
│   ├── REPRODUCIBILITY.md
│   ├── STRUCTURE.md
│   ├── CHANGELOG.md
│   └── CONTRIBUTING.md
│
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container configuration
├── CITATION.cff                       # Citation metadata
└── README.md
```

</details>

---

## 🖥️ Environment Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9+ | 3.10+ |
| **PyTorch** | 2.0 (CPU) | 2.0+ with CUDA 11.8+ |
| **RAM** | 4 GB | 8+ GB |
| **Storage** | 2 GB | 5+ GB (full evaluation) |
| **GPU** | Optional | NVIDIA GTX 1060+ |

### Python Dependencies

```bash
# Core scientific stack
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Deep learning
torch>=2.0.0
torchvision>=0.15.0

# Visualization & utilities
matplotlib>=3.4.0
tqdm>=4.60.0
jupyter>=1.0.0
ipywidgets>=7.7.0
```

> **Tip:** Use a virtual environment for isolation:
> ```bash
> python -m venv iout-env
> source iout-env/bin/activate    # Linux/macOS
> iout-env\Scripts\activate       # Windows
> pip install -r requirements.txt
> ```

### Troubleshooting

<details>
<summary><strong>Common issues and solutions</strong></summary>

<br>

| Issue | Solution |
|-------|----------|
| `CUDA not available` | Verify NVIDIA drivers and CUDA toolkit; use `--device cpu` flag |
| `MemoryError` | Reduce `--runs` or `--agents`; try `--batch-size 8` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt --force-reinstall` |
| `Docker permission denied` | Run `sudo usermod -aG docker $USER` and re-login |

</details>

---

## ♻️ Reproducibility

All code, simulation parameters, and evaluation procedures are provided for full transparency and scientific reproducibility.

### Reproduction Checklist

```
[ ] Python 3.9+ installed             →  python --version
[ ] Dependencies installed             →  pip install -r requirements.txt
[ ] Quick demo runs successfully       →  python scripts/run_full_pipeline.py --quick
[ ] Full evaluation executed           →  python scripts/reproduce_all.py --seed 42 --runs 30
[ ] Results match summary table        →  analysis/stats/summary_table.csv
[ ] Plots generated                    →  analysis/plots/
[ ] Docker build successful (optional) →  docker build -t iout-interrogator:latest .
```

### Version History

| Repository Version | Paper Version | Status |
|:-----------------:|:-------------:|:------:|
| `v1.0.0` (main) | Pre-print v1 | ✅ Stable |
| `v1.1.0` (dev) | Under Review | 🔄 Active Development |
| `v2.0.0` (future) | Published | 📅 Planned |

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Detailed reproduction guide, parameter specs, and result validation |
| [STRUCTURE.md](docs/STRUCTURE.md) | In-depth architecture documentation and module descriptions |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history, updates, and breaking changes |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Guidelines for contributions, bug reports, and feature requests |

---

## 📖 Citation

If you use this framework in your research, please cite:

**BibTeX**

```bibtex
@article{akarma2026interrogator,
  author    = {Akarma, Ali and Syed, Toqeer Ali and Jilani, Abdul Khadar and
               Jan, Salman and Muneer, Hammad and Khan, Muazzam A. and Yu, Changli},
  title     = {Interrogator-Based Behavioral Trust Inference for the Internet of
               Underwater Things Using Transformer-Based Temporal Modeling},
  journal   = {[Under Review]},
  year      = {2026},
  url       = {https://github.com/aliakarma/IoUT-Interrogator-Framework},
  note      = {Code and data available at the GitHub repository}
}
```

**APA**

> Akarma, A., Syed, T. A., Jilani, A. K., Jan, S., Muneer, H., Khan, M. A., & Yu, C. (2026). Interrogator-based behavioral trust inference for the internet of underwater things using transformer-based temporal modeling. *[Journal Name]*. (Under Review). https://github.com/aliakarma/IoUT-Interrogator-Framework

---

## 🤝 Contributing

Contributions from the research community are welcome.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

| Contribution Type | Guidelines |
|-------------------|-----------|
| **Bug Reports** | Include OS, Python version, full error traceback, and a minimal reproducible example |
| **Feature Requests** | Describe the use case, expected behavior, and potential implementation approach |
| **Documentation** | Follow existing style; update `docs/` and inline docstrings accordingly |
| **Code** | Add tests under `tests/`; verify all tests pass with `pytest tests/` |

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for full guidelines.

---

## 📬 Contact & Support

| Channel | Link |
|---------|------|
| **Bug Reports** | [Open an Issue](https://github.com/aliakarma/IoUT-Interrogator-Framework/issues/new) |
| **Discussions** | [GitHub Discussions](https://github.com/aliakarma/IoUT-Interrogator-Framework/discussions) |
| **Email** | 443059463@stu.iu.edu.sa |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

```
MIT License  ©  2026  Ali Akarma et al.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, subject to the above copyright notice and this
permission notice being included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

### Artifact Availability

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-4CAF50?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-Fixed%20Seeds-FF9800?style=flat-square&logo=checkmarx&logoColor=white)](#-reproducibility)
[![Containerized](https://img.shields.io/badge/Containerized-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](#-quick-start)
[![Cloud Ready](https://img.shields.io/badge/Cloud%20Ready-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](#-notebooks)
[![Code + Data](https://img.shields.io/badge/Code%20%2B%20Data-Included-2196F3?style=flat-square&logo=github&logoColor=white)](https://github.com/aliakarma/IoUT-Interrogator-Framework)

| Field | Details |
|-------|---------|
| **Paper Status** | Under Review |
| **Repository** | [IoUT-Interrogator-Framework](https://github.com/aliakarma/IoUT-Interrogator-Framework) |
| **License** | MIT |
| **Archival** | GitHub + Zenodo (DOI pending) |
| **Correspondence** | 443059463@stu.iu.edu.sa |

---

<div align="center">

**🌊 Advancing Trust in Autonomous Underwater Networks**

*Made with care by the IoUT Research Team · MIT Licensed · Reproducible by Design*

[⬆ Back to top](#-iout-interrogator-framework)

</div>
