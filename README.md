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
| Detection Accuracy (%) | 86.00 +- 0.00 | 72.03 +- 4.84 | **90.74 +- 1.58** |
| Precision (%) | 0.00 +- 0.00 | 26.75 +- 8.88 | **83.21 +- 10.05** |
| Recall (%) | 0.00 +- 0.00 | **46.60 +- 13.20** | 33.95 +- 11.25 |
| F1 Score (%) | 0.00 +- 0.00 | 32.45 +- 9.80 | **46.03 +- 11.60** |
| Packet Delivery Ratio (%) | **91.59 +- 0.88** | 82.73 +- 1.74 | 90.46 +- 1.24 |
| Energy Metric (%) | 73.08 +- 0.11 | 71.71 +- 0.11 | 71.38 +- 0.13 |

<details>
<summary><strong>Key findings</strong></summary>

<br>

1. **Static trust fails entirely** at adversarial detection (F1 = 0.00), confirming that fixed thresholds cannot generalize.
2. **The proposed transformer model** achieves the strongest overall detection profile (best accuracy, precision, and F1).
3. **Bayesian trust has the highest recall**, while the proposed model prioritizes fewer false positives.
4. **Network reliability remains high** for static and proposed approaches (PDR > 90%).
5. **All values are generated from 30 seeded runs** using the reproducibility pipeline.

**Source files:**
- `simulation/outputs/results.csv` — Aggregated interval-level metrics for the active run
- `simulation/outputs/raw_results.csv` — Long-format raw run-level metrics used for summary tables
- `analysis/stats/summary_table.csv` — Aggregated statistics
- `analysis/final_results/main_metrics.csv` — Publication-ready headline metrics
- `analysis/final_results/combined_results_table.csv` — Consolidated baseline/model comparison table
- `analysis/final_results/provenance_manifest.json` — SHA256 + artifact provenance metadata

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
- Statistics tables → `analysis/stats/`
- Exported result package → `analysis/final_results/`

### 3 · Full Reproducible Evaluation

```bash
# Run 30 independent experiments
python scripts/reproduce_all.py --seed 42 --runs 30

# Run sensitivity analysis
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42

# View aggregated results
python -c "import pandas as pd; print(pd.read_csv('analysis/stats/summary_table.csv').head(20).to_string(index=False))"
```

<details>
<summary><strong>Advanced configuration options</strong></summary>

<br>

```bash
# Full evaluation with custom parameters
python scripts/reproduce_all.py \
  --seed 42 \
  --runs 30 \
  --intervals 20 \
  --skip-training

# Threshold + sequence sensitivity study
python analysis/sensitivity_study.py \
  --runs 20 \
  --intervals 20 \
  --tau-values 0.55,0.60,0.65,0.70,0.75 \
  --sequence-lengths 16,32,64 \
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
| 05 | `enhancements_validation` | Validation of hardened settings and post-tuning enhancements | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/05_enhancements_validation.ipynb) |

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
<summary><strong>Repository map (aligned to current main branch)</strong></summary>

<br>

```
IoUT-Interrogator-Framework/
│
├── analysis/                          # Evaluation, statistics, and packaged outputs
│   ├── final_results/                 # Publication-ready consolidated result files
│   ├── plots/                         # Generated plots (PNG)
│   ├── smoke*/                        # Smoke and hardened experiment snapshots
│   ├── stats/                         # Statistical tables (CSV)
│   ├── plot_ablation.py
│   ├── plot_all.py
│   ├── plot_calibration.py
│   ├── plot_energy.py
│   ├── plot_ood_results.py
│   ├── plot_pdr.py
│   ├── plot_trust_accuracy.py
│   ├── sensitivity_study.py
│   └── statistical_summary.py
│
├── blockchain/                        # Conceptual governance module
│   ├── chaincode/trust_ledger.go
│   ├── configs/consortium_config.json
│   └── scripts/mock_ledger.py
│
├── data/                              # Raw, sample, and processed data
│   ├── raw/
│   ├── sample/sample_sequences.json
│   └── processed/trust_scores.csv
│
├── model/                             # Transformer model + baseline components
│   ├── checkpoints/
│   ├── configs/transformer_config.json
│   └── inference/
│       ├── baseline_models.py
│       ├── data_hardening.py
│       ├── infer.py
│       ├── lstm_baseline.py
│       ├── train.py
│       └── transformer_model.py
│
├── simulation/                        # Behavioral simulation environment
│   ├── configs/simulation_params.json
│   ├── scripts/
│   │   ├── environment.py
│   │   ├── generate_behavioral_data.py
│   │   └── run_simulation.py
│   └── outputs/
│       ├── results.csv
│       ├── raw_results.csv
│       └── multi_seed_raw_results.csv
│
├── scripts/                           # Main CLI entry points
│   ├── run_full_pipeline.py
│   ├── reproduce_all.py
│   ├── run_multi_seed_experiments.py
│   ├── run_ablation_study.py
│   ├── run_ood_evaluation.py
│   ├── smoke_validate_classifier.py
│   ├── profile_inference.py
│   └── export_all_results.py
│
├── tests/                             # Unit and integration tests
│   ├── test_environment.py
│   ├── test_model.py
│   └── test_blockchain.py
│
├── notebooks/
│   ├── 01_trust_inference_demo.ipynb
│   ├── 02_simulation_analysis.ipynb
│   ├── 03_blockchain_demo.ipynb
│   ├── 04_ablation_study.ipynb
│   └── 05_enhancements_validation.ipynb
│
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── STRUCTURE.md
│   ├── CHANGELOG.md
│   └── THRESHOLD_SWEEP_AND_FOCAL_LOSS.md
│
├── compat.py
├── requirements.txt
├── Dockerfile
├── CITATION.cff
└── README.md
```

</details>

---

## 🖥️ Environment Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10-3.11 |
| **PyTorch** | 2.0 (CPU) | 2.0+ with CUDA 11.8+ |
| **RAM** | 4 GB | 8+ GB |
| **Storage** | 2 GB | 5+ GB (full evaluation) |
| **GPU** | Optional | NVIDIA GTX 1060+ |

### Python Dependencies

```bash
# Install the tested, pinned dependency set
pip install -r requirements.txt
```

> **Version policy:** this project supports Python **3.9-3.11** only.
> Entry-point scripts include a runtime guard that fails fast on unsupported
> interpreters to avoid dependency resolution issues.
> The repository ships code and sample inputs; generated checkpoints and
> publication tables are recreated by the pipeline and tracked via
> `analysis/final_results/provenance_manifest.json`.

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
| `MemoryError` | Reduce `--runs` or `--intervals`; or run `python scripts/run_full_pipeline.py --quick` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt --force-reinstall` |
| `Docker permission denied` | Run `sudo usermod -aG docker $USER` and re-login |

</details>

---

## ♻️ Reproducibility

All code, simulation parameters, and evaluation procedures are provided for full transparency and scientific reproducibility.

### Reproduction Checklist

```
[ ] Python 3.9-3.11 installed         →  python --version
[ ] Dependencies installed             →  pip install -r requirements.txt
[ ] Quick demo runs successfully       →  python scripts/run_full_pipeline.py --quick
[ ] Full evaluation executed           →  python scripts/reproduce_all.py --seed 42 --runs 30
[ ] Results match summary table        →  analysis/stats/summary_table.csv
[ ] Plots generated                    →  analysis/plots/
[ ] Docker build successful (optional) →  docker build -t iout-interrogator:latest .
```

### Version History

| Branch | Purpose | Status |
|:------:|---------|:------:|
| `main` | Active development and reproducibility artifacts | ✅ Current |

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Detailed reproduction guide, parameter specs, and result validation |
| [STRUCTURE.md](docs/STRUCTURE.md) | In-depth architecture documentation and module descriptions |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history, updates, and breaking changes |
| [THRESHOLD_SWEEP_AND_FOCAL_LOSS.md](docs/THRESHOLD_SWEEP_AND_FOCAL_LOSS.md) | Threshold-sweep experiments and focal-loss analysis notes |

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

Use GitHub issues and pull requests for contribution discussion and review.

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
