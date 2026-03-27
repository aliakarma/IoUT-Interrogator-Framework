# 🌊 Interrogator-Based Behavioral Trust Inference for IoUT

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Paper-Under%20Review-orange?style=for-the-badge" alt="Paper Status">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  <img src="https://img.shields.io/badge/DOI-Dataset-blue?style=for-the-badge" alt="DOI">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white&style=for-the-badge" alt="Docker">
</p>

<p align="center">
  <strong>🔐 Advanced Trust Inference Framework for Autonomous Underwater Networks</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-getting-started">Quick Start</a> •
  <a href="#-interactive-notebooks">Notebooks</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📋 Interactive Table of Contents

<details open>
<summary><strong>🔍 Click to expand/collapse sections</strong></summary>

- [🔬 Overview](#-overview)
- [⚠️ Scope and Research Positioning](#️-scope-and-research-positioning)
- [✨ Key Features](#-key-features)
- [📈 Reproducible Results](#-reproducible-results)
- [🚀 Getting Started in 3 Steps](#-getting-started-in-3-steps)
- [📚 Interactive Notebooks](#-interactive-notebooks)
- [🗂️ Repository Structure](#️-repository-structure)
- [🖥️ Environment Setup](#️-environment-setup)
- [📋 Supplementary Materials](#-supplementary-materials)
- [📚 Artifact Availability](#-artifact-availability)
- [📖 Citation](#-citation)
- [📄 License](#-license)
- [🤝 Reproducibility Commitment](#-reproducibility-commitment)
- [🤝 Contributing](#-contributing)
- [📬 Support & Contact](#-support--contact)

</details>

---

## 🔬 Overview

<p align="center">
  <img src="https://img.shields.io/badge/Research-Grade-00A8E8?style=flat-square" alt="Research Grade">
  <img src="https://img.shields.io/badge/Reproducible-✅-4CAF50?style=flat-square" alt="Reproducible">
  <img src="https://img.shields.io/badge/Open-Source-FF6B6B?style=flat-square" alt="Open Source">
</p>

This repository provides a **research-grade, reproducible implementation** of an interrogator-based behavioral trust framework for autonomous **Internet of Underwater Things (IoUT)** environments.

graph LR
    A[Agent Behavior] --> B[Metadata Collection]
    B --> C[Transformer Model]
    C --> D[Trust Inference]
    D --> E[Anomaly Detection]
    E --> F[Trust-Based Regulation]

### 🎯 Core Innovation

> The framework introduces a **continuous behavioral trust inference mechanism** based on temporal sequence modeling, enabling detection of adversarial or faulty agents using **communication-level metadata** without accessing payload data.

### 🛠️ Designed For

| Purpose | Description |
|---------|-------------|
| 🔬 **Controlled Experimentation** | Isolate variables for rigorous behavioral analysis |
| ♻️ **Reproducible Evaluation** | Fixed seeds, documented parameters, containerized execution |
| 📊 **Trust Mechanism Analysis** | Study inference under stochastic behavioral conditions |
| 🧪 **Adversarial Testing** | Evaluate robustness against simulated attacks |

---

## ⚠️ Scope and Research Positioning

> ⚡ **Important**: This is a **behavioral-level simulation framework**, not a full network or acoustic simulation system.

<details>
<summary><strong>📌 Click to view scope clarifications</strong></summary>

### ✅ What This Framework Includes

- 📡 **Communication-metadata abstraction level** simulation
- 🤖 **Transformer-based temporal modeling** for trust inference
- 🔄 **Dynamic trust updating** with exponential smoothing
- 🚫 **Threshold-based gating** for anomaly mitigation
- 📈 **Statistical evaluation pipeline** with 30 independent runs

### ❌ What This Framework Does NOT Include

- 🌊 Physical-layer acoustic effects (propagation delay, multipath, Doppler)
- 🛰️ Full underwater network deployment simulation
- 🔗 Production-ready blockchain integration (conceptual module only)
- 📦 Real-world hardware deployment scripts

### 📚 For Full Details

See comprehensive documentation:
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) - Reproduction guidelines
- [`docs/STRUCTURE.md`](docs/STRUCTURE.md) - Architecture deep-dive

</details>

---

## ✨ Key Features

<p align="center">
  <img src="https://img.shields.io/badge/🤖-Transformer--Based-9C27B0?style=flat-square" alt="Transformer">
  <img src="https://img.shields.io/badge/📡-Metadata--Driven-2196F3?style=flat-square" alt="Metadata">
  <img src="https://img.shields.io/badge/🔁-Dynamic--Updating-FF9800?style=flat-square" alt="Dynamic">
  <img src="https://img.shields.io/badge/🚫-Anomaly--Mitigation-F44336?style=flat-square" alt="Mitigation">
  <img src="https://img.shields.io/badge/📊-Reproducible--Pipeline-4CAF50?style=flat-square" alt="Pipeline">
  <img src="https://img.shields.io/badge/🧪-Controlled--Simulation-607D8B?style=flat-square" alt="Simulation">
</p>

| Feature | Description | Benefit |
|---------|-------------|---------|
| 🤖 **Transformer-Based Trust Inference** | Temporal modeling of agent behavior using sequence-based learning | Captures long-range behavioral dependencies |
| 📡 **Metadata-Driven Monitoring** | Trust estimation based solely on communication-level features | Privacy-preserving, payload-agnostic analysis |
| 🔁 **Dynamic Trust Updating** | Continuous trust scoring with exponential smoothing | Adapts to evolving agent behavior in real-time |
| 🚫 **Trust-Based Agent Regulation** | Threshold-based gating mechanism for anomaly mitigation | Proactive defense against adversarial agents |
| 📊 **Reproducible Evaluation Pipeline** | End-to-end pipeline for simulation, training, and analysis | Ensures scientific rigor and result validation |
| 🧪 **Controlled Behavioral Simulation** | Synthetic AR(1)-based behavioral modeling with adversarial scenarios | Enables systematic testing under known conditions |

---

## 📈 Reproducible Results

### 🧪 Experimental Configuration

```yaml
evaluation_setup:
  independent_runs: 30
  monitoring_intervals: 20
  total_agents: 50
  adversarial_fraction: 0.15
  random_seed: 42
```

### 📊 Results Summary

| Metric | Static Trust | Bayesian Trust | **Proposed (Ours)** |
|--------|-------------|----------------|---------------------|
| 🔍 Detection Accuracy (%) | 86.00 ± 0.00 | 78.71 ± 8.19 | **✨ 91.87 ± 4.96** |
| 🎯 Precision (%) | 0.00 ± 0.00 | 44.99 ± 15.62 | **✨ 72.39 ± 18.34** |
| 📥 Recall (%) | 0.00 ± 0.00 | 58.52 ± 20.26 | **✨ 76.69 ± 12.50** |
| ⚖️ F1 Score (%) | 0.00 ± 0.00 | 46.61 ± 13.15 | **✨ 70.32 ± 13.14** |
| 📦 Packet Delivery Ratio (%) | 91.59 ± 3.85 | 92.31 ± 3.71 | **✨ 92.39 ± 3.75** |
| 🔋 Residual Energy (%) | **75.33 ± 0.05** | 73.95 ± 0.06 | 73.87 ± 0.05 |

<details>
<summary><strong>🔍 Click to view key observations</strong></summary>

### 🎯 Key Findings

1. **🚫 Static trust fails completely** in adversarial detection (F1 = 0.00)
2. **📈 Temporal modeling significantly improves** detection performance across all metrics
3. **📡 Communication reliability (PDR)** remains statistically comparable across methods
4. **🔋 Energy overhead is modest** relative to baseline approaches (~1.5% difference)
5. **✨ Proposed method achieves best trade-off** between detection accuracy and system efficiency

### 📁 Source Files

- Raw results: `simulation/outputs/results.csv`
- Aggregated stats: `analysis/stats/summary_table.csv`
- Visualization scripts: `analysis/plot_*.py`

</details>

---

## 🚀 Getting Started in 3 Steps

<p align="center">
  <img src="https://img.shields.io/badge/Step-1/3-2196F3?style=flat-square" alt="Step 1">
  <img src="https://img.shields.io/badge/Step-2/3-4CAF50?style=flat-square" alt="Step 2">
  <img src="https://img.shields.io/badge/Step-3/3-FF9800?style=flat-square" alt="Step 3">
</p>

### Step 1️⃣: Clone & Install

```bash
# 📥 Clone the repository
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework

# 📦 Install dependencies
pip install -r requirements.txt

# ✅ Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
```

### Step 2️⃣: Quick Demo (⏱️ ~5 minutes)

```bash
# 🎬 Run a minimal simulation with sample data
python scripts/run_full_pipeline.py --quick

# 📊 Expected outputs:
# ✅ Sample trust inference results
# ✅ Basic evaluation plots in analysis/plots/
# ✅ Environment setup verification
```

<details>
<summary><strong>💡 Pro Tip: Use Docker for guaranteed reproducibility</strong></summary>

```bash
# 🐳 Build and run with Docker
docker build -t iout-interrogator:latest .
docker run --gpus all -v $(pwd)/results:/app/results iout-interrogator:latest

# 🎯 Run quick demo inside container
python scripts/run_full_pipeline.py --quick
```

</details>

### Step 3️⃣: Full Reproducible Evaluation

```bash
# 🔬 Reproduce all main results (30 runs, ~2-4 hours)
python scripts/reproduce_all.py --seed 42 --runs 30

# 📈 Generate sensitivity analysis
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42

# 📊 View aggregated results
cat analysis/stats/summary_table.csv
```

<details>
<summary><strong>⚙️ Advanced Configuration Options</strong></summary>

```bash
# Customize simulation parameters
python scripts/reproduce_all.py \
  --seed 42 \
  --runs 30 \
  --agents 50 \
  --intervals 20 \
  --adversarial-fraction 0.15 \
  --output-dir ./custom_results

# Run ablation studies
python analysis/sensitivity_study.py \
  --study-type ablation \
  --components transformer,smoothing,threshold \
  --seed 42
```

</details>

---

## 📚 Interactive Notebooks

<p align="center">
  <a href="https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb">
    <img src="https://img.shields.io/badge/🔗-Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Colab">
  </a>
</p>

All notebooks are **fully compatible with Google Colab** for cloud-based execution—no local setup required! 🌥️

| Notebook | Description | Features | Colab |
|----------|-------------|----------|-------|
| 🔹 `01_trust_inference_demo.ipynb` | Model training, inference, and trust scoring visualization | 🎯 Interactive plots, 🔄 Live training metrics, 📊 Trust evolution charts | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb) |
| 🔹 `02_simulation_analysis.ipynb` | Performance metrics, comparisons with baselines, statistical analysis | 📈 Comparative bar charts, 📉 Confidence intervals, 🧪 Statistical tests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/02_simulation_analysis.ipynb) |
| 🔹 `03_blockchain_demo.ipynb` | Conceptual trust governance and blockchain integration flow | 🔗 Smart contract walkthrough, 🗳️ Consensus visualization, 📜 Ledger exploration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/03_blockchain_demo.ipynb) |
| 🔹 `04_ablation_study.ipynb` | Component sensitivity, parameter tuning, and ablation analysis | 🎛️ Interactive sliders, 📊 Heatmaps, 🔍 Parameter sweeps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/04_ablation_study.ipynb) |

<details>
<summary><strong>💻 Local Jupyter Setup</strong></summary>

```bash
# Install Jupyter dependencies
pip install jupyter notebook ipywidgets

# Launch Jupyter server
jupyter notebook notebooks/

# 🔗 Access at: http://localhost:8888
```

</details>

---

## 🗂️ Repository Structure

<details open>
<summary><strong>📁 Click to expand full directory tree</strong></summary>

```text
IoUT-Interrogator-Framework/
│
├── 📂 simulation/                    # 🎭 Behavioral Simulation Environment
│   ├── configs/
│   │   └── simulation_params.json       # ⚙️ Simulation hyperparameters
│   ├── scripts/
│   │   ├── environment.py               # 🌊 IoUT environment implementation
│   │   ├── generate_behavioral_data.py  # 📊 Synthetic AR(1) data generation
│   │   └── run_simulation.py            # 🎬 Simulation executor
│   └── outputs/
│       ├── results.csv                  # 📈 Aggregated performance metrics
│       └── results_*.csv                # 📄 Per-run detailed results
│
├── 📂 model/                         # 🤖 Transformer-Based Trust Model
│   ├── configs/
│   │   └── transformer_config.json      # 🧠 Model architecture & hyperparameters
│   ├── checkpoints/
│   │   └── best_model.pt                # 💾 Trained model weights
│   └── inference/
│       ├── transformer_model.py         # 🏗️ Transformer architecture definition
│       ├── train.py                     # 🎓 Model training pipeline
│       ├── infer.py                     # 🔮 Inference and trust scoring
│       └── __init__.py
│
├── 📂 analysis/                      # 📊 Evaluation & Visualization
│   ├── plots/                           # 🖼️ Generated evaluation figures
│   ├── stats/
│   │   ├── summary_table.csv            # 📋 Aggregated results across runs
│   │   ├── threshold_sensitivity.csv    # 🎚️ Sensitivity analysis outputs
│   │   └── sequence_length_ablation.csv # 🔪 Ablation study results
│   ├── plot_*.py                        # 🎨 Individual plot generation scripts
│   ├── sensitivity_study.py             # 🔬 Threshold and ablation analysis
│   ├── statistical_summary.py           # 📐 Statistical computation
│   └── __init__.py
│
├── 📂 blockchain/                    # 🔗 Conceptual Governance Module
│   ├── configs/
│   │   └── consortium_config.json       # ⛓️ Blockchain network configuration
│   ├── scripts/
│   │   └── mock_ledger.py               # 📝 Simulated blockchain ledger
│   ├── chaincode/
│   │   └── trust_ledger.go              # 💼 Smart contract template (Hyperledger)
│   └── __init__.py
│
├── 📂 data/                          # 🗄️ Datasets
│   ├── raw/
│   │   ├── behavioral_sequences.json    # 📥 Raw behavioral data
│   │   └── labels.csv                   # 🏷️ Ground truth agent labels
│   ├── processed/
│   │   ├── trust_scores.csv             # ✨ Processed trust inference outputs
│   │   └── trust_scores_only_legit.csv  # ✅ Filtered trust scores (legitimate)
│   └── sample/
│       └── sample_sequences.json        # 🎁 Sample data for quick testing
│
├── 📂 scripts/                       # 🎬 Execution Pipelines
│   ├── run_full_pipeline.py             # 🔄 End-to-end: simulation → training → eval
│   ├── reproduce_all.py                 # ♻️ Reproducible evaluation with seed mgmt
│   └── __init__.py
│
├── 📂 tests/                         # 🧪 Unit & Integration Tests
│   ├── test_environment.py              # 🔍 Simulation environment tests
│   ├── test_model.py                    # 🤖 Model inference tests
│   ├── test_blockchain.py               # 🔗 Blockchain module tests
│   └── __init__.py
│
├── 📂 notebooks/                     # 📓 Interactive Jupyter Notebooks
│   ├── 01_trust_inference_demo.ipynb    # 🎓 Model demo & training walkthrough
│   ├── 02_simulation_analysis.ipynb     # 📈 Results analysis & comparison
│   ├── 03_blockchain_demo.ipynb         # 🔗 Governance flow demonstration
│   └── 04_ablation_study.ipynb          # 🔪 Parameter sensitivity & ablation
│
├── 📂 docs/                          # 📚 Documentation
│   ├── REPRODUCIBILITY.md               # ♻️ Full reproducibility guidelines
│   ├── STRUCTURE.md                     # 🏗️ Detailed architecture documentation
│   ├── CHANGELOG.md                     # 📝 Version history & updates
│   └── README.md                        # 📄 This file (main documentation)
│
├── 📄 requirements.txt                # 📦 Python dependencies
├── 📄 Dockerfile                      # 🐳 Containerization config
├── 📄 LICENSE                         # ⚖️ MIT License
├── 📄 CITATION.cff                    # 📖 Citation metadata (BibTeX & APA)
└── 📄 README.md                       # 🏠 Project overview & quick start
```

</details>

### 📋 Key Files Quick Reference

| File | Purpose | Category |
|------|---------|----------|
| `requirements.txt` | 📦 All Python package dependencies | Setup |
| `Dockerfile` | 🐳 Containerized environment for reproducible execution | Deployment |
| `CITATION.cff` | 📖 Automated citation generation (GitHub & Zenodo) | Metadata |
| `docs/REPRODUCIBILITY.md` | ♻️ Complete reproduction instructions and parameter specs | Documentation |
| `simulation/outputs/results.csv` | 📊 Main results file with all metrics | Results |
| `analysis/stats/summary_table.csv` | 📈 Aggregated statistics across all runs | Analysis |

---

## 🖥️ Environment Setup

### 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| 🐍 **Python** | 3.9+ | 3.10+ |
| 🔥 **PyTorch** | 2.0 (CPU) | 2.0+ (CUDA 11.8+) |
| 💾 **RAM** | 4 GB | 8+ GB |
| 💿 **Storage** | 2 GB | 5+ GB (for full evaluation) |
| 🎮 **GPU** | Optional | NVIDIA GTX 1060+ (for faster training) |

### 📦 Dependencies

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

> 💡 **Tip**: Use a virtual environment for isolation:
> ```bash
> python -m venv iout-env
> source iout-env/bin/activate  # Linux/Mac
> # or
> iout-env\Scripts\activate  # Windows
> pip install -r requirements.txt
> ```

### 🐳 Docker Setup (Recommended for Reproducibility)

```bash
# 🏗️ Build Docker image
docker build -t iout-interrogator:latest .

# 🚀 Run container with GPU support (if available)
docker run --gpus all -it iout-interrogator:latest

# 💾 Run with volume mounts for results persistence
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/notebooks:/app/notebooks \
  iout-interrogator:latest

# 🎯 Run specific command inside container
docker run --rm iout-interrogator:latest \
  python scripts/run_full_pipeline.py --quick
```

<details>
<summary><strong>🔧 Troubleshooting Common Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| ❌ `CUDA not available` | Ensure NVIDIA drivers & CUDA toolkit are installed; use `--device cpu` flag |
| ❌ `MemoryError` | Reduce `--runs` or `--agents` parameters; use `--batch-size 8` |
| ❌ `ModuleNotFoundError` | Reinstall dependencies: `pip install -r requirements.txt --force-reinstall` |
| ❌ `Docker permission denied` | Add user to docker group: `sudo usermod -aG docker $USER` |

</details>

---

## 📋 Supplementary Materials

Additional resources are available in the `docs/` directory:

| Document | Content | Format |
|----------|---------|--------|
| 📘 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Detailed reproduction guide, parameter specifications, and result validation | Markdown |
| 🏗️ [STRUCTURE.md](docs/STRUCTURE.md) | In-depth architecture documentation and module descriptions | Markdown |
| 📝 [CHANGELOG.md](docs/CHANGELOG.md) | Version history, updates, and breaking changes | Markdown |
| 🎓 [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Guidelines for contributing code, documentation, or issues | Markdown |

---

## 📚 Artifact Availability

<p align="center">
  <img src="https://img.shields.io/badge/✅-Fully_Open_Source-4CAF50?style=flat-square" alt="Open Source">
  <img src="https://img.shields.io/badge/✅-Code+%2B_Data-2196F3?style=flat-square" alt="Code + Data">
  <img src="https://img.shields.io/badge/✅-Reproducible-FF9800?style=flat-square" alt="Reproducible">
  <img src="https://img.shields.io/badge/✅-Containerized-607D8B?style=flat-square" alt="Containerized">
  <img src="https://img.shields.io/badge/✅-Cloud--Friendly-F44336?style=flat-square" alt="Cloud">
  <img src="https://img.shields.io/badge/✅-Well--Documented-9C27B0?style=flat-square" alt="Documented">
</p>

This repository is designed for **full reproducibility and transparency**:

✅ **Fully Open Source** — MIT License  
✅ **Code + Data** — All research artifacts included  
✅ **Reproducible** — Fixed seeds, documented parameters  
✅ **Containerized** — Docker support for consistent environments  
✅ **Cloud-Friendly** — All notebooks run in Google Colab  
✅ **Well-Documented** — Comprehensive docs and inline comments  

### 📄 Paper Availability

| Field | Details |
|-------|---------|
| 📑 **Paper Status** | Under review |
| 🔗 **Repository** | [IoUT-Interrogator-Framework](https://github.com/aliakarma/IoUT-Interrogator-Framework) |
| ⚖️ **License** | MIT (Open Source) |
| 🗄️ **Permanence** | GitHub + Zenodo archival (DOI pending) |
| 📬 **Correspondence** | ali.akarma@institution.edu |

---

## 📖 Citation

### 🎓 BibTeX Format

```bibtex
@article{akarma2026interrogator,
  author  = {Akarma, Ali and Syed, Toqeer Ali and Jilani, Abdul Khadar and
             Jan, Salman and Muneer, Hammad and Khan, Muazzam A. and Yu, Changli},
  title   = {Interrogator-Based Behavioral Trust Inference for the Internet of Underwater Things Using Transformer-Based Temporal Modeling},
  journal = {[Under Review]},
  year    = {2026},
  url     = {https://github.com/aliakarma/IoUT-Interrogator-Framework},
  note    = {Code and data available at GitHub repository}
}
```

### 📚 APA Format

```text
Akarma, A., Syed, T. A., Jilani, A. K., Jan, S., Muneer, H., Khan, M. A., & Yu, C. (2026). 
Interrogator-based behavioral trust inference for the internet of underwater things using 
transformer-based temporal modeling. [Journal Name], [Volume(Issue)], pages. (Under Review). 
https://github.com/aliakarma/IoUT-Interrogator-Framework
```

<details>
<summary><strong>🔗 Export Citation</strong></summary>

```bash
# 📄 Download BibTeX
curl -O https://raw.githubusercontent.com/aliakarma/IoUT-Interrogator-Framework/main/CITATION.cff

# 📚 Generate APA/MLA via CFF convertor
# Visit: https://citation-file-format.github.io/
```

</details>

---

## 📄 License

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-00A8E8?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

```text
MIT License

Copyright (c) 2026 Ali Akarma et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

📄 Full license text: [LICENSE](LICENSE)

---

## 🤝 Reproducibility Commitment

<p align="center">
  <img src="https://img.shields.io/badge/🔬-Scientific_Rigor-4CAF50?style=flat-square" alt="Scientific Rigor">
  <img src="https://img.shields.io/badge/♻️-Full_Transparency-2196F3?style=flat-square" alt="Transparency">
  <img src="https://img.shields.io/badge/✨-Long_term_Maintenance-FF9800?style=flat-square" alt="Maintenance">
</p>

All code, simulation parameters, and evaluation procedures are provided to ensure **full transparency and reproducibility**.

✨ **Artifacts will be maintained and updated** in alignment with the final published version of the paper.

### ✅ Reproduction Checklist

<details open>
<summary><strong>Click to copy checklist</strong></summary>

```markdown
- [ ] ✅ Python 3.9+ installed (`python --version`)
- [ ] ✅ Dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ Quick demo runs successfully (`python scripts/run_full_pipeline.py --quick`)
- [ ] ✅ Full evaluation executed (`python scripts/reproduce_all.py --seed 42 --runs 30`)
- [ ] ✅ Results match [analysis/stats/summary_table.csv](analysis/stats/summary_table.csv)
- [ ] ✅ Plots generated in [analysis/plots/](analysis/plots/)
- [ ] ✅ Docker build successful (optional) (`docker build -t iout-interrogator:latest .`)
```

</details>

### 🔄 Version Compatibility

| Repository Version | Paper Version | Status |
|-------------------|---------------|--------|
| `v1.0.0` (main) | Pre-print v1 | ✅ Stable |
| `v1.1.0` (dev) | Under Review | 🚧 Active Development |
| `v2.0.0` (future) | Published | 📅 Planned |

---

## 🤝 Contributing

<p align="center">
  <img src="https://img.shields.io/badge/🤝-Contributions_Welcome-4CAF50?style=flat-square" alt="Contributions Welcome">
  <img src="https://img.shields.io/badge/🐛-Report_Issues-F44336?style=flat-square" alt="Report Issues">
  <img src="https://img.shields.io/badge/💡-Suggest_Features-2196F3?style=flat-square" alt="Suggest Features">
</p>

We welcome contributions from the research community! 🎉

### 🚀 How to Contribute

1. 🔍 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 **Push** to the branch (`git push origin feature/amazing-feature`)
5. 🔗 **Open** a Pull Request

### 📋 Contribution Guidelines

| Type | Guidelines |
|------|-----------|
| 🐛 **Bug Reports** | Include OS, Python version, error traceback, and minimal reproducible example |
| 💡 **Feature Requests** | Describe use case, expected behavior, and potential implementation approach |
| 📝 **Documentation** | Follow existing style; update `docs/` and inline docstrings |
| 🧪 **Code Contributions** | Add tests in `tests/`; ensure all tests pass (`pytest tests/`) |

📄 Full guidelines: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) *(coming soon)*

---

## 📬 Support & Contact

<p align="center">
  <a href="https://github.com/aliakarma/IoUT-Interrogator-Framework/issues">
    <img src="https://img.shields.io/badge/🐛-Report_Bug-F44336?style=for-the-badge&logo=github&logoColor=white" alt="Report Bug">
  </a>
  <a href="https://github.com/aliakarma/IoUT-Interrogator-Framework/discussions">
    <img src="https://img.shields.io/badge/💬-Start_Discussion-2196F3?style=for-the-badge&logo=github&logoColor=white" alt="Start Discussion">
  </a>
  <a href="mailto:ali.akarma@institution.edu">
    <img src="https://img.shields.io/badge/📧-Email_Us-00A8E8?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

### 🆘 Getting Help

| Issue Type | Recommended Action |
|-----------|-------------------|
| 🔧 Installation problems | Check [Environment Setup](#️-environment-setup) • Search [Issues](https://github.com/aliakarma/IoUT-Interrogator-Framework/issues) |
| 🤔 Usage questions | Read [Notebooks](#-interactive-notebooks) • Start a [Discussion](https://github.com/aliakarma/IoUT-Interrogator-Framework/discussions) |
| 🐛 Bug reports | Create a [New Issue](https://github.com/aliakarma/IoUT-Interrogator-Framework/issues/new) with reproduction steps |
| 💡 Feature ideas | Open a [Discussion](https://github.com/aliakarma/IoUT-Interrogator-Framework/discussions) or [Issue](https://github.com/aliakarma/IoUT-Interrogator-Framework/issues/new) |
| 📬 General inquiries | Email: `ali.akarma@institution.edu` |

### 🌟 Acknowledgments

- 🎓 Research supported by [Institution Name]
- 🤝 Collaborations with [Partner Institutions]
- 🙏 Thanks to the PyTorch, Hugging Face, and IoUT research communities

---

<p align="center">
  <strong>🌊 Advancing Trust in Autonomous Underwater Networks</strong><br>
  <sub>Made with ❤️ by the IoUT Research Team • MIT Licensed • Reproducible by Design</sub>
</p>

<p align="center">
  <a href="#-interrogatorbased-behavioral-trust-inference-for-iout">
    <img src="https://img.shields.io/badge/⬆️-Back_to_Top-607D8B?style=flat-square" alt="Back to Top">
  </a>
</p>
```