# Agents for Agents: Interrogator-Based Secure IoUT Framework

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen)](tests/)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)]()
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb)

🔬 A research-grade, reproducible Python artifact for trust inference and enforcement in autonomous Internet of Underwater Things (IoUT) networks.

## ⚠️ Scope and Research Disclosure

- The paper prototype references NS-3 Aqua-Sim and Hyperledger Fabric.
- This repository is a high-fidelity Python proxy artifact for transparent, repeatable evaluation.
- Use repository outputs as artifact-derived results, not as direct packet-level NS-3 reproductions.
- Full methodological notes are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## ✨ What This Framework Includes

- 🤖 Lightweight transformer for behavioral trust scoring.
- 🔗 Permissioned blockchain mock layer for trust governance flow.
- 🛡️ Multi-tier trust-aware enforcement policy simulation.
- 🌊 IoUT environment with acoustic-channel-aware delivery and energy modeling.
- 📊 Full evaluation pipeline with accuracy, precision, recall, F1, confusion statistics, PDR, energy, and sensitivity analysis.

## 📈 Latest Reproducible Results (30 runs x 20 intervals, seed base 42)

Source artifacts:
- [simulation/outputs/results.csv](simulation/outputs/results.csv)
- [analysis/stats/summary_table.csv](analysis/stats/summary_table.csv)

| Metric | Static | Bayesian | Proposed |
|---|---:|---:|---:|
| Detection Accuracy (%) | 86.00 ± 0.00 | 78.71 ± 8.19 | **91.87 ± 4.96** |
| Precision (%) | 0.00 ± 0.00 | 44.99 ± 15.62 | **72.39 ± 18.34** |
| Recall (%) | 0.00 ± 0.00 | 58.52 ± 20.26 | **76.69 ± 12.50** |
| F1 Score (%) | 0.00 ± 0.00 | 46.61 ± 13.15 | **70.32 ± 13.14** |
| Packet Delivery Ratio (%) | 91.59 ± 3.85 | 92.31 ± 3.71 | **92.39 ± 3.75** |
| Residual Energy (%) | **75.33 ± 0.05** | 73.95 ± 0.06 | 73.87 ± 0.05 |

Notes:
- Reported values are artifact-derived means and mean-std summaries across interval-level aggregates.
- Static baseline correctly yields zero anomaly recall and zero F1 under fixed-trust behavior.

## 🚀 Quick Start

```bash
# 1) Clone
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run complete pipeline (data -> train -> simulate -> analysis)
python scripts/run_full_pipeline.py
```

## 🧪 Reproduce Main Evaluation

```bash
python scripts/reproduce_all.py --seed 42 --runs 30
```

This regenerates:
- [simulation/outputs/results.csv](simulation/outputs/results.csv)
- [analysis/stats/summary_table.csv](analysis/stats/summary_table.csv)
- plots in [analysis/plots](analysis/plots)

## 📉 Sensitivity and Ablation Studies

Threshold sensitivity and sequence-length ablation are available via:

```bash
python analysis/sensitivity_study.py --runs 20 --intervals 20 --seed 42
```

Generated artifacts:
- [analysis/stats/threshold_sensitivity.csv](analysis/stats/threshold_sensitivity.csv)
- [analysis/stats/sequence_length_ablation.csv](analysis/stats/sequence_length_ablation.csv)

## 📚 Colab Notebooks

| Notebook | Purpose | Open |
|---|---|---|
| [notebooks/01_trust_inference_demo.ipynb](notebooks/01_trust_inference_demo.ipynb) | Model training and trust inference demo | [Colab](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/01_trust_inference_demo.ipynb) |
| [notebooks/02_simulation_analysis.ipynb](notebooks/02_simulation_analysis.ipynb) | Simulation baseline comparison and metrics analysis | [Colab](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/02_simulation_analysis.ipynb) |
| [notebooks/03_blockchain_demo.ipynb](notebooks/03_blockchain_demo.ipynb) | Trust governance flow demonstration | [Colab](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/03_blockchain_demo.ipynb) |
| [notebooks/04_ablation_study.ipynb](notebooks/04_ablation_study.ipynb) | Ablation and figure generation | [Colab](https://colab.research.google.com/github/aliakarma/IoUT-Interrogator-Framework/blob/main/notebooks/04_ablation_study.ipynb) |

## 🗂️ Repository Layout

```text
IoUT-Interrogator-Framework/
|- simulation/         # IoUT environment, runner, configs, outputs
|- model/              # Transformer configs, training, inference, checkpoints
|- analysis/           # Plots, summaries, sensitivity studies
|- blockchain/         # Governance logic mock and chaincode skeleton
|- data/               # Synthetic raw/processed/sample datasets
|- scripts/            # End-to-end orchestration commands
|- tests/              # Unit and integration tests
|- docs/               # Changelog, structure, reproducibility guide
```

## 🖥️ Environment

- Python 3.9+
- PyTorch 2.x (CPU-first workflow)
- NumPy, Pandas, Matplotlib, scikit-learn (see [requirements.txt](requirements.txt))

Optional Docker build:

```bash
docker build -t iout-framework .
```

## 📖 Citation

If you use this repository in academic work, please cite:

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

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE).
