# Repository Structure

```
IoUT-Interrogator-Framework/
│
├── README.md                        # Project overview, key results, quick start
├── requirements.txt                 # Python dependencies (pinned versions)
├── Dockerfile                       # Reproducible Docker environment
├── .gitignore                       # Git ignore rules
├── CITATION.cff                     # Machine-readable citation metadata
├── LICENSE                          # MIT License
│
├── data/
│   ├── raw/
│   │   ├── behavioral_sequences.json  # Generated: 500 agent sequences (K=64, d=5)
│   │   └── labels.csv                 # Generated: agent labels and attack types
│   ├── processed/
│   │   └── trust_scores.csv           # Generated: per-agent trust scores from inference
│   └── sample/
│       └── sample_sequences.json      # 10 pre-generated sequences for quick demos
│
├── simulation/
│   ├── configs/
│   │   └── simulation_params.json     # All simulation hyperparameters
│   ├── scripts/
│   │   ├── environment.py             # IoUT multi-agent simulation environment
│   │   ├── run_simulation.py          # Multi-run simulation orchestrator
│   │   └── generate_behavioral_data.py# Synthetic behavioral sequence generator
│   └── outputs/
│       └── results.csv                # Generated: aggregated simulation results
│
├── model/
│   ├── configs/
│   │   └── transformer_config.json    # Model architecture and training config
│   ├── inference/
│   │   ├── transformer_model.py       # TrustTransformer PyTorch model + dataset
│   │   ├── train.py                   # Training script (CLI entry point)
│   │   └── infer.py                   # Inference script (CLI entry point)
│   └── checkpoints/
│       └── best_model.pt              # Generated: trained model checkpoint
│
├── blockchain/
│   ├── configs/
│   │   └── consortium_config.json     # Hyperledger Fabric consortium config template
│   ├── chaincode/
│   │   └── trust_ledger.go            # Go smart contract for trust evidence storage
│   └── scripts/
│       └── mock_ledger.py             # Runnable Python mock of PBFT consortium
│
├── analysis/
│   ├── plot_trust_accuracy.py         # Figure 4 reproduction
│   ├── plot_energy.py                 # Figure 5 reproduction
│   ├── plot_pdr.py                    # Figure 6 reproduction
│   ├── plot_ablation.py               # Ablation study bar chart
│   ├── plot_all.py                    # Master script: generates all figures
│   ├── statistical_summary.py         # Summary statistics table generator
│   ├── plots/                         # Generated: PNG figures
│   └── stats/                         # Generated: CSV summary tables
│
├── notebooks/
│   ├── 01_trust_inference_demo.ipynb  # Colab: model training and inference demo
│   ├── 02_simulation_analysis.ipynb   # Colab: simulation run and figure reproduction
│   ├── 03_blockchain_demo.ipynb       # Colab: PBFT ledger demo
│   └── 04_ablation_study.ipynb        # Colab: ablation study reproduction
│
├── scripts/
│   ├── run_full_pipeline.py           # One-command pipeline (data → figures)
│   └── reproduce_all.py               # Exact paper reproduction (30 runs)
│
├── tests/
│   ├── test_environment.py            # Unit tests: simulation environment
│   ├── test_model.py                  # Unit tests: transformer trust model
│   └── test_blockchain.py             # Unit tests: mock PBFT ledger
│
├── docs/
│   ├── REPRODUCIBILITY.md             # Step-by-step reproduction guide
│   └── STRUCTURE.md                   # This file
│
└── .github/
    └── workflows/
        └── ci.yml                     # GitHub Actions CI pipeline
```

---

## Key Design Decisions

### Why Python instead of NS-3 / Hyperledger Fabric?

The paper's prototype uses NS-3 Aqua-Sim and Hyperledger Fabric for simulation and
blockchain governance, respectively. These systems require complex installation
and are not easily reproducible in standard Python environments. This repository
provides a **Python reimplementation artifact** that:

1. Preserve the core acoustic channel physics (Thorp model, energy equations)
2. Implement the same behavioral metadata generation logic
3. Simulate PBFT consensus with the same fault-tolerance parameters
4. Produces traceable, reproducible outputs for this Python pipeline

### Data Flow

```
generate_behavioral_data.py
    ↓ data/raw/behavioral_sequences.json
model/inference/train.py
    ↓ model/checkpoints/best_model.pt
model/inference/infer.py           simulation/scripts/run_simulation.py
    ↓                                   ↓
data/processed/trust_scores.csv    simulation/outputs/results.csv
                                        ↓
                               analysis/plot_all.py
                                        ↓
                               analysis/plots/*.png
```
