# Repository Structure

```text
IoUT-Interrogator-Framework/
|-- README.md                          # Project overview and reproducibility entrypoint
|-- requirements.txt                   # Python dependencies (pinned versions)
|-- Dockerfile                         # Reproducible Docker environment
|-- .gitignore                         # Git ignore rules
|-- CITATION.cff                       # Machine-readable citation metadata
|-- LICENSE                            # MIT License
|-- compat.py                          # Python version guard used by entry scripts
|-- data/
|   |-- raw/                           # Generated behavioral sequences and labels
|   |-- processed/
|   |   `-- trust_scores.csv           # Inference output trust scores
|   `-- sample/
|       `-- sample_sequences.json      # Demo input for inference pipeline
|-- simulation/
|   |-- configs/
|   |   `-- simulation_params.json     # Simulation hyperparameters
|   |-- scripts/
|   |   |-- environment.py             # IoUT multi-agent environment
|   |   |-- run_simulation.py          # Simulation orchestrator
|   |   `-- generate_behavioral_data.py# Synthetic sequence generator
|   `-- outputs/
|       |-- results.csv                # Aggregated interval-level metrics
|       |-- raw_results.csv            # Long-format run-level metrics
|       `-- multi_seed_raw_results.csv # Multi-seed long-format metrics
|-- model/
|   |-- configs/
|   |   `-- transformer_config.json    # Model architecture and training config
|   |-- inference/
|   |   |-- transformer_model.py       # TrustTransformer model definition
|   |   |-- train.py                   # Training entrypoint
|   |   |-- infer.py                   # Inference entrypoint
|   |   |-- baseline_models.py         # Classical model baselines
|   |   |-- lstm_baseline.py           # LSTM baseline implementation
|   |   `-- data_hardening.py          # Data hardening utilities
|   `-- checkpoints/
|       `-- best_model.pt              # Generated trained checkpoint
|-- blockchain/
|   |-- configs/
|   |   `-- consortium_config.json     # Consortium configuration template
|   |-- chaincode/
|   |   `-- trust_ledger.go            # Smart contract template
|   `-- scripts/
|       `-- mock_ledger.py             # Python mock PBFT ledger
|-- analysis/
|   |-- plot_trust_accuracy.py         # Trust accuracy plotting
|   |-- plot_energy.py                 # Energy plotting
|   |-- plot_pdr.py                    # PDR plotting
|   |-- plot_ablation.py               # Ablation plotting
|   |-- plot_calibration.py            # Calibration plotting
|   |-- plot_ood_results.py            # OOD result plotting
|   |-- plot_all.py                    # Generate all core plots
|   |-- sensitivity_study.py           # Threshold/sequence sensitivity pipeline
|   |-- statistical_summary.py         # Statistical summary generation
|   |-- plots/                         # Generated PNG figures
|   |-- stats/                         # Generated CSV summary tables
|   |-- final_results/                 # Exported publication-ready result tables
|   |-- smoke/                         # Smoke run artifacts
|   |-- smoke_hardened/                # Hardened smoke artifacts
|   `-- smoke_*/                       # Additional smoke variant artifacts
|-- notebooks/
|   |-- 01_trust_inference_demo.ipynb  # Model training and inference demo
|   |-- 02_simulation_analysis.ipynb   # Simulation and metrics analysis demo
|   |-- 03_blockchain_demo.ipynb       # Blockchain mock ledger demo
|   |-- 04_ablation_study.ipynb        # Ablation study demo
|   `-- 05_enhancements_validation.ipynb # Hardened/tuned enhancements validation
|-- scripts/
|   |-- run_full_pipeline.py           # End-to-end pipeline
|   |-- reproduce_all.py               # Reproduction wrapper
|   |-- run_multi_seed_experiments.py  # Multi-seed experiment runner
|   |-- run_ablation_study.py          # Ablation execution helper
|   |-- run_ood_evaluation.py          # OOD evaluation helper
|   |-- smoke_validate_classifier.py   # Smoke validation checks
|   |-- profile_inference.py           # Inference profiling utility
|   `-- export_all_results.py          # Export consolidated results package
|-- tests/
|   |-- test_environment.py            # Simulation environment tests
|   |-- test_model.py                  # Model and training tests
|   `-- test_blockchain.py             # Mock ledger tests
|-- docs/
|   |-- REPRODUCIBILITY.md             # Reproduction instructions
|   |-- CHANGELOG.md                   # Repository changelog
|   |-- THRESHOLD_SWEEP_AND_FOCAL_LOSS.md # Threshold/focal-loss experiment notes
|   `-- STRUCTURE.md                   # This file
`-- .github/
    `-- workflows/
        `-- ci.yml                     # GitHub Actions CI pipeline
```

---

## Key Design Decisions

### Why Python instead of NS-3 / Hyperledger Fabric?

The paper's prototype uses NS-3 Aqua-Sim and Hyperledger Fabric for simulation and
blockchain governance, respectively. These systems require complex installation
and are not easily reproducible in standard Python environments. This repository
provides a Python reimplementation artifact that:

1. Preserves core acoustic-channel behavior and trust dynamics.
2. Implements behavioral metadata generation with reproducible seeds.
3. Simulates consortium-style ledger behavior using a mock PBFT flow.
4. Produces traceable outputs for this Python pipeline.

### Data Flow

```text
simulation/scripts/generate_behavioral_data.py
  -> data/raw/behavioral_sequences.json + data/raw/labels.csv
model/inference/train.py
  -> model/checkpoints/best_model.pt
model/inference/infer.py
  -> data/processed/trust_scores.csv
simulation/scripts/run_simulation.py
  -> simulation/outputs/results.csv + simulation/outputs/raw_results.csv
analysis/statistical_summary.py
  -> analysis/stats/summary_table.csv
scripts/export_all_results.py
  -> analysis/final_results/*.csv + analysis/final_results/provenance_manifest.json
```
