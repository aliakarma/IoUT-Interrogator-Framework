# Dataset Documentation

## Data Source

The default pipeline dataset at data/raw/behavioral_sequences.json is synthetic behavioral telemetry generated for IoUT trust-inference experiments.

The pipeline also supports real-data evaluation with UNSW-NB15 CSV files via dataset mode unsw_nb15.

The synthetic design is used to emulate underwater communication and mobility uncertainty where collecting large labeled real-world underwater traces is expensive.

## Schema

Supported schema for pipeline ingestion:

1. Sequence JSON records
- sensor_id: unique node identifier
- sequence: array of time-ordered signal vectors
- label: binary class where 1 indicates adversarial or untrustworthy behavior
- timestamp: optional sequence-level timestamp

2. Long-form CSV or JSON records
- sensor_id: unique node identifier
- timestamp: sortable event timestamp
- signal: scalar or vector signal value at timestamp
- label: optional binary class

3. UNSW-NB15 CSV (real dataset mode)
- required columns: dur, spkts, ct_dst_ltm, ct_srv_src, state
- label source: label (preferred) or attack_cat
- mapping target: approximate behavioral features for timing, packet-rate, stability, churn, and protocol anomaly
- sequence construction: sliding window with fixed K=20

## Dataset Size and Structure

The effective sample count, sequence length distribution, and class balance are reported automatically during data loading and stored in run metadata.

Reported statistics include:
- n_samples
- sequence_length min, max, mean, std
- class_balance
- imbalance_ratio

## Signal Generation Process (Synthetic)

The synthetic signal traces are produced as time-series behavior signatures per sensor node. The generator models:
- baseline benign temporal dynamics
- adversarial perturbation patterns
- temporal drift and stochastic fluctuations

## Noise Model and Underwater Assumptions

The robustness pipeline injects additive Gaussian noise and dropout-like corruption to approximate underwater channel volatility, packet degradation, and intermittent sensing failure.

Assumptions used by robustness experiments:
- additive Gaussian noise for environmental and channel disturbances
- missing-segment corruption for packet/sampling loss
- sensor failure masking for node-level outages

## Limitations

- Synthetic data may not fully capture hydrodynamic channel physics.
- Class boundaries may be cleaner than real deployments.
- Domain shift to field data is expected and should be quantified with external validation.
- Reported benchmark metrics should be interpreted as controlled-comparison results, not deployment readiness guarantees.

## UNSW-NB15 Usage

Expected default path for UNSW runs:
- data/raw/unsw_nb15/UNSW_NB15_training-set.csv

Single run:
- python run_pipeline.py --dataset unsw_nb15 --data-path data/raw/unsw_nb15/UNSW_NB15_training-set.csv

Multi-seed real-data benchmark:
- python scripts/run_multi_seed_experiments.py --dataset unsw_nb15 --data-path data/raw/unsw_nb15/UNSW_NB15_training-set.csv --models hybrid_temporal,lstm,random_forest --seeds 42-61 --output-dir results/real_data_20seed --aggregate-output results/real_data_20seed/aggregate_metrics.json --tests-output results/real_data_20seed/statistical_tests.json --final-table-output results/real_data_20seed/final_results_table.csv

Real-data figure generation:
- python scripts/generate_figures.py --results-dir results/real_data_20seed --aggregate-path results/real_data_20seed/aggregate_metrics.json --figures-dir results/figures_real --figures-data-dir results/figures_data_real --name-suffix _real --skip-robustness
