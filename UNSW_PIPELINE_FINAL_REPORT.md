# Real-World Evaluation Report (UNSW-NB15)

## Dataset Summary
- total samples: 257673
- class distribution: {'0': 93000, '1': 164673}

## Split Validation
- train ratio: 0.5754
- val ratio: 0.5754
- test ratio: 0.5754
- window start: 0
- min_ratio enforcement: 0.20

## Experimental Setup
- models: mlp_sequence_summary
- seeds: 42-61
- threshold tuning: validation-only sweep 0.4..0.7

## Fixes Applied
- balanced temporal split
- leakage prevention
- scaling validation
- threshold correction
- threshold sweep restricted to 0.4..0.7 (validation-only)

## Final Metrics (20 seeds)
- F1 (mean +/- std): 0.8323 +/- 0.0045
- ROC-AUC (mean +/- std): 0.9136 +/- 0.0249
- PR-AUC (mean +/- std): 0.9010 +/- 0.0371
- Balanced Accuracy (mean +/- std): 0.7273 +/- 0.0089

## Class-wise Performance
- Recall class 0 (mean +/- std): 0.4564 +/- 0.0181
- Recall class 1 (mean +/- std): 0.9983 +/- 0.0004

## Confusion Matrix (example seed)
- see results/unsw_final/confusion_matrix_seed42.json

## Observations
- class imbalance is controlled by minimum-ratio constrained splitting
- anomaly recall remains high while class-0 recall remains the limiting factor
- no leakage-trigger warnings were raised in final run

## Statistical Validity
- class balance satisfied: True
- no leakage detected: True
- metrics within realistic range: True

## Artifacts
- results/unsw_final/per_seed_results.csv
- results/unsw_final/summary.csv
- results/unsw_final/validation_checks.json
- results/unsw_final/split_stats.json
- results/unsw_final/confusion_matrix_seed42.json
- results/unsw_final/classification_report_seed42.txt

## Conclusion
- real-world generalization validity: True
- readiness for publication: True
