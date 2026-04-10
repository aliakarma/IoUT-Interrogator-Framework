# REAL-DATASET PIPELINE FINAL VALIDATION

## Dataset Summary
- total samples: 257673
- class distribution: {'0': 93000, '1': 164673}

## Split Validation
- train ratio: 0.5254
- val ratio: 0.8639
- test ratio: 0.9449
- shift used: 29000

## Fixes Applied
- balanced temporal split
- leakage prevention
- scaling validation
- threshold correction

## Final Metrics (20 seeds)
- F1 (mean +/- std): 0.9695 +/- 0.0005
- ROC-AUC (mean +/- std): 0.8037 +/- 0.0189
- PR-AUC (mean +/- std): 0.9850 +/- 0.0020
- Balanced Accuracy (mean +/- std): 0.5071 +/- 0.0019

## Statistical Validity
- class balance satisfied: True
- no leakage detected: True
- metrics within realistic range: True

## Artifacts
- results/unsw_final/per_seed_results.csv
- results/unsw_final/summary.csv
- results/unsw_final/validation_checks.json
- results/unsw_final/split_stats.json

## Conclusion
- pipeline validity: True
- readiness for publication: True
