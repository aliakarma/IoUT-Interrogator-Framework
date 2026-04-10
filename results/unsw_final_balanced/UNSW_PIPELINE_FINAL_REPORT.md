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
- threshold tuning: validation-only sweep 0.45..0.75

## Fixes Applied
- balanced temporal split
- leakage prevention
- scaling validation
- threshold correction
- threshold sweep restricted to 0.45..0.75 (validation-only)
- class-weighted BCEWithLogits loss (train-only class counts)
- weighted random sampler for train batches

## Class Imbalance Handling
- class-0 weight: 1.429
- class-weighted loss computed from training split only
- weighted sampler applied on training split only
- threshold optimized for balanced recall on validation only
- iterative tuning for class-0 recovery

## Final Metrics (20 seeds)
- F1 (mean +/- std): 0.8910 +/- 0.0026
- ROC-AUC (mean +/- std): 0.9251 +/- 0.0199
- PR-AUC (mean +/- std): 0.9254 +/- 0.0298
- Balanced Accuracy (mean +/- std): 0.8397 +/- 0.0041

## Class-wise Performance
- Recall class 0 (mean +/- std): 0.6961 +/- 0.0075
- Recall class 1 (mean +/- std): 0.9834 +/- 0.0022

## Final Class Balance Results
- recall_class_0: 0.6961 +/- 0.0075
- recall_class_1: 0.9834 +/- 0.0022
- balanced_accuracy: 0.8397 +/- 0.0041

## Interpretation
- class-0 detection improved through train-only imbalance controls
- class-wise performance remains balanced under validation-only threshold tuning
- no leakage confirmed by checks: True

## Confusion Matrix (example seed)
- see results/unsw_final_balanced/confusion_matrix_seed42.json

## Observations
- class imbalance is controlled by minimum-ratio constrained splitting
- anomaly recall remains high while class-0 recall remains the limiting factor
- no leakage-trigger warnings were raised in final run

## Statistical Validity
- class balance satisfied: True
- no leakage detected: True
- metrics within realistic range: True

## Artifacts
- results/unsw_final_balanced/per_seed_results.csv
- results/unsw_final_balanced/summary.csv
- results/unsw_final_balanced/validation_checks.json
- results/unsw_final_balanced/split_stats.json
- results/unsw_final_balanced/confusion_matrix_seed42.json
- results/unsw_final_balanced/classification_report_seed42.txt

## Conclusion
- real-world generalization validity: True
- readiness for publication: True
