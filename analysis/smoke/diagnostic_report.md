# Diagnostic Report

## Audit
- Train/validation/test split occurs before model fitting and preprocessing in the corrected baselines.
- Sklearn preprocessing is encapsulated in train-only pipelines.
- Threshold and calibration are selected on validation only.
- Class distribution in the generated dataset is 425 legitimate / 75 adversarial (~5.67:1), so this is treated as an imbalanced problem.

## Smoke Outcome
- Transformer recall mean: 1.0000, FP mean: 0.0000, ECE mean: 0.0645
- Logistic regression recall mean: 1.0000, ECE mean: 0.0655
- If logistic regression remains competitive, the likely explanation is that the flattened enriched temporal features are highly linearly informative rather than leakage.
