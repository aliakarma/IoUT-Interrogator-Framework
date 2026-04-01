# Diagnostic Report

## Audit
- Train/validation/test split occurs before model fitting and preprocessing in the corrected baselines.
- Sklearn preprocessing is encapsulated in train-only pipelines.
- Threshold and calibration are selected on validation only.
- Dataset separability changed from 2.876x to 0.988x average standardized separation across base features.
- Class distribution remains 425 legitimate / 75 adversarial (adv_fraction=0.150).
- Harder generation uses heavier-tailed noise, mild burst loss, latency spikes, and a low-and-slow-heavy attack mix.
- Training data only is additionally hardened with Gaussian feature noise, overlap injection, 3% label noise, hard-sample duplication, and feature dropout.

## Smoke Outcome
- Transformer recall mean: 0.6944, FP mean: 1.6667, ECE mean: 0.0752
- Logistic regression recall mean: 0.5833, ECE mean: 0.0768
- If logistic regression remains competitive, the likely explanation is that the enriched temporal features still preserve strong linear structure rather than leakage.
