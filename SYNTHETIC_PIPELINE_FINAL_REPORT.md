# Synthetic Evaluation Report

## Dataset Description
- synthetic IoUT scenario
- controlled conditions

## Experimental Setup
- models used: hybrid_temporal, lstm, random_forest, logistic_regression
- seeds: 42-61
- metrics: F1, ROC-AUC, PR-AUC, Balanced Accuracy

## Results (20 seeds)
### hybrid_temporal
- F1 (mean +/- std): 0.7851 +/- 0.0594
- ROC-AUC (mean +/- std): 0.9637 +/- 0.0089
- PR-AUC (mean +/- std): 0.8851 +/- 0.0199
- Balanced Accuracy (mean +/- std): 0.8514 +/- 0.0437

### random_forest
- F1 (mean +/- std): 0.7081 +/- 0.0404
- ROC-AUC (mean +/- std): 0.9003 +/- 0.0124
- PR-AUC (mean +/- std): 0.8088 +/- 0.0222
- Balanced Accuracy (mean +/- std): 0.7870 +/- 0.0257

### logistic_regression
- F1 (mean +/- std): 0.6667 +/- 0.0000
- ROC-AUC (mean +/- std): 0.8638 +/- 0.0000
- PR-AUC (mean +/- std): 0.7572 +/- 0.0000
- Balanced Accuracy (mean +/- std): 0.7758 +/- 0.0000

### lstm
- F1 (mean +/- std): 0.6444 +/- 0.0403
- ROC-AUC (mean +/- std): 0.8199 +/- 0.0422
- PR-AUC (mean +/- std): 0.6973 +/- 0.0345
- Balanced Accuracy (mean +/- std): 0.7513 +/- 0.0216

## Observations
- best average F1 model: hybrid_temporal (0.7851)
- neural sequence model remains competitive under controlled imbalance
- classical baselines provide stable lower-complexity references

## Conclusion
- controlled validation success
