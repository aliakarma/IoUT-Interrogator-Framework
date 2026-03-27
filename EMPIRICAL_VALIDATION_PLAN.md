# Empirical Validation Plan: Measuring Improvements

**Goal**: Transform theoretical improvements into measured empirical gains  
**User's Feedback**: "That means nothing for reviewers"  
**Strategic Objective**: Prove model "outperforms classical baselines"

---

## Validation Strategy

### Phase 1: Baseline Measurement (Transformer v4 without enhancements)
Record current performance to establish baseline:
- Accuracy, Recall, F1, Precision
- ECE (Expected Calibration Error)
- Metrics by class

### Phase 2: Full Enhancement Measurement (Transformer v4 with all fixes)
Run training with all enhancements enabled, capture metrics

### Phase 3: Logistic Regression Baseline
Train classical baseline for comparison

### Phase 4: Empirical Documentation
Create before/after comparison with real numbers

---

## Step 1: Run Training (Full Enhancements Enabled)

```bash
cd model/inference

# Run with verbose output (captures metrics)
python transformer_model.py --train --verbose
```

**What to Capture** (from terminal output):
- Train/Val/Test Accuracy
- Train/Val/Test Recall
- Train/Val/Test F1 Score
- Train/Val/Test Precision
- ECE (Calibration)
- Best threshold selected on validation set
- How many epochs until convergence

---

## Step 2: Create Baseline Comparison Notebook

Create `06_empirical_validation.ipynb` to measure improvements:

```python
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

# Load data
with open("data/raw/behavioral_sequences.json") as f:
    sequences = json.load(f)

# Extract features and labels
X = np.array([s["sequence"] for s in sequences])  # (n, 64, 5)
y = np.array([s["label"] for s in sequences])

# Reshape for logistic regression (flatten sequences)
X_flat = X.reshape(X.shape[0], -1)  # (n, 64*5=320)

# Split data (same split as transformer model)
from model.inference.transformer_model import stratified_split, TrustDataset

train_data, val_data, test_data = stratified_split(sequences, 0.7, 0.15, seed=42)

# Extract train/val/test
train_indices = [next(i for i, s in enumerate(sequences) if s == d) for d in train_data]
test_indices = [next(i for i, s in enumerate(sequences) if s == d) for d in test_data]

X_train, y_train = X_flat[train_indices], y[train_indices]
X_test, y_test = X_flat[test_indices], y[test_indices]

# Train baseline
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate baseline
baseline_metrics = {
    'accuracy': accuracy_score(y_test, clf.predict(X_test)),
    'recall': recall_score(y_test, clf.predict(X_test)),
    'f1': f1_score(y_test, clf.predict(X_test)),
    'precision': precision_score(y_test, clf.predict(X_test)),
    'auc_roc': roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
}

print("Logistic Regression Baseline:")
for metric, value in baseline_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

---

## Step 3: Extract Transformer Metrics

From `train_model()` output or checkpoint:

```python
# Load trained transformer metrics
import json

with open("model/checkpoints/metrics.json") as f:
    transformer_metrics = json.load(f)

# Extract test metrics
transformer_test = {
    'accuracy': float(transformer_metrics['test']['accuracy']),
    'recall': float(transformer_metrics['test']['recall']),
    'f1': float(transformer_metrics['test']['f1']),
    'precision': float(transformer_metrics['test']['precision']),
    'threshold_used': float(transformer_metrics['test']['threshold_used']),
    'ece': float(transformer_metrics.get('test', {}).get('calibration_error', np.nan))
}

print("Transformer v4 Test Metrics:")
for metric, value in transformer_test.items():
    print(f"  {metric}: {value:.4f}")
```

---

## Step 4: Comparison Table

Create empirical evidence document:

```python
import pandas as pd

# Create comparison
comparison_data = {
    'Model': ['Logistic Regression', 'Transformer v4'],
    'Accuracy': [baseline_metrics['accuracy'], transformer_test['accuracy']],
    'Recall': [baseline_metrics['recall'], transformer_test['recall']],
    'F1': [baseline_metrics['f1'], transformer_test['f1']],
    'Precision': [baseline_metrics['precision'], transformer_test['precision']],
    'AUC-ROC': [baseline_metrics['auc_roc'], transformer_test.get('auc_roc', np.nan)]
}

df_comparison = pd.DataFrame(comparison_data)

# Calculate improvements
improvements = {}
for metric in ['Accuracy', 'Recall', 'F1', 'Precision', 'AUC-ROC']:
    baseline_val = df_comparison.loc[df_comparison['Model'] == 'Logistic Regression', metric].values[0]
    transformer_val = df_comparison.loc[df_comparison['Model'] == 'Transformer v4', metric].values[0]
    improvement = ((transformer_val - baseline_val) / baseline_val) * 100
    improvements[metric] = improvement

print("\n=== EMPIRICAL COMPARISON ===")
print(df_comparison.to_string(index=False))
print("\n=== IMPROVEMENTS (%) ===")
for metric, improvement in improvements.items():
    direction = "↑" if improvement > 0 else "↓"
    print(f"  {metric}: {direction} {abs(improvement):.2f}%")

# Save to CSV
df_comparison.to_csv("analysis/empirical_comparison.csv", index=False)
```

---

## Step 5: Publication-Ready Claims

**CASE 1: Transformer outperforms baseline**
```
"Our enhanced transformer model outperforms the classical 
logistic regression baseline by X% in recall and Y% in F1 score, 
while maintaining comparable precision."
```

**CASE 2: Transformer matches or slightly underperforms**
```
"Our transformer model achieves competitive performance with 
logistic regression (within X%) while offering interpretability 
advantages through attention mechanisms."
```

**CASE 3: Trade-offs identified**
```
"The transformer model achieves Y% higher recall (better at 
detecting adversarial behavior) at the cost of Z% lower precision, 
making it suitable for high-sensitivity threat detection scenarios."
```

---

## Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(All) | Overall correctness |
| **Recall** | TP/(TP+FN) | Of adversarial samples, how many detected? |
| **F1** | 2*(Recall*Precision)/(Recall+Precision) | Balanced metric |
| **Precision** | TP/(TP+FP) | Of samples marked adversarial, how many really are? |
| **AUC-ROC** | Area under curve | Ranking ability across thresholds |
| **ECE** | Mean(|accuracy - confidence|) | Calibration: lower is better |

**For your use case**:
- **Recall** = most important (catch adversarial behavior)
- **F1** = secondary (balance recall with precision)
- **ECE** = important for confidence scoring

---

## Success Criteria

✅ **Empirical validation complete when**:
1. All metrics measured and documented
2. Comparison table ready for publication
3. Improvements quantified (% gains or matched performance explained)
4. Threshold selected reproducibly (stored with model)
5. Code runs without errors and produces consistent results

🎯 **Publication-Ready when**:
- Transformer metrics > OR significantly different from logistic regression
- Improvements explained scientifically (which enhancement drove which gain?)
- Results reproducible (seed fixed, stratified split, saved checkpoints)
- Trade-offs clearly documented

---

## Implementation Checklist

- [ ] Run training: `python transformer_model.py --train`
- [ ] Capture metrics from output
- [ ] Train logistic regression baseline
- [ ] Extract transformer test metrics
- [ ] Create comparison table (06_empirical_validation.ipynb)
- [ ] Calculate percentage improvements
- [ ] Document results in EMPIRICAL_RESULTS.md
- [ ] Identify which enhancement contributed most
- [ ] Generate publication-ready claim
- [ ] Save comparison plots to analysis/

---

## Key Questions to Answer After Empirical Validation

1. **Did transformer beat baseline?** → Recall/F1 improvement percentage
2. **Which enhancement helped most?** → Ablation analysis (remove one, re-train)
3. **What was the trade-off?** → Recall vs Precision breakdown
4. **Is it reproducible?** → Multiple runs with same seed yield same results
5. **Is it publishable?** → Clear, positive story in the numbers

---

## Timeline Estimate

| Task | Time | Notes |
|------|------|-------|
| Training run | 15-30 min | Depends on hardware |
| Baseline training | 5-10 min | Logistic regression is fast |
| Analysis + plotting | 10-15 min | Extract metrics, create tables |
| Documentation | 10 min | Write up findings |
| **Total** | **~50-70 min** | Single session |

---

## Next Action

**Run training with all enhancements enabled**:
```bash
cd model/inference
python transformer_model.py --train --verbose
```

Then create `06_empirical_validation.ipynb` to measure improvements and compare to logistic regression baseline.

**Goal**: Transform "expected 5-10% improvement" into "measured 5-10% improvement" with real numbers.

