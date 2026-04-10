#!/usr/bin/env python
import json
import numpy as np
from sklearn.metrics import confusion_matrix

with open('results/unsw_smoke_fixed/predictions.json') as f:
    data = json.load(f)

y_true = np.array(data['y_true'])
y_pred = np.array(data['y_pred'])
y_proba = np.array(data['y_proba'])

print('=== Test Set Analysis ===')
print(f'Total samples: {len(y_true)}')
print(f'Positive (true anomalies): {y_true.sum()}')
print(f'Negative (true normal): {len(y_true) - y_true.sum()}')
print()

print('=== Predictions at threshold 0.2 ===')
print(f'Predicted positive: {y_pred.sum()}')
print(f'Predicted negative: {len(y_pred) - y_pred.sum()}')
print()

print('=== Probability Distribution ===')
print(f'Min probability: {y_proba.min():.6f}')
print(f'Max probability: {y_proba.max():.6f}')
print(f'Mean probability: {y_proba.mean():.6f}')
print(f'Median probability: {np.median(y_proba):.6f}')
print(f'Samples with prob < 0.2: {(y_proba < 0.2).sum()}')
print(f'Samples with prob >= 0.2: {(y_proba >= 0.2).sum()}')
print()

print('=== Confusion Matrix ===')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f'TP (correct positives): {tp}')
print(f'TN (correct negatives): {tn}')
print(f'FP (false positives): {fp}')
print(f'FN (false negatives): {fn}')
print()

print('=== Threshold Analysis ===')
print('Checking what happens at different thresholds:')
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred_at_thresh = (y_proba >= thresh).astype(int)
    acc = (pred_at_thresh == y_true).mean()
    print(f'Threshold {thresh:.1f}: {(pred_at_thresh >= thresh).sum()} positive predictions, accuracy={acc:.4f}')
print()

print('=== Probability of each sample ===')
print(f'First 20 probabilities: {y_proba[:20]}')
print(f'Last 20 probabilities: {y_proba[-20:]}')
