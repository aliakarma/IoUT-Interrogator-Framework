#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from data.unsw_loader import load_unsw_nb15_records

# Load and preprocess data
records = load_unsw_nb15_records(
    source="data/raw/unsw_nb15/Training and Testing Sets",
    seq_len=20,
    max_rows=15000,
)

X = np.array([r["sequence"] for r in records], dtype=np.float32)
y = np.array([r["label"] for r in records], dtype=np.int64)

print(f"Data shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")
print()

# Check raw feature statistics
X_flat = X.reshape(-1, X.shape[-1])
print("=== Raw Feature Statistics ===")
print(f"Min per feature: {X_flat.min(axis=0)}")
print(f"Max per feature: {X_flat.max(axis=0)}")
print(f"Mean per feature: {X_flat.mean(axis=0)}")
print(f"Std per feature: {X_flat.std(axis=0)}")
print()

# Perform stratified split
neg_indices = np.where(y == 0)[0]
pos_indices = np.where(y == 1)[0]

neg_train_cnt = int(len(neg_indices) * 0.70)
neg_val_cnt = int(len(neg_indices) * 0.15)
pos_train_cnt = int(len(pos_indices) * 0.70)
pos_val_cnt = int(len(pos_indices) * 0.15)

neg_train_idx = neg_indices[:neg_train_cnt]
pos_train_idx = pos_indices[:pos_train_cnt]

train_idx = np.sort(np.concatenate([neg_train_idx, pos_train_idx]))

X_train, y_train = X[train_idx], y[train_idx]

# Apply temporal gap
X_train = X_train[:-20]
y_train = y_train[:-20]

# Normalize
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
mean = X_train_flat.mean(axis=0, keepdims=True)
std = X_train_flat.std(axis=0, keepdims=True)
std = np.where(std < 1e-8, 1.0, std)

X_train_norm_flat = (X_train_flat - mean) / std
X_train_norm_flat = X_train_norm_flat.reshape(X_train.shape[0], -1)

print("=== Normalized Training Features ===")
print(f"Min: {X_train_norm_flat.min():.6f}")
print(f"Max: {X_train_norm_flat.max():.6f}")
print(f"Mean: {X_train_norm_flat.mean():.6f}")
print(f"Std: {X_train_norm_flat.std():.6f}")
print()

# Train model
print("=== Model Training ===")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_norm_flat, y_train)

print(f"Converged: {model.n_iter_ < model.max_iter}")
print(f"Coefficients shape: {model.coef_.shape}")
print(f"Coefficients min: {model.coef_.min():.10f}")
print(f"Coefficients max: {model.coef_.max():.10f}")
print(f"Intercept: {model.intercept_[0]:.10f}")
print()

# Test on training data
print("=== Training Performance ===")
proba_train = model.predict_proba(X_train_norm_flat)[:, 1]
print(f"Train probabilities min: {proba_train.min():.10f}")
print(f"Train probabilities max: {proba_train.max():.10f}")
print(f"Train probabilities mean: {proba_train.mean():.10f}")
print(f"Train probabilities median: {np.median(proba_train):.10f}")
print(f"Train accuracy (threshold 0.5): {(proba_train >= 0.5).mean():.4f}")
print(f"Actual positive rate in train: {y_train.mean():.4f}")
