from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def extract_behavioral_features(df: pd.DataFrame) -> np.ndarray:
    """Map UNSW-NB15 columns to approximate IoUT behavioral proxies.

    This is an approximation layer and does not claim one-to-one feature equivalence.
    """
    required = {"dur", "spkts", "ct_dst_ltm", "ct_srv_src", "state"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required UNSW feature columns: {missing}")

    dur = pd.to_numeric(df["dur"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    spkts = pd.to_numeric(df["spkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ct_dst_ltm = pd.to_numeric(df["ct_dst_ltm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ct_srv_src = pd.to_numeric(df["ct_srv_src"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    state = df["state"].fillna("UNK").astype(str)

    features = []

    # Timing proxy
    features.append(dur)

    # Packet rate proxy
    features.append(spkts / (dur + 1e-6))

    # Routing/traffic stability
    features.append(ct_dst_ltm)

    # Neighbor churn proxy
    features.append(ct_srv_src)

    # Protocol anomaly proxy
    features.append((state != "CON").astype(np.float32).to_numpy())

    return np.stack(features, axis=1).astype(np.float32)


def normalize_features(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Standard-score normalization.

    For 2D arrays [N, F], stats are computed over N.
    For 3D arrays [N, T, F], stats are computed per sequence over T to avoid cross-split leakage.
    """
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if arr.ndim == 2:
        axis = 0
        keepdims = True
    else:
        axis = 1
        keepdims = True

    if mean is None:
        mean = arr.mean(axis=axis, keepdims=keepdims)
    if std is None:
        std = arr.std(axis=axis, keepdims=keepdims)

    std = np.where(std < eps, 1.0, std)
    normalized = (arr - mean) / std
    return normalized.astype(np.float32), {"mean": np.asarray(mean), "std": np.asarray(std)}


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    sequences, labels = [], []
    for i in range(len(X) - seq_len):
        sequences.append(X[i : i + seq_len])
        labels.append(y[i + seq_len])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)
