"""
Train-only data hardening utilities for sequence classification.

Applies controlled difficulty without touching validation/test labels or
split logic:
- feature noise injection using training-set feature std
- class-overlap injection on discriminative features
- light label noise on training only
- hard-sample duplication with perturbation
- feature dropout on training samples
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class HardeningReport:
    n_original: int
    n_augmented: int
    class_counts_before: Dict[int, int]
    class_counts_after: Dict[int, int]
    label_flip_count: int
    hard_sample_duplicates: int
    top_feature_indices: List[int]
    train_feature_std_mean: float
    train_feature_std_median: float
    gaussian_noise_sigma_frac: float
    overlap_alpha_range: Tuple[float, float]
    feature_dropout_rate: float


def _stack_sequences(sequences: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([np.array(item["sequence"], dtype=np.float32) for item in sequences], dtype=np.float32)
    y = np.array([int(item["label"]) for item in sequences], dtype=np.int32)
    return x, y


def _flatten_for_selection(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)


def _compute_top_discriminative_features(x_train: np.ndarray, y_train: np.ndarray, seed: int, top_k: int) -> List[int]:
    flat = _flatten_for_selection(x_train)
    mi = mutual_info_classif(flat, y_train, random_state=seed, discrete_features=False)
    ranked = np.argsort(mi)[::-1]
    top = ranked[: max(1, min(top_k, len(ranked)))]
    return [int(idx) for idx in top.tolist()]


def _fit_borderline_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    flat = _flatten_for_selection(x_train)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=1000,
                ),
            ),
        ]
    )
    model.fit(flat, y_train)
    return model


def _apply_feature_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    feature_std: np.ndarray,
    sigma_frac: float,
) -> np.ndarray:
    sigma = np.clip(feature_std * sigma_frac, 1e-4, None)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def _apply_overlap(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    top_flat_indices: List[int],
    alpha_low: float,
    alpha_high: float,
) -> np.ndarray:
    result = x.copy()
    n, seq_len, feat_dim = result.shape
    flat = result.reshape(n, seq_len * feat_dim)
    neg_mask = y == 0
    pos_mask = y == 1
    if not np.any(neg_mask) or not np.any(pos_mask):
        return result
    neg_mean = flat[neg_mask].mean(axis=0)
    pos_mean = flat[pos_mask].mean(axis=0)

    for idx in np.where(pos_mask)[0]:
        alpha = float(rng.uniform(alpha_low, alpha_high))
        flat[idx, top_flat_indices] = (
            alpha * flat[idx, top_flat_indices]
            + (1.0 - alpha) * neg_mean[top_flat_indices]
        )
    # Also make a small subset of legitimate samples slightly ambiguous to
    # avoid the trivial FP=0 regime while preserving class semantics.
    legit_indices = np.where(neg_mask)[0]
    if legit_indices.size > 0:
        n_soft_legit = max(1, int(0.08 * len(legit_indices)))
        chosen = rng.choice(legit_indices, size=n_soft_legit, replace=False)
        for idx in chosen:
            alpha = float(rng.uniform(0.82, 0.94))
            flat[idx, top_flat_indices] = (
                alpha * flat[idx, top_flat_indices]
                + (1.0 - alpha) * pos_mean[top_flat_indices]
            )

    return flat.reshape(n, seq_len, feat_dim).astype(np.float32)


def _apply_feature_dropout(x: np.ndarray, rng: np.random.Generator, dropout_rate: float) -> np.ndarray:
    mask = rng.random(size=x.shape) >= dropout_rate
    return (x * mask.astype(np.float32)).astype(np.float32)


def _flip_labels(y: np.ndarray, rng: np.random.Generator, flip_fraction: float) -> Tuple[np.ndarray, int]:
    if flip_fraction <= 0.0:
        return y.copy(), 0
    y_out = y.copy()
    n_flip = max(1, int(round(len(y) * flip_fraction)))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_out[flip_idx] = 1 - y_out[flip_idx]
    return y_out, int(n_flip)


def _duplicate_hard_samples(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    feature_std: np.ndarray,
    seed: int,
    duplication_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if duplication_fraction <= 0.0:
        return x, y, 0
    probe = _fit_borderline_probe(x, y, seed=seed)
    probs = probe.predict_proba(_flatten_for_selection(x))[:, 1]
    margins = np.abs(probs - 0.5)
    n_dup = max(1, int(round(len(y) * duplication_fraction)))
    hard_idx = np.argsort(margins)[:n_dup]
    sigma = np.clip(feature_std * 0.05, 5e-5, None)
    dup_x = []
    dup_y = []
    for idx in hard_idx:
        perturbed = x[idx] + rng.normal(0.0, sigma, size=x[idx].shape).astype(np.float32)
        dup_x.append(perturbed.astype(np.float32))
        dup_y.append(int(y[idx]))
    x_out = np.concatenate([x, np.array(dup_x, dtype=np.float32)], axis=0)
    y_out = np.concatenate([y, np.array(dup_y, dtype=np.int32)], axis=0)
    return x_out, y_out, int(len(dup_x))


def harden_training_sequences(
    train_sequences: List[dict],
    seed: int,
    enabled: bool = True,
    gaussian_sigma_frac: float = 0.10,
    overlap_alpha_low: float = 0.60,
    overlap_alpha_high: float = 0.90,
    label_flip_fraction: float = 0.03,
    hard_duplicate_fraction: float = 0.08,
    feature_dropout_rate: float = 0.08,
    top_feature_fraction: float = 0.08,
) -> Tuple[List[dict], HardeningReport]:
    originals = [copy.deepcopy(item) for item in train_sequences]
    x_train, y_train = _stack_sequences(originals)
    class_before = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    flat_feature_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0).astype(np.float32)

    if not enabled:
        report = HardeningReport(
            n_original=len(originals),
            n_augmented=len(originals),
            class_counts_before=class_before,
            class_counts_after=class_before,
            label_flip_count=0,
            hard_sample_duplicates=0,
            top_feature_indices=[],
            train_feature_std_mean=float(flat_feature_std.mean()),
            train_feature_std_median=float(np.median(flat_feature_std)),
            gaussian_noise_sigma_frac=0.0,
            overlap_alpha_range=(0.0, 0.0),
            feature_dropout_rate=0.0,
        )
        return originals, report

    rng = np.random.default_rng(seed)
    flat_dim = x_train.shape[1] * x_train.shape[2]
    top_k = max(1, int(round(flat_dim * top_feature_fraction)))
    top_features = _compute_top_discriminative_features(x_train, y_train, seed=seed, top_k=top_k)

    x_aug = _apply_feature_noise(x_train, rng=rng, feature_std=flat_feature_std.reshape(1, 1, -1), sigma_frac=gaussian_sigma_frac)
    x_aug = _apply_overlap(
        x_aug,
        y_train,
        rng=rng,
        top_flat_indices=top_features,
        alpha_low=overlap_alpha_low,
        alpha_high=overlap_alpha_high,
    )
    x_aug = _apply_feature_dropout(x_aug, rng=rng, dropout_rate=feature_dropout_rate)
    y_aug, n_flipped = _flip_labels(y_train, rng=rng, flip_fraction=label_flip_fraction)
    x_aug, y_aug, n_dup = _duplicate_hard_samples(
        x_aug,
        y_aug,
        rng=rng,
        feature_std=flat_feature_std.reshape(1, -1),
        seed=seed,
        duplication_fraction=hard_duplicate_fraction,
    )

    hardened_sequences: List[dict] = []
    for idx, (seq, label) in enumerate(zip(x_aug, y_aug)):
        source = originals[idx] if idx < len(originals) else originals[int(idx % len(originals))]
        item = copy.deepcopy(source)
        item["sequence"] = seq.astype(np.float32).tolist()
        item["label"] = int(label)
        item["augmentation_tag"] = "train_hardened"
        hardened_sequences.append(item)

    class_after = {int(k): int(v) for k, v in zip(*np.unique(y_aug, return_counts=True))}
    report = HardeningReport(
        n_original=len(originals),
        n_augmented=len(hardened_sequences),
        class_counts_before=class_before,
        class_counts_after=class_after,
        label_flip_count=int(n_flipped),
        hard_sample_duplicates=int(n_dup),
        top_feature_indices=top_features[: min(20, len(top_features))],
        train_feature_std_mean=float(flat_feature_std.mean()),
        train_feature_std_median=float(np.median(flat_feature_std)),
        gaussian_noise_sigma_frac=float(gaussian_sigma_frac),
        overlap_alpha_range=(float(overlap_alpha_low), float(overlap_alpha_high)),
        feature_dropout_rate=float(feature_dropout_rate),
    )
    return hardened_sequences, report
