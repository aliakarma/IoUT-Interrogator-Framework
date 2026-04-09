from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.data_loader import IoUTDataset, dataset_to_arrays


@dataclass
class BaselineOutput:
    y_pred: np.ndarray
    y_prob: np.ndarray


class RandomBaseline:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.threshold = 0.5

    def fit(self, train_loader) -> "RandomBaseline":
        return self

    def predict_proba(self, loader) -> np.ndarray:
        probabilities = []
        for batch in loader:
            batch_size = batch["label"].shape[0]
            probabilities.extend(self.rng.random() for _ in range(batch_size))
        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, loader) -> np.ndarray:
        return (self.predict_proba(loader) >= self.threshold).astype(np.int64)


class MajorityClassBaseline:
    def __init__(self) -> None:
        self.majority_label = 0

    def fit(self, train_loader) -> "MajorityClassBaseline":
        labels = []
        for batch in train_loader:
            labels.extend(batch["label"].cpu().numpy().astype(int).tolist())
        if labels:
            counts = np.bincount(np.asarray(labels, dtype=np.int64))
            self.majority_label = int(counts.argmax())
        return self

    def predict_proba(self, loader) -> np.ndarray:
        total = 0
        for batch in loader:
            total += batch["label"].shape[0]
        probability = 1.0 if self.majority_label == 1 else 0.0
        return np.full(total, probability, dtype=np.float32)

    def predict(self, loader) -> np.ndarray:
        total = 0
        for batch in loader:
            total += batch["label"].shape[0]
        return np.full(total, self.majority_label, dtype=np.int64)


class RuleBasedBaseline:
    def __init__(self) -> None:
        self.threshold = 0.0

    @staticmethod
    def _score(signal: np.ndarray) -> float:
        if signal.ndim == 1:
            signal = signal[:, None]
        mean_level = float(signal.mean())
        volatility = float(signal.std())
        trend = float(signal[-1].mean() - signal[0].mean())
        return mean_level + 0.5 * volatility + 0.25 * trend

    def fit(self, train_loader) -> "RuleBasedBaseline":
        scores = []
        labels = []
        for batch in train_loader:
            for index in range(batch["signal"].shape[0]):
                length = int(batch["lengths"][index].item())
                signal = batch["signal"][index, :length].numpy()
                scores.append(self._score(signal))
                labels.append(int(batch["label"][index].item()))
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)
        legit_scores = scores_arr[labels_arr == 0]
        adv_scores = scores_arr[labels_arr == 1]
        if len(legit_scores) and len(adv_scores):
            self.threshold = float((legit_scores.mean() + adv_scores.mean()) / 2.0)
        else:
            self.threshold = float(scores_arr.mean()) if len(scores_arr) else 0.0
        return self

    def predict_proba(self, loader) -> np.ndarray:
        probabilities = []
        for batch in loader:
            for index in range(batch["signal"].shape[0]):
                length = int(batch["lengths"][index].item())
                signal = batch["signal"][index, :length].numpy()
                score = self._score(signal)
                probabilities.append(float(1.0 / (1.0 + np.exp(-(score - self.threshold)))))
        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, loader) -> np.ndarray:
        return (self.predict_proba(loader) >= 0.5).astype(np.int64)


class MovingAverageBaseline:
    def __init__(self, window: int = 5) -> None:
        self.window = int(window)
        self.threshold = 0.0

    def _score(self, signal: np.ndarray) -> float:
        if signal.ndim == 1:
            signal = signal[:, None]
        series = signal.mean(axis=1)
        window = min(self.window, max(len(series) // 2, 1))
        head = float(series[:window].mean())
        tail = float(series[-window:].mean())
        return tail - head

    def fit(self, train_loader) -> "MovingAverageBaseline":
        scores = []
        labels = []
        for batch in train_loader:
            for index in range(batch["signal"].shape[0]):
                length = int(batch["lengths"][index].item())
                signal = batch["signal"][index, :length].numpy()
                scores.append(self._score(signal))
                labels.append(int(batch["label"][index].item()))
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)
        if np.any(labels_arr == 0) and np.any(labels_arr == 1):
            self.threshold = float((scores_arr[labels_arr == 0].mean() + scores_arr[labels_arr == 1].mean()) / 2.0)
        elif len(scores_arr):
            self.threshold = float(scores_arr.mean())
        return self

    def predict_proba(self, loader) -> np.ndarray:
        probabilities = []
        for batch in loader:
            for index in range(batch["signal"].shape[0]):
                length = int(batch["lengths"][index].item())
                signal = batch["signal"][index, :length].numpy()
                score = self._score(signal)
                probabilities.append(float(1.0 / (1.0 + np.exp(-(score - self.threshold)))))
        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, loader) -> np.ndarray:
        return (self.predict_proba(loader) >= 0.5).astype(np.int64)


class SklearnSequenceBaseline:
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, train_loader) -> "SklearnSequenceBaseline":
        features, labels = dataset_to_arrays(train_loader.dataset)
        self.estimator.fit(features, labels)
        return self

    def predict_proba(self, loader) -> np.ndarray:
        features, _ = dataset_to_arrays(loader.dataset)
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(features)[:, 1]
        else:
            raw = self.estimator.decision_function(features)
            probabilities = 1.0 / (1.0 + np.exp(-raw))
        return np.asarray(probabilities, dtype=np.float32)

    def predict(self, loader) -> np.ndarray:
        return (self.predict_proba(loader) >= 0.5).astype(np.int64)


def _fft_features(dataset: IoUTDataset, n_bins: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for item in dataset:
        signal = item["signal"].numpy()
        if signal.ndim == 1:
            signal = signal[:, None]
        flattened = []
        for channel in signal.T:
            spectrum = np.abs(np.fft.rfft(channel))
            flattened.extend(spectrum[:n_bins].tolist())
            flattened.extend([float(channel.mean()), float(channel.std()), float(channel[-1] - channel[0])])
        features.append(np.asarray(flattened, dtype=np.float32))
        labels.append(int(item["label"].item()))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class FFTFeatureClassifier(SklearnSequenceBaseline):
    def __init__(self, seed: int = 42) -> None:
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
        super().__init__(estimator)

    def fit(self, train_loader) -> "FFTFeatureClassifier":
        features, labels = _fft_features(train_loader.dataset)
        self.estimator.fit(features, labels)
        return self

    def predict_proba(self, loader) -> np.ndarray:
        features, _ = _fft_features(loader.dataset)
        return np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=np.float32)


def count_learned_parameters(model: Any) -> int:
    if hasattr(model, "count_parameters"):
        return int(model.count_parameters())
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        classifier = model.named_steps["clf"]
        if hasattr(classifier, "coef_"):
            count = int(np.asarray(classifier.coef_).size)
            if hasattr(classifier, "intercept_"):
                count += int(np.asarray(classifier.intercept_).size)
            return count
    if hasattr(model, "coef_"):
        count = int(np.asarray(model.coef_).size)
        if hasattr(model, "intercept_"):
            count += int(np.asarray(model.intercept_).size)
        return count
    return 0


def build_baseline(model_type: str, seed: int = 42):
    normalized = model_type.lower()
    if normalized == "random":
        return RandomBaseline(seed=seed)
    if normalized in {"majority", "majority_class", "majority-class"}:
        return MajorityClassBaseline()
    if normalized in {"rule", "rule_based", "rule-based"}:
        return RuleBasedBaseline()
    if normalized in {"moving_average", "moving-average", "movingavg"}:
        return MovingAverageBaseline()
    if normalized in {"logistic_regression", "logreg"}:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        return SklearnSequenceBaseline(estimator)
    if normalized in {"random_forest", "rf"}:
        estimator = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
        return SklearnSequenceBaseline(estimator)
    if normalized in {"fft", "fft_classifier", "fft_logreg"}:
        return FFTFeatureClassifier(seed=seed)
    raise ValueError(f"Unsupported baseline/model type: {model_type}")
