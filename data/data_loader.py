from __future__ import annotations

import json
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SensorSequence:
    sensor_id: str
    timestamps: np.ndarray
    signals: np.ndarray
    label: int


class SequenceNormalizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, sequences: Sequence[np.ndarray]) -> "SequenceNormalizer":
        stacked = np.concatenate([np.asarray(seq, dtype=np.float32) for seq in sequences], axis=0)
        if stacked.ndim == 1:
            stacked = stacked[:, None]
        self.mean_ = stacked.mean(axis=0)
        self.std_ = stacked.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fit before transform().")
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return (arr - self.mean_) / self.std_


def _coerce_signal(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return np.asarray([float(value)], dtype=np.float32)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(source: str | Path | Sequence[dict], file_format: str = "auto") -> List[Dict[str, Any]]:
    if isinstance(source, (list, tuple)):
        raw = list(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        inferred_format = file_format.lower()
        if inferred_format == "auto":
            inferred_format = path.suffix.lower().lstrip(".")
        if inferred_format == "csv":
            raw = pd.read_csv(path).to_dict(orient="records")
        elif inferred_format == "json":
            raw = _read_json(path)
        else:
            raise ValueError(f"Unsupported dataset format: {file_format}")

    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("data", []))

    records: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if "sequence" in item:
            records.append(
                {
                    "sensor_id": str(item.get("sensor_id", item.get("agent_id", index))),
                    "timestamp": float(item.get("timestamp", index)),
                    "sequence": item["sequence"],
                    "label": int(item.get("label", 0)),
                }
            )
            continue

        missing = {key for key in ("sensor_id", "timestamp", "signal") if key not in item}
        if missing:
            raise ValueError(f"Missing required fields {sorted(missing)} in record {index}.")
        record = {
            "sensor_id": str(item["sensor_id"]),
            "timestamp": float(item["timestamp"]),
            "signal": item["signal"],
        }
        if "label" in item and item["label"] is not None:
            record["label"] = int(item["label"])
        records.append(record)
    return records


def _records_to_sequences(records: Sequence[Dict[str, Any]]) -> List[SensorSequence]:
    sequences: List[SensorSequence] = []

    for item in records:
        if "sequence" not in item:
            continue
        signal = np.asarray(item["sequence"], dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[:, None]
        timestamps = np.arange(signal.shape[0], dtype=np.float32)
        sequences.append(SensorSequence(str(item["sensor_id"]), timestamps, signal, int(item.get("label", 0))))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        if "sequence" in item:
            continue
        grouped[str(item["sensor_id"])].append(item)

    for sensor_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (float(row["timestamp"]), str(row["sensor_id"])))
        signals = np.stack([_coerce_signal(row["signal"]) for row in ordered], axis=0)
        labels = [int(row.get("label", ordered[0].get("label", 0))) for row in ordered if row.get("label") is not None]
        if not labels:
            raise ValueError(f"Sensor group {sensor_id} is missing labels.")
        if len(set(labels)) > 1:
            raise ValueError(f"Conflicting labels found within sensor group {sensor_id}.")
        timestamps = np.asarray([float(row["timestamp"]) for row in ordered], dtype=np.float32)
        sequences.append(SensorSequence(sensor_id, timestamps, signals, int(labels[0])))

    if not sequences:
        raise ValueError("No sequences could be built from the provided records.")
    return sequences


def compute_dataset_statistics(samples: Sequence[SensorSequence]) -> Dict[str, Any]:
    labels = [int(sample.label) for sample in samples]
    lengths = [int(sample.signals.shape[0]) for sample in samples]
    label_counts = {int(key): int(value) for key, value in Counter(labels).items()}
    total = max(len(labels), 1)
    imbalance_ratio = max(label_counts.values()) / total if label_counts else 0.0
    stats = {
        "n_samples": int(len(samples)),
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "std": float(np.std(lengths)) if lengths else 0.0,
        },
        "class_balance": label_counts,
        "imbalance_ratio": float(imbalance_ratio),
    }
    if imbalance_ratio > 0.80:
        warnings.warn(
            (
                "Dataset imbalance warning: dominant class exceeds 80%. "
                "Consider class-weighting, resampling, or threshold calibration."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return stats


def _sensor_overlap(a: Sequence[SensorSequence], b: Sequence[SensorSequence]) -> int:
    return len({sample.sensor_id for sample in a} & {sample.sensor_id for sample in b})


def _time_key(sample: SensorSequence) -> float:
    return float(sample.timestamps[-1]) if len(sample.timestamps) else float("inf")


def validate_splits(
    train: Sequence[SensorSequence],
    val: Sequence[SensorSequence],
    test: Sequence[SensorSequence],
    strategy: str,
) -> Dict[str, int]:
    overlaps = {
        "train_val": _sensor_overlap(train, val),
        "train_test": _sensor_overlap(train, test),
        "val_test": _sensor_overlap(val, test),
    }
    print(
        f"Split validation ({strategy}): train={len(train)} val={len(val)} test={len(test)} | "
        f"overlap(train,val)={overlaps['train_val']} overlap(train,test)={overlaps['train_test']} "
        f"overlap(val,test)={overlaps['val_test']}"
    )
    if any(value != 0 for value in overlaps.values()):
        raise AssertionError(f"Leakage detected across splits: {overlaps}")
    if strategy == "time":
        train_max = max((_time_key(sample) for sample in train), default=float("-inf"))
        val_min = min((_time_key(sample) for sample in val), default=float("inf"))
        test_min = min((_time_key(sample) for sample in test), default=float("inf"))
        if not (train_max <= val_min <= test_min):
            raise AssertionError(
                f"Temporal leakage detected: train_max={train_max}, val_min={val_min}, test_min={test_min}"
            )
    return overlaps


def create_splits(
    data: Sequence[SensorSequence] | Sequence[Dict[str, Any]],
    strategy: str = "group",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]:
    samples = list(data)
    if not samples:
        raise ValueError("Cannot split an empty dataset.")
    if isinstance(samples[0], dict):
        samples = _records_to_sequences(samples)  # type: ignore[assignment]

    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    strategy = strategy.lower().strip()
    rng = random.Random(seed)

    if strategy == "group":
        by_label: Dict[int, List[SensorSequence]] = defaultdict(list)
        for sample in samples:
            by_label[int(sample.label)].append(sample)

        train: List[SensorSequence] = []
        val: List[SensorSequence] = []
        test: List[SensorSequence] = []
        for label_samples in by_label.values():
            ordered = sorted(label_samples, key=lambda sample: sample.sensor_id)
            rng.shuffle(ordered)
            count = len(ordered)
            if count <= 2:
                train.extend(ordered[:1])
                val.extend(ordered[1:2])
                continue
            n_train = max(1, int(round(count * train_split)))
            n_val = max(1, int(round(count * val_split)))
            if n_train + n_val >= count:
                n_train = max(1, count - 2)
                n_val = 1
            n_test = count - n_train - n_val
            if n_test <= 0:
                n_test = 1
                n_train = max(1, n_train - 1)
            train.extend(ordered[:n_train])
            val.extend(ordered[n_train : n_train + n_val])
            test.extend(ordered[n_train + n_val : n_train + n_val + n_test])

    elif strategy == "time":
        ordered = sorted(samples, key=lambda sample: (_time_key(sample), sample.sensor_id))
        count = len(ordered)
        if count <= 2:
            train = ordered[:1]
            val = ordered[1:2]
            test = ordered[2:]
        else:
            n_train = max(1, int(round(count * train_split)))
            n_val = max(1, int(round(count * val_split)))
            if n_train + n_val >= count:
                n_train = max(1, count - 2)
                n_val = 1
            n_test = count - n_train - n_val
            if n_test <= 0:
                n_test = 1
                n_train = max(1, n_train - 1)
            train = ordered[:n_train]
            val = ordered[n_train : n_train + n_val]
            test = ordered[n_train + n_val : n_train + n_val + n_test]
    else:
        raise ValueError("strategy must be 'group' or 'time'")

    validate_splits(train, val, test, strategy=strategy)
    return train, val, test


class IoUTDataset(Dataset):
    def __init__(self, sequences: Sequence[SensorSequence], seq_len: int = 64, normalizer: SequenceNormalizer | None = None):
        self.sequences = list(sequences)
        self.seq_len = int(seq_len)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sequence = self.sequences[index]
        signal = np.asarray(sequence.signals, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[:, None]
        if self.normalizer is not None:
            signal = self.normalizer.transform(signal)
        if len(signal) > self.seq_len:
            signal = signal[: self.seq_len]
        return {
            "sensor_id": sequence.sensor_id,
            "timestamps": torch.tensor(sequence.timestamps[: len(signal)], dtype=torch.float32),
            "signal": torch.tensor(signal, dtype=torch.float32),
            "label": torch.tensor(float(sequence.label), dtype=torch.float32),
            "length": torch.tensor(int(signal.shape[0]), dtype=torch.long),
        }


def _collate_fn(seq_len: int):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = min(seq_len, max(item["signal"].shape[0] for item in batch))
        feature_dim = batch[0]["signal"].shape[-1]
        signals = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32)
        lengths = torch.zeros((len(batch),), dtype=torch.long)
        labels = torch.zeros((len(batch),), dtype=torch.float32)
        sensor_ids: List[str] = []
        for row_index, item in enumerate(batch):
            seq = item["signal"][:max_len]
            current_len = seq.shape[0]
            signals[row_index, :current_len] = seq
            lengths[row_index] = current_len
            labels[row_index] = item["label"]
            sensor_ids.append(item["sensor_id"])
        return {"sensor_id": sensor_ids, "signal": signals, "lengths": lengths, "label": labels}

    return collate


def build_dataloaders(
    records: Sequence[Dict[str, Any]] | Sequence[SensorSequence],
    seq_len: int,
    train_split: float,
    val_split: float,
    test_split: float,
    batch_size: int,
    num_workers: int,
    normalize: bool,
    seed: int,
    strategy: str = "group",
) -> Tuple[Dict[str, DataLoader], Dict[str, Any], Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]]:
    samples = list(records) if records and isinstance(records[0], SensorSequence) else _records_to_sequences(records)  # type: ignore[index]
    train_samples, val_samples, test_samples = create_splits(
        samples,
        strategy=strategy,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    normalizer = SequenceNormalizer().fit([sample.signals for sample in train_samples]) if normalize else None

    train_dataset = IoUTDataset(train_samples, seq_len=seq_len, normalizer=normalizer)
    val_dataset = IoUTDataset(val_samples, seq_len=seq_len, normalizer=normalizer)
    test_dataset = IoUTDataset(test_samples, seq_len=seq_len, normalizer=normalizer)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_fn(seq_len)),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn(seq_len)),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn(seq_len)),
    }

    feature_dim = int(train_samples[0].signals.shape[-1]) if train_samples else int(samples[0].signals.shape[-1])
    dataset_statistics = compute_dataset_statistics(samples)

    metadata = {
        "input_dim": feature_dim,
        "split_strategy": strategy,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "train_sensor_ids": [sample.sensor_id for sample in train_samples],
        "val_sensor_ids": [sample.sensor_id for sample in val_samples],
        "test_sensor_ids": [sample.sensor_id for sample in test_samples],
        "dataset_statistics": dataset_statistics,
    }
    return loaders, metadata, (train_samples, val_samples, test_samples)


def dataset_to_arrays(dataset: IoUTDataset) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    for item in dataset:
        signal = item["signal"].numpy()
        if signal.ndim == 1:
            signal = signal[:, None]
        summary = np.concatenate(
            [signal.mean(axis=0), signal.std(axis=0), signal.min(axis=0), signal.max(axis=0), signal[-1] - signal[0]]
        )
        features.append(summary.reshape(-1))
        labels.append(int(item["label"].item()))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def inject_gaussian_noise(samples: Sequence[SensorSequence], sigma: float, seed: int) -> List[SensorSequence]:
    rng = np.random.default_rng(seed)
    noisy: List[SensorSequence] = []
    for sample in samples:
        signals = np.asarray(sample.signals, dtype=np.float32)
        noise = rng.normal(0.0, sigma, size=signals.shape).astype(np.float32)
        noisy.append(SensorSequence(sample.sensor_id, sample.timestamps.copy(), signals + noise, sample.label))
    return noisy


def drop_missing_segments(samples: Sequence[SensorSequence], drop_fraction: float, seed: int) -> List[SensorSequence]:
    rng = np.random.default_rng(seed)
    masked: List[SensorSequence] = []
    for sample in samples:
        signals = np.asarray(sample.signals, dtype=np.float32).copy()
        total = signals.shape[0]
        n_drop = max(1, int(round(total * drop_fraction))) if total else 0
        if n_drop > 0:
            start = int(rng.integers(0, max(total - n_drop + 1, 1)))
            signals[start : start + n_drop] = 0.0
        masked.append(SensorSequence(sample.sensor_id, sample.timestamps.copy(), signals, sample.label))
    return masked


def simulate_sensor_failure(samples: Sequence[SensorSequence], failure_fraction: float, seed: int) -> List[SensorSequence]:
    rng = np.random.default_rng(seed)
    sensor_ids = [sample.sensor_id for sample in samples]
    n_fail = max(1, int(round(len(sensor_ids) * failure_fraction))) if sensor_ids else 0
    failed_ids = set(rng.choice(sensor_ids, size=min(n_fail, len(sensor_ids)), replace=False).tolist()) if n_fail > 0 else set()
    failed: List[SensorSequence] = []
    for sample in samples:
        signals = np.zeros_like(sample.signals, dtype=np.float32) if sample.sensor_id in failed_ids else np.asarray(sample.signals, dtype=np.float32)
        failed.append(SensorSequence(sample.sensor_id, sample.timestamps.copy(), signals, sample.label))
    return failed
