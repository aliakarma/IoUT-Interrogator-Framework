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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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

    def _assert_both_classes(split_name: str, split_samples: Sequence[SensorSequence]) -> None:
        labels = [int(sample.label) for sample in split_samples]
        positives = int(sum(labels))
        negatives = int(len(labels) - positives)
        if positives <= 0 or negatives <= 0:
            raise AssertionError(
                f"Invalid split '{split_name}': positives={positives}, negatives={negatives}. "
                "Both classes must be present in train/val/test."
            )

    _assert_both_classes("train", train)
    _assert_both_classes("val", val)
    _assert_both_classes("test", test)

    return overlaps


def save_split_indices(
    samples: Sequence[SensorSequence],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    output_path: str | Path,
) -> None:
    """Save split indices to JSON for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
        "test_indices": [int(i) for i in test_indices],
        "sample_ids": [sample.sensor_id for sample in samples],
        "n_samples": len(samples),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_split_indices(index_path: str | Path) -> Tuple[List[int], List[int], List[int]]:
    """Load pre-saved split indices."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Split index file not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return (
        payload["train_indices"],
        payload["val_indices"],
        payload["test_indices"],
    )


def create_fixed_splits(
    data: Sequence[SensorSequence] | Sequence[Dict[str, Any]],
    output_path: str | Path = "splits/split_v1.json",
    strategy: str = "group",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    temporal_gap: int = 0,
) -> Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]:
    """Create fixed splits and save indices. On subsequent calls, reuse saved indices.
    
    This ensures the SAME test set is used across all seeds.
    
    For time-based splits, temporal_gap prevents sequence overlap in sliding-window datasets.
    """
    output_path = Path(output_path)
    samples = list(data)
    if isinstance(samples[0], dict):
        samples = _records_to_sequences(samples)  # type: ignore[assignment]
    
    # Try to load existing splits
    if output_path.exists():
        try:
            train_idx, val_idx, test_idx = load_split_indices(output_path)
            train = [samples[i] for i in train_idx]
            val = [samples[i] for i in val_idx]
            test = [samples[i] for i in test_idx]
            validate_splits(train, val, test, strategy=strategy)
            return train, val, test
        except AssertionError as exc:
            print(f"[Split Validation] Existing fixed split is invalid ({exc}); regenerating.")
    
    # Create new splits
    train, val, test = create_splits(
        samples,
        strategy=strategy,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        temporal_gap=temporal_gap,
    )
    
    # Map back to indices
    sample_to_index = {id(sample): idx for idx, sample in enumerate(samples)}
    train_indices = [sample_to_index[id(s)] for s in train]
    val_indices = [sample_to_index[id(s)] for s in val]
    test_indices = [sample_to_index[id(s)] for s in test]
    
    # Save for reproducibility
    save_split_indices(samples, train_indices, val_indices, test_indices, output_path)
    
    return train, val, test


def create_splits(
    data: Sequence[SensorSequence] | Sequence[Dict[str, Any]],
    strategy: str = "group",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    temporal_gap: int = 0,
) -> Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]:
    """Create train/val/test splits.
    
    For time-based splits, temporal_gap skips samples between train-val and val-test
    to prevent sequence overlap in sliding-window datasets like UNSW-NB15.
    """
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
            
            # Apply temporal gap to prevent sequence overlap in sliding-window datasets
            train = ordered[:n_train]
            val_start = n_train + temporal_gap
            val = ordered[val_start : val_start + n_val]
            test_start = val_start + n_val + temporal_gap
            test = ordered[test_start:test_start + n_test]
            
            # Validate split boundaries
            if temporal_gap > 0:
                print(f"[Temporal Gap] Split indices: train[0:{n_train}], gap[{n_train}:{val_start}], val[{val_start}:{val_start+len(val)}], gap[{val_start+len(val)}:{test_start}], test[{test_start}:{test_start+len(test)}]")

    elif strategy == "stratified_ordered":
        # Preserve temporal order while ensuring both classes are represented in each split.
        ordered = sorted(samples, key=lambda sample: (_time_key(sample), sample.sensor_id))
        negatives = [sample for sample in ordered if int(sample.label) == 0]
        positives = [sample for sample in ordered if int(sample.label) == 1]

        def _split_one_class(class_samples: List[SensorSequence]) -> Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]:
            count = len(class_samples)
            if count < 3:
                raise ValueError("Need at least 3 samples per class for stratified_ordered split")
            n_train = max(1, int(round(count * train_split)))
            n_val = max(1, int(round(count * val_split)))
            if n_train + n_val >= count:
                n_train = max(1, count - 2)
                n_val = 1
            n_test = count - n_train - n_val
            if n_test <= 0:
                n_test = 1
                n_train = max(1, n_train - 1)

            cls_train = class_samples[:n_train]
            cls_val = class_samples[n_train:n_train + n_val]
            cls_test = class_samples[n_train + n_val:n_train + n_val + n_test]
            return cls_train, cls_val, cls_test

        neg_train, neg_val, neg_test = _split_one_class(negatives)
        pos_train, pos_val, pos_test = _split_one_class(positives)

        train = sorted(neg_train + pos_train, key=lambda sample: (_time_key(sample), sample.sensor_id))
        val = sorted(neg_val + pos_val, key=lambda sample: (_time_key(sample), sample.sensor_id))
        test = sorted(neg_test + pos_test, key=lambda sample: (_time_key(sample), sample.sensor_id))

        if temporal_gap > 0:
            if len(train) > temporal_gap:
                train = train[:-temporal_gap]
            if len(val) > (2 * temporal_gap):
                val = val[temporal_gap:-temporal_gap]
            if len(test) > temporal_gap:
                test = test[temporal_gap:]

    else:
        raise ValueError("strategy must be 'group', 'time', or 'stratified_ordered'")

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
    use_fixed_splits: bool = True,
    splits_path: str | Path = "splits/split_v1.json",
    temporal_gap: int = 0,
    use_weighted_sampler: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any], Tuple[List[SensorSequence], List[SensorSequence], List[SensorSequence]]]:
    samples = list(records) if records and isinstance(records[0], SensorSequence) else _records_to_sequences(records)  # type: ignore[index]
    
    if use_fixed_splits:
        train_samples, val_samples, test_samples = create_fixed_splits(
            samples,
            output_path=splits_path,
            strategy=strategy,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            temporal_gap=temporal_gap,
        )
    else:
        train_samples, val_samples, test_samples = create_splits(
            samples,
            strategy=strategy,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            temporal_gap=temporal_gap,
        )

    # ===== CRITICAL: Fit normalizer ONLY on training data =====
    normalizer = None
    if normalize:
        # Extract signals from training samples only
        train_signals = [sample.signals for sample in train_samples]
        normalizer = SequenceNormalizer().fit(train_signals)
        # Verify normalizer was fit on training data
        assert hasattr(normalizer, 'mean_') and hasattr(normalizer, 'std_'), \
            "Normalizer fit failed: missing mean_/std_ attributes"
        assert normalizer.mean_ is not None and normalizer.std_ is not None, \
            "Normalizer fit failed: mean_/std_ are None"
        print(f"[Scaler Validation] ✓ Normalizer fitted on {len(train_samples)} training sequences only")
        print(f"[Scaler Validation] ✓ Mean shape: {normalizer.mean_.shape}, Std shape: {normalizer.std_.shape}")

    train_dataset = IoUTDataset(train_samples, seq_len=seq_len, normalizer=normalizer)
    val_dataset = IoUTDataset(val_samples, seq_len=seq_len, normalizer=normalizer)
    test_dataset = IoUTDataset(test_samples, seq_len=seq_len, normalizer=normalizer)

    train_labels_array = np.asarray([int(sample.label) for sample in train_samples], dtype=np.int64)
    class_counts = np.bincount(train_labels_array, minlength=2)

    train_sampler = None
    if use_weighted_sampler and len(train_labels_array) > 0 and np.all(class_counts > 0):
        class_weights = (1.0 / class_counts.astype(np.float64)) ** 1.5
        sample_weights = class_weights[train_labels_array]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=_collate_fn(seq_len),
        ),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn(seq_len)),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn(seq_len)),
    }

    # ===== DEBUG: Class distribution validation =====
    train_labels = np.array([int(s.label) for s in train_samples])
    val_labels = np.array([int(s.label) for s in val_samples])
    test_labels = np.array([int(s.label) for s in test_samples])
    
    print(f"\n[Class Distribution]")
    print(f"  Train: {train_labels.sum()} anomalies, {(1-train_labels).sum()} normal (pos_rate={train_labels.mean():.4f})")
    print(f"  Val:   {val_labels.sum()} anomalies, {(1-val_labels).sum()} normal (pos_rate={val_labels.mean():.4f})")
    print(f"  Test:  {test_labels.sum()} anomalies, {(1-test_labels).sum()} normal (pos_rate={test_labels.mean():.4f})")
    
    # Warn if significant class distribution shift
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    overall_pos_rate = all_labels.mean()
    train_shift = abs(train_labels.mean() - overall_pos_rate)
    val_shift = abs(val_labels.mean() - overall_pos_rate)
    test_shift = abs(test_labels.mean() - overall_pos_rate)
    
    if train_shift > 0.05 or val_shift > 0.05 or test_shift > 0.05:
        print(f"⚠️  [Class Distribution Warning] Significant shift detected:")
        print(f"   Overall pos_rate: {overall_pos_rate:.4f}")
        print(f"   Train shift: {train_shift:.4f}  |  Val shift: {val_shift:.4f}  |  Test shift: {test_shift:.4f}")

    feature_dim = int(train_samples[0].signals.shape[-1]) if train_samples else int(samples[0].signals.shape[-1])
    dataset_statistics = compute_dataset_statistics(samples)

    metadata = {
        "input_dim": feature_dim,
        "split_strategy": strategy,
        "temporal_gap": temporal_gap,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "train_sensor_ids": [sample.sensor_id for sample in train_samples],
        "val_sensor_ids": [sample.sensor_id for sample in val_samples],
        "test_sensor_ids": [sample.sensor_id for sample in test_samples],
        "train_pos_rate": float(train_labels.mean()),
        "val_pos_rate": float(val_labels.mean()),
        "test_pos_rate": float(test_labels.mean()),
        "train_class_counts": {
            "0": int(class_counts[0]) if len(class_counts) > 0 else 0,
            "1": int(class_counts[1]) if len(class_counts) > 1 else 0,
        },
        "weighted_sampler": bool(train_sampler is not None),
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
