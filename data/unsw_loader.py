from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data.real_dataset_adapter import build_sequences, extract_behavioral_features, normalize_features


UNSW_REQUIRED_COLUMNS = {"dur", "spkts", "ct_dst_ltm", "ct_srv_src", "state"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip().lower() for col in df.columns}
    return df.rename(columns=renamed)


def _discover_csv_inputs(source_path: Path) -> List[Path]:
    """Resolve a user path into one or more UNSW CSV files.

    Supported layouts:
    - Direct file path (single CSV)
    - Directory containing "Training and Testing Sets" with training/testing CSVs
    - Directory containing UNSW-NB15_1.csv ... UNSW-NB15_4.csv shards
    """
    if source_path.is_file():
        return [source_path]

    if not source_path.is_dir():
        raise FileNotFoundError(f"UNSW-NB15 path not found: {source_path}")

    training_testing_candidates = sorted(
        [
            p
            for p in source_path.rglob("*.csv")
            if "training" in p.name.lower() or "testing" in p.name.lower()
        ]
    )
    if training_testing_candidates:
        return training_testing_candidates

    shard_candidates = sorted(
        [
            p
            for p in source_path.glob("*.csv")
            if p.stem.lower().startswith("unsw-nb15_") and p.stem[-1:].isdigit()
        ]
    )
    if shard_candidates:
        return shard_candidates

    raise ValueError(
        "No UNSW CSV files found. Provide either a training/testing CSV, "
        "a directory with training/testing files, or UNSW-NB15_1..4 shards."
    )


def _maybe_attach_gt_labels(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Attach labels from GT CSV only when direct row-wise alignment is possible."""
    if "label" in df.columns or "attack_cat" in df.columns:
        return df

    search_root = source_path.parent if source_path.is_file() else source_path
    gt_files = sorted([p for p in search_root.rglob("*.csv") if "gt" in p.stem.lower()])
    if not gt_files:
        return df

    gt_df = _normalize_columns(pd.read_csv(gt_files[0]))
    label_column = None
    for candidate in ["label", "attack_cat"]:
        if candidate in gt_df.columns:
            label_column = candidate
            break
    if label_column is None:
        return df

    if len(gt_df) != len(df):
        warnings.warn(
            (
                "GT file found but row count does not match UNSW data rows; "
                "ignoring GT labels to avoid unsafe joins."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return df

    df = df.copy()
    df[label_column] = gt_df[label_column].to_numpy()
    return df


def _extract_binary_labels(df: pd.DataFrame) -> np.ndarray:
    if "label" in df.columns:
        raw = df["label"]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            return (numeric.to_numpy(dtype=np.float32) > 0.0).astype(np.int64)
        text = raw.fillna("normal").astype(str).str.lower()
        return (~text.isin({"normal", "0", "benign"})).astype(np.int64).to_numpy()

    if "attack_cat" in df.columns:
        cat = df["attack_cat"].fillna("Normal").astype(str).str.lower()
        return (~cat.isin({"normal", "benign"})).astype(np.int64).to_numpy()

    raise ValueError("UNSW-NB15 CSV must contain either 'label' or 'attack_cat' for label extraction.")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(UNSW_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"UNSW-NB15 CSV is missing required columns: {missing}")


def load_unsw_nb15_records(
    source: str | Path,
    seq_len: int = 20,
    max_rows: int | None = None,
) -> List[Dict[str, Any]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"UNSW-NB15 CSV not found: {path}")

    csv_files = _discover_csv_inputs(path)
    frames = [_normalize_columns(pd.read_csv(file_path)) for file_path in csv_files]
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = _maybe_attach_gt_labels(df, source_path=path)

    if max_rows is not None and max_rows > 0:
        df = df.head(int(max_rows)).copy()

    _validate_columns(df)
    y = _extract_binary_labels(df)

    X = extract_behavioral_features(df)
    sequences, labels = build_sequences(X, y, seq_len=seq_len)

    if len(sequences) == 0:
        raise ValueError(
            f"Not enough rows ({len(X)}) to build sequences with seq_len={seq_len}."
        )

    # Normalize each sequence independently to avoid cross-split leakage.
    normalized_sequences, _ = normalize_features(sequences)

    records: List[Dict[str, Any]] = []
    for index, (sequence, label) in enumerate(zip(normalized_sequences, labels)):
        records.append(
            {
                "sensor_id": f"unsw_flow_{index:07d}",
                "timestamp": float(index),
                "sequence": sequence.tolist(),
                "label": int(label),
            }
        )
    return records
