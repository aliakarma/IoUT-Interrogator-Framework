from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data.real_dataset_adapter import build_sequences, extract_behavioral_features, normalize_features


UNSW_REQUIRED_COLUMNS = {"dur", "spkts", "ct_dst_ltm", "ct_srv_src", "state"}


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

    df = pd.read_csv(path)
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
