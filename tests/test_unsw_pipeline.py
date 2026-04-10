from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "unsw_final_balanced"


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / "summary.csv")


def _metric_value(summary: pd.DataFrame, metric: str) -> float:
    return float(summary.loc[summary["metric"] == metric, "mean"].iloc[0])


def test_split_has_both_classes() -> None:
    split_stats = json.loads((RESULT_DIR / "split_stats.json").read_text(encoding="utf-8"))
    for split_name in ["train", "val", "test"]:
        split = split_stats[split_name]
        assert int(split["pos"]) > 0
        assert int(split["neg"]) > 0


def test_metrics_valid_range() -> None:
    summary = _load_summary()
    f1 = _metric_value(summary, "f1")
    roc_auc = _metric_value(summary, "roc_auc")
    balanced_accuracy = _metric_value(summary, "balanced_accuracy")

    assert 0.5 < f1 < 0.95
    assert 0.7 < roc_auc <= 1.0
    assert 0.6 < balanced_accuracy <= 1.0


def test_no_leakage_flag() -> None:
    checks = json.loads((RESULT_DIR / "validation_checks.json").read_text(encoding="utf-8"))
    assert checks.get("no_leakage_warnings") is True
