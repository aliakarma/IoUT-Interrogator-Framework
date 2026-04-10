from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "unsw_smoke.yaml"
OUT_DIR = ROOT / "results" / "unsw_smoke_fixed"
ITER_DIR = OUT_DIR / "iterations"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_pipeline(config_path: Path) -> None:
    cmd = ["python", "run_pipeline.py", "--config", str(config_path)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate_conditions(metrics: Dict[str, Any], run_summary: Dict[str, Any]) -> Dict[str, Any]:
    metadata = run_summary.get("metadata", {})
    n_train = int(metadata.get("n_train", 0))
    n_val = int(metadata.get("n_val", 0))
    n_test = int(metadata.get("n_test", 0))
    train_pos = float(metadata.get("train_pos_rate", 0.0)) * n_train
    val_pos = float(metadata.get("val_pos_rate", 0.0)) * n_val
    test_pos = float(metadata.get("test_pos_rate", 0.0)) * n_test

    class_dist = metrics.get("class_distribution", {})
    test_pos_count = int(class_dist.get("1", class_dist.get(1, 0)))
    test_neg_count = int(class_dist.get("0", class_dist.get(0, 0)))

    checks = {
        "train_both_classes": bool(train_pos > 0 and (n_train - train_pos) > 0),
        "val_both_classes": bool(val_pos > 0 and (n_val - val_pos) > 0),
        "test_both_classes": bool(test_pos_count > 0 and test_neg_count > 0),
        "roc_auc_gt_07": float(metrics.get("roc_auc", 0.0)) > 0.7,
        "pr_auc_gt_05": float(metrics.get("pr_auc", 0.0)) > 0.5,
        "f1_lt_098": float(metrics.get("f1", 0.0)) < 0.98,
        "no_leakage_warning": bool(
            float(metrics.get("f1", 0.0)) <= 0.98 and float(metrics.get("roc_auc", 0.0)) <= 0.995
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    base = load_yaml(BASE_CONFIG)
    iteration_log: List[Dict[str, Any]] = []

    # Iteration schedule: keep split strategy fixed, increase epochs if needed.
    epochs_schedule = [1, 2, 3]

    for idx, epochs in enumerate(epochs_schedule, start=1):
        iter_name = f"iter_{idx:02d}"
        iter_out = ITER_DIR / iter_name
        iter_out.mkdir(parents=True, exist_ok=True)

        config = load_yaml(BASE_CONFIG)
        config.setdefault("data", {})["split_strategy"] = "stratified_ordered"
        config.setdefault("data", {})["use_fixed_splits"] = False
        config.setdefault("data", {})["temporal_gap"] = 20
        config.setdefault("evaluation", {})["tune_threshold"] = True
        config.setdefault("evaluation", {})["threshold_metric"] = "f1"
        config.setdefault("training", {})["epochs"] = int(epochs)
        config.setdefault("training", {})["early_stopping_patience"] = 1
        config.setdefault("training", {})["results_dir"] = str(iter_out.relative_to(ROOT)).replace("\\", "/")

        iter_config = iter_out / "config.yaml"
        save_yaml(iter_config, config)

        run_pipeline(iter_config)

        metrics_path = iter_out / "metrics.json"
        summary_path = iter_out / "run_summary.json"

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        checks = evaluate_conditions(metrics, summary)
        entry = {
            "iteration": idx,
            "epochs": epochs,
            "results_dir": str(iter_out.relative_to(ROOT)).replace("\\", "/"),
            "metrics": {
                "f1": float(metrics.get("f1", 0.0)),
                "roc_auc": float(metrics.get("roc_auc", 0.0)),
                "pr_auc": float(metrics.get("pr_auc", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                "threshold": float(metrics.get("threshold", 0.5)),
            },
            "checks": checks,
        }
        iteration_log.append(entry)

        if checks["all_pass"]:
            shutil.copy2(metrics_path, OUT_DIR / "metrics.json")
            shutil.copy2(summary_path, OUT_DIR / "run_summary.json")
            shutil.copy2(iter_out / "predictions.json", OUT_DIR / "predictions.json")
            shutil.copy2(iter_out / "pr_curve.json", OUT_DIR / "pr_curve.json")
            with (OUT_DIR / "iteration_log.json").open("w", encoding="utf-8") as handle:
                json.dump(iteration_log, handle, indent=2)
            write_report(iteration_log, OUT_DIR)
            print("[OK] All validation checks passed.")
            return

    with (OUT_DIR / "iteration_log.json").open("w", encoding="utf-8") as handle:
        json.dump(iteration_log, handle, indent=2)
    write_report(iteration_log, OUT_DIR)
    raise SystemExit("Validation criteria not met after all iterations.")


def write_report(iteration_log: List[Dict[str, Any]], out_dir: Path) -> None:
    latest = iteration_log[-1]
    passed = bool(latest.get("checks", {}).get("all_pass", False))

    lines: List[str] = []
    lines.append("# UNSW_PIPELINE_FINAL_REPORT")
    lines.append("")
    lines.append(f"Final status: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("## Audit Trail")
    lines.append("")
    for item in iteration_log:
        m = item["metrics"]
        c = item["checks"]
        lines.append(
            f"- Iteration {item['iteration']} (epochs={item['epochs']}): "
            f"F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
            f"BalancedAcc={m['balanced_accuracy']:.4f}, threshold={m['threshold']:.2f}, "
            f"all_pass={c['all_pass']}"
        )

    lines.append("")
    lines.append("## Validation Checks")
    lines.append("")
    for name, ok in latest["checks"].items():
        lines.append(f"- {name}: {ok}")

    lines.append("")
    lines.append("## Scientific Validity Notes")
    lines.append("")
    lines.append("- Split strategy: stratified_ordered")
    lines.append("- Temporal gap: 20 sequences")
    lines.append("- Threshold sweep: 0.2 to 0.8 (validation F1)")
    lines.append("- Scaler fit: training split only")
    lines.append("- Leakage flags: F1 > 0.98 or ROC-AUC > 0.995")

    report_path = out_dir / "UNSW_PIPELINE_FINAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
