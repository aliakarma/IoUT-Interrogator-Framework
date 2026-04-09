from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_pipeline import build_model, evaluate, load_config, load_data, train
from training.trainer import set_global_seed


BASELINE_MODELS = {"random", "majority", "moving_average", "rule", "logistic_regression", "random_forest", "fft"}


def _apply_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if model_name in {"gru", "lstm", "temporal_cnn", "tcnn", "cnn", "transformer_light", "light_transformer", "transformer", "hybrid_temporal", "hybrid_cnn", "hybrid"}:
        cfg.setdefault("model", {})["type"] = model_name
    else:
        cfg.setdefault("model", {})["type"] = "baseline"
        cfg.setdefault("baseline", {})["name"] = model_name
    return cfg


def _metrics_row(model_name: str, seed: int, train_summary: Dict[str, Any], eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": model_name,
        "seed": seed,
        "accuracy": float(eval_summary.get("accuracy", 0.0)),
        "balanced_accuracy": float(eval_summary.get("balanced_accuracy", 0.0)),
        "precision": float(eval_summary.get("precision", 0.0)),
        "recall": float(eval_summary.get("recall", 0.0)),
        "f1": float(eval_summary.get("f1", 0.0)),
        "roc_auc": float(eval_summary.get("roc_auc", 0.0)),
        "pr_auc": float(eval_summary.get("pr_auc", 0.0)),
        "params": int(train_summary.get("params", 0)),
        "runtime_seconds": float(train_summary.get("runtime_seconds", 0.0)),
    }


def _aggregate(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "runtime_seconds"]:
        values = frame[metric].astype(float).tolist()
        result[metric] = {
            "mean": float(mean(values)) if values else 0.0,
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        }
    result["params"] = {"mean": float(frame["params"].mean()) if len(frame) else 0.0, "std": 0.0}
    return result


def _find_best_baseline(model_summaries: Dict[str, Dict[str, Any]]) -> str | None:
    baseline_names = [name for name in model_summaries if name in BASELINE_MODELS]
    if not baseline_names:
        return None
    return max(baseline_names, key=lambda name: model_summaries[name]["aggregate"]["f1"]["mean"])


def _compute_statistical_tests(all_runs: pd.DataFrame, best_baseline: str | None) -> Dict[str, Any]:
    tests = {
        "gru_vs_baseline": {"p_value": None, "test_type": "paired_t_test", "baseline": best_baseline},
        "lstm_vs_baseline": {"p_value": None, "test_type": "paired_t_test", "baseline": best_baseline},
        "temporal_cnn_vs_baseline": {"p_value": None, "test_type": "paired_t_test", "baseline": best_baseline},
        "transformer_light_vs_baseline": {"p_value": None, "test_type": "paired_t_test", "baseline": best_baseline},
        "hybrid_temporal_vs_baseline": {"p_value": None, "test_type": "paired_t_test", "baseline": best_baseline},
    }
    if not best_baseline:
        return tests

    baseline_scores = all_runs[all_runs["model"] == best_baseline]["f1"].astype(float).to_numpy()
    for target in ["gru", "lstm", "temporal_cnn", "transformer_light", "hybrid_temporal"]:
        if target not in set(all_runs["model"].tolist()):
            continue
        target_scores = all_runs[all_runs["model"] == target]["f1"].astype(float).to_numpy()
        # Paired t-test: same number of samples (same seeds), aligned evaluation
        if len(target_scores) >= 2 and len(baseline_scores) >= 2 and len(target_scores) == len(baseline_scores):
            p_value = float(ttest_rel(target_scores, baseline_scores, alternative='two-sided').pvalue)
            tests[f"{target}_vs_baseline"]["p_value"] = p_value
            tests[f"{target}_vs_baseline"]["mean_diff"] = float(np.mean(target_scores) - np.mean(baseline_scores))
            tests[f"{target}_vs_baseline"]["std_diff"] = float(np.std(target_scores - baseline_scores))
    return tests


def _criteria_report(model_summaries: Dict[str, Dict[str, Any]], tests_payload: Dict[str, Any], best_baseline: str | None) -> Dict[str, Any]:
    neural_candidates = [
        name
        for name in model_summaries
        if name in {"gru", "lstm", "temporal_cnn", "tcnn", "cnn", "transformer_light", "light_transformer", "transformer", "hybrid_temporal", "hybrid_cnn", "hybrid"}
    ]
    if not neural_candidates:
        return {"status": "failed", "reason": "No neural candidate models evaluated."}

    best_model = max(neural_candidates, key=lambda name: model_summaries[name]["aggregate"]["f1"]["mean"])
    best_metrics = model_summaries[best_model]["aggregate"]
    baseline_f1_values = [
        model_summaries[name]["aggregate"]["f1"]["mean"]
        for name in model_summaries
        if name in BASELINE_MODELS
    ]
    best_baseline_f1 = max(baseline_f1_values) if baseline_f1_values else 0.0
    margin = float(best_metrics["f1"]["mean"] - best_baseline_f1)

    p_value = None
    if best_model == "gru":
        p_value = tests_payload.get("gru_vs_baseline", {}).get("p_value")
    elif best_model == "lstm":
        p_value = tests_payload.get("lstm_vs_baseline", {}).get("p_value")
    elif best_model in {"temporal_cnn", "tcnn", "cnn"}:
        p_value = tests_payload.get("temporal_cnn_vs_baseline", {}).get("p_value")
    elif best_model in {"transformer_light", "light_transformer", "transformer"}:
        p_value = tests_payload.get("transformer_light_vs_baseline", {}).get("p_value")
    elif best_model in {"hybrid_temporal", "hybrid_cnn", "hybrid"}:
        p_value = tests_payload.get("hybrid_temporal_vs_baseline", {}).get("p_value")

    checks = {
        "f1_ge_0_75": bool(best_metrics["f1"]["mean"] >= 0.75),
        "balanced_accuracy_ge_0_80": bool(best_metrics["balanced_accuracy"]["mean"] >= 0.80),
        "roc_auc_ge_0_90": bool(best_metrics["roc_auc"]["mean"] >= 0.90),
        "f1_std_le_0_10": bool(best_metrics["f1"]["std"] <= 0.10),
        "p_value_lt_0_05": bool(p_value is not None and p_value < 0.05),
        "baseline_margin_gt_0_05": bool(margin > 0.05),
    }

    return {
        "best_model": best_model,
        "best_baseline": best_baseline,
        "best_model_metrics": best_metrics,
        "best_baseline_f1": best_baseline_f1,
        "f1_margin_vs_best_baseline": margin,
        "p_value": p_value,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed IoUT benchmark experiments with statistical testing")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Single model shortcut (overrides --models)")
    parser.add_argument("--models", default="gru,lstm,temporal_cnn,fft,majority,moving_average")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", default="results/multi_seed")
    parser.add_argument("--aggregate-output", default="results/aggregate_metrics.json")
    parser.add_argument("--tests-output", default="results/statistical_tests.json")
    parser.add_argument("--final-table-output", default="results/final_results_table.csv")
    args = parser.parse_args()

    base_config = load_config(args.config)
    model_names = [args.model] if args.model else [token.strip() for token in args.models.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    model_summaries: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_rows: List[Dict[str, Any]] = []

        for seed in seeds:
            run_config = _apply_model_config(base_config, model_name)
            run_config.setdefault("training", {})["seed"] = seed
            run_config["training"]["results_dir"] = str(model_output_dir / f"seed_{seed}")
            set_global_seed(seed)

            loaders, metadata = load_data(run_config)
            model = build_model(run_config, input_dim=int(metadata["input_dim"]))
            trained_model, train_summary = train(model, loaders, run_config, metadata)
            eval_summary = evaluate(trained_model, loaders, run_config, metadata)

            row = _metrics_row(model_name, seed, train_summary, eval_summary)
            model_rows.append(row)
            all_rows.append(row)

            seed_dir = model_output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            with (seed_dir / "run_result.json").open("w", encoding="utf-8") as handle:
                json.dump({"model": model_name, "seed": seed, "train": train_summary, "eval": eval_summary, "metadata": metadata}, handle, indent=2)

        model_frame = pd.DataFrame(model_rows)
        model_frame.to_csv(model_output_dir / "multi_seed_results.csv", index=False)
        model_summaries[model_name] = {
            "aggregate": _aggregate(model_frame),
            "n_seeds": len(seeds),
        }

    all_frame = pd.DataFrame(all_rows)
    all_frame.to_csv(output_dir / "all_models_multi_seed_results.csv", index=False)

    aggregate_payload = {
        "models": model_summaries,
        "seeds": seeds,
    }
    Path(args.aggregate_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.aggregate_output).open("w", encoding="utf-8") as handle:
        json.dump(aggregate_payload, handle, indent=2)

    best_baseline = _find_best_baseline(model_summaries)
    tests_payload = _compute_statistical_tests(all_frame, best_baseline)
    Path(args.tests_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.tests_output).open("w", encoding="utf-8") as handle:
        json.dump(tests_payload, handle, indent=2)

    criteria_payload = _criteria_report(model_summaries, tests_payload, best_baseline)
    with (Path(args.final_table_output).parent / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(criteria_payload, handle, indent=2)

    table_rows = []
    for model_name, summary in model_summaries.items():
        aggregate = summary["aggregate"]
        table_rows.append(
            {
                "Model": model_name,
                "Accuracy": aggregate["accuracy"]["mean"],
                "Balanced Accuracy": aggregate["balanced_accuracy"]["mean"],
                "F1": aggregate["f1"]["mean"],
                "ROC-AUC": aggregate["roc_auc"]["mean"],
                "Params": int(round(aggregate["params"]["mean"])),
                "Runtime": aggregate["runtime_seconds"]["mean"],
                "Mean ± Std (multi-seed)": f"{aggregate['f1']['mean']:.4f} +- {aggregate['f1']['std']:.4f}",
            }
        )
    final_table = pd.DataFrame(table_rows).sort_values(by=["F1", "Accuracy"], ascending=False)
    Path(args.final_table_output).parent.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(args.final_table_output, index=False)

    print(
        json.dumps(
            {
                "aggregate_output": args.aggregate_output,
                "tests_output": args.tests_output,
                "final_table_output": args.final_table_output,
                "quality_report": str(Path(args.final_table_output).parent / "quality_report.json"),
                "all_checks_passed": criteria_payload.get("all_checks_passed", False),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
