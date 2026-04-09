from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        training_summary_path = run_dir / "training_summary.json"
        training_summary = _read_json(training_summary_path) if training_summary_path.exists() else {}
        run_summary_path = run_dir / "run_summary.json"
        run_summary = _read_json(run_summary_path) if run_summary_path.exists() else {}
        run_result_path = run_dir / "run_result.json"
        run_result = _read_json(run_result_path) if run_result_path.exists() else {}

        model_name = (
            run_summary.get("model")
            or run_summary.get("metadata", {}).get("model")
            or run_result.get("model")
            or run_dir.name
        )
        rows.append(
            {
                "Model": model_name,
                "Accuracy": float(metrics.get("accuracy", 0.0)),
                "F1": float(metrics.get("f1", 0.0)),
                "Params": int(run_summary.get("params", training_summary.get("params", 0))),
                "Runtime": float(training_summary.get("runtime_seconds", run_summary.get("runtime_seconds", 0.0))),
                "RunDir": str(run_dir),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables from IoUT experiment outputs")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--csv-output", default="results/summary_table.csv")
    parser.add_argument("--markdown-output", default="results/summary_table.md")
    args = parser.parse_args()

    rows = _collect_rows(Path(args.results_root))
    if not rows:
        raise SystemExit(f"No metrics.json files found under {args.results_root}")

    table = pd.DataFrame(rows)
    grouped = (
        table.groupby("Model", as_index=False)
        .agg({"Accuracy": "mean", "F1": "mean", "Params": "mean", "Runtime": "mean"})
        .sort_values(["Accuracy", "F1"], ascending=False)
    )

    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.csv_output, index=False)

    md_lines = ["| Model | Accuracy | F1 | Params | Runtime |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in grouped.itertuples(index=False):
        md_lines.append(f"| {row.Model} | {row.Accuracy:.4f} | {row.F1:.4f} | {int(round(row.Params))} | {row.Runtime:.2f} |")
    Path(args.markdown_output).write_text("\n".join(md_lines), encoding="utf-8")

    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
