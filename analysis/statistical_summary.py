"""
Statistical Summary
====================
Computes per-metric summary statistics from raw long-format simulation output,
including mean and bootstrap confidence intervals.

Usage:
    python analysis/statistical_summary.py \
    --input simulation/outputs/raw_results.csv \
        --output analysis/stats/summary_table.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


REQUIRED_RAW_COLUMNS = {"run_id", "seed", "interval", "metric_name", "value"}


def validate_raw_results_schema(df: pd.DataFrame) -> None:
    """Validate required raw long-format columns."""
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Input must be a raw long-format CSV with columns "
            "run_id, seed, interval, metric_name, value. "
            f"Missing: {missing_str}."
        )


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute sample mean and percentile bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty value array.")
    if n_bootstrap < 1000:
        raise ValueError("n_bootstrap must be at least 1000.")
    if not (0.0 < ci_level < 100.0):
        raise ValueError("ci_level must be in (0, 100).")

    rng = np.random.default_rng(seed)
    n = values.size

    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()

    alpha = (100.0 - ci_level) / 2.0
    lower = float(np.percentile(bootstrap_means, alpha))
    upper = float(np.percentile(bootstrap_means, 100.0 - alpha))
    return float(values.mean()), lower, upper


def summarize_raw_results(
    raw_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize each metric in raw long-format simulation output."""
    validate_raw_results_schema(raw_df)

    rows = []
    metric_names = sorted(raw_df["metric_name"].astype(str).unique().tolist())

    for metric_name in metric_names:
        metric_vals = raw_df.loc[
            raw_df["metric_name"] == metric_name,
            "value",
        ].astype(float).to_numpy(dtype=np.float64)

        mean, ci_lower, ci_upper = bootstrap_mean_ci(
            metric_vals,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )

        rows.append(
            {
                "metric_name": metric_name,
                "n": int(metric_vals.size),
                "mean": round(mean, 6),
                "std": round(float(metric_vals.std(ddof=1)) if metric_vals.size > 1 else 0.0, 6),
                f"ci{int(ci_level)}_lower": round(ci_lower, 6),
                f"ci{int(ci_level)}_upper": round(ci_upper, 6),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="simulation/outputs/raw_results.csv")
    parser.add_argument("--output", default="analysis/stats/summary_table.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=2000,
                        help="Number of bootstrap resamples (must be >= 1000).")
    parser.add_argument("--ci-level", type=float, default=95.0,
                        help="Confidence interval level in percent.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for bootstrap sampling.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Results CSV not found: {args.input}. "
            "Run the simulation pipeline first: python scripts/run_full_pipeline.py"
        )

    raw_df = pd.read_csv(args.input)
    summary = summarize_raw_results(
        raw_df,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("\n=== Performance Summary with Bootstrap CI ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
