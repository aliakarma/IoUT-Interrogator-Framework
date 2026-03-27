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
import re
import sys
from typing import Dict, List, Tuple

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


def aggregate_seed_level(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one value per (seed, metric_name) by averaging across runs/intervals.

    This avoids mixing interval-level rows as independent samples when the
    inferential unit is the seed.
    """
    validate_raw_results_schema(raw_df)
    grouped = (
        raw_df.groupby(["seed", "metric_name"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "seed_value"})
    )
    return grouped


def summarize_seed_level(
    raw_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize metrics using seed-level values (one sample per seed)."""
    seed_df = aggregate_seed_level(raw_df)

    rows = []
    metric_names = sorted(seed_df["metric_name"].astype(str).unique().tolist())
    for metric_name in metric_names:
        values = seed_df.loc[
            seed_df["metric_name"] == metric_name,
            "seed_value",
        ].astype(float).to_numpy(dtype=np.float64)

        mean, ci_lower, ci_upper = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )

        rows.append(
            {
                "metric_name": metric_name,
                "n_seeds": int(values.size),
                "mean": round(mean, 6),
                "std": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
                f"ci{int(ci_level)}_lower": round(ci_lower, 6),
                f"ci{int(ci_level)}_upper": round(ci_upper, 6),
            }
        )

    return pd.DataFrame(rows)


def _split_metric_model(metric_name: str) -> Tuple[str, str] | Tuple[None, None]:
    """Split metric names like accuracy_proposed into (accuracy, proposed)."""
    m = re.match(r"^(?P<base>.+)_(?P<model>proposed|bayesian|static)$", str(metric_name))
    if not m:
        return None, None
    return m.group("base"), m.group("model")


def _paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    if a.size != b.size:
        raise ValueError("Arrays must have the same length for paired Cohen's d.")
    if a.size < 2:
        return float("nan")
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def compute_model_significance(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare proposed model vs each baseline per base metric using seed-level values.

    Output columns:
      metric_name, model_a, model_b, p_value_ttest, p_value_wilcoxon,
      effect_size_cohens_d
    """
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError as exc:
        raise ImportError(
            "scipy is required for significance testing. "
            "Install dependencies from requirements.txt to enable --significance-output."
        ) from exc

    work = seed_level_df.copy()

    parsed = work["metric_name"].apply(_split_metric_model)
    work["metric_base"] = parsed.apply(lambda t: t[0])
    work["model_name"] = parsed.apply(lambda t: t[1])
    work = work.dropna(subset=["metric_base", "model_name"])

    if work.empty:
        return pd.DataFrame(
            columns=[
                "metric_name",
                "model_a",
                "model_b",
                "p_value_ttest",
                "p_value_wilcoxon",
                "effect_size_cohens_d",
            ]
        )

    rows: List[Dict[str, object]] = []
    for metric_base in sorted(work["metric_base"].unique().tolist()):
        sub = work[work["metric_base"] == metric_base]
        pivot = sub.pivot_table(index="seed", columns="model_name", values="seed_value", aggfunc="mean")
        if "proposed" not in pivot.columns:
            continue

        for baseline in ["bayesian", "static"]:
            if baseline not in pivot.columns:
                continue

            paired = pivot[["proposed", baseline]].dropna()
            n_pairs = int(len(paired))

            if n_pairs < 2:
                p_t = float("nan")
                p_w = float("nan")
                d = float("nan")
            else:
                a = paired["proposed"].to_numpy(dtype=np.float64)
                b = paired[baseline].to_numpy(dtype=np.float64)

                p_t = float(ttest_rel(a, b, nan_policy="omit").pvalue)
                try:
                    p_w = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
                except ValueError:
                    # Degenerate case: all differences are zero.
                    p_w = 1.0
                d = _paired_cohens_d(a, b)

            rows.append(
                {
                    "metric_name": metric_base,
                    "model_a": "proposed",
                    "model_b": baseline,
                    "p_value_ttest": round(p_t, 8) if not np.isnan(p_t) else np.nan,
                    "p_value_wilcoxon": round(p_w, 8) if not np.isnan(p_w) else np.nan,
                    "effect_size_cohens_d": round(d, 8) if not np.isnan(d) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="simulation/outputs/raw_results.csv")
    parser.add_argument("--output", default="analysis/stats/summary_table.csv")
    parser.add_argument(
        "--aggregate-level",
        choices=["row", "seed"],
        default="row",
        help=(
            "Aggregation unit for summary statistics. "
            "'row' preserves legacy behavior; 'seed' computes one value per seed "
            "before bootstrap/summary."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000,
                        help="Number of bootstrap resamples (must be >= 1000).")
    parser.add_argument("--ci-level", type=float, default=95.0,
                        help="Confidence interval level in percent.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for bootstrap sampling.")
    parser.add_argument(
        "--significance-output",
        default=None,
        help=(
            "Optional output CSV for paired significance tests and effect sizes. "
            "Computed from seed-level values and compares proposed vs baselines."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Results CSV not found: {args.input}. "
            "Run the simulation pipeline first: python scripts/run_full_pipeline.py"
        )

    raw_df = pd.read_csv(args.input)
    if args.aggregate_level == "seed":
        summary = summarize_seed_level(
            raw_df,
            n_bootstrap=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=args.seed,
        )
    else:
        summary = summarize_raw_results(
            raw_df,
            n_bootstrap=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output, index=False)

    if args.significance_output:
        seed_level_df = aggregate_seed_level(raw_df)
        significance_df = compute_model_significance(seed_level_df)
        os.makedirs(os.path.dirname(args.significance_output), exist_ok=True)
        significance_df.to_csv(args.significance_output, index=False)
        print(f"Saved significance/effect sizes to: {args.significance_output}")

    print("\n=== Performance Summary with Bootstrap CI ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
