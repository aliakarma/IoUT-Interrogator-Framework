#!/usr/bin/env python3
"""
Export all results from the full experimental pipeline into structured output
ready for publication in research papers.

This script acts as the SINGLE SOURCE OF TRUTH for all reported results,
aggregating results from:
- Main simulation outputs (raw_results.csv, results.csv)
- Multi-seed experiments
- OOD evaluation
- Baseline models
- Statistical significance tests
- Ablation studies
- Model calibration metrics

Usage:
    python export_all_results.py --output-dir results/final_results/
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metric name mapping (internal -> paper notation)
METRIC_MAPPING = {
    'accuracy': 'C',
    'recall': 'RP',
    'precision': 'P',
    'f1': 'A',
    'trust_mean': 'TI',
    'ece': 'ECE',
    'brier': 'Brier',
}

# List of models to include in comparison
MODELS_LIST = ['proposed', 'bayesian', 'static', 'logistic_regression', 'random_forest', 'lstm']
CORE_METRICS = ['C', 'RP', 'P', 'A', 'TI']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1']


def _to_unit_interval(value: float) -> float:
    """Convert metric value to [0,1] scale with safe clipping.

    Handles either fractional values (0..1) or percentage-style values (0..100).
    """
    v = float(value)
    if np.isnan(v):
        return np.nan
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return float(np.clip(v, 0.0, 1.0))


def _extract_sim_metric(metric_name: str, model: str) -> Optional[str]:
    suffix = f"_{model}"
    if not isinstance(metric_name, str) or not metric_name.endswith(suffix):
        return None
    return metric_name[: -len(suffix)]


def build_classification_results(
    calibration_metrics: Optional[Dict],
    baseline_results: Optional[pd.DataFrame],
    lstm_baseline: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build supervised classification-only table.

    Models: transformer_v4, logistic_regression, random_forest, lstm
    Metrics: accuracy, precision, recall, f1 in [0,1]
    """
    rows: List[Dict[str, object]] = []

    # Transformer v4 from evaluation_metrics.json
    if calibration_metrics is not None and 'test' in calibration_metrics:
        test = calibration_metrics['test']
        rows.append({
            'model': 'transformer_v4',
            'accuracy': _to_unit_interval(float(test.get('accuracy', np.nan))),
            'precision': _to_unit_interval(float(test.get('precision', np.nan))),
            'recall': _to_unit_interval(float(test.get('recall', np.nan))),
            'f1': _to_unit_interval(float(test.get('f1', np.nan))),
        })

    # Classical baselines from baseline_results.csv (test split)
    if baseline_results is not None and not baseline_results.empty:
        test_df = baseline_results[baseline_results['split'] == 'test'].copy()
        for model_name in ['logistic_regression', 'random_forest']:
            model_df = test_df[test_df['model'] == model_name]
            if model_df.empty:
                continue
            row = model_df.iloc[0]
            rows.append({
                'model': model_name,
                'accuracy': _to_unit_interval(float(row.get('accuracy', np.nan))),
                'precision': _to_unit_interval(float(row.get('precision', np.nan))),
                'recall': _to_unit_interval(float(row.get('recall', np.nan))),
                'f1': _to_unit_interval(float(row.get('f1', np.nan))),
            })

    # LSTM baseline from lstm_baseline_results.csv (test split)
    if lstm_baseline is not None and not lstm_baseline.empty:
        test_df = lstm_baseline[lstm_baseline['split'] == 'test'].copy() if 'split' in lstm_baseline.columns else lstm_baseline
        if not test_df.empty:
            row = test_df.iloc[0]
            rows.append({
                'model': 'lstm',
                'accuracy': _to_unit_interval(float(row.get('accuracy', np.nan))),
                'precision': _to_unit_interval(float(row.get('precision', np.nan))),
                'recall': _to_unit_interval(float(row.get('recall', np.nan))),
                'f1': _to_unit_interval(float(row.get('f1', np.nan))),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Enforce column order and deterministic model order
    result = result[['model'] + CLASSIFICATION_METRICS]
    model_order = ['transformer_v4', 'logistic_regression', 'random_forest', 'lstm']
    result['model'] = pd.Categorical(result['model'], categories=model_order, ordered=True)
    result = result.sort_values('model').reset_index(drop=True)
    result['model'] = result['model'].astype(str)
    return result


def build_simulation_results(
    multi_seed_summary: Optional[pd.DataFrame],
    raw_results: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build simulation-only table for proposed/bayesian/static metrics.

    Output is long format with consistent per-metric min-max normalization in [0,1].
    """
    sim_rows: List[Dict[str, object]] = []
    sim_models = ['proposed', 'bayesian', 'static']

    if raw_results is not None and not raw_results.empty:
        grouped = raw_results.groupby('metric_name', as_index=False)['value'].mean()
        for _, row in grouped.iterrows():
            metric_name = str(row.get('metric_name', ''))
            mean_val = float(row.get('value', np.nan))
            for model in sim_models:
                base_metric = _extract_sim_metric(metric_name, model)
                if base_metric is None:
                    continue
                sim_rows.append({
                    'model': model,
                    'metric_name': 'trust_index' if base_metric == 'trust_mean' else base_metric,
                    'value': mean_val,
                })
                break
    elif multi_seed_summary is not None and not multi_seed_summary.empty:
        for _, row in multi_seed_summary.iterrows():
            metric_name = str(row.get('metric_name', ''))
            mean_val = float(row.get('mean', np.nan))
            for model in sim_models:
                base_metric = _extract_sim_metric(metric_name, model)
                if base_metric is None:
                    continue
                sim_rows.append({
                    'model': model,
                    'metric_name': 'trust_index' if base_metric == 'trust_mean' else base_metric,
                    'value': mean_val,
                })
                break
    result = pd.DataFrame(sim_rows)
    if result.empty:
        return result

    # Consistent scaling across all simulation metrics: min-max per metric over models.
    result['value_scaled_0_1'] = np.nan
    for metric in result['metric_name'].unique():
        metric_mask = result['metric_name'] == metric
        values = result.loc[metric_mask, 'value'].astype(float)
        vmin = float(values.min())
        vmax = float(values.max())
        if np.isclose(vmax, vmin):
            result.loc[metric_mask, 'value_scaled_0_1'] = 0.5
        else:
            result.loc[metric_mask, 'value_scaled_0_1'] = (values - vmin) / (vmax - vmin)

    result['value_scaled_0_1'] = result['value_scaled_0_1'].astype(float).clip(0.0, 1.0)
    result['scaling'] = 'minmax_per_metric_over_models'
    result = result.sort_values(['metric_name', 'model']).reset_index(drop=True)
    return result


def ensure_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {path.resolve()}")
    return path


def load_raw_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load raw_results.csv and return long-format dataframe."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded raw_results.csv: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"raw_results.csv not found at {filepath}")
        return None


def load_aggregated_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load results.csv (aggregated by interval)."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded results.csv: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"results.csv not found at {filepath}")
        return None


def load_multi_seed_summary(filepath: str) -> Optional[pd.DataFrame]:
    """Load multi_seed_summary.csv with seed-level aggregation."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded multi_seed_summary.csv: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"multi_seed_summary.csv not found at {filepath}")
        return None


def load_multi_seed_raw(filepath: str) -> Optional[pd.DataFrame]:
    """Load multi_seed_raw_results.csv for seed-level processing."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded multi_seed_raw_results.csv: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"multi_seed_raw_results.csv not found at {filepath}")
        return None


def load_baseline_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load baseline model results."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded baseline results: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"Baseline results not found at {filepath}")
        return None


def load_ood_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load OOD evaluation results."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded OOD results: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"OOD results not found at {filepath}")
        return None


def load_ablation_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load ablation study results."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded ablation results: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"Ablation results not found at {filepath}")
        return None


def load_significance_results(filepath: str) -> Optional[pd.DataFrame]:
    """Load significance test results."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded significance results: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"Significance results not found at {filepath}")
        return None


def load_calibration_metrics(filepath: str) -> Optional[Dict]:
    """Load calibration metrics from JSON."""
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded calibration metrics from JSON")
        return metrics
    except FileNotFoundError:
        logger.warning(f"Calibration metrics not found at {filepath}")
        return None


def compute_bootstrap_ci(values: np.ndarray, ci: float = 95, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if len(values) == 0:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper


def compute_main_metrics(
    multi_seed_summary: Optional[pd.DataFrame],
    raw_results: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute main metrics for proposed model.
    
    Prefers current raw results so exported publication tables stay aligned
    with the active pipeline outputs. Falls back to multi-seed summary only
    when raw results are unavailable.
    """
    logger.info("Computing main metrics for proposed model...")
    
    if raw_results is not None and not raw_results.empty:
        # Aggregate from current raw results
        proposed_data = raw_results[
            raw_results['metric_name'].str.contains('proposed', case=False, na=False)
        ].copy()

        proposed_data['metric'] = proposed_data['metric_name'].str.replace('_proposed', '', regex=False)

        results = []
        for metric in proposed_data['metric'].unique():
            values = proposed_data[proposed_data['metric'] == metric]['value'].values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                ci_lower, ci_upper = compute_bootstrap_ci(values)
                results.append({
                    'metric_name': metric,
                    'mean': mean_val,
                    'std': std_val,
                    'ci95_lower': ci_lower,
                    'ci95_upper': ci_upper,
                })

        result_df = pd.DataFrame(results)
        logger.info(f"Using current raw results aggregation: {len(result_df)} metrics")
        return result_df

    elif multi_seed_summary is not None:
        # Filter for proposed model metrics
        proposed_df = multi_seed_summary[
            multi_seed_summary['metric_name'].str.contains('proposed', case=False, na=False)
        ].copy()
        
        # Extract metric name without model suffix
        proposed_df['metric'] = proposed_df['metric_name'].str.replace('_proposed', '', regex=False)
        
        # Select columns
        result_df = proposed_df[['metric', 'mean', 'std', 'ci95_lower', 'ci95_upper']].copy()
        result_df.rename(columns={'metric': 'metric_name'}, inplace=True)
        
        logger.info(f"Using seed-level aggregation: {len(result_df)} metrics")
        return result_df
    
    else:
        logger.error("No multi-seed or raw results available!")
        return pd.DataFrame()


def compute_baseline_comparison(
    baseline_results: Optional[pd.DataFrame],
    lstm_baseline: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Combine and align baseline model results."""
    logger.info("Computing baseline comparison metrics...")

    def _normalize_core_metric(metric_code: str, value: float) -> float:
        return _to_unit_interval(value) if metric_code in {'C', 'RP', 'P', 'A', 'TI'} else float(value)
    
    dfs = []
    
    if baseline_results is not None:
        # Get test split only
        test_df = baseline_results[baseline_results['split'] == 'test'].copy()
        
        for _, row in test_df.iterrows():
            model = row['model']
            for metric in CORE_METRICS:
                # Map paper notation back to column name
                col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
                
                if col_name in test_df.columns:
                    dfs.append({
                        'model': model,
                        'metric_name': metric,
                        'mean': _normalize_core_metric(metric, row[col_name]),
                    })
    
    if lstm_baseline is not None:
        # Get test split only
        test_df = lstm_baseline[lstm_baseline['split'] == 'test'].copy() if 'split' in lstm_baseline.columns else lstm_baseline
        
        for _, row in test_df.iterrows():
            model = row.get('model', 'lstm')
            for metric in CORE_METRICS:
                col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
                
                if col_name in lstm_baseline.columns:
                    dfs.append({
                        'model': model,
                        'metric_name': metric,
                        'mean': _normalize_core_metric(metric, row[col_name]),
                    })
    
    result_df = pd.DataFrame(dfs)
    logger.info(f"Baseline comparison: {len(result_df)} rows")
    return result_df


def compute_ood_summary(ood_results: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate OOD results by dataset condition."""
    logger.info("Computing OOD summary...")
    
    if ood_results is None or ood_results.empty:
        logger.warning("No OOD results available")
        return pd.DataFrame()
    
    # Group by dataset and metric, compute mean and CI
    grouped = ood_results.groupby(['dataset_name', 'metric_name'])['value'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
    ]).reset_index()
    
    # Compute CI
    ci_results = []
    for _, row in grouped.iterrows():
        # Get original values for CI computation
        mask = (ood_results['dataset_name'] == row['dataset_name']) & \
               (ood_results['metric_name'] == row['metric_name'])
        values = ood_results[mask]['value'].values
        ci_lower, ci_upper = compute_bootstrap_ci(values)
        
        ci_results.append({
            'dataset_name': row['dataset_name'],
            'metric_name': row['metric_name'],
            'mean': row['mean'],
            'ci95_lower': ci_lower,
            'ci95_upper': ci_upper,
        })
    
    result_df = pd.DataFrame(ci_results)
    logger.info(f"OOD summary: {len(result_df)} rows")
    return result_df


def compute_ablation_summary(ablation_results: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate ablation results by configuration."""
    logger.info("Computing ablation summary...")
    
    if ablation_results is None or ablation_results.empty:
        logger.warning("No ablation results available")
        return pd.DataFrame()
    
    # Group by configuration and metric, compute mean
    result_df = ablation_results.groupby(['configuration', 'metric_name'])['value'].mean().reset_index()
    result_df.rename(columns={'value': 'mean'}, inplace=True)
    
    logger.info(f"Ablation summary: {len(result_df)} rows")
    return result_df


def compute_calibration_summary(calibration_metrics: Optional[Dict]) -> pd.DataFrame:
    """Extract calibration summary from evaluation metrics."""
    logger.info("Computing calibration summary...")
    
    if calibration_metrics is None:
        logger.warning("No calibration metrics available")
        return pd.DataFrame()
    
    results = []
    
    # Extract validation and test metrics
    for split in ['validation', 'test']:
        if split in calibration_metrics:
            split_data = calibration_metrics[split]
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'ece', 'brier']:
                if metric in split_data:
                    results.append({
                        'split': split,
                        'metric_name': metric,
                        'value': split_data[metric],
                    })
    
    result_df = pd.DataFrame(results)
    logger.info(f"Calibration summary: {len(result_df)} rows")
    return result_df


def compute_combined_results_table(
    multi_seed_summary: Optional[pd.DataFrame],
    raw_results: Optional[pd.DataFrame],
    baseline_results: Optional[pd.DataFrame],
    lstm_baseline: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Create single table for paper with test metrics only.
    
    Rows: proposed, bayesian, static, logistic_regression, random_forest, lstm
    Columns: C, RP, P, A, TI
    """
    logger.info("Computing combined results table for paper...")

    def _normalize_core_metric(metric_code: str, value: float) -> float:
        return _to_unit_interval(value) if metric_code in {'C', 'RP', 'P', 'A', 'TI'} else float(value)
    
    table_data = {}
    
    # Extract proposed, bayesian, static from multi-seed or raw results
    for model in ['proposed', 'bayesian', 'static']:
        model_data = {}
        
        if raw_results is not None and not raw_results.empty:
            # Prefer current raw results so exported tables cannot silently mix
            # with stale multi-seed artifacts from earlier runs.
            for metric in CORE_METRICS:
                col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
                metric_col = f"{col_name}_{model}"

                values = raw_results[raw_results['metric_name'] == metric_col]['value'].values
                if len(values) > 0:
                    model_data[metric] = _normalize_core_metric(metric, np.mean(values))

        elif multi_seed_summary is not None:
            # Use seed-level means
            for metric in CORE_METRICS:
                col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
                metric_col = f"{col_name}_{model}"
                
                subset = multi_seed_summary[multi_seed_summary['metric_name'] == metric_col]
                if not subset.empty:
                    model_data[metric] = _normalize_core_metric(metric, subset['mean'].values[0])
        
        if model_data:
            table_data[model] = model_data
    
    # Add baselines (test split)
    if baseline_results is not None:
        test_df = baseline_results[baseline_results['split'] == 'test'].copy()
        for _, row in test_df.iterrows():
            model = row['model']
            model_data = {}
            for metric in CORE_METRICS:
                col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
                if col_name in test_df.columns:
                    model_data[metric] = _normalize_core_metric(metric, row[col_name])
            if model_data:
                table_data[model] = model_data
    
    if lstm_baseline is not None:
        test_df = lstm_baseline[lstm_baseline['split'] == 'test'].copy() if 'split' in lstm_baseline.columns else lstm_baseline
        model_data = {}
        for metric in CORE_METRICS:
            col_name = [k for k, v in METRIC_MAPPING.items() if v == metric][0] if metric in METRIC_MAPPING.values() else metric
            if col_name in lstm_baseline.columns:
                model_data[metric] = (
                    _normalize_core_metric(metric, test_df[col_name].values[0])
                    if len(test_df) > 0 else None
                )
        if model_data:
            table_data['lstm'] = model_data
    
    # Create dataframe
    result_df = pd.DataFrame(table_data).T.reset_index()
    result_df.rename(columns={'index': 'model'}, inplace=True)
    
    # Reorder columns to match paper notation
    cols = ['model'] + CORE_METRICS
    result_df = result_df[[c for c in cols if c in result_df.columns]]
    
    logger.info(f"Combined results table: {len(result_df)} models × {len(CORE_METRICS)} metrics")
    return result_df


def export_results(
    output_dir: str,
    sim_outputs_dir: str = 'results/simulation',
    analysis_stats_dir: str = 'results/stats',
    checkpoint_dir: str = 'results/checkpoints',
):
    """Main export orchestration."""
    
    output_path = ensure_output_dir(output_dir)
    
    # Construct filepaths
    raw_results_file = os.path.join(sim_outputs_dir, 'raw_results.csv')
    results_file = os.path.join(sim_outputs_dir, 'results.csv')
    multi_seed_raw_file = os.path.join(sim_outputs_dir, 'multi_seed_raw_results.csv')
    multi_seed_summary_file = os.path.join(analysis_stats_dir, 'multi_seed_summary.csv')
    baseline_file = os.path.join(analysis_stats_dir, 'baseline_results.csv')
    lstm_baseline_file = os.path.join(analysis_stats_dir, 'lstm_baseline_results.csv')
    ood_file = os.path.join(analysis_stats_dir, 'ood_results.csv')
    ablation_file = os.path.join(analysis_stats_dir, 'ablation_results.csv')
    significance_file = os.path.join(analysis_stats_dir, 'model_significance.csv')
    calibration_file = os.path.join(checkpoint_dir, 'evaluation_metrics.json')
    
    # Load all data
    logger.info("\n=== LOADING DATA ===")
    raw_results = load_raw_results(raw_results_file)
    aggregated_results = load_aggregated_results(results_file)
    multi_seed_raw = load_multi_seed_raw(multi_seed_raw_file)
    multi_seed_summary = load_multi_seed_summary(multi_seed_summary_file)
    baseline_results = load_baseline_results(baseline_file)
    lstm_baseline = load_baseline_results(lstm_baseline_file)
    ood_results = load_ood_results(ood_file)
    ablation_results = load_ablation_results(ablation_file)
    significance_results = load_significance_results(significance_file)
    calibration_metrics = load_calibration_metrics(calibration_file)
    
    # Compute summaries
    logger.info("\n=== COMPUTING SUMMARIES ===")
    main_metrics = compute_main_metrics(multi_seed_summary, raw_results)
    baseline_comparison = compute_baseline_comparison(baseline_results, lstm_baseline)
    ood_summary = compute_ood_summary(ood_results)
    ablation_summary = compute_ablation_summary(ablation_results)
    calibration_summary = compute_calibration_summary(calibration_metrics)
    combined_table = compute_combined_results_table(
        multi_seed_summary, raw_results, baseline_results, lstm_baseline
    )
    classification_results = build_classification_results(
        calibration_metrics=calibration_metrics,
        baseline_results=baseline_results,
        lstm_baseline=lstm_baseline,
    )
    simulation_results = build_simulation_results(
        multi_seed_summary=multi_seed_summary,
        raw_results=raw_results,
    )
    
    # Export results
    logger.info("\n=== EXPORTING RESULTS ===")
    
    if not main_metrics.empty:
        main_metrics.to_csv(output_path / 'main_metrics.csv', index=False)
        logger.info(f"✓ Exported main_metrics.csv ({len(main_metrics)} rows)")
    
    if not baseline_comparison.empty:
        baseline_comparison.to_csv(output_path / 'baseline_comparison.csv', index=False)
        logger.info(f"✓ Exported baseline_comparison.csv ({len(baseline_comparison)} rows)")
    
    if significance_results is not None and not significance_results.empty:
        significance_results.to_csv(output_path / 'significance_results.csv', index=False)
        logger.info(f"✓ Exported significance_results.csv ({len(significance_results)} rows)")
    else:
        logger.info("⊘ Skipping significance_results.csv (not available)")
    
    if not ood_summary.empty:
        ood_summary.to_csv(output_path / 'ood_summary.csv', index=False)
        logger.info(f"✓ Exported ood_summary.csv ({len(ood_summary)} rows)")
    
    if not ablation_summary.empty:
        ablation_summary.to_csv(output_path / 'ablation_summary.csv', index=False)
        logger.info(f"✓ Exported ablation_summary.csv ({len(ablation_summary)} rows)")
    
    if not calibration_summary.empty:
        calibration_summary.to_csv(output_path / 'calibration_summary.csv', index=False)
        logger.info(f"✓ Exported calibration_summary.csv ({len(calibration_summary)} rows)")
    else:
        logger.info("⊘ Skipping calibration_summary.csv (not available)")
    
    if not combined_table.empty:
        combined_table.to_csv(output_path / 'combined_results_table.csv', index=False)
        logger.info(f"✓ Exported combined_results_table.csv ({len(combined_table)} rows)")

    if not classification_results.empty:
        classification_results.to_csv(output_path / 'classification_results.csv', index=False)
        logger.info(f"✓ Exported classification_results.csv ({len(classification_results)} rows)")
    else:
        logger.info("⊘ Skipping classification_results.csv (not available)")

    if not simulation_results.empty:
        simulation_results.to_csv(output_path / 'simulation_results.csv', index=False)
        logger.info(f"✓ Exported simulation_results.csv ({len(simulation_results)} rows)")
    else:
        logger.info("⊘ Skipping simulation_results.csv (not available)")
    
    logger.info(f"\n=== SUCCESS ===")
    logger.info(f"All results exported to: {output_path.resolve()}")
    logger.info(f"Ready for publication in research papers.")


def main():
    parser = argparse.ArgumentParser(
        description='Export all results from experimental pipeline for publication.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/final_results',
        help='Output directory for final results (default: results/final_results)'
    )
    parser.add_argument(
        '--sim-outputs-dir',
        type=str,
        default='results/simulation',
        help='Path to simulation outputs directory'
    )
    parser.add_argument(
        '--analysis-stats-dir',
        type=str,
        default='results/stats',
        help='Path to statistics directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='results/checkpoints',
        help='Path to model checkpoints directory'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("IoUT RESULTS EXPORT PIPELINE")
    logger.info("=" * 60)
    
    try:
        export_results(
            output_dir=args.output_dir,
            sim_outputs_dir=args.sim_outputs_dir,
            analysis_stats_dir=args.analysis_stats_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
