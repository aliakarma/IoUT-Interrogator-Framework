# IoUT Interrogator Framework

An end-to-end, reproducible machine learning pipeline for Internet of Underwater Things (IoUT) signal classification and trust inference.

The repository now exposes a production-oriented entry point at `run_pipeline.py` that wires together data loading, deterministic splitting, preprocessing, model training, baseline evaluation, checkpointing, and metric export.

## Project Overview

The pipeline is organized around a simple contract:

- `data/` handles ingestion, normalization, padding, and deterministic group-based splitting.
- `models/` contains the GRU sequence classifier plus classical baselines.
- `training/` owns the fit loop, checkpointing, and epoch logging.
- `evaluation/` computes core metrics and writes run artifacts.

The default configuration runs immediately against the repository’s existing behavioral JSON data while also supporting CSV or JSON files with the schema `sensor_id,timestamp,signal[,label]`.

## Repository Layout

```text
configs/default.yaml        Default pipeline configuration
run_pipeline.py             Main CLI entry point
data/data_loader.py         Dataset, split, normalization, batching
data/download.sh            Placeholder for dataset acquisition
models/sequence_model.py    GRU sequence classifier
models/baselines.py         Random, rule-based, and sklearn baselines
training/trainer.py         Deterministic training loop and checkpoints
evaluation/metrics.py       Accuracy / precision / recall / F1
evaluation/evaluate.py      Inference and metric export
results/run_1/              Checkpoints and run artifacts
logs/train.log              Epoch-level training log
```

## Installation

### Pip

```bash
python -m pip install -r requirements.txt
```

### Docker

```bash
docker build -t iout-interrogator:latest .
docker run --rm -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs iout-interrogator:latest python run_pipeline.py --config configs/default.yaml
```

On Windows PowerShell, use the same `docker run` command with `${PWD}` as shown above.

## Dataset Instructions

The loader accepts two formats:

- Long-form CSV or JSON with columns or keys: `sensor_id`, `timestamp`, `signal`, and optionally `label`.
- Legacy sequence JSON from this repository, where each item contains a `sequence` array and a `label`. The loader converts that representation into grouped sensor trajectories automatically.

Recommended dataset conventions:

- Use a stable `sensor_id` per entity or device.
- Keep timestamps sortable and numeric.
- Provide labels at the sensor or group level whenever possible.
- Place raw files under `data/raw/` and point `configs/default.yaml` at the file you want to train on.

If you need a placeholder acquisition script, use `data/download.sh` as the location for your real download or extraction commands.

## Outputs

Each run writes artifacts to `results/run_1/` and `logs/`:

- `results/run_1/checkpoints/best.pt` - best validation checkpoint for the GRU model
- `results/run_1/metrics.json` - evaluation metrics plus confusion matrix metadata
- `results/run_1/confusion_matrix.json` - optional confusion matrix export
- `results/run_1/training_summary.json` - epoch history and best validation loss
- `logs/train.log` - epoch-level training and validation loss

The metrics file contains:

- accuracy
- precision
- recall
- F1

## Reproduce Results in 5 Steps

```bash
git clone https://github.com/aliakarma/IoUT-Interrogator-Framework.git
cd IoUT-Interrogator-Framework
python -m pip install -r requirements.txt
python run_pipeline.py --config configs/default.yaml
python -c "import json, pathlib; print(json.dumps(json.loads(pathlib.Path('results/run_1/metrics.json').read_text()), indent=2))"
```

## Baselines

The repository includes four drop-in baseline modes:

- `random`
- `rule`
- `logistic_regression`
- `random_forest`

These share the same dataset split and evaluation path as the GRU model, so comparisons are aligned and reproducible.

## Reproducibility Guarantees

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- deterministic group splits by `sensor_id`
- no train/validation/test overlap at the sensor level

## Notes

The repository still includes legacy analysis and simulation code for prior experiments, but the recommended production entry point is now `run_pipeline.py`.
