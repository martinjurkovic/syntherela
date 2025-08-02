# GNN Hyperparameter Tuning with Optuna

This directory contains scripts for hyperparameter tuning of GNN architectures using Optuna optimization library.

## Overview

The hyperparameter tuning system consists of two main components:

1. **`tmux_hyperparameter_tuning.py`** - Sets up parallel tmux sessions for each GNN architecture
2. **`run_optuna_hyperparameter_tuning.py`** - Performs actual hyperparameter optimization using Optuna

## Features

- **Parallel Optimization**: Each GNN architecture runs on a separate GPU simultaneously
- **Original Datasets**: Runs on full original datasets (not subsampled versions)
- **Comprehensive Search Space**: Optimizes learning rate, number of layers, number of neighbors, and weight decay
- **Multi-Dataset Evaluation**: Each trial evaluates across all 5 datasets and aggregates results
- **Persistent Storage**: Results stored in SQLite database for later analysis

## GNN Architectures

The system supports all 6 GNN architectures:
- `hetero-graphsage`
- `hetero-gin`
- `hetero-graphconv`
- `hetero-gat`
- `hetero-gatv2`
- `relgnn`

## Datasets

Uses original (full) datasets:
- `rossmann` (regression task)
- `walmart` (regression task)
- `airbnb-simplified` (binary classification)
- `f1` (binary classification)
- `Berka` (binary classification)

## Hyperparameter Search Space

- **Learning Rate**: 0.0001, 0.001, 0.01, 0.1 (categorical choice)
- **Number of Layers**: 1, 2, or 3 (categorical)
- **Number of Neighbors**: -1 (all neighbors) or 128 (categorical)
- **Weight Decay**: 1e-6 to 0.01 (log-uniform distribution)

## Usage

### Quick Start

1. **Launch parallel hyperparameter tuning**:
   ```bash
   python tmux_hyperparameter_tuning.py
   ```

2. **Monitor progress**:
   ```bash
   tmux attach-session -t gnn_hyperparameter_tuning
   ```

3. **Navigate between windows**:
   - `Ctrl+B + 0-5`: Switch to specific architecture window
   - `Ctrl+B + 6`: Switch to status/monitoring window
   - `Ctrl+B + d`: Detach (tuning continues in background)

### Individual Architecture Tuning

You can also run hyperparameter tuning for a specific architecture:

```bash
python run_optuna_hyperparameter_tuning.py --gnn_architecture hetero-graphsage --torch_device cuda:9
```

### Options

- `--n_trials`: Number of Optuna trials (default: 50)
- `--timeout`: Timeout per trial in seconds (default: 3600)
- `--torch_device`: GPU device to use

## GPU Allocation

Default GPU mapping:
- `hetero-graphsage`: cuda:9
- `hetero-gin`: cuda:8
- `hetero-graphconv`: cuda:7
- `hetero-gat`: cuda:6
- `hetero-gatv2`: cuda:5
- `relgnn`: cuda:4

## Results

### File Locations

- **JSON Results**: `results/hyperparameter_tuning/hyperparameter_results_<architecture>.json`
- **Optuna Database**: `results/hyperparameter_tuning/optuna_studies.db`

### Analysis

Load and analyze results:

```python
import optuna
import json

# Load specific study
study = optuna.load_study(
    study_name="gnn_study_hetero_graphsage", 
    storage="sqlite:///results/hyperparameter_tuning/optuna_studies.db"
)

print("Best trial:", study.best_trial.number)
print("Best score:", study.best_value)
print("Best params:", study.best_trial.params)

# Load summary results
with open("results/hyperparameter_tuning/hyperparameter_results_hetero_graphsage.json") as f:
    results = json.load(f)
```

### Optimization Metric

- **Regression tasks** (rossmann, walmart): Minimize MAE (Mean Absolute Error)
- **Classification tasks** (airbnb-simplified, f1, Berka): Maximize AUC (Area Under Curve)

The final score is the average across all datasets, with classification metrics negated since Optuna minimizes.

## Monitoring

### Real-time Monitoring

In the tmux status window (window 6), you can monitor progress:

```bash
# Check result files
ls -la results/hyperparameter_tuning/

# Check Optuna study progress
python -c "
import optuna
study = optuna.load_study(study_name='gnn_study_hetero_graphsage', storage='sqlite:///results/hyperparameter_tuning/optuna_studies.db')
print(f'Trials completed: {len(study.trials)}')
print(f'Best score so far: {study.best_value:.4f}')
"
```

### Window Layout

- **Window 0**: graphsage [cuda:9]
- **Window 1**: gin [cuda:8]
- **Window 2**: graphconv [cuda:7]
- **Window 3**: gat [cuda:6]
- **Window 4**: gatv2 [cuda:5]
- **Window 5**: relgnn [cuda:4]
- **Window 6**: status (monitoring)

## Expected Runtime

- **Per trial**: ~5-10 minutes (depends on dataset size and architecture)
- **50 trials per architecture**: ~4-8 hours per architecture
- **Total runtime**: ~24-48 hours for all architectures (running in parallel)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in the experiment configuration
2. **Import errors**: Ensure all dependencies are installed in the conda environment
3. **Database lock**: Only one process should write to the same Optuna study

### Debugging

Check individual window logs in tmux for detailed error messages:
```bash
tmux attach-session -t gnn_hyperparameter_tuning
# Switch to the problematic architecture window
# Check stdout/stderr output
```