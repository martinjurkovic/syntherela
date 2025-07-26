# Parallel GNN Benchmark Scripts

This directory contains scripts for running GNN benchmarks across multiple datasets in parallel using different GPUs.

## Scripts Overview

### 1. `tmux_benchmark.py` ⭐ **MAIN SCRIPT**
Creates a tmux session with 5 windows, one for each dataset, running benchmarks in parallel with real-time visibility.

**GPU Mapping:**
- `rossmann_subsampled` → `cuda:9`
- `walmart_subsampled` → `cuda:7`
- `airbnb-simplified_subsampled` → `cuda:6`
- `f1_subsampled` → `cuda:5`
- `Berka_subsampled` → `cuda:4`

### 2. `run_utility_benchmark.py` (Modified)
The original benchmark script, now with support for dataset filtering and GPU specification.

### 3. `merge_results.py`
Merges individual dataset result files into a single combined results file.

### 4. `test_parallel_benchmark.py`
Quick test script to verify the setup works before running full benchmarks.

## Usage

### Recommended Workflow

1. **Start the parallel benchmarks:**
   ```bash
   python tmux_benchmark.py
   ```

2. **Monitor progress in real-time:**
   ```bash
   tmux attach-session -t gnn_benchmark
   ```

3. **Navigate between datasets:**
   - `Ctrl+B + 0` - Rossmann
   - `Ctrl+B + 1` - Walmart
   - `Ctrl+B + 2` - Airbnb
   - `Ctrl+B + 3` - F1
   - `Ctrl+B + 4` - Berka
   - `Ctrl+B + 5` - Status monitor

4. **Detach (benchmarks continue running):**
   - `Ctrl+B + d`

5. **After completion, merge results:**
   ```bash
   python merge_results.py
   ```

### Alternative: Manual Single Dataset

1. **Run a specific dataset:**
   ```bash
   python run_utility_benchmark.py --dataset_filter rossmann_subsampled --torch_device cuda:9
   ```

## Testing

Before running the full benchmark, test the setup:

```bash
python test_parallel_benchmark.py
```

This runs a quick test (1 epoch, 2 steps) on each dataset to verify everything works.

## Results

### Individual Results
Each dataset creates its own results file:
- `results/rdl_utility/gnn_utility_results_rossmann_subsampled.json`
- `results/rdl_utility/gnn_utility_results_walmart_subsampled.json`
- `results/rdl_utility/gnn_utility_results_airbnb-simplified_subsampled.json`
- `results/rdl_utility/gnn_utility_results_f1_subsampled.json`
- `results/rdl_utility/gnn_utility_results_Berka_subsampled.json`

### Merged Results
After running `merge_results.py`:
- `results/rdl_utility/gnn_utility_results_merged.json`

## Result Structure

```json
{
  "dataset_name": {
    "method_name": {
      "gnn_architecture": {
        "run_id": {
          "metric1": value,
          "metric2": value,
          ...
        }
      }
    }
  }
}
```

## Benchmarked Configurations

**GNN Architectures:**
- `hetero-graphsage` (default)
- `hetero-gin`
- `hetero-graphconv`
- `hetero-gat`
- `hetero-gatv2`
- `relgnn`

**Methods per dataset:**
- `ORIGINAL` (real data)
- `CLAVADDPM`, `MOSTLYAI`, `RCTGAN`, `REALTABFORMER`, `RGCLD`, `SDV` (synthetic)

**Runs:** 3 per combination (for statistical significance)

## Troubleshooting

### Environment Setup
This setup uses **micromamba** instead of conda. The scripts automatically handle:
```bash
source ~/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate syntherela_new
```

### Tmux Commands
- **List sessions:** `tmux list-sessions`
- **Kill session:** `tmux kill-session -t gnn_benchmark`
- **Create new session:** `tmux new-session -s gnn_benchmark`

### GPU Memory Issues
If you encounter GPU memory issues:
1. Reduce batch size in `run_gnn.py`
2. Reduce number of layers
3. Use fewer GNN architectures

### Monitoring Progress
Check results directory:
```bash
ls -la results/rdl_utility/gnn_utility_results_*.json
```

### Resume Failed Datasets
If a dataset fails, you can restart just that dataset:
```bash
python run_utility_benchmark.py --dataset_filter failed_dataset --torch_device cuda:X
```

## Performance

**Expected Runtime:**
- Per dataset: ~2-4 hours (depends on dataset size and number of methods)
- Total (parallel): ~2-4 hours (limited by slowest dataset)
- Total (sequential): ~10-20 hours

**Resource Usage:**
- 5 GPUs in parallel
- Each GPU runs 6 GNN architectures × 7 methods × 3 runs = ~126 experiments
- Total: ~630 experiments across all datasets 