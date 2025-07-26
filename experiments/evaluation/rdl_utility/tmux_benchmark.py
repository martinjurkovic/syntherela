#!/usr/bin/env python3

import os
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

"""
Tmux-based GNN Utility Benchmark Script

Creates a tmux session with 5 windows, one for each dataset.
Each window runs the benchmark for its assigned dataset on its assigned GPU.
You can attach to the tmux session to monitor progress in real-time.

Usage:
    python tmux_benchmark.py

Then attach to the session:
    tmux attach-session -t gnn_benchmark
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Dataset to GPU mapping
DATASET_GPU_MAPPING = {
    "rossmann_subsampled": "cuda:9",
    "walmart_subsampled": "cuda:7", 
    "airbnb-simplified_subsampled": "cuda:6",
    "f1_subsampled": "cuda:5",
    "Berka_subsampled": "cuda:4",
}

def run_tmux_command(cmd):
    """Run a tmux command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Tmux command failed: {cmd}")
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def create_tmux_session():
    """Create or recreate the tmux session"""
    session_name = "gnn_benchmark"
    
    # Kill existing session if it exists
    subprocess.run(f"tmux kill-session -t {session_name}", shell=True, capture_output=True)
    
    # Create new session (detached)
    if not run_tmux_command(f"tmux new-session -d -s {session_name}"):
        print("Failed to create tmux session")
        return False
    
    print(f"✓ Created tmux session: {session_name}")
    return True

def setup_dataset_window(session_name, window_index, dataset, gpu_device):
    """Set up a window for a specific dataset"""
    
    # Create new window (except for the first one which already exists)
    if window_index > 0:
        if not run_tmux_command(f"tmux new-window -t {session_name}"):
            print(f"Failed to create window for {dataset}")
            return False
    
    # Rename the window
    window_name = dataset.replace("_subsampled", "").replace("-simplified", "")
    if not run_tmux_command(f"tmux rename-window -t {session_name}:{window_index} '{window_name}[{gpu_device}]'"):
        print(f"Failed to rename window for {dataset}")
    
    # Create the benchmark command
    benchmark_cmd = (
        f"cd {PROJECT_PATH} && "
        f"source ../.zshrc && "
        f"conda activate syntherela_new && "
        f"echo 'Starting {dataset} on {gpu_device}' && "
        f"python experiments/evaluation/rdl_utility/run_utility_benchmark.py "
        f"--dataset_filter {dataset} --torch_device {gpu_device} && "
        f"echo 'COMPLETED {dataset}' || "
        f"echo 'FAILED {dataset}'"
    )
    
    # Send the command to the window
    if not run_tmux_command(f"tmux send-keys -t {session_name}:{window_index} '{benchmark_cmd}' Enter"):
        print(f"Failed to send command to window for {dataset}")
        return False
    
    print(f"✓ Set up window {window_index}: {dataset} → {gpu_device}")
    return True

def create_status_window(session_name):
    """Create a status/monitoring window"""
    # Create status window
    run_tmux_command(f"tmux new-window -t {session_name}")
    run_tmux_command(f"tmux rename-window -t {session_name} 'status'")
    
    # Very simple status command
    status_cmd = (
        f"cd {PROJECT_PATH} && "
        f"source ../.zshrc && "
        f"conda activate syntherela_new && "
        f"echo 'Status window ready. Use Ctrl+B + number to switch windows.' && "
        f"echo 'Check results: ls -la results/rdl_utility/'"
    )
    
    run_tmux_command(f"tmux send-keys -t {session_name} '{status_cmd}' Enter")
    print("✓ Set up status monitoring window")

def main():
    print("=== Tmux GNN Benchmark Setup ===")
    print(f"Setting up parallel benchmarks across {len(DATASET_GPU_MAPPING)} GPUs")
    print()
    
    # Create tmux session
    if not create_tmux_session():
        return
    
    session_name = "gnn_benchmark"
    
    # Set up a window for each dataset
    datasets = list(DATASET_GPU_MAPPING.items())
    for i, (dataset, gpu_device) in enumerate(datasets):
        if not setup_dataset_window(session_name, i, dataset, gpu_device):
            print(f"Failed to set up window for {dataset}")
            return
    
    # Create status window
    create_status_window(session_name)
    
    # Switch to status window
    run_tmux_command(f"tmux select-window -t {session_name}:5")
    
    print()
    print("=" * 60)
    print("✓ Tmux session created successfully!")
    print()
    print("To monitor the benchmarks:")
    print(f"  tmux attach-session -t {session_name}")
    print()
    print("Window layout:")
    for i, (dataset, gpu) in enumerate(datasets):
        window_name = dataset.replace("_subsampled", "").replace("-simplified", "")
        print(f"  {i}: {window_name} [{gpu}]")
    print(f"  5: status (monitoring)")
    print()
    print("In tmux:")
    print("  Ctrl+B + number   - Switch to window")
    print("  Ctrl+B + d        - Detach (benchmarks continue running)")
    print("  Ctrl+B + c        - Create new window")
    print()
    print("Results will be saved to:")
    print(f"  {PROJECT_PATH}/results/gnn_utility_results_<dataset>.json")
    print("=" * 60)

if __name__ == "__main__":
    main() 