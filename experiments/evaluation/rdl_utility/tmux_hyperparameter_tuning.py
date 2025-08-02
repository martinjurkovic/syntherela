#!/usr/bin/env python3

import os
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

"""
Tmux-based GNN Hyperparameter Tuning Script with Optuna

Creates a tmux session with 6 windows, one for each GNN architecture.
Each window runs hyperparameter tuning for its assigned architecture on original datasets.
Uses Optuna for efficient hyperparameter optimization.

Usage:
    python tmux_hyperparameter_tuning.py

Then attach to the session:
    tmux attach-session -t gnn_hyperparameter_tuning
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

# GNN architecture to GPU mapping
ARCHITECTURE_GPU_MAPPING = {
    "hetero-graphsage": "cuda:9",
    "hetero-gin": "cuda:8", 
    "hetero-graphconv": "cuda:7",
    "hetero-gat": "cuda:6",
    "hetero-gatv2": "cuda:5",
    "relgnn": "cuda:4",
}

# Datasets to tune on
DATASETS = [
    "rossmann_subsampled",
    "walmart_subsampled", 
    "airbnb-simplified_subsampled",
    "f1_subsampled",
    "Berka_subsampled",
]

def run_tmux_command(cmd):
    """Run a tmux command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Tmux command failed: {cmd}")
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def create_tmux_session(session_name):
    """Create or recreate a tmux session"""
    
    # Kill existing session if it exists
    subprocess.run(f"tmux kill-session -t {session_name}", shell=True, capture_output=True)
    
    # Create new session (detached)
    if not run_tmux_command(f"tmux new-session -d -s {session_name}"):
        print(f"Failed to create tmux session: {session_name}")
        return False
    
    print(f"✓ Created tmux session: {session_name}")
    return True

def setup_dataset_window(session_name, window_index, architecture, dataset, gpu_device):
    """Set up a window for a specific architecture-dataset combination"""
    
    # Create new window (except for the first one which already exists)
    if window_index > 0:
        if not run_tmux_command(f"tmux new-window -t {session_name}"):
            print(f"Failed to create window for {architecture}-{dataset}")
            return False
    
    # Rename the window
    dataset_short = dataset.replace("_subsampled", "").replace("-simplified", "")
    if not run_tmux_command(f"tmux rename-window -t {session_name}:{window_index} '{dataset_short}'"):
        print(f"Failed to rename window for {dataset}")
    
    # Create the hyperparameter tuning command
    tuning_cmd = (
        f"cd {PROJECT_PATH} && "
        f"source ../.zshrc && "
        f"conda activate syntherela_new && "
        f"echo 'Starting hyperparameter tuning for {architecture} on {dataset} using {gpu_device}' && "
        f"python experiments/evaluation/rdl_utility/run_optuna_hyperparameter_tuning.py "
        f"--gnn_architecture {architecture} --dataset {dataset} --torch_device {gpu_device} "
        f"--n_trials 30 && "
        f"echo 'COMPLETED {architecture}-{dataset}' || "
        f"echo 'FAILED {architecture}-{dataset}'"
    )
    
    # Send the command to the window
    if not run_tmux_command(f"tmux send-keys -t {session_name}:{window_index} '{tuning_cmd}' Enter"):
        print(f"Failed to send command to window for {architecture}-{dataset}")
        return False
    
    print(f"✓ Set up window {window_index}: {dataset} → {gpu_device}")
    return True

def create_status_window(session_name, architecture):
    """Create a status/monitoring window for an architecture"""
    # Create status window
    run_tmux_command(f"tmux new-window -t {session_name}")
    run_tmux_command(f"tmux rename-window -t {session_name} 'status'")
    
    # Status command with monitoring
    status_cmd = (
        f"cd {PROJECT_PATH} && "
        f"source ../.zshrc && "
        f"conda activate syntherela_new && "
        f"echo 'Status window for {architecture}' && "
        f"echo 'Check results: ls -la results/hyperparameter_tuning/' && "
        f"echo 'Use Ctrl+B + number to switch between dataset windows' && "
        f"echo '' && "
        f"echo 'Monitor progress for each dataset:' && "
        f"echo 'ls -la results/hyperparameter_tuning/hyperparameter_results_{architecture.replace('-', '_')}_*.json'"
    )
    
    run_tmux_command(f"tmux send-keys -t {session_name} '{status_cmd}' Enter")
    print(f"✓ Set up status monitoring window for {architecture}")

def main():
    print("=== Tmux GNN Hyperparameter Tuning Setup ===")
    print(f"Setting up parallel hyperparameter tuning across {len(ARCHITECTURE_GPU_MAPPING)} GPUs")
    print(f"Datasets: {DATASETS}")
    print(f"GNN architectures: {list(ARCHITECTURE_GPU_MAPPING.keys())}")
    print(f"Total combinations: {len(ARCHITECTURE_GPU_MAPPING)} architectures × {len(DATASETS)} datasets = {len(ARCHITECTURE_GPU_MAPPING) * len(DATASETS)} studies")
    print()
    
    # Create separate tmux session for each architecture
    architectures = list(ARCHITECTURE_GPU_MAPPING.items())
    
    for architecture, gpu_device in architectures:
        session_name = f"gnn_hp_{architecture.replace('hetero-', '').replace('-', '_')}"
        
        # Create tmux session for this architecture
        if not create_tmux_session(session_name):
            print(f"Failed to create session for {architecture}")
            continue
        
        # Set up a window for each dataset
        for i, dataset in enumerate(DATASETS):
            if not setup_dataset_window(session_name, i, architecture, dataset, gpu_device):
                print(f"Failed to set up window for {architecture}-{dataset}")
                continue
        
        # Create status window
        create_status_window(session_name, architecture)
        
        # Switch to status window
        run_tmux_command(f"tmux select-window -t {session_name}:5")
        
        print(f"✓ Set up session '{session_name}' for {architecture}")
    
    print()
    print("=" * 80)
    print("✓ Tmux hyperparameter tuning sessions created successfully!")
    print()
    print("To monitor the hyperparameter tuning:")
    
    for architecture, gpu in architectures:
        session_name = f"gnn_hp_{architecture.replace('hetero-', '').replace('-', '_')}"
        print(f"  tmux attach-session -t {session_name}  # {architecture} [{gpu}]")
    
    print()
    print("Session layout (each session has these windows):")
    for i, dataset in enumerate(DATASETS):
        dataset_short = dataset.replace("_subsampled", "").replace("-simplified", "")
        print(f"  {i}: {dataset_short}")
    print(f"  5: status (monitoring)")
    print()
    print("In tmux:")
    print("  Ctrl+B + number   - Switch to window")
    print("  Ctrl+B + d        - Detach (tuning continues running)")
    print("  Ctrl+B + s        - List all sessions")
    print()
    print("Results will be saved to:")
    print(f"  {PROJECT_PATH}/results/hyperparameter_tuning/")
    print("  Files: hyperparameter_results_<architecture>_<dataset>.json")
    print("  SQLite database: optuna_studies.db")
    print()
    print("Hyperparameter search space:")
    print("  - Learning rate: 0.0001, 0.001, 0.01, 0.1 (categorical)")
    print("  - Num layers: 1, 2, 3")
    print("  - Num neighbors: -1, 128")
    print("  - Weight decay: 0.0 to 0.01 (log uniform)")
    print("  - Trials per architecture-dataset: 30")
    print("=" * 80)

if __name__ == "__main__":
    main()