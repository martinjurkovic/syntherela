#!/usr/bin/env python3

import os
import subprocess
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

"""
Single Architecture Hyperparameter Tuning Script

Creates a tmux session for one specific GNN architecture.
Useful for rerunning failed architectures or running on different GPUs.

Usage:
    python run_single_architecture_hyperparameter_tuning.py --architecture hetero-gin --gpu cuda:0
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

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
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for single GNN architecture')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=["hetero-graphsage", "hetero-gin", "hetero-graphconv", 
                                "hetero-gat", "hetero-gatv2", "relgnn"],
                        help='GNN architecture to tune')
    parser.add_argument('--gpu', type=str, required=True,
                        help='GPU device to use (e.g., cuda:0, cuda:7)')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials per dataset')
    
    args = parser.parse_args()
    
    print(f"=== Single Architecture Hyperparameter Tuning ===")
    print(f"Architecture: {args.architecture}")
    print(f"GPU: {args.gpu}")
    print(f"Datasets: {DATASETS}")
    print(f"Trials per dataset: {args.n_trials}")
    print()
    
    # Create session name
    session_name = f"gnn_hp_{args.architecture.replace('hetero-', '').replace('-', '_')}"
    
    # Create tmux session for this architecture
    if not create_tmux_session(session_name):
        print(f"Failed to create session for {args.architecture}")
        return
    
    # Set up a window for each dataset
    for i, dataset in enumerate(DATASETS):
        if not setup_dataset_window(session_name, i, args.architecture, dataset, args.gpu):
            print(f"Failed to set up window for {args.architecture}-{dataset}")
            continue
    
    # Create status window
    create_status_window(session_name, args.architecture)
    
    # Switch to status window
    run_tmux_command(f"tmux select-window -t {session_name}:5")
    
    print()
    print("=" * 60)
    print(f"✓ Tmux session '{session_name}' created successfully!")
    print()
    print("To monitor the hyperparameter tuning:")
    print(f"  tmux attach-session -t {session_name}")
    print()
    print("Window layout:")
    for i, dataset in enumerate(DATASETS):
        dataset_short = dataset.replace("_subsampled", "").replace("-simplified", "")
        print(f"  {i}: {dataset_short}")
    print(f"  5: status (monitoring)")
    print()
    print("In tmux:")
    print("  Ctrl+B + number   - Switch to window")
    print("  Ctrl+B + d        - Detach (tuning continues running)")
    print()
    print("Results will be saved to:")
    print(f"  results/hyperparameter_tuning/hyperparameter_results_{args.architecture.replace('-', '_')}_*.json")
    print("=" * 60)

if __name__ == "__main__":
    main()