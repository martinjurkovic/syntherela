#!/usr/bin/env python3

import argparse
import ast
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import optuna
from dotenv import load_dotenv

load_dotenv()

"""
Optuna-based Hyperparameter Tuning for GNNs

This script performs hyperparameter optimization for a specific GNN architecture
on a specific dataset using Optuna. 

The search space includes:
- Learning rate: 0.0001, 0.001, 0.01, 0.1 (categorical)
- Number of layers: 1, 2, 3
- Number of neighbors: -1 (all), 128
- Weight decay: 0.0 to 0.01 (log uniform)
- MLP layers: 1, 2, 3

Usage:
    python run_optuna_hyperparameter_tuning.py --gnn_architecture hetero-graphsage --dataset rossmann_subsampled --torch_device cuda:9
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Original datasets with their task configurations
DATASET_CONFIGS = {
    "rossmann_subsampled": {
        "task_type": "REGRESSION",
        "entity_table": "historical",
        "target_col": "Customers",
        "task": "autocomplete",
    },
    "walmart_subsampled": {
        "task_type": "REGRESSION",
        "entity_table": "depts",
        "target_col": "Weekly_Sales",
        "task": "autocomplete",
    },
    "airbnb-simplified_subsampled": {
        "task_type": "BINARY_CLASSIFICATION",
        "entity_table": "users",
        "target_col": "country_destination",
        "task": "autocomplete",
    },
    "f1_subsampled": {
        "task_type": "BINARY_CLASSIFICATION",
        "task": "driver-top3",
    },
    "Berka_subsampled": {
        "task_type": "BINARY_CLASSIFICATION",
        "entity_table": "loan",
        "target_col": "status",
        "task": "autocomplete",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter tuning for GNNs')
    parser.add_argument('--gnn_architecture', type=str, required=True,
                        choices=["hetero-graphsage", "hetero-gin", "hetero-graphconv", 
                                "hetero-gat", "hetero-gatv2", "relgnn"],
                        help='GNN architecture to tune')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to tune on')
    parser.add_argument('--torch_device', type=str, default='cuda:9',
                        help='GPU device to use (e.g., cuda:7)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per trial in seconds (default: 1 hour)')
    return parser.parse_args()

def run_gnn_experiment(dataset: str, gnn_architecture: str, torch_device: str, 
                      lr: float, num_layers: int, num_neighbors: int, 
                      weight_decay: float, mlp_layers: int, run_id: int = 1) -> Dict[str, float]:
    """Run a single GNN experiment and return metrics"""
    
    config = DATASET_CONFIGS[dataset]
    
    # Base command arguments
    cmd_args = [
        "python", 
        "experiments/evaluation/rdl_utility/run_gnn.py",
        "--dataset", dataset,
        "--gnn_architecture", gnn_architecture,
        "--method", "ORIGINAL",
        "--run_id", str(run_id),
        "--torch_device", torch_device,
        "--lr", str(lr),
        "--num_layers", str(num_layers),
        "--num_neighbors", str(num_neighbors),
        "--weight_decay", str(weight_decay),
        "--mlp_layers", str(mlp_layers),
        "--epochs", "30",  # Reasonable number for hyperparameter tuning
        "--max_steps_per_epoch", "1000",  # Faster iterations for tuning
        "--task_type", config["task_type"],
        "--task", config["task"],
    ]
    
    # Add dataset-specific arguments
    if "entity_table" in config:
        cmd_args.extend(["--entity_table", config["entity_table"]])
    if "target_col" in config:
        cmd_args.extend(["--target_col", config["target_col"]])
    
    try:
        # Run the experiment
        result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=3600)
        
        # Clean up temporary torch_geometric files (same as utility benchmark)
        subprocess.run(["rm", "-f", "torch_geometric.*"])
        
        if result.returncode != 0:
            print(f"GNN experiment failed for {dataset}")
            print(f"Error: {result.stderr}")
            return {"error": "experiment_failed"}
        
        # Parse the output to extract metrics (same as utility benchmark)
        try:
            lines = result.stdout.splitlines()
            final_line = lines[-1]
            
            best_test_metrics = final_line.split("Best test metrics: ")[1]
            # Convert string to dictionary
            metrics = ast.literal_eval(best_test_metrics)
            return metrics
            
        except Exception as parse_error:
            print(f"Failed to parse output for {dataset}")
            print(f"Parse error: {parse_error}")
            print(f"Final line: {lines[-1] if lines else 'No output'}")
            return {"error": "parse_failed"}
        
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out for {dataset}")
        return {"error": "timeout"}
    except Exception as e:
        print(f"Exception during experiment for {dataset}: {e}")
        return {"error": str(e)}

def objective(trial, gnn_architecture: str, dataset: str, torch_device: str) -> float:
    """Optuna objective function"""
    
    # Define hyperparameter search space
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    num_neighbors = trial.suggest_categorical('num_neighbors', [-1, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 0.01, log=True)
    mlp_layers = trial.suggest_categorical('mlp_layers', [1, 2, 3])
    # Store hyperparameters in trial
    trial.set_user_attr('hyperparameters', {
        'lr': lr,
        'num_layers': num_layers,
        'num_neighbors': num_neighbors,
        'weight_decay': weight_decay,
        'mlp_layers': mlp_layers
    })
    
    print(f"Running trial {trial.number} for {dataset} with {gnn_architecture}")
    print(f"Hyperparameters: lr={lr:.4f}, layers={num_layers}, neighbors={num_neighbors}, decay={weight_decay:.6f}, mlp_layers={mlp_layers}")
    
    # Run experiment on the specific dataset
    metrics = run_gnn_experiment(
        dataset=dataset,
        gnn_architecture=gnn_architecture,
        torch_device=torch_device,
        lr=lr,
        num_layers=num_layers,
        num_neighbors=num_neighbors,
        weight_decay=weight_decay,
        mlp_layers=mlp_layers
    )
    
    # Store results in trial for later analysis
    trial.set_user_attr('results', metrics)
    
    if "error" in metrics:
        print(f"Error in {dataset}: {metrics['error']}")
        # Return a bad score for failed experiments
        return float('inf')
    
    # Choose the appropriate metric to optimize based on task type
    config = DATASET_CONFIGS[dataset]
    if config["task_type"] == "REGRESSION":
        # For regression, we want to minimize MAE or RMSE
        if "mae" in metrics:
            score = metrics["mae"]  # Lower is better
        elif "rmse" in metrics:
            score = metrics["rmse"]  # Lower is better
        elif "test_mae" in metrics:
            score = metrics["test_mae"]  # Lower is better
        elif "test_rmse" in metrics:
            score = metrics["test_rmse"]  # Lower is better
        else:
            print(f"Warning: No regression metric found in {metrics.keys()}")
            score = float('inf')  # No valid metric found
    else:
        # For classification, we want to maximize AUC or accuracy
        if "roc_auc" in metrics:
            score = -metrics["roc_auc"]  # Negative because Optuna minimizes
        elif "accuracy" in metrics:
            score = -metrics["accuracy"]  # Negative because Optuna minimizes
        elif "f1" in metrics:
            score = -metrics["f1"]  # Negative because Optuna minimizes
        elif "test_auc" in metrics:
            score = -metrics["test_auc"]  # Negative because Optuna minimizes
        elif "test_accuracy" in metrics:
            score = -metrics["test_accuracy"]  # Negative because Optuna minimizes
        elif "test_f1" in metrics:
            score = -metrics["test_f1"]  # Negative because Optuna minimizes
        else:
            print(f"Warning: No classification metric found in {metrics.keys()}")
            score = float('inf')  # No valid metric found
    
    print(f"Trial {trial.number} completed. Score: {score:.4f}")
    
    return score

def main():
    args = parse_args()
    
    print(f"=== Optuna Hyperparameter Tuning ===")
    print(f"Architecture: {args.gnn_architecture}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.torch_device}")
    print(f"Trials: {args.n_trials}")
    print(f"Timeout per trial: {args.timeout}s")
    print("=" * 50)
    
    # Create results directory
    results_dir = os.path.join(PROJECT_PATH, "results", "hyperparameter_tuning")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create Optuna study with unique name for architecture + dataset combination
    study_name = f"gnn_study_{args.gnn_architecture.replace('-', '_')}_{args.dataset.replace('-', '_')}_categorical_lr"
    storage_url = f"sqlite:///{os.path.join(results_dir, 'optuna_studies.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"Created/loaded study: {study_name}")
    print(f"Storage: {storage_url}")
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, args.gnn_architecture, args.dataset, args.torch_device),
            n_trials=args.n_trials,
            timeout=args.timeout * args.n_trials,  # Total timeout
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
    
    # Save results
    print("\n=== Optimization Complete ===")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save detailed results
    results_file = os.path.join(results_dir, f"hyperparameter_results_{args.gnn_architecture.replace('-', '_')}_{args.dataset.replace('-', '_')}.json")
    
    results_data = {
        "gnn_architecture": args.gnn_architecture,
        "dataset": args.dataset,
        "device": args.torch_device,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_score": study.best_value,
        "best_hyperparameters": study.best_trial.params,
        "best_results": study.best_trial.user_attrs.get('results', {}),
        "study_name": study_name,
        "storage_url": storage_url
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Optuna database: {storage_url}")
    print(f"\nTo analyze results later:")
    print(f"  import optuna")
    print(f"  study = optuna.load_study(study_name='{study_name}', storage='{storage_url}')")
    print(f"  print(study.best_trial)")

if __name__ == "__main__":
    main()