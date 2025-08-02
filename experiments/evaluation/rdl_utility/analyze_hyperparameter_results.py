#!/usr/bin/env python3

import json
import os
from pathlib import Path
import pandas as pd

def analyze_hyperparameter_results(results_dir="results/hyperparameter_tuning"):
    """Analyze all hyperparameter tuning results and create summary"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all result JSON files
    json_files = list(results_path.glob("hyperparameter_results_*.json"))
    
    if not json_files:
        print("No hyperparameter result files found!")
        return
    
    print(f"Found {len(json_files)} result files")
    print("=" * 80)
    
    all_results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            result = {
                'architecture': data['gnn_architecture'],
                'dataset': data['dataset'],
                'n_trials': data['n_trials'],
                'best_score': data['best_score'],
                'best_lr': data['best_hyperparameters']['lr'],
                'best_num_layers': data['best_hyperparameters']['num_layers'],
                'best_num_neighbors': data['best_hyperparameters']['num_neighbors'],
                'best_weight_decay': data['best_hyperparameters']['weight_decay'],
                'best_results': data.get('best_results', {}),
                'file': json_file.name
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid results found!")
        return
    
    # Sort by architecture and dataset
    df = df.sort_values(['architecture', 'dataset'])
    
    print("BEST HYPERPARAMETERS FOR EACH ARCHITECTURE-DATASET COMBINATION:")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n{row['architecture']} + {row['dataset']}:")
        print(f"  Best Score: {row['best_score']:.4f}")
        print(f"  Learning Rate: {row['best_lr']}")
        print(f"  Num Layers: {row['best_num_layers']}")
        print(f"  Num Neighbors: {row['best_num_neighbors']}")
        print(f"  Weight Decay: {row['best_weight_decay']:.6f}")
        print(f"  Trials: {row['n_trials']}")
        # Show detailed metrics if available
        if row['best_results']:
            print(f"  Detailed Metrics: {row['best_results']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY ARCHITECTURE:")
    print("=" * 80)
    
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        print(f"\n{arch}:")
        print(f"  Average Score: {arch_data['best_score'].mean():.4f}")
        print(f"  Best Dataset: {arch_data.loc[arch_data['best_score'].idxmin(), 'dataset']}")
        print(f"  Most Common LR: {arch_data['best_lr'].mode().iloc[0]}")
        print(f"  Most Common Layers: {arch_data['best_num_layers'].mode().iloc[0]}")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY DATASET:")
    print("=" * 80)
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Average Score: {dataset_data['best_score'].mean():.4f}")
        print(f"  Best Architecture: {dataset_data.loc[dataset_data['best_score'].idxmin(), 'architecture']}")
        print(f"  Most Common LR: {dataset_data['best_lr'].mode().iloc[0]}")
    
    # Save summary to CSV
    summary_file = results_path / "hyperparameter_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return df

def get_best_config_for_combination(architecture, dataset, results_dir="results/hyperparameter_tuning"):
    """Get best hyperparameters for specific architecture-dataset combination"""
    
    filename = f"hyperparameter_results_{architecture.replace('-', '_')}_{dataset.replace('-', '_')}.json"
    filepath = Path(results_dir) / filename
    
    if not filepath.exists():
        print(f"Results file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Best hyperparameters for {architecture} + {dataset}:")
    print(f"  Score: {data['best_score']:.4f}")
    print(f"  Learning Rate: {data['best_hyperparameters']['lr']}")
    print(f"  Num Layers: {data['best_hyperparameters']['num_layers']}")
    print(f"  Num Neighbors: {data['best_hyperparameters']['num_neighbors']}")
    print(f"  Weight Decay: {data['best_hyperparameters']['weight_decay']:.6f}")
    
    # Show detailed metrics if available
    if 'best_results' in data and data['best_results']:
        print(f"  Detailed Metrics: {data['best_results']}")
    
    return data['best_hyperparameters']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--architecture', type=str, help='Specific architecture to analyze')
    parser.add_argument('--dataset', type=str, help='Specific dataset to analyze')
    parser.add_argument('--results_dir', type=str, default='results/hyperparameter_tuning', 
                        help='Results directory')
    
    args = parser.parse_args()
    
    if args.architecture and args.dataset:
        # Get specific combination
        get_best_config_for_combination(args.architecture, args.dataset, args.results_dir)
    else:
        # Analyze all results
        analyze_hyperparameter_results(args.results_dir)