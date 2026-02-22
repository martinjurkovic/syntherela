#!/usr/bin/env python3

import os
import json
import glob


"""
Merge Results Script

Merges individual dataset result files into a single combined results file.
Run this after the tmux benchmark completes.
"""

PROJECT_PATH = __file__.split("experiments")[0]

def merge_results():
    """Merge all dataset-specific result files into a single file"""
    results_dir = os.path.join(PROJECT_PATH, "results", "rdl_utility")

    # Find all dataset-specific result files
    pattern = os.path.join(results_dir, "gnn_utility_results_*_subsampled.json")
    result_files = glob.glob(pattern)

    if not result_files:
        print("No dataset-specific result files found!")
        print(f"Looking for pattern: {pattern}")
        return None

    print(f"Found {len(result_files)} result files:")
    for file in result_files:
        print(f"  {os.path.basename(file)}")

    merged_results = {}

    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                dataset_results = json.load(f)

            # Merge into main results
            for dataset_name, methods in dataset_results.items():
                if dataset_name not in merged_results:
                    merged_results[dataset_name] = {}
                merged_results[dataset_name].update(methods)

            print(f"✓ Merged: {os.path.basename(result_file)}")

        except Exception as e:
            print(f"✗ Error reading {result_file}: {e}")

    # Save merged results in the same directory
    merged_file = os.path.join(results_dir, "gnn_utility_results_merged.json")

    try:
        with open(merged_file, "w") as f:
            json.dump(merged_results, f, indent=4)

        print(f"\n✓ Merged results saved to: {merged_file}")

        # Print summary
        total_experiments = 0
        for dataset in merged_results:
            for method in merged_results[dataset]:
                for gnn_arch in merged_results[dataset][method]:
                    for run_id in merged_results[dataset][method][gnn_arch]:
                        if merged_results[dataset][method][gnn_arch][run_id] != {}:
                            total_experiments += 1

        print(f"✓ Total experiments in merged file: {total_experiments}")
        return merged_file

    except Exception as e:
        print(f"✗ Error saving merged file: {e}")
        return None

def main():
    print("=== Merging GNN Benchmark Results ===")
    print()

    merged_file = merge_results()

    if merged_file:
        print("\n" + "="*50)
        print("Merge completed successfully!")
        print(f"Combined results available at: {merged_file}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Merge failed!")
        print("="*50)

if __name__ == "__main__":
    main()
