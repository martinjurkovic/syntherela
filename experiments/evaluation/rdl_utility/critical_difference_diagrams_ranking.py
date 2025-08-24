import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
import glob
from dotenv import load_dotenv

# Note: This script generates critical difference diagrams for each dataset
# comparing all GNN-method pairs across the datasets
# 
# Critical Difference (CD) diagrams show statistical significance of differences
# between methods using the Nemenyi post-hoc test after Friedman test
#
# Install required packages:
# pip install matplotlib seaborn scipy pandas numpy

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Configuration
USE_HYPERPARAMETER_TUNING_RESULTS = True
ALPHA = 0.05  # Significance level for statistical tests

# Define which metric to use for each dataset
dataset_metrics = {
    "rossmann_subsampled": "mae",
    "walmart_subsampled": "mae", 
    "airbnb-simplified_subsampled": "roc_auc",
    "Berka_subsampled": "roc_auc",
    "f1_subsampled": "roc_auc",
}

# Read results data (same as in rdl_utility.py)
results_dir = os.path.join(PROJECT_PATH, "results", "rdl_utility")
results_file = os.path.join(results_dir, "gnn_utility_results_merged.json")

with open(results_file, "r") as f:
    data = json.load(f)

if USE_HYPERPARAMETER_TUNING_RESULTS:
    # Override ORIGINAL method results with hyperparameter tuning results
    hyperparameter_dir = os.path.join(PROJECT_PATH, "results", "hyperparameter_tuning_100")
    hyperparameter_files = glob.glob(os.path.join(hyperparameter_dir, "hyperparameter_results_*.json"))
    
    for file_path in hyperparameter_files:
        with open(file_path, "r") as f:
            result = json.load(f)
        
        dataset = result["dataset"]
        gnn_arch = result["gnn_architecture"]
        best_results = result["best_results"]
        
        # Initialize nested structure if needed for this dataset
        if dataset not in data:
            data[dataset] = {}
        if "ORIGINAL" not in data[dataset]:
            data[dataset]["ORIGINAL"] = {}
        if gnn_arch not in data[dataset]["ORIGINAL"]:
            data[dataset]["ORIGINAL"][gnn_arch] = {}
        
        # Override the ORIGINAL method results with hyperparameter tuning results
        data[dataset]["ORIGINAL"][gnn_arch]["1"] = best_results
        data[dataset]["ORIGINAL"][gnn_arch]["2"] = best_results
        data[dataset]["ORIGINAL"][gnn_arch]["3"] = best_results

# Helper functions
def compute_mean_and_se(values):
    mean = np.mean(values)
    if len(values) == 1:
        se = 0.0
    else:
        se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, se

def critical_difference(num_algorithms, num_datasets, alpha=0.05):
    """
    Calculate critical difference for Nemenyi post-hoc test
    """
    q_alpha = {
        0.05: {4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
        0.10: {4: 2.291, 5: 2.420, 6: 2.514, 7: 2.589, 8: 2.650, 9: 2.701, 10: 2.746}
    }
    
    if num_algorithms in q_alpha[alpha]:
        q_val = q_alpha[alpha][num_algorithms]
    else:
        # Approximate for larger numbers
        q_val = 2.576  # Conservative estimate
    
    cd = q_val * np.sqrt((num_algorithms * (num_algorithms + 1)) / (6.0 * num_datasets))
    return cd

def draw_cd_diagram(ranks, names, cd, title, output_path, scores=None):
    """
    Draw critical difference diagram matching the reference style exactly
    """
    # Sort by rank (best to worst)
    sorted_indices = np.argsort(ranks)
    sorted_ranks = ranks[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    
    # If scores are provided, use them for display, otherwise use ranks
    if scores is not None:
        sorted_scores = [scores[i] for i in sorted_indices]
        display_values = sorted_scores
    else:
        display_values = sorted_ranks
    

    
    n_algorithms = len(sorted_ranks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set up coordinate system
    min_rank = min(sorted_ranks)
    max_rank = max(sorted_ranks)
    rank_range = max_rank - min_rank
    x_margin = rank_range * 0.1
    
    # Main horizontal line position
    y_main = 0.4
    line_start = min_rank - x_margin
    line_end = max_rank + x_margin
    
    # Draw main horizontal line
    ax.plot([line_start, line_end], [y_main, y_main], 'k-', linewidth=2)
    
    # Color mapping by method
    method_colors = {
        'ORIG': '#1f77b4',      # Blue
        'RELDIFF': '#ff7f0e',   # Orange  
        'SDV': '#2ca02c',       # Green
        'RCTGAN': '#d62728',    # Red
        'REALTF': '#9467bd',    # Purple
        'CLAVA': '#8c564b',     # Brown
        'TARGN': '#e377c2',     # Pink
        'RGCLD': '#7f7f7f',     # Gray
    }
    
    # Extract method from algorithm name and assign colors
    def get_method_color(name):
        for method in method_colors.keys():
            if method in name:
                return method_colors[method]
        return '#000000'  # Default black
    
    # Draw dots only (no extra vertical lines)
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        color = get_method_color(name)
        
        # Draw colored dot on the line
        ax.plot(rank, y_main, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Position algorithm names (special left/right pattern)
    left_positions = []
    right_positions = []
    
    # Split algorithms into two halves
    half_point = n_algorithms // 2
    
    # First half goes on left (top to bottom order)
    for i in range(half_point):
        left_positions.append((sorted_ranks[i], sorted_names[i], i, display_values[i]))
    
    # Second half goes on right (bottom to top order - reverse the indices)
    right_half_indices = list(range(half_point, n_algorithms))
    right_half_indices.reverse()  # Reverse for bottom to top
    
    for list_idx, orig_idx in enumerate(right_half_indices):
        right_positions.append((sorted_ranks[orig_idx], sorted_names[orig_idx], orig_idx, display_values[orig_idx]))
    
    # Draw algorithm names on the left
    for idx, (rank, name, orig_idx, score) in enumerate(left_positions):
        color = get_method_color(name)
        y_pos = y_main - 0.15 - (idx * 0.04)
        
        # Draw L-shaped line: vertical down from score position, then horizontal to text
        ax.plot([rank, rank], [y_main, y_pos], color=color, linewidth=1.5)  # Vertical line
        ax.plot([rank, line_start - 0.2], [y_pos, y_pos], color=color, linewidth=1.5)  # Horizontal line
        
        # Add algorithm name with score (or rank if no scores)
        if scores is not None:
            score_text = f"[{score:.3f}]"
            ax.text(line_start - 0.25, y_pos, f"{name} {score_text}", 
                   ha='right', va='center', fontsize=10, color=color, fontweight='bold')
        else:
            rank_text = f"[{rank:.1f}]"
            ax.text(line_start - 0.25, y_pos, f"{name} {rank_text}", 
                   ha='right', va='center', fontsize=10, color=color, fontweight='bold')
    
    # Draw algorithm names on the right  
    for idx, (rank, name, orig_idx, score) in enumerate(right_positions):
        color = get_method_color(name)
        y_pos = y_main - 0.15 - (idx * 0.04)
        
        # Draw L-shaped line: vertical down from score position, then horizontal to text
        ax.plot([rank, rank], [y_main, y_pos], color=color, linewidth=1.5)  # Vertical line
        ax.plot([rank, line_end + 0.2], [y_pos, y_pos], color=color, linewidth=1.5)  # Horizontal line
        
        # Add algorithm name with score (or rank if no scores)
        if scores is not None:
            score_text = f"[{score:.3f}]"
            ax.text(line_end + 0.25, y_pos, f"{score_text} {name}", 
                   ha='left', va='center', fontsize=10, color=color, fontweight='bold')
        else:
            rank_text = f"[{rank:.1f}]"
            ax.text(line_end + 0.25, y_pos, f"{rank_text} {name}", 
                   ha='left', va='center', fontsize=10, color=color, fontweight='bold')
    
    # Find and draw significance groups based on critical difference
    groups = []
    
    # Use proper critical difference method based on rank differences
    # Two algorithms are in the same group if their rank difference is <= CD
    used = [False] * n_algorithms
    
    for i in range(n_algorithms):
        if used[i]:
            continue
        
        # Start a new group with algorithm i
        current_group = [i]
        used[i] = True
        
        # Find all algorithms whose rank difference from algorithm i is <= CD
        for j in range(i + 1, n_algorithms):
            if used[j]:
                continue
            
            # Check if rank difference is within critical difference
            rank_diff = abs(sorted_ranks[j] - sorted_ranks[i])
            if rank_diff <= cd:
                current_group.append(j)
                used[j] = True
        
        # Only add groups with more than one member
        if len(current_group) > 1:
            groups.append(current_group)
    
    # Draw significance brackets below the line (between the connecting lines)
    for group_idx, group in enumerate(groups):
        if len(group) > 1:
            start_rank = sorted_ranks[group[0]]
            end_rank = sorted_ranks[group[-1]]
            
            # Position brackets below the main line, staggered by group
            bracket_y = y_main - 0.05 - (group_idx % 4) * 0.03
            
            # Draw bracket in black
            ax.plot([start_rank, end_rank], [bracket_y, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
            ax.plot([start_rank, start_rank], [y_main - 0.02, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
            ax.plot([end_rank, end_rank], [y_main - 0.02, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
    
    # Add scale values above the main horizontal line
    # Calculate scale values: start with minimum, then multiples of 5 up to maximum
    min_rank = min(sorted_ranks)
    max_rank = max(sorted_ranks)
    
    # Start with the minimum value
    scale_values = [min_rank]
    
    # Find the next multiple of 5 after the minimum
    next_multiple = int(np.ceil(min_rank / 5.0) * 5)
    
    # Add multiples of 5 up to the maximum
    current_value = next_multiple
    while current_value <= max_rank:
        scale_values.append(current_value)
        current_value += 5
    
    # If the maximum isn't already included, check if it's too close to the last value
    if max_rank not in scale_values:
        # If the difference between max and last value is small, replace the last multiple with max
        if len(scale_values) > 1 and abs(max_rank - scale_values[-1]) < 2.0:
            scale_values[-1] = max_rank  # Replace last multiple with max
        else:
            scale_values.append(max_rank)  # Add max as separate value
    
    # Add tick marks and values above the main line
    for value in scale_values:
        # Only draw tick if value is within our range
        if line_start <= value <= line_end:
            # Tick mark above the main line
            ax.plot([value, value], [y_main, y_main + 0.05], 'k-', linewidth=1)
            # Value label above the tick
            ax.text(value, y_main + 0.1, f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Removed top scale bar as requested
    
    # Title removed as requested
    
    # Set plot limits and remove axes
    text_margin = (line_end - line_start) * 0.3
    ax.set_xlim(line_start - text_margin, line_end + text_margin)
    
    # Calculate y limits based on number of algorithms
    max_left = len(left_positions)
    max_right = len(right_positions)
    max_labels = max(max_left, max_right)
    y_bottom = y_main - 0.15 - (max_labels * 0.04) - 0.1
    
    ax.set_ylim(y_bottom, y_main + 0.2)
    
    # Remove all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Extract datasets, methods, gnn architectures
datasets = [
    "rossmann_subsampled",
    "walmart_subsampled", 
    "airbnb-simplified_subsampled",
    "Berka_subsampled",
    "f1_subsampled",
]

# Get all GNN architectures and methods from the data
gnn_architectures = set()
methods = set()
for dataset, method_data in data.items():
    for method, gnn_data in method_data.items():
        methods.add(method)
        for gnn_arch in gnn_data.keys():
            gnn_architectures.add(gnn_arch)

gnn_architectures = sorted(list(gnn_architectures))
methods = sorted(list(methods))

# Process results: dataset -> gnn_arch -> method -> (mean, se)
results = {}
for dataset in datasets:
    results[dataset] = {}
    for gnn_arch in gnn_architectures:
        results[dataset][gnn_arch] = {}
        for method in methods:
            if (dataset in data and 
                method in data[dataset] and 
                gnn_arch in data[dataset][method]):
                
                runs = data[dataset][method][gnn_arch]
                metric_values = []
                
                for run in runs.values():
                    if isinstance(run, dict) and run:
                        metric_name = dataset_metrics.get(dataset, "mae")
                        if metric_name in run:
                            metric_values.append(run[metric_name])
                
                if metric_values:
                    mean_value, se_value = compute_mean_and_se(metric_values)
                    results[dataset][gnn_arch][method] = (mean_value, se_value)

# Method and architecture renaming for better display
method_rename = {
    "ORIGINAL": "ORIG",
    "RELDIFF": "RELDIFF", 
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTF",
    "CLAVADDPM": "CLAVA",
    "MOSTLYAI": "TARGN",
    "RGCLD": "RGCLD",
}

gnn_arch_rename = {
    "hetero-graphsage": "GSAGE",
    "hetero-gin": "GIN",
    "hetero-graphconv": "GCONV", 
    "hetero-gat": "GAT",
    "hetero-gatv2": "GATv2",
    "relgnn": "RelGNN",
}

dataset_rename = {
    "f1_subsampled": "F1",
    "Berka_subsampled": "Berka", 
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
}

# Create output directory
output_dir = os.path.join(PROJECT_PATH, "experiments", "evaluation", "rdl_utility", "cd_diagrams_ranking")
os.makedirs(output_dir, exist_ok=True)

print("Generating Critical Difference Diagrams...")
print("=" * 50)

# Generate CD diagram for each dataset
for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    
    # Collect all GNN-method combinations that have results for this dataset
    algorithm_results = []
    algorithm_names = []
    algorithm_std_errors = []
    
    for gnn_arch in gnn_architectures:
        for method in methods:
            if (dataset in results and 
                gnn_arch in results[dataset] and 
                method in results[dataset][gnn_arch]):
                
                mean_value, se_value = results[dataset][gnn_arch][method]
                algorithm_results.append(mean_value)
                algorithm_std_errors.append(se_value)
                
                # Create readable name for GNN-method combination
                gnn_display = gnn_arch_rename.get(gnn_arch, gnn_arch)
                method_display = method_rename.get(method, method)
                algorithm_names.append(f"{gnn_display}-{method_display}")
    
    if len(algorithm_results) < 4:
        print(f"  Skipping {dataset}: insufficient algorithms ({len(algorithm_results)} < 4)")
        continue
    
    # For a single dataset, rank the algorithms by their performance
    algorithm_results = np.array(algorithm_results)
    
    # Determine ranking (lower is better for MAE, higher is better for AUC)
    metric_type = dataset_metrics.get(dataset, "mae")
    if metric_type == "roc_auc":
        # Higher is better - rank in descending order
        ranks = rankdata(-algorithm_results, method='average')
    else:
        # Lower is better - rank in ascending order  
        ranks = rankdata(algorithm_results, method='average')
    
    # For single dataset, we can't use proper Nemenyi critical difference
    # Instead, use a conservative threshold based on algorithm count
    # This is for visualization purposes only - not statistically rigorous
    num_algorithms = len(algorithm_results)
    # Use a more conservative threshold for single dataset
    cd_heuristic = 2.0  # Conservative fixed threshold for visualization
    
    # Create the diagram
    dataset_display = dataset_rename.get(dataset, dataset)
    metric_display = "AUC (higher is better)" if metric_type == "roc_auc" else "MAE (lower is better)"
    title = f"Algorithm Ranking for {dataset_display}\n({metric_display})"
    
    output_path = os.path.join(output_dir, f"cd_diagram_{dataset}.png")
    
    print(f"  Creating diagram with {num_algorithms} algorithms")
    print(f"  Heuristic CD threshold: {cd_heuristic:.3f}")
    
    draw_cd_diagram(ranks, algorithm_names, cd_heuristic, title, output_path, scores=algorithm_results)
    
    # Also save ranking data
    ranking_data = pd.DataFrame({
        'Algorithm': algorithm_names,
        'Performance': algorithm_results,
        'Rank': ranks
    }).sort_values('Rank')
    
    ranking_path = os.path.join(output_dir, f"rankings_{dataset}.csv")
    ranking_data.to_csv(ranking_path, index=False)
    
    print(f"  Saved diagram: {output_path}")
    print(f"  Saved rankings: {ranking_path}")
    print(f"  Top 3 algorithms:")
    for i, (_, row) in enumerate(ranking_data.head(3).iterrows()):
        print(f"    {i+1}. {row['Algorithm']}: {row['Performance']:.4f} (rank {row['Rank']:.1f})")

# Generate combined analysis across all datasets
print(f"\nGenerating combined analysis across all datasets...")

# Create a matrix where rows are GNN-method combinations and columns are datasets
algorithm_names_global = []
performance_matrix = []

# First pass: collect all unique algorithm combinations
all_combinations = set()
for dataset in datasets:
    for gnn_arch in gnn_architectures:
        for method in methods:
            if (dataset in results and 
                gnn_arch in results[dataset] and 
                method in results[dataset][gnn_arch]):
                
                gnn_display = gnn_arch_rename.get(gnn_arch, gnn_arch)
                method_display = method_rename.get(method, method)
                all_combinations.add((gnn_arch, method, f"{method_display}-{gnn_display}"))

all_combinations = sorted(list(all_combinations))

# Create performance matrix
for gnn_arch, method, display_name in all_combinations:
    algorithm_names_global.append(display_name)
    performance_row = []
    
    for dataset in datasets:
        if (dataset in results and 
            gnn_arch in results[dataset] and 
            method in results[dataset][gnn_arch]):
            
            mean_value, _ = results[dataset][gnn_arch][method]
            performance_row.append(mean_value)
        else:
            performance_row.append(np.nan)
    
    performance_matrix.append(performance_row)

performance_matrix = np.array(performance_matrix)

# Filter out algorithms that don't have results for all datasets
complete_mask = ~np.isnan(performance_matrix).any(axis=1)
complete_algorithms = [name for i, name in enumerate(algorithm_names_global) if complete_mask[i]]
complete_performance = performance_matrix[complete_mask]

if len(complete_algorithms) >= 4:
    print(f"Found {len(complete_algorithms)} algorithms with complete results across all datasets")
    
    # Calculate ranks for each dataset
    rank_matrix = np.zeros_like(complete_performance)
    for j, dataset in enumerate(datasets):
        metric_type = dataset_metrics.get(dataset, "mae")
        if metric_type == "roc_auc":
            # Higher is better
            rank_matrix[:, j] = rankdata(-complete_performance[:, j], method='average')
        else:
            # Lower is better
            rank_matrix[:, j] = rankdata(complete_performance[:, j], method='average')
    
    # Calculate average ranks
    avg_ranks = np.mean(rank_matrix, axis=1)
    
    # Calculate critical difference using Nemenyi test
    num_algorithms = len(complete_algorithms)
    num_datasets = len(datasets)
    cd = critical_difference(num_algorithms, num_datasets, ALPHA)
    
    # Create combined CD diagram
    title = f"Critical Difference Diagram Across All Datasets\n(Nemenyi test, α={ALPHA})"
    output_path = os.path.join(output_dir, "cd_diagram_combined.png")
    
    # For combined analysis, only ranks are meaningful (scores have different scales)
    # Don't pass scores since averaging AUC and MAE values is meaningless
    draw_cd_diagram(avg_ranks, complete_algorithms, cd, title, output_path, scores=None)
    
    # Save combined ranking data
    combined_ranking = pd.DataFrame({
        'Algorithm': complete_algorithms,
        'Average_Rank': avg_ranks
    })
    
    # Add individual dataset ranks
    for j, dataset in enumerate(datasets):
        combined_ranking[f'Rank_{dataset_rename.get(dataset, dataset)}'] = rank_matrix[:, j]
    
    combined_ranking = combined_ranking.sort_values('Average_Rank')
    ranking_path = os.path.join(output_dir, "rankings_combined.csv")
    combined_ranking.to_csv(ranking_path, index=False)
    
    print(f"Saved combined diagram: {output_path}")
    print(f"Saved combined rankings: {ranking_path}")
    print(f"Critical difference threshold: {cd:.3f}")
    print(f"Top 5 algorithms overall:")
    for i, (_, row) in enumerate(combined_ranking.head(5).iterrows()):
        print(f"  {i+1}. {row['Algorithm']}: avg rank {row['Average_Rank']:.2f}")

else:
    print(f"Insufficient algorithms with complete results ({len(complete_algorithms)} < 4)")

print(f"\nAll diagrams saved to: {output_dir}")
print("=" * 50)
