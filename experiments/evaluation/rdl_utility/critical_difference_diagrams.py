import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata, friedmanchisquare, studentized_range
import scikit_posthocs as sp
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

def nemenyi_test_with_friedman(rank_matrix, alpha=0.05):
    """
    Perform Friedman test followed by Nemenyi test using Critical Difference (CD) calculation
    
    Formula: CD = q_alpha * sqrt(k(k+1)/(6N))
    where:
    - k = number of models/algorithms
    - N = number of datasets  
    - q_alpha = critical value from Studentized range statistic
    
    Returns significant pairs based on rank differences exceeding CD
    """
    print("Performing Friedman test...")
    
    # Remove rows/columns with all NaN values
    valid_algorithms = ~np.all(np.isnan(rank_matrix), axis=1)
    valid_datasets = ~np.all(np.isnan(rank_matrix), axis=0)
    
    clean_rank_matrix = rank_matrix[valid_algorithms][:, valid_datasets]
    
    # Debug information
    print(f"Original rank matrix shape: {rank_matrix.shape}")
    print(f"Clean rank matrix shape: {clean_rank_matrix.shape}")
    print(f"Average ranks: {np.nanmean(rank_matrix, axis=1)[:10]}...")  # Show first 10
    
    # Check if we have any NaN values remaining (should not happen with pre-filtered data)
    if np.any(np.isnan(clean_rank_matrix)):
        print("Error: NaN values detected in rank matrix even after pre-filtering.")
        print("This indicates the input data was not properly filtered for complete algorithms.")
        return [], None
    
    # Friedman test expects data in format: (n_observations, n_algorithms)
    # Our rank_matrix is (n_algorithms, n_datasets), so we need to transpose
    data_for_friedman = clean_rank_matrix.T
    
    # Perform Friedman test
    try:
        n_datasets, n_algorithms = data_for_friedman.shape
        print(f"Data for Friedman: {n_datasets} datasets, {n_algorithms} algorithms")
        
        # Show sample of the data
        print("Sample rank data (first 5 algorithms, all datasets):")
        for i in range(min(5, n_algorithms)):
            print(f"  Algorithm {i}: {data_for_friedman[:, i]}")
        
        # Friedman chi-square test
        friedman_stat, friedman_p = friedmanchisquare(*[data_for_friedman[:, i] for i in range(n_algorithms)])
        
        print(f"Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.6f}")
        print(f"Critical p-value: {alpha}")
        
        if friedman_p < alpha:
            print(f"Friedman test significant (p < {alpha}), proceeding with Nemenyi CD calculation")
            
            # Step 4: Compute Critical Difference (CD) with Nemenyi test
            # Formula: CD = q_alpha * sqrt(k(k+1)/(6N))
            
            k = n_algorithms  # number of models/algorithms
            N = n_datasets    # number of datasets
            
            # Get critical value from Studentized range statistic using scipy
            # For Nemenyi test, we use the studentized range statistic q_alpha
            # Using infinite degrees of freedom for the error term (standard for Nemenyi)
            q_alpha = studentized_range.ppf(1 - alpha, k, float('inf'))
            print(f"Using scipy.stats.studentized_range for q_alpha with k={k}, df=∞")
            
            # Calculate Critical Difference
            cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
            
            print(f"Nemenyi Critical Difference calculation:")
            print(f"  k (algorithms) = {k}")
            print(f"  N (datasets) = {N}")
            print(f"  q_alpha (α={alpha}) = {q_alpha:.3f}")
            print(f"  CD = {q_alpha:.3f} * sqrt({k}*{k+1}/(6*{N})) = {cd:.6f}")
            
            # Calculate average ranks for each algorithm
            avg_ranks = np.mean(data_for_friedman, axis=0)
            print(f"Average ranks: {avg_ranks}")
            
            # Find significant pairs (rank differences > CD)
            significant_pairs = []
            for i in range(n_algorithms):
                for j in range(i + 1, n_algorithms):
                    rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                    if rank_diff > cd:
                        significant_pairs.append((i, j))
                        print(f"  Significant pair: {i} vs {j}, rank diff = {rank_diff:.6f} > CD = {cd:.6f}")
            
            print(f"Found {len(significant_pairs)} significant pairs from Nemenyi CD test")
            print(f"Critical difference threshold: {cd:.6f}")
            
            return significant_pairs, cd
        else:
            print(f"Friedman test not significant (p = {friedman_p:.6f} >= {alpha})")
            print("No post-hoc testing performed")
            return [], None
            
    except Exception as e:
        print(f"Error in Friedman/Nemenyi test: {e}")
        print("Falling back to no significant pairs")
        return [], None







def draw_cd_diagram(ranks, names, cd, title, output_path, scores=None, significant_pairs=None):
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
    
    # Draw dots only (no labels on each dot)
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        color = get_method_color(name)
        
        # Draw colored dot on the line
        ax.plot(rank, y_main, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Add interval ranking markers based on actual rank values
    min_rank = min(sorted_ranks)
    max_rank = max(sorted_ranks)
    
    # Generate multiples of 5 within the range
    rank_start = int(np.floor(min_rank))
    rank_end = int(np.ceil(max_rank))
    
    intervals = []
    
    # Add multiples of 5 within the actual rank range
    first_multiple = ((rank_start // 5) + 1) * 5  # First multiple of 5 after start
    current = first_multiple
    while current < max_rank:  # Only include if less than actual max rank
        intervals.append(current)
        current += 5
    
    # Always add the actual min and max ranks (not rounded)
    intervals.append(min_rank)
    intervals.append(max_rank)
    
    # Remove duplicates and sort
    intervals = sorted(list(set(intervals)))
    
    # Draw tick marks and labels for these intervals
    for interval in intervals:
        # Draw tick mark and label
        ax.plot([interval, interval], [y_main + 0.05, y_main + 0.1], 'k-', linewidth=1)
        if interval == min_rank or interval == max_rank:
            # Show actual rank value for min/max
            ax.text(interval, y_main + 0.15, f'{interval:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # Show integer for multiples of 5
            ax.text(interval, y_main + 0.15, f'{int(interval)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
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
    
    # Find and draw significance groups
    groups = []
    
    if significant_pairs is not None:
        # Use pairwise significance results to determine groups
        # Create adjacency matrix for non-significant pairs
        adjacency = np.ones((n_algorithms, n_algorithms), dtype=bool)
        np.fill_diagonal(adjacency, True)  # Algorithm is equivalent to itself
        
        # Convert significant pairs from original indices to sorted indices
        # Create mapping from original index to sorted index
        original_to_sorted = {orig_idx: sort_idx for sort_idx, orig_idx in enumerate(sorted_indices)}
        
        # Mark significant pairs as NOT equivalent (using sorted indices)
        for i, j in significant_pairs:
            if i in original_to_sorted and j in original_to_sorted:
                sorted_i = original_to_sorted[i]
                sorted_j = original_to_sorted[j]
                adjacency[sorted_i, sorted_j] = False
                adjacency[sorted_j, sorted_i] = False
                print(f"DEBUG: Marked pair ({i},{j}) -> ({sorted_i},{sorted_j}) as NOT equivalent")
        
        print(f"DEBUG: Adjacency matrix shape: {adjacency.shape}")
        print(f"DEBUG: Adjacency matrix:\n{adjacency}")
        
        # For CD diagrams, we want to find maximal groups where NO algorithms 
        # within the group are significantly different from each other
        
        groups = []
        used = [False] * n_algorithms
        
        # Find all maximal cliques (complete subgraphs) in the adjacency matrix
        for i in range(n_algorithms):
            if used[i]:
                continue
                
            # Start building a group starting with algorithm i
            current_group = [i]
            
            # Try to add more algorithms to this group
            for j in range(i + 1, n_algorithms):
                if used[j]:
                    continue
                    
                # Check if j can be added to current group
                # j can be added only if it's not significantly different from ALL algorithms in current group
                can_add = True
                for k in current_group:
                    if not adjacency[k, j]:  # j is significantly different from k
                        can_add = False
                        break
                
                if can_add:
                    current_group.append(j)
            
            # Mark all algorithms in this group as used
            for alg in current_group:
                used[alg] = True
            
            # Only add groups with more than one member
            if len(current_group) > 1:
                groups.append(current_group)
                print(f"DEBUG: Found valid group: {current_group}")
            else:
                print(f"DEBUG: Algorithm {i} is in its own group (significantly different from others)")
        
        print(f"DEBUG: Total groups found from significant pairs: {len(groups)}")
    else:
        # Use critical difference method based on rank differences
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
    print(f"DEBUG: Drawing {len(groups)} groups: {groups}")
    for group_idx, group in enumerate(groups):
        print(f"DEBUG: Processing group {group_idx}: {group} with {len(group)} members")
        if len(group) > 1:
            start_rank = sorted_ranks[group[0]]
            end_rank = sorted_ranks[group[-1]]
            
            print(f"DEBUG: Drawing bracket from rank {start_rank} to {end_rank}")
            
            # Position brackets below the main line, staggered by group
            bracket_y = y_main - 0.05 - (group_idx % 4) * 0.03
            
            # Draw bracket in black
            ax.plot([start_rank, end_rank], [bracket_y, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
            ax.plot([start_rank, start_rank], [y_main - 0.02, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
            ax.plot([end_rank, end_rank], [y_main - 0.02, bracket_y], 
                   color='black', linewidth=2, alpha=0.8)
        else:
            print(f"DEBUG: Skipping single-member group {group}")
    
    # Top scale removed - rankings now shown directly on dots
    
    # Add title
    ax.text((line_start + line_end) / 2, y_main + 0.5, title, 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Set plot limits and remove axes
    text_margin = (line_end - line_start) * 0.3
    ax.set_xlim(line_start - text_margin, line_end + text_margin)
    
    # Calculate y limits based on number of algorithms
    max_left = len(left_positions)
    max_right = len(right_positions)
    max_labels = max(max_left, max_right)
    y_bottom = y_main - 0.15 - (max_labels * 0.04) - 0.1
    
    ax.set_ylim(y_bottom, y_main + 0.6)
    
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
output_dir = os.path.join(PROJECT_PATH, "experiments", "evaluation", "rdl_utility", "cd_diagrams")
os.makedirs(output_dir, exist_ok=True)

print("Generating Critical Difference Diagrams...")
print("=" * 50)


# Generate separate analysis for each GNN architecture
print(f"\nGenerating separate analysis for each GNN architecture...")

print(f"DEBUG: Available GNN architectures: {gnn_architectures}")
print(f"DEBUG: Available methods: {methods}")
print(f"DEBUG: Available datasets: {datasets}")

# Debug: Print a sample of the data structure
print(f"DEBUG: Sample data structure:")
for dataset in list(datasets)[:2]:  # Just show first 2 datasets
    print(f"  {dataset}:")
    if dataset in results:
        for method in list(results[dataset].keys())[:2]:  # Just show first 2 methods
            print(f"    {method}: {list(results[dataset][method].keys())}")
    else:
        print(f"    NOT FOUND in results")

for gnn_arch in gnn_architectures:
    print(f"\n{'='*80}")
    print(f"Processing GNN Architecture: {gnn_arch_rename.get(gnn_arch, gnn_arch)}")
    print(f"{'='*80}")
    print(f"DEBUG: Starting analysis for {gnn_arch}")
    
    # Create a matrix where rows are methods and columns are datasets
    # Only for this specific GNN architecture
    method_names = []
    performance_matrix = []
    
    # Collect methods that have results for this GNN architecture
    available_methods = set()
    print(f"DEBUG: Checking datasets: {datasets}")
    for dataset in datasets:
        print(f"DEBUG: Checking dataset {dataset}")
        print(f"DEBUG: dataset in results: {dataset in results}")
        if (dataset in results and gnn_arch in results[dataset]):
            print(f"DEBUG: {gnn_arch} found in {dataset}")
            print(f"DEBUG: Available methods in {dataset}[{gnn_arch}]: {list(results[dataset][gnn_arch].keys())}")
            for method in methods:
                if method in results[dataset][gnn_arch]:
                    available_methods.add(method)
                    print(f"DEBUG: Added method {method} for {gnn_arch} in {dataset}")
        else:
            print(f"DEBUG: {gnn_arch} NOT found in {dataset}")
    
    available_methods = sorted(list(available_methods))
    print(f"DEBUG: Available methods for {gnn_arch}: {available_methods}")
    
    # Remove ORIGINAL method from analysis (always performs best, skews comparisons)
    # filtered_methods = [method for method in available_methods if method != "ORIGINAL"]
    filtered_methods = [method for method in available_methods]
    
    print(f"DEBUG: Filtered methods: {filtered_methods}")
    print(f"DEBUG: Number of filtered methods: {len(filtered_methods)}")
    
    if len(filtered_methods) < 2:
        print(f"Insufficient methods for analysis ({len(filtered_methods)} < 2)")
        continue
    
    print(f"Found {len(filtered_methods)} methods for {gnn_arch}: {filtered_methods}")
    
    # Create performance matrix for this GNN architecture
    print(f"DEBUG: Creating performance matrix for {gnn_arch}")
    for method in filtered_methods:
        method_display = method_rename.get(method, method)
        method_names.append(method_display)
        performance_row = []
        
        print(f"DEBUG: Processing method {method} -> {method_display}")
        for dataset in datasets:
            if (dataset in results and 
                gnn_arch in results[dataset] and 
                method in results[dataset][gnn_arch]):
                
                mean_value, _ = results[dataset][gnn_arch][method]
                performance_row.append(mean_value)
                print(f"DEBUG: {method} in {dataset}: {mean_value}")
            else:
                performance_row.append(np.nan)
                print(f"DEBUG: {method} in {dataset}: NaN (missing)")
        
        performance_matrix.append(performance_row)
        print(f"DEBUG: Performance row for {method}: {performance_row}")
    
    performance_matrix = np.array(performance_matrix)
    print(f"DEBUG: Performance matrix shape: {performance_matrix.shape}")
    print(f"DEBUG: Performance matrix:\n{performance_matrix}")
    
    # Calculate ranks for each dataset (handling missing data)
    rank_matrix = np.full_like(performance_matrix, np.nan)
    print(f"DEBUG: Starting ranking calculation for {gnn_arch}")
    for j, dataset in enumerate(datasets):
        # Get non-missing values for this dataset
        dataset_scores = performance_matrix[:, j]
        valid_mask = ~np.isnan(dataset_scores)
        
        print(f"DEBUG: Dataset {dataset} (column {j}):")
        print(f"DEBUG: Raw scores: {dataset_scores}")
        print(f"DEBUG: Valid mask: {valid_mask}")
        print(f"DEBUG: Valid scores: {dataset_scores[valid_mask]}")
        print(f"DEBUG: Number of valid methods: {np.sum(valid_mask)}")
        
        if np.sum(valid_mask) > 1:  # Need at least 2 methods for ranking
            metric_type = dataset_metrics.get(dataset, "mae")
            print(f"DEBUG: Metric type for {dataset}: {metric_type}")
            
            if metric_type == "roc_auc":
                # Higher is better
                ranks = rankdata(-dataset_scores[valid_mask], method='average')
                print(f"DEBUG: ROC-AUC ranking (higher=better): {ranks}")
            else:
                # Lower is better  
                ranks = rankdata(dataset_scores[valid_mask], method='average')
                print(f"DEBUG: MAE ranking (lower=better): {ranks}")
            
            # Assign ranks back to the full matrix
            rank_matrix[valid_mask, j] = ranks
            print(f"DEBUG: Assigned ranks to column {j}: {rank_matrix[:, j]}")
        else:
            print(f"Warning: Dataset {dataset} has insufficient valid methods for ranking")
            print(f"DEBUG: Only {np.sum(valid_mask)} valid methods found")
    
    # Calculate average ranks (ignoring NaN values for methods with missing datasets)
    avg_ranks = np.nanmean(rank_matrix, axis=1)
    
    # Count how many datasets each method was evaluated on
    dataset_counts = np.sum(~np.isnan(rank_matrix), axis=1)
    
    print(f"DEBUG: Final rank matrix for {gnn_arch}:")
    print(f"DEBUG: Rank matrix shape: {rank_matrix.shape}")
    print(f"DEBUG: Rank matrix:\n{rank_matrix}")
    print(f"DEBUG: Average ranks: {avg_ranks}")
    print(f"DEBUG: Dataset counts: {dataset_counts}")
    print(f"DEBUG: Method names: {method_names}")
    
    # Filter methods that have complete data across ALL datasets for statistical testing
    complete_data_mask = dataset_counts == len(datasets)
    print(f"Methods with complete data across all {len(datasets)} datasets: {np.sum(complete_data_mask)}")
    print(f"DEBUG: Complete data mask: {complete_data_mask}")
    
    if np.sum(complete_data_mask) >= 2:
        # Create filtered versions for statistical testing
        complete_methods = [method_names[i] for i in range(len(method_names)) if complete_data_mask[i]]
        complete_rank_matrix = rank_matrix[complete_data_mask]
        complete_avg_ranks = avg_ranks[complete_data_mask]
        
        print(f"Using {len(complete_methods)} methods with complete data for Nemenyi test:")
        for i, method_name in enumerate(complete_methods):
            print(f"  {method_name}: avg rank {complete_avg_ranks[i]:.2f}")
            
        print(f"DEBUG: Complete methods: {complete_methods}")
        print(f"DEBUG: Complete rank matrix shape: {complete_rank_matrix.shape}")
        print(f"DEBUG: Complete rank matrix:\n{complete_rank_matrix}")
    else:
        print(f"Warning: Only {np.sum(complete_data_mask)} methods have complete data. Need at least 2.")
        print("Proceeding with all available data but statistical tests may be unreliable.")
        complete_methods = method_names
        complete_rank_matrix = rank_matrix
        complete_avg_ranks = avg_ranks
    
    # Save ranking data for this GNN architecture
    gnn_ranking = pd.DataFrame({
        'Method': method_names,
        'Average_Rank': avg_ranks,
        'Datasets_Count': dataset_counts
    })
    
    # Add individual dataset ranks
    for j, dataset in enumerate(datasets):
        gnn_ranking[f'Rank_{dataset_rename.get(dataset, dataset)}'] = rank_matrix[:, j]
    
    gnn_ranking = gnn_ranking.sort_values('Average_Rank')
    gnn_arch_display = gnn_arch_rename.get(gnn_arch, gnn_arch)
    ranking_path = os.path.join(output_dir, f"rankings_{gnn_arch_display}.csv")
    gnn_ranking.to_csv(ranking_path, index=False)
    print(f"Saved rankings: {ranking_path}")
    
    # Run Nemenyi analysis if we have enough methods
    if len(complete_methods) >= 2:
        print(f"\n{'='*60}")
        print(f"Running NEMENYI analysis for {gnn_arch_display}...")
        print(f"{'='*60}")
        
        # Proper Friedman + Nemenyi post-hoc test with Critical Difference calculation
        significant_pairs, cd = nemenyi_test_with_friedman(complete_rank_matrix, ALPHA)
        title = f"Critical Difference Diagram - {gnn_arch_display}\n(Friedman + Nemenyi CD, α={ALPHA})"
        
        output_path = os.path.join(output_dir, f"cd_diagram_{gnn_arch_display}.png")
        
        # For method comparison, only ranks are meaningful (scores have different scales)
        draw_cd_diagram(complete_avg_ranks, complete_methods, cd, title, output_path, 
                       scores=None, significant_pairs=significant_pairs)
        
        print(f"Saved diagram: {output_path}")
        print(f"Method ranking for {gnn_arch_display}:")
        for i, (_, row) in enumerate(gnn_ranking.iterrows()):
            print(f"  {i+1}. {row['Method']}: avg rank {row['Average_Rank']:.2f}")
    else:
        print(f"Insufficient methods for statistical analysis ({len(complete_methods)} < 2)")

print(f"\nAll diagrams saved to: {output_dir}")
print("=" * 50)
