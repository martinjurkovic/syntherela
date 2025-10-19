import os
import json
import numpy as np
import glob

from dotenv import load_dotenv

# Note: The generated LaTeX table uses cell coloring for highlighting
# Make sure to include \usepackage[table]{xcolor} in your LaTeX document preamble

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Configuration for reading results
USE_HYPERPARAMETER_TUNING_RESULTS = True  # Set to True to read from hyperparameter_tuning_100 directory

# Configuration for table highlighting
HIGHLIGHT_COLOR = "green!10"  # Change this to adjust the highlight color (e.g., "blue!15", "yellow!8", etc.)

# Define which metric to use for each dataset
dataset_metrics = {
    "rossmann_subsampled": "mae",
    "walmart_subsampled": "mae",
    "airbnb-simplified_subsampled": "roc_auc",
    "Berka_subsampled": "roc_auc",
    "f1_subsampled": "roc_auc",
}

# Read from existing merged results file (contains all synthetic data methods)
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

# Compute mean and standard error
def compute_mean_and_se(values):
    mean = np.mean(values)
    if len(values) == 1:
        # For single values (like hyperparameter tuning results), no standard error
        se = 0.0
    else:
        se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, se


# Extract datasets, methods, gnn architectures, and calculate metrics
datasets = [
    "rossmann_subsampled",
    "walmart_subsampled",
    "airbnb-simplified_subsampled",
    "Berka_subsampled",
    "f1_subsampled",
]

# Use only hetero-graphsage architecture
gnn_architectures = ["hetero-graphsage"]
methods = set()
for dataset, method_data in data.items():
    for method, gnn_data in method_data.items():
        methods.add(method)

# Process results: dataset -> method -> best_across_gnn_archs -> (mean, se)
results = {}

for dataset in datasets:
    if dataset not in data:
        continue
        
    results[dataset] = {}
    
    for method in methods:
        if method not in data[dataset]:
            continue
            
        # Use only hetero-graphsage architecture
        gnn_arch = "hetero-graphsage"
        if gnn_arch not in data[dataset][method]:
            continue
            
        runs = data[dataset][method][gnn_arch]
        metric_values = []
        
        for run in runs.values():
            if isinstance(run, dict) and run:  # Skip empty runs
                # Get the appropriate metric for this dataset
                metric_name = dataset_metrics.get(dataset, "mae")
                if metric_name in run:
                    metric_values.append(run[metric_name])
        
        if metric_values:
            mean_value, se_value = compute_mean_and_se(metric_values)
            results[dataset][method] = (mean_value, se_value)

# Set the desired order of methods
method_order = [
    "ORIGINAL",
    "MOSTLYAI", 
    "CLAVADDPM",
    "RCTGAN",
    "REALTABFORMER",
    "SDV",
    # "RELDIFF",
]

# Note: Removing MOSTLYAI/TARGN from the table as it appears to be missing from the image
method_rename = {
    "ORIGINAL": "ORIG.",
    # "RELDIFF": "RelDiff",
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTABF.",
    "CLAVADDPM": "CLAVADDPM",
    "MOSTLYAI": "TabularARGN",
}

dataset_rename = {
    "f1_subsampled": "F1",
    "Berka_subsampled": "Berka",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
}

baseline_scores = {
    "f1_subsampled": "0.5",
    "Berka_subsampled": "0.5",
    "rossmann_subsampled": "324",
    "walmart_subsampled": "14.7k",
    "airbnb-simplified_subsampled": "0.5",
}

score_types_with_arrow = {
    "f1_subsampled": "AUC ($\\uparrow$)",
    "Berka_subsampled": "AUC ($\\uparrow$)",
    "rossmann_subsampled": "MAE ($\\downarrow$)",
    "walmart_subsampled": "MAE ($\\downarrow$)",
    "airbnb-simplified_subsampled": "AUC ($\\uparrow$)",
}

# Filter methods to only include those that exist in our data and are in the desired order
available_methods = [method for method in method_order if method in methods]
available_method_names = [method_rename[method] for method in available_methods]

# Generate LaTeX table
num_columns = len(available_method_names) + 3  # Dataset + Metric + Methods + Improvement
latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lc" + "c" * len(available_method_names) + "c}\n"
)
latex_table += "\\toprule\n"
latex_table += "\\textbf{Dataset} & \\textbf{Metric} & " + " & ".join([f"\\textbf{{{name}}}" for name in available_method_names]) + " & \\textbf{Improv.} \\\\\n"
latex_table += "\\midrule\n"

for dataset in datasets:
    if dataset not in results:
        continue
        
    dataset_name = dataset_rename.get(dataset, dataset)
    score_type = score_types_with_arrow[dataset].replace("($\\uparrow$)", "($\\uparrow$)").replace("($\\downarrow$)", "($\\downarrow$)")
    
    # Collect scores for this dataset (excluding ORIGINAL)
    scores = []
    for method in available_methods:
        if method == "ORIGINAL":
            continue
        if method in results[dataset]:
            score, se = results[dataset][method]
            scores.append((score, se, method))
    
    # Sort to find best and second best
    metric_type = dataset_metrics.get(dataset, "mae")
    if metric_type == "roc_auc":
        # For ROC AUC, higher is better
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    else:
        # For MAE, lower is better
        sorted_scores = sorted(scores, key=lambda x: x[0])
    
    best_score = None
    second_best_score = None
    improvement = 0.0
    best_methods = []  # List to store all methods that tie for best
    
    if len(sorted_scores) >= 1:
        best_score = sorted_scores[0][0]
        # Find all methods that tie for best based on displayed rounded values
        best_displayed_value = round(best_score) if best_score >= 1 else round(best_score, 2)
        for score, se, method in sorted_scores:
            displayed_value = round(score) if score >= 1 else round(score, 2)
            if displayed_value == best_displayed_value:
                best_methods.append(method)
        
        # Find second best score (first score that's not equal to best)
        for score, se, method in sorted_scores:
            if score != best_score:
                second_best_score = score
                break
        
        # Calculate improvement as percentage using rounded display values: (RelDiff - other_best) / other_best * 100
        reldiff_score = None
        other_best_score = None
        
        # Find RelDiff score (rounded as displayed)
        for score, se, method in scores:
            if method == "RELDIFF":
                reldiff_score = round(score) if score >= 1 else round(score, 2)
                break
        
        # Find the best score among non-RelDiff methods (rounded as displayed)
        non_reldiff_scores = []
        for score, se, method in scores:
            if method != "RELDIFF":
                rounded_score = round(score) if score >= 1 else round(score, 2)
                non_reldiff_scores.append((rounded_score, se, method))
        
        if non_reldiff_scores:
            if metric_type == "roc_auc":
                # For ROC AUC, higher is better
                other_best_score = max(non_reldiff_scores, key=lambda x: x[0])[0]
            else:
                # For MAE, lower is better
                other_best_score = min(non_reldiff_scores, key=lambda x: x[0])[0]
        
        # Calculate improvement using rounded values
        if reldiff_score is not None and other_best_score is not None and other_best_score > 0:
            if metric_type == "roc_auc":
                # For AUC: (RelDiff - other_best) / other_best * 100
                improvement = ((reldiff_score - other_best_score) / other_best_score) * 100
            else:
                # For MAE: (other_best - RelDiff) / other_best * 100 (since lower is better)
                improvement = ((other_best_score - reldiff_score) / other_best_score) * 100
        else:
            improvement = 0.0
    
    # Build table row
    row = [dataset_name, score_type]
    
    # Add method columns
    for method in available_methods:
        if method in results[dataset]:
            score, se = results[dataset][method]
            
            if method == "ORIGINAL":
                # ORIGINAL method with baseline score in parentheses
                if score >= 1:
                    score_str = f"{score:.0f}"
                else:
                    score_str = f"{score:.2f}"
                baseline = baseline_scores[dataset]
                row.append(f"{score_str} ({baseline})")
            else:
                # Format score
                if score >= 1:
                    score_str = f"{score:.0f}"
                else:
                    score_str = f"{score:.2f}"
                
                # Add standard error - use 0.01 if se is 0 or very small
                display_se = 0.01 if se < 0.005 else se
                if display_se >= 1:
                    se_str = f"\\pm{display_se:.0f}"
                else:
                    se_str = f"\\pm{display_se:.2f}"
                formatted_score = f"${score_str}${{\\tiny${se_str}$}}"
                
                # Check if this method is among the best (tied for first)
                try:
                    original_method = next(k for k, v in method_rename.items() if v == method)
                except StopIteration:
                    original_method = method  # fallback if not found in rename map
                is_best = original_method in best_methods
                
                if is_best:
                    # Bold all methods that tie for first place (regardless of how many)
                    row.append(f"$\\mathbf{{{score_str}}}${{\\tiny${se_str}$}}")
                else:
                    row.append(formatted_score)
        else:
            row.append("-")
    
    # Add improvement column with cyan color
    if improvement > 0:
        row.append(f"\\textcolor{{cyan}}{{{improvement:.1f}}}")
    else:
        row.append("\\textcolor{cyan}{0.0}")
    
    latex_table += " & ".join(row) + " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Comparison of synthetic data methods across datasets. Improvement shows percentage improvement of the best method over the second-best method.}\n\\label{tab:synthetic_data_comparison}\n\\end{table}"

# Output the LaTeX table
print(latex_table)
