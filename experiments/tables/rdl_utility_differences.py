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

# Get all GNN architectures from the data
gnn_architectures = set()
methods = set()
for dataset, method_data in data.items():
    for method, gnn_data in method_data.items():
        methods.add(method)
        for gnn_arch in gnn_data.keys():
            gnn_architectures.add(gnn_arch)

# Sort GNN architectures for consistent ordering
gnn_architectures = sorted(list(gnn_architectures))

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
                    if isinstance(run, dict) and run:  # Skip empty runs
                        # Get the appropriate metric for this dataset
                        metric_name = dataset_metrics.get(dataset, "mae")
                        if metric_name in run:
                            metric_values.append(run[metric_name])
                
                if metric_values:
                    mean_value, se_value = compute_mean_and_se(metric_values)
                    results[dataset][gnn_arch][method] = (mean_value, se_value)

# Set the desired order of methods
method_order = [
    "ORIGINAL",
    "RELDIFF",
    "MOSTLYAI",
    "RGCLD",
    "CLAVADDPM",
    "RCTGAN",
    "REALTABFORMER",
    "SDV",
]

method_rename = {
    "ORIGINAL": "ORIG.",
    "RELDIFF": "RELDIFF",
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTF.",
    "CLAVADDPM": "CLAVA",
    "MOSTLYAI": "TARGN",
    "RGCLD": "RGCLD",
}

dataset_rename = {
    "f1_subsampled": "F1",
    "Berka_subsampled": "Berka",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
}

gnn_arch_rename = {
    "hetero-graphsage": "G-SAGE",
    "hetero-gin": "GIN",
    "hetero-graphconv": "G-Conv",
    "hetero-gat": "GAT",
    "hetero-gatv2": "GATv2",
    "relgnn": "RelGNN",
}

score_types_with_arrow = {
    "f1_subsampled": "AUC ($\\uparrow$)",
    "Berka_subsampled": "AUC ($\\uparrow$)",
    "rossmann_subsampled": "MAE ($\\downarrow$)",
    "walmart_subsampled": "MAE ($\\downarrow$)",
    "airbnb-simplified_subsampled": "AUC ($\\uparrow$)",
}



# Calculate differences between synthetic methods and ORIGINAL
difference_results = {}

# Calculate differences at the run level
for dataset in datasets:
    difference_results[dataset] = {}
    
    for gnn_arch in gnn_architectures:
        difference_results[dataset][gnn_arch] = {}
        
        # Get ORIGINAL runs for this dataset and architecture
        if (dataset in data and 
            "ORIGINAL" in data[dataset] and 
            gnn_arch in data[dataset]["ORIGINAL"]):
            
            original_runs = data[dataset]["ORIGINAL"][gnn_arch]
            
            # Calculate differences for each synthetic method
            for method in methods:
                if method != "ORIGINAL":
                    if (dataset in data and 
                        method in data[dataset] and 
                        gnn_arch in data[dataset][method]):
                        
                        synthetic_runs = data[dataset][method][gnn_arch]
                        
                        # Calculate run-level differences
                        run_differences = []
                        
                        # Get all run combinations
                        for orig_run_key, orig_run_data in original_runs.items():
                            for synth_run_key, synth_run_data in synthetic_runs.items():
                                if (isinstance(orig_run_data, dict) and orig_run_data and
                                    isinstance(synth_run_data, dict) and synth_run_data):
                                    
                                    # Get the appropriate metric for this dataset
                                    metric_name = dataset_metrics.get(dataset, "mae")
                                    
                                    if metric_name in orig_run_data and metric_name in synth_run_data:
                                        orig_value = orig_run_data[metric_name]
                                        synth_value = synth_run_data[metric_name]
                                        
                                        # Calculate raw difference (synthetic - original)
                                        difference = synth_value - orig_value
                                        run_differences.append(difference)
                        
                        # Calculate mean and SE of run-level differences
                        if run_differences:
                            diff_mean, diff_se = compute_mean_and_se(run_differences)
                            difference_results[dataset][gnn_arch][method] = (diff_mean, diff_se)



# Generate difference table
synthetic_methods = [method for method in method_order if method != "ORIGINAL" and method in methods]
available_synthetic_methods = [method_rename[method] for method in synthetic_methods]

# Filter datasets and architectures to only include those with data
filtered_datasets = []
for dataset in datasets:
    if dataset in difference_results and any(
        len(difference_results[dataset].get(gnn_arch, {})) > 0 
        for gnn_arch in gnn_architectures
    ):
        filtered_datasets.append(dataset)

num_columns = len(available_synthetic_methods) + 3  # Dataset + GNN Arch + Score Type + Methods
difference_latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{c" + "c" * num_columns + "}\n"
)
difference_latex_table += "\\toprule\n"
difference_latex_table += "Dataset & GNN Architecture & & " + " & ".join(available_synthetic_methods) + " \\\\\n"
difference_latex_table += "\\midrule\n"

for dataset_idx, dataset in enumerate(filtered_datasets):
    dataset_name = dataset_rename.get(dataset, dataset)
    score_type = score_types_with_arrow[dataset]
    
    # Filter GNN architectures to only include those with data for this dataset
    available_gnn_archs = []
    for gnn_arch in gnn_architectures:
        if (dataset in difference_results and 
            gnn_arch in difference_results[dataset] and 
            len(difference_results[dataset][gnn_arch]) > 0):
            available_gnn_archs.append(gnn_arch)
    
    num_gnn_archs = len(available_gnn_archs)
    
    for gnn_idx, gnn_arch in enumerate(available_gnn_archs):
        # Build table row
        row = []
        
        # Dataset column (multirow for first entry)
        if gnn_idx == 0:
            row.append(f"\\multirow{{{num_gnn_archs}}}{{*}}{{{dataset_name}}}")
        else:
            row.append("")
        
        # GNN Architecture column
        gnn_arch_display = gnn_arch_rename.get(gnn_arch, gnn_arch)
        row.append(gnn_arch_display)
        
        # Score type column (multirow for first entry)
        if gnn_idx == 0:
            row.append(f"\\multirow{{{num_gnn_archs}}}{{*}}{{{score_type}}}")
        else:
            row.append("")
        
        # Method columns (differences)
        for method in synthetic_methods:
            if (dataset in difference_results and 
                gnn_arch in difference_results[dataset] and 
                method in difference_results[dataset][gnn_arch]):
                
                difference_mean, difference_se = difference_results[dataset][gnn_arch][method]
                
                # Format the difference mean (adjust precision based on metric type)
                metric_type = dataset_metrics.get(dataset, "mae")
                if metric_type == "mae":
                    # MAE differences might be larger, use appropriate precision
                    if abs(difference_mean) >= 10:
                        mean_str = f"{difference_mean:+.0f}"
                        se_str = f"{difference_se:.0f}" if difference_se >= 1 else f"{difference_se:.1f}"
                    else:
                        mean_str = f"{difference_mean:+.1f}"
                        se_str = f"{difference_se:.1f}"
                else:
                    # AUC differences are typically small
                    mean_str = f"{difference_mean:+.3f}"
                    se_str = f"{difference_se:.3f}"
                
                # Prepare the ±SE part
                pm_se_str_core = ""
                if not np.isclose(difference_se, 0, atol=1e-10):
                    pm_se_str_core = f"\\pm {se_str}"
                
                # Combine mean and SE
                if pm_se_str_core:
                    formatted_diff = f"${mean_str}${{\\tiny${pm_se_str_core}$}}"
                else:
                    formatted_diff = f"${mean_str}$"
                
                row.append(formatted_diff)
            else:
                row.append("-")  # Placeholder for missing data
        
        difference_latex_table += " & ".join(row) + " \\\\\n"
    
    # Add midrule between datasets (except after last dataset)
    if dataset_idx < len(filtered_datasets) - 1:
        difference_latex_table += "\\midrule\n"

difference_latex_table += "\\bottomrule\n\\end{tabular}\n"
difference_latex_table += "\\caption{Performance differences between synthetic data methods and original data. "
difference_latex_table += "For MAE datasets (↓), negative differences indicate better performance (lower error). "
difference_latex_table += "For AUC datasets (↑), positive differences indicate better performance (higher score). "
difference_latex_table += "Values shown as difference ± standard error.}\n"
difference_latex_table += "\\label{tab:gnn_differences}\n\\end{table}"

# Output the difference table
print(difference_latex_table)

print("\n" + "="*80 + "\n")
print("AVERAGED DIFFERENCES TABLE (Across GNN Architectures)")
print("="*80 + "\n")

# Calculate average differences across GNN architectures for each dataset and method
architecture_averaged_results = {}

for dataset in filtered_datasets:
    architecture_averaged_results[dataset] = {}
    
    for method in synthetic_methods:
        # Collect all differences for this method across all architectures
        method_differences = []
        
        for gnn_arch in gnn_architectures:
            if (dataset in difference_results and 
                gnn_arch in difference_results[dataset] and 
                method in difference_results[dataset][gnn_arch]):
                diff_mean, diff_se = difference_results[dataset][gnn_arch][method]
                method_differences.append(diff_mean)
        
        # Calculate mean and SE across architectures
        if method_differences:
            arch_mean, arch_se = compute_mean_and_se(method_differences)
            architecture_averaged_results[dataset][method] = (arch_mean, arch_se)

# Generate architecture-averaged table
arch_avg_latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{c" + "c" * len(available_synthetic_methods) + "}\n"
)
arch_avg_latex_table += "\\toprule\n"
arch_avg_latex_table += "Dataset & " + " & ".join(available_synthetic_methods) + " \\\\\n"
arch_avg_latex_table += "\\midrule\n"

for dataset_idx, dataset in enumerate(filtered_datasets):
    dataset_name = dataset_rename.get(dataset, dataset)
    score_type = score_types_with_arrow[dataset]
    metric_type = dataset_metrics.get(dataset, "mae")
    
    # Collect all scores for this dataset to determine best and within-margin methods
    scores = []
    for method in synthetic_methods:
        if (dataset in architecture_averaged_results and 
            method in architecture_averaged_results[dataset]):
            arch_mean, arch_se = architecture_averaged_results[dataset][method]
            scores.append((arch_mean, arch_se, method))
    
    # Sort scores to find best method
    if metric_type == "mae":
        # For MAE, lower (more negative) differences are better
        sorted_scores = sorted(scores, key=lambda x: x[0])
    else:
        # For AUC, higher (more positive) differences are better
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    
    best_method_tuple = sorted_scores[0] if sorted_scores else None
    underlined_methods = []
    best_method = None
    
    if best_method_tuple:
        best_mean, best_se, best_method = best_method_tuple
        # Calculate the margin: multiply the best method's SE by sqrt(3)
        margin = best_se * np.sqrt(6)
        
        # Find methods to underline (within margin of best)
        for current_mean, current_se, current_method in sorted_scores:
            if current_method == best_method:
                continue  # The best method itself is bolded, not underlined
            
            if abs(current_mean - best_mean) <= margin:
                underlined_methods.append(current_method)
    
    # Build table row
    row = [f"{dataset_name} ({score_type})"]
    
    # Method columns (averaged differences)
    for method in synthetic_methods:
        if (dataset in architecture_averaged_results and 
            method in architecture_averaged_results[dataset]):
            
            arch_mean, arch_se = architecture_averaged_results[dataset][method]
            
            # Format the difference mean (adjust precision based on metric type)
            if metric_type == "mae":
                # MAE differences might be larger, use appropriate precision
                if abs(arch_mean) >= 10:
                    mean_str = f"{arch_mean:+.0f}"
                    se_str = f"{arch_se:.0f}" if arch_se >= 1 else f"{arch_se:.1f}"
                else:
                    mean_str = f"{arch_mean:+.1f}"
                    se_str = f"{arch_se:.1f}"
            else:
                # AUC differences are typically small
                mean_str = f"{arch_mean:+.3f}"
                se_str = f"{arch_se:.3f}"
            
            # Prepare the ±SE part
            pm_se_str_core = ""
            if not np.isclose(arch_se, 0, atol=1e-10):
                pm_se_str_core = f"\\pm {se_str}"
            
            # Format with bolding, underlining, or regular formatting
            if method == best_method:
                # Bold the best method
                if pm_se_str_core:
                    formatted_diff = f"$\\mathbf{{{mean_str}}}${{\\tiny${pm_se_str_core}$}}"
                else:
                    formatted_diff = f"$\\mathbf{{{mean_str}}}$"
            elif method in underlined_methods:
                # Underline methods within margin of best
                if pm_se_str_core:
                    formatted_diff = f"$\\underline{{{mean_str}}}${{\\tiny${pm_se_str_core}$}}"
                else:
                    formatted_diff = f"$\\underline{{{mean_str}}}$"
            else:
                # Regular formatting
                if pm_se_str_core:
                    formatted_diff = f"${mean_str}${{\\tiny${pm_se_str_core}$}}"
                else:
                    formatted_diff = f"${mean_str}$"
            
            row.append(formatted_diff)
        else:
            row.append("-")  # Placeholder for missing data
    
    arch_avg_latex_table += " & ".join(row) + " \\\\\n"
    
    # Add midrule between datasets (except after last dataset)
    if dataset_idx < len(filtered_datasets) - 1:
        arch_avg_latex_table += "\\midrule\n"

arch_avg_latex_table += "\\bottomrule\n\\end{tabular}\n"
arch_avg_latex_table += "\\caption{Average performance differences between synthetic data methods and original data across all GNN architectures. "
arch_avg_latex_table += "For MAE datasets (↓), negative differences indicate better performance (lower error). "
arch_avg_latex_table += "For AUC datasets (↑), positive differences indicate better performance (higher score). "
arch_avg_latex_table += "Values shown as difference ± standard error across architectures. "
arch_avg_latex_table += "Best method per dataset is shown in bold, methods within margin (SE × √6) are underlined.}\n"
arch_avg_latex_table += "\\label{tab:gnn_differences_averaged}\n\\end{table}"

# Output the architecture-averaged table
print(arch_avg_latex_table)