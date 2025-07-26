import os
import json
import numpy as np

from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Define which metric to use for each dataset
dataset_metrics = {
    "rossmann_subsampled": "mae",
    "walmart_subsampled": "mae",
    "airbnb-simplified_subsampled": "roc_auc",
    "Berka_subsampled": "roc_auc",
    "f1_subsampled": "roc_auc",
}

results_dir = os.path.join(PROJECT_PATH, "results", "rdl_utility")
results_file = os.path.join(results_dir, "gnn_utility_results_merged.json")

with open(results_file, "r") as f:
    data = json.load(f)


# Compute mean and standard error
def compute_mean_and_se(values):
    mean = np.mean(values)
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
    "MOSTLYAI",
    "RGCLD",
    "CLAVADDPM",
    "RCTGAN",
    "REALTABFORMER",
    "SDV",
]

method_rename = {
    "ORIGINAL": "ORIG.",
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTABF.",
    "CLAVADDPM": "CLAVADDPM",
    "MOSTLYAI": "TabularARGN",
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
    "hetero-graphsage": "GraphSAGE",
    "hetero-gin": "GIN",
    "hetero-graphconv": "GraphConv",
    "hetero-gat": "GAT",
    "hetero-gatv2": "GATv2",
    "relgnn": "RelGNN",
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

# Filter methods to only include those that exist in our data
available_methods = [method_rename[method] for method in method_order if method in methods]

# Generate LaTeX table
num_columns = len(available_methods) + 3  # Dataset + GNN Arch + Score Type + Methods
latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{c" + "c" * num_columns + "}\n"
)
latex_table += "\\toprule\n"
latex_table += "Dataset & GNN Architecture & & " + " & ".join(available_methods) + " \\\\\n"
latex_table += "\\midrule\n"

for dataset_idx, dataset in enumerate(datasets):
    dataset_name = dataset_rename.get(dataset, dataset)
    score_type = score_types_with_arrow[dataset]
    
    # Filter GNN architectures to only include those with data for this dataset
    available_gnn_archs = []
    for gnn_arch in gnn_architectures:
        if (dataset in results and 
            gnn_arch in results[dataset] and 
            any(method in results[dataset][gnn_arch] for method in method_order if method in methods)):
            available_gnn_archs.append(gnn_arch)
    
    num_gnn_archs = len(available_gnn_archs)
    
    for gnn_idx, gnn_arch in enumerate(available_gnn_archs):
        # Collect all scores for this dataset and GNN architecture to determine best and second best
        scores = []
        for method in available_methods:
            original_method = next(k for k, v in method_rename.items() if v == method)
            # Skip ORIGINAL in the comparison for best/worst determination
            if original_method not in ["ORIGINAL"]:
                if (dataset in results and 
                    gnn_arch in results[dataset] and 
                    original_method in results[dataset][gnn_arch]):
                    mean, se = results[dataset][gnn_arch][original_method]
                    scores.append((mean, se, method))
                else:
                    scores.append((float("inf"), 0, method))

        # Sort scores to find best and second best
        metric_type = dataset_metrics.get(dataset, "mae")
        if metric_type == "roc_auc":
            # For ROC AUC, higher is better
            sorted_scores = sorted(
                (s for s in scores if s[0] != float("inf")),
                key=lambda x: x[0],
                reverse=True,
            )
        else:
            # For MAE, lower is better
            sorted_scores = sorted(
                (s for s in scores if s[0] != float("inf")), key=lambda x: x[0]
            )

        best_method_tuple = sorted_scores[0] if sorted_scores else None
        underlined_methods = []
        best_method_name_for_bolding = None

        if best_method_tuple:
            best_mean, best_se, best_method_name_for_bolding = best_method_tuple
            # Calculate the margin: multiply the best method's SE by sqrt(3)
            margin = best_se * np.sqrt(3)

            # Find methods to underline (within margin of best)
            for current_mean, _current_se, current_method_name in sorted_scores:
                if current_method_name == best_method_name_for_bolding:
                    continue  # The best method itself is bolded, not underlined

                if best_mean - margin <= current_mean <= best_mean + margin:
                    underlined_methods.append(current_method_name)

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

        # Method columns
        for method in available_methods:
            original_method = next(k for k, v in method_rename.items() if v == method)
            if (dataset in results and 
                gnn_arch in results[dataset] and 
                original_method in results[dataset][gnn_arch]):
                
                mean, se = results[dataset][gnn_arch][original_method]

                mean_val_str = f"{mean:.2f}" if mean < 1 else f"{mean:.0f}"

                # Prepare the ±SE part
                pm_se_str_core = ""
                if not np.isclose(se, 0, atol=1e-10):
                    se_val_for_format = f"{se:.2f}" if se < 1 else f"{se:.0f}"
                    pm_se_str_core = f"\\pm {se_val_for_format}"

                # Format based on highlighting rules
                if original_method not in ["ORIGINAL"]:
                    if method == best_method_name_for_bolding:
                        # Bold for best method
                        formatted_score = f"$\\mathbf{{{mean_val_str}}}$"
                        if pm_se_str_core:
                            formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                        row.append(formatted_score)
                    elif method in underlined_methods:
                        # Underline for methods within margin
                        formatted_score = f"$\\underline{{{mean_val_str}}}$"
                        if pm_se_str_core:
                            formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                        row.append(formatted_score)
                    else:
                        # Regular formatting
                        formatted_score = f"${mean_val_str}$"
                        if pm_se_str_core:
                            formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                        row.append(formatted_score)
                elif original_method == "ORIGINAL":
                    # ORIGINAL method with baseline score
                    base_score_part = f"${mean_val_str}$"
                    if pm_se_str_core:
                        base_score_part += f"{{\\tiny${pm_se_str_core}$}}"
                    row.append(f"{base_score_part} $({baseline_scores[dataset]})$")
                else:
                    # Other methods (shouldn't reach here with current logic)
                    formatted_score = f"${mean_val_str}$"
                    if pm_se_str_core:
                        formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                    row.append(formatted_score)
            else:
                row.append("-")  # Placeholder for missing data

        latex_table += " & ".join(row) + " \\\\\n"
    
    # Add midrule between datasets (except after last dataset)
    if dataset_idx < len(datasets) - 1:
        latex_table += "\\midrule\n"

latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{GNN Architecture Comparison: Mean metrics ± SE for each dataset, GNN architecture, and synthetic data method.}\n\\label{tab:gnn_results}\n\\end{table}"

# Output the LaTeX table
print(latex_table) 