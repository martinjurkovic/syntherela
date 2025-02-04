import os
import json
import numpy as np

from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Define which metric to use for each dataset
dataset_metrics = {
    "f1": "mae",
    "Berka_subsampled": "mae",
    "rossmann_subsampled": "mae",
    "walmart_subsampled": "mae",
    "airbnb-simplified_subsampled": "roc_auc"
}

results_dir = os.path.join(PROJECT_PATH, "results")
results_file = os.path.join(results_dir, "gnn_utility_results.json")

with open(results_file, "r") as f:
    data = json.load(f)


# Compute mean and standard error
def compute_mean_and_se(values):
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, se


# Extract datasets, methods, and calculate metrics
datasets = list(data.keys())
methods = set()
results = {}

for dataset, method_data in data.items():
    results[dataset] = {}
    for method, runs in method_data.items():
        methods.add(method)
        # Try to get MAE values, if not available use AUC
        metric_values = []
        for run in runs.values():
            if "mae" in run:
                metric_values.append(run["mae"])
            elif "roc_auc" in run:
                metric_values.append(run["roc_auc"])
            else:
                raise ValueError(f"No valid metric found for method {method} in dataset {dataset}")
        mean_value, se_value = compute_mean_and_se(metric_values)
        results[dataset][method] = (mean_value, se_value)

# Set the desired order of methods
method_order = [
    # "BASELINE",
    "ORIGINAL",
    "SDV",
    "RCTGAN",
    "REALTABFORMER",
    "CLAVADDPM",
    "MOSTLYAI",
    "RGCLD",
]

method_rename = {
    "BASELINE": "BASELINE",
    "ORIGINAL": "ORIG.",
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTABF.",
    "CLAVADDPM": "CLAVADDPM",
    "MOSTLYAI": "MOSTLYAI",
    "RGCLD": "RGCLD",
}

dataset_rename = {
    "f1": "F1",
    "Berka_subsampled": "Berka",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
}

methods = [method_rename[method] for method in method_order if method in methods]

# Generate LaTeX table
latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * len(methods) + "}\n"
)
latex_table += "\\toprule\n"
latex_table += "Dataset & " + " & ".join(methods) + " \\\\\n"
latex_table += "\\midrule\n"

for dataset in datasets:
    # Use renamed dataset if available, otherwise use original name
    dataset_name = dataset_rename.get(dataset, dataset)
    row = [dataset_name]

    # Collect all scores for this dataset to determine best and second best
    scores = []
    for method in methods:
        original_method = next(k for k, v in method_rename.items() if v == method)
        # Skip ORIGINAL and BASELINE in the comparison
        if original_method not in ["ORIGINAL", "BASELINE"]:
            if original_method in results[dataset]:
                mean, se = results[dataset][original_method]
                scores.append((mean, se, method))
            else:
                scores.append((float("inf"), 0, method))

    # Sort scores to find best and second best
    metric_type = dataset_metrics.get(dataset, "mae")  # Default to mae if not specified
    if metric_type == "roc_auc":
        # For ROC AUC, higher is better
        sorted_scores = sorted(
            (s for s in scores if s[0] != float("inf")), 
            key=lambda x: x[0], 
            reverse=True
        )
    else:
        # For MAE, lower is better
        sorted_scores = sorted(
            (s for s in scores if s[0] != float("inf")), 
            key=lambda x: x[0]
        )
    best_method = sorted_scores[0][2] if sorted_scores else None
    second_best_method = sorted_scores[1][2] if len(sorted_scores) > 1 else None

    # Generate row entries
    for method in methods:
        original_method = next(k for k, v in method_rename.items() if v == method)
        if original_method in results[dataset]:
            mean, se = results[dataset][original_method]
            score_str = f"{mean:.2f}" if mean < 1 else f"{mean:.0f}"
            if se > 0:
                score_str += f" \\pm {se:.2f}" if se < 1 else f" \\pm {se:.0f}"

            # Only highlight if method is not ORIGINAL or BASELINE
            if original_method not in ["ORIGINAL", "BASELINE"]:
                if method == best_method:
                    row.append(f"$\\mathbf{{{score_str}}}$")
                elif method == second_best_method:
                    row.append(f"$\\underline{{{score_str}}}$")
                else:
                    row.append(f"${score_str}$")
            else:
                row.append(f"${score_str}$")
        else:
            row.append("-")  # Placeholder for missing data
    latex_table += " & ".join(row) + " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Mean RMSE ± SE for each dataset and method.}\n\\label{tab:results}\n\\end{table}"

# Output the LaTeX table
print(latex_table)
