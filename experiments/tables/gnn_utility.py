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
# datasets = list(data.keys())
datasets = [
    "rossmann_subsampled",
    "walmart_subsampled",
    "airbnb-simplified_subsampled",
    "Berka_subsampled",
    "f1_subsampled",
]
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
                raise ValueError(
                    f"No valid metric found for method {method} in dataset {dataset}"
                )
        mean_value, se_value = compute_mean_and_se(metric_values)
        results[dataset][method] = (mean_value, se_value)

# Set the desired order of methods
method_order = [
    # "BASELINE",
    "ORIGINAL",
    "MOSTLYAI",
    "RGCLD",
    "CLAVADDPM",
    "RCTGAN",
    "REALTABFORMER",
    "SDV",
]

method_rename = {
    "BASELINE": "BASELINE",
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

baseline_scores = {
    "f1_subsampled": "0.5",
    "Berka_subsampled": "0.5",
    "rossmann_subsampled": "324",
    "walmart_subsampled": "14.7k",
    "airbnb-simplified_subsampled": "0.5",
}

score_types = {
    "f1_subsampled": "AUC",
    "Berka_subsampled": "AUC",
    "rossmann_subsampled": "MAE",
    "walmart_subsampled": "MAE",
    "airbnb-simplified_subsampled": "AUC",
}

score_trypes_with_arrow = {
    "f1_subsampled": "AUC ($\\uparrow$)",
    "Berka_subsampled": "AUC ($\\uparrow$)",
    "rossmann_subsampled": "MAE ($\\downarrow$)",
    "walmart_subsampled": "MAE ($\\downarrow$)",
    "airbnb-simplified_subsampled": "AUC ($\\uparrow$)",
}

methods = [method_rename[method] for method in method_order if method in methods]

# Generate LaTeX table
latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{c" + "c" * (len(methods) + 1) + "}\n"
)
latex_table += "\\toprule\n"
latex_table += "Dataset & & " + " & ".join(methods) + " \\\\\n"
latex_table += "\\midrule\n"

for dataset in datasets:
    # Use renamed dataset if available, otherwise use original name
    dataset_name = dataset_rename.get(dataset, dataset)
    row = [dataset_name, score_trypes_with_arrow[dataset]]

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
            reverse=True,
        )
    else:
        # For MAE, lower is better
        sorted_scores = sorted(
            (s for s in scores if s[0] != float("inf")), key=lambda x: x[0]
        )

    best_method_tuple = sorted_scores[0] if sorted_scores else None
    underlined_methods = []
    # This variable will store the name of the best method for bolding.
    best_method_name_for_bolding = None

    if best_method_tuple:
        best_mean, best_se, best_method_name_for_bolding = best_method_tuple
        # Calculate the margin: user specified to multiply the best method's SE by sqrt(3).
        # np.sqrt is used as numpy is imported as np.
        margin = best_se * np.sqrt(3)

        # Iterate through all scores (which are already filtered not to be ORIGINAL/BASELINE,
        # and are sorted) to find methods to underline.
        for current_mean, _current_se, current_method_name in sorted_scores:
            # _current_se is used as the SE of the current method is not needed for this comparison.
            if current_method_name == best_method_name_for_bolding:
                continue  # The best method itself is bolded, not underlined.

            # Check if the current method's mean score falls into the interval:
            # [best_mean - margin, best_mean + margin].
            if best_mean - margin <= current_mean <= best_mean + margin:
                underlined_methods.append(current_method_name)
    # If best_method_tuple was None (e.g., no valid scores),
    # best_method_name_for_bolding remains None, and underlined_methods remains empty.

    # Generate row entries
    for method in methods:
        original_method = next(k for k, v in method_rename.items() if v == method)
        if original_method in results[dataset]:
            mean, se = results[dataset][original_method]

            mean_val_str = f"{mean:.2f}" if mean < 1 else f"{mean:.0f}"

            # Prepare the \pm SE part, without any $ or \tiny yet
            pm_se_str_core = ""
            if not np.isclose(se, 0, atol=1e-10):
                se_val_for_format = f"{se:.2f}" if se < 1 else f"{se:.0f}"
                pm_se_str_core = f"\\pm {se_val_for_format}"

            # Now, build the cell string based on highlighting rules
            if original_method not in ["ORIGINAL", "BASELINE"]:
                if method == best_method_name_for_bolding:
                    # $\mathbf{MEAN}${\tiny$\pm SE$}
                    formatted_score = f"$\\mathbf{{{mean_val_str}}}$"
                    if pm_se_str_core:
                        formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                    row.append(formatted_score)
                elif method in underlined_methods:
                    # $\underline{MEAN}${\tiny$\pm SE$}
                    formatted_score = f"$\\underline{{{mean_val_str}}}$"
                    if pm_se_str_core:
                        formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                    row.append(formatted_score)
                else:
                    # $MEAN${\tiny$\pm SE$}
                    formatted_score = f"${mean_val_str}$"
                    if pm_se_str_core:
                        formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                    row.append(formatted_score)
            elif original_method == "ORIGINAL":
                # $MEAN${\tiny$\pm SE$} (BASELINE)
                base_score_part = f"${mean_val_str}$"
                if pm_se_str_core:
                    base_score_part += f"{{\\tiny${pm_se_str_core}$}}"
                row.append(f"{base_score_part} $({baseline_scores[dataset]})$")
            else: # This is for 'BASELINE' method if it's in method_order and not filtered out
                # $MEAN${\tiny$\pm SE$}
                formatted_score = f"${mean_val_str}$"
                if pm_se_str_core:
                    formatted_score += f"{{\\tiny${pm_se_str_core}$}}"
                row.append(formatted_score)
        else:
            row.append("-")  # Placeholder for missing data
    latex_table += " & ".join(row) + " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Mean RMSE ± SE for each dataset and method.}\n\\label{tab:results}\n\\end{table}"

# Output the LaTeX table
print(latex_table)
