import os
import json
import numpy as np

from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

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
        rmse_values = [run["rmse"] for run in runs.values()]
        mean_rmse, se_rmse = compute_mean_and_se(rmse_values)
        results[dataset][method] = (mean_rmse, se_rmse)

# Set the desired order of methods
method_order = [
    "BASELINE",
    "ORIGINAL",
    "SDV",
    "RCTGAN",
    "REALTABFORMER",
    "CLAVADDPM",
    "GRETEL_ACTGAN",
    "GRETEL_LSTM",
    "MOSTLYAI",
]
methods = [method for method in method_order if method in methods]

# Generate LaTeX table
latex_table = (
    "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * len(methods) + "}\n"
)
latex_table += "\\toprule\n"
latex_table += "Dataset & " + " & ".join(methods) + " \\\\\n"
latex_table += "\\midrule\n"

for dataset in datasets:
    row = [dataset]
    for method in methods:
        if method in results[dataset]:
            mean, se = results[dataset][method]
            # Only print SE if it is greater than 0
            if se > 0:
                row.append(f"${mean:.2f} \\pm {se:.2f}$")
            else:
                row.append(f"${mean:.2f}$")
        else:
            row.append("-")  # Placeholder for missing data
    latex_table += " & ".join(row) + " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Mean RMSE ± SE for each dataset and method.}\n\\label{tab:results}\n\\end{table}"

# Output the LaTeX table
print(latex_table)