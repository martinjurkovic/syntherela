import os
import subprocess
import json
import ast
from dotenv import load_dotenv

load_dotenv()

"""
GNN Utility Benchmark Script

This script runs benchmarks across:
- Multiple datasets (RUN_DATASETS)
- Multiple synthetic data generation methods (ORIGINAL, CLAVADDPM, MOSTLYAI, etc.)
- Multiple GNN architectures (hetero-graphsage, hetero-gin, hetero-graphconv, hetero-gat, hetero-gatv2)
- Multiple runs (1, 2, 3) for statistical significance

Results are saved in JSON format with structure:
results[dataset][method][gnn_architecture][run_id] = metrics
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

RUN_DATASETS = [
    "rossmann_subsampled",
    # "walmart_subsampled",
    # "airbnb-simplified_subsampled",
    # "f1_subsampled",
    # "Berka_subsampled",
]

# GNN architectures to test
GNN_ARCHITECTURES = [
    "hetero-graphsage",
    "hetero-gin",
    "hetero-graphconv",
    "hetero-gat",
    "hetero-gatv2",
    "relgnn",
]

UTILITY_TASKS = [
    {
        "dataset": "rossmann_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "historical",
        "entity_col": "Id",
        "time_col": "Date",
        "target_col": "Customers",
        "methods": [
            "ORIGINAL",
            "CLAVADDPM",
            "MOSTLYAI",
            "RCTGAN",
            "REALTABFORMER",
            "RGCLD",
            "SDV",
        ],
        "task": "autocomplete",
    },
    {
        "dataset": "walmart_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "depts",
        "entity_col": None,
        "time_col": "Date",
        "target_col": "Weekly_Sales",
        "methods": [
            "ORIGINAL",
            "CLAVADDPM",
            "MOSTLYAI",
            "RCTGAN",
            "REALTABFORMER",
            "RGCLD",
            "SDV",
        ],
        "--lr": 0.1,
        "task": "autocomplete",
    },
    # {
    #     "dataset": "f1_subsampled",
    #     "entity_table": "constructor_standings",
    #     "entity_col": "constructorStandingsId",
    #     "time_col": "date",
    #     "target_col": "position",
    #     "task_type": "REGRESSION",
    #     "methods": [
    #                 "ORIGINAL",
    #                 "CLAVADDPM",
    #                 "RGCLD",
    #                 "MOSTLYAI",
    #                 "RCTGAN",
    #                 "SDV",
    #                 ],
    #     "--lr": 0.005,
    #     "task": "autocomplete",
    # },
    {
        "dataset": "f1_subsampled",
        "task_type": "BINARY_CLASSIFICATION",
        "methods": [
            "ORIGINAL",
            "CLAVADDPM",
            "RGCLD",
            "MOSTLYAI",
            "RCTGAN",
            "SDV",
        ],
        "--lr": 0.005,
        "task": "driver-top3",
    },
    {
        "dataset": "airbnb-simplified_subsampled",
        "task_type": "BINARY_CLASSIFICATION",
        "entity_table": "users",
        "entity_col": "id",
        "time_col": "date_account_created",
        "target_col": "country_destination",
        "methods": [
            "ORIGINAL",
            "CLAVADDPM",
            "MOSTLYAI",
            "RCTGAN",
            "RGCLD",
            "SDV",
        ],
        "--lr": 0.01,
        "task": "autocomplete",
    },
    {
        "dataset": "Berka_subsampled",
        "task_type": "BINARY_CLASSIFICATION",
        "entity_table": "loan",
        "entity_col": "loan_id",
        "time_col": "date",
        "target_col": "status",
        "methods": [
            "ORIGINAL",
            "CLAVADDPM",
            "MOSTLYAI",
            "RGCLD",
        ],
        "--lr": 0.1,
        "--num_layers": 3,
        "task": "autocomplete",
    },
]

results_dir = os.path.join(PROJECT_PATH, "results")
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, "gnn_utility_results_multi_arch.json")

if not os.path.exists(results_file):
    with open(results_file, "w") as f:
        json.dump({}, f)

with open(results_file, "r") as f:
    existing_results = json.load(f)

# print(existing_results)

print(f"=== GNN Utility Benchmark ===")
print(f"Testing {len(RUN_DATASETS)} datasets: {RUN_DATASETS}")
print(f"Testing {len(GNN_ARCHITECTURES)} GNN architectures: {GNN_ARCHITECTURES}")
print(f"Results will be saved to: {results_file}")
print(f"{'='*50}")

for task in UTILITY_TASKS:
    dataset = task["dataset"]
    task_type = task["task_type"]

    if dataset not in RUN_DATASETS:
        continue

    if dataset not in existing_results:
        existing_results[dataset] = {}

    for method in task["methods"]:
        if method not in existing_results[dataset]:
            existing_results[dataset][method] = {}
        try:
            for gnn_arch in GNN_ARCHITECTURES:
                if gnn_arch not in existing_results[dataset][method]:
                    existing_results[dataset][method][gnn_arch] = {}
                    
                for run_id in (1, 2, 3):
                    # check if the result already exists
                    if str(run_id) in existing_results[dataset][method][gnn_arch]:
                        if existing_results[dataset][method][gnn_arch][str(run_id)] != {}:
                            print(
                                f"SKIPPING: {task['dataset']}, Method: {method}, GNN: {gnn_arch}, Run ID: {run_id}"
                            )
                            continue

                    existing_results[dataset][method][gnn_arch][str(run_id)] = {}

                    print(
                        f"Running task: {task['dataset']}, Method: {method}, GNN: {gnn_arch}, Run ID: {run_id}"
                    )

                    command = [
                        "python",
                        "experiments/evaluation/rdl_utility/run_gnn.py",
                        "--dataset",
                        dataset,
                        "--task_type",
                        task_type,
                        "--run_id",
                        str(run_id),
                        "--method",
                        method,
                        "--gnn_architecture",
                        gnn_arch,
                        "--torch_device",
                        "cuda:9",
                        "--task",
                        task["task"],
                    ]
                    if "entity_table" in task:
                        command.extend(["--entity_table", task["entity_table"]])
                    # if "time_col" in task:
                    #     command.extend(["--time_col", task["time_col"]])
                    if "target_col" in task:
                        command.extend(["--target_col", task["target_col"]])
                    # if "entity_col" in task and task["entity_col"] is not None:
                    #     command.extend(["--entity_col", task["entity_col"]])
                    if "--lr" in task:
                        command.extend(["--lr", str(task["--lr"])])
                    if "--batch_size" in task:
                        command.extend(["--batch_size", str(task["--batch_size"])])
                    if "--num_layers" in task:
                        command.extend(["--num_layers", str(task["--num_layers"])])

                    result = subprocess.run(command, capture_output=True, text=True)

                    # Clean up temporary torch_geometric files
                    subprocess.run(["rm", "-f", "torch_geometric.*"])

                    # print(f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}")
                    best_test_metrics = None
                    try:
                        lines = result.stdout.splitlines()
                        final_line = lines[-1]

                        best_test_metrics = final_line.split("Best test metrics: ")[1]
                        print(f"BEST TEST METRICS: {best_test_metrics}")
                    except Exception as e:
                        print(
                            f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}"
                        )
                        print(f"Error: {e}")
                        continue

                    # convert string to dictionary
                    best_test_metrics = ast.literal_eval(best_test_metrics)
                    # print(f"JSON TEST METRICS: {best_test_metrics}")
                    existing_results[dataset][method][gnn_arch][str(run_id)] = best_test_metrics

                    with open(results_file, "w") as f:
                        json.dump(existing_results, f, indent=4)

                    if method == "ORIGINAL":
                        existing_results[dataset][method][gnn_arch]["2"] = best_test_metrics
                        existing_results[dataset][method][gnn_arch]["3"] = best_test_metrics
                        break

                    with open(results_file, "w") as f:
                        json.dump(existing_results, f, indent=4)

        except ValueError as e:
            print(f"Task: {task['dataset']}, Method: {method}, Error: {e}")
            continue

with open(results_file, "w") as f:
    json.dump(existing_results, f, indent=4)

print(f"{'='*50}")
print(f"=== Benchmark Complete ===")
print(f"Results saved to: {results_file}")

# Count total experiments
total_experiments = 0
for dataset in existing_results:
    for method in existing_results[dataset]:
        for gnn_arch in existing_results[dataset][method]:
            for run_id in existing_results[dataset][method][gnn_arch]:
                if existing_results[dataset][method][gnn_arch][run_id] != {}:
                    total_experiments += 1

print(f"Total completed experiments: {total_experiments}")
print(f"{'='*50}")
