import os
import subprocess
import json
import ast
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

RUN_DATASETS = [
    "rossmann_subsampled",
    "walmart_subsampled",
    "airbnb-simplified_subsampled",
    "f1_subsampled",
    "Berka_subsampled",
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

results_file = os.path.join(results_dir, "gnn_baseline_results.json")

if not os.path.exists(results_file):
    with open(results_file, "w") as f:
        json.dump({}, f)

with open(results_file, "r") as f:
    existing_results = json.load(f)

# print(existing_results)

for task in UTILITY_TASKS:
    dataset = task["dataset"]
    task_type = task["task_type"]

    if dataset not in RUN_DATASETS:
        continue

    if dataset not in existing_results:
        existing_results[dataset] = {}

    for method in task["methods"]:
        if method != "ORIGINAL":
            continue
        if method not in existing_results[dataset]:
            existing_results[dataset][method] = {}
        try:
            for run_id in (1, 2, 3):
                # check if the result already exists
                if str(run_id) in existing_results[dataset][method]:
                    if existing_results[dataset][method][str(run_id)] != {}:
                        print(
                            f"SKIPPING: {task['dataset']}, Method: {method}, Run ID: {run_id}"
                        )
                        continue

                existing_results[dataset][method][str(run_id)] = {}

                print(
                    f"Running task: {task['dataset']}, Method: {method}, Run ID: {run_id}"
                )

                command = [
                    "python",
                    "experiments/evaluation/rdl_utility/run_baseline.py",
                    "--dataset",
                    dataset,
                    "--run",
                    str(run_id),
                    "--method",
                    method,
                    "--task",
                    task["task"],
                    "--task_type",
                    task_type,
                ]
                if "entity_table" in task:
                    command.extend(["--entity_table", task["entity_table"]])
                if "target_col" in task:
                    command.extend(["--target_col", task["target_col"]])

                result = subprocess.run(command, capture_output=True, text=True)

                # Parse the baseline results from the output
                baseline_results = None
                try:
                    lines = result.stdout.splitlines()

                    # Find the baseline results in the last part of the output
                    baseline_results = {}
                    current_method = None

                    # Look for baseline method lines (ending with ':')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line and line.endswith(':') and not line.startswith('Train:') and not line.startswith('Val:') and not line.startswith('Test:'):
                            current_method = line[:-1]  # Remove the ':'
                            baseline_results[current_method] = {}

                            # Extract Train/Val/Test results for this method
                            for j in range(i + 1, min(i + 4, len(lines))):
                                if j < len(lines):
                                    result_line = lines[j].strip()
                                    if result_line.startswith('Train:'):
                                        metrics_str = result_line.split('Train: ')[1]
                                        baseline_results[current_method]['Train'] = ast.literal_eval(metrics_str)
                                    elif result_line.startswith('Val:'):
                                        metrics_str = result_line.split('Val: ')[1]
                                        baseline_results[current_method]['Val'] = ast.literal_eval(metrics_str)
                                    elif result_line.startswith('Test:'):
                                        metrics_str = result_line.split('Test: ')[1]
                                        baseline_results[current_method]['Test'] = ast.literal_eval(metrics_str)

                    print(f"BASELINE RESULTS: {baseline_results}")
                except Exception as e:
                    print(
                        f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}"
                    )
                    print(f"Parsing Error: {e}")
                    continue

                # Store the results
                existing_results[dataset][method][str(run_id)] = baseline_results

                # Save results to file after each run
                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=4)

                if method == "ORIGINAL":
                    existing_results[dataset][method]["2"] = baseline_results
                    existing_results[dataset][method]["3"] = baseline_results
                    break

                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=4)



        except ValueError as e:
            print(f"Task: {task['dataset']}, Method: {method}, Error: {e}")
            continue

with open(results_file, "w") as f:
    json.dump(existing_results, f, indent=4)
