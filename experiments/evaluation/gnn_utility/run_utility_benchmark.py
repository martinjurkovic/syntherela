import os
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

RUN_DATASETS = ["f1_subsampled"]

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
        "task": "predict-column",
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
        "task": "predict-column",
    },
    {
        "dataset": "f1_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "constructor_standings",
        "entity_col": "constructorStandingsId",
        "time_col": "date",
        "target_col": "position",
        "methods": ["ORIGINAL", "CLAVADDPM", "MOSTLYAI", "RCTGAN", "RGCLD", "SDV"],
        "--lr": 0.001,
        "task": "driver-top3",
    },
    {
        "dataset": "airbnb-simplified_subsampled",
        "task_type": "BINARY_CLASSIFICATION",
        "entity_table": "users",
        "entity_col": "id",
        "time_col": "date_account_created",
        "target_col": "country_destination",
        "methods": ["ORIGINAL", "CLAVADDPM", "MOSTLYAI", "RCTGAN", "RGCLD", "SDV"],
        "--lr": 0.1,
        "task": "predict-column",
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
            "RCTGAN",
            "REALTABFORMER",
            "SDV",
            "RGCLD",
        ],
        "--lr": 0.1,
        "--batch_size": 64,
        "task": "predict-column",
    },
]

results_dir = os.path.join(PROJECT_PATH, "results")
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, "gnn_utility_results.json")

if not os.path.exists(results_file):
    with open(results_file, "w") as f:
        json.dump({}, f)

with open(results_file, "r") as f:
    existing_results = json.load(f)

# print(existing_results)

for task in UTILITY_TASKS:
    dataset = task["dataset"]
    task_type = task["task_type"]
    entity_table = task["entity_table"]
    entity_col = task["entity_col"]
    time_col = task["time_col"]
    target_col = task["target_col"]

    if dataset not in RUN_DATASETS:
        continue

    if dataset not in existing_results:
        existing_results[dataset] = {}

    for method in task["methods"]:
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

                existing_results[dataset][method][run_id] = {}

                print(
                    f"Running task: {task['dataset']}, Method: {method}, Run ID: {run_id}"
                )

                command = [
                    "python",
                    "experiments/evaluation/gnn_utility/run_gnn.py",
                    "--dataset",
                    dataset,
                    "--task_type",
                    task_type,
                    "--entity_table",
                    entity_table,
                    "--time_col",
                    time_col,
                    "--target_col",
                    target_col,
                    "--run",
                    str(run_id),
                    "--method",
                    method,
                    "--torch_device",
                    "cuda:9",
                    "--task",
                    task,
                ]
                if entity_col is not None:
                    command.extend(["--entity_col", entity_col])
                if "--lr" in task:
                    command.extend(["--lr", str(task["--lr"])])
                if "--batch_size" in task:
                    command.extend(["--batch_size", str(task["--batch_size"])])
                # subprocess.run(command)
                result = subprocess.run(command, capture_output=True, text=True)

                # Clean up temporary torch_geometric files
                subprocess.run(["rm", "-f", "torch_geometric.*"])

                # print(f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}")
                best_test_metrics = None
                try:
                    lines = result.stdout.splitlines()
                    final_line = lines[-1]

                    best_test_metrics = final_line.split("Best test metrics: ")[1]
                    print(f"STRING BEST TEST METRICS: {best_test_metrics}")
                except Exception as e:
                    print(
                        f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}"
                    )
                    raise e

                # convert string to dictionary
                best_test_metrics = json.loads(best_test_metrics.replace("'", '"'))
                print(f"JSON TEST METRICS: {best_test_metrics}")
                existing_results[dataset][method][run_id] = best_test_metrics

                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=4)

                if method == "ORIGINAL":
                    existing_results[dataset][method][2] = best_test_metrics
                    existing_results[dataset][method][3] = best_test_metrics
                    break

                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=4)

        except ValueError as e:
            print(f"Task: {task['dataset']}, Method: {method}, Error: {e}")
            continue

with open(results_file, "w") as f:
    json.dump(existing_results, f, indent=4)
