import os
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

RUN_DATASETS = [
    # "rossmann_subsampled",
    # "walmart_subsampled",
    # "airbnb-simplified_subsampled",
    # "f1_subsampled",
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
        "entity_table": "constructor_standings",
        "entity_col": "constructorStandingsId",
        "time_col": "date",
        "target_col": "position",
        "task_type": "REGRESSION",
        "methods": [
                    "ORIGINAL", 
                    "CLAVADDPM", 
                    "RGCLD", 
                    "MOSTLYAI", 
                    "RCTGAN", 
                    "SDV",
                    ],
        "--lr": 0.005,
        "task": "predict-column",
    },
    {
        "dataset": "airbnb-simplified_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "users",
        "entity_col": "id",
        "time_col": "date_account_created",
        "target_col": "country_destination",
        "methods": ["ORIGINAL", "CLAVADDPM", "MOSTLYAI", "RCTGAN", "RGCLD", "SDV"],
        "--lr": 0.01,
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
            # "RCTGAN",
            # "SDV",
            # "ORIGINAL",
            # "CLAVADDPM",
            # "MOSTLYAI",
            "RGCLD",
        ],
        "--lr": 0.1,
        "--num_layers": 3,
        "task": "predict-column",
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

                existing_results[dataset][method][run_id] = {}

                print(
                    f"Running task: {task['dataset']}, Method: {method}, Run ID: {run_id}"
                )

                command = [
                    "python",
                    "experiments/evaluation/gnn_utility/run_baseline.py",
                    "--dataset",
                    dataset,
                    "--run",
                    str(run_id),
                    "--method",
                    method,
                    "--task",
                    task["task"],
                ]
                if "entity_table" in task:
                    command.extend(["--entity_table", task["entity_table"]])
                if "time_col" in task:
                    command.extend(["--time_col", task["time_col"]])
                if "target_col" in task:
                    command.extend(["--target_col", task["target_col"]])
                if "entity_col" in task and task["entity_col"] is not None:
                    command.extend(["--entity_col", task["entity_col"]])
                # if "--lr" in task:
                #     command.extend(["--lr", str(task["--lr"])])
                # if "--batch_size" in task:
                #     command.extend(["--batch_size", str(task["--batch_size"])])
                # if "--num_layers" in task:
                #     command.extend(["--num_layers", str(task["--num_layers"])])
                
                result = subprocess.run(command, capture_output=False, text=True)

                # Clean up temporary torch_geometric files
                # subprocess.run(["rm", "-f", "torch_geometric.*"])

                # print(f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}")
                # best_test_metrics = None
                # try:
                #     lines = result.stdout.splitlines()
                #     final_line = lines[-1]

                #     best_test_metrics = final_line.split("Best test metrics: ")[1]
                #     print(f"BEST TEST METRICS: {best_test_metrics}")
                # except Exception as e:
                #     print(
                #         f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}"
                #     )
                #     print(f"Error: {e}")
                #     continue

                # # convert string to dictionary
                # best_test_metrics = json.loads(best_test_metrics.replace("'", '"'))
                # # print(f"JSON TEST METRICS: {best_test_metrics}")
                # existing_results[dataset][method][run_id] = best_test_metrics

                # with open(results_file, "w") as f:
                #     json.dump(existing_results, f, indent=4)

                # if method == "ORIGINAL":
                #     existing_results[dataset][method][2] = best_test_metrics
                #     existing_results[dataset][method][3] = best_test_metrics
                #     break

                # with open(results_file, "w") as f:
                #     json.dump(existing_results, f, indent=4)

        except ValueError as e:
            print(f"Task: {task['dataset']}, Method: {method}, Error: {e}")
            continue

# with open(results_file, "w") as f:
#     json.dump(existing_results, f, indent=4)
