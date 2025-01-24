import subprocess


UTILITY_TASKS = [
    {
        "dataset": "rossmann_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "historical",
        "entity_col": "Id",
        "time_col": "Date",
        "target_col": "Customers",
    },
    {
        "dataset": "walmart_subsampled",
        "task_type": "REGRESSION",
        "entity_table": "depts",
        "entity_col": None,
        "time_col": "Date",
        "target_col": "Weekly_Sales",
    },
]

for task in UTILITY_TASKS:
    dataset = task["dataset"]
    task_type = task["task_type"]
    entity_table = task["entity_table"]
    entity_col = task["entity_col"]
    time_col = task["time_col"]
    target_col = task["target_col"]

    if dataset != "walmart_subsampled":
        continue

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
    ]
    if entity_col is not None:
        command.extend(["--entity_col", entity_col])
    subprocess.run(command)
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Task: {task['dataset']}, Output: {result.stdout}, Error: {result.stderr}")
