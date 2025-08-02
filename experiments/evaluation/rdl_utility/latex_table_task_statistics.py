from gnn_datasets import (
    RossmannDataset,
    WalmartDataset,
    F1Dataset,
    AirbnbDataset,
    BerkaDataset,
)
import pandas as pd
from relbench.tasks import get_task 
from relbench.base import BaseTask, EntityTask, TaskType
from relbench.tasks.f1 import DriverTop3Task
from relbench.base.task_autocomplete import AutoCompleteTask

# Configuration
INCLUDE_DST_ENTITIES_COLUMN = False  # Set to False to exclude the #Dst Entities column

TASKS = {
    "driver-top3": DriverTop3Task,
    "autocomplete": AutoCompleteTask,
}

DATASETS = {
    RossmannDataset.name: RossmannDataset,
    WalmartDataset.name: WalmartDataset,
    F1Dataset.name: F1Dataset,
    AirbnbDataset.name: AirbnbDataset,
    BerkaDataset.name: BerkaDataset,
}

def get_task_statistics():
    """Extract task statistics for each dataset and return as a list of dictionaries."""
    
    # Task configurations based on run_utility_benchmark.py
    tasks_info = [
        {
            "dataset_class": RossmannDataset,
            "dataset_name": "rossmann_subsampled",
            "display_name": "rossmann",
            "task_name": "autocomplete",
            "task_type": "REGRESSION",
            "entity_table": "historical",
            "entity_col": "Id",
            "time_col": "Date",
            "target_col": "Customers",
        },
        {
            "dataset_class": WalmartDataset,
            "dataset_name": "walmart_subsampled", 
            "display_name": "Walmart",
            "task_name": "autocomplete",
            "task_type": "REGRESSION",
            "entity_table": "depts",
            "entity_col": None,
            "time_col": "Date",
            "target_col": "Weekly_Sales",
        },
        {
            "dataset_class": AirbnbDataset,
            "dataset_name": "airbnb-simplified_subsampled",
            "display_name": "Airbnb",
            "task_name": "autocomplete", 
            "task_type": "BINARY_CLASSIFICATION",
            "entity_table": "users",
            "entity_col": "id",
            "time_col": "date_account_created",
            "target_col": "country_destination",
        },
        {
            "dataset_class": BerkaDataset,
            "dataset_name": "Berka_subsampled",
            "display_name": "berka",
            "task_name": "autocomplete",
            "task_type": "BINARY_CLASSIFICATION", 
            "entity_table": "loan",
            "entity_col": "loan_id",  # Primary key from the dataset
            "time_col": "date",
            "target_col": "status",
        },
        {
            "dataset_class": F1Dataset,
            "dataset_name": "f1_subsampled",
            "display_name": "f1",
            "task_name": "driver-top3",
            "task_type": "BINARY_CLASSIFICATION",
            "entity_table": "drivers",  # Main entity table for F1 predictions
            "entity_col": "driverId",
            "time_col": "date", 
            "target_col": "qualifying",  # Will be transformed to binary for top3
        }
    ]
    
    statistics = []
    
    for task_info in tasks_info:
        print(f"Processing {task_info['display_name']} - {task_info['task_name']}...")

        try:
            # Instantiate dataset
            dataset = task_info["dataset_class"](method="ORIGINAL", type="train")
            dataset_test = task_info["dataset_class"](method="ORIGINAL", type="test")
            # Get database
            

            if task_info["task_name"] == "autocomplete":
                dataset.target_col = task_info["target_col"]
                dataset.entity_table = task_info["entity_table"]
                dataset_test.target_col = task_info["target_col"]
                dataset_test.entity_table = task_info["entity_table"]
                task = AutoCompleteTask(dataset=dataset, task_type=TaskType[task_info["task_type"]], entity_table=task_info["entity_table"], target_col=task_info["target_col"])
                task_test = AutoCompleteTask(dataset=dataset_test, task_type=TaskType[task_info["task_type"]], entity_table=task_info["entity_table"], target_col=task_info["target_col"])
            else:
                task: BaseTask = TASKS[task_info["task_name"]](dataset=dataset)
                task_test: EntityTask = get_task("rel-f1", task_info["task_name"], download=False)

            train_table = task.get_table("train", mask_input_cols=True)
            val_table = task.get_table("val", mask_input_cols=False)
            test_table = task_test.get_table("test", mask_input_cols=False)
            
            # Calculate basic statistics
            train_df = train_table.df
            val_df = val_table.df
            test_df = test_table.df
            
            train_rows = len(train_df)
            val_rows = len(val_df) 
            test_rows = len(test_df)
            
            # Calculate unique entities
            entity_col = task_info["entity_col"]
            # If entity_col is None, use "primary_key" as created in dataset construction
            if entity_col is None:
                entity_col = "primary_key"
            
            if entity_col and entity_col in train_df.columns:
                unique_entities = train_df[entity_col].nunique()
                
                # Calculate train/test entity overlap if we have both splits
                if entity_col != "primary_key" and len(test_df) > 0 and entity_col in test_df.columns:
                    train_entities = set(train_df[entity_col].unique())
                    val_entities = set(val_df[entity_col].unique())
                    train_val_entities = set(train_entities).union(val_entities)
                    test_entities = set(test_df[entity_col].unique())
                    overlap = len(train_val_entities.intersection(test_entities))
                    total_test_entities = len(test_entities)
                    overlap_pct = (overlap / total_test_entities * 100) if total_test_entities > 0 else 0
                else:
                    overlap_pct = 0
            else:
                unique_entities = 0
                overlap_pct = 0
            
            # For #Dst Entities - this seems to be destination entities for recommendation tasks
            # Based on the image, only some tasks have this (recommendation tasks)
            dst_entities = "—"  # Default to em dash
            if INCLUDE_DST_ENTITIES_COLUMN:
                if "recommendation" in task_info.get("task_type", "").lower() or "purchase" in task_info.get("target_col", "").lower():
                    # For recommendation tasks, count unique target items
                    target_col = task_info["target_col"]
                    if target_col in train_df.columns:
                        dst_entities = f"{train_df[target_col].nunique():,}"
            
            statistics.append({
                "dataset": task_info["display_name"],
                "task_name": task_info["task_name"].capitalize(),
                "task_type": "Classification" if task_info["task_type"] == "BINARY_CLASSIFICATION" else "Regression",
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "unique_entities": unique_entities,
                "overlap_pct": overlap_pct,
                "dst_entities": dst_entities
            })
            
            print(f"  Train: {train_rows:,}, Val: {val_rows:,}, Test: {test_rows:,}")
            print(f"  Unique entities: {unique_entities:,}, Overlap: {overlap_pct:.1f}%")
            
        except Exception as e:
            print(f"Error processing {task_info['display_name']}: {e}")
            continue
    
    return statistics

def generate_latex_table(statistics):
    """Generate LaTeX table from the task statistics."""
    
    if INCLUDE_DST_ENTITIES_COLUMN:
        tabular_spec = "llcrrrcrc"
        header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Task name} & \multirow{2}{*}{Task type} & \multicolumn{3}{c}{\#Rows of training table} & \multirow{2}{*}{\#Unique Entities} & \multirow{2}{*}{\%Train-Val/test Entity Overlap} & \multirow{2}{*}{\#Dst Entities} \\"
        cmidrule = r"\cmidrule(lr){4-6}"
        header2 = r" &  &  & Train & Validation & Test &  &  &  \\"
    else:
        tabular_spec = "llcrrrcc"
        header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Task name} & \multirow{2}{*}{Task type} & \multicolumn{3}{c}{\#Rows of training table} & \multirow{2}{*}{\#Unique Entities} & \multirow{2}{*}{\%Train-Val/test Entity Overlap} \\"
        cmidrule = r"\cmidrule(lr){4-6}"
        header2 = r" &  &  & Train & Validation & Test &  &  \\"
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{{tabular_spec}}}
\\toprule
{header1}
{cmidrule}
{header2}
\\midrule
"""
    
    for stat in statistics:
        if INCLUDE_DST_ENTITIES_COLUMN:
            latex += f"{stat['dataset']} & {stat['task_name']} & {stat['task_type']} & {stat['train_rows']:,} & {stat['val_rows']:,} & {stat['test_rows']:,} & {stat['unique_entities']:,} & {stat['overlap_pct']:.1f} & {stat['dst_entities']} \\\\\n"
        else:
            latex += f"{stat['dataset']} & {stat['task_name']} & {stat['task_type']} & {stat['train_rows']:,} & {stat['val_rows']:,} & {stat['test_rows']:,} & {stat['unique_entities']:,} & {stat['overlap_pct']:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\caption{Task statistics for relational datasets used in evaluation.}
\label{tab:task_statistics}
\end{table}
"""
    
    return latex

def main():
    """Main function to generate the LaTeX table."""
    print("Extracting task statistics...")
    statistics = get_task_statistics()
    
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(statistics)
    
    print("\nLaTeX Table:")
    print("=" * 120)
    print(latex_table)
    print("=" * 120)
    
    # Also print a summary
    print("\nTask Summary:")
    if INCLUDE_DST_ENTITIES_COLUMN:
        print("-" * 120)
        print(f"{'Dataset':<15} | {'Task':<15} | {'Type':<12} | {'Train':<10} | {'Val':<8} | {'Test':<8} | {'Entities':<8} | {'Overlap%':<8} | {'Dst':<8}")
        print("-" * 120)
        for stat in statistics:
            print(f"{stat['dataset']:<15} | {stat['task_name']:<15} | {stat['task_type']:<12} | {stat['train_rows']:<10,} | {stat['val_rows']:<8,} | {stat['test_rows']:<8,} | {stat['unique_entities']:<8,} | {stat['overlap_pct']:<8.1f} | {stat['dst_entities']:<8}")
    else:
        print("-" * 100)
        print(f"{'Dataset':<15} | {'Task':<15} | {'Type':<12} | {'Train':<10} | {'Val':<8} | {'Test':<8} | {'Entities':<8} | {'Overlap%':<8}")
        print("-" * 100)
        for stat in statistics:
            print(f"{stat['dataset']:<15} | {stat['task_name']:<15} | {stat['task_type']:<12} | {stat['train_rows']:<10,} | {stat['val_rows']:<8,} | {stat['test_rows']:<8,} | {stat['unique_entities']:<8,} | {stat['overlap_pct']:<8.1f}")

if __name__ == "__main__":
    main()
