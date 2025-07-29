from gnn_datasets import (
    RossmannDataset,
    WalmartDataset,
    F1Dataset,
    AirbnbDataset,
    BerkaDataset,
)
import pandas as pd
from relbench.tasks import get_task
from relbench.base.task_autocomplete import AutoCompleteTask

# Configuration
INCLUDE_DST_ENTITIES_COLUMN = False  # Set to False to exclude the #Dst Entities column

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
            "display_name": "walmart",
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
            "display_name": "airbnb",
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
            dataset.target_col = task_info["target_col"]
            dataset.entity_table = task_info["entity_table"]
            dataset_test = task_info["dataset_class"](method="ORIGINAL", type="test")
            dataset_test.target_col = task_info["target_col"]
            dataset_test.entity_table = task_info["entity_table"]
            # Get database
            db = dataset.get_db(upto_test_timestamp=True)
            db_test = dataset_test.get_db(upto_test_timestamp=False)
            
            # Get the entity table
            entity_table_name = task_info["entity_table"]
            entity_table = db.table_dict[entity_table_name]
            entity_df = entity_table.df

            entity_table_test = db_test.table_dict[entity_table_name]
            entity_df_test = entity_table_test.df
            
            # Calculate basic statistics
            if hasattr(dataset, 'val_timestamp') and hasattr(dataset, 'test_timestamp'):
                val_timestamp = dataset.val_timestamp
                test_timestamp = dataset.test_timestamp
                time_col = task_info["time_col"]
                
                if time_col in entity_df.columns:
                    # Convert time column to datetime
                    entity_df[time_col] = pd.to_datetime(entity_df[time_col], errors='coerce')
                    
                    # Split into train/val/test based on timestamps
                    train_df = entity_df[entity_df[time_col] < val_timestamp]
                    val_df = entity_df[(entity_df[time_col] >= val_timestamp) & (entity_df[time_col] < test_timestamp)]
                    test_df = entity_df_test[entity_df_test[time_col] >= test_timestamp]
                    
                    train_rows = len(train_df)
                    val_rows = len(val_df) 
                    test_rows = len(test_df)
                else:
                    # If no time column, use total rows for train and 0 for others
                    train_rows = len(entity_df)
                    val_rows = 0
                    test_rows = 0
                    train_df = entity_df
                    test_df = pd.DataFrame()
            else:
                # No temporal split available
                train_rows = len(entity_df)
                val_rows = 0
                test_rows = 0
                train_df = entity_df
                test_df = pd.DataFrame()
            
            # Calculate unique entities
            entity_col = task_info["entity_col"]
            # If entity_col is None, use "primary_key" as created in dataset construction
            if entity_col is None:
                entity_col = "primary_key"
            
            if entity_col and entity_col in entity_df.columns:
                unique_entities = entity_df[entity_col].nunique()
                
                # Calculate train/test entity overlap if we have both splits
                if entity_col != "primary_key" and len(test_df) > 0 and entity_col in test_df.columns:
                    train_entities = set(train_df[entity_col].unique())
                    test_entities = set(test_df[entity_col].unique())
                    overlap = len(train_entities.intersection(test_entities))
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
                    if target_col in entity_df.columns:
                        dst_entities = f"{entity_df[target_col].nunique():,}"
            
            statistics.append({
                "dataset": task_info["display_name"],
                "task_name": task_info["task_name"],
                "task_type": "entity-reg" if task_info["task_type"] == "REGRESSION" else "entity-cls",
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
        header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Task name} & \multirow{2}{*}{Task type} & \multicolumn{3}{c}{\#Rows of training table} & \multirow{2}{*}{\#Unique Entities} & \multirow{2}{*}{\%train/test Entity Overlap} & \multirow{2}{*}{\#Dst Entities} \\"
        cmidrule = r"\cmidrule(lr){4-6}"
        header2 = r" &  &  & Train & Validation & Test &  &  &  \\"
    else:
        tabular_spec = "llcrrrcc"
        header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Task name} & \multirow{2}{*}{Task type} & \multicolumn{3}{c}{\#Rows of training table} & \multirow{2}{*}{\#Unique Entities} & \multirow{2}{*}{\%train/test Entity Overlap} \\"
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
