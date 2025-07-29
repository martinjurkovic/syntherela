from gnn_datasets import (
    RossmannDataset,
    WalmartDataset,
    F1Dataset,
    AirbnbDataset,
    BerkaDataset,
)
import pandas as pd

# Configuration
INCLUDE_TASKS_COLUMN = False  # Set to False to exclude the #Tasks column

def get_dataset_statistics():
    """Extract statistics for each dataset and return as a list of dictionaries."""
    
    datasets_info = [
        {
            "class": RossmannDataset,
            "name": "rossmann",
            "domain": "E-commerce",
            "tasks": 1,  # Assuming one main prediction task per dataset
        },
        {
            "class": WalmartDataset,
            "name": "walmart", 
            "domain": "E-commerce",
            "tasks": 1,
        },
        {
            "class": AirbnbDataset,
            "name": "airbnb",
            "domain": "E-commerce",
            "tasks": 1,
        },
        {
            "class": BerkaDataset,
            "name": "berka",
            "domain": "Financial",
            "tasks": 1,
        },
        {
            "class": F1Dataset,
            "name": "f1",
            "domain": "Sports", 
            "tasks": 1,
        }
    ]
    
    statistics = []
    
    for dataset_info in datasets_info:
        print(f"Processing {dataset_info['name']}...")
        
        try:
            # Instantiate dataset
            dataset = dataset_info["class"](method="ORIGINAL", type="train")
            
            # Get database
            db = dataset.make_db()
            
            # Extract table statistics
            num_tables = len(db.table_dict)
            total_rows = sum(len(table.df) for table in db.table_dict.values())
            total_cols = sum(len(table.df.columns) for table in db.table_dict.values())
            
            # Get timestamps
            val_timestamp = getattr(dataset, 'val_timestamp', None)
            test_timestamp = getattr(dataset, 'test_timestamp', None)
            from_timestamp = getattr(dataset, 'from_timestamp', None)
            
            # If from_timestamp is not defined, extract from the actual data
            if from_timestamp is None:
                earliest_dates = []
                for table_name, table in db.table_dict.items():
                    if table.time_col is not None and table.time_col in table.df.columns:
                        # Get the earliest non-null date from this time column
                        time_series = pd.to_datetime(table.df[table.time_col], errors='coerce')
                        earliest_date = time_series.min()
                        if pd.notna(earliest_date):
                            earliest_dates.append(earliest_date)
                
                if earliest_dates:
                    from_timestamp = min(earliest_dates)
            
            # Format timestamps as year-mon-day
            start_date = from_timestamp.strftime("%Y-%m-%d") if from_timestamp else "N/A"
            val_date = val_timestamp.strftime("%Y-%m-%d") if val_timestamp else "N/A"
            test_date = test_timestamp.strftime("%Y-%m-%d") if test_timestamp else "N/A"
            
            statistics.append({
                "name": dataset_info["name"],
                "domain": dataset_info["domain"],
                "tasks": dataset_info["tasks"],
                "tables": num_tables,
                "rows": total_rows,
                "cols": total_cols,
                "start": start_date,
                "val": val_date,
                "test": test_date
            })
            
            print(f"  Tables: {num_tables}, Rows: {total_rows:,}, Cols: {total_cols}")
            
        except Exception as e:
            print(f"Error processing {dataset_info['name']}: {e}")
            continue
    
    return statistics

def generate_latex_table(statistics):
    """Generate LaTeX table from the statistics."""
    
    if INCLUDE_TASKS_COLUMN:
        tabular_spec = "lllrrrrccc"
        header1 = r"\multirow{2}{*}{Name} & \multirow{2}{*}{Domain} & \multirow{2}{*}{\#Tasks} & \multicolumn{3}{c}{Tables} & \multicolumn{3}{c}{Timestamp (year-mon-day)} \\"
        cmidrule = r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}"
        header2 = r" &  &  & \#Tables & \#Rows & \#Cols & Start & Val & Test \\"
    else:
        tabular_spec = "llrrrrrrr"
        header1 = r"\multirow{2}{*}{Name} & \multirow{2}{*}{Domain} & \multicolumn{3}{c}{Tables} & \multicolumn{3}{c}{Timestamp (year-mon-day)} \\"
        cmidrule = r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}"
        header2 = r" &  & \#Tables & \#Rows & \#Cols & Start & Val & Test \\"
    
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
    
    # Add data rows
    total_tasks = 0
    total_tables = 0
    total_rows = 0
    total_cols = 0
    
    for stat in statistics:
        total_tasks += stat["tasks"]
        total_tables += stat["tables"]
        total_rows += stat["rows"]
        total_cols += stat["cols"]
        
        if INCLUDE_TASKS_COLUMN:
            latex += f"{stat['name']} & {stat['domain']} & {stat['tasks']} & {stat['tables']} & {stat['rows']:,} & {stat['cols']} & {stat['start']} & {stat['val']} & {stat['test']} \\\\\n"
        else:
            latex += f"{stat['name']} & {stat['domain']} & {stat['tables']} & {stat['rows']:,} & {stat['cols']} & {stat['start']} & {stat['val']} & {stat['test']} \\\\\n"
    
    # Add total row
    latex += r"\midrule" + "\n"
    if INCLUDE_TASKS_COLUMN:
        latex += f"\\multicolumn{{2}}{{l}}{{Total}} & {total_tasks} & {total_tables} & {total_rows:,} & {total_cols} & / & / & / \\\\\n"
    else:
        latex += f"\\multicolumn{{2}}{{l}}{{Total}} & {total_tables} & {total_rows:,} & {total_cols} & / & / & / \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\caption{Dataset statistics for relational datasets used in evaluation.}
\label{tab:dataset_statistics}
\end{table}
"""
    
    return latex

def main():
    """Main function to generate the LaTeX table."""
    print("Extracting dataset statistics...")
    statistics = get_dataset_statistics()
    
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(statistics)
    
    print("\nLaTeX Table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
    
    # Also print a summary
    print("\nDataset Summary:")
    print("-" * 60)
    for stat in statistics:
        print(f"{stat['name']:15} | {stat['domain']:12} | {stat['tables']:2d} tables | {stat['rows']:8,} rows | {stat['cols']:3d} cols")

if __name__ == "__main__":
    main()


