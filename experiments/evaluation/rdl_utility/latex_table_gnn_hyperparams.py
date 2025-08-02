"""
Generate LaTeX table for RDL Utility GNN Hyperparameters
Shows custom hyperparameters for each dataset from run_utility_benchmark.py
and default values from run_gnn.py
"""

def generate_latex_table():
    # Dataset renaming for cleaner display
    dataset_rename = {
        "f1_subsampled": "F1",
        "Berka_subsampled": "Berka",
        "rossmann_subsampled": "Rossmann",
        "walmart_subsampled": "Walmart",
        "airbnb-simplified_subsampled": "Airbnb",
    }
    
    # Default hyperparameters from run_gnn.py
    defaults = {
        "lr": 0.1,
        "epochs": 30,
        "batch_size": 512,
        "channels": 128,
        "aggr": "sum",
        "num_layers": 2,
        "gnn_architecture": "hetero-graphsage",
        "num_neighbors": 128,
        "temporal_strategy": "uniform",
        "max_steps_per_epoch": 2000
    }
    
    # Custom hyperparameters for each dataset from UTILITY_TASKS
    dataset_hyperparams = {
        "rossmann_subsampled": {
            # Uses all defaults
        },
        "walmart_subsampled": {
            "lr": 0.1  # Same as default
        },
        "f1_subsampled": {
            "lr": 0.005
        },
        "airbnb-simplified_subsampled": {
            "lr": 0.01
        },
        "Berka_subsampled": {
            "lr": 0.1,
            "num_layers": 2  # Same as default
        }
    }
    
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{GNN Hyperparameters for RDL Utility Evaluation}
\label{tab:gnn_hyperparams}
\begin{tabular}{l|c|c|c|c|c}
\hline
\textbf{Dataset} & \textbf{Learning Rate} & \textbf{Num Layers} & \textbf{Epochs} & \textbf{Batch Size} & \textbf{Channels} \\
\hline
"""
    
    for dataset, custom_params in dataset_hyperparams.items():
        # Get learning rate (custom or default)
        lr = custom_params.get("lr", defaults["lr"])
        
        # Get num_layers (custom or default)
        num_layers = custom_params.get("num_layers", defaults["num_layers"])
        
        # All other parameters use defaults
        epochs = defaults["epochs"]
        batch_size = defaults["batch_size"]
        channels = defaults["channels"]
        
        # Format dataset name using renaming dictionary
        dataset_display = dataset_rename.get(dataset, dataset.replace("_", "\\_"))
        
        # Add row to table
        latex_table += f"{dataset_display} & {lr} & {num_layers} & {epochs} & {batch_size} & {channels} \\\\\n"
    
    latex_table += r"""\hline
\end{tabular}
\end{table}

% Additional hyperparameters (same for all datasets):
% - GNN Architecture: hetero-graphsage
% - Aggregation: sum
% - Num Neighbors: 128
% - Temporal Strategy: uniform
% - Max Steps per Epoch: 2000
"""
    
    return latex_table

def generate_detailed_latex_table():
    """Generate a more detailed table with all relevant hyperparameters"""
    
    # Dataset renaming for cleaner display
    dataset_rename = {
        "f1_subsampled": "F1",
        "Berka_subsampled": "Berka",
        "rossmann_subsampled": "Rossmann",
        "walmart_subsampled": "Walmart",
        "airbnb-simplified_subsampled": "Airbnb",
    }
    
    # Default hyperparameters from run_gnn.py
    defaults = {
        "lr": 0.1,
        "epochs": 30,
        "batch_size": 512,
        "channels": 128,
        "aggr": "sum",
        "num_layers": 2,
        "gnn_architecture": "hetero-graphsage",
        "num_neighbors": 128,
        "temporal_strategy": "uniform",
        "max_steps_per_epoch": 2000
    }
    
    # Custom hyperparameters for each dataset
    dataset_hyperparams = {
        "rossmann_subsampled": {},
        "walmart_subsampled": {"lr": 0.1},
        "f1_subsampled": {"lr": 0.005},
        "airbnb-simplified_subsampled": {"lr": 0.01},
        "Berka_subsampled": {"lr": 0.1, "num_layers": 2}
    }
    
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Detailed GNN Hyperparameters for RDL Utility Evaluation}
\label{tab:gnn_hyperparams_detailed}
\small
\begin{tabular}{lccccccc}
\hline
\textbf{Dataset} & \textbf{LR} & \textbf{Layers} & \textbf{Epochs} & \textbf{Batch} & \textbf{Channels} & \textbf{Neighbors} & \textbf{Max Steps} \\
\hline
"""
    
    for dataset, custom_params in dataset_hyperparams.items():
        # Get parameters (custom or default)
        lr = custom_params.get("lr", defaults["lr"])
        num_layers = custom_params.get("num_layers", defaults["num_layers"])
        epochs = defaults["epochs"]
        batch_size = defaults["batch_size"]
        channels = defaults["channels"]
        num_neighbors = defaults["num_neighbors"]
        max_steps = defaults["max_steps_per_epoch"]
        
        # Format dataset name using renaming dictionary
        dataset_display = dataset_rename.get(dataset, dataset.replace("_", "\\_").replace("-", "-"))
        
        # Format parameters without bold formatting
        lr_display = str(lr)
        layers_display = str(num_layers)
        
        # Add row to table
        latex_table += f"{dataset_display} & {lr_display} & {layers_display} & {epochs} & {batch_size} & {channels} & {num_neighbors} & {max_steps} \\\\\n"
    
    latex_table += r"""\hline
\end{tabular}
\end{table}

% Note: Values show both custom and default hyperparameters used for each dataset
% Common settings for all datasets:
% - GNN Architecture: hetero-graphsage
% - Aggregation: sum  
% - Temporal Strategy: uniform
"""
    
    return latex_table

if __name__ == "__main__":
    print("=== Basic LaTeX Table ===")
    print(generate_latex_table())
    
    print("\n=== Detailed LaTeX Table ===")
    print(generate_detailed_latex_table())
