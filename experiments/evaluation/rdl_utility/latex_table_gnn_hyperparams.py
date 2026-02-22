"""
Generate LaTeX table for RDL Utility GNN Hyperparameters
Uses actual hyperparameters from hyperparameter tuning results
Creates separate tables for each GNN architecture
"""

import json
import os


def load_hyperparameters():
    """Load hyperparameters from JSON files organized by GNN architecture and dataset"""

    # Path to hyperparameter results
    results_dir = "/lfs/hyperturing1/0/martinj1/syntherela/results/hyperparameter_tuning_100"

    # Initialize nested dictionary: {gnn_architecture: {dataset: hyperparams}}
    hyperparams = {}

    # GNN architectures to process
    gnn_architectures = [
        "hetero_gat", "hetero_gatv2", "hetero_gin", "hetero_graphconv",
        "hetero_graphsage", "relgnn"
    ]

    # Datasets to process
    datasets = [
        "f1_subsampled", "Berka_subsampled", "rossmann_subsampled",
        "walmart_subsampled", "airbnb_simplified_subsampled"
    ]

    for gnn_arch in gnn_architectures:
        hyperparams[gnn_arch] = {}

        for dataset in datasets:
            # Use the dataset name as-is for file naming
            filename = f"hyperparameter_results_{gnn_arch}_{dataset}.json"
            filepath = os.path.join(results_dir, filename)

            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        hyperparams[gnn_arch][dataset] = data[
                            'best_hyperparameters']
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")

    return hyperparams


def generate_gnn_hyperparameter_table(gnn_arch, hyperparams):
    """Generate LaTeX table for a specific GNN architecture"""

    # Dataset renaming for cleaner display
    dataset_rename = {
        "f1_subsampled": "F1",
        "Berka_subsampled": "Berka",
        "rossmann_subsampled": "Rossmann",
        "walmart_subsampled": "Walmart",
        "airbnb_simplified_subsampled": "Airbnb",
    }

    # GNN architecture display names
    gnn_display_names = {
        "hetero_gat": "Hetero-GAT",
        "hetero_gatv2": "Hetero-GATv2",
        "hetero_gin": "Hetero-GIN",
        "hetero_graphconv": "Hetero-GraphConv",
        "hetero_graphsage": "Hetero-GraphSAGE",
        "relgnn": "RelGNN"
    }

    gnn_display = gnn_display_names.get(gnn_arch, gnn_arch)

    # Determine hyperparameters to include based on available data
    all_params = set()
    dataset_params = hyperparams.get(gnn_arch, {})

    for dataset_hyperparams in dataset_params.values():
        all_params.update(dataset_hyperparams.keys())

    # Order parameters consistently
    param_order = [
        "lr", "num_layers", "num_neighbors", "weight_decay", "mlp_layers",
        "aggr"
    ]
    params_to_show = [p for p in param_order if p in all_params]

    # Create table header
    header_map = {
        "lr": "LR",
        "num_layers": "Layers",
        "num_neighbors": "Neighbors",
        "weight_decay": "Weight Decay",
        "mlp_layers": "MLP Layers",
        "aggr": "Aggregation"
    }

    headers = ["Dataset"] + [header_map.get(p, p) for p in params_to_show]
    header_str = " & ".join(f"\\textbf{{{h}}}" for h in headers)

    # Calculate number of columns for table format
    num_cols = len(headers)
    col_format = "l" + "c" * (num_cols - 1)

    latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{gnn_display} Hyperparameters for RDL Utility Evaluation}}
\\label{{tab:gnn_hyperparams_{gnn_arch}}}
\\begin{{tabular}}{{{col_format}}}
\\hline
{header_str} \\\\
\\hline
"""

    # Add rows for each dataset
    datasets_order = [
        "rossmann_subsampled", "walmart_subsampled",
        "airbnb_simplified_subsampled", "Berka_subsampled", "f1_subsampled"
    ]

    for dataset in datasets_order:
        if dataset in dataset_params:
            dataset_display = dataset_rename.get(dataset,
                                                 dataset.replace("_", "\\_"))

            row_values = [dataset_display]

            for param in params_to_show:
                if param in dataset_params[dataset]:
                    value = dataset_params[dataset][param]
                    # Format the value appropriately
                    if isinstance(value, float):
                        if value < 0.001:
                            formatted_value = f"{value:.2e}"
                        else:
                            formatted_value = f"{value:.3f}".rstrip(
                                '0').rstrip('.')
                    else:
                        formatted_value = str(value)
                    row_values.append(formatted_value)
                else:
                    row_values.append("-")

            latex_table += " & ".join(row_values) + " \\\\\n"

    latex_table += r"""\hline
\end{tabular}
\end{table}
"""

    return latex_table


def generate_latex_table():
    """Generate all GNN hyperparameter tables"""
    hyperparams = load_hyperparameters()

    all_tables = []
    gnn_architectures = [
        "hetero_gat", "hetero_gatv2", "hetero_gin", "hetero_graphconv",
        "hetero_graphsage", "relgnn"
    ]

    for gnn_arch in gnn_architectures:
        if gnn_arch in hyperparams and hyperparams[gnn_arch]:
            table = generate_gnn_hyperparameter_table(gnn_arch, hyperparams)
            all_tables.append(table)

    return "\n\n".join(all_tables)


def generate_detailed_latex_table():
    """Generate a detailed combined table showing hyperparameters for all GNN architectures"""
    hyperparams = load_hyperparameters()

    # Dataset renaming for cleaner display
    dataset_rename = {
        "f1_subsampled": "F1",
        "Berka_subsampled": "Berka",
        "rossmann_subsampled": "Rossmann",
        "walmart_subsampled": "Walmart",
        "airbnb_simplified_subsampled": "Airbnb",
    }

    # GNN architecture display names
    gnn_display_names = {
        "hetero_gat": "Hetero-GAT",
        "hetero_gatv2": "Hetero-GATv2",
        "hetero_gin": "Hetero-GIN",
        "hetero_graphconv": "Hetero-GraphConv",
        "hetero_graphsage": "Hetero-GraphSAGE",
        "relgnn": "RelGNN"
    }

    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Detailed GNN Hyperparameters Summary for RDL Utility Evaluation}
\label{tab:gnn_hyperparams_detailed}
\small
\begin{tabular}{llcccccc}
\hline
\textbf{GNN} & \textbf{Dataset} & \textbf{LR} & \textbf{Layers} & \textbf{Neighbors} & \textbf{Weight Decay} & \textbf{MLP Layers} & \textbf{Aggr} \\
\hline
"""

    gnn_architectures = [
        "hetero_gat", "hetero_gatv2", "hetero_gin", "hetero_graphconv",
        "hetero_graphsage", "relgnn"
    ]
    datasets_order = [
        "rossmann_subsampled", "walmart_subsampled",
        "airbnb_simplified_subsampled", "Berka_subsampled", "f1_subsampled"
    ]

    for gnn_arch in gnn_architectures:
        if gnn_arch in hyperparams and hyperparams[gnn_arch]:
            gnn_display = gnn_display_names.get(gnn_arch, gnn_arch)

            for i, dataset in enumerate(datasets_order):
                if dataset in hyperparams[gnn_arch]:
                    dataset_display = dataset_rename.get(
                        dataset, dataset.replace("_", "\\_"))
                    params = hyperparams[gnn_arch][dataset]

                    # Show GNN name only for first dataset
                    gnn_cell = gnn_display if i == 0 else ""

                    # Format parameters
                    lr = params.get("lr", "-")
                    num_layers = params.get("num_layers", "-")
                    num_neighbors = params.get("num_neighbors", "-")
                    weight_decay = params.get("weight_decay", "-")
                    mlp_layers = params.get("mlp_layers", "-")
                    aggr = params.get("aggr", "-")

                    # Format weight decay in scientific notation if very small
                    if isinstance(weight_decay,
                                  float) and weight_decay < 0.001:
                        weight_decay = f"{weight_decay:.1e}"
                    elif isinstance(weight_decay, float):
                        weight_decay = f"{weight_decay:.5f}".rstrip(
                            '0').rstrip('.')

                    latex_table += f"{gnn_cell} & {dataset_display} & {lr} & {num_layers} & {num_neighbors} & {weight_decay} & {mlp_layers} & {aggr} \\\\\n"

            # Add separator line between GNN architectures
            if gnn_arch != "relgnn":  # Don't add after last GNN
                latex_table += "\\hline\n"

    latex_table += r"""\hline
\end{tabular}
\end{table}

% Note: Hyperparameters obtained from hyperparameter tuning with 100 trials each
"""

    return latex_table


if __name__ == "__main__":
    print("=== Individual GNN Hyperparameter Tables ===")
    print(generate_latex_table())

    print("\n\n=== Combined Detailed Hyperparameter Table ===")
    print(generate_detailed_latex_table())
