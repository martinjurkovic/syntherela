import os
import json

import numpy as np

import matplotlib.pyplot as plt

USE_DISTANCES = False

datasets = [
    'airbnb-simplified_subsampled',
    'rossmann_subsampled',
    'walmart_subsampled',
    "Berka_subsampled",
    "f1_subsampled",
    'imdb_MovieLens_v1',
    'Biodegradability_v1',
    'CORA_v1',
]

dataset_names = {
    "airbnb-simplified_subsampled": "Airbnb",
    "Berka_subsampled": "Berka",
    "Biodegradability_v1": "Biodegradability",
    "CORA_v1": "Cora",
    "imdb_MovieLens_v1": "IMDB",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "f1_subsampled": "F1",
}

methods = [
    'MOSTLYAI',
    'RGCLD',
    'CLAVADDPM',
    'RCTGAN',
    'REALTABFORMER',
    'SDV',
]

names = {
    'CLAVADDPM': "CLAVADDPM\n2024",
    'RGCLD': "RGCLD\n2024",
    'MOSTLYAI': "TabularARGN\n2025",
    'RCTGAN': "RCTGAN\n2023",
    'REALTABFORMER': "REALTABFORMER\n2023",
    'SDV': "SDV\n2016",
}

runs = [1, 2, 3]
results = {}
multi_table_results = {}
single_table_results = {}
single_column_results = {}

for run in runs:
    results[run] = {}
    run_results = results[run]
    for dataset in datasets:
        run_results[dataset] = {}
        for method in methods:
            try:
                with open(
                        f'results/{run}/{dataset}_{method}_{run}_sample1.json'
                ) as f:
                    run_results[dataset][method] = json.load(f)
            except FileNotFoundError:
                print(f"Missing {dataset} {method}")

    multi_table_results[run] = {}
    single_table_results[run] = {}
    single_column_results[run] = {}

    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        multi_table_results[run].setdefault(dataset_name, {})
        single_table_results[run].setdefault(dataset_name, {})
        single_column_results[run].setdefault(dataset_name, {})
        multi_run = multi_table_results[run]
        single_run = single_table_results[run]
        column_run = single_column_results[run]
        for method in methods:
            multi_run[dataset_name].setdefault(method, [])
            single_run[dataset_name].setdefault(method, [])
            column_run[dataset_name].setdefault(method, [])
            if method not in results[run][dataset]:
                continue
            for table, single_results in results[run][dataset][method][
                    'single_table_metrics'][
                        'SingleTableDetection-XGBClassifier'].items():
                multi_results = results[run][dataset][method][
                    'multi_table_metrics'][
                        'AggregationDetection-XGBClassifier']
                if table in multi_results:
                    multi_run[dataset_name][method].append(
                        multi_results[table]['accuracy'])
                single_run[dataset_name][method].append(
                    single_results['accuracy'])


def multi_comparison_chart(single_table_results,
                           multi_table_results,
                           models,
                           tabular_sota=None,
                           datasets=None,
                           runs=[1, 2, 3],
                           names=None,
                           bold_best=True,
                           width=0.1,
                           capsize=4,
                           scalex=1.25,
                           aspect=0.75,
                           save_path=None):
    if names is None:
        names = {model: model for model in models}

    # models = list(single_table_results['Airbnb'].keys())
    if datasets is None:
        datasets = list(single_table_results[runs[0]].keys())
    x = np.arange(len(models)) * width  # the label locations
    # the width of the bars

    xscale = len(models) * scalex
    yscale = xscale * aspect
    fig, ax = plt.subplots(figsize=(xscale, yscale))

    # Extract data and calculate means and standard deviations
    single_means, single_stds = [], []
    multi_means, multi_stds = [], []

    for model in models:
        single_data = []
        multi_data = []
        for run in runs:
            for dataset in datasets:
                if len(single_table_results[run][dataset][model]) == 0:
                    continue
                single_data += single_table_results[run][dataset][model]
                multi_data += multi_table_results[run][dataset][model]
        single_means.append(np.mean(single_data))
        single_stds.append(np.std(single_data) / np.sqrt(len(single_data)))
        multi_means.append(np.mean(multi_data))
        multi_stds.append(np.std(multi_data) / np.sqrt(len(single_data)))

    # Calculate delta values
    deltas = [
        multi_mean - single_mean
        for multi_mean, single_mean in zip(multi_means, single_means)
    ]

    # # Create bars
    ax.errorbar(x,
                multi_means,
                yerr=multi_stds,
                fmt='o',
                label='Avg. Multi Table',
                color='seagreen',
                capsize=capsize)
    ax.errorbar(x,
                single_means,
                yerr=single_stds,
                fmt='o',
                label='Avg. Single Table',
                color='slateblue',
                capsize=capsize)

    # Add dashed lines and delta values
    d = width / 10

    for i in range(len(models)):
        if deltas[i] == min(deltas) and bold_best:
            delta = deltas[i].round(2)  # Round to 2 decimal places
            if delta == 0:
                delta = deltas[i].round(3)  # Round to 3 decimal places
            ax.text(x[i] + d, (single_means[i] + multi_means[i]) / 2,
                    f' Δ = {delta}',
                    va='center',
                    color='gray',
                    fontweight='bold')
        else:
            ax.text(x[i] + d, (single_means[i] + multi_means[i]) / 2,
                    f' Δ = {deltas[i]:.2f}',
                    va='center',
                    color='gray')
        ax.plot([x[i] + d, x[i] + d], [single_means[i], multi_means[i]],
                '--',
                color='gray',
                zorder=3)
        ax.plot([x[i], x[i] + d], [single_means[i], single_means[i]],
                '--',
                color='gray',
                zorder=3)
        ax.plot([x[i], x[i] + d], [multi_means[i], multi_means[i]],
                '--',
                color='gray',
                zorder=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Detection Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([names[model] for model in models], fontsize=12)
    plt.xticks(rotation=0)  # Rotate x labels if needed

    # Add horizontal lines
    ax.plot([0, 0], [0.5, 0.5],
            '--',
            color='gray',
            zorder=0,
            label='Relational Δ')
    if tabular_sota is not None:
        mean, std, method = tabular_sota
        ax.axhline(y=mean, color='C4', linestyle='--',
                   alpha=0.8)  # Tabular SOTA line
        ax.fill_between([-d, len(models) * width + d],
                        mean - std,
                        mean + std,
                        color='C4',
                        alpha=0.1)
        ax.text(0,
                mean,
                f'Tabular SOTA - {method}',
                va='bottom',
                ha='left',
                color='C4',
                fontsize=12,
                alpha=0.8)
    ax.axhline(y=0.5, color='C3', linestyle='--',
               alpha=0.8)  # Perfect fidelity line
    ax.text(0,
            0.5,
            'Perfect Fidelity',
            va='bottom',
            ha='left',
            color='C3',
            fontsize=12,
            alpha=0.8)

    ax.legend(fontsize=12)
    ax.set_yticks(np.arange(0.5, 1.001, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='lightgrey')
    ax.set_xlim(-d, (len(models) - 0.5) * width)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400)

    plt.show()


databases = [dataset_names[dataset] for dataset in datasets]
models = ['SDV', 'REALTABFORMER', 'RCTGAN', 'CLAVADDPM', 'RGCLD', 'MOSTLYAI']

multi_comparison_chart(single_table_results,
                       multi_table_results,
                       models=models,
                       datasets=databases,
                       bold_best=False,
                       width=0.25,
                       aspect=0.6,
                       names=names,
                       scalex=1.4,
                       save_path='results/figures/figure2.png')
