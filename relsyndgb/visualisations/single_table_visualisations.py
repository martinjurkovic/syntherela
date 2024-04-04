import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
sns.set_theme(font_scale=1.5)

from relsyndgb.visualisations.utils import get_x_tick_width_coef

def visualize_single_table_distance_metrics(all_results, datasets, methods, **kwargs):
    for dataset in datasets:
        base_metrics = list(all_results[dataset][methods[0]]['single_table_metrics'].keys())
        # remove all metrics that are detection
        base_metrics = [metric for metric in base_metrics if "detection" not in metric.lower()]
        base_metric_names = base_metrics

        save_figs = kwargs.get("save_figs", False)
        save_figs_path = kwargs.get("save_figs_path", "./figs")
        save_figs_path = Path(save_figs_path) / "single_table" / "distance"

        method_order = kwargs.get("method_order", ['SDV', 'RCTGAN', 'MOSTLYAI', 'REALTABFORMER'])
        if method_order is not None:
            methods = [method for method in method_order if method in methods]
            methods += sorted([method for method in methods if method not in method_order])

        for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
            metrics = [
                base_metric
                ]

            metric_names = [
                base_metric_name
                ]
            if len(methods) == 0 or len(metrics) == 0:
                continue
            dataset_table_results = {}

            tables = all_results[dataset][methods[0]]['single_table_metrics'][base_metric].keys()
                
            N = len(methods) # number of methods
            M = len(tables) # number of tables
            
            ind = np.arange(M)
            width = 0.15

            fig, ax = plt.subplots(figsize=(10,7))
            # set dpi
            fig.dpi = 300
            # make font size bigger
            # plt.rcParams.update({'font.size': 20})

            colors = plt.cm.viridis(np.linspace(0.5, 1, N)) # create a color map

            for j, method in enumerate(methods):
                method_means = [all_results[dataset][method]['single_table_metrics'][base_metric][table]["value"] for table in tables]
                method_ses = [all_results[dataset][method]['single_table_metrics'][base_metric][table]["bootstrap_se"] for table in tables]
                ax.bar(ind + width*j, method_means, width, yerr=method_ses, color=colors[j])
            
            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef*width)
            rotation = 20 if len(tables) > 6 else 0
            ax.set_xticklabels(tables, fontsize = 10, rotation=rotation)
            # y_min = 0

            # max_value = max([all_results[dataset][method]['single_table_metrics'][base_metric][table][column]["value"] for column in tables])
            # y_max = max_value * 1.2
            ax.set_ylim(0)
            # ax.set_yticks(np.arange(y_min, 1.01, 0.1))
            ax.set_ylabel("Metric Value")

            # Create a legend
            
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(custom_lines, methods, loc='upper center', ncol=N, fontsize=11)

            for j, table in enumerate(tables):
                ci95 = all_results[dataset][method]['single_table_metrics'][base_metric][table]["reference_ci"]
                # ci95 = np.mean(ci95, axis=0)
                ax.axhline(y=ci95[1], color='black', linestyle='--', linewidth=1, 
                        xmin=j/len(tables), 
                        xmax=(j+1)/len(tables))
                ax.axhline(y=ci95[0], color='black', linestyle='--', linewidth=1, 
                        xmin=j/len(tables), 
                        xmax=(j+1)/len(tables))
            
                if ci95[1] > ax.get_ylim()[1]:
                    y_max = ci95[1] * 1.1
                    ax.set_ylim(0, y_max)


            # set title
            plt.title(f"{base_metric_name} for dataset {dataset}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(f"{save_figs_path}/{dataset}_{base_metric}.png", dpi=300)

def visualize_single_table_detection_metrics_per_classifier(all_results, datasets, methods, **kwargs):
    for dataset in datasets:
        base_metrics = list(all_results[dataset][methods[0]]['single_table_metrics'].keys())
        # remove all metrics that are detection
        base_metrics = [metric for metric in base_metrics if "detection" in metric.lower()]
        base_metric_names = base_metrics

        save_figs = kwargs.get("save_figs", False)
        save_figs_path = kwargs.get("save_figs_path", "./figs")
        save_figs_path = Path(save_figs_path) / "single_table" / "detection" 

        method_order = kwargs.get("method_order", ['SDV', 'RCTGAN', 'MOSTLYAI', 'REALTABFORMER'])
        if method_order is not None:
            methods = [method for method in method_order if method in methods]
            methods += sorted([method for method in methods if method not in method_order])

        for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
            metrics = [
                base_metric
                ]

            metric_names = [
                base_metric_name
                ]
            if len(methods) == 0 or len(metrics) == 0:
                continue
            dataset_table_results = {}

            tables = all_results[dataset][methods[0]]['single_table_metrics'][base_metric].keys()
                
            N = len(methods) # number of methods
            M = len(tables) # number of tables
            
            ind = np.arange(M)
            width = 0.15

            fig, ax = plt.subplots(figsize=(10,7))
            # set dpi
            fig.dpi = 300
            # make font size bigger
            # plt.rcParams.update({'font.size': 20})

            colors = plt.cm.viridis(np.linspace(0.5, 1, N)) # create a color map

            min_mean = 1
            for j, method in enumerate(methods):
                method_means = [all_results[dataset][method]['single_table_metrics'][base_metric][table]["accuracy"] for table in tables]
                min_mean = min(min_mean, min(method_means))
                method_ses = [all_results[dataset][method]['single_table_metrics'][base_metric][table]["SE"] for table in tables]
                baseline_means = np.array([all_results[dataset][method]['single_table_metrics'][base_metric][table]["baseline_mean"] for table in tables])
                baseline_ses = np.array([all_results[dataset][method]['single_table_metrics'][base_metric][table]["baseline_se"] for table in tables])
                ax.bar(ind + width*j, method_means, width, yerr=method_ses, color=colors[j])
                # draw a horizontal line for the baseline and standard error
                ax.hlines(baseline_means, ind + width*j - width/2, ind + width*j + width/2, color='k')#, linestyle='--')
                ax.hlines(baseline_means + baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
                ax.hlines(baseline_means - baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
            
            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef*width)
            rotation = 20 if len(tables) > 6 else 0
            ax.set_xticklabels(tables, fontsize = 10, rotation=rotation)

            y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1)*10)/10
            ax.set_ylim(y_min, 1.1)
            ax.set_ylabel("Metric Value")

            # Create a legend
            
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(custom_lines, methods, loc='upper center', ncol=N, fontsize=11)

            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)

            # set title
            plt.title(f"{base_metric_name} for dataset {dataset}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(f"{save_figs_path}/{dataset}_{base_metric}.png", dpi=300)



def visualize_single_table_detection_metrics_per_table(all_results, datasets, methods, **kwargs):
    for dataset in datasets:
        metrics = kwargs.get('detection_metrics', 
                            [metric for metric in list(all_results[dataset][methods[0]]['single_table_metrics'].keys()) if 'detection' in metric.lower()])
        metric_names = kwargs.get('detection_metric_names', metrics)

        aggregation_metrics = kwargs.get(
            'aggregation_metrics', 
            [metric for metric in list(all_results[datasets[0]][methods[0]]['multi_table_metrics'].keys()) if 'SingleTableAggregationDetection' in metric])


        # aggregation_metrics = [
        #     # "SingleTableAggregationDetection-LogisticRegression", 
        #     "SingleTableAggregationDetection-XGBClassifier",
        #     ]
        aggregation_metric_names = aggregation_metrics

        save_figs = kwargs.get("save_figs", False)
        save_figs_path = kwargs.get("save_figs_path", "./figs")
        save_figs_path = Path(save_figs_path) / "single_table" / "detection"

        method_order = kwargs.get("method_order", ['SDV', 'RCTGAN', 'MOSTLYAI', 'REALTABFORMER'])
        if method_order is not None:
            methods = [method for method in method_order if method in methods]
            methods += sorted([method for method in methods if method not in method_order])

    
        if len(methods) == 0 or len(metrics) == 0:
            continue
        tables = all_results[dataset][methods[0]]['single_table_metrics'][metrics[0]].keys()
        for table in tables:
            agg_metrics = []
            if aggregation_metrics and table in all_results[dataset][methods[0]]['multi_table_metrics'][aggregation_metrics[0]]:
                agg_metrics = aggregation_metrics
            
            N = len(metrics + agg_metrics) # number of metrics
            M = len(methods) # number of methods
            ind = np.arange(M) # the x locations for the groups
            width = 0.15       # the width of the bars

            fig, ax = plt.subplots(figsize=(10,7))
            # set dpi
            fig.dpi = 300
            # make font size bigger
            # plt.rcParams.update({'font.size': 20})

            colors = plt.cm.viridis(np.linspace(0, 1, N)) # create a color map

            min_mean = 1
            for j, metric in enumerate(metrics):
                metric_means = [all_results[dataset][method]['single_table_metrics'][metric][table]['accuracy'] for method in methods]
                min_mean = min(min_mean, min(metric_means))
                metric_ses = [all_results[dataset][method]['single_table_metrics'][metric][table]['SE'] for method in methods]
                baseline_means = np.array([all_results[dataset][method]['single_table_metrics'][metric][table]["baseline_mean"] for method in methods])
                baseline_ses = np.array([all_results[dataset][method]['single_table_metrics'][metric][table]["baseline_se"] for method in methods])

                ax.bar(ind + width*j, metric_means, width, yerr=metric_ses, color=colors[j])
                ax.hlines(baseline_means, ind + width*j - width/2, ind + width*j + width/2, color='k')#, linestyle='--')
                ax.hlines(baseline_means + baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
                ax.hlines(baseline_means - baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')

            for j, agg_metric in enumerate(agg_metrics):
                metric_means = [all_results[dataset][method]['multi_table_metrics'][agg_metric][table]['accuracy'] for method in methods]
                min_mean = min(min_mean, min(metric_means))
                metric_ses = [all_results[dataset][method]['multi_table_metrics'][agg_metric][table]['SE'] for method in methods]
                baseline_means = np.array([all_results[dataset][method]['multi_table_metrics'][agg_metric][table]["baseline_mean"] for method in methods])
                baseline_ses = np.array([all_results[dataset][method]['multi_table_metrics'][agg_metric][table]["baseline_se"] for method in methods])

                ax.bar(ind + width*(j+len(metrics)), metric_means, width, yerr=metric_ses, color=colors[j+len(metrics)])
                ax.hlines(baseline_means, ind + width*(j+len(metrics)) - width/2, ind + width*(j+len(metrics)) + width/2, color='k')
                ax.hlines(baseline_means + baseline_ses, ind + width*(j+len(metrics)) - width/2, ind + width*(j+len(metrics)) + width/2, color='k', linestyle='--')
                ax.hlines(baseline_means - baseline_ses, ind + width*(j+len(metrics)) - width/2, ind + width*(j+len(metrics)) + width/2, color='k', linestyle='--')

            ax.set_ylabel('Means')
            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef*width)
            rotation = 20 if len(methods) > 6 else 0
            ax.set_xticklabels(methods, fontsize = 10, rotation=rotation)

            y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1)*10)/10
            ax.set_ylim(y_min, 1.3)
            ax.set_yticks(np.arange(y_min, 1.01, 0.1))
            ax.set_ylabel("Classification Accuracy")

            # Create a legend
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(custom_lines, metric_names + agg_metrics, loc='upper left') # move the legend

            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)

            # set title
            plt.title(f"Dataset {dataset}, table {table}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(f"{save_figs_path}/{dataset}_{table}_per_table.png", dpi=300)

                

                
