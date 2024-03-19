import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
sns.set_theme(font_scale=1.5)

from relsyndgb.visualisations.utils import get_x_tick_width_coef

def visualize_single_column_distance_metrics(all_results, datasets, methods, **kwargs):
    base_metrics = list(all_results[datasets[0]][methods[0]]['single_column_metrics'].keys())
    # remove all metrics that are detection
    base_metrics = [metric for metric in base_metrics if "detection" not in metric.lower()]
    base_metrics = [metric for metric in base_metrics if "test" not in metric.lower()]
    base_metric_names = base_metrics

    for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
        metrics = [
            base_metric
            ]

        metric_names = [
            base_metric_name
            ]
        for dataset in datasets:
            dataset_table_results = {}
            for table in all_results[dataset][methods[0]]['single_column_metrics'][base_metric].keys():
                try:
                
                    N = len(methods) # number of methods
                    M = len(all_results[dataset][methods[0]]['single_column_metrics'][base_metric][table].keys()) # number of columns
                    
                    ind = np.arange(M)
                    width = 0.15

                    fig, ax = plt.subplots(figsize=(10,7))
                    # set dpi
                    fig.dpi = 300
                    # make font size bigger
                    # plt.rcParams.update({'font.size': 20})

                    colors = plt.cm.viridis(np.linspace(0.5, 1, N)) # create a color map


                    columns = all_results[dataset][methods[0]]['single_column_metrics'][base_metric][table].keys()
                    for j, method in enumerate(methods):
                        method_means = [all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["value"] for column in columns]
                        method_ses = [all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["bootstrap_se"] for column in columns]
                        ax.bar(ind + width*j, method_means, width, yerr=method_ses, color=colors[j])

                    def get_x_tick_width_coef(N):
                        if N == 5:
                            return 2
                        elif N == 4:
                            return 1.5
                        elif N == 3:
                            return 1
                        elif N == 2:
                            return 0.5
                        else:
                            return 0
                    
                    x_tick_width_coef = get_x_tick_width_coef(N)
                    ax.set_xticks(ind + x_tick_width_coef*width)
                    rotation = 20 if len(columns) > 6 else 0
                    ax.set_xticklabels(columns, fontsize = 10, rotation=rotation)
                    # y_min = 0

                    # max_value = max([all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["value"] for column in columns])
                    # y_max = max_value * 1.2
                    ax.set_ylim(0)
                    # ax.set_yticks(np.arange(y_min, 1.01, 0.1))
                    ax.set_ylabel("Metric Value")

                    # Create a legend
                    from matplotlib.lines import Line2D
                    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
                    ax.legend(custom_lines, methods, loc='upper center', ncol=N, fontsize=11)

                    for j, column in enumerate(columns):
                        ci95 = all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["reference_ci"]
                        # ci95 = np.mean(ci95, axis=0)
                        ax.axhline(y=ci95[1], color='black', linestyle='--', linewidth=1, 
                                xmin=j/len(columns), 
                                xmax=(j+1)/len(columns))
                        # ax.axhline(y=ci95[0], color='black', linestyle='--', linewidth=1, 
                        #         xmin=j/len(columns), 
                        #         xmax=(j+1)/len(columns))
                    
                        if ci95[1] > ax.get_ylim()[1]:
                            y_max = ci95[1] * 1.1
                            ax.set_ylim(0, y_max)


                    # set title
                    plt.title(f"{base_metric_name} for dataset {dataset}, table {table}")

                    # directory_path = f"./figs/single_column/{base_metric}"

                    # os.makedirs(directory_path, exist_ok=True)

                    # plt.savefig(f"{directory_path}/{dataset}_{table}.png", dpi=300)
                except Exception as e:
                    print(f"{base_metric_name} for dataset {dataset}, table {table}")
                    print(e)
                    pass

def visualize_single_column_detection_metrics(all_results, datasets, methods, **kwargs):
    base_metrics = list(all_results[datasets[0]][methods[0]]['single_column_metrics'].keys())
    # keep only detection metrics
    base_metrics = [metric for metric in base_metrics if "detection" in metric.lower()]
    base_metric_names = base_metrics

    for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
        metrics = [
            base_metric
            ]

        metric_names = [
            base_metric_name
            ]
        for dataset in datasets:
            dataset_table_results = {}
            for table in all_results[dataset][methods[0]]['single_column_metrics'][base_metric].keys():
                
                N = len(methods) # number of methods
                M = len(all_results[dataset][methods[0]]['single_column_metrics'][base_metric][table].keys()) # number of columns
                
                ind = np.arange(M)
                width = 0.15

                fig, ax = plt.subplots(figsize=(10,7))
                # set dpi
                fig.dpi = 300
                # make font size bigger
                # plt.rcParams.update({'font.size': 20})

                colors = plt.cm.viridis(np.linspace(0.5, 1, N)) # create a color map

                min_mean = 1
                columns = all_results[dataset][methods[0]]['single_column_metrics'][base_metric][table].keys()
                for j, method in enumerate(methods):
                    method_means = [all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["accuracy"] for column in columns]
                    min_mean = min(min_mean, min(method_means))
                    method_ses = [all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["SE"] for column in columns]
                    baseline_means = np.array([all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["baseline_mean"] for column in columns])
                    baseline_ses = np.array([all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["baseline_se"] for column in columns])
                    ax.bar(ind + width*j, method_means, width, yerr=method_ses, color=colors[j])
                    # draw a line for baseline accuracy
                    ax.hlines(baseline_means, ind + width*j - width/2, ind + width*j + width/2, color='k')
                    ax.hlines(baseline_means + 1.96*baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
                    ax.hlines(baseline_means - 1.96*baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
                
                x_tick_width_coef = get_x_tick_width_coef(N)
                ax.set_xticks(ind + x_tick_width_coef*width)
                rotation = 20 if len(columns) > 6 else 0
                ax.set_xticklabels(columns, fontsize = 10, rotation=rotation)
                # y_min = 0

                # max_value = max([all_results[dataset][method]['single_column_metrics'][base_metric][table][column]["value"] for column in columns])
                # y_max = max_value * 1.2
                y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1)*10)/10

                ax.set_ylim(y_min, 1.1)
                # ax.set_yticks(np.arange(y_min, 1.01, 0.1))
                ax.set_ylabel("Metric Value")

                # Create a legend
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
                ax.legend(custom_lines, methods, loc='upper center', ncol=N, fontsize=11)

                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)


                # set title
                plt.title(f"{base_metric_name} for dataset {dataset}, table {table}")

                # directory_path = f"./figs/single_column/{base_metric}"

                # os.makedirs(directory_path, exist_ok=True)

                # plt.savefig(f"{directory_path}/{dataset}_{table}.png", dpi=300)


                


                