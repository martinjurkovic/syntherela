import os
import math
from pathlib import Path

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

from syntherela.visualisations.utils import (
    get_x_tick_width_coef,
    prettify_dataset_name,
    prettify_method_name,
    get_color,
    get_dataset_info,
)


def visualize_single_table_distance_metrics(
    granularity_level,
    metric_type,
    all_results,
    datasets,
    methods,
    title=True,
    log_scale=False,
    fontsize=35,
    **kwargs,
):
    for dataset in datasets:
        base_metrics, base_metric_names, save_figs, save_figs_path, methods = (
            get_dataset_info(
                granularity_level, metric_type, all_results, dataset, methods, **kwargs
            )
        )

        for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
            if len(methods) == 0:
                continue

            tables = all_results[dataset][list(all_results[dataset].keys())[0]][
                "single_table_metrics"
            ][base_metric].keys()

            N = len(methods)  # number of methods
            M = len(tables)  # number of tables

            ind = np.arange(M)
            width = 0.12

            fig, ax = plt.subplots(figsize=(10, 7))
            # set dpi
            fig.dpi = 300

            if log_scale:
                ax.set_yscale("log")

            colors = [get_color(method) for method in methods]
            if None in colors:
                colors = plt.cm.viridis(np.linspace(0.5, 1, N))

            max_value = 0

            for j, method in enumerate(methods):
                method_means = [
                    all_results[dataset][method]["single_table_metrics"][base_metric][
                        table
                    ]["value"]
                    for table in tables
                ]
                method_ses = [
                    all_results[dataset][method]["single_table_metrics"][base_metric][
                        table
                    ]["bootstrap_se"]
                    for table in tables
                ]
                ax.bar(
                    ind + width * j,
                    method_means,
                    width,
                    yerr=method_ses,
                    color=colors[j],
                    alpha=0.8,
                    edgecolor="white",
                )
                ci95s = [
                    all_results[dataset][method]["single_table_metrics"][base_metric][
                        table
                    ]["reference_ci"][1]
                    for table in tables
                ]

                max_value = max(max_value, max(method_means), max(ci95s))

            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef * width)
            rotation = 20 if len(tables) > 6 else 0
            ax.set_xticklabels(tables, rotation=rotation, fontsize=fontsize)
            # y_min = 0

            if not log_scale:
                ax.set_ylim(bottom=0, top=max_value * 1.25)
            else:
                p = kwargs.get(f"{base_metric}_pow", 20)
                top = math.pow(max_value, p)
                ax.set_ylim(top=top)
            # ax.set_yticks(np.arange(y_min, 1.01, 0.1))
            ax.set_ylabel("Metric Value", fontsize=fontsize)
            ax.tick_params(axis="y", labelsize=20)

            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(
                custom_lines,
                [prettify_method_name(method) for method in methods],
                loc="upper center",
                ncol=3,
                fontsize=18,
            )

            for j, table in enumerate(tables):
                ci95 = all_results[dataset][method]["single_table_metrics"][
                    base_metric
                ][table]["reference_ci"]
                # ci95 = np.mean(ci95, axis=0)
                ax.axhline(
                    y=ci95[1],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    xmin=j / len(tables),
                    xmax=(j + 1) / len(tables),
                )
                ax.axhline(
                    y=ci95[0],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    xmin=j / len(tables),
                    xmax=(j + 1) / len(tables),
                )

                if ci95[1] > ax.get_ylim()[1]:
                    y_max = ci95[1] * 1.1
                    ax.set_ylim(0, y_max)

            # set title
            if title:
                plt.title(
                    f"{base_metric_name} for dataset {prettify_dataset_name(dataset)}"
                )

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(f"{save_figs_path}/{dataset}_{base_metric}.png", dpi=300)


def visualize_single_table_detection_metrics_per_classifier(
    granularity_level, metric_type, all_results, datasets, methods, **kwargs
):
    for dataset in datasets:
        base_metrics, base_metric_names, save_figs, save_figs_path, methods = (
            get_dataset_info(
                granularity_level, metric_type, all_results, dataset, methods, **kwargs
            )
        )

        for base_metric, base_metric_name in zip(base_metrics, base_metric_names):
            if len(methods) == 0:
                continue

            tables = all_results[dataset][methods[0]]["single_table_metrics"][
                base_metric
            ].keys()

            N = len(methods)  # number of methods
            M = len(tables)  # number of tables

            ind = np.arange(M)
            width = 0.15

            fig, ax = plt.subplots(figsize=(10, 7))
            # set dpi
            fig.dpi = 300

            colors = plt.cm.viridis(np.linspace(0.5, 1, N))  # create a color map

            min_mean = 1
            for j, method in enumerate(methods):
                method_means = [
                    all_results[dataset][method]["single_table_metrics"][base_metric][
                        table
                    ]["accuracy"]
                    for table in tables
                ]
                min_mean = min(min_mean, min(method_means))
                method_ses = [
                    all_results[dataset][method]["single_table_metrics"][base_metric][
                        table
                    ]["SE"]
                    for table in tables
                ]
                baseline_means = np.array(
                    [
                        all_results[dataset][method]["single_table_metrics"][
                            base_metric
                        ][table]["baseline_mean"]
                        for table in tables
                    ]
                )
                baseline_ses = np.array(
                    [
                        all_results[dataset][method]["single_table_metrics"][
                            base_metric
                        ][table]["baseline_se"]
                        for table in tables
                    ]
                )
                ax.bar(
                    ind + width * j,
                    method_means,
                    width,
                    yerr=method_ses,
                    color=colors[j],
                )
                # draw a horizontal line for the baseline and standard error
                ax.hlines(
                    baseline_means,
                    ind + width * j - width / 2,
                    ind + width * j + width / 2,
                    color="k",
                )  # , linestyle='--')
                ax.hlines(
                    baseline_means + baseline_ses,
                    ind + width * j - width / 2,
                    ind + width * j + width / 2,
                    color="k",
                    linestyle="--",
                )
                ax.hlines(
                    baseline_means - baseline_ses,
                    ind + width * j - width / 2,
                    ind + width * j + width / 2,
                    color="k",
                    linestyle="--",
                )

            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef * width)
            rotation = 20 if len(tables) > 6 else 0
            ax.set_xticklabels(tables, fontsize=10, rotation=rotation)

            y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1) * 10) / 10
            ax.set_ylim(y_min, 1.1)
            ax.set_ylabel("Metric Value")

            # Create a legend

            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(custom_lines, methods, loc="upper center", ncol=N, fontsize=11)

            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)

            # set title
            plt.title(f"{base_metric_name} for dataset {dataset}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(f"{save_figs_path}/{dataset}_{base_metric}.png", dpi=300)


def visualize_single_table_detection_metrics_per_table(
    all_results, datasets, methods, title=True, log_scale=False, fontsize=20, **kwargs
):
    for dataset in datasets:
        metrics = kwargs.get(
            "detection_metrics",
            [
                metric
                for metric in list(
                    all_results[dataset][list(all_results[dataset].keys())[0]][
                        "single_table_metrics"
                    ].keys()
                )
                if "detection" in metric.lower()
            ],
        )
        metric_names = kwargs.get("detection_metric_names", metrics)

        aggregation_metrics = kwargs.get(
            "aggregation_metrics",
            [
                metric
                for metric in list(
                    all_results[dataset][list(all_results[dataset].keys())[0]][
                        "multi_table_metrics"
                    ].keys()
                )
                if "AggregationDetection" in metric and "parent" not in metric.lower()
            ],
        )

        save_figs = kwargs.get("save_figs", False)
        save_figs_path = kwargs.get("save_figs_path", "./figs")
        save_figs_path = Path(save_figs_path) / "single_table" / "detection"

        method_order = kwargs.get(
            "method_order",
            [
                "SDV",
                "RCTGAN",
                "REALTABFORMER",
                "MOSTLYAI",
                "GRETEL_ACTGAN",
                "GRETEL_LSTM",
                "ClavaDDPM",
            ],
        )

        if method_order is not None:
            methods = [
                method
                for method in method_order
                if method in methods and method in all_results[dataset]
            ]
            methods += sorted(
                [method for method in methods if method not in method_order]
            )

        if len(methods) == 0 or len(metrics) == 0:
            continue
        method_ = list(all_results[dataset].keys())[0]
        tables = all_results[dataset][method_]["single_table_metrics"][
            list(all_results[dataset][method_]["single_table_metrics"].keys())[0]
        ].keys()
        for table in tables:
            agg_metrics = []
            if (
                aggregation_metrics
                and table
                in all_results[dataset][methods[-1]]["multi_table_metrics"][
                    aggregation_metrics[-1]
                ]
            ):
                agg_metrics = aggregation_metrics

            N = len(metrics + agg_metrics)  # number of metrics
            M = len(methods)  # number of methods
            ind = np.arange(M)  # the x locations for the groups
            width = 0.15  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 7))
            # set dpi
            fig.dpi = 300
            # make font size bigger
            # plt.rcParams.update({'font.size': 20})

            colors = plt.cm.tab20(np.linspace(0, 1, N))  # create a color map

            tab20 = plt.colormaps["tab20"]
            colors = [tab20(1), tab20(3), tab20(0), tab20(2)]

            min_mean = 1
            for j, metric in enumerate(metrics):
                metric_means = [
                    all_results[dataset][method]["single_table_metrics"][metric][table][
                        "accuracy"
                    ]
                    for method in methods
                ]
                min_mean = min(min_mean, min(metric_means))
                metric_ses = [
                    all_results[dataset][method]["single_table_metrics"][metric][table][
                        "SE"
                    ]
                    for method in methods
                ]
                # baseline_means = np.array([all_results[dataset][method]['single_table_metrics'][metric][table]["baseline_mean"] for method in methods])
                baseline_means = np.array([0.5 for method in methods])
                # baseline_ses = np.array([all_results[dataset][method]['single_table_metrics'][metric][table]["baseline_se"] for method in methods])
                baseline_ses = np.array([0 for method in methods])

                ax.bar(
                    ind + width * j,
                    metric_means,
                    width,
                    yerr=metric_ses,
                    color=colors[j],
                )
                # ax.hlines(baseline_means, ind + width*j - width/2, ind + width*j + width/2, color='k')#, linestyle='--')
                # ax.hlines(baseline_means + baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')
                # ax.hlines(baseline_means - baseline_ses, ind + width*j - width/2, ind + width*j + width/2, color='k', linestyle='--')

            for j, agg_metric in enumerate(agg_metrics):
                metric_means = [
                    all_results[dataset][method]["multi_table_metrics"][agg_metric][
                        table
                    ]["accuracy"]
                    for method in methods
                ]
                min_mean = min(min_mean, min(metric_means))
                metric_ses = [
                    all_results[dataset][method]["multi_table_metrics"][agg_metric][
                        table
                    ]["SE"]
                    for method in methods
                ]
                # baseline_means = np.array([all_results[dataset][method]['multi_table_metrics'][agg_metric][table]["baseline_mean"] for method in methods])
                baseline_means = np.array([0.5 for method in methods])
                # baseline_ses = np.array([all_results[dataset][method]['multi_table_metrics'][agg_metric][table]["baseline_se"] for method in methods])
                baseline_ses = np.array([0 for method in methods])

                ax.bar(
                    ind + width * (j + len(metrics)),
                    metric_means,
                    width,
                    yerr=metric_ses,
                    color=colors[j + len(metrics)],
                )

            ax.set_ylabel("Means")
            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef * width)
            pretty_methods = [prettify_method_name(method) for method in methods]
            ax.set_xticklabels(pretty_methods, fontsize=9.4)

            # y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1)*10)/10
            y_min = 0.3
            ax.set_ylim(y_min, 1.4)
            ax.set_yticks(np.arange(y_min, 1.01, 0.1))
            ax.set_ylabel("Classification Accuracy")

            # Create a legend
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(
                custom_lines, metric_names + agg_metrics, loc="upper left"
            )  # move the legend

            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)

            # make figsize smaller
            fig.set_size_inches(6.5, 4)

            # set title
            if title:
                plt.title(f"Dataset {prettify_dataset_name(dataset)}, table {table}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(
                    f"{save_figs_path}/{dataset}_{table}_per_table.png", dpi=300
                )
