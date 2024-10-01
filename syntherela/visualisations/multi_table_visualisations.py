import os
from pathlib import Path

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from syntherela.visualisations.utils import get_x_tick_width_coef

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)


def visualize_parent_child_multi_table(all_results, datasets, methods, **kwargs):
    for dataset in datasets:
        metrics = kwargs.get(
            "detection_metrics",
            [
                metric
                for metric in list(
                    all_results[dataset][methods[0]]["multi_table_metrics"].keys()
                )
                if "ParentChildDetection" in metric
            ],
        )
        metric_names = kwargs.get("detection_metric_names", metrics)

        aggregation_metrics = kwargs.get(
            "aggregation_metrics",
            [
                metric
                for metric in list(
                    all_results[datasets[0]][methods[0]]["multi_table_metrics"].keys()
                )
                if "ParentChildAggregationDetection" in metric
            ],
        )

        save_figs = kwargs.get("save_figs", False)
        save_figs_path = kwargs.get("save_figs_path", "./figs")
        save_figs_path = Path(save_figs_path) / "multi_table" / "detection"

        method_order = kwargs.get(
            "method_order", ["SDV", "RCTGAN", "MOSTLYAI", "REALTABFORMER"]
        )
        if method_order is not None:
            methods = [method for method in method_order if method in methods]
            methods += sorted(
                [method for method in methods if method not in method_order]
            )

        if len(methods) == 0 or len(metrics) == 0:
            continue
        tables = all_results[dataset][methods[0]]["multi_table_metrics"][
            metrics[0]
        ].keys()
        for table in tables:
            agg_metrics = []
            if (
                aggregation_metrics
                and table
                in all_results[dataset][methods[0]]["multi_table_metrics"][
                    aggregation_metrics[0]
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

            colors = plt.cm.viridis(np.linspace(0, 1, N))  # create a color map

            min_mean = 1
            for j, metric in enumerate(metrics):
                metric_means = [
                    all_results[dataset][method]["multi_table_metrics"][metric][table][
                        "accuracy"
                    ]
                    for method in methods
                ]
                min_mean = min(min_mean, min(metric_means))
                metric_ses = [
                    all_results[dataset][method]["multi_table_metrics"][metric][table][
                        "SE"
                    ]
                    for method in methods
                ]
                # baseline_means = np.array([all_results[dataset][method]['multi_table_metrics'][metric][table]["baseline_mean"] for method in methods])
                # baseline_ses = np.array([all_results[dataset][method]['multi_table_metrics'][metric][table]["baseline_se"] for method in methods])

                baseline_means = np.array([0.5 for method in methods])
                baseline_ses = np.array([0.00 for method in methods])

                ax.bar(
                    ind + width * j,
                    metric_means,
                    width,
                    yerr=metric_ses,
                    color=colors[j],
                )
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
                # baseline_ses = np.array([all_results[dataset][method]['multi_table_metrics'][agg_metric][table]["baseline_se"] for method in methods])

                baseline_means = np.array([0.5 for method in methods])
                baseline_ses = np.array([0.00 for method in methods])

                ax.bar(
                    ind + width * (j + len(metrics)),
                    metric_means,
                    width,
                    yerr=metric_ses,
                    color=colors[j + len(metrics)],
                )
                ax.hlines(
                    baseline_means,
                    ind + width * (j + len(metrics)) - width / 2,
                    ind + width * (j + len(metrics)) + width / 2,
                    color="k",
                )
                ax.hlines(
                    baseline_means + baseline_ses,
                    ind + width * (j + len(metrics)) - width / 2,
                    ind + width * (j + len(metrics)) + width / 2,
                    color="k",
                    linestyle="--",
                )
                ax.hlines(
                    baseline_means - baseline_ses,
                    ind + width * (j + len(metrics)) - width / 2,
                    ind + width * (j + len(metrics)) + width / 2,
                    color="k",
                    linestyle="--",
                )

            ax.set_ylabel("Means")
            x_tick_width_coef = get_x_tick_width_coef(N)
            ax.set_xticks(ind + x_tick_width_coef * width)
            rotation = 20 if len(methods) > 6 else 0
            ax.set_xticklabels(methods, fontsize=10, rotation=rotation)

            y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1) * 10) / 10
            ax.set_ylim(y_min, 1.3)
            ax.set_yticks(np.arange(y_min, 1.01, 0.1))
            ax.set_ylabel("Classification Accuracy")

            # Create a legend
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
            ax.legend(
                custom_lines, metric_names + agg_metrics, loc="upper left"
            )  # move the legend

            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)

            # set title
            plt.title(f"Dataset {dataset}, table {table}")

            if save_figs:
                os.makedirs(save_figs_path, exist_ok=True)
                plt.savefig(
                    f"{save_figs_path}/{dataset}_{table}_parent_child_detection.png",
                    dpi=300,
                )


def visualize_multi_table(all_results, datasets, methods, **kwargs):
    save_figs = kwargs.get("save_figs", False)
    save_figs_path = kwargs.get("save_figs_path", "./figs/")
    save_figs_path = Path(save_figs_path) / "multi_table" / "detection"

    methods_all = methods.copy()

    for dataset in datasets:
        methods = methods_all.copy()
        method_order = kwargs.get(
            "method_order",
            [
                "SDV",
                "RCTGAN",
                "REALTABFORMER",
                "MOSTLYAI",
                "GRETEL_ACTGAN",
                "GRETEL_LSTM",
            ],
        )
        metrics = kwargs.get(
            "detection_metrics",
            [
                metric
                for metric in list(
                    all_results[dataset][method_order[1]]["multi_table_metrics"].keys()
                )
                if "singletable" not in metric.lower()
                and "parent" not in metric.lower()
                and "detection" in metric.lower()
            ],
        )
        metric_names = kwargs.get("detection_metric_names", metrics)

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

        N = len(metrics)  # number of metrics
        M = len(methods)  # number of methods
        ind = np.arange(M)  # the x locations for the groups
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 7))
        # set dpi
        fig.dpi = 300

        colors = plt.cm.viridis(np.linspace(0, 1, N))  # create a color map

        min_mean = 1

        for j, metric in enumerate(metrics):
            metric_means = [
                all_results[dataset][method]["multi_table_metrics"][metric]["accuracy"]
                for method in methods
            ]
            min_mean = min(min_mean, min(metric_means))
            metric_ses = [
                all_results[dataset][method]["multi_table_metrics"][metric]["SE"]
                for method in methods
            ]

            baseline_means = np.array([0.5 for method in methods])
            baseline_ses = np.array([0.00 for method in methods])

            ax.bar(
                ind + width * j, metric_means, width, yerr=metric_ses, color=colors[j]
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

        ax.set_ylabel("Means")
        x_tick_width_coef = get_x_tick_width_coef(N)
        ax.set_xticks(ind + x_tick_width_coef * width)
        rotation = 20 if len(methods) > 6 else 0
        ax.set_xticklabels(methods, fontsize=10, rotation=rotation)

        y_min = 0.4 if min_mean > 0.4 else np.floor((min_mean - 0.1) * 10) / 10
        ax.set_ylim(y_min, 1.3)
        ax.set_yticks(np.arange(y_min, 1.01, 0.1))
        ax.set_ylabel("Classification Accuracy")

        # Create a legend
        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(N)]
        ax.legend(custom_lines, metric_names, loc="upper left")  # move the legend

        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)

        # set title
        plt.title(f"Dataset {dataset}")

        if save_figs:
            os.makedirs(save_figs_path, exist_ok=True)
            plt.savefig(save_figs_path / f"{dataset}_multi_table_detection.png")
