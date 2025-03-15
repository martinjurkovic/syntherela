"""Utility functions for visualizations."""

import numpy as np
from pathlib import Path

COLORMAP = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

COLOR_DICT = {
    "SDV": COLORMAP[0],
    "RCTGAN": COLORMAP[1],
    "REALTABFORMER": COLORMAP[2],
    "REALTABF.": COLORMAP[2],
    "MOSTLYAI": COLORMAP[8],
    "GRE-ACTGAN": COLORMAP[3],
    "GRETEL_ACTGAN": COLORMAP[3],
    "GRE-LSTM": COLORMAP[5],
    "GRETEL_LSTM": COLORMAP[5],
    "REAL": COLORMAP[6],
    "CLAVADDPM": COLORMAP[4],
}


def get_color(method_name):
    """Get the color for a specific method.

    Parameters
    ----------
    method_name : str
        The name of the method to get the color for.

    Returns
    -------
    str or None
        The hex color code for the method, or None if not found.

    """
    if method_name in COLOR_DICT:
        return COLOR_DICT[method_name]
    return None


def get_x_tick_width_coef(N):
    """Calculate the coefficient for x-tick width based on the number of ticks.

    Parameters
    ----------
    N : int
        The number of ticks.

    Returns
    -------
    float
        The coefficient for x-tick width.

    """
    return (N - 1) * 0.5


def get_bins(data):
    """Determine appropriate bins for histogram visualization.

    Parameters
    ----------
    data : pandas.Series
        The data to determine bins for.

    Returns
    -------
    int or array
        Number of bins or array of bin edges.

    """
    if data.dtype.name == "category" or data.dtype.name == "object":
        return len(data.unique())
    if data.dtype.name == "bool":
        return 2
    if data.dtype.name == "datetime64" or data.dtype.name == "datetime64[ns]":
        return "auto"
    return np.histogram_bin_edges(data.dropna())


def prettify_metric_name(metric_name):
    """Convert metric names into a more readable format.

    Parameters
    ----------
    metric_name : str
        The name of the metric to prettify.

    Returns
    -------
    str
        The prettified metric name.

    """
    if metric_name == "WassersteinDistance":
        return "Wasserstein Distance"
    if metric_name == "KSTest":
        return "Kolmogorov-Smirnov Test"
    if metric_name == "ChiSquareTest":
        return "Chi-Square Test"
    if metric_name == "JSDivergence":
        return "Jensen-Shannon Divergence"
    if metric_name == "MaximumMeanDiscrepancy":
        return "Maximum Mean Discrepancy"
    if metric_name == "PairwiseCorrelationDifference":
        return "Pairwise Correlation Difference"
    return metric_name


def prettify_dataset_name(dataset_name):
    """Convert dataset names into a more readable format.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to prettify.

    Returns
    -------
    str
        The prettified dataset name.

    """
    if dataset_name in ("rossmann", "rossmann_store_sales", "rossmann_subsampled"):
        return "Rossmann"
    if dataset_name in ("airbnb-simplified_subsampled", "airbnb-simplified"):
        return "Airbnb"
    if dataset_name in ("Biodegradability_v1"):
        return "Biodegradability"
    if dataset_name in ("Cora_v1"):
        return "Cora"
    if dataset_name in ("imdb_MovieLens_v1"):
        return "IMDB MovieLens"
    if dataset_name in ("walmart", "walmart_subsampled"):
        return "Walmart"
    return dataset_name


def prettify_method_name(method_name):
    """Convert method names into a more readable format.

    Parameters
    ----------
    method_name : str
        The name of the method to prettify.

    Returns
    -------
    str
        The prettified method name.

    """
    if method_name == "REALTABFORMER":
        return "REALTABF"
    if method_name == "GRETEL_ACTGAN":
        return "G-ACTGAN"
    if method_name == "GRETEL_LSTM":
        return "G-LSTM"
    return method_name


def get_dataset_info(
    granularity_level, metric_type, all_results, dataset, methods, **kwargs
):
    """Retrieve information about datasets and methods.

    Parameters
    ----------
    granularity_level : str
        The granularity level of the metrics ('single_table' or 'single_column').
    metric_type : str
        The type of metrics ('distance' or 'detection').
    all_results : dict
        Dictionary containing all results.
    dataset : str
        The name of the dataset.
    methods : list
        List of method names.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    tuple
        A tuple containing base metrics, base metric names, save_figs flag,
        save_figs_path, and methods.

    Raises
    ------
    ValueError
        If an unknown metric type or granularity level is provided.

    """
    base_metrics = list(
        all_results[dataset][list(all_results[dataset].keys())[0]][
            f"{granularity_level}_metrics"
        ].keys()
    )

    if granularity_level == "single_table":
        if metric_type == "distance":
            base_metrics = [
                metric for metric in base_metrics if "detection" not in metric.lower()
            ]
        elif metric_type == "detection":
            base_metrics = [
                metric for metric in base_metrics if "detection" in metric.lower()
            ]
        else:
            raise ValueError(
                f"Unknown metric type {metric_type}. Should be either 'distance' or 'detection'."
            )
    elif granularity_level == "single_column":
        if metric_type == "distance":
            base_metrics = [
                metric for metric in base_metrics if "detection" not in metric.lower()
            ]
            base_metrics = [
                metric for metric in base_metrics if "test" not in metric.lower()
            ]
        elif metric_type == "detection":
            base_metrics = [
                metric for metric in base_metrics if "detection" in metric.lower()
            ]
        else:
            raise ValueError(
                f"Unknown metric type {metric_type}. Should be either 'distance' or 'detection'."
            )
    else:
        raise ValueError(
            f"Unknown granularity level {granularity_level}. Should be either 'single_table' or 'single_column'."
        )

    base_metric_names = [prettify_metric_name(metric) for metric in base_metrics]

    save_figs = kwargs.get("save_figs", False)
    save_figs_path = kwargs.get("save_figs_path", "./figs")
    save_figs_path = Path(save_figs_path) / granularity_level / metric_type

    method_order = kwargs.get(
        "method_order",
        [
            "SDV",
            "RCTGAN",
            "REALTABFORMER",
            "MOSTLYAI",
            "GRETEL_ACTGAN",
            "GRETEL_LSTM",
            "CLAVADDPM",
        ],
    )

    if method_order is not None:
        methods = [
            method
            for method in method_order
            if method in methods and method in all_results[dataset]
        ]
        methods += sorted([method for method in methods if method not in method_order])

    return base_metrics, base_metric_names, save_figs, save_figs_path, methods
