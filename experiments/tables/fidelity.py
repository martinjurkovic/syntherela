import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

datasets = [
    "airbnb-simplified_subsampled",
    "rossmann_subsampled",
    "walmart_subsampled",
    "Biodegradability_v1",
    "imdb_MovieLens_v1",
    "CORA_v1",
]

dataset_names_dict = {
    "airbnb-simplified_subsampled": "AirBnB",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "Biodegradability_v1": "Biodeg.",
    "imdb_MovieLens_v1": "MovieLens",
    "CORA_v1": "CORA",
}

single_table_methods = [
    "bayesian_network",
    "ddpm",
    "ctgan",
    "nflow",
    "tvae",
]

all_methods = [
    "SDV",
    "RCTGAN",
    "REALTABFORMER",
    "MOSTLYAI",
    "GRETEL_ACTGAN",
    "GRETEL_LSTM",
] + single_table_methods

method_names_dict = {
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTABF.",
    "MOSTLYAI": "MOSTLYAI",
    "GRETEL_ACTGAN": "G-ACTGAN",
    "GRETEL_LSTM": "G-LSTM",
    "ddpm": "DDPM",
    "ctgan": "CTGAN",
    "nflow": "NFLOW",
    "tvae": "TVAE",
    "bayesian_network": "BN",
}

dataset_method_dict = {
    "airbnb-simplified_subsampled": [
        "SDV",
        "RCTGAN",
        "REALTABFORMER",
        "MOSTLYAI",
        "GRETEL_ACTGAN",
        "GRETEL_LSTM",
        "bayesian_network",
        "ctgan",
        "ddpm",
        "nflow",
        "tvae",
    ],
    "rossmann_subsampled": [
        "SDV",
        "RCTGAN",
        "REALTABFORMER",
        "MOSTLYAI",
        "GRETEL_ACTGAN",
        "GRETEL_LSTM",
        "bayesian_network",
        "ctgan",
        "ddpm",
        "nflow",
        "tvae",
    ],
    "walmart_subsampled": [
        "SDV",
        "RCTGAN",
        "REALTABFORMER",
        "MOSTLYAI",
        "GRETEL_ACTGAN",
        "GRETEL_LSTM",
        "bayesian_network",
        "ctgan",
        "ddpm",
        "nflow",
        "tvae",
    ],
    "Biodegradability_v1": [
        "SDV",
        "RCTGAN",
        "MOSTLYAI",
        "GRETEL_ACTGAN",
        "GRETEL_LSTM",
        "bayesian_network",
        "ctgan",
        "ddpm",
        "nflow",
        "tvae",
    ],
    "imdb_MovieLens_v1": ["RCTGAN", "MOSTLYAI", "GRETEL_ACTGAN", "ddpm"],
    "CORA_v1": [
        "SDV",
        "RCTGAN",
        "GRETEL_ACTGAN",
        "GRETEL_LSTM",
        "bayesian_network",
        "ctgan",
        "ddpm",
        "nflow",
        "tvae",
    ],
}

RESULTS_PATH = Path("./results")
RUNS = ["1", "2", "3"]

ALPHA_STATISTICAL = 0.05
ALPHA_DETECTION = 0.05

INCLUDE_SINGLE_TABLE = True


def create_table(metric_type, table_name="table1"):
    if metric_type == "multi_table_metrics" or not INCLUDE_SINGLE_TABLE:
        for dataset in datasets:
            dict_ = dataset_method_dict[dataset]
            dict_ = [method for method in dict_ if method not in single_table_methods]
            dataset_method_dict[dataset] = dict_

    all_results = {}
    for run_id in RUNS:
        all_results[run_id] = {}
        for dataset in datasets:
            all_results[run_id][dataset] = {}
            methods = dataset_method_dict[dataset]
            # take only methods that are in all_methods
            methods = [method for method in methods if method in all_methods]
            for method in methods:
                for file in os.listdir(f"{RESULTS_PATH}/{run_id}"):
                    if file.startswith(f"{dataset}_{method}"):
                        with open(RESULTS_PATH / run_id / file, "r") as f:
                            all_results[run_id][dataset][method] = json.load(f)

    base_metrics = []

    if metric_type == "single_column_metrics":
        base_metrics = [
            "ChiSquareTest",
            "KolmogorovSmirnovTest",
            "HellingerDistance",
            "JensenShannonDistance",
            "TotalVariationDistance",
            "WassersteinDistance",
            "SingleColumnDetection-LogisticRegression",
            "SingleColumnDetection-XGBClassifier",
        ]

    if metric_type == "single_table_metrics":
        base_metrics = [
            "MaximumMeanDiscrepancy",
            "PairwiseCorrelationDifference",
            "SingleTableDetection-LogisticRegression",
            "SingleTableDetection-XGBClassifier",
        ]

    if metric_type == "multi_table_metrics":
        base_metrics = [
            "CardinalityShapeSimilarity",
            "AggregationDetection-LogisticRegression",
            "AggregationDetection-XGBClassifier",
        ]

    df_run_list = []
    df_copy_list = []
    df_how_many_runs = pd.DataFrame(columns=["dataset", "methods"] + base_metrics)
    for run_id in RUNS:
        df = pd.DataFrame(columns=["dataset", "methods"] + base_metrics)
        df_latex = df.copy()
        for idx, dataset in enumerate(datasets):
            cnt = True
            methods = dataset_method_dict[dataset]
            for method in methods:
                # append row with dataset, method and empty values for all metrics
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [[dataset, method] + [""] * len(base_metrics)],
                            columns=["dataset", "methods"] + base_metrics,
                        ),
                    ],
                    ignore_index=True,
                )
                df_how_many_runs = pd.concat(
                    [
                        df_how_many_runs,
                        pd.DataFrame(
                            [[dataset, method] + [""] * len(base_metrics)],
                            columns=["dataset", "methods"] + base_metrics,
                        ),
                    ],
                    ignore_index=True,
                )
                dataset_latex_text = f"\\multirow{{{len(dataset_method_dict[dataset])}}}{{*}}{{{dataset_names_dict[dataset]}}}"
                df_latex = pd.concat(
                    [
                        df_latex,
                        pd.DataFrame(
                            [
                                [
                                    dataset_latex_text if cnt else "",
                                    method_names_dict[method],
                                ]
                                + [""] * len(base_metrics)
                            ],
                            columns=["dataset", "methods"] + base_metrics,
                        ),
                    ],
                    ignore_index=True,
                )
                cnt = False

        df["AGG"] = 0
        df_how_many_runs["AGG"] = 0

        df_copy = df.copy()

        for dataset in datasets:
            detection_dict = {
                "SDV": {},
                "RCTGAN": {},
                "REALTABFORMER": {},
                "MOSTLYAI": {},
                "GRETEL_ACTGAN": {},
                "GRETEL_LSTM": {},
                "bayesian_network": {},
                "ddpm": {},
                "ctgan": {},
                "nflow": {},
                "tvae": {},
            }
            if metric_type == "single_column_metrics":
                for metric in base_metrics:
                    methods = dataset_method_dict[dataset]
                    for method in methods:
                        if (
                            metric
                            not in all_results[run_id][dataset][method][metric_type]
                        ):
                            continue
                        for table in all_results[run_id][dataset][method][metric_type][
                            metric
                        ]:
                            for column in all_results[run_id][dataset][method][
                                metric_type
                            ][metric][table]:

                                detection_dict[method].setdefault(metric, {})
                                detection_dict[method][metric].setdefault("detected", 0)
                                detection_dict[method][metric].setdefault("all", 0)
                                detection_dict[method][metric].setdefault("copied", 0)
                                detection_dict[method][metric]["all"] += 1

                                if (
                                    "p_value"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["p_value"]
                                    if metric_value <= ALPHA_STATISTICAL:
                                        detection_dict[method][metric]["detected"] += 1
                                if (
                                    "p_val"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["p_val"]
                                    if metric_value <= ALPHA_STATISTICAL:
                                        detection_dict[method][metric]["detected"] += 1
                                if (
                                    "pval"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["pval"]
                                    if metric_value <= ALPHA_STATISTICAL:
                                        detection_dict[method][metric]["detected"] += 1
                                if (
                                    "value"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["value"]
                                    lower_bound = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["reference_ci"][0]
                                    upper_bound = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["reference_ci"][1]
                                    if (
                                        metric_value < lower_bound
                                        or metric_value > upper_bound
                                    ):
                                        detection_dict[method][metric]["detected"] += 1
                                if (
                                    "bin_test_p_val"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["bin_test_p_val"]
                                    if metric_value <= ALPHA_DETECTION:
                                        detection_dict[method][metric]["detected"] += 1
                                if (
                                    "copying_p_val"
                                    in all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]
                                ):
                                    metric_value = all_results[run_id][dataset][method][
                                        metric_type
                                    ][metric][table][column]["copying_p_val"]
                                    if metric_value <= ALPHA_DETECTION:
                                        detection_dict[method][metric]["copied"] += 1

            if metric_type == "single_table_metrics":
                for metric in base_metrics:
                    methods = dataset_method_dict[dataset]
                    for method in methods:
                        if (
                            metric
                            not in all_results[run_id][dataset][method][metric_type]
                        ):
                            continue
                        for table in all_results[run_id][dataset][method][metric_type][
                            metric
                        ]:
                            detection_dict[method].setdefault(metric, {})
                            detection_dict[method][metric].setdefault("detected", 0)
                            detection_dict[method][metric].setdefault("all", 0)
                            detection_dict[method][metric].setdefault("copied", 0)
                            detection_dict[method][metric]["all"] += 1
                            if (
                                "p_value"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["p_value"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["p_val"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "pval"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["pval"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "value"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["value"]
                                lower_bound = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["reference_ci"][0]
                                upper_bound = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["reference_ci"][1]
                                if (
                                    metric_value < lower_bound
                                    or metric_value > upper_bound
                                ):
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "bin_test_p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["bin_test_p_val"]
                                if metric_value <= ALPHA_DETECTION:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "copying_p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["copying_p_val"]
                                if metric_value <= ALPHA_DETECTION:
                                    detection_dict[method][metric]["copied"] += 1

            if metric_type == "multi_table_metrics":
                for metric in base_metrics:
                    methods = dataset_method_dict[dataset]
                    for method in methods:
                        if (
                            metric
                            not in all_results[run_id][dataset][method][metric_type]
                        ):
                            continue
                        for table in all_results[run_id][dataset][method][metric_type][
                            metric
                        ]:
                            detection_dict[method].setdefault(metric, {})
                            detection_dict[method][metric].setdefault("detected", 0)
                            detection_dict[method][metric].setdefault("all", 0)
                            detection_dict[method][metric].setdefault("copied", 0)
                            detection_dict[method][metric]["all"] += 1
                            if (
                                "p_value"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["p_value"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["p_val"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "pval"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["pval"]
                                if metric_value <= ALPHA_STATISTICAL:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "value"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["value"]
                                lower_bound = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["reference_ci"][0]
                                upper_bound = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["reference_ci"][1]
                                if (
                                    metric_value < lower_bound
                                    or metric_value > upper_bound
                                ):
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "bin_test_p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["bin_test_p_val"]
                                if metric_value <= ALPHA_DETECTION:
                                    detection_dict[method][metric]["detected"] += 1
                            if (
                                "copying_p_val"
                                in all_results[run_id][dataset][method][metric_type][
                                    metric
                                ][table]
                            ):
                                metric_value = all_results[run_id][dataset][method][
                                    metric_type
                                ][metric][table]["copying_p_val"]
                                if metric_value <= ALPHA_DETECTION:
                                    detection_dict[method][metric]["copied"] += 1

            for method in dataset_method_dict[dataset]:
                for metric in base_metrics:
                    if metric not in detection_dict[method]:
                        continue
                    df.loc[
                        (df["methods"] == method) & (df["dataset"] == dataset), metric
                    ] = detection_dict[method][metric]["detected"]
                    df_copy.loc[
                        (df_copy["methods"] == method)
                        & (df_copy["dataset"] == dataset),
                        metric,
                    ] = detection_dict[method][metric]["copied"]
                    df_how_many_runs.loc[
                        (df_how_many_runs["methods"] == method)
                        & (df_how_many_runs["dataset"] == dataset),
                        metric,
                    ] = detection_dict[method][metric]["all"]
                    df.loc[
                        (df["methods"] == method) & (df["dataset"] == dataset), "AGG"
                    ] += detection_dict[method][metric]["detected"]
                    df_how_many_runs.loc[
                        (df_how_many_runs["methods"] == method)
                        & (df_how_many_runs["dataset"] == dataset),
                        "AGG",
                    ] += detection_dict[method][metric]["all"]

        # set all columns except dataset and methods to be float
        # fill empty values with nan
        for col in df.columns[2:]:
            df[col] = df[col].replace("", np.nan)
            df[col] = df[col].astype(float)
        df_run_list.append(df.copy())
        df_copy_list.append(df_copy.copy())

    df_average = (
        pd.concat(df_run_list)
        .groupby(["dataset", "methods"], sort=False)
        .mean()
        .reset_index()
    )
    df_sum = (
        pd.concat(df_run_list)
        .groupby(["dataset", "methods"], sort=False)
        .sum()
        .reset_index()
    )

    def assign_multicolumn_string(average, se):
        return f"\\multicolumn {{1}}{{c}}{{${average:.2f}\\pm{se:.3f}$}}"

    # iterate all rows and columns
    for i in range(df_latex.shape[0]):
        for j in range(df_latex.shape[1]):
            if j < 2:
                continue

            if df_average.iloc[i, j] != df_average.iloc[i, j]:
                df_latex.iloc[i, j] = "\multicolumn {1}{c}{-}"
                continue
            temp_result_string = ""
            for idx_, df_temp in enumerate(df_run_list):
                temp_result_string += f"{int(df_temp.iloc[i, j])}"
                temp_result_string += ", "

            temp_result_string = temp_result_string[:-2]

            temp_result_latex = f"{temp_result_string} ({df_how_many_runs.iloc[i, j]})"

            df_latex.iloc[i, j] = f"\\multicolumn {{1}}{{c}}{{{temp_result_latex}}}"

    for dataset in datasets:
        for metric in base_metrics:
            metric_results = df_average.loc[(df_average["dataset"] == dataset), metric]
            # get the lowest number of column [0]
            metric_results = metric_results.apply(pd.to_numeric)

            min_indexes = []

            # Iterate over columns
            min_val = metric_results.min()
            for idx, value in metric_results.items():
                if metric_results[idx] == min_val:
                    min_indexes.append(idx)

            min_indexes
            if len(min_indexes) == len(dataset_method_dict[dataset]):
                continue
            for idx in min_indexes:
                temp_result_list = []
                for df_temp in df_run_list:
                    temp_result_list.append(int(df_temp.loc[idx, metric]))

                temp_result_string = ""
                for idx_, df_temp in enumerate(df_run_list):
                    temp_result_string += f"{int(df_temp.loc[idx, metric])}"
                    temp_result_string += ", "

                temp_result_string = temp_result_string[:-2]
                temp_result_latex = (
                    f"{temp_result_string} ({df_how_many_runs.loc[idx, metric]})"
                )

                df_latex.loc[idx, metric] = (
                    f"\\multicolumn {{1}}{{c}}{{\\textbf{{{temp_result_latex}}}}}"
                )

    for dataset in datasets:
        agg_res = df_sum.loc[(df_sum["dataset"] == dataset), ["AGG"]]

        min_indexes = []

        min_val = agg_res["AGG"].min()
        for idx, row in agg_res.iterrows():
            if agg_res["AGG"][idx] == min_val:
                min_indexes.append(idx)

        min_indexes
        if len(min_indexes) == len(dataset_method_dict[dataset]):
            continue
        for idx in min_indexes:
            df_latex.loc[idx, "methods"] = (
                f"\\multirow{{1}}{{*}}{{\\textbf{{{method_names_dict[df_sum.loc[idx, 'methods']]}}}}}"
            )

    latex_table = df_latex.to_latex(header=False, index=False, escape=False)
    cline = f"\\ \cline{{2-{len(base_metrics) + 2}}}\n"
    latex_table = latex_table.replace("\\\n", cline)

    # iterate latex_table string row by row
    latex_table = latex_table.split("\n")

    for idx, row in enumerate(latex_table):
        # if row starts with multirow
        if row.lstrip().startswith("\multirow"):
            # remove cline from previous row
            latex_table[idx - 1] = latex_table[idx - 1].replace(cline[:-1], "\\ \hline")
            # add \n to previous row
            latex_table[idx - 1] += "\n"

    latex_table[-4] = latex_table[-4].replace(cline[:-1], "\\ \hline") + "\n"

    # join all rows back to one string
    latex_table = "\n".join(latex_table)

    latex_table = latex_table.replace("\\textbackslash ", "\\")

    table_latex = re.sub(" +", " ", latex_table)
    # save the latex table to file
    print(f"Saving table {table_name}.")
    with open(f"results/tables/table{table_name}.tex", "w") as f:
        f.write(table_latex)


create_table("multi_table_metrics", "1")
create_table("single_column_metrics", "8")
create_table("single_table_metrics", "9")
