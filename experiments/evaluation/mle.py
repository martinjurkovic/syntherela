import os
import json
import argparse
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau, weightedtau
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns
from syntherela.metrics.utility import MachineLearningEfficacyMetric
from syntherela.utils import NpEncoder


## DATA LOADING
def load_rossmann(method, run="1"):
    dataset_name = "rossmann_subsampled"

    metadata = Metadata().load_from_json(
        f"{PROJECT_PATH}/data/original/{dataset_name}/metadata.json"
    )

    tables = load_tables(f"{PROJECT_PATH}/data/original/{dataset_name}/", metadata)
    tables_synthetic = load_tables(
        f"{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1", metadata
    )

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic, metadata = remove_sdv_columns(
        tables_synthetic, metadata, update_metadata=False
    )

    tables_test = load_tables(
        f'{PROJECT_PATH}/data/original/{dataset_name.split("_")[0]}/', metadata
    )
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)
    # split the test data
    min_date = datetime.strptime("2014-10-01", "%Y-%m-%d")
    max_date = datetime.strptime("2014-11-01", "%Y-%m-%d")
    tables_test["historical"] = tables_test["historical"][
        (tables_test["historical"]["Date"] >= min_date)
        & (tables_test["historical"]["Date"] < max_date)
    ]

    return tables, tables_synthetic, tables_test, metadata


def load_airbnb(method, run="1"):
    dataset_name = "airbnb-simplified_subsampled"

    metadata = Metadata().load_from_json(
        f"{PROJECT_PATH}/data/original/{dataset_name}/metadata.json"
    )

    tables = load_tables(f"{PROJECT_PATH}/data/original/{dataset_name}/", metadata)
    tables_synthetic = load_tables(
        f"{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1", metadata
    )

    tables, metadata = remove_sdv_columns(tables, metadata)
    for table in tables:
        if "index" in tables_synthetic[table].columns:
            tables_synthetic[table].drop(columns=["index"], inplace=True)
    tables_synthetic, metadata = remove_sdv_columns(
        tables_synthetic, metadata, update_metadata=False
    )

    tables_test = load_tables(
        f'{PROJECT_PATH}/data/original/{dataset_name.split("_")[0]}/', metadata
    )
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)

    # select users with at most 50 sessions
    sessions_count = (
        tables_test["sessions"].reset_index().groupby("user_id").index.count()
    )
    eligable_users = sessions_count[sessions_count <= 50].index
    no_sessions = tables_test["users"][
        ~tables_test["users"]["id"].isin(tables_test["sessions"]["user_id"])
    ]["id"]
    eligable_users = eligable_users.union(no_sessions)
    # select users that are not in the synthetic data
    eligable_users = eligable_users[~eligable_users.isin(tables["users"]["id"])]

    selected_users = np.random.choice(eligable_users, 2000, replace=False)
    tables_test["users"] = tables_test["users"][
        tables_test["users"]["id"].isin(selected_users)
    ]
    tables_test["sessions"] = tables_test["sessions"][
        tables_test["sessions"]["user_id"].isin(selected_users)
    ]

    return tables, tables_synthetic, tables_test, metadata


def load_walmart(method, run="1"):
    dataset_name = "walmart_subsampled"

    metadata = Metadata().load_from_json(
        f"{PROJECT_PATH}/data/original/{dataset_name}/metadata.json"
    )

    tables = load_tables(f"{PROJECT_PATH}/data/original/{dataset_name}/", metadata)
    tables_synthetic = load_tables(
        f"{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1", metadata
    )

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic, metadata = remove_sdv_columns(
        tables_synthetic, metadata, update_metadata=False
    )

    # split the test data
    tables_test = load_tables(
        f'{PROJECT_PATH}/data/original/{dataset_name.split("_")[0]}/', metadata
    )
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)

    min_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
    max_date = datetime.strptime("2012-02-01", "%Y-%m-%d")
    tables_test["depts"] = tables_test["depts"][
        (tables_test["depts"]["Date"] >= min_date)
        & (tables_test["depts"]["Date"] < max_date)
    ]
    tables_test["features"] = tables_test["features"][
        (tables_test["features"]["Date"] >= min_date)
        & (tables_test["features"]["Date"] < max_date)
    ]

    return tables, tables_synthetic, tables_test, metadata


## DATA PREPARATION
def process_rossmann(tables, metadata):
    df = tables["historical"].merge(tables["store"], on="Store")
    numerical_columns = []
    for table in metadata.get_tables():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata["columns"].items():
            if column_info["sdtype"] == "numerical":
                numerical_columns.append(column)
            elif column_info["sdtype"] == "id":
                if column in df.columns and column != "Store":
                    df.drop(columns=[column], inplace=True)

    # drop the StateHoliday column as it is constant causing problems with standardization
    # drop the DayOfWeek as we will be aggregating the data by month
    df.drop(columns=["StateHoliday", "DayOfWeek"], inplace=True)

    # set the categories for categorical variables (in case methods do not generate certain values)
    df["Open"] = pd.Categorical(df["Open"], categories=["0", "1"])
    df["Promo"] = pd.Categorical(df["Promo"], categories=["0", "1"])
    df["SchoolHoliday"] = pd.Categorical(df["SchoolHoliday"], categories=["0", "1"])
    df["StoreType"] = pd.Categorical(df["StoreType"], categories=["a", "b", "c", "d"])
    df["Assortment"] = pd.Categorical(df["Assortment"], categories=["a", "b", "c"])
    df["PromoInterval"] = pd.Categorical(
        df["PromoInterval"].astype("object").fillna("missing"),
        categories=[
            "missing",
            "Jan,Apr,Jul,Oct",
            "Mar,Jun,Sept,Dec",
            "Feb,May,Aug,Nov",
        ],
    )
    # one-hot encode the categorical columns
    df = pd.get_dummies(
        df,
        columns=[
            "Open",
            "Promo",
            "SchoolHoliday",
            "StoreType",
            "Assortment",
            "PromoInterval",
        ],
        drop_first=True,
    )
    # aggregate the data by store and month
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df = df.groupby(["Store", "Month", "Year"]).mean().reset_index(drop=True)
    # drop the dates due to subsampling
    df.drop(columns=["Date"], inplace=True)

    df[numerical_columns] = df[numerical_columns].fillna(0)
    y = df.pop("Customers")

    # remove missing / infinite y values
    mask = y.isna() | y.isin([np.inf, -np.inf])
    X = df[~mask]
    y = y[~mask]
    return X, y


def process_airbnb(tables, metadata, categories):
    df = tables["users"].copy()
    df.drop(columns=["date_first_booking"], inplace=True)

    for table in metadata.get_tables():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata["columns"].items():
            if column_info["sdtype"] == "categorical":
                if column in df.columns:
                    df[column] = pd.Categorical(
                        df[column], categories=categories[column]
                    )

    # impute missing values
    df[["age"]] = df[["age"]].fillna(0)

    # aggregate the sessions data by user_id to obtain average session duration
    sessions_data = (
        tables["sessions"][["secs_elapsed", "user_id"]].groupby("user_id").mean()
    )
    # add the sessions count
    sessions_count = (
        tables["sessions"]["user_id"].value_counts().rename("sessions_count")
    )
    # join the sessions data and count of sessions
    sessions_data = sessions_data.join(sessions_count, on="user_id")

    # merge the sessions data with the users data and fill missing values with 0
    df = df.merge(sessions_data, left_on="id", right_index=True, how="left")
    df[sessions_data.columns] = df[sessions_data.columns].fillna(0)
    df.drop(columns=["id"], inplace=True)

    y = df.pop("country_destination")
    X = df.copy()

    # convert y to binary variable determining if the user booked a trip or not
    y = y != "NDF"

    return X, y


def process_walmart(tables, metadata):
    df = (
        tables["depts"]
        .merge(tables["stores"], on="Store")
        .merge(tables["features"], on=["Store", "Date"], suffixes=("", "_y"))
    )
    df.drop(df.filter(regex="_y$").columns, axis=1, inplace=True)
    df.drop(columns=["Dept"], inplace=True)

    # one-hot encode the categorical columns
    df["Type"] = pd.Categorical(df["Type"], categories=["A", "B", "C"])
    df = pd.get_dummies(df, columns=["Type"])
    # obtain average daily sales across all departments
    df = df.groupby(["Store", "Date"]).mean().reset_index(drop=True)

    y = df.pop("Weekly_Sales")

    # remove missing / infinite y values
    mask = y.isna() | y.isin([np.inf, -np.inf])
    X = df[~mask]
    y = y[~mask]
    return X, y


## UTILITY
def load_dataset(dataset_name, method, run="1"):
    if dataset_name == "rossmann":
        tables, tables_synthetic, tables_test, metadata = load_rossmann(method, run=run)
        feature_engineering_function = process_rossmann
        target = ("historical", "Customers", "regression")
    elif dataset_name == "airbnb":
        tables, tables_synthetic, tables_test, metadata = load_airbnb(method, run=run)
        # Store the categories from the real data to avoid mistakes in case methods do not generate certain values
        categories = {}
        for _, table in tables_test.items():
            for column in table.columns:
                if table[column].dtype.name == "category":
                    categories[column] = table[column].cat.categories.tolist()
        feature_engineering_function = partial(process_airbnb, categories=categories)
        # feature_engineering_function = None
        target = ("users", "country_destination", "classification")
    elif dataset_name == "walmart":
        tables, tables_synthetic, tables_test, metadata = load_walmart(method, run=run)
        feature_engineering_function = process_walmart
        target = ("depts", "Weekly_Sales", "regression")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return (
        tables,
        tables_synthetic,
        tables_test,
        metadata,
        feature_engineering_function,
        target,
    )


methods = [
    "SDV",
    "RCTGAN",
    "REALTABFORMER",
    "MOSTLYAI",
    "GRETEL_ACTGAN",
    "GRETEL_LSTM",
]

datasets = [
    "rossmann",
    "airbnb",
    "walmart",
]

classifiers = {
    "regression": {
        "xgboost": xgb.XGBRegressor,
        "linear": LinearRegression,
        "random_forest": RandomForestRegressor,
        "decision_tree": DecisionTreeRegressor,
        "knn": KNeighborsRegressor,
        "svr": SVR,
        "mlp": MLPRegressor,
    },
    "classification": {
        "xgboost": xgb.XGBClassifier,
        "linear": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "svc": SVC,
        "gaussian_nb": GaussianNB,
        "mlp": MLPClassifier,
    },
}

cls_args = {
    "xgboost": {"random_state": None, "seed": None},
    "linear": {},
    "random_forest": {"random_state": None},
    "decision_tree": {"random_state": None},
    "knn": {},
    "svr": {},
    "svc": {"random_state": None, "probability": True},
    "gaussian_nb": {},
    "mlp": {"random_state": None},
}


feature_selection_models = {"regression": ["xgboost"], "classification": ["xgboost"]}

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_PATH = os.getenv("PROJECT_PATH")

    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset-name",
        type=str,
        default="rossmann",
        choices=datasets,
        help="Dataset name to run the experiment on.",
    )
    args.add_argument(
        "--methods",
        "-m",
        default=None,
        help="List of methods to evaluate.",
        action="append",
    )
    args.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    args.add_argument(
        "--bootstrap_repetitions",
        type=int,
        default=100,
        help="Number of bootstrap samples.",
    )
    args.add_argument("--run", type=str, default="1", help="Run number.")
    args = args.parse_args()

    dataset_name = args.dataset_name
    if args.methods is None:
        methods_to_run = methods
    else:
        methods_to_run = args.methods
    seed = args.seed
    m = args.bootstrap_repetitions
    run = args.run

    results = {}
    results[dataset_name] = {}
    for method in methods_to_run:
        print(f"Method: {method}, Dataset: {dataset_name}")
        results[dataset_name][method] = {}
        (
            tables,
            tables_synthetic,
            tables_test,
            metadata,
            feature_engineering_function,
            target,
        ) = load_dataset(dataset_name, method, run=run)
        task = target[2]
        for classifier, classifier_cls in classifiers[task].items():
            classifier_args_ = cls_args[classifier]
            if "random_state" in classifier_args_:
                classifier_args_["random_state"] = seed
            if "seed" in classifier_args_:
                classifier_args_["seed"] = seed

            ml_metric = MachineLearningEfficacyMetric(
                target=target,
                feature_engineering_function=feature_engineering_function,
                classifier_cls=classifier_cls,
                random_state=seed,
                classifier_args=classifier_args_,
            )

            result = ml_metric.run(
                tables,
                tables_synthetic,
                metadata,
                tables_test,
                m=m,
                feature_importance=classifier in feature_selection_models[task],
            )
            print(
                f"Classifier: {classifier} real_score: {result['real_score'] :.3f} +- {result['real_score_se']:.3f}, synthetic_score: {result['synthetic_score']:.3f} +- {result['synthetic_score_se']:.3f}"
            )
            importances_real = result.pop("importance_real", None)
            importances_syn = result.pop("importance_synthetic", None)
            results[dataset_name][method][classifier] = result
            if classifier in feature_selection_models[task]:
                feature_importances_spearman = []
                feature_importances_tau = []
                feature_importances_weighted = []
                feature_names = np.array(result["feature_names"])

                true_feature_importance_real = np.array(
                    result["true_feature_importance_real"]
                )
                true_feature_importance_synthetic = np.array(
                    result["true_feature_importance_synthetic"]
                )
                real_rank = np.argsort(true_feature_importance_real)
                synthetic_rank = np.argsort(true_feature_importance_synthetic)
                true_features_spearman_rank = spearmanr(
                    feature_names[real_rank], feature_names[synthetic_rank]
                ).statistic
                real_feature_order = feature_names[real_rank]
                synthetic_feature_order = feature_names[synthetic_rank]
                true_features_tau_rank = kendalltau(
                    feature_names[real_rank], feature_names[synthetic_rank]
                ).statistic
                indexed_real_rank = np.arange(len(real_rank))
                indexed_synthetic_rank = np.array(
                    [real_rank.tolist().index(feature) for feature in synthetic_rank]
                )
                true_features_weighted_rank = weightedtau(
                    indexed_real_rank, indexed_synthetic_rank, rank=None
                ).statistic

                for i in range(m):
                    # importance_real = importances_real[i]
                    importance_syn = importances_syn[i]
                    # real_rank = np.argsort(importance_real)
                    synthetic_rank = np.argsort(importance_syn)
                    features_spearman_rank = spearmanr(
                        feature_names[real_rank], feature_names[synthetic_rank]
                    ).statistic
                    features_tau_rank = kendalltau(
                        feature_names[real_rank], feature_names[synthetic_rank]
                    ).statistic

                    indexed_real_rank = np.arange(len(real_rank))
                    indexed_synthetic_rank_boot = np.array(
                        [
                            real_rank.tolist().index(feature)
                            for feature in synthetic_rank
                        ]
                    )

                    features_weighted_rank = weightedtau(
                        indexed_real_rank, indexed_synthetic_rank_boot, rank=None
                    ).statistic
                    feature_importances_spearman.append(features_spearman_rank)
                    feature_importances_tau.append(features_tau_rank)
                    feature_importances_weighted.append(features_weighted_rank)

        # rank the classifiers
        real_classifier_rank = list(
            dict(
                sorted(
                    results[dataset_name][method].items(),
                    key=lambda x: x[1]["real_score"],
                    reverse=True,
                )
            ).keys()
        )
        synthetic_classifier_rank = list(
            dict(
                sorted(
                    results[dataset_name][method].items(),
                    key=lambda x: x[1]["synthetic_score"],
                    reverse=True,
                )
            ).keys()
        )
        print(f"Real data classifier rank: {real_classifier_rank}")
        print(f"Synthetic data classifier rank: {synthetic_classifier_rank}")

        true_classifier_rank_spearman = spearmanr(
            real_classifier_rank, synthetic_classifier_rank
        ).statistic
        true_classifier_rank_kendall = kendalltau(
            real_classifier_rank, synthetic_classifier_rank
        ).statistic

        indexed_real_rank = np.array(
            [
                len(real_classifier_rank) - real_classifier_rank.index(classifier)
                for classifier in real_classifier_rank
            ]
        )
        indexed_synthetic_rank = np.array(
            [
                len(real_classifier_rank) - real_classifier_rank.index(classifier)
                for classifier in synthetic_classifier_rank
            ]
        )
        true_classifier_rank_weighted = weightedtau(
            indexed_real_rank, indexed_synthetic_rank, rank=None
        ).statistic

        classifier_rank_array_spearman = []
        classifier_rank_array_kendall = []
        classifier_rank_array_weighted = []

        for bootstrap_index in range(m):
            synthetic_classifier_rank_boot = list(
                dict(
                    sorted(
                        results[dataset_name][method].items(),
                        key=lambda x: x[1]["synthetic_score_array"][bootstrap_index],
                        reverse=True,
                    )
                ).keys()
            )

            classifier_rank_array_spearman.append(
                spearmanr(
                    real_classifier_rank, synthetic_classifier_rank_boot
                ).statistic
            )
            classifier_rank_array_kendall.append(
                kendalltau(
                    real_classifier_rank, synthetic_classifier_rank_boot
                ).statistic
            )

            indexed_real_rank_boot = np.array(
                [
                    len(real_classifier_rank) - real_classifier_rank.index(classifier)
                    for classifier in real_classifier_rank
                ]
            )
            indexed_synthetic_rank_boot = np.array(
                [
                    len(real_classifier_rank) - real_classifier_rank.index(classifier)
                    for classifier in synthetic_classifier_rank_boot
                ]
            )
            classifier_rank_array_weighted.append(
                weightedtau(
                    indexed_real_rank_boot, indexed_synthetic_rank_boot, rank=None
                ).statistic
            )

        spearman_rank = spearmanr(
            list(real_classifier_rank), list(synthetic_classifier_rank)
        )
        results[dataset_name][method]["classifier_rank"] = spearman_rank.statistic
        results[dataset_name][method]["feature_importance_spearman"] = (
            true_features_spearman_rank
        )
        results[dataset_name][method]["feature_importance_spearman_mean"] = np.mean(
            feature_importances_spearman
        )
        results[dataset_name][method]["feature_importance_spearman_se"] = np.std(
            feature_importances_spearman
        ) / np.sqrt(m)
        results[dataset_name][method]["feature_importance_tau"] = true_features_tau_rank
        results[dataset_name][method]["feature_importance_tau_mean"] = np.mean(
            feature_importances_tau
        )
        results[dataset_name][method]["feature_importance_tau_se"] = np.std(
            feature_importances_tau
        ) / np.sqrt(m)
        results[dataset_name][method]["feature_importance_weighted"] = (
            true_features_weighted_rank
        )
        results[dataset_name][method]["feature_importance_weighted_mean"] = np.mean(
            feature_importances_weighted
        )
        results[dataset_name][method]["feature_importance_weighted_se"] = np.std(
            feature_importances_weighted
        ) / np.sqrt(m)
        results[dataset_name][method]["spearman"] = true_classifier_rank_spearman
        results[dataset_name][method]["spearman_mean"] = np.mean(
            classifier_rank_array_spearman
        )
        results[dataset_name][method]["spearman_se"] = np.std(
            classifier_rank_array_spearman
        ) / np.sqrt(m)
        results[dataset_name][method]["kendall"] = true_classifier_rank_kendall
        results[dataset_name][method]["kendall_mean"] = np.mean(
            classifier_rank_array_kendall
        )
        results[dataset_name][method]["kendall_se"] = np.std(
            classifier_rank_array_kendall
        ) / np.sqrt(m)
        results[dataset_name][method]["weighted"] = true_classifier_rank_weighted
        results[dataset_name][method]["weighted_mean"] = np.mean(
            classifier_rank_array_weighted
        )
        results[dataset_name][method]["weighted_se"] = np.std(
            classifier_rank_array_weighted
        ) / np.sqrt(m)
        results[dataset_name][method]["real_feature_order"] = real_feature_order
        results[dataset_name][method]["synthetic_feature_order"] = (
            synthetic_feature_order
        )

        print()
        print(
            f"Boot spearman: {np.mean(classifier_rank_array_spearman):.3f}+-{np.std(classifier_rank_array_spearman) / np.sqrt(m):.4f}"
        )
        print(
            f"Boot kendall: {np.mean(classifier_rank_array_kendall) :.3f}+-{np.std(classifier_rank_array_kendall) / np.sqrt(m):.4f}"
        )
        print(
            f"Boot weighted: {np.mean(classifier_rank_array_weighted) :.3f}+-{np.std(classifier_rank_array_weighted) / np.sqrt(m):.4f}"
        )

        print(
            f"Feature importance spearman: {np.mean(feature_importances_spearman) :.3f}+-{np.std(feature_importances_spearman) / np.sqrt(m):.4f}"
        )
        print(
            f"Feature importance tau: {np.mean(feature_importances_tau) :.3f}+-{np.std(feature_importances_tau) / np.sqrt(m):.4f}"
        )
        print(
            f"Feature importance weighted: {np.mean(feature_importances_weighted) :.3f}+-{np.std(feature_importances_weighted) / np.sqrt(m):.4f}"
        )
        print()

        if len(methods_to_run) < len(methods):
            with open(
                f"{PROJECT_PATH}/results/mle_{dataset_name}_{run}_{seed}_{method}.json",
                "w",
            ) as f:
                json.dump(results, f, indent=4, cls=NpEncoder)

    with open(f"{PROJECT_PATH}/results/mle_{dataset_name}_{run}_{seed}.json", "w") as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
