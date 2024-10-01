import os
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd
from syntherela.metadata import Metadata
from syntherela.data import save_tables, load_tables, remove_sdv_columns

from ClavaDDPM.preprocess_utils import table_label_decode, reconstruct_dates


def revert_ids(df, metadata, table_name):
    primary_key = metadata.get_primary_key(table_name)
    if primary_key is not None:
        df[primary_key] = df[f"{table_name}_id"]
        df.drop(columns=[f"{table_name}_id"], inplace=True)
    else:
        df.drop(columns=[f"{table_name}_id"], inplace=True)

    for parent in metadata.get_parents(table_name):
        foreign_keys = metadata.get_foreign_keys(parent, table_name)
        assert len(foreign_keys) == 1
        foreign_key = foreign_keys[0]
        df[foreign_key] = df[f"{parent}_id"]
        df.drop(columns=[f"{parent}_id"], inplace=True)

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--run-id", type=int, default=1, help="Run id")
    parser.add_argument(
        "--real-data-path",
        type=str,
        default="data/original/",
        help="Path to the original data.",
    )
    parser.add_argument(
        "--synthetic-data-path",
        type=str,
        default="data/synthetic/",
        help="Path to the original data.",
    )

    return parser.parse_args()


def main(args):
    run = str(args.run_id)
    dataset_name = args.dataset_name
    real_data_path = args.real_data_path
    synthetic_data_path = args.synthetic_data_path

    model_name = "CLAVADDPM"

    metadata = Metadata().load_from_json(
        Path(real_data_path) / f"{dataset_name}/metadata.json"
    )

    tables = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)

    processed_data_path = os.path.join("ClavaDDPM", "complex_data", dataset_name)
    generated_data_path = os.path.join(
        "ClavaDDPM", f"clavaDDPM_workspace_run{run}", dataset_name
    )
    synthetic_data_path = os.path.join(synthetic_data_path, dataset_name)

    if os.path.exists(os.path.join(processed_data_path, "first_dates.json")):
        with open(os.path.join(processed_data_path, "first_dates.json"), "r") as f:
            first_dates = json.load(f)

    tables_synthetic = dict()
    for table_name in metadata.get_tables():
        table_meta = metadata.get_table_meta(table_name)["columns"]
        table_path = os.path.join(
            generated_data_path, table_name, "_final", f"{table_name}_synthetic.csv"
        )
        df = pd.read_csv(table_path)

        datetime_columns = metadata.get_column_names(table_name, sdtype="datetime")
        numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")

        le_path = os.path.join(processed_data_path, f"{table_name}_label_encoders.pkl")
        with open(le_path, "rb") as f:
            label_encoders = pickle.load(f)

        for column in label_encoders.keys():
            if column in df.columns:
                df[column] = df[column].astype(int)

        df = table_label_decode(df, label_encoders)

        if len(datetime_columns):
            for col in datetime_columns:
                first_date_str = first_dates[table_name][col]
                df[col] = reconstruct_dates(df[col], first_date_str)
                df[col] = pd.to_datetime(df[col], format="%y%m%d")
                column_format = table_meta[col].get("datetime_format", "%Y-%m-%d")
                df[col] = df[col].dt.strftime(column_format)

        for col in numerical_columns:
            dtype = table_meta[col]["computer_representation"]
            if dtype == "Int64":
                df[col] = df[col].round().astype(dtype)

        df = revert_ids(df, metadata, table_name)
        tables_synthetic[table_name] = df

    save_data_path = os.path.join(synthetic_data_path, model_name, run, "sample1")
    save_tables(tables_synthetic, save_data_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
