import os
from typing import Dict, Optional, Union

import pandas as pd
from sdv.datasets.demo import get_available_demos, download_demo
from syntherela.metadata import Metadata


def load_tables(data_path: Union[str, os.PathLike], metadata: Metadata):
    tables = {}
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".csv"):
            continue
        table_name = file_name.split(".")[0]
        dtypes = {}
        parse_dates = []
        datetime_formats = {}
        for column, column_info in metadata.tables[table_name].columns.items():
            if column_info["sdtype"] == "categorical":
                dtypes[column] = "category"
            elif column_info["sdtype"] == "boolean":
                dtypes[column] = "bool"
            elif column_info["sdtype"] == "datetime":
                parse_dates.append(column)
                datetime_format = column_info.get("datetime_format")
                if not datetime_format:
                    raise ValueError(
                        f'"datetime_format" not found in metadata for column "{column}" in table "{table_name}"'
                    )
                datetime_formats[column] = datetime_format

        table = pd.read_csv(
            f"{data_path}/{file_name}",
            low_memory=False,
            dtype=dtypes,
            parse_dates=parse_dates,
            date_format=datetime_formats,
        )
        tables[table_name] = table
    return tables


def remove_sdv_columns(
    tables: Dict[str, pd.DataFrame],
    metadata: Metadata,
    update_metadata=True,
    validate=True,
):
    """
    "_v1" Versions of the relational demo datasets in SDV have some columns that are not present in the original datasets.
    We created this function to remove these columns from the tables and the metadata.
    """
    for table_name, table in tables.items():
        for column in table.columns:
            if any(
                prefix in column
                for prefix in ["add_numerical", "nb_rows_in", "min(", "max(", "sum("]
            ):
                table = table.drop(columns=column, axis=1)

                if not update_metadata:
                    continue
                metadata.tables[table_name].columns.pop(column)

        tables[table_name] = table
    if validate:
        metadata.validate()
        metadata.validate_data(tables)
    return tables, metadata


def save_tables(
    tables: Dict[str, pd.DataFrame],
    path: Union[str, os.PathLike],
    metadata: Optional[Metadata] = None,
):
    if not os.path.exists(path):
        os.makedirs(path)
    if metadata:
        metadata.save_to_json(os.path.join(path, "metadata.json"))
    for table_name, table in tables.items():
        if metadata:
            for col in table.columns:
                # if col in metadata is datetime, convert to string with datetime_format
                if metadata.tables[table_name].columns[col]["sdtype"] == "datetime":
                    datetime_format = (
                        metadata.tables[table_name].columns[col].get("datetime_format")
                    )
                    if datetime_format:
                        table[col] = table[col].dt.strftime(datetime_format)
        table.to_csv(os.path.join(path, f"{table_name}.csv"), index=False)


def download_sdv_relational_datasets(
    data_path: Union[str, os.PathLike] = "data/original"
):
    """
    Download all the available relational datasets from SDV
    https://docs.sdv.dev/sdv/single-table-data/data-preparation/loading-data
    """
    sdv_relational_datasets = get_available_demos("multi_table")

    # iterate through the dataframe
    for dataset_name in sdv_relational_datasets.dataset_name:
        print(f"Downloading {dataset_name}...", end=" ")
        download_demo(
            "multi_table",
            dataset_name,
            output_folder_name=f"{data_path}/{dataset_name}",
        )
        print("Done.")


def denormalize_tables(tables: Dict[str, pd.DataFrame], metadata: Metadata):
    relationships = metadata.relationships.copy()
    denormalized_table = tables[relationships[0]["parent_table_name"]]
    already_merged_tables = [relationships[0]["parent_table_name"]]

    while len(relationships) > 0:
        if relationships[0]["parent_table_name"] in already_merged_tables:
            parent_table = denormalized_table
            child_table = tables[relationships[0]["child_table_name"]]
            already_merged_tables.append(relationships[0]["child_table_name"])

        elif relationships[0]["child_table_name"] in already_merged_tables:
            parent_table = tables[relationships[0]["parent_table_name"]]
            child_table = denormalized_table
            already_merged_tables.append(relationships[0]["parent_table_name"])
        else:
            relationships.append(relationships.pop(0))
            continue

        denormalized_table = parent_table.merge(
            child_table,
            left_on=relationships[0]["parent_primary_key"],
            right_on=relationships[0]["child_foreign_key"],
            suffixes=(None, f"_{relationships[0]['child_table_name']}"),
            how="outer",
        )

        # Drop the foreign key column with suffix from the denormalized table
        for column, column_info in metadata.tables[
            relationships[0]["child_table_name"]
        ].columns.items():
            if column_info["sdtype"] != "id":
                continue
            denormalized_table = drop_column_if_in_table(
                denormalized_table, f"{column}_{relationships[0]['child_table_name']}"
            )

        relationships.pop(0)

    return denormalized_table


def drop_column_if_in_table(table: pd.DataFrame, column: str):
    if column in table.columns:
        table = table.drop(columns=column, axis=1)
    return table


def make_column_names_unique(
    real_data: Dict[str, pd.DataFrame],
    synthetic_data: Dict[str, pd.DataFrame],
    metadata: Metadata,
    validate=True,
):
    for table_name in metadata.get_tables():
        if not real_data[table_name].columns.equals(synthetic_data[table_name].columns):
            raise ValueError("Real and synthetic data column names are not the same")

        table_metadata = metadata.tables[table_name].to_dict()

        for column in table_metadata["columns"]:
            real_data[table_name] = real_data[table_name].rename(
                columns={column: f"{table_name}_{column}"}
            )
            synthetic_data[table_name] = synthetic_data[table_name].rename(
                columns={column: f"{table_name}_{column}"}
            )
            metadata = metadata.rename_column(
                table_name, column, f"{table_name}_{column}"
            )

    if validate:
        metadata.validate()
        metadata.validate_data(real_data)
        metadata.validate_data(synthetic_data)

    return real_data, synthetic_data, metadata
