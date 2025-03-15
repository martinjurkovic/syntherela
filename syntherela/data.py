"""Data handling utilities for synthetic data evaluation.

This module provides functions for loading, preprocessing, and manipulating
real and synthetic data for evaluation purposes.
"""

import os
from typing import Optional, Union
from syntherela.typing import Tables

import pandas as pd
from sdv.datasets.demo import get_available_demos, download_demo
from syntherela.metadata import Metadata


def get_dataset_stats(tables: Tables, metadata: Metadata) -> dict:
    """Get statistics about the dataset.

    Parameters
    ----------
    tables: Tables
        Dictionary mapping table names to pandas DataFrames.
    metadata: Metadata
        Metadata object containing information about the tables.

    Returns
    -------
    dict
        Dictionary containing statistics about the dataset:
        - num_tables: Number of tables in the dataset.
        - num_rows: Total number of rows across all tables.
        - num_columns: Total number of non-id columns across all tables.
        - num_relationships: Number of relationships between tables.

    """
    total_rows = 0
    total_columns = 0
    for table_name, table in tables.items():
        total_rows += len(table)
        id_columns = metadata.get_column_names(table_name, sdtype="id")
        total_columns += len(table.columns) - len(id_columns)

    return {
        "num_tables": len(tables),
        "num_rows": total_rows,
        "num_columns": total_columns,
        "num_relationships": len(metadata.relationships),
    }


def load_tables(data_path: Union[str, os.PathLike], metadata: Metadata):
    """Load tables from CSV files based on metadata.

    Parameters
    ----------
    data_path: Union[str, os.PathLike]
        Path to the directory containing CSV files.
    metadata: Metadata
        Metadata object containing information about the tables.

    Returns
    -------
    Tables
        Dictionary mapping table names to pandas DataFrames.

    Raises
    ------
    ValueError
        If datetime_format is not found in metadata for a datetime column.

    """
    tables: Tables = {}
    table_names = metadata.get_tables()
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".csv"):
            continue
        table_name = file_name.split(".")[0]
        if table_name not in table_names:
            continue
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
        for column, format in datetime_formats.items():
            # If pandas can't parse the datetime format set it manually,
            # if the format is correct, this will not change the column.
            table[column] = pd.to_datetime(table[column], format="ISO8601").dt.strftime(
                format
            )
            table[column] = pd.to_datetime(table[column], format=format)

        tables[table_name] = table
    return tables


# FIXME: This function should be removed
def remove_sdv_columns(
    tables: Tables,
    metadata: Metadata,
    update_metadata=True,
    validate=True,
):
    """Remove SDV-specific columns from tables.

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
    tables: Tables,
    path: Union[str, os.PathLike],
    metadata: Optional[Metadata] = None,
    save_metadata: bool = False,
):
    """Save tables to CSV files.

    Parameters
    ----------
    tables: Tables
        Dictionary mapping table names to pandas DataFrames.
    path: Union[str, os.PathLike]
        Path to the directory where CSV files will be saved.
    metadata: Optional[Metadata], default=None
        Optional metadata object. If provided and save_metadata is True,
        metadata will be saved as well.
    save_metadata: bool, default=False
        Whether to save metadata.

    """
    if not os.path.exists(path):
        os.makedirs(path)
    if metadata and save_metadata:
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
                        # If the column is already a string, convert it to datetime first
                        # to ensure the datetime_format is applied correctly.
                        table[col] = pd.to_datetime(table[col]).dt.strftime(
                            datetime_format
                        )
        table.to_csv(os.path.join(path, f"{table_name}.csv"), index=False)


def download_sdv_relational_datasets(
    data_path: Union[str, os.PathLike] = "data/original",
):
    """Download SDV relational datasets.

    The datasets are available at https://docs.sdv.dev/sdv/single-table-data/data-preparation/loading-data.

    Parameters
    ----------
    data_path: Union[str, os.PathLike], default="data/original"
        Path to the directory where datasets will be saved.

    Returns
    -------
    list
        List of downloaded dataset names.

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


def denormalize_tables(tables: Tables, metadata: Metadata):
    """Denormalize tables by joining them based on relationships in metadata.

    Parameters
    ----------
    tables: Tables
        Dictionary mapping table names to pandas DataFrames.
    metadata: Metadata
        Metadata object containing information about the tables and their relationships.

    Returns
    -------
    pd.DataFrame
        Denormalized table containing all data.

    """
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
    """Drop a column from a table if it exists.

    Parameters
    ----------
    table: pd.DataFrame
        pandas DataFrame.
    column: str
        Name of the column to drop.

    Returns
    -------
    pd.DataFrame
        Table with the column dropped if it existed.

    """
    if column in table.columns:
        table = table.drop(columns=column, axis=1)
    return table


def make_column_names_unique(
    real_data: Tables,
    synthetic_data: Tables,
    metadata: Metadata,
    validate=True,
):
    """Make column names unique across all tables.

    Parameters
    ----------
    real_data: Tables
        Dictionary mapping table names to pandas DataFrames for real data.
    synthetic_data: Tables
        Dictionary mapping table names to pandas DataFrames for synthetic data.
    metadata: Metadata
        Metadata object containing information about the tables.
    validate: bool, default=True
        Whether to validate the tables after making column names unique.

    Returns
    -------
    tuple
        Tuple containing:
        - real_data: Dictionary mapping table names to pandas DataFrames with unique column names.
        - synthetic_data: Dictionary mapping table names to pandas DataFrames with unique column names.
        - metadata: Updated metadata object with unique column names.

    """
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
