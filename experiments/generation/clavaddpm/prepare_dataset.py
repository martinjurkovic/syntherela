import os
import json
import argparse
from pathlib import Path

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

from ClavaDDPM.preprocess_utils import encode_and_save, calculate_days_since_earliest_date


def remap_ids(metadata, tables):
    # Transform the ids to 0, 1, 2, ...
    id_map = {}

    for parent_table_name in metadata.get_tables():
        primary_key = f"{parent_table_name}_id"

        if parent_table_name not in id_map:
            id_map[parent_table_name] = {}

        if primary_key not in id_map[parent_table_name]:
            id_map[parent_table_name][primary_key] = {}
            idx = 0
            for primary_key_val in tables[parent_table_name][primary_key].unique():
                id_map[parent_table_name][primary_key][primary_key_val] = idx
                idx += 1

        for relationship in metadata.relationships:
            if relationship["parent_table_name"] != parent_table_name:
                continue
            if relationship["child_table_name"] not in id_map:
                id_map[relationship["child_table_name"]] = {}
            fk = f"{parent_table_name}_id"
            id_map[relationship["child_table_name"]][fk] = id_map[parent_table_name][fk]

    # remap the ids
    for table_name in id_map.keys():
        for column_name in id_map[table_name].keys():
            if column_name not in tables[table_name].columns:
                raise ValueError(
                    f"Column {column_name} not found in table {table_name}"
                )
            tables[table_name][column_name] = tables[table_name][column_name].map(
                id_map[table_name][column_name]
            )

    return tables


def rename_ids(metadata, tables):
    non_integer_ids = False
    ids = dict()
    for table_name in metadata.get_tables():
        ids[table_name] = []
        primary_key = metadata.get_primary_key(table_name)
        if primary_key is not None:
            tables[table_name][f"{table_name}_id"] = tables[table_name][primary_key]
            pk = tables[table_name].pop(primary_key)
            ids[table_name].append(f"{table_name}_id")
            if pk.dtype != "int64":
                non_integer_ids = True
        else:
            tables[table_name][f"{table_name}_id"] = range(len(tables[table_name]))
            ids[table_name].append(f"{table_name}_id")
        for parent in metadata.get_parents(table_name):
            foreign_keys = metadata.get_foreign_keys(parent, table_name)
            assert (
                len(foreign_keys) == 1
            ), "CLAVADDPM only supports one foreign key per table pair."
            foreign_key = foreign_keys[0]
            tables[table_name][f"{parent}_id"] = tables[table_name][foreign_key]
            fk = tables[table_name].pop(foreign_key)
            ids[table_name].append(f"{parent}_id")
            if fk.dtype != "int64":
                non_integer_ids = True

    if non_integer_ids:
        tables = remap_ids(metadata, tables)
    return tables, ids


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=50,
        help="Number of clusters to use for clustering.",
    )
    parser.add_argument(
        "--real-data-path",
        type=str,
        default="data/original/",
        help="Path to the original data.",
    )
    parser.add_argument("--run-id", type=str, default="1")

    return parser.parse_args()


def main(args):
    num_clusters = args.num_clusters
    dataset_name = args.dataset_name
    real_data_path = args.real_data_path
    run_id = 1
    
    metadata = Metadata().load_from_json(
        Path(real_data_path) / f"{dataset_name}/metadata.json"
    )
    real_data = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)
    real_data, metadata = remove_sdv_columns(real_data, metadata)

    processed_data_path = os.path.join("complex_data", dataset_name)
    os.makedirs(processed_data_path, exist_ok=True)

    tables = dict()
    relation_order = []
    for table in metadata.get_tables():
        parents = metadata.get_parents(table)
        children = metadata.get_children(table)
        tables[table] = {
            "children": list(children),
            "parents": list(parents),
        }
        if not len(parents):
            relation_order.append([None, table])

    for relation in metadata.relationships:
        parent = relation["parent_table_name"]
        child = relation["child_table_name"]
        relation_order.append([parent, child])

    dataset_meta = {
        "relation_order": relation_order,
        "tables": tables,
    }

    # Create a dataset_meta.json, in which tables should be manually created to
    #    specify all foreign key relationships in a multi-table dataset.
    with open(os.path.join(processed_data_path, "dataset_meta.json"), "w") as f:
        json.dump(dataset_meta, f, indent=4)

    first_dates = dict()

    # Save all tables as .csv files. All id columns should be named <column_name>_id.
    real_data, ids = rename_ids(metadata, real_data)
    for table_name, df in real_data.items():
        id_columns = ids[table_name]
        categorical_columns = metadata.get_column_names(
            table_name, sdtype="categorical"
        )
        datetime_columns = metadata.get_column_names(table_name, sdtype="datetime")
        numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")

        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].mean())

        # Calculate days since the earliest date for each datetime column
        if len(datetime_columns) > 0:
            first_dates[table_name] = dict()
        for col in datetime_columns:
            # TODO: handle missing values in datetime columns
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            df[col], first_dates[table_name][col] = calculate_days_since_earliest_date(
                df[col].dt.strftime("%y%m%d")
            )

        # Create a domain file for each table, id columns excluded.
        encode_and_save(
            df,
            categorical_columns,
            id_columns,
            processed_data_path,
            table_name,
        )

    if len(first_dates) > 0:
        with open(os.path.join(processed_data_path, "first_dates.json"), "w") as f:
            json.dump(first_dates, f, indent=4)

    # create config.json
    config = {
        "general": {
            "data_dir": processed_data_path.replace("\\", "/"),
            "exp_name": f"{dataset_name}_train",
            "workspace_dir": f"clavaDDPM_workspace_run{run_id}/{dataset_name}",
            "sample_prefix": "",
            "test_data_dir": None,
        },
        "clustering": {
            "parent_scale": 1.0,
            "num_clusters": num_clusters,
            "clustering_method": "both",
        },
        "diffusion": {
            "d_layers": [512, 1024, 1024, 1024, 1024, 512],
            "dropout": 0.0,
            "num_timesteps": 2000,
            "model_type": "mlp",
            "iterations": 200000,
            "batch_size": 4096,
            "lr": 6e-4,
            "gaussian_loss_type": "mse",
            "weight_decay": 1e-5,
            "scheduler": "cosine",
        },
        "classifier": {
            "d_layers": [128, 256, 512, 1024, 512, 256, 128],
            "lr": 0.0001,
            "dim_t": 128,
            "batch_size": 4096,
            "iterations": 20000,
        },
        "sampling": {"batch_size": 20000, "classifier_scale": 1.0},
        "matching": {
            "num_matching_clusters": 1,
            "matching_batch_size": 1000,
            "unique_matching": True,
            "no_matching": False,
        },
    }

    with open(
        os.path.join(
            "ClaVaDDPM",
            "configs",
            f"{dataset_name}_run{run_id}.json",
        ),
        "w",
    ) as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
