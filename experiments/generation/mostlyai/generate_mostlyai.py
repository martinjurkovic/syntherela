import argparse
from pathlib import Path
from time import sleep

import pandas as pd
from mostlyai.sdk import MostlyAI
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns, save_tables


def create_config_from_metadata(
    metadata: Metadata, tables: dict[pd.DataFrame], name: str
) -> dict:
    config = {"name": name, "tables": []}
    for table in metadata.get_tables():
        data = tables[table]
        primary_key = metadata.get_primary_key(table)
        table_config = {
            "name": table,
            "data": data,
            "primary_key": primary_key,
            "columns": [],
            "modelConfiguration": {
                "model": "MOSTLY_AI/Large",
                "maxSampleSize": None,
                "batchSize": None,
                "maxTrainingTime": 120,
                "maxEpochs": 100,
                "maxSequenceWindow": 100,
                "enableFlexibleGeneration": False,
                "valueProtection": False,
                "rareCategoryReplacementMethod": "CONSTANT",
                "differentialPrivacy": None,
            },
        }
        table_meta = metadata.get_table_metadata(table)
        for column, column_info in table_meta.columns.items():
            column_type = column_info["sdtype"]
            if column_type == "id":
                encoding = "AUTO"
            elif column_type == "categorical" or column_type == "boolean":
                encoding = "TABULAR_CATEGORICAL"
            elif column_type == "numerical":
                encoding = "TABULAR_NUMERIC_AUTO"
            elif column_type == "datetime":
                encoding = "TABULAR_DATETIME"
            else:
                raise ValueError(
                    f"Unsupported column type: {column_type} for column {column}"
                )
            column_config = {"name": column, "model_encoding_type": encoding}
            table_config["columns"].append(column_config)

        is_context = True
        for parent in metadata.get_parents(table):
            table_config.setdefault("foreign_keys", [])
            for foreign_key in metadata.get_foreign_keys(parent, table):
                fk_dict = {
                    "column": foreign_key,
                    "referenced_table": parent,
                    "is_context": is_context,
                }
                table_config["foreign_keys"].append(fk_dict)
                # Use only the first foreign key of the first parent as context
                is_context = False
        config["tables"].append(table_config)
    return config


def postprocess_data(synthetic_data, metadata):
    for table in metadata.get_tables():
        table_meta = metadata.get_table_metadata(table)
        for column, column_info in table_meta.columns.items():
            column_type = column_info["sdtype"]
            if column_type == "datetime":
                synthetic_data[table][column] = pd.to_datetime(
                    synthetic_data[table][column]
                )
            elif column_type == "boolean":
                synthetic_data[table][column] = synthetic_data[table][column] == "True"
    return synthetic_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset-name", type=str, default="airbnb-simplified_subsampled"
    )
    args.add_argument("--real-data-path", type=str, default="data/original")
    args.add_argument("--synthetic-data-path", type=str, default="data/synthetic")
    args.add_argument("--api-key", type=str, required=True)
    args.add_argument("--run-id", type=str, default="1")
    args.add_argument("--generator-id", type=str)
    args.add_argument("--num-samples", type=int, default=3)
    args.add_argument("--sleep-for", type=int, default=3)
    args = args.parse_args()

    dataset_name = args.dataset_name
    real_data_path = args.real_data_path
    synthetic_data_path = args.synthetic_data_path
    api_key = args.api_key
    run_id = args.run_id
    generator_id = args.generator_id
    sleep_for = args.sleep_for
    num_samples = args.num_samples

    # load the data
    metadata = Metadata().load_from_json(
        Path(real_data_path) / f"{dataset_name}/metadata.json"
    )
    real_data = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)
    real_data, metadata = remove_sdv_columns(real_data, metadata)

    # create MOSTLY AI configuration
    config = create_config_from_metadata(
        metadata, tables=real_data, name=f"{dataset_name} - {run_id}"
    )

    # Train the model
    mostly = MostlyAI(api_key=api_key)
    if generator_id:
        # Load the trained generator
        g = mostly.generators.get(generator_id=generator_id)
    else:
        # Train a new generator
        g = mostly.train(config=config)

    # Sample the model three times
    for sample in range(num_samples):
        sample_id = str(sample + 1)
        sd = mostly.generate(
            g, config=config, name=f"{dataset_name} - {run_id} - {sample_id}"
        )
        sleep(sleep_for)
        synthetic_data = sd.data()
        synthetic_data = postprocess_data(synthetic_data, metadata)
        # Ensure the data follows the metadata
        metadata.validate_data(synthetic_data)
        path_synthetic = (
            f"{synthetic_data_path}/{dataset_name}/MOSTLYAI/{run_id}/sample{sample_id}"
        )
        print(f"Saving sample {sample_id}")
        save_tables(synthetic_data, path=Path(path_synthetic))
