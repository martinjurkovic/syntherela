import os
import sys
import pickle
import logging
import argparse
from pathlib import Path


import pandas as pd
from rctgan import Metadata
from rctgan.relational import RCTGAN


MODEL_NAME = "RCTGAN"

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="airbnb-simplified_subsampled")
args.add_argument("--real-data-path", type=str, default="data/original")
args.add_argument("--synthetic-data-path", type=str, default="data/synthetic")
args.add_argument("--model-save-path", type=str, default="checkpoints")
args.add_argument("--run-id", type=str, default="1")
args = args.parse_args()

dataset_name = args.dataset_name
real_data_path = args.real_data_path
synthetic_data_path = args.synthetic_data_path
model_save_path = args.model_save_path
run_id = args.run_id

logger = logging.getLogger(f"{MODEL_NAME}_logger")

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"START LOGGING RUN {run_id}...")


logger.info(f"START {MODEL_NAME}...")


# utils
def load_tables(data_path, metadata):
    tables = {}
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".csv"):
            continue
        table_name = file_name.split(".")[0]
        dtypes = {}
        parse_dates = []
        for column, column_info in metadata.to_dict()["tables"][table_name][
            "fields"
        ].items():
            if column_info["type"] == "categorical":
                dtypes[column] = "object"
            elif column_info["type"] == "boolean":
                dtypes[column] = "bool"
            elif column_info["type"] == "datetime":
                parse_dates.append(column)
            # for ids and numerical values let pandas infer the type
        table = pd.read_csv(
            f"{data_path}/{file_name}",
            low_memory=False,
            dtype=dtypes,
            parse_dates=parse_dates,
        )
        tables[table_name] = table
    return tables


def remove_sdv_columns(tables, metadata, update_metadata=True):
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
                metadata._metadata["tables"][table_name]["fields"].pop(column)

        tables[table_name] = table
    metadata.validate(tables)
    return tables, metadata


def save_tables(tables, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for table_name, table in tables.items():
        table.to_csv(os.path.join(path, f"{table_name}.csv"), index=False)


# GENERATE SYNTHETIC DATA ---------------------------------
logger.debug(f"Loading real data... for {dataset_name}")
metadata = Metadata(
    metadata=str(Path(real_data_path) / f"{dataset_name}/metadata_v0.json")
)
real_data = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)
real_data, metadata = remove_sdv_columns(real_data, metadata)
logger.debug("Real data loaded")

synthetic_data = {}

model = RCTGAN(metadata=metadata)
logger.info("Fitting model...")
model.seed = int(run_id)
model.fit(real_data)

# SAVE MODEL CHECKPOINT -----------------------------------
logger.debug("Saving model checkpoint...")
model_path = os.path.join(model_save_path, MODEL_NAME, dataset_name)
os.makedirs(model_path, exist_ok=True)
model_path = os.path.join(model_path, f"model_{run_id}.pkl")
pickle.dump(model, open(model_path, "wb"))

# SAMPLE AND SAVE DATA ------------------------------------
logger.info("Sampling and saving synthetic data...")
for i in range(1, 4):
    model.seed = i
    synthetic_data = model.sample()
    save_data_path = (
        Path(synthetic_data_path) / dataset_name / MODEL_NAME / run_id / f"sample{i}"
    )
    save_tables(synthetic_data, save_data_path)


logger.info("COMPLETE GENERATION DONE.")
