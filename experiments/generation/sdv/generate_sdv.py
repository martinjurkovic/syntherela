import os
import sys
import logging
import argparse
from pathlib import Path


from sdv.multi_table import HMASynthesizer
from syntherela.metadata import Metadata
from syntherela.data import load_tables, save_tables, remove_sdv_columns

MODEL_NAME = "SDV"

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

logger = logging.getLogger(f'{MODEL_NAME}_logger')

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"START LOGGING DATASET {dataset_name} RUN {run_id}...")


logger.debug("Loading real data...")
metadata = Metadata().load_from_json(Path(real_data_path) / f'{dataset_name}/metadata.json')

if dataset_name == 'Biodegradability_v1':
    metadata.update_column('bond', 'type', sdtype='numerical', computer_representation = "Int64")

real_data = load_tables(Path(real_data_path) / f'{dataset_name}', metadata)
real_data, metadata = remove_sdv_columns(real_data, metadata)
metadata.validate_data(real_data)
logger.debug("Real data loaded")



synthetic_data = {}

# GENERATE SYNTHETIC DATA ---------------------------------
model = HMASynthesizer(metadata=metadata)
model.fit(real_data)

# SAVE MODEL CHECKPOINT -----------------------------------
logger.debug("Saving model checkpoint...")
model_path = os.path.join(model_save_path, MODEL_NAME, dataset_name)
os.makedirs(model_path, exist_ok=True)
model_path = os.path.join(model_path, f"model_{run_id}.pkl")
model.save(model_path)

# SAMPLE AND SAVE DATA ------------------------------------
logger.info("Sampling and saving synthetic data...")
for i in range(1, 4):
    logger.debug(f"Sampling sample {i}")
    model.seed = i
    synthetic_data = model.sample()
    save_data_path = Path(synthetic_data_path) / dataset_name / MODEL_NAME / run_id / f"sample{i}"
    save_tables(synthetic_data, save_data_path)
    logger.debug(f"Done! Sample {i} saved!")
    

logger.info("COMPLETE GENERATION DONE.")

