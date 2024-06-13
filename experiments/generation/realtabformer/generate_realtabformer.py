import os
import sys
import glob
import logging
import argparse
from pathlib import Path


from realtabformer import REaLTabFormer
from syntherela.metadata import Metadata
from syntherela.data import load_tables, save_tables, remove_sdv_columns

MODEL_NAME = "REALTABFORMER"
BATCH_SIZE_PARENT = 1024
BATCH_SIZE_CHILD = 1

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="airbnb-simplified_subsampled")
args.add_argument("--real-data-path", type=str, default="data/original")
args.add_argument("--synthetic-data-path", type=str, default="data/synthetic")
args.add_argument("--full-sensitivity", type=bool, default=True)
args.add_argument("--retrain", type=bool, default=True)
args.add_argument("--run-id", type=str, default="1")
args = args.parse_args()

dataset_name = args.dataset_name
full_sensitivity = args.full_sensitivity
retrain = args.retrain
real_data_path = args.real_data_path
synthetic_data_path = args.synthetic_data_path
run_id = args.run_id

logger = logging.getLogger(f"{MODEL_NAME}_logger")

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"START LOGGING Dataset: {dataset_name} RUN {run_id}...")

os.environ["WANDB_PROJECT"] = f"{MODEL_NAME}_{dataset_name}_run_{run_id}"
os.environ["WANDB__SERVICE_WAIT"] = "300"

report_to = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


logger.info(f"START {MODEL_NAME}...")


logger.debug("Loading real data...")
metadata = Metadata().load_from_json(
    Path(real_data_path) / f"{dataset_name}/metadata.json"
)
real_data = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)
real_data, metadata = remove_sdv_columns(real_data, metadata)
metadata.validate_data(real_data)
logger.debug("Real data loaded")

synthetic_data = {}

added_keys = {}

for table in real_data:
    if metadata.get_primary_key(table) is None:
        logger.debug(f"Table {table} has no primary key. Adding one...")
        real_data[table].reset_index(inplace=True)
        added_keys[table] = "index"
        logger.debug(f"Primary key added to table {table}")

renamed_keys = {}
join_on_name = "join_on_RTF"
for relationship in metadata.relationships:
    parent_key = relationship["parent_primary_key"]
    child_key = relationship["child_foreign_key"]

    if parent_key == child_key:
        continue

    # rename child key in child table to parent key
    real_data[relationship["parent_table_name"]].rename(
        columns={parent_key: join_on_name}, inplace=True
    )
    real_data[relationship["child_table_name"]].rename(
        columns={child_key: join_on_name}, inplace=True
    )
    renamed_keys[relationship["child_table_name"]] = child_key
    renamed_keys[relationship["parent_table_name"]] = parent_key
    logger.debug(f"Renamed {child_key} and {parent_key} to {join_on_name}")

# GENERATE SYNTHETIC DATA ---------------------------------

root_table_name = metadata.get_root_tables()[0]
parent_df = real_data[root_table_name]

join_on = (
    metadata.get_primary_key(root_table_name)
    if len(renamed_keys.keys()) == 0
    else join_on_name
)
parent_model_path = (
    f"models/{MODEL_NAME}/{dataset_name}/checkpoint_{root_table_name}_run_{run_id}"
)

if retrain or not (
    os.path.exists(parent_model_path) and len(os.listdir(parent_model_path)) > 0
):
    logger.debug("Starting parent table generation...")
    logger.debug("Fitting parent model...")
    parent_model = REaLTabFormer(
        model_type="tabular",
        batch_size=BATCH_SIZE_PARENT,
        checkpoints_dir=f"checkpoints/{MODEL_NAME}/{dataset_name}/checkpoint_{root_table_name}_run_{run_id}",
        random_state=int(run_id),
        report_to=report_to,
    )
    os.makedirs(parent_model_path, exist_ok=True)
    # fit and save parent model
    parent_model.fit(parent_df.drop(join_on, axis=1), full_sensitivity=full_sensitivity)
    parent_model.save(parent_model_path)
    logger.debug("Parent fitted and saved")
else:
    logger.debug("Skipping parent table generation and reading last trained model...")

# load trained parent model
directories = list(filter(os.path.isdir, glob.glob(f"{parent_model_path}/id*")))
directories.sort(key=lambda x: os.path.getmtime(x))
parent_model_path = directories[-1]
parent_model = REaLTabFormer(
    model_type="tabular",
    parent_realtabformer_path=parent_model_path,
    batch_size=BATCH_SIZE_PARENT,
    checkpoints_dir=f"checkpoints/{MODEL_NAME}/{dataset_name}/checkpoint_{root_table_name}_run_{run_id}",
    random_state=int(run_id),
    report_to=report_to,
)

parent_model = parent_model.load_from_dir(parent_model_path)

for i in range(1, 4):
    # sample from parent model
    logger.debug(f"Sampling from parent model, table {root_table_name}...")
    parent_samples = parent_model.sample(real_data[root_table_name].shape[0])
    parent_samples.index.name = join_on
    parent_samples = parent_samples.reset_index()
    synthetic_data[root_table_name] = parent_samples
    logger.debug("Done.")

    # GENERATE CHILD TABLES
    for child_table_name in sorted(metadata.get_children(root_table_name)):
        logger.info(f"Starting child table generation for {child_table_name}...")
        child_model_path = f"models/{MODEL_NAME}/{dataset_name}/checkpoint_{child_table_name}_run_{run_id}"
        child_df = real_data[child_table_name]
        assert (join_on in parent_df.columns) and (join_on in child_df.columns)
        if retrain or not (
            os.path.exists(child_model_path) and len(os.listdir(child_model_path)) > 0
        ):
            child_model = REaLTabFormer(
                model_type="relational",
                parent_realtabformer_path=parent_model_path,
                output_max_length=None,
                train_size=0.95,
                batch_size=BATCH_SIZE_CHILD,
                checkpoints_dir=f"checkpoints/{MODEL_NAME}/{dataset_name}/checkpoint_{child_table_name}_run_{run_id}",
                random_state=int(run_id),
                report_to=report_to,
            )

            logger.debug(f"Fitting child model for table {child_table_name}...")
            # fit child model
            child_model.fit(df=child_df, in_df=parent_df, join_on=join_on)
            # save child model
            os.makedirs(child_model_path, exist_ok=True)
            child_model.save(child_model_path)
            logger.debug("Child fitted and saved")
        else:
            logger.debug(
                f"Skipping child table {child_table_name} generation and reading last trained model..."
            )

        directories = list(filter(os.path.isdir, glob.glob(f"{child_model_path}/id*")))
        directories.sort(key=lambda x: os.path.getmtime(x))
        child_model_path = directories[-1]
        child_model = REaLTabFormer(
            model_type="relational",
            parent_realtabformer_path=parent_model_path,
            output_max_length=None,
            train_size=0.95,
            batch_size=BATCH_SIZE_CHILD,
            checkpoints_dir=f"checkpoints/{MODEL_NAME}/{dataset_name}/checkpoint_{child_table_name}_run_{run_id}",
            random_state=int(run_id),
            report_to=report_to,
        )
        child_model = child_model.load_from_dir(child_model_path)

        logger.info("Starting sampling and saving synthetic data...")

        logger.debug(f"Sampling from child model, table {child_table_name}...")
        child_samples = child_model.sample(
            input_unique_ids=parent_samples[join_on],
            input_df=parent_samples.drop(join_on, axis=1),
            gen_batch=64,
        )

        child_samples.index.name = join_on
        child_samples = child_samples.reset_index()

        child_primary_key = (
            metadata.get_primary_key(child_table_name)
            if child_table_name not in added_keys
            else added_keys[child_table_name]
        )
        child_samples = child_samples.drop(columns=child_primary_key)
        child_samples.index.name = child_primary_key
        child_samples.reset_index(inplace=True)

        synthetic_data[child_table_name] = child_samples

        for table in synthetic_data:
            if table in added_keys:
                try:
                    synthetic_data[table].drop(columns=added_keys[table], inplace=True)
                except:
                    pass
            if table in renamed_keys:
                synthetic_data[table].rename(
                    columns={join_on_name: renamed_keys[table]}, inplace=True
                )

        logger.info("Saving synthetic data...")
        save_data_path = (
            Path(synthetic_data_path)
            / dataset_name
            / MODEL_NAME
            / run_id
            / f"sample{i}"
        )
        save_tables(synthetic_data, save_data_path)
        logger.debug("Done.")

logger.info("COMPLETE GENERATION DONE.")
