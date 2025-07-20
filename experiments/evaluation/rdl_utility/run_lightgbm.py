import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything
from tqdm import tqdm


from relbench.base import Dataset, TaskType, EntityTask, BaseTask, AutoCompleteTask
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task
from relbench.tasks.f1 import DriverPositionTask, DriverTop3Task, DriverDNFTask
from gnn_datasets import (
    RossmannDataset,
    WalmartDataset,
    F1Dataset,
    AirbnbDataset,
    BerkaDataset,
)

DATASETS = {
    RossmannDataset.name: RossmannDataset,
    WalmartDataset.name: WalmartDataset,
    F1Dataset.name: F1Dataset,
    AirbnbDataset.name: AirbnbDataset,
    BerkaDataset.name: BerkaDataset,
}

TASKS = {
    "driver-position": DriverPositionTask,
    "driver-top3": DriverTop3Task,
    "driver-dnft": DriverDNFTask,
    "predict-column": AutoCompleteTask,
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="predict-column")
parser.add_argument("--method", type=str, default="ORIGINAL")
parser.add_argument("--run_id", type=str, default="1")

parser.add_argument(
    "--task_type",
    type=str,
    default="BINARY_CLASSIFICATION",
    choices=["BINARY_CLASSIFICATION", "REGRESSION", "MULTILABEL_CLASSIFICATION"],
)

parser.add_argument("--dataset", type=str, default="airbnb-simplified_subsampled")
parser.add_argument("--entity_table", type=str, default="users")
parser.add_argument("--target_col", type=str, default="country_destination")


parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50_000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--left_join_fkey", action="store_true", default=False)
parser.add_argument(
    "--download",
    action="store_true",
    default=False,
    help="Download the dataset if not already present.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

predict_column_task_config = {
    "task_type": TaskType[args.task_type],
    "entity_table": args.entity_table,
    "target_col": args.target_col,
}

# dataset: Dataset = get_dataset(args.dataset, download=False)
dataset: Dataset = DATASETS[args.dataset](method=args.method, run_id=args.run_id)
dataset_test: Dataset = DATASETS[args.dataset](
    method=args.method, run_id=args.run_id, type="test"
)

# task = PredictColumnTask(dataset=dataset, **predict_column_task_config)
if args.task == "predict-column":
    dataset.target_col = args.target_col
    dataset.entity_table = args.entity_table
    dataset_test.target_col = args.target_col
    dataset_test.entity_table = args.entity_table
    task: AutoCompleteTask = TASKS[args.task](
        dataset=dataset, **predict_column_task_config
    )
    task_test: AutoCompleteTask = TASKS[args.task](
        dataset=dataset_test, **predict_column_task_config
    )
else:
    task: BaseTask = TASKS[args.task](dataset=dataset)
    # task_test: BaseTask = TASKS[args.task](dataset=dataset_test)
    task_test: EntityTask = get_task("rel-f1", args.task, download=False)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task_test.get_table("test")


dfs: Dict[str, pd.DataFrame] = {}
entity_table = dataset.get_db().table_dict[task.entity_table]
entity_df = entity_table.df

entity_table_test = dataset_test.get_db(upto_test_timestamp=False if args.task == "predict-column" else True).table_dict[task.entity_table]
entity_df_test = entity_table_test.df

stypes_cache_path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/stypes.json"
)
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        orig_columns = dataset.get_db().table_dict[table].df.columns
        # Collect keys to delete first to avoid modifying dict during iteration
        keys_to_delete = []
        for col, stype_str in col_to_stype.items():
            if col not in orig_columns:
                keys_to_delete.append(col)
                continue
            col_to_stype[col] = stype(stype_str)
        # Delete the keys after iteration
        for col in keys_to_delete:
            del col_to_stype[col]
except FileNotFoundError:
    raise ValueError(f"Stypes cache file not found for {args.dataset}. Please run the metadata_sdv_to_relbench.py script to generate the cache file.")
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

col_to_stype = col_to_stype_dict[task.entity_table]
remove_pkey_fkey(col_to_stype, entity_table)
remove_pkey_fkey(col_to_stype, entity_table_test)
for col in dataset.remove_columns:
    if col in col_to_stype:
        del col_to_stype[col]

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.categorical
elif task.task_type == TaskType.REGRESSION:
    col_to_stype[task.target_col] = torch_frame.numerical
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.embedding
elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.categorical
    # task.metrics = task.metrics[:1]  # NOTE: Probabilistic multiclass predictions
    # are not supported by torch_frame LightGBM to enable probabilities:
    #  install torch_frame from https://github.com/ValterH/pytorch-frame
else:
    raise ValueError(f"Unsupported task type called {task.task_type}")

# randomly subsample in case training data size is too large.
if args.sample_size > 0 and args.sample_size < len(train_table):
    sampled_idx = np.random.permutation(len(train_table))[: args.sample_size]
    train_table.df = train_table.df.iloc[sampled_idx]

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    if split == "test":
        entity_df = entity_df_test
        entity_table = entity_table_test

    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
    entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})
    # Remove duplicated columns from entity_df that are already in the table df
    for col in set(entity_df.columns).intersection(set(table.df.columns)):
        if col != entity_table.pkey_col:
            entity_df.pop(col)
    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
    )
    if args.left_join_fkey:
        for fkey_col, pkey_table_name in entity_table.fkey_col_to_pkey_table.items():
            pkey_table = dataset.get_db().table_dict[pkey_table_name]
            dfs[split] = dfs[split].merge(
                pkey_table.df,
                how="left",
                left_on=fkey_col,
                right_on=pkey_table.pkey_col,
                suffixes=("", f"_{pkey_table_name}"),
            )
            pkey_col_to_stype = col_to_stype_dict[pkey_table_name]
            # add appropriate stypes and remove id columns
            for col, stype_str in pkey_col_to_stype.items():
                if col == pkey_table.pkey_col:
                    # remove pkey column
                    if f"{col}_{pkey_table_name}" in dfs[split].columns:
                        dfs[split].pop(f"{col}_{pkey_table_name}")
                    elif col in dfs[split].columns:
                        dfs[split].pop(col)
                elif col in pkey_table.fkey_col_to_pkey_table:
                    # remove fkey columns
                    if col in dfs[split].columns:
                        dfs[split].pop(col)
                    elif f"{col}_{pkey_table_name}" in dfs[split].columns:
                        dfs[split].pop(f"{col}_{pkey_table_name}")
                elif col not in col_to_stype:
                    # add stype for the column
                    col_to_stype[col] = stype(stype_str)
                elif (
                    f"{col}_{pkey_table_name}" not in col_to_stype
                    and f"{col}_{pkey_table_name}" in dfs[split].columns
                ):
                    # add stype for the column with suffix to avoid name collision
                    col_to_stype[f"{col}_{pkey_table_name}"] = stype(stype_str)


train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    # col_to_text_embedder_cfg=TextEmbedderConfig(
    #     text_embedder=GloveTextEmbedding(device=device),
    #     batch_size=256,
    # ),
)
path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/{args.method}/{args.run_id}/node_train{'_join' if args.left_join_fkey else ''}.pt"
)
path.parent.mkdir(parents=True, exist_ok=True)
train_dataset = train_dataset.materialize(path=path)

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

if task.task_type in [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
]:
    tune_metric = Metric.ROCAUC
elif task.task_type == TaskType.REGRESSION:
    tune_metric = Metric.MAE
elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    tune_metric = Metric.ACCURACY
else:
    raise ValueError(f"Task task type is unsupported {task.task_type}")

if task.task_type in [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.REGRESSION,
    TaskType.MULTICLASS_CLASSIFICATION,
]:
    model = LightGBM(
        task_type=train_dataset.task_type,
        metric=tune_metric,
        num_classes=(
            task.num_classes
            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION
            else None
        ),
    )
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)

    pred = model.predict(tf_test=tf_train).numpy()
    train_metrics = task.evaluate(pred, train_table)

    pred = model.predict(tf_test=tf_val).numpy()
    val_metrics = task.evaluate(pred, val_table)

    pred = model.predict(tf_test=tf_test).numpy()
    test_metrics = task_test.evaluate(pred)
else:
    raise ValueError(f"Task task type is unsupported {task.task_type}")

print(f"Train: {train_metrics}")
print(f"Val: {val_metrics}")
print(f"Test: {test_metrics}")