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
from torch_frame.gbdt import LightGBM, XGBoost
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import featuretools as ft
from torch_frame.utils import infer_df_stype


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
    "autocomplete": AutoCompleteTask,
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="autocomplete")
parser.add_argument("--method", type=str, default="ORIGINAL")
parser.add_argument("--run_id", type=str, default="1")

parser.add_argument(
    "--task_type",
    type=str,
    default="REGRESSION",
    choices=["BINARY_CLASSIFICATION", "REGRESSION", "MULTILABEL_CLASSIFICATION"],
)

parser.add_argument("--dataset", type=str, default="walmart_subsampled")
parser.add_argument("--entity_table", type=str, default="depts")
parser.add_argument("--target_col", type=str, default="Weekly_Sales")


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
if args.task == "autocomplete":
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
test_table = task_test.get_table("test", mask_input_cols=False)


dfs: Dict[str, pd.DataFrame] = {}
# entity_table = dataset.get_db().table_dict[task.entity_table]
# entity_df = entity_table.df

# entity_table_test = dataset_test.get_db(upto_test_timestamp=False if args.task == "autocomplete" else True).table_dict[task.entity_table]
# entity_df_test = entity_table_test.df

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")

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
# remove_pkey_fkey(col_to_stype, entity_table)
# remove_pkey_fkey(col_to_stype, entity_table_test)
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
    # are not supported by torch_frame LightGBM
else:
    raise ValueError(f"Unsupported task type called {task.task_type}")

DROPPED_COLS = False

for split, table in [
    ("test", test_table),
    ("train", train_table),
    ("val", val_table),
]:
    print(f"\n=== Processing {split} split ===")

    db = None
    # Get database for this split
    if split == "test":
        db = dataset_test.get_db(upto_test_timestamp=False if args.task == "autocomplete" else True)
    else:
        db = dataset.get_db()

    # Create EntitySet for DFS
    es = None
    es = ft.EntitySet(id=f"{split}_entityset")

    print(f"Adding tables to EntitySet...")
    for table_name, table_obj in db.table_dict.items():
        df = table_obj.df.copy()

        # Remove duplicates if any
        if table_obj.pkey_col and table_obj.pkey_col in df.columns:
            before_len = len(df)
            df = df.drop_duplicates(subset=[table_obj.pkey_col])
            if len(df) < before_len:
                print(f"  Removed {before_len - len(df)} duplicates from {table_name}")

        # Create logical types mapping
        logical_types = {}
        if table_name in col_to_stype_dict:
            col_to_stype = col_to_stype_dict[table_name]

            for col in df.columns:
                if col in col_to_stype:
                    stype_val = col_to_stype[col]
                    # Map torch_frame stypes to featuretools logical types
                    if stype_val == torch_frame.categorical:
                        logical_types[col] = 'categorical'
                    elif stype_val == torch_frame.numerical:
                        if df[col].dtype in ['int64', 'int32']:
                            logical_types[col] = 'integer'
                        else:
                            logical_types[col] = 'double'
                    elif stype_val == torch_frame.embedding:
                        logical_types[col] = 'categorical'
                    elif stype_val == torch_frame.timestamp:
                        logical_types[col] = 'datetime'
                    else:
                        logical_types[col] = 'categorical'
                else:
                    # Fallback for columns not in stypes
                    if df[col].dtype == 'object':
                        logical_types[col] = 'categorical'
                    elif df[col].dtype == 'bool':
                        logical_types[col] = 'boolean'
                    elif 'datetime' in str(df[col].dtype):
                        logical_types[col] = 'datetime'
                    elif df[col].dtype in ['int64', 'int32']:
                        logical_types[col] = 'integer'
                    elif df[col].dtype in ['float64', 'float32']:
                        logical_types[col] = 'double'
                    else:
                        logical_types[col] = 'categorical'

        try:
            # Handle primary key
            if table_obj.pkey_col is None:
                # Create artificial primary key
                df = df.reset_index(drop=True)
                artificial_pkey = f"{table_name}_id"
                df[artificial_pkey] = range(len(df))
                logical_types[artificial_pkey] = 'integer'
                pkey_to_use = artificial_pkey
                print(f"  Created artificial primary key for {table_name}: {artificial_pkey}")
            else:
                pkey_to_use = table_obj.pkey_col

            if pkey_to_use not in df.columns:
                print(f"  ✗ Skipped {table_name}: primary key '{pkey_to_use}' not found")
                continue

            if df[pkey_to_use].nunique() != len(df):
                print(f"  ✗ Skipped {table_name}: primary key '{pkey_to_use}' not unique")
                continue

            es = es.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=pkey_to_use,
                logical_types=logical_types
            )
            print(f"  ✓ Added {table_name}: {len(df)} rows, pkey='{pkey_to_use}'")

        except Exception as e:
            print(f"  ✗ Failed to add {table_name}: {e}")
            continue

    # Add relationships
    print(f"Adding relationships...")
    relationships_added = 0
    for table_name, table_obj in db.table_dict.items():
        if hasattr(table_obj, 'fkey_col_to_pkey_table') and table_obj.fkey_col_to_pkey_table:
            for fkey_col, parent_table in table_obj.fkey_col_to_pkey_table.items():
                if parent_table in es.dataframe_dict and table_name in es.dataframe_dict:
                    try:
                        parent_obj = db.table_dict[parent_table]
                        parent_pkey = parent_obj.pkey_col if parent_obj.pkey_col else f"{parent_table}_id"

                        es = es.add_relationship(
                            parent_dataframe_name=parent_table,
                            child_dataframe_name=table_name,
                            parent_column_name=parent_pkey,
                            child_column_name=fkey_col
                        )
                        print(f"  ✓ {parent_table}.{parent_pkey} -> {table_name}.{fkey_col}")
                        relationships_added += 1
                    except Exception as e:
                        print(f"  ✗ Failed relationship {parent_table} -> {table_name}: {e}")

    print(f"EntitySet created with {relationships_added} relationships")

    # Run DFS to create features
    # if split == "test":
    print(f"Running DFS on target table: {task.entity_table}")
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name=task.entity_table,
        agg_primitives=["count", "mean", "sum"],
        trans_primitives=["month", "year"],
        max_depth=2,
        features_only=False,
        verbose=True
    )
    #     print(f"Generated {len(feature_defs)} features for training")
    # else:
    #     print(f"Applying training features to {split} data")
    #     feature_matrix = ft.calculate_feature_matrix(
    #         features=feature_defs,
    #         entityset=es,
    #         verbose=True
    #     )

    print(f"{split} feature matrix shape: {feature_matrix.shape}")

    # Join task table with DFS features (following the pattern from run_lightgbm.py)
    # if split == "test":
    #     entity_df = entity_df_test
    #     entity_table = entity_table_test
    # else:
    entity_df = db.table_dict[task.entity_table].df
    entity_table = db.table_dict[task.entity_table]

    # Get foreign key column name
    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]

    # Ensure dtype compatibility between entity table primary key and task table foreign key
    entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})

    # Use DFS features instead of raw entity_df
    # Reset index to make entity IDs a column for joining
    dfs_features = feature_matrix.copy()
    # dfs_features[entity_table.pkey_col] = entity_df[entity_table.pkey_col].iloc[dfs_features.index].values
    dfs_features = dfs_features.reset_index()

    # Remove duplicated columns from DFS features that are already in the task table
    for col in set(dfs_features.columns).intersection(set(table.df.columns)):
        if col != entity_table.pkey_col:
            dfs_features = dfs_features.drop(columns=[col])

    # Join task table with DFS features
    merged_df = table.df.merge(
        dfs_features,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
    )
    # Drop rows where categorical columns are NaN
    if args.method == "RGCLD":
        categorical_cols = []
        merged_df_len = len(merged_df)
        for col in merged_df.columns:
            dtype = merged_df[col].dtype
            if (pd.api.types.is_object_dtype(dtype) or 
                pd.api.types.is_string_dtype(dtype) or 
                pd.api.types.is_bool_dtype(dtype) or 
                isinstance(dtype, pd.CategoricalDtype)):
                categorical_cols.append(col)
        
        
        # Drop rows where categorical columns are NaN
        if split in ["train", "val"]:
            merged_df = merged_df.dropna(subset=categorical_cols)
            if len(merged_df) < merged_df_len:
                print(f"Dropped {merged_df_len - len(merged_df)} rows with NaN categorical values")
                DROPPED_COLS = True

    print(f"Joined {split} data: task table {table.df.shape} + DFS features -> {merged_df.shape}")

    # Store the merged result
    dfs[split] = merged_df
    print(f"Stored {split} feature matrix: {merged_df.shape}")

# Convert dtypes from dfs features back to stype dict for proper torch_frame handling
col_to_stype = {}
for col in dfs["train"].columns:
    if col == task.target_col:
        # set target column stype based on task type
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            col_to_stype[col] = stype.categorical
        elif task.task_type == TaskType.REGRESSION:
            col_to_stype[col] = stype.numerical
        elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            col_to_stype[col] = stype.embedding
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            col_to_stype[col] = stype.categorical
        else:
            raise ValueError(f"Unsupported task type called {task.task_type}")
        continue
    
    # Get the pandas dtype
    dtype = dfs["train"][col].dtype
    
    # Map pandas dtypes to torch_frame stypes
    if pd.api.types.is_integer_dtype(dtype):
        col_to_stype[col] = stype.numerical
    elif pd.api.types.is_float_dtype(dtype):
        col_to_stype[col] = stype.numerical
    elif pd.api.types.is_bool_dtype(dtype):
        col_to_stype[col] = stype.categorical
    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        # All text fields are categorical
        col_to_stype[col] = stype.categorical
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        col_to_stype[col] = stype.timestamp
    elif isinstance(dtype, pd.CategoricalDtype):
        # Handle pandas categorical dtype
        col_to_stype[col] = stype.categorical
    else:
        print(f"Unknown dtype for column {col}: {dtype}")
        # Default to categorical for unknown types
        col_to_stype[col] = stype.categorical

print(f"Mapped {len(col_to_stype)} columns to stypes for DFS features")



train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    # col_to_text_embedder_cfg=TextEmbedderConfig(
    #     text_embedder=GloveTextEmbedding(device=device),
    #     batch_size=256,
    # ),
)
# path = Path(
#     f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/dfs/materialized/{args.method}/{args.run_id}/node_train{'_join' if args.left_join_fkey else ''}.pt"
# )
# path.parent.mkdir(parents=True, exist_ok=True)
train_dataset = train_dataset.materialize(path=None)

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
    if not DROPPED_COLS:
        train_metrics = task.evaluate(pred, train_table)

    pred = model.predict(tf_test=tf_val).numpy()
    if not DROPPED_COLS:
        val_metrics = task.evaluate(pred, val_table)

    pred = model.predict(tf_test=tf_test).numpy()
    test_metrics = task_test.evaluate(pred)
else:
    raise ValueError(f"Task task type is unsupported {task.task_type}")


def clean_metrics(metrics):
    """Convert numpy values to regular Python numbers for cleaner output."""
    return {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}

if not DROPPED_COLS:
    print(f"Train: {clean_metrics(train_metrics)}")
    print(f"Val: {clean_metrics(val_metrics)}")
print(f"Test: {clean_metrics(test_metrics)}")