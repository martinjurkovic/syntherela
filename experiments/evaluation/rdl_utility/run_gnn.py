import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import faulthandler

faulthandler.enable()

# Set CUDA_LAUNCH_BLOCKING=1 to get better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

"""
Usage examples for different GNN architectures:

# Default HeteroGraphSAGE
python run_gnn.py --gnn_architecture hetero-graphsage

# Use HeteroGNN with GIN convolution
python run_gnn.py --gnn_architecture hetero-gin

# Use HeteroGNN with GraphConv convolution (GCN-like but supports heterogeneous graphs)
python run_gnn.py --gnn_architecture hetero-graphconv

# Use HeteroGNN with GAT convolution (uses default 4 heads, concat=True)
python run_gnn.py --gnn_architecture hetero-gat

# Use HeteroGNN with GAT v2 convolution (uses default 4 heads, concat=True)
python run_gnn.py --gnn_architecture hetero-gatv2

# Use RelGNN_Model directly (relational GNN with message passing)
python run_gnn.py --gnn_architecture relgnn
"""

import numpy as np
import torch
from model import Model, create_hetero_gin, create_hetero_graphconv, create_hetero_gat, create_hetero_gatv2
from relgnn_nn import RelGNN_Model, get_atomic_routes
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, BaseTask
from relbench.base.task_autocomplete import AutoCompleteTask
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
parser.add_argument("--run_id", type=str, default="1")
parser.add_argument("--method", type=str, default="ORIGINAL")

parser.add_argument(
    "--task_type",
    type=str,
    default="REGRESSION",
    choices=["BINARY_CLASSIFICATION", "REGRESSION", "MULTILABEL_CLASSIFICATION"],
)
parser.add_argument("--dataset", type=str, default="walmart_subsampled")
parser.add_argument("--entity_table", type=str, default="depts")
parser.add_argument("--target_col", type=str, default="Weekly_Sales")

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--gnn_architecture", type=str, default="hetero-gin", 
                    choices=["hetero-graphsage", "hetero-gin", "hetero-graphconv", "hetero-gat", "hetero-gatv2", "relgnn"],
                    help="GNN architecture to use")
parser.add_argument("--num_neighbors", type=int, default=-1)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--mlp_layers", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--torch_device", type=str, default="cuda:0")
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()


device = torch.device(args.torch_device if torch.cuda.is_available() else "cpu")
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

# task_test.get_table("test", mask_input_cols=False)


stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)

    # remove target column from stypes.json
    if args.task == "autocomplete": 
        col_to_stype_dict[args.entity_table].pop(args.target_col)
except FileNotFoundError:
    print(f"No stypes.json found for {args.dataset}, generating new ones.")
    print(f"Please consider generating them with the metadata_sdv_to_relbench.py script.")
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    # Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    # with open(stypes_cache_path, "w") as f:
    #     json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict_train = make_pkey_fkey_graph(
    dataset.get_db(
        upto_test_timestamp=False if args.task == "autocomplete" else True,
    ),
    col_to_stype_dict=col_to_stype_dict,
    # text_embedder_cfg=TextEmbedderConfig(
    #     text_embedder=GloveTextEmbedding(device=device), batch_size=256
    # ),
    # cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)
data_test, col_stats_dict_test = make_pkey_fkey_graph(
    dataset_test.get_db(
        upto_test_timestamp=False if args.task == "autocomplete" else True,
    ),
    col_to_stype_dict=col_to_stype_dict,
    # text_embedder_cfg=TextEmbedderConfig(
    #     text_embedder=GloveTextEmbedding(device=device), batch_size=256
    # ),
    # cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

g = torch.Generator()
g.manual_seed(args.seed)

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    tmp_task = task_test if split == "test" else task
    table = tmp_task.get_table(split, mask_input_cols=False)
    table_input = get_node_train_table_input(table=table, task=tmp_task)
    tmp_data = data if split in ("train", "val") else data_test
    loader_dict[split] = NeighborLoader(
        tmp_data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        # num_neighbors=[-1 for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        generator=g,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        # if task.task_type == TaskType.REGRESSION:
        #     assert clamp_min is not None
        #     assert clamp_max is not None
        #     pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


# Mapping from GNN architecture strings to factory functions
GNN_FACTORY_MAP = {
    "hetero-graphsage": None,  # None means use default HeteroGraphSAGE
    "hetero-gin": create_hetero_gin,
    "hetero-graphconv": create_hetero_graphconv,
    "hetero-gat": create_hetero_gat,
    "hetero-gatv2": create_hetero_gatv2,
    "relgnn": None,  # RelGNN_Model will be created directly (special case)
}

# Get the selected GNN factory
selected_gnn_factory = GNN_FACTORY_MAP[args.gnn_architecture]

# Create model with selected GNN architecture
if args.gnn_architecture == "relgnn":
    # Use RelGNN_Model directly (special case)
    atomic_routes_list = get_atomic_routes(data.edge_types)
    
    model = RelGNN_Model(
        data=data,
        col_stats_dict=col_stats_dict_test,
        num_model_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        norm="batch_norm",
        atomic_routes=atomic_routes_list,
        num_heads=1,  # Default number of heads
        simplified_MP=False,  # Default simplified message passing
        mlp_layers=args.mlp_layers,
    ).to(device)
else:
    # Use standard Model class with factory pattern
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict_test,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        norm="batch_norm",
        # norm=None,
        gnn_factory=selected_gnn_factory,  # None for default HeteroGraphSAGE
        mlp_layers=args.mlp_layers,
    ).to(device)

print(f"Using GNN architecture: {args.gnn_architecture}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    test_pred = test(loader_dict["test"])
    test_metrics = task_test.evaluate(test_pred)
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")
    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

# model.data = data_test

test_pred = test(loader_dict["test"])
test_metrics = task_test.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
