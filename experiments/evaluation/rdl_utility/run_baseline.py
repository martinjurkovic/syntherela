import argparse

import numpy as np
import pandas as pd
import torch
from gnn_datasets import (
    AirbnbDataset,
    BerkaDataset,
    F1Dataset,
    RossmannDataset,
    WalmartDataset,
)
from relbench.base import (
    AutoCompleteTask,
    BaseTask,
    Dataset,
    EntityTask,
    Table,
    TaskType,
)
from relbench.tasks import get_task
from relbench.tasks.f1 import DriverDNFTask, DriverPositionTask, DriverTop3Task
from scipy.stats import mode
from torch_geometric.seed import seed_everything

DATASETS = {
    RossmannDataset.name: RossmannDataset,
    WalmartDataset.name: WalmartDataset,
    F1Dataset.name: F1Dataset,
    AirbnbDataset.name: AirbnbDataset,
    BerkaDataset.name: BerkaDataset,
}

TASKS = {
    'driver-position': DriverPositionTask,
    'driver-top3': DriverTop3Task,
    'driver-dnft': DriverDNFTask,
    'autocomplete': AutoCompleteTask,
}

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='f1_subsampled')
parser.add_argument('--task', type=str, default='driver-top3')
parser.add_argument('--run_id', type=str, default='1')
parser.add_argument('--method', type=str, default='ORIGINAL')

parser.add_argument(
    '--task_type',
    type=str,
    default='BINARY_CLASSIFICATION',
    choices=[
        'BINARY_CLASSIFICATION',
        'REGRESSION',
        'MULTILABEL_CLASSIFICATION',
    ],
)
parser.add_argument('--entity_table', type=str, default='users')
parser.add_argument('--target_col', type=str, default='country_destination')

parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(args.seed)

predict_column_task_config = {
    'task_type': TaskType[args.task_type],
    'entity_table': args.entity_table,
    'target_col': args.target_col,
}

# dataset: Dataset = get_dataset(args.dataset, download=False)
dataset: Dataset = DATASETS[args.dataset](
    method=args.method, run_id=args.run_id
)
dataset_test: Dataset = DATASETS[args.dataset](
    method=args.method, run_id=args.run_id, type='test'
)

# task = PredictColumnTask(dataset=dataset, **predict_column_task_config)
if args.task == 'autocomplete':
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
    task_test: EntityTask = get_task('rel-f1', args.task, download=False)
    dataset_test = task_test.dataset

train_table = task.get_table('train')
val_table = task.get_table('val')
test_table = task_test.get_table('test')


def evaluate(
    task: BaseTask, train_table: Table, pred_table: Table, name: str
) -> dict[str, float]:
    is_test = task.target_col not in pred_table.df
    if name == 'global_zero':
        pred = np.zeros(len(pred_table))
    elif name == 'global_mean':
        mean = train_table.df[task.target_col].astype(float).values.mean()
        pred = np.ones(len(pred_table)) * mean
    elif name == 'global_median':
        median = np.median(train_table.df[task.target_col].astype(float).values)
        pred = np.ones(len(pred_table)) * median
    elif name == 'entity_mean':
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: 'mean'})
        df.rename(columns={task.target_col: '__target__'}, inplace=True)
        df = pred_table.df.merge(df, how='left', on=fkey)
        pred = df['__target__'].fillna(0).astype(float).values
    elif name == 'entity_median':
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: 'median'})
        df.rename(columns={task.target_col: '__target__'}, inplace=True)
        df = pred_table.df.merge(df, how='left', on=fkey)
        pred = df['__target__'].fillna(0).astype(float).values
    elif name == 'random':
        pred = np.random.rand(len(pred_table))
    elif name == 'majority':
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        pred = torch.full((len(pred_table),), fill_value=majority_label)
    elif name == 'majority_multilabel':
        past_target = train_table.df[task.target_col]
        majority = mode(np.stack(past_target.values), axis=0).mode[0]
        pred = np.stack([majority] * len(pred_table.df))
    elif name == 'random_multilabel':
        num_labels = train_table.df[task.target_col].values[0].shape[0]
        pred = np.random.rand(len(pred_table), num_labels)
    else:
        raise ValueError('Unknown eval name called {name}.')
    return task.evaluate(pred, None if is_test else pred_table)


trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)

if task.task_type == TaskType.REGRESSION:
    eval_name_list = [
        'global_zero',
        'global_mean',
        'global_median',
        'entity_mean',
        'entity_median',
    ]

    for name in eval_name_list:
        train_metrics = evaluate(task, train_table, train_table, name=name)
        val_metrics = evaluate(task, train_table, val_table, name=name)
        test_metrics = evaluate(
            task_test, trainval_table, test_table, name=name
        )
        print(f'{name}:')
        print(f'Train: {train_metrics}')
        print(f'Val: {val_metrics}')
        print(f'Test: {test_metrics}')

elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    eval_name_list = ['random', 'majority']
    for name in eval_name_list:
        train_metrics = evaluate(task, train_table, train_table, name=name)
        val_metrics = evaluate(task, train_table, val_table, name=name)
        test_metrics = evaluate(
            task_test, trainval_table, test_table, name=name
        )
        print(f'{name}:')
        print(f'Train: {train_metrics}')
        print(f'Val: {val_metrics}')
        print(f'Test: {test_metrics}')

elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    eval_name_list = ['random_multilabel', 'majority_multilabel']
    for name in eval_name_list:
        train_metrics = evaluate(task, train_table, train_table, name=name)
        val_metrics = evaluate(task, train_table, val_table, name=name)
        test_metrics = evaluate(
            task_test, trainval_table, test_table, name=name
        )
        print(f'{name}:')
        print(f'Train: {train_metrics}')
        print(f'Val: {val_metrics}')
        print(f'Test: {test_metrics}')
