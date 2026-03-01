import argparse
import logging
import sys

from syntherela.benchmark import Benchmark
from syntherela.metrics.multi_table.detection import (
    AggregationDetection,
    ParentChildDetection,
)
from xgboost import XGBClassifier


class AggregationDetectionWithoutChildCounts(AggregationDetection):
    """A subclass of AggregationDetection that does not add child counts.
    This is useful for comparing aggregation detection with and without child counts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_child_counts = False


args = argparse.ArgumentParser()
args.add_argument('--dataset-name', type=str, default='imdb_MovieLens_v1')
args.add_argument('--methods', '-m', action='append', default=None)
args.add_argument('--run-id', type=str, default='1')
args = args.parse_args()
dataset_name = args.dataset_name
methods = args.methods
run_id = args.run_id

logger = logging.getLogger(f'{dataset_name}_logger')

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f'START LOGGING Dataset: {dataset_name}')

xgb_cls = XGBClassifier
xgb_args = {'seed': 0}

single_column_metrics = []
single_table_metrics = []
multi_table_metrics = [
    AggregationDetection(
        classifier_cls=xgb_cls,
        classifier_args=xgb_args,
        random_state=42,
    ),
    ParentChildDetection(
        classifier_cls=xgb_cls,
        classifier_args=xgb_args,
        random_state=42,
    ),
    AggregationDetectionWithoutChildCounts(
        classifier_cls=xgb_cls,
        classifier_args=xgb_args,
        random_state=42,
    ),
]

benchmark = Benchmark(
    real_data_dir='data/original',
    synthetic_data_dir='data/synthetic',
    results_dir=f'results/multi-table/{run_id}',
    benchmark_name='Muliti-Table Detection Comparison',
    single_column_metrics=single_column_metrics,
    single_table_metrics=single_table_metrics,
    multi_table_metrics=multi_table_metrics,
    run_id=run_id,
    sample_id='sample1',
    datasets=[dataset_name],
    methods=methods,
    compute_trends=False,
)

benchmark.run()
