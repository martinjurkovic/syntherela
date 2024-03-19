from datetime import datetime
import json
import os
from pathlib import Path

from tqdm import tqdm

from relsyndgb.utils import NpEncoder
from relsyndgb.metadata import Metadata
from relsyndgb.report import Report
from relsyndgb.metrics.single_column.statistical import ChiSquareTest
from relsyndgb.metrics.single_table.distance import MaximumMeanDiscrepancy
from relsyndgb.data import load_tables, remove_sdv_columns

class Benchmark():

    def __init__(self, 
                real_data_dir,
                synthetic_data_dir,
                results_dir,
                benchmark_name, 
                single_column_metrics = [
                    ChiSquareTest(), 
                   ],
                single_table_metrics = [
                    MaximumMeanDiscrepancy(),
                    ], 
                multi_table_metrics = [],
                methods=None,
                datasets=None,
                run_id=None,
                sample_id=None,
                ):
        # metadata.validate_data(real_data)
        # metadata.validate_data(synthetic_data)
        # self.real_data = real_data
        # self.synthetic_data = synthetic_data

        # reorder synthetic data columns to match real data
        # for table in metadata.get_tables():
        #     self.synthetic_data[table] = self.synthetic_data[table][self.real_data[table].columns]
        #     assert (self.real_data[table].columns == self.synthetic_data[table].columns).all(), f"Columns in real and synthetic data do not match for table {table}"
        # self.metadata = metadata
        self.methods = methods if methods is not None else {}
        self.datasets = datasets
        self.run_id = str(run_id)
        self.sample_id = str(sample_id)

        self.benchmark_name = benchmark_name, 
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.results_dir = Path(results_dir)

        if self.datasets is None:
            self.datasets = [d for d in os.listdir(self.synthetic_data_dir) if os.path.isdir(os.path.join(self.synthetic_data_dir, d))]

        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.benchmark_datetime = datetime.now()

        


    def run(self):
        for dataset in self.datasets:
            if dataset not in self.methods:
                self.methods[dataset] = [d for d in os.listdir(self.synthetic_data_dir / dataset) if os.path.isdir(os.path.join(self.synthetic_data_dir / dataset, d))]
            
            for method in self.methods[dataset]:
                real_data_path = self.real_data_dir / dataset
                synthetic_data_path = self.synthetic_data_dir / dataset / method

                if self.run_id is not None:
                    synthetic_data_path = synthetic_data_path / self.run_id
                if self.sample_id is not None:
                    synthetic_data_path = synthetic_data_path / self.sample_id

                metadata = Metadata().load_from_json(real_data_path / 'metadata.json')

                real_data = load_tables(real_data_path, metadata)
                synthetic_data = load_tables(synthetic_data_path, metadata)

                real_data, metadata = remove_sdv_columns(real_data, metadata)
                synthetic_data, metadata = remove_sdv_columns(synthetic_data, metadata, update_metadata=False)

                print(f"Starting benchmark for {dataset}, method {method}")
                report = Report(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    metadata=metadata,
                    report_name=f"{self.benchmark_name}_{dataset}_{method}",
                    single_column_metrics=self.single_column_metrics,
                    single_table_metrics=self.single_table_metrics,
                    multi_table_metrics=self.multi_table_metrics,
                )

                results = report.generate()

                file_name = f"{dataset}_{method}"
                if self.run_id is not None:
                    file_name += f"_{self.run_id}"
                    if self.sample_id is not None:  
                        file_name += f"_{self.sample_id}"
                file_name += ".json"

                report.save_results(self.results_dir, file_name)



