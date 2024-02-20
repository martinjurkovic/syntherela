from datetime import datetime
import json
import os
from pathlib import Path

from tqdm import tqdm

from relsyndgb.utils import NpEncoder
from relsyndgb.metrics.single_column.statistical import ChiSquareTest
from relsyndgb.metrics.single_table.distance import MaximumMeanDiscrepancy

class Report():

    def __init__(self, 
                real_data,
                synthetic_data, 
                metadata,
                report_name, 
                single_column_metrics = [
                    ChiSquareTest(), 
                   ],
                single_table_metrics = [
                    MaximumMeanDiscrepancy(),
                    ], 
                multi_table_metrics = [],
                ):
        metadata.validate_data(real_data)
        metadata.validate_data(synthetic_data)
        self.real_data = real_data
        self.synthetic_data = synthetic_data

        # reorder synthetic data columns to match real data
        for table in metadata.get_tables():
            self.synthetic_data[table] = self.synthetic_data[table][self.real_data[table].columns]
            assert (self.real_data[table].columns == self.synthetic_data[table].columns).all(), f"Columns in real and synthetic data do not match for table {table}"
        self.metadata = metadata
        self.report_name = report_name
        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.report_datetime = datetime.now()
        self.results = {
            "single_column_metrics": {},
            "single_table_metrics": {},
            "multi_table_metrics": {},
        }


    def generate(self):
        """
        Generate the report.
        """
        column_count = sum([len(self.real_data[table].columns) for table in self.metadata.get_tables()])
        table_count = len(self.metadata.get_tables())

        # single_column_metrics
        if len(self.single_column_metrics) == 0:
            print("No single column metrics to run. Skipping.")
        else:
            with tqdm(total=len(self.single_column_metrics) * column_count, desc="Running Single Column Metrics") as pbar:
                for table in self.metadata.get_tables():
                    for metric in self.single_column_metrics:
                        for column, column_info in self.metadata.tables[table].columns.items():
                            if not metric.is_applicable(column_info["sdtype"]):
                                pbar.update(1)
                                continue
                            self.results["single_column_metrics"].setdefault(metric.name, {}).setdefault(table, {})[column] = metric.run(
                                self.real_data[table][column],
                                self.synthetic_data[table][column],
                                metadata = 'TODO',
                            )
                            pbar.update(1)
                    
        # single_table_metrics
        if len(self.single_table_metrics) == 0:
            print("No single table metrics to run. Skipping.")
        else:
            with tqdm(total=len(self.single_table_metrics) * table_count, desc="Running Single Table Metrics") as pbar:
                for table in self.metadata.get_tables():
                    for metric in self.single_table_metrics:
                        if not metric.is_applicable(self.metadata.to_dict()['tables'][table]):
                                pbar.update(1)
                                continue
                        self.results["single_table_metrics"].setdefault(metric.name, {})[table] = metric.run(
                            self.real_data[table],
                            self.synthetic_data[table],
                            metadata = self.metadata.to_dict()['tables'][table],
                        )
                        pbar.update(1)

        # multi_table_metrics
        if len(self.multi_table_metrics) == 0:
            print("No multi table metrics to run. Skipping.")
        else:
            for metric in tqdm(self.multi_table_metrics, desc="Running Multi Table Metrics"):
                self.results["multi_table_metrics"][metric.name] = metric.run(
                    self.real_data,
                    self.synthetic_data,
                    metadata = self.metadata,
                )


        self.report_datetime = datetime.now()

        return self.results
    
    def print_results(self):
        """
        Print the results.
        """
        print(json.dumps(self.results, sort_keys=True, indent=4, cls=NpEncoder))
    
    def save_results(self, path, filename=None):
        """
        Save the results to a file.
        """
        path = Path(path)

        if filename is None:
            filename = f"{self.report_name}_{self.report_datetime.strftime('%Y_%m_%d')}.json"

        path = path / filename

        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.results, f, sort_keys=True, indent=4, cls=NpEncoder)

    def get_metric_instance(self, metric_name):
        """
        Get the metric instance.
        """
        for metric in self.single_column_metrics + self.single_table_metrics + self.multi_table_metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"Metric with name \"{metric_name}\" not found in the report. Available metrics: {[metric.name for metric in self.single_column_metrics + self.single_table_metrics + self.multi_table_metrics]}")
