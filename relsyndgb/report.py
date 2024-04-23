from datetime import datetime
import json
import os
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from relsyndgb.utils import NpEncoder
from relsyndgb.metrics.single_column.statistical import ChiSquareTest
from relsyndgb.metrics.single_table.distance import MaximumMeanDiscrepancy
from relsyndgb.visualisations.distribution_visualisations import visualize_bivariate_distributions, visualize_marginals, visualize_parent_child_bivariates

from multiprocessing import Pool

class Report():

    def __init__(self, 
                real_data,
                synthetic_data, 
                metadata,
                report_name, 
                test_data = None,
                single_column_metrics = [
                    ChiSquareTest(), 
                   ],
                single_table_metrics = [
                    MaximumMeanDiscrepancy(),
                    ], 
                multi_table_metrics = [],
                utility_metrics = [],
                ):
        metadata.validate_data(real_data)
        metadata.validate_data(synthetic_data)
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.test_data = None

        if test_data is not None:
            metadata.validate_data(test_data)
            self.test_data = test_data

        # reorder synthetic data columns to match real data
        for table in metadata.get_tables():
            self.synthetic_data[table] = self.synthetic_data[table][self.real_data[table].columns]
            assert (self.real_data[table].columns == self.synthetic_data[table].columns).all(), f"Columns in real and synthetic data do not match for table {table}"
        self.metadata = metadata
        self.report_name = report_name
        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.utility_metrics = utility_metrics
        self.report_datetime = datetime.now()
        self.results = {
            "single_column_metrics": {},
            "single_table_metrics": {},
            "multi_table_metrics": {},
            "utility_metrics": {},
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
                            try:
                                self.results["single_column_metrics"].setdefault(metric.name, {}).setdefault(table, {})[column] = metric.run(
                                    self.real_data[table][column],
                                    self.synthetic_data[table][column],
                                    metadata = self.metadata.to_dict()['tables'][table]['columns'][column],  
                                )
                            except Exception as e:
                                print(f"There was a problem with metric {metric.name}, table {table}, column {column}")
                                print(e)
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
                        try:
                            self.results["single_table_metrics"].setdefault(metric.name, {})[table] = metric.run(
                                self.real_data[table],
                                self.synthetic_data[table],
                                metadata = self.metadata.to_dict()['tables'][table],
                            )
                        except Exception as e:
                                print(f"There was a problem with metric {metric.name}, table {table}")
                                print(e)
                        pbar.update(1)

        # multi_table_metrics
        if len(self.multi_table_metrics) == 0:
            print("No multi table metrics to run. Skipping.")
        else:
            for metric in tqdm(self.multi_table_metrics, desc="Running Multi Table Metrics"):
                try:
                    self.results["multi_table_metrics"][metric.name] = metric.run(
                        self.real_data,
                        self.synthetic_data,
                        metadata = self.metadata,
                    )
                except Exception as e:
                    print(f"There was a problem with metric {metric.name}")
                    print(e)
        
        # utility_metrics
        if len(self.utility_metrics) == 0:
            print("No utility metrics to run. Skipping.")
        else:
            for metric in tqdm(self.utility_metrics, desc="Running Utility Metrics"):
                if self.test_data is None:
                    print(f"Test data is None. Skipping utility metric {metric.name}")
                    continue
                try:
                    self.results["utility_metrics"][metric.name] = metric.run(
                        self.real_data,
                        self.synthetic_data,
                        metadata = self.metadata,
                        test_data = self.test_data,
                    )
                except Exception as e:
                    print(f"There was a problem with metric {metric.name}")
                    print(e)


        self.report_datetime = datetime.now()

        return self.results

    def load_from_json(self, path):
        """
        Read the results from a file.
        """
        path = Path(path)
        with open(path, 'r') as f:
            self.results = json.load(f)
        return self
    
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

    def visualize_distributions(self, marginals=True, bivariate=True, parent_child_bivariate=True):
        """
        Visualize the distributions.
        """
        if marginals:
            visualize_marginals(self.real_data, self.synthetic_data, self.metadata)
        if bivariate:
            visualize_bivariate_distributions(self.real_data, self.synthetic_data, self.metadata)
        if parent_child_bivariate:
            visualize_parent_child_bivariates(self.real_data, self.synthetic_data, self.metadata)

    def get_metric_instance(self, metric_name):
        """
        Get the metric instance.
        """
        for metric in self.single_column_metrics + self.single_table_metrics + self.multi_table_metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"Metric with name \"{metric_name}\" not found in the report. Available metrics: {[metric.name for metric in self.single_column_metrics + self.single_table_metrics + self.multi_table_metrics]}")
