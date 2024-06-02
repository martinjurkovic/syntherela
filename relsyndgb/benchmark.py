import os
import warnings
from pathlib import Path
from datetime import datetime



from relsyndgb.report import Report
from relsyndgb.metadata import Metadata
from relsyndgb.data import load_tables, remove_sdv_columns
from relsyndgb.metrics.single_column.statistical import ChiSquareTest
from relsyndgb.metrics.single_table.distance import MaximumMeanDiscrepancy
from relsyndgb.visualisations.multi_table_visualisations import visualize_multi_table, visualize_parent_child_multi_table
from relsyndgb.visualisations.single_column_visualisations import visualize_single_column_detection_metrics, visualize_single_column_distance_metrics
from relsyndgb.visualisations.single_table_visualisations import visualize_single_table_detection_metrics_per_classifier, visualize_single_table_detection_metrics_per_table, visualize_single_table_distance_metrics


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
        self.datasets = datasets
        self.run_id = str(run_id)
        self.sample_id = str(sample_id)

        self.benchmark_name = benchmark_name, 
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.results_dir = Path(results_dir)

        if self.datasets is None:
            self.datasets = [d for d in os.listdir(self.synthetic_data_dir) if os.path.isdir(os.path.join(self.synthetic_data_dir, d))]

        if methods is not None:
            # if self.methods is dict
            if isinstance(methods, dict):
                self.methods = methods
            if isinstance(methods, list):
                self.methods = {}
                for dataset_name in self.datasets:
                    self.methods[dataset_name] = methods
        else:
            self.methods = {}
            for dataset_name in self.datasets:
                if dataset_name not in self.methods:
                        self.methods[dataset_name] = [d for d in os.listdir(self.synthetic_data_dir / dataset_name) if os.path.isdir(os.path.join(self.synthetic_data_dir / dataset_name, d))]


        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.benchmark_datetime = datetime.now()

        self.all_results = {}
        self.reports = {}

    def load_data(self, dataset_name, method_name):
        real_data_path = self.real_data_dir / dataset_name
        synthetic_data_path = self.synthetic_data_dir / dataset_name / method_name
        test_data_path = self.real_data_dir / dataset_name / 'test_data'

        if self.run_id is not None:
            synthetic_data_path = synthetic_data_path / self.run_id
        if self.sample_id is not None:
            synthetic_data_path = synthetic_data_path / self.sample_id

        metadata = Metadata().load_from_json(real_data_path / 'metadata.json')

        real_data = load_tables(real_data_path, metadata)
        synthetic_data = load_tables(synthetic_data_path, metadata)

        real_data, metadata = remove_sdv_columns(real_data, metadata)
        synthetic_data, metadata = remove_sdv_columns(synthetic_data, metadata, update_metadata=False)

        return real_data, synthetic_data, metadata

    def run(self):
        for dataset_name in self.datasets:
            for method_name in self.methods[dataset_name]:
                try:
                    real_data, synthetic_data, metadata = self.load_data(dataset_name, method_name)

                    print(f"Starting benchmark for {dataset_name}, method_name {method_name}")
                    report = Report(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        metadata=metadata,
                        report_name=f"{self.benchmark_name}_{dataset_name}_{method_name}",
                        single_column_metrics=self.single_column_metrics,
                        single_table_metrics=self.single_table_metrics,
                        multi_table_metrics=self.multi_table_metrics,
                    )

                    self.reports.setdefault(dataset_name, {})[method_name] = report

                    self.all_results.setdefault(dataset_name, {})[method_name] = report.generate()

                    file_name = self.build_file_name(dataset_name, method_name)

                    report.save_results(self.results_dir, file_name)
                except Exception as e:
                    print(f"There was an error with dataset: {dataset_name}, method: {method_name}.")
                    print(e)

    def read_results(self):
        for dataset_name in self.datasets:
            for method_name in self.methods[dataset_name]:
                file_name = self.build_file_name(dataset_name, method_name)
                with open(self.results_dir / file_name, 'r') as f:
                    real_data, synthetic_data, metadata = self.load_data(dataset_name, method_name)
                    temp_report = Report(real_data, synthetic_data, metadata, f"{dataset_name}_{method_name}").load_from_json(self.results_dir / file_name)
                    self.reports.setdefault(dataset_name, {})[method_name] = temp_report
                    self.all_results.setdefault(dataset_name, {})[method_name] = temp_report.results
        if not self.all_results:
            warnings.warn("No results found.")

    def build_file_name(self, dataset_name, method_name):
        file_name = f"{dataset_name}_{method_name}"
        if self.run_id is not None:
            file_name += f"_{self.run_id}"
            if self.sample_id is not None:  
                file_name += f"_{self.sample_id}"
        file_name += ".json"
        return file_name

    def visualize_single_table_metrics(self, distance=True, detection=True, **kwargs):
        datasets = kwargs.get('datasets', self.datasets)
        methods = kwargs.get('methods', self.methods[datasets[0]])
        if distance:
            visualize_single_table_distance_metrics(granularity_level="single_table", 
                                                    metric_type="distance", 
                                                    all_results=self.all_results, 
                                                    datasets=datasets, 
                                                    methods=methods, **kwargs)

        if detection:
            # visualize_single_table_detection_metrics_per_classifier(self.all_results, datasets, methods, **kwargs)
            visualize_single_table_detection_metrics_per_table(granularity_level="single_table", 
                                                    metric_type="detection", 
                                                    all_results=self.all_results, 
                                                    datasets=datasets, 
                                                    methods=methods, **kwargs)

    def visualize_single_column_metrics(self, distance=True, detection=True, **kwargs):
        datasets = kwargs.get('datasets', self.datasets)
        methods = kwargs.get('methods', self.methods[datasets[0]])
        if distance:
            visualize_single_column_distance_metrics(granularity_level="single_column", 
                                                    metric_type="distance", 
                                                    all_results=self.all_results, 
                                                    datasets=datasets, 
                                                    methods=methods, **kwargs)

        if detection:
            visualize_single_column_detection_metrics(granularity_level="single_column", 
                                                    metric_type="detection", 
                                                    all_results=self.all_results, 
                                                    datasets=datasets, 
                                                    methods=methods, **kwargs)

    def visualize_multi_table_metrics(self, parent_child = True, **kwargs):
        datasets = kwargs.get('datasets', self.datasets)
        methods = kwargs.get('methods', self.methods[datasets[0]])
        visualize_multi_table(self.all_results, datasets, methods, **kwargs)
        if parent_child:
            visualize_parent_child_multi_table(self.all_results, datasets, methods, **kwargs)
            
