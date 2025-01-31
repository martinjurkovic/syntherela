import os
import warnings
from pathlib import Path
from datetime import datetime


from syntherela.report import Report
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns
from syntherela.metrics.single_column.statistical import ChiSquareTest
from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy
from syntherela.visualisations.multi_table_visualisations import (
    visualize_multi_table,
)
from syntherela.visualisations.single_column_visualisations import (
    visualize_single_column_detection_metrics,
    visualize_single_column_distance_metrics,
)
from syntherela.visualisations.single_table_visualisations import (
    visualize_single_table_detection_metrics_per_table,
    visualize_single_table_distance_metrics,
)


class Benchmark:
    def __init__(
        self,
        real_data_dir,
        synthetic_data_dir,
        results_dir,
        benchmark_name,
        single_column_metrics=[
            ChiSquareTest(),
        ],
        single_table_metrics=[
            MaximumMeanDiscrepancy(),
        ],
        multi_table_metrics=[],
        methods=None,
        datasets=None,
        run_id=None,
        sample_id=None,
        validate_metadata=True,
        compute_trends=True,
    ):
        self.datasets = datasets
        self.run_id = str(run_id)
        self.sample_id = str(sample_id)
        self.validate_metadata = validate_metadata
        self.compute_trends = compute_trends

        self.benchmark_name = (benchmark_name,)
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.results_dir = Path(results_dir)

        if self.datasets is None:
            self.datasets = [
                d
                for d in os.listdir(self.synthetic_data_dir)
                if os.path.isdir(os.path.join(self.synthetic_data_dir, d))
            ]

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
                    self.methods[dataset_name] = [
                        d
                        for d in os.listdir(self.synthetic_data_dir / dataset_name)
                        if os.path.isdir(
                            os.path.join(self.synthetic_data_dir / dataset_name, d)
                        )
                    ]

        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.benchmark_datetime = datetime.now()

        # Initialize results containers
        self.all_results = {}
        self.reports = {}

        # Try to read existing results during initialization
        try:
            self.read_results()
        except Exception as e:
            warnings.warn(
                f"No existing results found or could not read them: {str(e)}. This is expected if running the benchmark for the first time."
            )

    def load_data(self, dataset_name, method_name):
        real_data_path = self.real_data_dir / dataset_name
        synthetic_data_path = self.synthetic_data_dir / dataset_name / method_name

        if self.run_id is not None:
            synthetic_data_path = synthetic_data_path / self.run_id
        if self.sample_id is not None:
            synthetic_data_path = synthetic_data_path / self.sample_id

        metadata = Metadata().load_from_json(real_data_path / "metadata.json")

        real_data = load_tables(real_data_path, metadata)
        synthetic_data = load_tables(synthetic_data_path, metadata)

        real_data, metadata = remove_sdv_columns(
            real_data, metadata, validate=self.validate_metadata
        )
        synthetic_data, metadata = remove_sdv_columns(
            synthetic_data,
            metadata,
            update_metadata=False,
            validate=self.validate_metadata,
        )

        return real_data, synthetic_data, metadata

    def merge_results(self, dataset_name, method_name, new_results):
        """Merge new results with existing results for a given dataset and method.

        Args:
            dataset_name (str): Name of the dataset
            method_name (str): Name of the method
            new_results (dict): New results to merge

        Returns:
            dict: Merged results
        """
        existing_results = None
        if (
            dataset_name in self.all_results
            and method_name in self.all_results[dataset_name]
        ):
            existing_results = self.all_results[dataset_name][method_name]

        if existing_results:
            # Update only the metrics that were run
            for metric_type, metrics in new_results.items():
                if metric_type in existing_results:
                    # If the existing metric is a dictionary, update it
                    if isinstance(existing_results[metric_type], dict) and isinstance(metrics, dict):
                        existing_results[metric_type].update(metrics)
                    else:
                        # If either is not a dictionary, just replace with new value
                        existing_results[metric_type] = metrics
                else:
                    existing_results[metric_type] = metrics
            return existing_results
        else:
            return new_results

    def run(self):
        for dataset_name in self.datasets:
            for method_name in self.methods[dataset_name]:
                try:
                    real_data, synthetic_data, metadata = self.load_data(
                        dataset_name, method_name
                    )

                    print(
                        f"Starting benchmark for {dataset_name}, method_name {method_name}"
                    )

                    report = Report(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        metadata=metadata,
                        report_name=f"{self.benchmark_name}_{dataset_name}_{method_name}",
                        method_name=method_name,
                        dataset_name=dataset_name,
                        run_id=self.run_id,
                        single_column_metrics=self.single_column_metrics,
                        single_table_metrics=self.single_table_metrics,
                        multi_table_metrics=self.multi_table_metrics,
                        validate_metadata=self.validate_metadata,
                        compute_trends=self.compute_trends,
                        sample_id=self.sample_id,
                    )

                    self.reports.setdefault(dataset_name, {})[method_name] = report

                    # Generate and merge new results
                    new_results = report.generate()
                    merged_results = self.merge_results(
                        dataset_name, method_name, new_results
                    )
                    self.all_results.setdefault(dataset_name, {})[
                        method_name
                    ] = merged_results

                    # Update report results with merged results before saving
                    report.results = merged_results
                    file_name = self.build_file_name(dataset_name, method_name)
                    report.save_results(self.results_dir, file_name)

                except Exception as e:
                    print(
                        f"There was an error with dataset: {dataset_name}, method: {method_name}."
                    )
                    print(e)

    def read_results(self):
        for dataset_name in self.datasets:
            for method_name in self.methods[dataset_name]:
                file_name = self.build_file_name(dataset_name, method_name)
                try:
                    real_data, synthetic_data, metadata = self.load_data(
                        dataset_name, method_name
                    )
                except FileNotFoundError:
                    warnings.warn(
                        f"Results for {dataset_name}, method {method_name} not found."
                    )
                    continue
                temp_report = Report(
                    real_data,
                    synthetic_data,
                    metadata,
                    f"{dataset_name}_{method_name}",
                    validate_metadata=self.validate_metadata,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    run_id=self.run_id,
                    sample_id=self.sample_id,
                ).load_from_json(self.results_dir / file_name)
                self.reports.setdefault(dataset_name, {})[method_name] = temp_report
                self.all_results.setdefault(dataset_name, {})[
                    method_name
                ] = temp_report.results
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
        datasets = kwargs.pop("datasets", self.datasets)
        methods = kwargs.pop("methods", self.methods[datasets[0]])
        if distance:
            visualize_single_table_distance_metrics(
                granularity_level="single_table",
                metric_type="distance",
                all_results=self.all_results,
                datasets=datasets,
                methods=methods,
                **kwargs,
            )

        if detection:
            # visualize_single_table_detection_metrics_per_classifier(self.all_results, datasets, methods, **kwargs)
            visualize_single_table_detection_metrics_per_table(
                granularity_level="single_table",
                metric_type="detection",
                all_results=self.all_results,
                datasets=datasets,
                methods=methods,
                **kwargs,
            )

    def visualize_single_column_metrics(self, distance=True, detection=True, **kwargs):
        datasets = kwargs.get("datasets", self.datasets)
        methods = kwargs.get("methods", self.methods[datasets[0]])
        if distance:
            visualize_single_column_distance_metrics(
                granularity_level="single_column",
                metric_type="distance",
                all_results=self.all_results,
                datasets=datasets,
                methods=methods,
                **kwargs,
            )

        if detection:
            visualize_single_column_detection_metrics(
                granularity_level="single_column",
                metric_type="detection",
                all_results=self.all_results,
                datasets=datasets,
                methods=methods,
                **kwargs,
            )

    def visualize_multi_table_metrics(self, **kwargs):
        datasets = kwargs.get("datasets", self.datasets)
        methods = kwargs.get("methods", self.methods[datasets[0]])
        visualize_multi_table(self.all_results, datasets, methods, **kwargs)
