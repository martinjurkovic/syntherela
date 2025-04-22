"""Reporting functionality for synthetic data evaluation.

This module provides tools for generating reports on the quality
of synthetic data across various metrics for a specific dataset and method.
"""

import os
import json
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from syntherela.utils import NpEncoder
from syntherela.metrics.multi_table.trends import multi_table_trends
from syntherela.metrics.single_table.trends import ColumnPairsReport
from syntherela.metrics.single_column.trends import ColumnShapesReport
from syntherela.metrics.single_column.statistical import ChiSquareTest
from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy
from syntherela.visualisations.distribution_visualisations import (
    visualize_bivariate_distributions,
    visualize_marginals,
    visualize_parent_child_bivariates,
)


class Report:
    """Report class for evaluating synthetic data quality.

    This class provides functionality to generate reports comparing synthetic data
    against real data using various metrics at different levels.

    Parameters
    ----------
    real_data : dict
        Dictionary mapping table names to pandas DataFrames for real data.
    synthetic_data : dict
        Dictionary mapping table names to pandas DataFrames for synthetic data.
    metadata : Metadata
        Metadata object containing information about the tables.
    report_name : str
        Name of the report.
    method_name : str
        Name of the synthetic data generation method.
    dataset_name : str
        Name of the dataset.
    run_id : str
        Identifier for the report run.
    single_column_metrics : list, default=[ChiSquareTest()]
        List of single column metrics to compute.
    single_table_metrics : list, default=[MaximumMeanDiscrepancy()]
        List of single table metrics to compute.
    multi_table_metrics : list, default=[]
        List of multi table metrics to compute.
    validate_metadata : bool, default=True
        Whether to validate metadata against data.
    compute_trends : bool, default=True
        Whether to compute trends.
    sample_id : str, default=None
        Identifier for the data sample.

    """

    def __init__(
        self,
        real_data,
        synthetic_data,
        metadata,
        report_name,
        method_name,
        dataset_name,
        run_id,
        single_column_metrics=[
            ChiSquareTest(),
        ],
        single_table_metrics=[
            MaximumMeanDiscrepancy(),
        ],
        multi_table_metrics=[],
        validate_metadata=True,
        compute_trends=True,
        sample_id=None,
    ):
        """Initialize the Report object.

        Parameters
        ----------
        real_data : dict
            Dictionary mapping table names to pandas DataFrames for real data.
        synthetic_data : dict
            Dictionary mapping table names to pandas DataFrames for synthetic data.
        metadata : Metadata
            Metadata object containing information about the tables.
        report_name : str
            Name of the report.
        method_name : str
            Name of the synthetic data generation method.
        dataset_name : str
            Name of the dataset.
        run_id : str
            Identifier for the report run.
        single_column_metrics : list, default=[ChiSquareTest()]
            List of single column metrics to compute.
        single_table_metrics : list, default=[MaximumMeanDiscrepancy()]
            List of single table metrics to compute.
        multi_table_metrics : list, default=[]
            List of multi table metrics to compute.
        validate_metadata : bool, default=True
            Whether to validate metadata against data.
        compute_trends : bool, default=True
            Whether to compute trends.
        sample_id : str, default=None
            Identifier for the data sample.

        """
        if validate_metadata:
            metadata.validate_data(real_data)
            metadata.validate_data(synthetic_data)
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.evaluate_trends = compute_trends

        # reorder synthetic data columns to match real data
        for table in metadata.get_tables():
            self.synthetic_data[table] = self.synthetic_data[table][
                self.real_data[table].columns
            ]
            assert (
                self.real_data[table].columns == self.synthetic_data[table].columns
            ).all(), (
                f"Columns in real and synthetic data do not match for table {table}"
            )
        self.metadata = metadata
        self.report_name = report_name
        self.single_column_metrics = single_column_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.report_datetime = datetime.now()
        self.results = {
            "method_name": method_name,
            "dataset_name": dataset_name,
            "run_id": run_id,
            "sample_id": sample_id,
            "single_column_metrics": {},
            "single_table_metrics": {},
            "multi_table_metrics": {},
        }

    def generate(self):
        """Generate the report by computing all metrics.

        This method computes all configured metrics for single columns,
        single tables, and multi tables.

        Returns
        -------
        dict
            Dictionary containing all report results.

        """
        column_count = sum(
            [len(self.real_data[table].columns) for table in self.metadata.get_tables()]
        )
        table_count = len(self.metadata.get_tables())

        # single_column_metrics
        if len(self.single_column_metrics) == 0:
            print("No single column metrics to run. Skipping.")
        else:
            with tqdm(
                total=len(self.single_column_metrics) * column_count,
                desc="Running Single Column Metrics",
            ) as pbar:
                for table in self.metadata.get_tables():
                    for metric in self.single_column_metrics:
                        for column, column_info in self.metadata.tables[
                            table
                        ].columns.items():
                            if not metric.is_applicable(column_info["sdtype"]):
                                pbar.update(1)
                                continue
                            try:
                                self.results["single_column_metrics"].setdefault(
                                    metric.name, {}
                                ).setdefault(table, {})[column] = metric.run(
                                    self.real_data[table][column],
                                    self.synthetic_data[table][column],
                                    metadata=self.metadata.to_dict()["tables"][table][
                                        "columns"
                                    ][column],
                                )
                            except Exception as e:
                                print(
                                    f"There was a problem with metric {metric.name}, table {table}, column {column}"
                                )
                                print(e)
                            pbar.update(1)
            if self.evaluate_trends:
                self.compute_trends(
                    single_column=True, single_table=False, multi_table=False
                )
        # single_table_metrics
        if len(self.single_table_metrics) == 0:
            print("No single table metrics to run. Skipping.")
        else:
            with tqdm(
                total=len(self.single_table_metrics) * table_count,
                desc="Running Single Table Metrics",
            ) as pbar:
                for table in self.metadata.get_tables():
                    for metric in self.single_table_metrics:
                        if not metric.is_applicable(
                            self.metadata.to_dict()["tables"][table]
                        ):
                            pbar.update(1)
                            continue
                        try:
                            self.results["single_table_metrics"].setdefault(
                                metric.name, {}
                            )[table] = metric.run(
                                self.real_data[table],
                                self.synthetic_data[table],
                                metadata=self.metadata.to_dict()["tables"][table],
                            )
                        except Exception as e:
                            print(
                                f"There was a problem with metric {metric.name}, table {table}"
                            )
                            print(e)
                        pbar.update(1)
            if self.evaluate_trends:
                self.compute_trends(
                    single_column=False, single_table=True, multi_table=False
                )
        # multi_table_metrics
        if len(self.multi_table_metrics) == 0:
            print("No multi table metrics to run. Skipping.")
        else:
            for metric in tqdm(
                self.multi_table_metrics, desc="Running Multi Table Metrics"
            ):
                try:
                    self.results["multi_table_metrics"][metric.name] = metric.run(
                        self.real_data,
                        self.synthetic_data,
                        metadata=self.metadata,
                    )
                except Exception as e:
                    print(f"There was a problem with metric {metric.name}")
                    print(e)
            if self.evaluate_trends:
                self.compute_trends(
                    single_column=False, single_table=False, multi_table=True
                )

        self.report_datetime = datetime.now()

        return self.results

    def compute_trends(
        self, single_table=True, single_column=True, multi_table=True, verbose=True
    ):
        """Compute trends for the data.

        Parameters
        ----------
        single_table : bool, default=True
            Whether to compute single table trends.
        single_column : bool, default=True
            Whether to compute single column trends.
        multi_table : bool, default=True
            Whether to compute multi table trends.
        verbose : bool, default=True
            Whether to display progress information.

        Returns
        -------
        dict
            Dictionary containing trend results.

        """
        if single_column:
            trends_report = ColumnShapesReport()
            trends_report.generate(
                self.real_data,
                self.synthetic_data,
                self.metadata.to_dict(),
                verbose=verbose,
            )
            shapes_df = trends_report.get_details("Column Shapes")
            self.results["single_column_metrics"]["Trends"] = {
                "shapes": {
                    "mean": shapes_df.Score.mean(),
                    "se": shapes_df.Score.std() / len(shapes_df) ** 0.5,
                }
            }
            # TODO: we could also store individual scores for each column
        if single_table:
            trends_report = ColumnPairsReport()
            trends_report.generate(
                self.real_data,
                self.synthetic_data,
                self.metadata.to_dict(),
                verbose=verbose,
            )
            pairs_df = trends_report.get_details("Column Pair Trends")
            if not pairs_df.empty:
                # This can happen with tables with only one column
                self.results["single_table_metrics"]["Trends"] = {
                    "pairs": {
                        "mean": pairs_df.Score.mean(),
                        "se": pairs_df.Score.std() / len(pairs_df) ** 0.5,
                    }
                }
        if multi_table:
            multi_table_trends_results = multi_table_trends(
                self.real_data,
                self.synthetic_data,
                self.metadata,
                verbose=verbose,
            )
            self.results["multi_table_metrics"]["Trends"] = {
                "cardinality": multi_table_trends_results["cardinality"],
                "k_hop_similarity": {},
            }
            hop_scores = multi_table_trends_results["avg_scores"]
            hop_se = multi_table_trends_results["scores_se"]
            for hop in multi_table_trends_results["hop_relation"]:
                self.results["multi_table_metrics"]["Trends"]["k_hop_similarity"][
                    hop
                ] = {
                    "mean": hop_scores[hop],
                    "se": hop_se[hop],
                }

    def load_from_json(self, path):
        """Load report results from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.

        Returns
        -------
        dict
            Dictionary containing report results.

        """
        path = Path(path)
        with open(path, "r") as f:
            self.results = json.load(f)
        return self

    def print_results(self):
        """Print report results to the console."""
        print(json.dumps(self.results, sort_keys=True, indent=4, cls=NpEncoder))

    def save_results(self, path, filename=None):
        """Save report results to a JSON file.

        Parameters
        ----------
        path : str or Path
            Directory where the file will be saved.
        filename : str, default=None
            Name of the file. If None, a default name will be generated.

        Returns
        -------
        str
            Path to the saved file.

        """
        path = Path(path)

        if filename is None:
            filename = (
                f"{self.report_name}_{self.report_datetime.strftime('%Y_%m_%d')}.json"
            )

        path = path / filename

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=4, cls=NpEncoder)

    def visualize_distributions(
        self, marginals=True, bivariate=True, parent_child_bivariate=True
    ):
        """Visualize data distributions.

        Parameters
        ----------
        marginals : bool, default=True
            Whether to visualize marginal distributions.
        bivariate : bool, default=True
            Whether to visualize bivariate distributions.
        parent_child_bivariate : bool, default=True
            Whether to visualize parent-child bivariate distributions.

        """
        if marginals:
            visualize_marginals(self.real_data, self.synthetic_data, self.metadata)
        if bivariate:
            visualize_bivariate_distributions(
                self.real_data, self.synthetic_data, self.metadata
            )
        if parent_child_bivariate:
            visualize_parent_child_bivariates(
                self.real_data, self.synthetic_data, self.metadata
            )

    def get_metric_instance(self, metric_name):
        """Get a metric instance by name.

        Parameters
        ----------
        metric_name : str
            Name of the metric.

        Returns
        -------
        object
            Metric instance.

        Raises
        ------
        ValueError
            If the metric is not found.

        """
        for metric in (
            self.single_column_metrics
            + self.single_table_metrics
            + self.multi_table_metrics
        ):
            if metric.name == metric_name:
                return metric
        raise ValueError(
            f'Metric with name "{metric_name}" not found in the report. Available metrics: {[metric.name for metric in self.single_column_metrics + self.single_table_metrics + self.multi_table_metrics]}'
        )
