"""Pairwise correlation difference metric for single tables.

This module implements a metric that measures the difference between correlation matrices
of real and synthetic data, evaluating how well the synthetic data preserves linear relationships
between variables in the original dataset.
"""

import numpy as np
import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

from syntherela.metadata import drop_ids
from syntherela.metrics.base import DistanceBaseMetric, SingleTableMetric


class PairwiseCorrelationDifference(DistanceBaseMetric, SingleTableMetric):
    """Pairwise correlation difference metric."""

    def __init__(self, norm_order="fro", correlation_method="pearson", **kwargs):
        super().__init__(**kwargs)
        self.name = "PairwiseCorrelationDifference"
        self.goal = Goal.MINIMIZE
        self.norm_order = norm_order
        self.correlation_method = correlation_method
        self.min_value = 0.0
        self.max_value = 1.0

    @staticmethod
    def is_applicable(metadata):
        """Check if the table contains at least one non-id column."""
        numeric_count = 0
        for column_name in metadata["columns"].keys():
            if (
                metadata["columns"][column_name]["sdtype"] == "numerical"
                or metadata["columns"][column_name]["sdtype"] == "datetime"
            ):
                numeric_count += 1
        return numeric_count > 1

    def compute(self, original_table, sythetic_table, metadata, **kwargs):
        """Compute pairwise correlation difference between original and synthetic data.

        Based on:
            Andre Goncalves, Priyadip Ray, Braden Soper, Jennifer Stevens, Linda Coyle & Ana Paula Sales (2020).
            Generation and evaluation of synthetic patient data.
            https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1

        Parameters
        ----------
        original_table : pd.DataFrame
            The original data table.
        sythetic_table : pd.DataFrame
            The synthetic data table.
        metadata : dict
            Table metadata.

        Returns
        -------
        float
            The pairwise correlation difference between the original and synthetic data.

        """
        orig = original_table.copy()
        synth = sythetic_table.copy()

        orig = drop_ids(orig, metadata)
        synth = drop_ids(synth, metadata)

        zero_variance_columns = []
        for col in orig.columns:
            if orig[col].dtype.name in ("object", "category"):
                orig.drop(col, axis=1, inplace=True)
                synth.drop(col, axis=1, inplace=True)
                continue
            elif is_datetime(orig[col]):
                orig[col] = pd.to_numeric(
                    orig[col], errors="coerce", downcast="integer"
                )
                synth[col] = pd.to_numeric(
                    synth[col], errors="coerce", downcast="integer"
                )

        # drop nan values
        orig.dropna(inplace=True)
        synth.dropna(inplace=True)

        # drop columns with zero variance
        std_orig = orig.std()
        zero_variance_columns = std_orig[std_orig == 0].index.tolist()
        orig.drop(columns=zero_variance_columns, inplace=True)
        synth.drop(columns=zero_variance_columns, inplace=True)
        assert (synth.std() > 0).all(), (
            "Synthetic data includes invalid columns with zero variance."
        )

        # compute the correlation matrix
        orig_corr = orig.corr(method=self.correlation_method)
        synth_corr = synth.corr(method=self.correlation_method)

        return np.linalg.norm(orig_corr - synth_corr, ord=self.norm_order).astype(float)
