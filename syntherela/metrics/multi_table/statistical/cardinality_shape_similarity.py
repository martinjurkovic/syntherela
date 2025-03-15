"""Cardinality shape similarity metrics for multi-table data.

This module provides metrics for evaluating how well synthetic data preserves
the cardinality of relationships between tables in multi-table datasets.
"""

from scipy.stats import ks_2samp
from sdmetrics.utils import get_cardinality_distribution

from syntherela.metrics.base import StatisticalBaseMetric


class CardinalityShapeSimilarity(StatisticalBaseMetric):
    """Cardinality shape similarity metric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CardinalityShapeSimilarity"

    def validate(self, real_data, synthetic_data):
        """Validate the input data."""
        return sorted(real_data.keys()) == sorted(synthetic_data.keys())

    def run(self, real_data, synthetic_data, metadata, **kwargs):
        """Execute the cardinality shape similarity metric."""
        self.validate(real_data, synthetic_data)
        return self.compute(real_data, synthetic_data, metadata)

    @staticmethod
    def compute(real_data, synthetic_data, metadata, **kwargs):
        """Compute the cardinality shape similarity between real and synthetic data."""
        results = {}
        for rel in metadata.relationships:
            cardinality_real = get_cardinality_distribution(
                real_data[rel["parent_table_name"]][rel["parent_primary_key"]],
                real_data[rel["child_table_name"]][rel["child_foreign_key"]],
            )
            cardinality_synthetic = get_cardinality_distribution(
                synthetic_data[rel["parent_table_name"]][rel["parent_primary_key"]],
                synthetic_data[rel["child_table_name"]][rel["child_foreign_key"]],
            )
            statistic, pval = ks_2samp(cardinality_real, cardinality_synthetic)
            results[f"{rel['parent_table_name']}_{rel['child_table_name']}"] = {
                "statistic": statistic,
                "pval": pval,
            }
        return results
