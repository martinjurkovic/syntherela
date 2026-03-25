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
        self.name = 'CardinalityShapeSimilarity'

    @staticmethod
    def validate(data):
        """Validate that the input looks like a multi-table mapping."""
        return isinstance(data, dict) and len(data) > 0

    def run(self, real_data, synthetic_data, **kwargs):
        """Execute the cardinality shape similarity metric."""
        self.validate(real_data)
        self.validate(synthetic_data)
        return self.compute(real_data, synthetic_data, **kwargs)

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """Compute the cardinality metric."""
        metadata = kwargs['metadata']
        results = {}
        for rel in metadata.relationships:
            cardinality_real = get_cardinality_distribution(
                real_data[rel['parent_table_name']][rel['parent_primary_key']],
                real_data[rel['child_table_name']][rel['child_foreign_key']],
            )
            cardinality_synthetic = get_cardinality_distribution(
                synthetic_data[rel['parent_table_name']][
                    rel['parent_primary_key']
                ],
                synthetic_data[rel['child_table_name']][
                    rel['child_foreign_key']  #
                ],
            )
            statistic, pval = ks_2samp(cardinality_real, cardinality_synthetic)
            key = f'{rel["parent_table_name"]}_{rel["child_table_name"]}'
            results[key] = {
                'statistic': statistic,
                'pval': pval,
            }
        return results
