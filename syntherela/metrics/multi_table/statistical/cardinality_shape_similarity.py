from scipy.stats import ks_2samp
from sdmetrics.utils import get_cardinality_distribution

from syntherela.metrics.base import StatisticalBaseMetric


class CardinalityShapeSimilarity(StatisticalBaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CardinalityShapeSimilarity"

    def validate(self, real_data, synthetic_data):
        return sorted(real_data.keys()) == sorted(synthetic_data.keys())

    def run(self, real_data, synthetic_data, metadata, **kwargs):
        self.validate(real_data, synthetic_data)
        return self.compute(real_data, synthetic_data, metadata)

    @staticmethod
    def compute(real_data, synthetic_data, metadata, **kwargs):
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
