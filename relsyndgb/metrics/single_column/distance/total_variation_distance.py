import numpy as np
from sdmetrics.goal import Goal

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class TotalVariationDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TotalVariationDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float('inf')

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]
    
    @staticmethod
    def compute(real_data, synthetic_data, bins, **kwargs):
        """
        Total Variation Distance metric.
        """

        f_exp, f_obs = get_histograms(
            real_data, synthetic_data, normalize=True, bins=bins)
        total_variation = 0
        for i in range(len(f_obs)):
            total_variation += abs(f_obs[i] - f_exp[i])

        return total_variation
    
    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': [0, 0], 'bootstrap_mean': 0, 'bootstrap_se': 0}
        if real_data.dtype.name in ("object", "category"):
            bins = None
        else:
            bins = np.histogram_bin_edges(real_data.dropna())
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)