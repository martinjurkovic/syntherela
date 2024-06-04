import numpy as np
import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

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
        return column_type in ["categorical", "numerical", "datetime", "boolean"]
    
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
        # check for datetime
        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data, errors='coerce', downcast='integer')
            synthetic_data = pd.to_numeric(synthetic_data, errors='coerce', downcast='integer')
        if real_data.dtype.name in ("object", "category", "bool"):
            bins = None
        else:
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()
            bins = np.histogram_bin_edges(real_data)
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)