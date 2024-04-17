import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class WassersteinDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WassersteinDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float('inf')

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical", "numerical", "datetime", "boolean"]

    @staticmethod
    def compute(orig_col, synth_col, bins, **kwargs):
        # sample real and synthetic data to have the same length
        n = min(len(orig_col), len(synth_col))
        orig_col = orig_col.sample(n, random_state=0)
        synth_col = synth_col.sample(n, random_state=1)
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, bins=bins, normalize=False)
        return wasserstein_distance(gt_freq, synth_freq)
    
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
            bins = np.histogram_bin_edges(real_data.dropna())
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)

