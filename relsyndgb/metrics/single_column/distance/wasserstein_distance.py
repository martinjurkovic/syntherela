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
        # TODO: add continous version using sinkhorn loss
        return column_type in ["categorical", "boolean"]

    @staticmethod
    def compute(orig_col, synth_col, **kwargs):
        # sample real and synthetic data to have the same length
        n = min(len(orig_col), len(synth_col))
        orig_col = orig_col.sample(n, random_state=0)
        synth_col = synth_col.sample(n, random_state=1)
        (gt_freq, synth_freq), keys = get_histograms(
            orig_col, synth_col, normalize=False, return_keys=True)
        return wasserstein_distance(keys, keys, u_weights=gt_freq, v_weights=synth_freq)
    
    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': [0, 0], 'bootstrap_mean': 0, 'bootstrap_se': 0}
        return super().run(real_data, synthetic_data, **kwargs)

