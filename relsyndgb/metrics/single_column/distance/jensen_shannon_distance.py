import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sdmetrics.goal import Goal

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class JensenShannonDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "JensenShannonDistance"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]

    @staticmethod
    def compute(orig_col, synth_col, normalize_histograms=True, bins='doane', **kwargs):
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=normalize_histograms, bins=bins)
        return jensenshannon(gt_freq, synth_freq)
    
    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': 0, 'bootstrap_mean': 0, 'bootstrap_se': 0}
        return super().run(real_data, synthetic_data, **kwargs)
