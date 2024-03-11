import numpy as np
from scipy.spatial.distance import jensenshannon
from sdmetrics.goal import Goal

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class JensenShannonDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, base = np.e, **kwargs):
        super().__init__(**kwargs)
        self.name = "JensenShannonDistance"
        self.goal = Goal.MINIMIZE
        self.base = base
        self.min_value = 0.0
        self.max_value = np.emath.logn(base, 2)

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]

    @staticmethod
    def compute(orig_col, synth_col, bins, normalize_histograms=True, base = np.e, **kwargs):
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=normalize_histograms, bins=bins)
        return jensenshannon(gt_freq, synth_freq, base=base)
    
    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': [0, 0], 'bootstrap_mean': 0, 'bootstrap_se': 0}
        # compute bin values on the original data
        if real_data.dtype.name in ("object", "category"):
            bins = None
        else:
            bins = np.histogram_bin_edges(real_data.dropna())
        return super().run(real_data, synthetic_data, bins=bins, base = self.base, **kwargs)
