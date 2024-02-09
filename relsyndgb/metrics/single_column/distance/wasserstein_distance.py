import numpy as np
import pandas as pd

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from scipy.stats import wasserstein_distance
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class WassersteinDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WassersteinDistance"

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]

    @staticmethod
    def compute(orig_col, synth_col, **kwargs):
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=False)
        return wasserstein_distance(gt_freq, synth_freq)

