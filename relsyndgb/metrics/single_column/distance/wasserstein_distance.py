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
        return column_type in ["numerical", "datetime"]

    @staticmethod
    def compute(orig_col, synth_col, **kwargs):
        if orig_col.dtype == 'datetime64[ns]':
            orig_col = orig_col.astype('int64')
            synth_col = synth_col.astype('int64')

        orig_col = orig_col.dropna()
        synth_col = synth_col.dropna()
        return wasserstein_distance(orig_col, synth_col)
