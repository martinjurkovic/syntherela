import numpy as np
import pandas as pd

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from scipy.spatial.distance import jensenshannon
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class JensenShannonDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "JensenShannonDistance"

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]

    @staticmethod
    def compute(orig_col, synth_col, normalize_histograms=True, bins='doane', **kwargs):
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=normalize_histograms, bins=bins)
        return jensenshannon(gt_freq, synth_freq)
