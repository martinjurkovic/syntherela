import numpy as np
import pandas as pd
from sdmetrics.goal import Goal

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class HellingerDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HellingerDistance"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical", "numerical"]
    
    @staticmethod
    def hellinger(p, q):
        return sum([(np.sqrt(t[0])-np.sqrt(t[1]))*(np.sqrt(t[0])-np.sqrt(t[1]))\
                    for t in zip(p,q)])/np.sqrt(2.)

    @classmethod
    def compute(cls, orig_col, synth_col, normalize_histograms=True, bins='doane', **kwargs):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        gt_freq, synth_freq = get_histograms(orig_col, synth_col, normalize=normalize_histograms, bins='doane')
        return cls.hellinger(gt_freq, synth_freq)
