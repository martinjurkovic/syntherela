import numpy as np
import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

from syntherela.metrics.base import SingleColumnMetric, DistanceBaseMetric
from syntherela.metrics.single_column.distance.utils import get_histograms


_SQRT2 = np.sqrt(2)


class HellingerDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HellingerDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = 1

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical", "numerical", "datetime", "boolean"]

    @staticmethod
    def hellinger(p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    @classmethod
    def compute(cls, orig_col, synth_col, bins, normalize_histograms=True, **kwargs):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.
            bins:
                The bins to use for the histogram.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=normalize_histograms, bins=bins
        )
        return cls.hellinger(gt_freq, synth_freq)

    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {
                "value": 0,
                "reference_ci": [0, 0],
                "bootstrap_mean": 0,
                "bootstrap_se": 0,
            }
        # check for datetime
        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data, errors="coerce", downcast="integer")
            synthetic_data = pd.to_numeric(
                synthetic_data, errors="coerce", downcast="integer"
            )
        # compute bin values on the original data
        if real_data.dtype.name in ("object", "category", "bool"):
            bins = None
        else:
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()
            bins = np.histogram_bin_edges(real_data)
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)
