"""Hellinger distance metric for single columns."""

import numpy as np
import pandas as pd
from sdmetrics.utils import is_datetime

from syntherela.metrics.base import DistanceBaseMetric, Goal, SingleColumnMetric
from syntherela.metrics.single_column.distance.utils import get_histograms

_SQRT2 = np.sqrt(2)


class HellingerDistance(DistanceBaseMetric, SingleColumnMetric):
    """Hellinger distance metric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'HellingerDistance'
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = 1

    @staticmethod
    def is_applicable(column_type):
        """Check if the metric is applicable to the given column type."""  # noqa: DOC201
        return column_type in [
            'categorical',
            'numerical',
            'datetime',
            'boolean',
        ]

    @staticmethod
    def hellinger(p, q):
        """Hellinger distance between two histograms."""  # noqa: DOC201
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """Compute Hellinger distance between two histograms.

        Parameters
        ----------
        orig_col:
            The values from the real dataset.
        synth_col:
            The values from the synthetic dataset.
        bins:
            The bins to use for the histogram.
        normalize_histograms:
            Whether to normalize the histograms.


        Returns
        -------
            Union[float, tuple[float]]:
                Metric output or outputs.

        """
        bins = kwargs.get('bins')
        normalize_histograms = kwargs.get('normalize_histograms', True)
        gt_freq, synth_freq = get_histograms(
            real_data,
            synthetic_data,
            normalize=normalize_histograms,
            bins=bins,
        )
        return HellingerDistance.hellinger(gt_freq, synth_freq)

    def run(self, real_data, synthetic_data, **kwargs):
        """Run the Hellinger distance metric.

        Returns
        -------
        dict
            Dictionary with results.

        """
        if self.is_constant(real_data):
            return {
                'value': 0,
                'reference_ci': [0, 0],
                'bootstrap_mean': 0,
                'bootstrap_se': 0,
            }
        # check for datetime
        if is_datetime(real_data):
            real_data = pd.to_numeric(
                real_data, errors='coerce', downcast='integer'
            )
            synthetic_data = pd.to_numeric(
                synthetic_data, errors='coerce', downcast='integer'
            )
        # compute bin values on the original data
        if real_data.dtype.name in ('object', 'category', 'bool'):
            bins = None
        else:
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()
            bins = np.histogram_bin_edges(real_data)
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)
