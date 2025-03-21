"""Wasserstein distance metric for single columns."""

import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

from syntherela.metrics.base import SingleColumnMetric, DistanceBaseMetric


class WassersteinDistance(DistanceBaseMetric, SingleColumnMetric):
    """Wasserstein distance metric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WassersteinDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float("inf")

    @staticmethod
    def is_applicable(column_type):
        """Check if the metric is applicable to the given column type."""
        return column_type in ["numerical", "datetime"]

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """Compute the Wasserstein distance between two columns."""
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        orig_col = pd.Series(real_data).dropna()
        synth_col = pd.Series(synthetic_data).dropna()
        # sample real and synthetic data to have the same length
        n = min(len(orig_col), len(synth_col))
        orig_col = orig_col.sample(n, random_state=0)
        synth_col = synth_col.sample(n, random_state=1)

        # scale data to [0, 1]
        scaler = MinMaxScaler()
        scaler.fit(combined_data.values.reshape(-1, 1))
        x_orig = scaler.transform(orig_col.values.reshape(-1, 1)).flatten()
        x_synth = scaler.transform(synth_col.values.reshape(-1, 1)).flatten()
        return wasserstein_distance(x_orig, x_synth)

    def run(self, real_data, synthetic_data, **kwargs):
        """Execute the Wasserstein distance metric."""
        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data, errors="coerce", downcast="integer")
            synthetic_data = pd.to_numeric(
                synthetic_data, errors="coerce", downcast="integer"
            )
        if self.is_constant(real_data):
            return {
                "value": 0,
                "reference_ci": [0, 0],
                "bootstrap_mean": 0,
                "bootstrap_se": 0,
            }
        return super().run(real_data, synthetic_data, **kwargs)
