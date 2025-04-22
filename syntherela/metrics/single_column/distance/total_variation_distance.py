"""Total Variation distance metric for single columns."""

import numpy as np
import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

from syntherela.metrics.base import SingleColumnMetric, DistanceBaseMetric
from syntherela.metrics.single_column.distance.utils import get_histograms


class TotalVariationDistance(DistanceBaseMetric, SingleColumnMetric):
    """Total Variation Distance metric for comparing distributions.

    This metric computes the total variation distance between the distributions
    of real and synthetic data columns. It is applicable to categorical, numerical,
    datetime, and boolean columns.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    Attributes
    ----------
    name : str
        Name of the metric.
    goal : Goal
        Goal of the metric (minimize).
    min_value : float
        Minimum value of the metric (0.0).
    max_value : float
        Maximum value of the metric (infinity).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TotalVariationDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float("inf")

    @staticmethod
    def is_applicable(column_type):
        """Check if the column type is applicable for this metric.

        Parameters
        ----------
        column_type : str
            The type of the column.

        Returns
        -------
        bool
            True if the metric is applicable to the column type, False otherwise.

        """
        return column_type in ["categorical", "numerical", "datetime", "boolean"]

    @staticmethod
    def compute(real_data, synthetic_data, bins, **kwargs):
        """Compute the Total Variation Distance between two columns.

        Parameters
        ----------
        real_data : pandas.Series
            The real data column.
        synthetic_data : pandas.Series
            The synthetic data column.
        bins : int or array-like
            The bins to use for the histograms.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The Total Variation Distance between the two columns.

        """
        f_exp, f_obs = get_histograms(
            real_data, synthetic_data, normalize=True, bins=bins
        )
        total_variation = 0
        for i in range(len(f_obs)):
            total_variation += abs(f_obs[i] - f_exp[i])

        return total_variation

    def run(self, real_data, synthetic_data, **kwargs):
        """Run the Total Variation Distance metric.

        Parameters
        ----------
        real_data : pandas.Series
            The real data column.
        synthetic_data : pandas.Series
            The synthetic data column.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing the metric results, including:
            - value: The Total Variation Distance.
            - reference_ci: Reference confidence interval.
            - bootstrap_mean: Bootstrap mean estimate.
            - bootstrap_se: Bootstrap standard error.

        """
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
        if real_data.dtype.name in ("object", "category", "bool"):
            bins = None
        else:
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()
            bins = np.histogram_bin_edges(real_data)
        return super().run(real_data, synthetic_data, bins=bins, **kwargs)
