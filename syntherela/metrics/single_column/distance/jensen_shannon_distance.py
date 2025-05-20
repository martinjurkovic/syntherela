"""Jensen-Shannon distance metric for single columns."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime

from syntherela.metrics.base import SingleColumnMetric, DistanceBaseMetric
from syntherela.metrics.single_column.distance.utils import get_histograms


class JensenShannonDistance(DistanceBaseMetric, SingleColumnMetric):
    """Jensen-Shannon distance metric for comparing distributions.

    This metric computes the Jensen-Shannon distance between the distributions
    of real and synthetic data columns. It is applicable to categorical, numerical,
    datetime, and boolean columns.

    Parameters
    ----------
    base : float, default=np.e
        The base of the logarithm used in the calculation.
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
        Maximum value of the metric (log_base(2)).

    """

    def __init__(self, base=np.e, **kwargs):
        super().__init__(**kwargs)
        self.name = "JensenShannonDistance"
        self.goal = Goal.MINIMIZE
        self.base = base
        self.min_value = 0.0
        self.max_value = np.emath.logn(base, 2)

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
    def compute(
        orig_col, synth_col, bins, normalize_histograms=True, base=np.e, **kwargs
    ):
        """Compute the Jensen-Shannon distance between two columns.

        Parameters
        ----------
        orig_col : pandas.Series
            The original column.
        synth_col : pandas.Series
            The synthetic column.
        bins : int or array-like
            The bins to use for the histograms.
        normalize_histograms : bool, default=True
            Whether to normalize the histograms.
        base : float, default=np.e
            The base of the logarithm used in the calculation.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The Jensen-Shannon distance between the two columns.

        """
        gt_freq, synth_freq = get_histograms(
            orig_col, synth_col, normalize=normalize_histograms, bins=bins
        )
        return jensenshannon(gt_freq, synth_freq, base=base)

    def run(self, real_data, synthetic_data, **kwargs):
        """Run the Jensen-Shannon distance metric.

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
            - value: The Jensen-Shannon distance.
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
        # compute bin values on the original data
        if real_data.dtype.name in ("object", "category", "bool"):
            bins = None
        else:
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()
            bins = np.histogram_bin_edges(real_data)
        return super().run(
            real_data, synthetic_data, bins=bins, base=self.base, **kwargs
        )
