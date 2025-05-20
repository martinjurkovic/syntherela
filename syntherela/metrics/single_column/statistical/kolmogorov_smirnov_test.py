"""Kolmogorov-Smirnov statistical test for single columns."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdmetrics.utils import is_datetime
from sdmetrics.goal import Goal

from syntherela.metrics.base import SingleColumnMetric, StatisticalBaseMetric


class KolmogorovSmirnovTest(StatisticalBaseMetric, SingleColumnMetric):
    """Kolmogorov-Smirnov test metric for comparing marginal distributions.

    This metric computes the Kolmogorov-Smirnov test statistic between the distributions
    of real and synthetic data columns. It is applicable to numerical and datetime columns.

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

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KolmogorovSmirnovTest"
        self.goal = Goal.MINIMIZE

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
        return column_type == "numerical" or column_type == "datetime"

    def validate(self, column):
        """Validate that the column is numerical or datetime.

        Parameters
        ----------
        column : pandas.Series
            The column to validate.

        Raises
        ------
        ValueError
            If the column is not numerical or datetime.

        """
        column_dtype = column.dtypes
        if np.issubdtype(column_dtype, np.number) or np.issubdtype(
            column_dtype, np.datetime64
        ):
            return

        raise ValueError(
            f"{self.name} can only be applied to numerical columns, but column {column.name} is of type {column.dtype}"
        )

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute the Kolmogorov-Smirnov test statistic and p-value.

        Parameters
        ----------
        real_data : pandas.Series
            The real data column.
        synthetic_data : pandas.Series
            The synthetic data column.

        Returns
        -------
        dict
            Dictionary containing:
            - statistic: The Kolmogorov-Smirnov test statistic.
            - p_val: The p-value of the test.

        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        statistic, p_val = ks_2samp(real_data, synthetic_data)

        return {"statistic": statistic, "p_val": p_val}
