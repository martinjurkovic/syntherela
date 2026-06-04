"""Kolmogorov-Smirnov statistical test for single columns."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdmetrics.utils import is_datetime

from syntherela.metrics.base import (
    Goal,
    SingleColumnMetric,
    StatisticalBaseMetric,
)


class KolmogorovSmirnovTest(StatisticalBaseMetric, SingleColumnMetric):
    """Kolmogorov-Smirnov test metric for comparing marginal distributions.

    This metric computes the Kolmogorov-Smirnov test statistic between the
    distributions of real and synthetic data columns. It is applicable to
    numerical and datetime columns.

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
        self.name: str = 'KolmogorovSmirnovTest'
        self.goal: Goal = Goal.MINIMIZE

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
            Whether the metric is applicable to the column type.

        """
        return column_type == 'numerical' or column_type == 'datetime'

    @staticmethod
    def validate(data: Any) -> None:
        """Validate that the column is numerical or datetime.

        Parameters
        ----------
        data : pandas.Series
            The column to validate.

        Raises
        ------
        ValueError
            If the column is not numerical or datetime.

        """
        column_dtype = data.dtypes
        if np.issubdtype(column_dtype, np.number) or np.issubdtype(
            column_dtype, np.datetime64
        ):
            return

        raise ValueError(
            'KolmogorovSmirnovTest can only be applied to numerical '
            f'columns, but column {data.name} is of type {data.dtype}'
        )

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
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

        return {'statistic': statistic, 'p_val': p_val}
