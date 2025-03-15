"""Chi-square statistical test for single columns."""

import pandas as pd
from scipy.stats import chi2_contingency
from sdmetrics.goal import Goal

from syntherela.metrics.base import SingleColumnMetric, StatisticalBaseMetric


class ChiSquareTest(StatisticalBaseMetric, SingleColumnMetric):
    """ChiSquare test metric.

    Attributes
    ----------
        name (str): The name of the metric, set to "ChiSquareTest".
        goal (Goal): The goal of the metric, set to Goal.MINIMIZE.

    Methods
    -------
        is_applicable(column_type):
            Checks if the metric is applicable to the given column type.
        validate(column):
            Validates if the column is of a type that can be used with the chi-square test.
        compute(real_data, synthetic_data):
            Computes the chi-square statistic and p-value for the given real and synthetic data.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ChiSquareTest"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        """Check if the metric is applicable to the given column type."""
        return column_type == "categorical" or column_type == "boolean"

    def validate(self, column):
        """Validate if the data type that can be used with the chi-square test."""
        if column.dtype.name not in ("object", "category", "bool"):
            raise ValueError(
                f"{self.name} can only be applied to categorical columns, but column {column.name} is of type {column.dtype}"
            )

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute the chi-square test statistic and p-value.

        Parameters
        ----------
        real_data:  pd.Series
            The real data as a pandas Series.
        synthetic_data : pd.Series
            The synthetic data as a pandas Series.

        Returns
        -------
            dict: A dictionary containing the chi-square test statistic and p-value.
                - "statistic" (float): The chi-square test statistic.
                - "p_value" (float): The p-value of the chi-square test.

        Raises
        ------
            AssertionError: If the indexes of the frequency counts do not match.

        """
        orig_col = pd.Categorical(real_data)
        synth_col = pd.Categorical(synthetic_data, categories=orig_col.categories)
        freq_orig = orig_col.value_counts()
        freq_synth = synth_col.value_counts()
        if freq_synth.sum() == 0:
            return {"statistic": -1, "p_value": 0}
        freq_synth = freq_synth / freq_synth.sum() * freq_orig.sum()
        assert (freq_orig.index == freq_synth.index).all(), (
            "Indexes do not match for column"
        )
        # calculate the chi-square test
        statistic, pval, _, _ = chi2_contingency([freq_orig, freq_synth])

        return {"statistic": statistic, "p_value": pval}
