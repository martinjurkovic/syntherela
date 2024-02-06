from relsyndgb.base import StatisticalBaseMetric

import pandas as pd
from scipy.stats import chi2_contingency


class ChiSquareTest(StatisticalBaseMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_column(self, column):
        if column.dtype != 'object':
            raise ValueError(f"ChiSquareTest can only be applied to categorical columns, but column {column.name} is of type {column.dtype}")

    def compute(self, orig_column, synth_column):
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
        self.validate_column(orig_column)
        self.validate_column(synth_column)

        orig_col = pd.Categorical(orig_column)
        synth_col = pd.Categorical(synth_column, categories=orig_col.categories)
        freq_orig = orig_col.value_counts()
        freq_synth = synth_col.value_counts()
        freq_synth = freq_synth / freq_synth.sum() * freq_orig.sum()
        assert (freq_orig.index == freq_synth.index).all(), f"Indexes do not match for column"
        # calculate the chi-square test
        statistic, pval, _, _ = chi2_contingency([freq_orig, freq_synth]) 

        return statistic, pval