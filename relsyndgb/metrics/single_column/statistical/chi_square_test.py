import pandas as pd
from scipy.stats import chi2_contingency

from relsyndgb.metrics.base import SingleColumnMetric, StatisticalBaseMetric


class ChiSquareTest(StatisticalBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ChiSquareTest"

    @staticmethod
    def is_applicable(column_type):
        return column_type == "categorical"

    def validate(self, column):
        if column.dtype.name not in ("object", "category"):
            raise ValueError(f"{self.name} can only be applied to categorical columns, but column {column.name} is of type {column.dtype}")
    
    @staticmethod
    def compute(real_data, synthetic_data):
        orig_col = pd.Categorical(real_data)
        synth_col = pd.Categorical(synthetic_data, categories=orig_col.categories)
        freq_orig = orig_col.value_counts()
        freq_synth = synth_col.value_counts()
        freq_synth = freq_synth / freq_synth.sum() * freq_orig.sum()
        assert (freq_orig.index == freq_synth.index).all(), f"Indexes do not match for column"
        # calculate the chi-square test
        statistic, pval, _, _ = chi2_contingency([freq_orig, freq_synth]) 

        return statistic, pval