import pandas as pd
from scipy.stats import chi2_contingency
from sdmetrics.goal import Goal

from syntherela.metrics.base import SingleColumnMetric, StatisticalBaseMetric


class ChiSquareTest(StatisticalBaseMetric, SingleColumnMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ChiSquareTest"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        return column_type == "categorical" or column_type == "boolean"

    def validate(self, column):
        if column.dtype.name not in ("object", "category", "bool"):
            raise ValueError(
                f"{self.name} can only be applied to categorical columns, but column {column.name} is of type {column.dtype}"
            )

    @staticmethod
    def compute(real_data, synthetic_data):
        orig_col = pd.Categorical(real_data)
        synth_col = pd.Categorical(synthetic_data, categories=orig_col.categories)
        freq_orig = orig_col.value_counts()
        freq_synth = synth_col.value_counts()
        if freq_synth.sum() == 0:
            return {"statistic": -1, "p_value": 0}
        freq_synth = freq_synth / freq_synth.sum() * freq_orig.sum()
        assert (
            freq_orig.index == freq_synth.index
        ).all(), "Indexes do not match for column"
        # calculate the chi-square test
        statistic, pval, _, _ = chi2_contingency([freq_orig, freq_synth])

        return {"statistic": statistic, "p_value": pval}
