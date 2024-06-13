import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdmetrics.utils import is_datetime
from sdmetrics.goal import Goal

from syntherela.metrics.base import SingleColumnMetric, StatisticalBaseMetric


class KolmogorovSmirnovTest(StatisticalBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KolmogorovSmirnovTest"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        return column_type == "numerical" or column_type == "datetime"

    def validate(self, column):
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
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        statistic, p_val = ks_2samp(real_data, synthetic_data)

        return {"statistic": statistic, "p_val": p_val}
