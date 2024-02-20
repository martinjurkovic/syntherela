from sdmetrics.goal import Goal
from sdmetrics.errors import IncomputableMetricError
import pandas as pd
from sdmetrics.utils import get_frequencies
from sdmetrics.single_column.statistical import TVComplement

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class TotalVariationDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TotalVariationDistance"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical"]
    
    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """
        Total Variation Distance metric.
        Score transformed from sdmetrics.single_column.statistical.TVComplement to return the total variation distance.
        """
        return (1 - TVComplement.compute(real_data, synthetic_data)) / 0.5
    
    def run(self, real_data, synthetic_data, **kwargs):
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': 0, 'bootstrap_mean': 0, 'bootstrap_se': 0}
        return super().run(real_data, synthetic_data, **kwargs)