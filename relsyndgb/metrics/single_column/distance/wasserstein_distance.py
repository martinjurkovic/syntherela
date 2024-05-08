import pandas as pd
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime
from ot.lp import wasserstein_1d

from relsyndgb.metrics.base import SingleColumnMetric, DistanceBaseMetric
from relsyndgb.metrics.single_column.distance.utils import get_histograms


class WassersteinDistance(DistanceBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "WassersteinDistance"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float('inf')

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["numerical", "datetime"]

    @staticmethod
    def compute(orig_col, synth_col, xmin, xmax, **kwargs):
        # sample real and synthetic data to have the same length
        n = min(len(orig_col), len(synth_col))
        orig_col = orig_col.sample(n, random_state=0)
        synth_col = synth_col.sample(n, random_state=1)
        
        # scale data to [0, 1]
        x_orig = ((orig_col - xmin) / (xmax - xmin)).values
        x_synth = ((synth_col - xmin) / (xmax - xmin)).values
        return wasserstein_1d(x_orig, x_synth)
    
    def run(self, real_data, synthetic_data, **kwargs):
        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data, errors='coerce', downcast='integer')
            synthetic_data = pd.to_numeric(synthetic_data, errors='coerce', downcast='integer')
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        xmin, xmax = combined_data.min(), combined_data.max()
        if self.is_constant(real_data):
            return {'value': 0, 'reference_ci': [0, 0], 'bootstrap_mean': 0, 'bootstrap_se': 0}
        return super().run(real_data, synthetic_data, xmin=xmin, xmax=xmax, **kwargs)

