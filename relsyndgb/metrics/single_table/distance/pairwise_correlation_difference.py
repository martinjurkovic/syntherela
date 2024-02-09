import numpy as np
import pandas as pd
from sdmetrics.goal import Goal

from relsyndgb.metrics.base import DistanceBaseMetric, SingleTableMetric


class PairwiseCorrelationDifference(DistanceBaseMetric, SingleTableMetric):
    def __init__(self, norm_order = 'fro', correlation_method='pearson', **kwargs):
        super().__init__(**kwargs)
        self.name = "PairwiseCorrelationDifference"
        self.goal = Goal.MINIMIZE
        self.norm_order = norm_order
        self.correlation_method = correlation_method

    @staticmethod
    def is_applicable(metadata):
        """
        Check if the table contains at least one column that is not an id.
        """
        numeric_count = 0
        for column_name in metadata['columns'].keys():
            if metadata['columns'][column_name]['sdtype'] == 'numerical':
                numeric_count += 1
        return numeric_count > 1


    def compute(self, original_table, sythetic_table, metadata, **kwargs):
        """
        Based on:
        Andre Goncalves, Priyadip Ray, Braden Soper, Jennifer Stevens, Linda Coyle & Ana Paula Sales (2020). 
        Generation and evaluation of synthetic patient data.
        https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1
        """
        orig = original_table.copy()
        synth = sythetic_table.copy()

        orig.drop(metadata['primary_key'], axis=1, inplace=True)
        synth.drop(metadata['primary_key'], axis=1, inplace=True)
        for col in orig.columns:
            if orig[col].dtype.name in ("object", "category"):
                orig.drop(col, axis=1, inplace=True)
                synth.drop(col, axis=1, inplace=True)
            elif "datetime" in str(orig[col].dtype):
                orig[col] =  pd.to_numeric(orig[col])
                synth[col] =  pd.to_numeric(synth[col])

        # drop nan values
        orig.dropna(inplace=True)
        synth.dropna(inplace=True)

        # compute the correlation matrix
        orig_corr = orig.corr(method=self.correlation_method)
        synth_corr = synth.corr(method=self.correlation_method)

        return np.linalg.norm(orig_corr - synth_corr, ord=self.norm_order).astype(float)
    