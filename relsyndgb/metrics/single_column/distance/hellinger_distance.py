from relsyndgb.base import StatisticalBaseMetric
from relsyndgb.single_column.base import SingleColumnMetric

import numpy as np
import pandas as pd

class HellingerDistance(StatisticalBaseMetric, SingleColumnMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HellingerDistance"

    def is_applicable(self, column_type):
        return column_type in ["categorical", "numerical"]

    def get_histograms(
        self,
        original, synthetic,
        normalize: bool = False, 
    ) -> dict:
        """Get percentual frequencies or counts for each possible real categorical value.

        Returns:
            The observed and expected frequencies.
        """
        
        if "date" in str(type(original.dtype)):
            original = pd.to_numeric(pd.to_datetime(original), errors = 'coerce', downcast='integer')
            synthetic = pd.to_numeric(pd.to_datetime(synthetic), errors = 'coerce', downcast='integer')
            
        if original.dtype.name in ("object", "category"):  # categorical
            gt = original.value_counts().to_dict()
            synth = synthetic.value_counts().to_dict()
            # add missing values with smoothing constant to avoid division by zero
            for val in gt:
                if val not in synth:
                    synth[val] = 0
            for val in synth:
                if val not in gt:
                    gt[val] = 0
        elif np.issubdtype(original.dtype, np.number):  # continuous
            original = original.dropna()
            synthetic = synthetic.dropna()
            gt_vals, bins = np.histogram(original, bins='doane', 
                range=(min(min(original), min(synthetic)),
                    max(max(original), max(synthetic))))
            synth_vals, _ = np.histogram(synthetic, bins=bins)
            gt = {k: v  for k, v in zip(bins, gt_vals)}
            synth = {k: v for k, v in zip(bins, synth_vals)}
        else:
            raise ValueError(f"Column is not categorical or continouous")
            
        # order the keys
        gt = {k: v for k, v in sorted(gt.items(), key=lambda item: item[0])}
        synth = {k: v for k, v in sorted(synth.items(), key=lambda item: item[0])}

        assert gt.keys() == synth.keys(), f"Keys do not match for column"

        if normalize:
            gt_sum = sum(gt.values())
            synth_sum = sum(synth.values())
            gt = {k: v / gt_sum for k, v in gt.items()}
            synth = {k: v / synth_sum for k, v in synth.items()}

        frequencies = (np.array(list(gt.values())), np.array(list(synth.values())))

        return frequencies
    
    def hellinger(self, p, q):
        return sum([(np.sqrt(t[0])-np.sqrt(t[1]))*(np.sqrt(t[0])-np.sqrt(t[1]))\
                    for t in zip(p,q)])/np.sqrt(2.)


    def compute(self, orig_col, synth_col):
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
        gt_freq, synth_freq = self.get_histograms(orig_col, synth_col, normalize=True)

        return self.hellinger(gt_freq, synth_freq)
