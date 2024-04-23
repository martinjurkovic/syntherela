from typing import Union
import numpy as np
import pandas as pd
from sdmetrics.utils import is_datetime

def get_histograms(original: pd.Series, synthetic: pd.Series, normalize: bool = True, bins: Union[str, np.array] = 'doane', return_keys: bool=False) -> tuple:
    """Get percentual frequencies or counts for each possible real categorical value.

    Returns:
        The observed and expected frequencies.
    """
    
    if is_datetime(original):
        original = pd.to_numeric(original, errors = 'coerce', downcast='integer')
        synthetic = pd.to_numeric(synthetic, errors = 'coerce', downcast='integer')
        
    if original.dtype.name in ("object", "category", "bool"):  # categorical
        gt = original.value_counts().to_dict()
        synth = synthetic.value_counts().to_dict()
        all_keys = gt.keys() | synth.keys()
        for key in all_keys:
            gt.setdefault(key, 0)
            synth.setdefault(key, 0)
    elif np.issubdtype(original.dtype, np.number):  # continuous
        original = original.dropna()
        synthetic = synthetic.dropna()
        if type(bins) != np.ndarray:
            combined = pd.concat([original, synthetic])
            bins = np.histogram_bin_edges(combined, bins=bins)
        gt_vals, _ = np.histogram(original, bins=bins)
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
    if return_keys:
        return frequencies, list(gt.keys())

    return frequencies