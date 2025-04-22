"""Utility functions for distance metrics in single columns."""

from typing import Union

import numpy as np
import pandas as pd
from sdmetrics.utils import is_datetime


def get_histograms(
    original: pd.Series,
    synthetic: pd.Series,
    normalize: bool = True,
    bins: Union[str, np.array] = "doane",
    return_keys: bool = False,
) -> tuple:
    """Compute histograms for the given data.

    Discretize numerical data into bins and compute the frequencies of each bin or
    compute the frequencies of each category in categorical data.

    Parameters
    ----------
    original: pd.Series
        The original column.
    synthetic: pd.Series
        The synthetic column.
    normalize: bool
        Whether to normalize the frequencies.
    bins: Union[str, np.array]
        The number of bins or the bin edges.
    return_keys: bool
        Whether to return the keys.

    Returns
    -------
        The observed and expected frequencies and the keys if return_keys is True.

    """
    if is_datetime(original):
        original = pd.to_numeric(original, errors="coerce", downcast="integer")
        synthetic = pd.to_numeric(synthetic, errors="coerce", downcast="integer")

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
        if type(bins) is not np.ndarray:
            combined = pd.concat([original, synthetic])
            bins = np.histogram_bin_edges(combined, bins=bins)
        gt_vals, _ = np.histogram(original, bins=bins)
        synth_vals, _ = np.histogram(synthetic, bins=bins)
        gt = {k: v for k, v in zip(bins, gt_vals)}
        synth = {k: v for k, v in zip(bins, synth_vals)}
    else:
        raise ValueError("Column is not categorical or continouous")

    # order the keys
    gt = {k: v for k, v in sorted(gt.items(), key=lambda item: item[0])}
    synth = {k: v for k, v in sorted(synth.items(), key=lambda item: item[0])}

    assert gt.keys() == synth.keys(), "Keys do not match for column"

    if normalize:
        gt_sum = sum(gt.values())
        synth_sum = sum(synth.values())
        gt = {k: v / gt_sum for k, v in gt.items()}
        synth = {k: v / synth_sum for k, v in synth.items()}

    frequencies = (np.array(list(gt.values())), np.array(list(synth.values())))
    if return_keys:
        return frequencies, list(gt.keys())

    return frequencies
