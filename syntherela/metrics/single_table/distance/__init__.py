"""Distance metrics for measuring fidelity of synthetic tables."""

from .maximum_mean_discrepancy import MaximumMeanDiscrepancy
from .pairwise_correlation_difference import PairwiseCorrelationDifference

__all__ = ["MaximumMeanDiscrepancy", "PairwiseCorrelationDifference"]
