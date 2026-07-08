"""Metrics for evaluating synthetic RDB quality across related tables."""

from .detection import AggregationDetection
from .statistical import CardinalityShapeSimilarity
from .trends import multi_table_trends

__all__ = [
    'AggregationDetection',
    'CardinalityShapeSimilarity',
    'multi_table_trends',
]
