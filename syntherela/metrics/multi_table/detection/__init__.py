"""Detection metrics (C2ST) for comparing real and synthetic databases."""

from .aggregation_detection import (
    AggregationDetection,
    ParentChildAggregationDetection,
)
from .parent_child import ParentChildDetection

__all__ = [
    'AggregationDetection',
    'ParentChildDetection',
    'ParentChildAggregationDetection',
]
