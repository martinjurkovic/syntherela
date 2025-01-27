"""
Based on https://github.com/sdv-dev/SDMetrics/blob/main/sdmetrics/reports/multi_table_report/quality_report.py
"""

from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    InterTableTrends,
)
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class MultiTableTrendsReport(BaseMultiTableReport):
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along Intertable Trends, and Cardinality.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            "Cardinality": Cardinality(),
            "Intertable Trends": InterTableTrends(),
        }
