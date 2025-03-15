"""Single table column pair trends report for multi-table data.

Based on https://github.com/sdv-dev/SDMetrics/blob/main/sdmetrics/reports/multi_table_report/quality_report.py.
"""

from sdmetrics.reports.multi_table._properties import ColumnPairTrends
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class ColumnPairsReport(BaseMultiTableReport):
    """Single table column pair trends report.

    This class creates a quality report for multi-table data. It calculates the column pair
    trends for all of the tables in the dataset.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            "Column Pair Trends": ColumnPairTrends(),
        }
