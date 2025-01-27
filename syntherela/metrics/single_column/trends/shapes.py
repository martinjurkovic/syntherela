from sdmetrics.reports.multi_table._properties import ColumnShapes
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class ColumnShapesReport(BaseMultiTableReport):
    """Single column shapes report

    This class creates a quality report for multi-table data. It calculates the column Shapes
    trends for all of the tables in the dataset.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            "Column Shapes": ColumnShapes(),
        }
