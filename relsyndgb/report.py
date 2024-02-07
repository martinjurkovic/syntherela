from datetime import datetime

from relsyndgb.metrics.single_column.distance import HellingerDistance
from relsyndgb.metrics.single_column.statistical import ChiSquareTest

class Report():

    def __init__(self, 
                real_data,
                synthetic_data, 
                metadata,
                report_name, 
                single_col_metrics = [ChiSquareTest(), HellingerDistance()], 
                single_table_metrics = [], 
                multi_table_metrics = [],
                ):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata
        self.report_name = report_name
        self.single_col_metrics = single_col_metrics
        self.single_table_metrics = single_table_metrics
        self.multi_table_metrics = multi_table_metrics
        self.report_datetime = datetime.now()
        self.results = {
            "single_col_metrics": {},
            "single_table_metrics": {},
            "multi_table_metrics": {},
        }

    def generate(self):
        
        # TODO: Validate the input data

        for table in self.metadata.get_tables():
            # single_col_metrics
            for column, column_info in self.metadata.tables[table].columns.items():
                for metric in self.single_col_metrics:
                    if metric.is_applicable(column_info["sdtype"]):
                        self.results["single_col_metrics"].setdefault(metric.name, {}).setdefault(table, {})[column] = metric.run(
                            self.real_data[table][column],
                            self.synthetic_data[table][column],
                        )

            # single_table_metrics
            for metric in self.single_table_metrics:
                self.results["single_table_metrics"].setdefault(metric.name, {})[table] = metric.run(
                    self.real_data[table],
                    self.synthetic_data[table],
                )

        self.report_datetime = datetime.now()

        return self.results
