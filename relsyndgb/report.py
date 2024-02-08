from datetime import datetime
from sklearn.linear_model import LogisticRegression

from relsyndgb.metrics.single_column.distance import HellingerDistance
from relsyndgb.metrics.single_column.statistical import ChiSquareTest
from relsyndgb.metrics.single_table.distance import MaximumMeanDiscrepancy
from relsyndgb.metrics.single_column.detection import SingleColumnDetection

class Report():

    def __init__(self, 
                real_data,
                synthetic_data, 
                metadata,
                report_name, 
                single_col_metrics = [
                    ChiSquareTest(), 
                    HellingerDistance(),
                    SingleColumnDetection(classifier_cls=LogisticRegression, classifier_args={"solver": "lbfgs"})],
                single_table_metrics = [MaximumMeanDiscrepancy()], 
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
                    if not metric.is_applicable(column_info["sdtype"]):
                        continue
                    self.results["single_col_metrics"].setdefault(metric.name, {}).setdefault(table, {})[column] = metric.run(
                        self.real_data[table][column],
                        self.synthetic_data[table][column],
                        metadata = 'TODO',
                    )

            # single_table_metrics
            for metric in self.single_table_metrics:
                self.results["single_table_metrics"].setdefault(metric.name, {})[table] = metric.run(
                    self.real_data[table],
                    self.synthetic_data[table],
                    metadata = self.metadata.to_dict()['tables'][table],
                )

        self.report_datetime = datetime.now()

        return self.results
