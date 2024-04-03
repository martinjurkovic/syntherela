from relsyndgb.metrics.base import DetectionBaseMetric, SingleColumnMetric


class SingleColumnDetection(DetectionBaseMetric, SingleColumnMetric):
    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical", "datetime", "numerical"]
        