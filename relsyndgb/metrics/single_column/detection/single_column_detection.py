from relsyndgb.metrics.base import DetectionBaseMetric, SingleColumnMetric


class SingleColumnDetection(DetectionBaseMetric, SingleColumnMetric):

    def __init__(self, classifier_cls, classifier_args = {}, **kwargs):
        super().__init__(classifier_cls, classifier_args=classifier_args, **kwargs)
        self.name = f"SingleColumnDetection-{classifier_cls.__name__}"

    @staticmethod
    def is_applicable(column_type):
        return column_type in ["categorical", "datetime", "numerical"]
        