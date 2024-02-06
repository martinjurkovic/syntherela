from sdmetrics.base import BaseMetric
import sdv

class SingleColumnMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_applicable(self, column_type):
        raise NotImplementedError()
