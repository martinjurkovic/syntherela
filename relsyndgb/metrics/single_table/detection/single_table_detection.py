from relsyndgb.metadata import drop_ids
from relsyndgb.metrics.base import DetectionBaseMetric, SingleTableMetric

class SingleTableDetection(DetectionBaseMetric, SingleTableMetric):

    def __init__(self, classifier_cls, classifier_args = {}, **kwargs):
        super().__init__(classifier_cls, classifier_args=classifier_args, **kwargs)
        self.name = f"SingleTableDetection-{classifier_cls.__name__}"

    def prepare_data(self, real_data, synthetic_data, metadata, **kwargs):
        real_data = drop_ids(real_data, metadata)
        synthetic_data = drop_ids(synthetic_data, metadata)
        return super().prepare_data(real_data, synthetic_data)
    