from syntherela.metadata import drop_ids
from syntherela.metrics.base import DetectionBaseMetric, SingleTableMetric

class SingleTableDetection(DetectionBaseMetric, SingleTableMetric):
    def prepare_data(self, real_data, synthetic_data, metadata, **kwargs):
        real_data = drop_ids(real_data, metadata)
        synthetic_data = drop_ids(synthetic_data, metadata)
        return super().prepare_data(real_data, synthetic_data)
    