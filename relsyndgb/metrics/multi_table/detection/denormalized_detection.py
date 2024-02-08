from relsyndgb.metadata import drop_ids
from relsyndgb.data import denormalize_tables
from relsyndgb.metrics.base import DetectionBaseMetric

class DenormalizedDetection(DetectionBaseMetric):

    def __init__(self, classifier_cls, classifier_args = {}, **kwargs):
        super().__init__(classifier_cls, classifier_args=classifier_args, **kwargs)
        self.name = f"DenormalizedDetection-{classifier_cls.__name__}"

    def prepare_data(self, real_data, synthetic_data, metadata):
        denormalized_real_data = denormalize_tables(real_data, metadata)
        denormalized_synthetic_data = denormalize_tables(synthetic_data, metadata)
        for table in metadata.get_tables():
            table_metadata = metadata.tables[table]
            denormalized_real_data = drop_ids(denormalized_real_data, table_metadata)
            denormalized_synthetic_data = drop_ids(denormalized_synthetic_data, table_metadata)

        return super().prepare_data(denormalized_real_data, denormalized_synthetic_data, metadata)
    