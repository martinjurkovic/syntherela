from relsyndgb.metadata import drop_ids
from relsyndgb.data import denormalize_tables, make_column_names_unique
from relsyndgb.metrics.base import DetectionBaseMetric
from copy import deepcopy

class DenormalizedDetection(DetectionBaseMetric):

    def __init__(self, classifier_cls, classifier_args = {}, **kwargs):
        super().__init__(classifier_cls, classifier_args=classifier_args, **kwargs)
        self.name = f"DenormalizedDetection-{classifier_cls.__name__}"

    def prepare_data(self, real_data, synthetic_data, metadata):
        real_data_unique, synthetic_data_unique, metadata_unique = make_column_names_unique(real_data.copy(), synthetic_data.copy(), deepcopy(metadata))
        denormalized_real_data = denormalize_tables(real_data_unique, metadata_unique)
        denormalized_synthetic_data = denormalize_tables(synthetic_data_unique, metadata_unique)
        for table in metadata_unique.get_tables():
            table_metadata = metadata_unique.tables[table].to_dict()
            denormalized_real_data = drop_ids(denormalized_real_data, table_metadata)
            denormalized_synthetic_data = drop_ids(denormalized_synthetic_data, table_metadata)
        return super().prepare_data(denormalized_real_data, denormalized_synthetic_data)
    