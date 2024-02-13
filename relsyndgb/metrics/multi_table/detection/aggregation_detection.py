from copy import deepcopy

import pandas as pd
import numpy as np 

from relsyndgb.metadata import drop_ids
from relsyndgb.metrics.base import DetectionBaseMetric, SingleTableMetric
from .denormalized_detection import DenormalizedDetection

class AggregationDetection(DetectionBaseMetric):
    @staticmethod
    def add_aggregations(data, metadata, update_metadata=True):
        aggregated_data = deepcopy(data)
        for relationship in metadata.relationships:
            parent_table_name = relationship['parent_table_name']
            child_table_name = relationship['child_table_name']
            parent_column = relationship['parent_primary_key']
            child_column = relationship['child_foreign_key']

            child_df = pd.DataFrame({f'{child_table_name}_{child_column}_counts': data[child_table_name][child_column].value_counts()})
            cardinality_df = pd.DataFrame({'parent': data[parent_table_name][parent_column]}).join(
                child_df, on='parent').fillna(0)
            aggregated_data[parent_table_name] = aggregated_data[parent_table_name].merge(
                cardinality_df, how='left', left_on=parent_column, right_on='parent').drop(columns='parent')
            
            if not update_metadata:
                continue
            
            metadata.add_column(parent_table_name, f'{child_table_name}_{child_column}_counts', sdtype='numerical')

        return aggregated_data, metadata


class SingleTableAggregationDetection(AggregationDetection, DetectionBaseMetric, SingleTableMetric):

    def __init__(self, classifier_cls, classifier_args={}, **kwargs):
        super().__init__(classifier_cls, classifier_args, **kwargs)
        self.name = f"SingleTableAggregationDetection-{classifier_cls.__name__}"

    def run(self, real_data, synthetic_data, metadata, **kwargs):
        real_data_with_aggregations, metadata = self.add_aggregations(real_data, deepcopy(metadata))
        synthetic_data_with_aggregations, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        results = {}
        for table in metadata.get_tables():
            table_metadata = metadata.tables[table].to_dict()
            if not self.is_applicable(table_metadata):
                continue
            real_data_with_aggregations[table] = drop_ids(real_data_with_aggregations[table], table_metadata)
            synthetic_data_with_aggregations[table] = drop_ids(synthetic_data_with_aggregations[table], table_metadata)
            
            scores = self.compute(real_data_with_aggregations[table], synthetic_data_with_aggregations[table], metadata=metadata, **kwargs)
            _, bin_test_p_val = self.binomial_test(sum(scores), len(scores))
            standard_error = np.std(scores) / np.sqrt(len(scores))
            results[table] = { "accuracy": np.mean(scores), "SE": standard_error, "bin_test_p_val" : bin_test_p_val }
        return results


class DenormalizedAggregationDetection(DenormalizedDetection, AggregationDetection):

    def __init__(self, classifier_cls, classifier_args = {}, **kwargs):
        super().__init__(classifier_cls, classifier_args=classifier_args, **kwargs)
        self.name = f"DenormalizedAggregationDetection-{classifier_cls.__name__}"

    def prepare_data(self, real_data, synthetic_data, metadata):
        aggregated_real_data, metadata = self.add_aggregations(real_data, deepcopy(metadata))
        aggregated_synthetic_data, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        return super().prepare_data(aggregated_real_data, aggregated_synthetic_data, metadata)
    
    