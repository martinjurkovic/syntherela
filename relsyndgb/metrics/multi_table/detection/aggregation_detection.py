from copy import deepcopy

import pandas as pd
import numpy as np 

from relsyndgb.metadata import drop_ids
from relsyndgb.metrics.base import DetectionBaseMetric, SingleTableMetric
from .denormalized_detection import DenormalizedDetection
from .parent_child import ParentChildDetection

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
    @staticmethod
    def is_applicable(metadata, table):
        """
        Check if the table contains at least one column that is not an id.
        And if the table has a relationship with another table.
        """
        nonid = False
        table_metadata = metadata.tables[table].to_dict()
        for column_name in table_metadata['columns'].keys():
            if table_metadata['columns'][column_name]['sdtype'] != 'id':
                nonid = True
                break
        has_children = len(metadata.get_children(table)) > 0
        return nonid and has_children
    
    
    def run(self, real_data, synthetic_data, metadata, **kwargs):
        real_data_with_aggregations, metadata = self.add_aggregations(real_data, deepcopy(metadata))
        synthetic_data_with_aggregations, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        results = {}
        for table in metadata.get_tables():
            table_metadata = metadata.tables[table].to_dict()
            if not self.is_applicable(metadata, table):
                continue
            real_data_with_aggregations[table] = drop_ids(real_data_with_aggregations[table], table_metadata)
            synthetic_data_with_aggregations[table] = drop_ids(synthetic_data_with_aggregations[table], table_metadata)
            
            results[table] = super().run(real_data_with_aggregations[table], synthetic_data_with_aggregations[table], metadata=metadata, **kwargs)
        return results


class DenormalizedAggregationDetection(DenormalizedDetection, AggregationDetection):
    def prepare_data(self, real_data, synthetic_data, metadata):
        aggregated_real_data, metadata = self.add_aggregations(real_data, deepcopy(metadata))
        aggregated_synthetic_data, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        return super().prepare_data(aggregated_real_data, aggregated_synthetic_data, metadata)
    

class ParentChildAggregationDetection(ParentChildDetection, AggregationDetection):
    def run(self, real_data, synthetic_data, metadata, **kwargs):
        real_data_with_aggregations, metadata = self.add_aggregations(real_data, deepcopy(metadata))
        synthetic_data_with_aggregations, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        return super().run(real_data_with_aggregations, synthetic_data_with_aggregations, metadata, **kwargs)