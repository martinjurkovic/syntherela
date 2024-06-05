from copy import deepcopy

import pandas as pd
import numpy as np 

from relsyndgb.metadata import drop_ids
from relsyndgb.metrics.base import DetectionBaseMetric, SingleTableMetric
from .denormalized_detection import DenormalizedDetection
from .parent_child import ParentChildDetection

class BaseAggregationDetection(DetectionBaseMetric):
    @staticmethod
    def add_aggregations(data, metadata, update_metadata=True):
        aggregated_data = deepcopy(data)
        for relationship in metadata.relationships:
            parent_table_name = relationship['parent_table_name']
            child_table_name = relationship['child_table_name']
            parent_column = relationship['parent_primary_key']
            child_fk = relationship['child_foreign_key']

            # add child counts
            child_df = pd.DataFrame({f'{child_table_name}_{child_fk}_counts': data[child_table_name][child_fk].value_counts()})
            cardinality_df = pd.DataFrame({'parent': data[parent_table_name][parent_column]}).join(
                child_df, on='parent').fillna(0)
            aggregated_data[parent_table_name] = aggregated_data[parent_table_name].merge(
                cardinality_df, how='left', left_on=parent_column, right_on='parent').drop(columns='parent')
            
            if update_metadata:
                metadata.add_column(parent_table_name, f'{child_table_name}_{child_fk}_counts', sdtype='numerical')

            # add categorical counts
            categorical_columns = []
            for column_name, column_info in metadata.tables[child_table_name].to_dict()['columns'].items():
                if column_info['sdtype'] == 'categorical':
                    categorical_columns.append(column_name)

            if len(categorical_columns) > 0:
                categorical_df = data[child_table_name][categorical_columns + [child_fk]]
                categorical_column_names = [f'{child_table_name}_{child_fk}_{column}_nunique' for column in categorical_columns]
                categorical_df.columns = categorical_column_names + [child_fk]

                aggregated_data[parent_table_name] = aggregated_data[parent_table_name].merge(
                    categorical_df.groupby(child_fk).nunique(), how='left', left_on=parent_column, right_index=True, suffixes=('', '_nunique'))
                aggregated_data[parent_table_name][categorical_column_names] = aggregated_data[parent_table_name][categorical_column_names].fillna(0)

                if update_metadata:
                    for column in categorical_column_names:
                        metadata.add_column(parent_table_name, column, sdtype='numerical')

            # add numerical means
            numerical_columns = []
            for column_name, column_info in metadata.tables[child_table_name].to_dict()['columns'].items():
                if column_info['sdtype'] == 'numerical':
                    numerical_columns.append(column_name)

            if len(numerical_columns) > 0:
                numerical_df = data[child_table_name][numerical_columns + [child_fk]]
                numerical_column_names = [f'{child_table_name}_{child_fk}_{column}_mean' for column in numerical_columns]
                numerical_df.columns = numerical_column_names + [child_fk]

                aggregated_data[parent_table_name] = aggregated_data[parent_table_name].merge(
                    numerical_df.groupby(child_fk).mean(), how='left', left_on=parent_column, right_index=True, suffixes=('', '_mean'))
                
                if update_metadata:
                    for column in numerical_column_names:
                        metadata.add_column(parent_table_name, column, sdtype='numerical')

        return aggregated_data, metadata


class AggregationDetection(BaseAggregationDetection, DetectionBaseMetric, SingleTableMetric):
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
    
    # TODO: add baseline
    def run(self, real_data, synthetic_data, metadata, target_table=None, **kwargs):
        real_data_with_aggregations, updated_metadata = self.add_aggregations(real_data, deepcopy(metadata))
        synthetic_data_with_aggregations, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        results = {}
        if target_table is not None:
            table_metadata = metadata.tables[target_table].to_dict()
            real_data_with_aggregations[target_table] = drop_ids(real_data_with_aggregations[target_table], table_metadata)
            synthetic_data_with_aggregations[target_table] = drop_ids(synthetic_data_with_aggregations[target_table], table_metadata)
            return super().run(real_data_with_aggregations[target_table], synthetic_data_with_aggregations[target_table], metadata=updated_metadata, **kwargs)
        
        for table in metadata.get_tables():
            table_metadata = metadata.tables[table].to_dict()
            if not self.is_applicable(updated_metadata, table):
                continue
            real_data_with_aggregations[table] = drop_ids(real_data_with_aggregations[table], table_metadata)
            synthetic_data_with_aggregations[table] = drop_ids(synthetic_data_with_aggregations[table], table_metadata)
            
            results[table] = super().run(real_data_with_aggregations[table], synthetic_data_with_aggregations[table], metadata=updated_metadata, **kwargs)
        return results


class DenormalizedAggregationDetection(DenormalizedDetection, BaseAggregationDetection):
    def prepare_data(self, real_data, synthetic_data, metadata):
        aggregated_real_data, updated_metadata = self.add_aggregations(real_data, deepcopy(metadata))
        aggregated_synthetic_data, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        return super().prepare_data(aggregated_real_data, aggregated_synthetic_data, updated_metadata)
    

class ParentChildAggregationDetection(ParentChildDetection, BaseAggregationDetection):
    def prepare_data(self, real_data, synthetic_data, metadata, parent_table, child_table, pair_metadata):
        aggregated_real_data, updated_metadata = self.add_aggregations(real_data, deepcopy(metadata))
        aggregated_synthetic_data, _ = self.add_aggregations(synthetic_data, metadata, update_metadata=False)
        return super().prepare_data(aggregated_real_data, aggregated_synthetic_data, updated_metadata, parent_table, child_table, pair_metadata)