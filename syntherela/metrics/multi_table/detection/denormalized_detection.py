from typing import Union

import pandas as pd
from syntherela.metadata import drop_ids, Metadata
from syntherela.data import denormalize_tables, make_column_names_unique
from syntherela.metrics.base import DetectionBaseMetric
from copy import deepcopy

class DenormalizedDetection(DetectionBaseMetric):
    @staticmethod
    def bootstrap_sample(real_data: dict, metadata: Metadata, random_state: Union[int, None] = None) -> dict:
        bootstrapped_tables = dict()
        root_tables = metadata.get_root_tables()
        # TODO: deal with duplicated parent samples (should reindex them --> and their children)
        for table in root_tables:
            primary_key = metadata.get_primary_key(table)
            bootstrapped_tables[table] = real_data[table].sample(frac=1, replace=True, random_state=random_state)
            bootstrapped_tables[table][f'{primary_key}_duplicated'] = bootstrapped_tables[table][primary_key].duplicated()
        for relationship in metadata.relationships:
            child_table = relationship['child_table_name']
            child_fk = relationship['child_foreign_key']
            parent_table = relationship['parent_table_name']
            parent_pk = relationship['parent_primary_key']
            if child_table in bootstrapped_tables:
                child_table_pk = metadata.get_primary_key(child_table)
                new_data = real_data[child_table][real_data[child_table][child_fk].isin(bootstrapped_tables[parent_table][parent_pk])]
                new_data.set_index(child_table_pk, inplace=True)
                bootstrapped_tables[child_table].set_index(child_table_pk, inplace=True)
                bootstrapped_tables[child_table] = bootstrapped_tables[child_table].combine_first(new_data).reset_index()
            else:
                bootstrapped_tables[child_table] = real_data[child_table][real_data[child_table][child_fk].isin(bootstrapped_tables[parent_table][parent_pk])]

        return bootstrapped_tables
    

    def prepare_data(self, real_data, synthetic_data, metadata):
        real_data_unique, synthetic_data_unique, metadata_unique = make_column_names_unique(real_data.copy(), synthetic_data.copy(), deepcopy(metadata), validate=False)
        denormalized_real_data = denormalize_tables(real_data_unique, metadata_unique)
        denormalized_synthetic_data = denormalize_tables(synthetic_data_unique, metadata_unique)
        for table in metadata_unique.get_tables():
            table_metadata = metadata_unique.tables[table].to_dict()
            denormalized_real_data = drop_ids(denormalized_real_data, table_metadata)
            denormalized_synthetic_data = drop_ids(denormalized_synthetic_data, table_metadata)
        return super().prepare_data(denormalized_real_data, denormalized_synthetic_data)
    