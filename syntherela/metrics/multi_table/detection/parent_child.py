from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection.denormalized_detection import DenormalizedDetection

class ParentChildDetection(DenormalizedDetection):
    @staticmethod
    def is_applicable(metadata: Metadata, table1: str, table2: str):
        """
        Check if the table contains at least one column that is not an id.
        And if the table has a relationship with another table.
        """
        nonid1 = False
        table_metadata = metadata.tables[table1].to_dict()
        for column_name in table_metadata['columns'].keys():
            if table_metadata['columns'][column_name]['sdtype'] != 'id':
                nonid1 = True
                break
        nonid2 = False
        table_metadata = metadata.tables[table2].to_dict()
        for column_name in table_metadata['columns'].keys():
            if table_metadata['columns'][column_name]['sdtype'] != 'id':
                nonid2 = True
                break
        return nonid1 and nonid2
    

    def prepare_data(self, real_data, synthetic_data, metadata, parent_table, child_table, pair_metadata):
        real_data_pair = {parent_table: real_data[parent_table], child_table: real_data[child_table]}
        synthetic_data_pair = {parent_table: synthetic_data[parent_table], child_table: synthetic_data[child_table]}
        return super().prepare_data(real_data_pair, synthetic_data_pair, pair_metadata)


    def run(self, real_data: dict, synthetic_data: dict, metadata: Metadata, **kwargs):
        results = {}
        for relationship in metadata.relationships:
            child_table = relationship['child_table_name']
            child_fk = relationship['child_foreign_key']
            parent_table = relationship['parent_table_name']
            if not self.is_applicable(metadata, parent_table, child_table):
                continue
            pair_meta = metadata.to_dict()
            for table in metadata.get_tables():
                if table != parent_table and table != child_table:
                    pair_meta['tables'].pop(table)
            pair_meta['relationships'] = [relationship]
            pair_metadata = Metadata.load_from_dict(pair_meta)
            results[f'{parent_table}_{child_table}_{child_fk}'] = super().run(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata, parent_table=parent_table, child_table=child_table, pair_metadata=pair_metadata)
        return results