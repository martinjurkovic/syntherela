from sdv.metadata import MultiTableMetadata


class Metadata(MultiTableMetadata):

    def __init__(self, dataset_name=''):
        super().__init__()
        self.dataset_name = dataset_name

    def get_tables(self):
        return list(self.tables.keys())
    
    def get_primary_key(self, table_name):
        return self.tables[table_name].primary_key
    
    def get_children(self, table_name):
        children = []
        for relation in self.relationships:
            if relation['parent_table_name'] == table_name:
                children.append(relation['child_table_name'])
        return children
    
    def get_foreign_keys(self, parent_table_name, child_table_name):
        return self._get_foreign_keys(parent_table_name, child_table_name)
    

def drop_ids(table, metadata):
    for column, column_info in metadata['columns'].items():
        if column_info['sdtype'] == 'id':
            table = table.drop(columns = column, axis=1)
    return table