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
    
    def rename_column(self, table_name, old_column_name, new_column_name):
        self.tables[table_name].columns[new_column_name] = self.tables[table_name].columns.pop(old_column_name)
        if self.tables[table_name].columns[new_column_name]['sdtype'] != 'id':
            return self
        
        if self.tables[table_name].primary_key == old_column_name:
            self.tables[table_name].primary_key = new_column_name
        
        for relationship in self.relationships:
            if relationship['parent_table_name'] == table_name and relationship['parent_primary_key'] == old_column_name:
                relationship['parent_primary_key'] = new_column_name
            if relationship['child_table_name'] == table_name and relationship['child_foreign_key'] == old_column_name:
                relationship['child_foreign_key'] = new_column_name
        return self

def drop_ids(table, metadata: dict):
    for column, column_info in metadata['columns'].items():
        if column_info['sdtype'] == 'id' and column in table.columns:
            table = table.drop(columns = column, axis=1)
    return table