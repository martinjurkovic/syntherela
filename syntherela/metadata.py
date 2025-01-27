import pandas as pd
from sdv.metadata import MultiTableMetadata


class Metadata(MultiTableMetadata):
    def __init__(self, dataset_name=""):
        super().__init__()
        self.dataset_name = dataset_name

    def get_tables(self):
        return list(self.tables.keys())

    def get_primary_key(self, table_name):
        return self.tables[table_name].primary_key

    def get_table_meta(self, table_name, to_dict=True):
        table_meta = self.tables[table_name]
        if to_dict:
            return table_meta.to_dict()
        return table_meta

    def get_children(self, table_name):
        children = set()
        for relation in self.relationships:
            if relation["parent_table_name"] == table_name:
                children.add(relation["child_table_name"])
        return children

    def get_parents(self, table_name):
        parents = set()
        for relation in self.relationships:
            if relation["child_table_name"] == table_name:
                parents.add(relation["parent_table_name"])
        return parents

    def get_foreign_keys(self, parent_table_name, child_table_name):
        return self._get_foreign_keys(parent_table_name, child_table_name)

    def rename_column(self, table_name, old_column_name, new_column_name):
        self.tables[table_name].columns[new_column_name] = self.tables[
            table_name
        ].columns.pop(old_column_name)
        if self.tables[table_name].columns[new_column_name]["sdtype"] != "id":
            return self

        if self.tables[table_name].primary_key == old_column_name:
            self.tables[table_name].primary_key = new_column_name

        for relationship in self.relationships:
            if (
                relationship["parent_table_name"] == table_name
                and relationship["parent_primary_key"] == old_column_name
            ):
                relationship["parent_primary_key"] = new_column_name
            if (
                relationship["child_table_name"] == table_name
                and relationship["child_foreign_key"] == old_column_name
            ):
                relationship["child_foreign_key"] = new_column_name
        return self

    def get_root_tables(self):
        root_tables = set(self.tables.keys())
        for relation in self.relationships:
            root_tables.discard(relation["child_table_name"])
        return list(root_tables)

    def get_table_levels(self):
        # return the length of the path from any root table
        root_tables = self.get_root_tables()
        table_levels = {}
        for root_table in root_tables:
            table_levels[root_table] = 0

        relationships = self.relationships.copy()
        while len(relationships) > 0:
            relationship = relationships.pop(0)
            if relationship["parent_table_name"] in table_levels:
                table_levels[relationship["child_table_name"]] = (
                    table_levels[relationship["parent_table_name"]] + 1
                )
            else:
                relationships.append(relationship)
        return table_levels


def drop_ids(table: pd.DataFrame, metadata: dict):
    for column, column_info in metadata["columns"].items():
        if column_info["sdtype"] == "id" and column in table.columns:
            table = table.drop(columns=column, axis=1)
    return table


def convert_metadata_to_v0(metadata):
    metadata_v1 = metadata.to_dict()
    metadata_v0 = {"tables": {}}
    for table_name, table_info in metadata_v1["tables"].items():
        metadata_v0["tables"][table_name] = {"fields": {}}
        for column, column_info in table_info["columns"].items():
            metadata_v0["tables"][table_name]["fields"][column] = {
                "type": column_info["sdtype"]
            }
            if column_info["sdtype"] == "boolean":
                # convert boolean to categorical
                metadata_v0["tables"][table_name]["fields"][column]["type"] = (
                    "categorical"
                )
            if column_info["sdtype"] == "datetime":
                metadata_v0["tables"][table_name]["fields"][column]["format"] = (
                    column_info["datetime_format"]
                )

        if "primary_key" in table_info:
            metadata_v0["tables"][table_name]["fields"][table_info["primary_key"]] = {
                "type": "id",
                "subtype": "string",
            }
            metadata_v0["tables"][table_name]["primary_key"] = table_info["primary_key"]

    for relationship in metadata_v1["relationships"]:
        parent_table_name = relationship["parent_table_name"]
        child_table_name = relationship["child_table_name"]
        parent_primary_key = relationship["parent_primary_key"]
        child_foreign_key = relationship["child_foreign_key"]
        metadata_v0["tables"][child_table_name]["fields"][child_foreign_key]["ref"] = {
            "table": parent_table_name,
            "field": parent_primary_key,
        }
        metadata_v0["tables"][child_table_name]["fields"][child_foreign_key][
            "subtype"
        ] = "string"
    return metadata_v0
