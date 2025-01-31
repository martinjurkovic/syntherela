from typing import Union

import graphviz
import pandas as pd
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from sdv.metadata.visualization import _get_graphviz_extension


class Metadata(MultiTableMetadata):
    def __init__(self, dataset_name=""):
        super().__init__()
        self.dataset_name = dataset_name

    def get_tables(self):
        return list(self.tables.keys())

    def get_primary_key(self, table_name: str) -> str:
        table_meta: SingleTableMetadata = self.tables[table_name]
        return table_meta.primary_key

    def get_table_meta(
        self, table_name: str, to_dict: bool = True
    ) -> Union[dict, SingleTableMetadata]:
        table_meta: SingleTableMetadata = self.tables[table_name]
        if to_dict:
            return table_meta.to_dict()
        return table_meta

    def get_children(self, table_name: str) -> set:
        children = set()
        for relation in self.relationships:
            if relation["parent_table_name"] == table_name:
                children.add(relation["child_table_name"])
        return children

    def get_parents(self, table_name: str) -> set:
        parents = set()
        for relation in self.relationships:
            if relation["child_table_name"] == table_name:
                parents.add(relation["parent_table_name"])
        return parents

    def get_foreign_keys(self, parent_table_name: str, child_table_name: str) -> list:
        return self._get_foreign_keys(parent_table_name, child_table_name)

    def rename_column(
        self, table_name: str, old_column_name: str, new_column_name: str
    ):
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

    def get_root_tables(self) -> list:
        root_tables = set(self.tables.keys())
        for relation in self.relationships:
            root_tables.discard(relation["child_table_name"])
        return list(root_tables)

    def get_table_levels(self) -> dict:
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

    def visualize(self, output_filename=None) -> graphviz.Digraph:
        """Generates a Graphviz node with an HTML-like table label.

        Args:
            output_filename: The name of the output file.
        """
        try:
            filename, graphviz_extension = _get_graphviz_extension(output_filename)
        except ValueError:
            raise ValueError(
                "Unable to save a visualization with this file type. Try a supported file type like "
                "'png', 'jpg' or 'pdf'. For a full list, see 'https://graphviz.org/docs/outputs/'"
            )

        colors = {
            "id": "#e2edf1",
            "numerical": "#f2f2f2",
            "categorical": "#f2f2f2",
            "datetime": "#f2f2f2",
        }

        def create_table_node(table_name: str, metadata: Metadata, font: str = "Arial"):
            table_meta = metadata.get_table_meta(table_name)
            table_label = (
                '< <table cellpadding="0" cellborder="0" cellspacing="0" border="0">'
            )
            table_label += f'<tr><td bgcolor="#476893">  </td> <td align="left" bgcolor="#476893"><font color="white"><b>{table_name}</b></font></td> <td align="right" bgcolor="#476893"></td></tr>'
            primary_key = metadata.get_primary_key(table_name)
            for col, info in table_meta["columns"].items():
                sdtype = info["sdtype"]
                fontspec = f'face="{font}"'
                if col == primary_key:
                    col = f"<u><b>{col}</b></u>"
                table_label += f'<tr><td bgcolor="{colors[sdtype]}">  </td> <td align="left" bgcolor="{colors[sdtype]}"><font color="#6e6e6e"  {fontspec}>{col} </font></td> <td align="right" bgcolor="{colors[sdtype]}"><font color="#9b9c9c" {fontspec}>{sdtype}</font></td></tr>'
            table_label += "</table> >"
            return table_label

        dot = graphviz.Digraph(
            graph_attr={"splines": "ortho", "ranksep": "0.8"},
            node_attr={"shape": "plaintext"},
        )

        for table_name in self.get_tables():
            dot.node(
                table_name,
                shape="plain",
                label=create_table_node(table_name, self),
                fontname="Arial",
            )

        for relationship in self.relationships:
            parent_table = relationship["parent_table_name"]
            child_table = relationship["child_table_name"]
            dot.edge(
                parent_table,
                child_table,
                arrowhead="crow",
                arrowtail="tee",
                color="#78a9d2",
                arrowsize="0.9",
            )

        if filename:
            dot.render(filename=filename, cleanup=True, format=graphviz_extension)
        else:
            try:
                graphviz.version()
            except graphviz.ExecutableNotFound:
                from warnings import warn

                warning_message = (
                    "Graphviz does not seem to be installed on this system. For full "
                    "metadata visualization capabilities, please make sure to have its "
                    "binaries propertly installed: https://graphviz.gitlab.io/download/"
                )
                warn(warning_message, RuntimeWarning)
        return dot


def drop_ids(table: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    for column, column_info in metadata["columns"].items():
        if column_info["sdtype"] == "id" and column in table.columns:
            table = table.drop(columns=column, axis=1)
    return table


def convert_metadata_to_v0(metadata: Metadata) -> dict:
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
