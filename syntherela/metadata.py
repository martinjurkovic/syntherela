"""Metadata handling for synthetic data evaluation.

This module provides classes and functions for managing metadata objects
describing the database schema: tables and their relationships. The on-disk
JSON format is interoperable with SDV's metadata specification.
"""

import copy
import json
import os
from typing import cast

import graphviz
import pandas as pd

MULTI_TABLE_SPEC_VERSION = 'MULTI_TABLE_V1'
SINGLE_TABLE_SPEC_VERSION = 'SINGLE_TABLE_V1'


class InvalidMetadataError(ValueError):
    """Raised when a metadata object is internally inconsistent."""


class InvalidDataError(ValueError):
    """Raised when data does not conform to its metadata."""


def _get_graphviz_extension(
    filepath: str | os.PathLike | None,
) -> tuple[str | None, str | None]:
    """Split a filepath into its name and graphviz output extension.

    Parameters
    ----------
    filepath: str | os.PathLike | None
        Output file path. If None, no file is written.

    Returns
    -------
    tuple[str | None, str | None]
        A ``(filename, extension)`` tuple. Both elements are None when
        ``filepath`` is None.

    Raises
    ------
    ValueError
        If ``filepath`` is provided without a file extension.

    """
    if filepath is None:
        return None, None

    path_str = os.fspath(filepath)
    if '.' not in os.path.basename(path_str):
        raise ValueError('Provide a filepath with a valid extension.')
    filename, extension = path_str.rsplit('.', 1)
    return filename, extension


class SingleTableMetadata:
    """Metadata describing the columns of a single table.

    Columns are stored as plain dictionaries keyed by column name, e.g.
    ``{'sdtype': 'numerical', 'computer_representation': 'Int64'}``. This
    keeps arbitrary column attributes (``datetime_format``, ``regex_format``,
    ...) round-tripping without dedicated per-type classes.
    """

    def __init__(self) -> None:
        """Initialize an empty single-table metadata object."""
        self.columns: dict[str, dict] = {}
        self.primary_key: str | None = None

    def add_column(self, column_name: str, **kwargs) -> None:
        """Add a column to the table.

        Parameters
        ----------
        column_name: str
            Name of the column.
        **kwargs
            Column attributes. Must include ``sdtype``.

        Raises
        ------
        InvalidMetadataError
            If the column already exists or ``sdtype`` is missing.

        """
        if column_name in self.columns:
            raise InvalidMetadataError(
                f"Column '{column_name}' already exists."
            )
        if 'sdtype' not in kwargs:
            raise InvalidMetadataError(
                f"Cannot add column '{column_name}': 'sdtype' is required."
            )
        self.columns[column_name] = dict(kwargs)

    def update_column(self, column_name: str, **kwargs) -> None:
        """Update the attributes of an existing column.

        Parameters
        ----------
        column_name: str
            Name of the column to update.
        **kwargs
            Column attributes to set or overwrite.

        Raises
        ------
        InvalidMetadataError
            If the column does not exist.

        """
        if column_name not in self.columns:
            raise InvalidMetadataError(
                f"Column '{column_name}' does not exist."
            )
        self.columns[column_name].update(kwargs)

    def set_primary_key(self, column_name: str) -> None:
        """Set the primary key of the table.

        Parameters
        ----------
        column_name: str
            Name of the primary key column.

        Raises
        ------
        InvalidMetadataError
            If the column does not exist.

        """
        if column_name not in self.columns:
            raise InvalidMetadataError(
                f"Cannot set primary key to unknown column '{column_name}'."
            )
        self.primary_key = column_name

    def get_column_names(self, **kwargs) -> list[str]:
        """Get column names, optionally filtered by attribute.

        Parameters
        ----------
        **kwargs
            Attribute filters, e.g. ``sdtype='id'``. A column matches when
            all provided attributes equal the column's values.

        Returns
        -------
        list[str]
            List of matching column names.

        """
        return [
            column
            for column, info in self.columns.items()
            if all(info.get(key) == value for key, value in kwargs.items())
        ]

    def detect_from_dataframe(self, data: pd.DataFrame) -> None:
        """Infer column sdtypes from a dataframe.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame to inspect. Booleans map to ``categorical``,
            datetimes to ``datetime``, other numerics to ``numerical`` and
            everything else to ``categorical``.

        """
        for column in data.columns:
            series = data[column]
            if pd.api.types.is_bool_dtype(series):
                sdtype = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(series):
                sdtype = 'datetime'
            elif pd.api.types.is_numeric_dtype(series):
                sdtype = 'numerical'
            else:
                sdtype = 'categorical'
            self.columns[column] = {'sdtype': sdtype}

    def _to_table_dict(self) -> dict:
        """Return the table metadata without the spec version.

        Returns
        -------
        dict
            Dictionary with ``columns`` and, when set, ``primary_key``.

        """
        table_dict: dict = {'columns': copy.deepcopy(self.columns)}
        if self.primary_key is not None:
            table_dict['primary_key'] = self.primary_key
        return table_dict

    def to_dict(self) -> dict:
        """Convert the metadata to a dictionary.

        Returns
        -------
        dict
            Dictionary representation including ``METADATA_SPEC_VERSION``.

        """
        return {
            'METADATA_SPEC_VERSION': SINGLE_TABLE_SPEC_VERSION,
            **self._to_table_dict(),
        }

    @classmethod
    def load_from_dict(cls, metadata_dict: dict) -> 'SingleTableMetadata':
        """Create a SingleTableMetadata from a dictionary.

        Parameters
        ----------
        metadata_dict: dict
            Dictionary with ``columns`` and optional ``primary_key``.

        Returns
        -------
        SingleTableMetadata
            The reconstructed metadata object.

        """
        instance = cls()
        instance.columns = copy.deepcopy(metadata_dict.get('columns', {}))
        instance.primary_key = metadata_dict.get('primary_key')
        return instance


class Metadata:
    """Metadata describing a relational database: tables and relationships.

    Parameters
    ----------
    dataset_name: str, default=""
        Name of the dataset.

    """

    def __init__(self, dataset_name: str = '') -> None:
        """Initialize the Metadata object.

        Parameters
        ----------
        dataset_name: str, default=""
            Name of the dataset.

        """
        self.dataset_name = dataset_name
        self.tables: dict[str, SingleTableMetadata] = {}
        self.relationships: list[dict] = []

    # -- Construction / mutation ------------------------------------------

    def add_table(self, table_name: str) -> None:
        """Add an empty table to the metadata.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Raises
        ------
        InvalidMetadataError
            If the table already exists.

        """
        if table_name in self.tables:
            raise InvalidMetadataError(f"Table '{table_name}' already exists.")
        self.tables[table_name] = SingleTableMetadata()

    def _get_table(self, table_name: str) -> SingleTableMetadata:
        """Return a table's metadata, raising if it is unknown.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Returns
        -------
        SingleTableMetadata
            The table's metadata object.

        Raises
        ------
        InvalidMetadataError
            If the table does not exist.

        """
        if table_name not in self.tables:
            raise InvalidMetadataError(f"Unknown table '{table_name}'.")
        return self.tables[table_name]

    def add_column(self, table_name: str, column_name: str, **kwargs) -> None:
        """Add a column to a table.

        Parameters
        ----------
        table_name: str
            Name of the table.
        column_name: str
            Name of the column.
        **kwargs
            Column attributes. Must include ``sdtype``.

        """
        self._get_table(table_name).add_column(column_name, **kwargs)

    def update_column(
        self, table_name: str, column_name: str, **kwargs
    ) -> None:
        """Update the attributes of a column in a table.

        Parameters
        ----------
        table_name: str
            Name of the table.
        column_name: str
            Name of the column.
        **kwargs
            Column attributes to set or overwrite.

        """
        self._get_table(table_name).update_column(column_name, **kwargs)

    def set_primary_key(self, table_name: str, column_name: str) -> None:
        """Set the primary key of a table.

        Parameters
        ----------
        table_name: str
            Name of the table.
        column_name: str
            Name of the primary key column.

        """
        self._get_table(table_name).set_primary_key(column_name)

    def add_relationship(
        self,
        parent_table_name: str,
        child_table_name: str,
        parent_primary_key: str,
        child_foreign_key: str,
    ) -> None:
        """Add a parent-child relationship between two tables.

        Parameters
        ----------
        parent_table_name: str
            Name of the parent table.
        child_table_name: str
            Name of the child table.
        parent_primary_key: str
            Primary key column in the parent table.
        child_foreign_key: str
            Foreign key column in the child table.

        Raises
        ------
        InvalidMetadataError
            If a referenced table or key column does not exist.

        """
        parent = self._get_table(parent_table_name)
        child = self._get_table(child_table_name)
        if parent_primary_key not in parent.columns:
            raise InvalidMetadataError(
                f"Unknown primary key '{parent_primary_key}' in table "
                f"'{parent_table_name}'."
            )
        if child_foreign_key not in child.columns:
            raise InvalidMetadataError(
                f"Unknown foreign key '{child_foreign_key}' in table "
                f"'{child_table_name}'."
            )
        self.relationships.append(
            {
                'parent_table_name': parent_table_name,
                'parent_primary_key': parent_primary_key,
                'child_table_name': child_table_name,
                'child_foreign_key': child_foreign_key,
            }
        )

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """Convert the metadata to a dictionary.

        Returns
        -------
        dict
            Dictionary representation following the ``MULTI_TABLE_V1`` spec.

        """
        return {
            'tables': {
                name: table._to_table_dict()
                for name, table in self.tables.items()
            },
            'relationships': copy.deepcopy(self.relationships),
            'METADATA_SPEC_VERSION': MULTI_TABLE_SPEC_VERSION,
        }

    @classmethod
    def load_from_dict(
        cls, metadata_dict: dict, dataset_name: str = ''
    ) -> 'Metadata':
        """Create a Metadata object from a dictionary.

        Parameters
        ----------
        metadata_dict: dict
            Dictionary following the ``MULTI_TABLE_V1`` spec.
        dataset_name: str, default=""
            Name of the dataset.

        Returns
        -------
        Metadata
            The reconstructed metadata object.

        """
        instance = cls(dataset_name=dataset_name)
        for table_name, table_dict in metadata_dict.get('tables', {}).items():
            instance.tables[table_name] = SingleTableMetadata.load_from_dict(
                table_dict
            )
        instance.relationships = copy.deepcopy(
            metadata_dict.get('relationships', [])
        )
        return instance

    @classmethod
    def load_from_json(cls, filepath: str | os.PathLike) -> 'Metadata':
        """Load metadata from a JSON file.

        Parameters
        ----------
        filepath: str | os.PathLike
            Path to the metadata JSON file.

        Returns
        -------
        Metadata
            The loaded metadata object.

        """
        with open(filepath) as f:
            metadata_dict = json.load(f)
        return cls.load_from_dict(metadata_dict)

    def save_to_json(
        self, filepath: str | os.PathLike, mode: str = 'write'
    ) -> None:
        """Save the metadata to a JSON file.

        Parameters
        ----------
        filepath: str | os.PathLike
            Path to write the metadata JSON file.
        mode: str, default="write"
            Accepted for API compatibility; the file is always written.

        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    # -- Queries ----------------------------------------------------------

    def get_tables(self) -> list:
        """Get a list of all table names in the metadata.

        Returns
        -------
        list
            List of table names.

        """
        return list(self.tables.keys())

    def get_primary_key(self, table_name: str) -> str | None:
        """Get the primary key of a table.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Returns
        -------
        str | None
            Name of the primary key column, or None if unset.

        """
        return self._get_table(table_name).primary_key

    def get_table_meta(
        self, table_name: str, to_dict: bool = True
    ) -> dict | SingleTableMetadata:
        """Get metadata for a specific table.

        Parameters
        ----------
        table_name: str
            Name of the table.
        to_dict: bool, default=True
            Whether to return the metadata as a dictionary.

        Returns
        -------
        dict | SingleTableMetadata
            Table metadata as a dictionary or SingleTableMetadata object.

        """
        table_meta = self._get_table(table_name)
        if to_dict:
            return table_meta.to_dict()
        return table_meta

    def get_column_names(self, table_name: str, **kwargs) -> list[str]:
        """Get column names of a table, optionally filtered by attribute.

        Parameters
        ----------
        table_name: str
            Name of the table.
        **kwargs
            Attribute filters, e.g. ``sdtype='id'``.

        Returns
        -------
        list[str]
            List of matching column names.

        """
        return self._get_table(table_name).get_column_names(**kwargs)

    def get_children(self, table_name: str) -> set:
        """Get all child tables of a given table.

        Parameters
        ----------
        table_name: str
            Name of the parent table.

        Returns
        -------
        set
            Set of child table names.

        """
        children = set()
        for relation in self.relationships:
            if relation['parent_table_name'] == table_name:
                children.add(relation['child_table_name'])
        return children

    def get_parents(self, table_name: str) -> set:
        """Get all parent tables of a given table.

        Parameters
        ----------
        table_name: str
            Name of the child table.

        Returns
        -------
        set
            Set of parent table names.

        """
        parents = set()
        for relation in self.relationships:
            if relation['child_table_name'] == table_name:
                parents.add(relation['parent_table_name'])
        return parents

    def _get_foreign_keys(
        self, parent_table_name: str, child_table_name: str
    ) -> list:
        """Get foreign keys for a parent-child table pair.

        Parameters
        ----------
        parent_table_name: str
            Name of the parent table.
        child_table_name: str
            Name of the child table.

        Returns
        -------
        list
            List of foreign key column names.

        """
        return [
            relation['child_foreign_key']
            for relation in self.relationships
            if relation['parent_table_name'] == parent_table_name
            and relation['child_table_name'] == child_table_name
        ]

    def get_foreign_keys(
        self, parent_table_name: str, child_table_name: str
    ) -> list:
        """Get foreign keys between parent and child tables.

        Parameters
        ----------
        parent_table_name: str
            Name of the parent table.
        child_table_name: str
            Name of the child table.

        Returns
        -------
        list
            List of foreign key column names.

        """
        return self._get_foreign_keys(parent_table_name, child_table_name)

    def rename_column(
        self, table_name: str, old_column_name: str, new_column_name: str
    ):
        """Rename a column in a table."""  # noqa: DOC201
        self.tables[table_name].columns[new_column_name] = self.tables[
            table_name
        ].columns.pop(old_column_name)
        if self.tables[table_name].columns[new_column_name]['sdtype'] != 'id':
            return self

        if self.tables[table_name].primary_key == old_column_name:
            self.tables[table_name].primary_key = new_column_name

        for relationship in self.relationships:
            if (
                relationship['parent_table_name'] == table_name
                and relationship['parent_primary_key'] == old_column_name
            ):
                relationship['parent_primary_key'] = new_column_name
            if (
                relationship['child_table_name'] == table_name
                and relationship['child_foreign_key'] == old_column_name
            ):
                relationship['child_foreign_key'] = new_column_name
        return self

    def get_root_tables(self) -> list:
        """Get all root tables (tables with no parents).

        Returns
        -------
        list
            List of root table names.

        """
        root_tables = set(self.tables.keys())
        for relation in self.relationships:
            root_tables.discard(relation['child_table_name'])
        return list(root_tables)

    def get_table_levels(self) -> dict:
        """Get the level of each table in the hierarchy.

        The level is determined by the length of the path from any root table.

        Returns
        -------
        dict
            Dictionary mapping table names to their levels.

        """
        # return the length of the path from any root table
        root_tables = self.get_root_tables()
        table_levels = {}
        for root_table in root_tables:
            table_levels[root_table] = 0

        relationships = self.relationships.copy()
        while len(relationships) > 0:
            relationship = relationships.pop(0)
            if relationship['parent_table_name'] in table_levels:
                table_levels[relationship['child_table_name']] = (
                    table_levels[relationship['parent_table_name']] + 1
                )
            else:
                relationships.append(relationship)
        return table_levels

    # -- Validation -------------------------------------------------------

    def validate(self) -> None:
        """Validate the internal consistency of the metadata.

        Checks that every relationship references existing tables and key
        columns.

        Raises
        ------
        InvalidMetadataError
            If a relationship references an unknown table or key column.

        """
        for relationship in self.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            parent = self._get_table(parent_name)
            child = self._get_table(child_name)
            if relationship['parent_primary_key'] not in parent.columns:
                raise InvalidMetadataError(
                    f'Relationship references unknown primary key '
                    f"'{relationship['parent_primary_key']}' in table "
                    f"'{parent_name}'."
                )
            if relationship['child_foreign_key'] not in child.columns:
                raise InvalidMetadataError(
                    f'Relationship references unknown foreign key '
                    f"'{relationship['child_foreign_key']}' in table "
                    f"'{child_name}'."
                )

    def validate_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Validate that data conforms to the metadata.

        Checks that all tables are present, that columns match the
        metadata, that columns conform to their declared
        ``computer_representation``, that primary keys are unique, and that
        foreign keys reference existing primary keys.

        Parameters
        ----------
        data: dict[str, pd.DataFrame]
            Dictionary mapping table names to pandas DataFrames.

        Raises
        ------
        InvalidDataError
            If the data does not match the metadata.

        """
        missing = set(self.tables) - set(data)
        if missing:
            raise InvalidDataError(
                f'The provided data is missing the tables {missing}.'
            )

        for table_name, table_meta in self.tables.items():
            table = data[table_name]
            metadata_columns = set(table_meta.columns)
            data_columns = set(table.columns)
            extra = data_columns - metadata_columns
            if extra:
                raise InvalidDataError(
                    f"Columns {sorted(extra)} in table '{table_name}' are "
                    f'not present in the metadata.'
                )
            absent = metadata_columns - data_columns
            if absent:
                raise InvalidDataError(
                    f'Columns {sorted(absent)} are missing from table '
                    f"'{table_name}'."
                )

            for column, info in table_meta.columns.items():
                representation = info.get('computer_representation')
                if representation is None:
                    continue
                series = table[column].dropna()
                if not pd.api.types.is_numeric_dtype(series):
                    raise InvalidDataError(
                        f"Column '{column}' in table '{table_name}' has "
                        f"computer_representation '{representation}' but is "
                        f'not numeric.'
                    )
                if (
                    representation.lower().startswith(('int', 'uint'))
                    and not (series % 1 == 0).all()
                ):
                    raise InvalidDataError(
                        f"Column '{column}' in table '{table_name}' has "
                        f"integer computer_representation '{representation}' "
                        f'but contains non-integer values.'
                    )

            primary_key = table_meta.primary_key
            if primary_key is not None:
                key_values = table[primary_key]
                if key_values.duplicated().any():
                    raise InvalidDataError(
                        f"Primary key '{primary_key}' in table "
                        f"'{table_name}' contains repeating values."
                    )

        for relationship in self.relationships:
            parent = data[relationship['parent_table_name']]
            child = data[relationship['child_table_name']]
            parent_keys = set(
                parent[relationship['parent_primary_key']].dropna()
            )
            child_keys = child[relationship['child_foreign_key']].dropna()
            unknown = set(child_keys) - parent_keys
            if unknown:
                raise InvalidDataError(
                    f"Foreign key '{relationship['child_foreign_key']}' in "
                    f"table '{relationship['child_table_name']}' contains "
                    f'unknown references.'
                )

    # -- Visualisation ----------------------------------------------------

    def visualize(
        self,
        show_table_details='full',
        show_relationship_labels=True,
        output_filepath=None,
    ) -> graphviz.Digraph:
        """Visualize the database schema.

        Parameters
        ----------
        show_table_details: str, default='full'
            Ignored (kept for compatibility with SDV's API).
        show_relationship_labels: bool, default=True
            Ignored (kept for compatibility with SDV's API).
        output_filepath: str | os.PathLike | None, default=None
            Output file path. If None, the graph is not saved.

        Returns
        -------
        graphviz.Digraph
            Graph visualization of the metadata.

        """
        filename, graphviz_extension = _get_graphviz_extension(output_filepath)

        def create_table_node(
            table_name: str, metadata: 'Metadata', font: str = 'Arial'
        ):
            """Create a node for a table in the graph.

            Parameters
            ----------
            table_name: str
                Name of the table.
            metadata: Metadata
                Metadata object.
            font: str, default="Arial"
                Font to use for the node.

            Returns
            -------
            str
                HTML-like label for the node.

            """
            table_meta = cast(
                dict, metadata.get_table_meta(table_name, to_dict=True)
            )
            table_label = '< <table cellpadding="0" cellborder="0" cellspacing="0" border="0">'  # noqa: E501
            table_label += f'<tr><td bgcolor="#476893">  </td> <td align="left" bgcolor="#476893"><font color="white"><b>{table_name}</b></font></td> <td align="right" bgcolor="#476893"></td></tr>'  # noqa: E501
            primary_key = metadata.get_primary_key(table_name)
            for col, info in table_meta['columns'].items():
                sdtype = info['sdtype']
                fontspec = f'face="{font}"'
                if col == primary_key:
                    col = f'<u><b>{col}</b></u>'
                color = '#e2edf1' if sdtype == 'id' else '#f2f2f2'
                table_label += f'<tr><td bgcolor="{color}">  </td> <td align="left" bgcolor="{color}"><font color="#6e6e6e"  {fontspec}>{col} </font></td> <td align="right" bgcolor="{color}"><font color="#9b9c9c" {fontspec}>{sdtype}</font></td></tr>'  # noqa: E501
            table_label += '</table> >'
            return table_label

        dot = graphviz.Digraph(
            graph_attr={'splines': 'ortho', 'ranksep': '0.8'},
            node_attr={'shape': 'plaintext'},
        )

        for table_name in self.get_tables():
            dot.node(
                table_name,
                shape='plain',
                label=create_table_node(table_name, self),
                fontname='Arial',
            )

        for relationship in self.relationships:
            parent_table = relationship['parent_table_name']
            child_table = relationship['child_table_name']
            dot.edge(
                parent_table,
                child_table,
                arrowhead='crow',
                arrowtail='tee',
                color='#78a9d2',
                arrowsize='0.9',
            )

        if filename:
            dot.render(
                filename=filename, cleanup=True, format=graphviz_extension
            )
        else:
            try:
                graphviz.version()
            except graphviz.ExecutableNotFound:
                from warnings import warn

                warning_message = (
                    'Graphviz does not seem to be installed on this system. '
                    'For full metadata visualization capabilities, please '
                    'make sure to have its binaries properly installed: '
                    'https://graphviz.gitlab.io/download/'
                )
                warn(warning_message, RuntimeWarning, stacklevel=2)
        return dot


def drop_ids(table: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Drop ID columns from a table.

    Parameters
    ----------
    table: pd.DataFrame
        DataFrame to process.
    metadata: dict
        Metadata dictionary for the table.

    Returns
    -------
    pd.DataFrame
        DataFrame with ID columns removed.

    """
    for column, column_info in metadata['columns'].items():
        if column_info['sdtype'] == 'id' and column in table.columns:
            table = table.drop(columns=column, axis=1)
    return table


def convert_metadata_to_v0(metadata: Metadata) -> dict:
    """Convert Metadata object to v0 format.

    Parameters
    ----------
    metadata: Metadata
        Metadata object to convert.

    Returns
    -------
    dict
        Metadata in v0 format.

    """
    metadata_v1 = metadata.to_dict()
    metadata_v0 = {'tables': {}}
    for table_name, table_info in metadata_v1['tables'].items():
        metadata_v0['tables'][table_name] = {'fields': {}}
        for column, column_info in table_info['columns'].items():
            metadata_v0['tables'][table_name]['fields'][column] = {
                'type': column_info['sdtype']
            }
            if column_info['sdtype'] == 'boolean':
                # convert boolean to categorical
                metadata_v0['tables'][table_name]['fields'][column]['type'] = (
                    'categorical'
                )
            if column_info['sdtype'] == 'datetime':
                metadata_v0['tables'][table_name]['fields'][column][
                    'format'
                ] = column_info['datetime_format']

        if 'primary_key' in table_info:
            pkey_metadata = {
                'type': 'id',
                'subtype': 'string',
            }
            metadata_v0['tables'][table_name]['fields'][
                table_info['primary_key']
            ] = pkey_metadata
            metadata_v0['tables'][table_name]['primary_key'] = table_info[
                'primary_key'
            ]

    for relationship in metadata_v1['relationships']:
        parent_table_name = relationship['parent_table_name']
        child_table_name = relationship['child_table_name']
        parent_primary_key = relationship['parent_primary_key']
        child_foreign_key = relationship['child_foreign_key']
        table_pkey = {
            'table': parent_table_name,
            'field': parent_primary_key,
        }
        metadata_v0['tables'][child_table_name]['fields'][child_foreign_key][
            'ref'
        ] = table_pkey
        metadata_v0['tables'][child_table_name]['fields'][child_foreign_key][
            'subtype'
        ] = 'string'
    return metadata_v0


def convert_and_save_metadata_v0(metadata: Metadata, path: str | os.PathLike):
    """Convert Metadata object to v0 format and save it to a file.

    Parameters
    ----------
    metadata: Metadata
        Metadata object to convert and save.
    path: Union[str, os.PathLike]
        Path to save the metadata to.

    """
    metadata_v0 = convert_metadata_to_v0(metadata)
    with open(os.path.join(path, 'metadata_v0.json'), 'w') as f:
        json.dump(metadata_v0, f, indent=4)
