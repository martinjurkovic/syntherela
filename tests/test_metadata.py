import json

import pandas as pd
import pytest
from data.data_generators import (
    generate_real_data,
    generate_real_data_three_tables,
)
from syntherela.metadata import (
    InvalidDataError,
    InvalidMetadataError,
    Metadata,
    SingleTableMetadata,
    convert_metadata_to_v0,
)


def test_metadata():
    data, metadata = generate_real_data()
    metadata.validate()
    metadata.validate_data(data)

    assert metadata.get_tables() == list(data.keys())

    assert metadata.get_primary_key('table1') == 'pk1'
    assert metadata.get_primary_key('table2') == 'pk2'

    table1_meta_dict = metadata.get_table_meta('table1', to_dict=True)
    assert table1_meta_dict['primary_key'] == 'pk1'
    for column in data['table1'].columns:
        assert column in table1_meta_dict['columns']

    table2_meta = metadata.get_table_meta('table2', to_dict=False)
    assert table2_meta.primary_key == 'pk2'
    assert type(table2_meta) is SingleTableMetadata
    for column in data['table2'].columns:
        assert column in table2_meta.columns

    children = metadata.get_children('table1')
    assert 'table2' in children

    parents = metadata.get_parents('table2')
    assert 'table1' in parents

    foreign_keys = metadata.get_foreign_keys('table1', 'table2')
    assert 'fk2' in foreign_keys

    root_tables = metadata.get_root_tables()
    assert 'table1' in root_tables
    assert set(root_tables) == parents

    # Rename a column
    metadata.rename_column('table1', 'normal', 'new_normal')
    assert (
        'new_normal'
        in metadata.get_table_meta('table1', to_dict=True)['columns']
    )

    metadata.rename_column('table1', 'pk1', 'pk')
    assert metadata.get_primary_key('table1') == 'pk'

    metadata.rename_column('table2', 'fk2', 'new_fk2')
    assert 'new_fk2' in metadata.get_foreign_keys('table1', 'table2')


def test_metadata_conversion():
    data, metadata = generate_real_data()
    # TODO: add these datattypes in the generator
    metadata.add_column(
        'table1', 'date', sdtype='datetime', datetime_format='%Y-%m-%d'
    )
    data['table1']['date'] = pd.date_range(
        '2020-01-01', periods=len(data['table1']), freq='D'
    )
    metadata.add_column('table2', 'bool', sdtype='boolean')
    data['table2']['bool'] = [True, False] * (len(data['table2']) // 2)
    metadata.validate()
    metadata.validate_data(data)

    metadata_v0 = convert_metadata_to_v0(metadata)

    for table in metadata.get_tables():
        assert table in metadata_v0['tables']
        for column in data[table].columns:
            assert column in metadata_v0['tables'][table]['fields']
        pk = metadata.get_primary_key(table)
        assert pk == metadata_v0['tables'][table]['primary_key']


def test_visualize_basic():
    """Test that the visualize method returns a graphviz.Digraph object."""
    _, metadata = generate_real_data()

    # Test visualization
    graph = metadata.visualize()
    assert graph is not None
    assert hasattr(graph, 'render')  # Check it's a graphviz.Digraph object


# ---------------------------------------------------------------------------
# Serialisation: to_dict / load_from_dict / load_from_json / save_to_json
# ---------------------------------------------------------------------------


def test_to_dict_structure():
    _, metadata = generate_real_data()
    meta_dict = metadata.to_dict()

    assert set(meta_dict) >= {'tables', 'relationships'}
    assert meta_dict['METADATA_SPEC_VERSION'] == 'MULTI_TABLE_V1'
    assert set(meta_dict['tables']) == {'table1', 'table2'}
    for table_dict in meta_dict['tables'].values():
        assert 'columns' in table_dict
        assert 'primary_key' in table_dict
    relationship = meta_dict['relationships'][0]
    assert relationship['parent_table_name'] == 'table1'
    assert relationship['child_table_name'] == 'table2'
    assert relationship['parent_primary_key'] == 'pk1'
    assert relationship['child_foreign_key'] == 'fk2'


def test_load_from_dict_roundtrip():
    _, metadata = generate_real_data()
    meta_dict = metadata.to_dict()

    restored = Metadata.load_from_dict(meta_dict)

    assert set(restored.get_tables()) == set(metadata.get_tables())
    for table in metadata.get_tables():
        assert restored.get_primary_key(table) == metadata.get_primary_key(
            table
        )
        assert restored.tables[table].columns == metadata.tables[table].columns
    assert restored.relationships == metadata.relationships
    # to_dict is stable across a round-trip
    assert restored.to_dict() == meta_dict


def test_load_from_json(tmp_path, mock_metadata_dict):
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(mock_metadata_dict))

    metadata = Metadata().load_from_json(path)

    assert set(metadata.get_tables()) == {'users', 'orders'}
    assert metadata.get_primary_key('users') == 'user_id'
    assert 'users' in metadata.get_parents('orders')
    # computer_representation and datetime_format survive the round-trip.
    orders = metadata.tables['orders'].columns
    assert orders['amount']['computer_representation'] == 'Float'
    assert orders['created']['datetime_format'] == '%Y-%m-%d'
    assert metadata.to_dict() == mock_metadata_dict


def test_save_to_json(tmp_path):
    _, metadata = generate_real_data()
    out = tmp_path / 'metadata.json'

    metadata.save_to_json(out)

    assert out.exists()
    reloaded = Metadata.load_from_dict(json.loads(out.read_text()))
    assert reloaded.to_dict() == metadata.to_dict()


# ---------------------------------------------------------------------------
# Queries: column names, foreign keys, hierarchy
# ---------------------------------------------------------------------------


def test_get_column_names():
    _, metadata = generate_real_data()

    assert metadata.get_column_names('table2', sdtype='id') == ['pk2', 'fk2']
    numerical = metadata.get_column_names('table1', sdtype='numerical')
    assert set(numerical) == {'normal', 'uniform'}
    # No filter returns every column.
    assert set(metadata.get_column_names('table1')) == {
        'pk1',
        'normal',
        'uniform',
        'categorical',
    }


def test_get_foreign_keys_multi_hop():
    _, metadata = generate_real_data_three_tables()

    assert metadata.get_foreign_keys('table1', 'table2') == ['fk2']
    assert metadata.get_foreign_keys('table2', 'table3') == ['fk3']
    # Unrelated table pair has no foreign keys.
    assert metadata.get_foreign_keys('table1', 'table3') == []


def test_hierarchy_helpers():
    _, metadata = generate_real_data_three_tables()

    assert metadata.get_root_tables() == ['table1']
    assert metadata.get_children('table2') == {'table3'}
    assert metadata.get_parents('table3') == {'table2'}
    assert metadata.get_table_levels() == {
        'table1': 0,
        'table2': 1,
        'table3': 2,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_data_detects_foreign_key_violation():
    data, metadata = generate_real_data()
    data['table2'].loc[0, 'fk2'] = 9999  # not a valid table1 primary key

    with pytest.raises(InvalidDataError):
        metadata.validate_data(data)


def test_validate_data_detects_duplicate_primary_key():
    data, metadata = generate_real_data()
    data['table1'].loc[1, 'pk1'] = data['table1'].loc[0, 'pk1']

    with pytest.raises(InvalidDataError):
        metadata.validate_data(data)


def test_validate_data_detects_extra_column():
    data, metadata = generate_real_data()
    data['table1']['unexpected'] = 1

    with pytest.raises(InvalidDataError):
        metadata.validate_data(data)


def test_validate_data_detects_missing_table():
    data, metadata = generate_real_data()
    del data['table2']

    with pytest.raises(InvalidDataError):
        metadata.validate_data(data)


def test_validate_data_accepts_valid_computer_representation(
    mock_metadata_dict, mock_data
):
    metadata = Metadata.load_from_dict(mock_metadata_dict)
    # Integral values stored as float satisfy an integer representation.
    metadata.validate_data(mock_data)


def test_validate_data_detects_computer_representation_mismatch(
    mock_metadata_dict, mock_data
):
    metadata = Metadata.load_from_dict(mock_metadata_dict)
    mock_data['users'].loc[0, 'age'] = 30.5  # not integral for 'Int64'

    with pytest.raises(InvalidDataError):
        metadata.validate_data(mock_data)


def test_validate_rejects_relationship_to_unknown_table():
    metadata = Metadata()
    metadata.add_table('table1')
    metadata.add_column('table1', 'pk1', sdtype='id')
    metadata.set_primary_key('table1', 'pk1')

    with pytest.raises(InvalidMetadataError):
        metadata.add_relationship('table1', 'missing', 'pk1', 'fk')


# ---------------------------------------------------------------------------
# Single-table metadata
# ---------------------------------------------------------------------------


def test_single_table_add_and_update_column():
    table_meta = SingleTableMetadata()
    table_meta.add_column('a', sdtype='numerical')
    table_meta.add_column('pk', sdtype='id')
    table_meta.set_primary_key('pk')

    assert table_meta.primary_key == 'pk'
    assert table_meta.get_column_names(sdtype='id') == ['pk']

    table_meta.update_column('a', sdtype='categorical')
    assert table_meta.columns['a']['sdtype'] == 'categorical'

    table_dict = table_meta.to_dict()
    assert table_dict['columns']['a']['sdtype'] == 'categorical'
    assert table_dict['primary_key'] == 'pk'


def test_single_table_detect_from_dataframe():
    df = pd.DataFrame(
        {
            'f': [0.1, 0.2, 0.3],
            'd': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            's': ['x', 'y', 'z'],
        }
    )
    table_meta = SingleTableMetadata()
    table_meta.detect_from_dataframe(df)

    assert table_meta.columns['f']['sdtype'] == 'numerical'
    assert table_meta.columns['d']['sdtype'] == 'datetime'
    assert table_meta.columns['s']['sdtype'] == 'categorical'

    # Detected columns can be overridden with authoritative metadata.
    table_meta.update_column(
        'f', sdtype='numerical', computer_representation='Float'
    )
    assert table_meta.columns['f']['computer_representation'] == 'Float'
