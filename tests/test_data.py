import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from data.data_generators import generate_real_data
from syntherela.data import (
    download_sdv_relational_datasets,
    get_dataset_stats,
    load_tables,
    remove_sdv_columns,
    save_tables,
)


def test_dataset_download():
    """Test download_sdv_relational_datasets with mocked SDV demo calls."""
    pytest.importorskip('sdv')
    # Avoid network: mock get_available_demos and download_demo
    mock_demos = pd.DataFrame({'dataset_name': ['fake_dataset']})

    def create_output_dir(*args, **kwargs):
        out = kwargs.get('output_folder_name')
        if out:
            os.makedirs(out, exist_ok=True)

    with (
        patch('sdv.datasets.demo.get_available_demos', return_value=mock_demos),
        patch(
            'sdv.datasets.demo.download_demo', side_effect=create_output_dir
        ) as mock_download,
    ):
        download_sdv_relational_datasets('tests/tmp')

    assert mock_download.called
    first_call_kwargs = mock_download.call_args_list[0][1]
    path_created = first_call_kwargs['output_folder_name']
    assert os.path.exists(path_created)


def test_loading_and_saving():
    tables, metadata = generate_real_data()
    save_tables(tables, path='tests/tmp/test_data')
    assert os.path.exists('tests/tmp/test_data')

    loaded_tables = load_tables('tests/tmp/test_data', metadata)
    metadata.validate_data(loaded_tables)
    for table_name in tables.keys():
        table_meta = metadata.get_table_meta(table_name, to_dict=False)
        for column in tables[table_name].columns:
            assert column in loaded_tables[table_name].columns
            if table_meta.columns[column]['sdtype'] == 'numerical':
                assert np.isclose(
                    tables[table_name][column].values,
                    loaded_tables[table_name][column].values,
                    rtol=1e-16,
                ).all()
            else:
                assert (
                    tables[table_name][column]
                    == loaded_tables[table_name][column]
                ).all()


def test_get_dataset_stats():
    """Test get_dataset_stats returns expected keys and values."""
    tables, metadata = generate_real_data()
    stats = get_dataset_stats(tables, metadata)
    assert 'num_tables' in stats
    assert 'num_rows' in stats
    assert 'num_columns' in stats
    assert 'num_relationships' in stats
    assert stats['num_tables'] == 2
    assert stats['num_rows'] == sum(len(t) for t in tables.values())
    assert stats['num_relationships'] == len(metadata.relationships)


def test_remove_sdv_columns():
    """Test remove_sdv_columns drops extra SDV columns and updates metadata."""
    tables, metadata = generate_real_data()
    # Add a column that should be removed
    tables['table1']['add_numerical_extra'] = 1
    metadata.add_column('table1', 'add_numerical_extra', sdtype='numerical')
    metadata.validate()
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        out_tables, out_metadata = remove_sdv_columns(
            tables, metadata, update_metadata=True, validate=True
        )
    assert 'add_numerical_extra' not in out_tables['table1'].columns
    assert (
        'add_numerical_extra'
        not in out_metadata.get_table_meta('table1', to_dict=True)['columns']
    )


def test_save_tables_with_metadata_and_save_metadata():
    """Test save_tables with metadata writes metadata.json."""
    tables, metadata = generate_real_data()
    path = 'tests/tmp/test_save_meta'
    save_tables(tables, path=path, metadata=metadata, save_metadata=True)
    assert os.path.isfile(os.path.join(path, 'metadata.json'))
    assert os.path.isfile(os.path.join(path, 'table1.csv'))


def test_save_and_load_tables_with_datetime_column():
    """Test save_tables and load_tables round-trip with a datetime column."""
    tables, metadata = generate_real_data()
    metadata.add_column(
        'table1', 'date_col', sdtype='datetime', datetime_format='%Y-%m-%d'
    )
    tables['table1']['date_col'] = pd.date_range(
        '2020-01-01', periods=len(tables['table1']), freq='D'
    )
    metadata.validate()
    metadata.validate_data(tables)

    path = 'tests/tmp/test_datetime'
    save_tables(tables, path=path, metadata=metadata)
    loaded = load_tables(path, metadata)
    metadata.validate_data(loaded)
    orig = pd.to_datetime(tables['table1']['date_col'])
    loaded_dates = pd.to_datetime(loaded['table1']['date_col'])
    pd.testing.assert_series_equal(
        orig,
        loaded_dates,
        check_names=True,
        check_dtype=False,
    )


def test_load_tables_raises_when_datetime_format_missing():
    """Test load_tables raises ValueError if datetime_format is missing."""
    tables, metadata = generate_real_data()
    metadata.add_column(
        'table1', 'date_col', sdtype='datetime', datetime_format='%Y-%m-%d'
    )
    tables['table1']['date_col'] = pd.date_range(
        '2020-01-01', periods=len(tables['table1']), freq='D'
    )
    metadata.validate()
    path = 'tests/tmp/test_datetime_missing_fmt'
    save_tables(tables, path=path, metadata=metadata)
    # Remove datetime_format from column info so load_tables will raise
    col_info = metadata.tables['table1'].columns['date_col']
    if isinstance(col_info, dict):
        col_info.pop('datetime_format', None)
    else:
        # SDV may use an object; try to remove the attribute
        if hasattr(col_info, '__dict__'):
            col_info.__dict__.pop('datetime_format', None)
        elif hasattr(col_info, 'datetime_format'):
            delattr(col_info, 'datetime_format')
    with pytest.raises(ValueError, match='datetime_format.*not found'):
        load_tables(path, metadata)
