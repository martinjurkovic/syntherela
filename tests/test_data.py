import os
from shutil import rmtree
from unittest.mock import patch

import numpy as np
from data.data_generators import generate_real_data
from syntherela.data import (
    download_sdv_relational_datasets,
    get_dataset_stats,
    load_tables,
    remove_sdv_columns,
    save_tables,
)


def test_dataset_download():
    """Test download_sdv_relational_datasets with mocked download_demo."""

    def create_output_dir(*args, **kwargs):
        out = kwargs.get('output_folder_name')
        if out:
            os.makedirs(out, exist_ok=True)

    with patch(
        'syntherela.data.download_demo', side_effect=create_output_dir
    ) as mock_download:
        download_sdv_relational_datasets('tests/tmp')

    assert mock_download.called
    first_call_kwargs = mock_download.call_args_list[0][1]
    path_created = first_call_kwargs['output_folder_name']
    assert os.path.exists(path_created)
    rmtree('tests/tmp')


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
    rmtree('tests/tmp')


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
