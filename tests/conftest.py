import os
from shutil import rmtree

import pandas as pd
import pytest
from syntherela.metadata import MULTI_TABLE_SPEC_VERSION


@pytest.fixture(scope='session', autouse=True)
def cleanup_tmp_dirs():
    """Remove tests/tmp once after all tests in the session have finished."""
    yield
    if os.path.exists('tests/tmp'):
        rmtree('tests/tmp')


@pytest.fixture()
def mock_metadata_dict():
    """Return a small self-contained relational schema (users → orders)."""
    return {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'age': {
                        'sdtype': 'numerical',
                        'computer_representation': 'Int64',
                    },
                },
            },
            'orders': {
                'primary_key': 'order_id',
                'columns': {
                    'order_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'amount': {
                        'sdtype': 'numerical',
                        'computer_representation': 'Float',
                    },
                    'created': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d',
                    },
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'orders',
                'child_foreign_key': 'user_id',
            }
        ],
        'METADATA_SPEC_VERSION': MULTI_TABLE_SPEC_VERSION,
    }


@pytest.fixture()
def mock_data():
    """DataFrames matching the mock_metadata_dict schema."""
    users = pd.DataFrame({'user_id': [1, 2], 'age': [30.0, 40.0]})
    orders = pd.DataFrame(
        {
            'order_id': [1, 2],
            'user_id': [1, 2],
            'amount': [9.99, 5.0],
            'created': pd.to_datetime(['2020-01-01', '2020-01-02']),
        }
    )
    return {'users': users, 'orders': orders}
