"""Shared fixtures for single table metric tests."""

import pytest
from data.data_generators import generate_real_data


@pytest.fixture
def sample_data():
    """Real data and metadata."""
    data, metadata = generate_real_data()
    return data, metadata


@pytest.fixture
def table_meta(sample_data):
    """Table1 metadata as dict for single-table metrics."""
    _, metadata = sample_data
    return metadata.get_table_meta('table1', to_dict=True)


@pytest.fixture
def detection_data(sample_data):
    """Real table, synthetic table, and table metadata for detection metrics."""
    data, metadata = sample_data
    table_meta = metadata.get_table_meta('table1', to_dict=True)
    return data['table1'], data['table1'].copy(), table_meta
