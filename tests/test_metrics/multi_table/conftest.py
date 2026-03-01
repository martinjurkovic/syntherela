"""Shared fixtures for multi-table metric tests."""

import pytest
from data.data_generators import generate_real_data


@pytest.fixture
def sample_data():
    """Real data and metadata for multi-table tests."""
    data, metadata = generate_real_data()
    return data, metadata
