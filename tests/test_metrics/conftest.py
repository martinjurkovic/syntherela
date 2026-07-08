"""Shared fixtures for metrics tests (test_base, test_utility)."""

import numpy as np
import pandas as pd
import pytest
from data.data_generators import generate_real_data


@pytest.fixture
def sample_data():
    """Real data and metadata from data_generators."""
    return generate_real_data(seed=42)


@pytest.fixture
def column_data():
    """Generate column data for testing (single-column metrics)."""
    np.random.seed(42)
    real_data = pd.Series(np.random.normal(0, 1, 100))
    synthetic_data = pd.Series(np.random.normal(0.1, 1.1, 100))
    return real_data, synthetic_data


@pytest.fixture
def table_data():
    """Generate table data for testing (single-table metrics)."""
    np.random.seed(42)
    real_data = pd.DataFrame(
        {
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.choice(['A', 'B', 'C'], size=100),
            'col3': np.random.uniform(0, 1, 100),
        }
    )
    synthetic_data = pd.DataFrame(
        {
            'col1': np.random.normal(0.1, 1.1, 100),
            'col2': np.random.choice(['A', 'B', 'C'], size=100),
            'col3': np.random.uniform(0.1, 1.1, 100),
        }
    )
    return real_data, synthetic_data
