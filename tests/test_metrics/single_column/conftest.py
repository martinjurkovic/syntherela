"""Shared fixtures for single column metric tests."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numerical_data():
    """Generate numerical data for testing."""
    np.random.seed(42)
    real_data = pd.Series(np.random.normal(0, 1, 100))
    synthetic_data = pd.Series(np.random.normal(0.1, 1.1, 100))
    return real_data, synthetic_data


@pytest.fixture
def categorical_data():
    """Generate categorical data for testing."""
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D']
    real_probs = [0.4, 0.3, 0.2, 0.1]
    synth_probs = [0.3, 0.3, 0.3, 0.1]

    real_data = pd.Series(np.random.choice(categories, size=100, p=real_probs))
    synthetic_data = pd.Series(
        np.random.choice(categories, size=100, p=synth_probs)
    )
    return real_data, synthetic_data


@pytest.fixture
def datetime_data():
    """Generate datetime data for testing."""
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    days_real = pd.Series(
        pd.to_datetime(start_date)
        + pd.to_timedelta(np.random.normal(50, 10, 100), unit='D')
    )
    days_synth = pd.Series(
        pd.to_datetime(start_date)
        + pd.to_timedelta(np.random.normal(55, 12, 100), unit='D')
    )
    return days_real, days_synth
