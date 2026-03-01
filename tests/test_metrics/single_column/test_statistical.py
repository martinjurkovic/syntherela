"""Tests for single column statistical metrics."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from syntherela.metrics.single_column.statistical import (
    ChiSquareTest,
    KolmogorovSmirnovTest,
)


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


def test_kolmogorov_smirnov_initialization():
    """Test initialization of KolmogorovSmirnovTest metric."""
    metric = KolmogorovSmirnovTest()
    assert metric.name == 'KolmogorovSmirnovTest'
    assert metric.goal.name == 'MINIMIZE'


def test_kolmogorov_smirnov_is_applicable():
    """Test is_applicable method of KolmogorovSmirnovTest."""
    assert KolmogorovSmirnovTest.is_applicable('numerical') is True
    assert KolmogorovSmirnovTest.is_applicable('datetime') is True
    assert KolmogorovSmirnovTest.is_applicable('categorical') is False
    assert KolmogorovSmirnovTest.is_applicable('boolean') is False


def test_kolmogorov_smirnov_validate(numerical_data, categorical_data):
    """Test validate method of KolmogorovSmirnovTest."""
    real_num, _ = numerical_data
    real_cat, _ = categorical_data

    metric = KolmogorovSmirnovTest()

    # Should not raise an error for numerical data
    metric.validate(real_num)

    # Should raise an error for categorical data
    with pytest.raises(ValueError):
        metric.validate(real_cat)


def test_kolmogorov_smirnov_compute_numerical(numerical_data):
    """Test compute method of KolmogorovSmirnovTest with numerical data."""
    real_data, synthetic_data = numerical_data

    result = KolmogorovSmirnovTest.compute(real_data, synthetic_data)

    assert 'statistic' in result
    assert 'p_val' in result
    assert 0 <= result['statistic'] <= 1
    assert 0 <= result['p_val'] <= 1


def test_kolmogorov_smirnov_compute_datetime(datetime_data):
    """Test compute method of KolmogorovSmirnovTest with datetime data."""
    real_data, synthetic_data = datetime_data

    result = KolmogorovSmirnovTest.compute(real_data, synthetic_data)

    assert 'statistic' in result
    assert 'p_val' in result
    assert 0 <= result['statistic'] <= 1
    assert 0 <= result['p_val'] <= 1


def test_chi_square_initialization():
    """Test initialization of ChiSquareTest metric."""
    metric = ChiSquareTest()
    assert metric.name == 'ChiSquareTest'
    assert metric.goal.name == 'MINIMIZE'


def test_chi_square_is_applicable():
    """Test is_applicable method of ChiSquareTest."""
    assert ChiSquareTest.is_applicable('categorical') is True
    assert ChiSquareTest.is_applicable('boolean') is True
    assert ChiSquareTest.is_applicable('numerical') is False
    assert ChiSquareTest.is_applicable('datetime') is False


def test_chi_square_compute(categorical_data):
    """Test compute method of ChiSquareTest."""
    real_data, synthetic_data = categorical_data

    result = ChiSquareTest.compute(real_data, synthetic_data)

    assert 'statistic' in result
    assert 'p_val' in result
    assert (
        result['statistic'] >= 0
    )  # Chi-square statistic is always non-negative
    assert 0 <= result['p_val'] <= 1
