"""Tests for the base metrics module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from syntherela.metrics.base import (
    DetectionBaseMetric,
    DistanceBaseMetric,
    MultiTableMetric,
    SingleColumnMetric,
    SingleTableMetric,
    StatisticalBaseMetric,
)


class TestSingleColumnMetric(SingleColumnMetric):
    """Test implementation of SingleColumnMetric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TestSingleColumnMetric'

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        return {'score': 0.5}


class TestSingleTableMetric(SingleTableMetric):
    """Test implementation of SingleTableMetric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TestSingleTableMetric'

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        return {'score': 0.5}


class TestMultiTableMetric(MultiTableMetric):
    """Test implementation of MultiTableMetric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TestMultiTableMetric'

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        return {'score': 0.5}


class TestStatisticalMetric(StatisticalBaseMetric):
    """Test implementation of StatisticalBaseMetric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TestStatisticalMetric'

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        return {'statistic': 0.3, 'p_val': 0.7}


class TestDistanceMetric(DistanceBaseMetric):
    """Test implementation of DistanceBaseMetric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'TestDistanceMetric'

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        return 0.3


class TestDetectionMetric(DetectionBaseMetric):
    """Test implementation of DetectionBaseMetric."""

    def __init__(self, **kwargs):
        super().__init__(classifier_cls=RandomForestClassifier, **kwargs)
        self.name = 'TestDetectionMetric'

    def prepare_data(self, real_data, synthetic_data, **kwargs):
        return super().prepare_data(real_data, synthetic_data)


@pytest.fixture
def column_data():
    """Generate column data for testing."""
    np.random.seed(42)
    real_data = pd.Series(np.random.normal(0, 1, 100))
    synthetic_data = pd.Series(np.random.normal(0.1, 1.1, 100))
    return real_data, synthetic_data


@pytest.fixture
def table_data():
    """Generate table data for testing."""
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


def test_single_column_is_constant():
    """Test is_constant method of SingleColumnMetric."""
    constant_column = pd.Series([5] * 100)
    varying_column = pd.Series(range(100))

    assert SingleColumnMetric.is_constant(constant_column) is True
    assert SingleColumnMetric.is_constant(varying_column) is False


def test_distance_base_metric(column_data):
    """Test DistanceBaseMetric."""
    real_data, synthetic_data = column_data

    metric = TestDistanceMetric()
    result = metric.run(real_data, synthetic_data)

    assert 'value' in result
    assert 'bootstrap_mean' in result
    assert 'bootstrap_se' in result
    assert 'reference_mean' in result
    assert 'reference_ci' in result


def test_detection_base_metric(table_data):
    """Test DetectionBaseMetric."""
    real_data, synthetic_data = table_data

    metric = TestDetectionMetric()

    # Test prepare_data
    X, y = metric.prepare_data(real_data, synthetic_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(real_data) + len(synthetic_data)
    assert len(y) == len(real_data) + len(synthetic_data)
    assert set(np.unique(y)) == {0, 1}

    # Test stratified_kfold returns per-sample 0-1 loss
    scores = metric.stratified_kfold(X, y)
    assert isinstance(scores, list)
    assert len(scores) == len(X)
