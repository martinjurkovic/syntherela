"""Tests for the base metrics module."""

import numpy as np
import pandas as pd
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


def test_single_column_is_constant():
    """Test is_constant method of SingleColumnMetric."""
    constant_column = pd.Series([5] * 100)
    varying_column = pd.Series(range(100))

    assert SingleColumnMetric.is_constant(constant_column)
    assert not SingleColumnMetric.is_constant(varying_column)


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


def test_detection_feature_importance_after_run(table_data):
    """After `run`, feature importance is available."""
    real_data, synthetic_data = table_data
    metric = TestDetectionMetric(random_state=42, folds=2)
    metric.run(real_data, synthetic_data, metadata=None)
    importance = metric.feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    for _name, scores in importance.items():
        assert len(scores) == metric.folds


def test_single_table_metric_is_applicable():
    """Test SingleTableMetric.is_applicable with metadata dict."""
    only_id = {'columns': {'pk': {'sdtype': 'id'}}}
    assert not SingleTableMetric.is_applicable(only_id)
    with_non_id = {
        'columns': {'pk': {'sdtype': 'id'}, 'x': {'sdtype': 'numerical'}},
    }
    assert SingleTableMetric.is_applicable(with_non_id)


def test_detection_base_baseline(table_data):
    """Test DetectionBaseMetric.baseline returns mean and se."""
    real_data, synthetic_data = table_data
    metric = TestDetectionMetric(random_state=42)
    mean_acc, se = metric.baseline(real_data, metadata=None, m=3)
    assert 0 <= mean_acc <= 1
    assert se >= 0


def test_detection_base_binomial_test():
    """Test DetectionBaseMetric.binomial_test."""
    stat_g, p_g = DetectionBaseMetric.binomial_test(
        8, 10, p=0.5, alternative='greater'
    )
    assert 0 <= p_g <= 1
    stat_l, p_l = DetectionBaseMetric.binomial_test(
        2, 10, p=0.5, alternative='less'
    )
    assert 0 <= p_l <= 1
