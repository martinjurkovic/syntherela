"""Tests for single column detection metric."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from syntherela.metrics.single_column.detection import SingleColumnDetection


def test_single_column_detection_init():
    """Test initialization of SingleColumnDetection metric."""
    metric = SingleColumnDetection(classifier_cls=RandomForestClassifier)
    assert 'SingleColumnDetection' in metric.name
    assert metric.classifier_cls == RandomForestClassifier
    assert metric.folds == 5


def test_single_column_detection_is_applicable():
    """Test is_applicable for SingleColumnDetection."""
    assert SingleColumnDetection.is_applicable('categorical') is True
    assert SingleColumnDetection.is_applicable('numerical') is True
    assert SingleColumnDetection.is_applicable('boolean') is True
    assert SingleColumnDetection.is_applicable('datetime') is True


def test_single_column_detection_prepare_data(numerical_data):
    """Test prepare_data for SingleColumnDetection using numerical fixture."""
    real_data, synthetic_data = numerical_data
    metric = SingleColumnDetection(
        classifier_cls=RandomForestClassifier, random_state=42
    )
    X, y = metric.prepare_data(real_data, synthetic_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(real_data) + len(synthetic_data)
    assert set(np.unique(y)) == {0, 1}


def test_single_column_detection_run_numerical(numerical_data):
    """Test run for SingleColumnDetection with numerical_data fixture."""
    real_data, synthetic_data = numerical_data
    metric = SingleColumnDetection(
        classifier_cls=RandomForestClassifier, random_state=42, folds=2
    )
    result = metric.run(real_data, synthetic_data, metadata=None)
    assert isinstance(result, dict)
    assert 'accuracy' in result
    assert 'bin_test_p_val' in result
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['bin_test_p_val'] <= 1


def test_single_column_detection_run_categorical(categorical_data):
    """Test run for SingleColumnDetection with categorical_data fixture."""
    real_data, synthetic_data = categorical_data
    metric = SingleColumnDetection(
        classifier_cls=RandomForestClassifier, random_state=42, folds=2
    )
    result = metric.run(real_data, synthetic_data, metadata=None)
    assert isinstance(result, dict)
    assert 'accuracy' in result
    assert 'bin_test_p_val' in result
    assert 0 <= result['accuracy'] <= 1
