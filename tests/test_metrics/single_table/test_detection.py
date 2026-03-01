"""Tests for single table detection metrics."""

import numpy as np
import pandas as pd
import pytest
from data.data_generators import generate_real_data
from sklearn.ensemble import RandomForestClassifier
from syntherela.metrics.single_table.detection import SingleTableDetection


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    data, metadata = generate_real_data()
    return (
        data['table1'],
        data['table1'].copy(),
        metadata.get_table_meta('table1'),
    )


def test_single_table_detection_initialization():
    """Test initialization of SingleTableDetection metric."""
    # Test with default parameters
    metric = SingleTableDetection(classifier_cls=RandomForestClassifier)
    assert metric.name == 'SingleTableDetection-RandomForestClassifier'
    assert metric.classifier_cls == RandomForestClassifier
    assert metric.classifier_args == {}
    assert metric.folds == 5

    # Test with custom parameters
    custom_args = {'n_estimators': 100, 'max_depth': 5}
    metric = SingleTableDetection(
        classifier_cls=RandomForestClassifier,
        classifier_args=custom_args,
        folds=3,
        random_state=42,
    )
    assert metric.classifier_args == custom_args
    assert metric.folds == 3
    assert metric.random_state == 42


def test_single_table_detection_prepare_data(sample_data):
    """Test data preparation for SingleTableDetection metric."""
    real_data, synthetic_data, metadata = sample_data

    # Create the metric
    metric = SingleTableDetection(
        classifier_cls=RandomForestClassifier, random_state=42
    )

    # Test data preparation
    X, y = metric.prepare_data(real_data, synthetic_data, metadata)

    # Check that the output has the expected shape and types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(real_data) + len(synthetic_data)
    assert len(y) == len(real_data) + len(synthetic_data)

    # Check that ID columns are removed
    primary_key = metadata['primary_key']
    assert primary_key not in X.columns


def test_single_table_detection_run(sample_data):
    """Test running SingleTableDetection metric."""
    real_data, synthetic_data, metadata = sample_data

    # Create the metric
    metric = SingleTableDetection(
        classifier_cls=RandomForestClassifier, random_state=42
    )

    # Run the metric
    result = metric.run(real_data, synthetic_data, metadata)

    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert 'accuracy' in result
    assert 'bin_test_p_val' in result
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['bin_test_p_val'] <= 1
