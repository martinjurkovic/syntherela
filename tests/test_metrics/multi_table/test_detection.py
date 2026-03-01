"""Tests for multi-table detection metrics."""

from sklearn.ensemble import RandomForestClassifier
from syntherela.metrics.multi_table.detection import (
    AggregationDetection,
    ParentChildDetection,
)


def test_aggregation_detection_initialization():
    """Test initialization of AggregationDetection metric."""
    # Test with default parameters
    metric = AggregationDetection(classifier_cls=RandomForestClassifier)
    assert (
        metric.name == f'AggregationDetection-{RandomForestClassifier.__name__}'
    )
    assert metric.classifier_cls == RandomForestClassifier
    assert metric.classifier_args == {}
    assert metric.folds == 5

    # Test with custom parameters
    custom_args = {'n_estimators': 100, 'max_depth': 5}
    metric = AggregationDetection(
        classifier_cls=RandomForestClassifier,
        classifier_args=custom_args,
        folds=3,
        random_state=42,
    )
    assert metric.classifier_args == custom_args
    assert metric.folds == 3
    assert metric.random_state == 42

    assert metric.classifier_args == custom_args
    assert metric.folds == 3
    assert metric.random_state == 42


def test_aggregation_detection_run(sample_data):
    """Test computation of AggregationDetection metric."""
    data, metadata = sample_data

    # Create the metric
    metric = AggregationDetection(
        classifier_cls=RandomForestClassifier,
        random_state=42,
        folds=2,
    )

    # run the metric
    result = metric.run(data, data, metadata)

    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert 'accuracy' in result['table1']
    assert 'SE' in result['table1']
    assert 'bin_test_p_val' in result['table1']
    assert 'copying_p_val' in result['table1']
    assert 0 <= result['table1']['accuracy'] <= 1
    assert 0 <= result['table1']['SE'] <= 1
    assert 0 <= result['table1']['bin_test_p_val'] <= 1
    assert 0 <= result['table1']['copying_p_val'] <= 1


def test_parent_child_detection_run(sample_data):
    """Test computation of ParentChildDetection metric."""
    data, metadata = sample_data

    # Create the metric
    metric = ParentChildDetection(
        classifier_cls=RandomForestClassifier, random_state=42
    )

    # run the metric
    result = metric.run(data, data, metadata)

    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert 'accuracy' in result['table1_table2_fk2']
    assert 'SE' in result['table1_table2_fk2']
    assert 'bin_test_p_val' in result['table1_table2_fk2']
    assert 'copying_p_val' in result['table1_table2_fk2']
    assert 0 <= result['table1_table2_fk2']['accuracy'] <= 1
    assert 0 <= result['table1_table2_fk2']['SE'] <= 1
    assert 0 <= result['table1_table2_fk2']['bin_test_p_val'] <= 1
    assert 0 <= result['table1_table2_fk2']['copying_p_val'] <= 1
