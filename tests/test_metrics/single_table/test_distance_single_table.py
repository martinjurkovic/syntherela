"""Tests for single table distance metrics."""

import numpy as np
import pytest
from syntherela.metrics.single_table.distance import (
    MaximumMeanDiscrepancy,
    PairwiseCorrelationDifference,
)


def test_mmd_init_and_applicable(table_meta):
    m = MaximumMeanDiscrepancy()
    assert m.name == 'MaximumMeanDiscrepancy'
    assert MaximumMeanDiscrepancy.is_applicable(table_meta) is True
    only_id = {'columns': {'pk': {'sdtype': 'id'}}}
    assert MaximumMeanDiscrepancy.is_applicable(only_id) is False


def test_mmd_run(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    synth['normal'] = synth['normal'] + 0.5
    m = MaximumMeanDiscrepancy()
    result = m.run(real, synth, metadata=table_meta)
    assert 'value' in result
    assert result['value'] >= 0
    assert 'bootstrap_mean' in result


def test_mmd_compute_linear(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    score = MaximumMeanDiscrepancy.compute(
        real, synth, table_meta, kernel='linear'
    )
    assert isinstance(score, (float, np.floating))
    assert score >= 0


def test_mmd_compute_rbf(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    score = MaximumMeanDiscrepancy.compute(
        real, synth, table_meta, kernel='rbf'
    )
    assert isinstance(score, (float, np.floating))
    assert score >= 0


def test_mmd_compute_polynomial(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    score = MaximumMeanDiscrepancy.compute(
        real, synth, table_meta, kernel='polynomial'
    )
    assert isinstance(score, (float, np.floating))
    assert score >= 0


def test_mmd_compute_invalid_kernel_raises(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    with pytest.raises(ValueError, match=r'Unsupported kernel'):
        MaximumMeanDiscrepancy.compute(
            real, synth, table_meta, kernel='invalid'
        )


def test_pcd_init():
    m = PairwiseCorrelationDifference()
    assert m.name == 'PairwiseCorrelationDifference'
    assert m.norm_order == 'fro'
    m2 = PairwiseCorrelationDifference(
        norm_order='nuc', correlation_method='spearman'
    )
    assert m2.norm_order == 'nuc'
    assert m2.correlation_method == 'spearman'


def test_pcd_is_applicable(table_meta):
    assert PairwiseCorrelationDifference.is_applicable(table_meta) is True
    meta = {'columns': {'a': {'sdtype': 'id'}, 'b': {'sdtype': 'numerical'}}}
    assert not PairwiseCorrelationDifference.is_applicable(meta)


def test_pcd_run(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    synth['normal'] = synth['normal'] + 0.3
    m = PairwiseCorrelationDifference()
    result = m.run(real, synth, metadata=table_meta)
    assert 'value' in result
    assert 0 <= result['value'] <= 1
    assert 'bootstrap_mean' in result


def test_pcd_compute(sample_data, table_meta):
    data, _ = sample_data
    real = data['table1']
    synth = data['table1'].copy()
    m = PairwiseCorrelationDifference()
    score = m.compute(real, synth, table_meta)
    assert isinstance(score, (float, np.floating))
    assert 0 <= score <= 1
