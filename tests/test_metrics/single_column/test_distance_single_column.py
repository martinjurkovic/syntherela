"""Tests for single column distance metrics."""

import numpy as np
import pandas as pd
import pytest
from syntherela.metrics.single_column.distance import (
    HellingerDistance,
    JensenShannonDistance,
    TotalVariationDistance,
    WassersteinDistance,
)


@pytest.fixture
def numerical_data():
    np.random.seed(42)
    real = pd.Series(np.random.normal(0, 1, 80))
    synth = pd.Series(np.random.normal(0.1, 1.1, 80))
    return real, synth


@pytest.fixture
def categorical_data():
    np.random.seed(42)
    real = pd.Series(np.random.choice(['A', 'B', 'C'], 80))
    synth = pd.Series(np.random.choice(['A', 'B', 'C'], 80))
    return real, synth


def test_wasserstein_init_and_applicable():
    m = WassersteinDistance()
    assert m.name == 'WassersteinDistance'
    assert WassersteinDistance.is_applicable('numerical') is True
    assert WassersteinDistance.is_applicable('datetime') is True
    assert WassersteinDistance.is_applicable('categorical') is False


def test_wasserstein_compute_and_run(numerical_data):
    real, synth = numerical_data
    m = WassersteinDistance()
    out = m.run(real, synth)
    assert 'value' in out
    assert out['value'] >= 0
    assert 'bootstrap_mean' in out


def test_wasserstein_constant_column():
    c = pd.Series([1.0] * 50)
    m = WassersteinDistance()
    out = m.run(c, c)
    assert out['value'] == 0


def test_jensen_shannon_init_and_applicable():
    m = JensenShannonDistance()
    assert m.name == 'JensenShannonDistance'
    assert JensenShannonDistance.is_applicable('categorical') is True
    assert JensenShannonDistance.is_applicable('numerical') is True


def test_jensen_shannon_run(numerical_data, categorical_data):
    m = JensenShannonDistance()
    real_n, synth_n = numerical_data
    out = m.run(real_n, synth_n)
    assert 'value' in out
    assert 0 <= out['value'] <= 2
    real_c, synth_c = categorical_data
    out_c = m.run(real_c, synth_c)
    assert 'value' in out_c


def test_total_variation_init_and_run(categorical_data):
    m = TotalVariationDistance()
    assert m.name == 'TotalVariationDistance'
    real, synth = categorical_data
    out = m.run(real, synth)
    assert 'value' in out
    assert out['value'] >= 0


def test_hellinger_init_and_applicable():
    m = HellingerDistance()
    assert m.name == 'HellingerDistance'
    assert m.max_value == 1
    assert HellingerDistance.is_applicable('boolean') is True


def test_hellinger_run(numerical_data):
    real, synth = numerical_data
    m = HellingerDistance()
    out = m.run(real, synth)
    assert 'value' in out
    assert 0 <= out['value'] <= 1


def test_hellinger_hellinger_method():
    p, q = np.array([0.5, 0.5]), np.array([0.5, 0.5])
    assert HellingerDistance.hellinger(p, q) == 0
