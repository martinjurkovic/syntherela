"""Smoke tests for multi-table trends."""

from data.data_generators import generate_real_data, generate_synthetic_data
from syntherela.metrics.multi_table.trends import multi_table_trends


def test_multi_table_trends_smoke():
    """Run multi_table_trends on generated real/synthetic data."""
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()
    result = multi_table_trends(
        real_data,
        synthetic_data,
        metadata,
        verbose=False,
    )
    assert 'hop_relation' in result
    assert 'avg_scores' in result
    assert 'scores_se' in result
    assert 'all_avg_score' in result
    assert 'cardinality' in result
