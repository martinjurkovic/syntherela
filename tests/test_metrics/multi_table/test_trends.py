"""Tests for multi-table trends (multi_table_trends module)."""

import numpy as np
import pandas as pd
import pytest
from data.data_generators import (
    generate_real_data,
    generate_real_data_three_tables,
    generate_synthetic_data,
    generate_synthetic_data_three_tables,
)
from syntherela.metrics.multi_table.trends import multi_table_trends
from syntherela.metrics.multi_table.trends.multi_table_trends import (
    PairTrendsReport,
    evaluate_long_path,
    find_paths_with_length_greater_than_one,
    get_avg_long_range_scores,
    get_joint_table,
    get_long_range,
    recursive_merge,
)


@pytest.fixture
def sample_data_two_tables():
    """Real + synthetic data for two-table structure (table1 -> table2)."""
    real, metadata = generate_real_data(seed=42)
    syn = generate_synthetic_data(seed=43)
    return real, syn, metadata


@pytest.fixture
def sample_data_three_tables():
    """Real + synthetic for multi-hop (table1 -> table2 -> table3)."""
    real, metadata = generate_real_data_three_tables(seed=42)
    syn = generate_synthetic_data_three_tables(seed=43)
    return real, syn, metadata


# ----- recursive_merge -----


def test_recursive_merge_two_tables():
    """recursive_merge with two dataframes merges on the given key pair."""
    df1 = pd.DataFrame({'pk': [1, 2], 'a': [10, 20]})
    df2 = pd.DataFrame({'fk': [1, 1, 2], 'b': [100, 101, 200]})
    merged = recursive_merge([df1, df2], [('fk', 'pk')])
    assert len(merged) == 3
    assert list(merged.columns) == ['fk', 'b', 'pk', 'a']
    assert merged['a'].tolist() == [10, 10, 20]


def test_recursive_merge_three_tables():
    """recursive_merge with three dataframes produces correct join."""
    df1 = pd.DataFrame({'p1': [1, 2], 'x': [1, 2]})
    # f1 references df1.p1 so row p2=1->f1=1, p2=2->f1=2
    df2 = pd.DataFrame({'p2': [1, 2, 3], 'f1': [1, 2, 2], 'y': [10, 11, 20]})
    df3 = pd.DataFrame({'p3': [1, 2], 'f2': [1, 2], 'z': [100, 200]})
    merged = recursive_merge(
        [df1, df2, df3],
        [('f1', 'p1'), ('f2', 'p2')],
    )
    assert len(merged) == 2
    assert 'x' in merged.columns and 'z' in merged.columns
    assert merged['x'].tolist() == [1, 2]
    assert merged['z'].tolist() == [100, 200]


# ----- get_joint_table -----


def test_get_joint_table_two_tables(sample_data_two_tables):
    """get_joint_table on two-table path returns joined table and metadata."""
    real, _, metadata = sample_data_two_tables
    path = ['table1', 'table2']
    joined, table_meta = get_joint_table(path, real, metadata)
    assert isinstance(joined, pd.DataFrame)
    assert len(joined) == len(real['table2'])
    assert table_meta is not None
    # End tables' columns present
    assert 'pk1' in joined.columns or 'normal' in joined.columns
    assert 'pk2' in joined.columns or 'lognormal' in joined.columns


def test_get_joint_table_three_tables(sample_data_three_tables):
    """get_joint_table on a three-table path drops intermediate columns."""
    real, _, metadata = sample_data_three_tables
    path = ['table1', 'table2', 'table3']
    joined, table_meta = get_joint_table(path, real, metadata)
    assert isinstance(joined, pd.DataFrame)
    assert len(joined) == len(real['table3'])
    # table2-only columns should be removed (intermediate table)
    table2_cols = {'pk2', 'fk2', 'y', 'label', 'z'}
    for col in table2_cols:
        assert col not in joined.columns
    assert 'pk1' in joined.columns or 'x' in joined.columns
    assert 'pk3' in joined.columns or 'w' in joined.columns


# ----- find_paths_with_length_greater_than_one -----


def test_find_paths_two_tables_empty():
    """Two-table schema has no path with length > 1 (no multi-hop)."""
    from syntherela.metadata import Metadata

    meta = Metadata()
    meta.add_table('t1')
    meta.add_column('t1', 'pk1', sdtype='id')
    meta.set_primary_key('t1', 'pk1')
    meta.add_table('t2')
    meta.add_column('t2', 'pk2', sdtype='id')
    meta.add_column('t2', 'fk', sdtype='id')
    meta.set_primary_key('t2', 'pk2')
    meta.add_relationship('t1', 't2', 'pk1', 'fk')
    meta.validate()

    paths = find_paths_with_length_greater_than_one(meta)
    assert paths == []


def test_find_paths_three_tables_one_path(sample_data_three_tables):
    """Three-table chain yields one path [table1, table2, table3]."""
    _, _, metadata = sample_data_three_tables
    paths = find_paths_with_length_greater_than_one(metadata)
    assert len(paths) == 1
    assert paths[0] == ['table1', 'table2', 'table3']


# ----- get_avg_long_range_scores -----


def test_get_avg_long_range_scores_empty():
    """get_avg_long_range_scores with empty dict returns empty dicts."""
    avg, se = get_avg_long_range_scores({})
    assert avg == {}
    assert se == {}


def test_get_avg_long_range_scores_skips_empty_hop():
    """get_avg_long_range_scores skips hops with no scores."""
    res = {1: {}, 2: {'a': 0.8, 'b': 1.0}}
    avg, se = get_avg_long_range_scores(res)
    assert 1 not in avg and 1 not in se
    assert 2 in avg and 2 in se
    assert avg[2] == pytest.approx(0.9)
    assert se[2] == pytest.approx(0.1 / np.sqrt(2))


# ----- PairTrendsReport -----


def test_pair_trends_report_init():
    """PairTrendsReport has Column Pair Trends property."""
    report = PairTrendsReport()
    assert 'Column Pair Trends' in report._properties


# ----- evaluate_long_path -----


def test_evaluate_long_path(sample_data_three_tables):
    """evaluate_long_path returns dict of pair keys to scores."""
    real, syn, metadata = sample_data_three_tables
    from copy import deepcopy

    from syntherela.data import make_column_names_unique

    real_u, syn_u, meta_u = make_column_names_unique(
        deepcopy(real), deepcopy(syn), deepcopy(metadata), validate=True
    )
    path = ['table1', 'table2', 'table3']
    real_joined, table_meta = get_joint_table(path, real_u, meta_u)
    syn_joined, _ = get_joint_table(path, syn_u, meta_u)
    top = path[0]
    bottom = path[-1]
    top_cols = list(meta_u.tables[top].columns.keys())
    bottom_cols = list(meta_u.tables[bottom].columns.keys())

    scores = evaluate_long_path(
        real_joined,
        syn_joined,
        table_meta,
        top_cols,
        bottom_cols,
        top,
        bottom,
        verbose=False,
    )
    assert isinstance(scores, dict)
    for k, v in scores.items():
        assert top in k and bottom in k
        assert isinstance(v, (int, float))
        assert 0 <= v <= 1 or np.isnan(v)


# ----- get_long_range -----


def test_get_long_range_two_tables_empty(sample_data_two_tables):
    real, syn, metadata = sample_data_two_tables
    res = get_long_range(real, syn, metadata, verbose=False)
    # Only paths with length > 1 (3+ tables) are considered
    assert res == {} or all(h >= 2 for h in res.keys())


def test_get_long_range_three_tables(sample_data_three_tables):
    """Three-table schema yields long-range scores for 2-hop path."""
    real, syn, metadata = sample_data_three_tables
    res = get_long_range(real, syn, metadata, verbose=False)
    assert 2 in res
    assert isinstance(res[2], dict)
    for key in res[2]:
        assert 'table1' in key and 'table3' in key


# ----- multi_table_trends (public API) -----


def test_multi_table_trends_smoke(sample_data_two_tables):
    """Run multi_table_trends on two-table generated real/synthetic data."""
    real_data, synthetic_data, metadata = sample_data_two_tables
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
    assert 1 in result['hop_relation']
    assert isinstance(result['cardinality'], (int, float))


def test_multi_table_trends_three_tables_structure(sample_data_three_tables):
    """multi_table_trends on 3-table data has hop_relation for 1 and 2 hops."""
    real, syn, metadata = sample_data_three_tables
    result = multi_table_trends(real, syn, metadata, verbose=False)
    assert 'hop_relation' in result
    assert 1 in result['hop_relation']
    assert 2 in result['hop_relation']
    assert len(result['hop_relation'][1]) > 0
    assert len(result['hop_relation'][2]) > 0
    assert 'avg_scores' in result
    assert 'scores_se' in result
    assert 'all_avg_score' in result
    assert 'cardinality' in result
    assert isinstance(result['all_avg_score'], (int, float))
    assert isinstance(result['cardinality'], (int, float))


def test_multi_table_trends_three_tables_columns_subset(
    sample_data_three_tables,
):
    """multi_table_trends restricts syn_tables to real table columns."""
    real, syn, metadata = sample_data_three_tables
    # Add extra column to synthetic only
    syn['table1'] = syn['table1'].copy()
    syn['table1']['extra'] = 999
    result = multi_table_trends(real, syn, metadata, verbose=False)
    assert 'hop_relation' in result
    assert result['hop_relation'] is not None
