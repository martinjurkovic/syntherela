import os

from data.data_generators import generate_real_data, generate_synthetic_data
from syntherela.benchmark import Benchmark
from syntherela.data import save_tables
from syntherela.metrics.multi_table import CardinalityShapeSimilarity
from syntherela.metrics.single_column.statistical import ChiSquareTest
from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy


def test_benchmark():
    real_data, metadata = generate_real_data()
    synthetic_data_good = generate_synthetic_data()
    synthetic_data_bad = generate_synthetic_data(good_fit=False)
    save_tables(real_data, path='tests/tmp/original/TEST')
    if not os.path.isfile('tests/tmp/original/TEST/metadata.json'):
        metadata.save_to_json('tests/tmp/original/TEST/metadata.json')
    save_tables(synthetic_data_good, path='tests/tmp/synthetic/TEST/good')
    save_tables(synthetic_data_bad, path='tests/tmp/synthetic/TEST/bad')

    benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='test_benchmark',
        datasets=['TEST'],
    )
    benchmark.run()
    assert os.path.exists('tests/tmp/results')

    # TODO: available methods are hard-coded in the Benchmark class
    benchmark.visualize_single_table_metrics()


def test_benchmark_datasets_inferred_when_none():
    """Benchmark with datasets=None infers datasets from synthetic_data_dir."""
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()
    save_tables(real_data, path='tests/tmp/original/TEST')
    if not os.path.isfile('tests/tmp/original/TEST/metadata.json'):
        metadata.save_to_json('tests/tmp/original/TEST/metadata.json')
    save_tables(synthetic_data, path='tests/tmp/synthetic/TEST/m1')

    benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='test',
        datasets=None,
    )
    assert benchmark.datasets == ['TEST']


def test_benchmark_methods_as_list():
    """Benchmark with methods as list assigns same methods to each dataset."""
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()
    save_tables(real_data, path='tests/tmp/original/TEST')
    if not os.path.isfile('tests/tmp/original/TEST/metadata.json'):
        metadata.save_to_json('tests/tmp/original/TEST/metadata.json')
    save_tables(synthetic_data, path='tests/tmp/synthetic/TEST/m1')

    benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='test',
        datasets=['TEST'],
        methods=['m1'],
    )
    assert benchmark.methods == {'TEST': ['m1']}


def test_benchmark_methods_as_dict():
    """Benchmark with methods as dict uses per-dataset method lists."""
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()
    save_tables(real_data, path='tests/tmp/original/TEST')
    if not os.path.isfile('tests/tmp/original/TEST/metadata.json'):
        metadata.save_to_json('tests/tmp/original/TEST/metadata.json')
    save_tables(synthetic_data, path='tests/tmp/synthetic/TEST/m1')

    benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='test',
        datasets=['TEST'],
        methods={'TEST': ['m1']},
    )
    assert benchmark.methods == {'TEST': ['m1']}


def test_load_results_from_finished_benchmark():
    """Load single-column, single-table, and multi-table results benchmark."""
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()
    save_tables(real_data, path='tests/tmp/original/TEST')
    if not os.path.isfile('tests/tmp/original/TEST/metadata.json'):
        metadata.save_to_json('tests/tmp/original/TEST/metadata.json')
    save_tables(synthetic_data, path='tests/tmp/synthetic/TEST/m1')

    # Run benchmark with one metric per level so Trends are computed and saved
    run_benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='load_test',
        datasets=['TEST'],
        methods=['m1'],
        single_column_metrics=[ChiSquareTest()],
        single_table_metrics=[MaximumMeanDiscrepancy()],
        multi_table_metrics=[CardinalityShapeSimilarity()],
    )
    run_benchmark.run()
    assert os.path.isfile('tests/tmp/results/TEST_m1.json')

    # New Benchmark instance: load results from disk (no run)
    load_benchmark = Benchmark(
        real_data_dir='tests/tmp/original',
        synthetic_data_dir='tests/tmp/synthetic',
        results_dir='tests/tmp/results',
        benchmark_name='load_test',
        datasets=['TEST'],
        methods=['m1'],
    )
    single_col = load_benchmark.get_single_column_results('TEST', 'm1')
    single_tbl = load_benchmark.get_single_table_results('TEST', 'm1')
    multi_tbl = load_benchmark.get_multi_table_results('TEST', 'm1')

    assert isinstance(single_col, dict)
    assert isinstance(single_tbl, dict)
    assert isinstance(multi_tbl, dict)
    assert 'Trends' in single_col
    assert 'Trends' in single_tbl
    assert 'Trends' in multi_tbl
