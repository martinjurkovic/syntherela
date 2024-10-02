import os
from shutil import rmtree

from syntherela.data import save_tables
from syntherela.benchmark import Benchmark
from data.data_generators import generate_real_data, generate_synthetic_data


def test_benchmark():
    real_data, metadata = generate_real_data()
    synthetic_data_good = generate_synthetic_data()
    synthetic_data_bad = generate_synthetic_data(good_fit=False)
    save_tables(real_data, path="tests/tmp/original/TEST")
    if not os.path.isfile("tests/tmp/original/TEST/metadata.json"):
        metadata.save_to_json("tests/tmp/original/TEST/metadata.json")
    save_tables(synthetic_data_good, path="tests/tmp/synthetic/TEST/good")
    save_tables(synthetic_data_bad, path="tests/tmp/synthetic/TEST/bad")

    benchmark = Benchmark(
        real_data_dir="tests/tmp/original",
        synthetic_data_dir="tests/tmp/synthetic",
        results_dir="tests/tmp/results",
        benchmark_name="test_benchmark",
        datasets=["TEST"],
    )
    benchmark.run()
    assert os.path.exists("tests/tmp/results")

    # TODO: available methods are hard-coded in the Benchmark class
    benchmark.visualize_single_table_metrics()

    rmtree("tests/tmp")


test_benchmark()
