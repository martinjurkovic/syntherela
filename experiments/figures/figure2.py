import shutil
from syntherela.benchmark import Benchmark

benchmark = Benchmark(
    real_data_dir="data/original",
    synthetic_data_dir="data/synthetic",
    results_dir=f"results/3",
    benchmark_name="Benchmark",
    single_column_metrics=[],
    single_table_metrics=[],
    multi_table_metrics=[],
    run_id="3",
    sample_id="sample1",
    datasets=["rossmann_subsampled"],
    methods=["SDV", "RCTGAN", "REALTABFORMER", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM", "CLAVADDPM"],
    validate_metadata=False,
)

benchmark.read_results()

benchmark.visualize_single_table_metrics(
    save_figs=True,
    save_figs_path="results/figures/tmp/",
    datasets=["rossmann_subsampled"],
    detection=False,
    log_scale=True,
    title=False,
    MaximumMeanDiscrepancy_pow=7,
)

shutil.copy(
    "results/figures/tmp/single_table/distance/rossmann_subsampled_MaximumMeanDiscrepancy.png",
    "results/figures/figure2a.png",
)
shutil.copy(
    "results/figures/tmp/single_table/distance/rossmann_subsampled_PairwiseCorrelationDifference.png",
    "results/figures/figure2b.png",
)
# remove the tmp folder
shutil.rmtree("results/figures/tmp")
