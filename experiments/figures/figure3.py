import shutil
from syntherela.benchmark import Benchmark


benchmark = Benchmark(
    real_data_dir="data/original",
    synthetic_data_dir="data/synthetic",
    results_dir=f"results/1",
    benchmark_name="Benchmark",
    run_id="1",
    sample_id="sample1",
    datasets=["Biodegradability_v1"],
    methods=["SDV", "RCTGAN", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM"],
    validate_metadata=False,
)

benchmark.read_results()

benchmark.visualize_single_table_metrics(
    save_figs=True,
    save_figs_path="results/figures/tmp/",
    distance=False,
    detection=True,
    title=False,
)

shutil.copy(
    "results/figures/tmp/single_table/detection/Biodegradability_v1_molecule_per_table.png",
    "results/figures/figure3a.png",
)

benchmark = Benchmark(
    real_data_dir="data/original",
    synthetic_data_dir="data/synthetic",
    results_dir=f"results/1",
    benchmark_name="Benchmark",
    run_id="1",
    sample_id="sample1",
    datasets=["rossmann_subsampled"],
    methods=["SDV", "RCTGAN", "REALTABFORMER", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM", "CLAVADDPM"],
    validate_metadata=False,
)

benchmark.read_results()

benchmark.visualize_single_table_metrics(
    save_figs=True,
    save_figs_path="results/figures/tmp/",
    distance=False,
    detection=True,
    title=False,
)

shutil.copy(
    "results/figures/tmp/single_table/detection/rossmann_subsampled_store_per_table.png",
    "results/figures/figure3b.png",
)
# remove the tmp folder
shutil.rmtree("results/figures/tmp")
