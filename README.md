# SyntheRela - Synthetic Relational Data Generation Benchmark

<h2 align="center">
    <img src="https://raw.githubusercontent.com/martinjurkovic/syntherela/refs/heads/main/docs/SyntheRela.png" height="150px">
    <div align="center">
      <a href="https://pypi.org/project/syntherela/">
        <img src="https://img.shields.io/pypi/v/syntherela" alt="PyPI">
      </a>
      <a href="https://github.com/martinjurkovic/syntherela/blob/main/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
      </a>
      <a href="https://openreview.net/forum?id=Mi8XioazWy">
        <img alt="Paper URL" src="https://img.shields.io/badge/OpenReview-Mi8XioazWy-B31B1B.svg">
      </a>
      <a href="https://huggingface.co/spaces/SyntheRela/leaderboard">
        <img src="https://img.shields.io/badge/🤗-Leaderboard-yellow.svg" alt="Hugging Face Leaderboard">
      </a>
  </div>
</h2>

## About SyntheRela

SyntheRela is a comprehensive benchmark designed to evaluate and compare synthetic relational database generation methods. It provides a standardized framework for assessing both the fidelity and utility of synthetic data across multiple real-world databases. The benchmark includes novel evaluation metrics, particularly for relational data, and supports various open-source and commercial synthetic data generation methods.

SyntheRela is highly extensible, allowing users to benchmark on their own custom datasets and implement new evaluation metrics to suit specific use cases.

Our research on SyntheRela is presented in the TMLR paper **"SyntheRela: A Benchmark For Synthetic Relational Database Generation"**, available on [OpenReview](https://openreview.net/forum?id=Mi8XioazWy).

We maintain a [public leaderboard on Hugging Face](https://huggingface.co/spaces/SyntheRela/leaderboard) where you can compare the performance of different synthetic data generation methods.

## Installation

To install only the benchmark package, run the following command:

```bash
pip install syntherela
```

## Using SyntheRela

To evaluate your synthetic relational data, configure the `Benchmark` class with your desired metrics and run the evaluation pipeline:

```python
from syntherela.benchmark import Benchmark
from syntherela.metrics.single_column.statistical import ChiSquareTest
from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy
from syntherela.metrics.multi_table.statistical import CardinalityShapeSimilarity
from syntherela.metrics.multi_table.detection import AggregationDetection
from xgboost import XGBClassifier

# Initialize the benchmark with specific metrics
benchmark = Benchmark(
    real_data_dir="path/to/real_data",
    synthetic_data_dir="path/to/synthetic_data",
    results_dir="results",
    benchmark_name="my_benchmark",
    single_column_metrics=[ChiSquareTest()],
    single_table_metrics=[MaximumMeanDiscrepancy()],
    multi_table_metrics=[
        CardinalityShapeSimilarity(),
        AggregationDetection(classifier_cls=XGBClassifier, random_state=42)
    ],
    datasets=["your_dataset_name"],
    methods=["your_method_name"]
)

# Execute evaluation
benchmark.run()
```

## Examples

We provide example notebooks to help you get started with `syntherela` in the [examples/](examples/) directory.

- [Evaluating Rossmann Subsampled Dataset](examples/evaluate_rossmann_subsampled.ipynb): A step-by-step guide to evaluating a subsampled version of the Rossmann dataset using various metrics.

## Replicating the paper's results

For detailed instructions on how to replicate the paper's results, please refer to [docs/REPLICATING_RESULTS.md](/docs/REPLICATING_RESULTS.md).

## Adding a new metric

The documentation for adding a new metric can be found in [docs/ADDING_A_METRIC.md](/docs/ADDING_A_METRIC.md).

\* Denotes the method does not have a public implementation available.

## 🏆 Leaderboard Submission

We maintain an official leaderboard to benchmark synthetic relational data generation methods. To ensure fairness and reproducibility, **all evaluations are performed by the SyntheRela maintainers** on standardized hardware.

### Evaluation Overview

| Feature                  | Specification                                     |
| :----------------------- | :------------------------------------------------ |
| **Compute**              | Single NVIDIA H100 (80GB)                         |
| **Time Limit**           | 48 hours execution time **per dataset**           |
| **Submission Frequency** | 1 submission per 30-day period                    |
| **Capacity**             | Up to 2 model variants/checkpoints per submission |

### How to Submit

1. **Prepare your code:** Ensure your method is reproducible and includes a clear `README` and `requirements.txt`.
2. **Open an Issue:** Create a new [GitHub Issue](https://github.com/martinjurkovic/syntherela/issues) using the title prefix `[Model Submission]`.

For the complete requirements regarding environment setup, logging, and our privacy/confidentiality policy, please refer to our **[Full Submission Guidelines](https://docs.google.com/document/d/1ae16L_vvT5PFt2OeN7FJauA_ayd_A6xCkhVJFoYcx04)**.

## Conflicts of Interest

The authors declare no conflict of interest and are not associated with any of the evaluated commercial synthetic data providers.

## Citation

If you use SyntheRela in your work, please cite our paper:

```
@article{
hudovernik2026syntherela,
title={SyntheRela: A Benchmark For Synthetic Relational Database Generation},
author={Valter Hudovernik and Martin Jurkovic and Erik {\v{S}}trumbelj},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=Mi8XioazWy},
}
```

## License

This project is licensed under the [MIT License](/LICENSE).
