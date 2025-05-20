# SyntheRela - Synthetic Relational Data Generation Benchmark

<h2 align="center">
    <img src="docs/SyntheRela.png" height="150px">
    <div align="center">
      <a href="anonymized">
        <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
      </a>
      <a href="anonymized">
        <img src="https://img.shields.io/badge/🤗-Leaderboard-yellow.svg" alt="Hugging Face Leaderboard">
      </a>
  </div>
</h2>

## About SyntheRela

SyntheRela is a comprehensive benchmark designed to evaluate and compare synthetic relational database generation methods. It provides a standardized framework for assessing both the fidelity and utility of synthetic data across multiple real-world databases. The benchmark includes novel evaluation metrics, particularly for relational data, and supports various open-source and commercial synthetic data generation methods.

SyntheRela is highly extensible, allowing users to benchmark on their own custom datasets and implement new evaluation metrics to suit specific use cases.

## Installation
To install only the benchmark package, run the following command:

```bash
pip install .
```

## Replicating the paper's results

For detailed instructions on how to replicate the paper's results, please refer to [docs/REPLICATING_RESULTS.md](/docs/REPLICATING_RESULTS.md).

## Adding a new metric
The documentation for adding a new metric can be found in [docs/ADDING_A_METRIC.md](/docs/ADDING_A_METRIC.md).



\* Denotes the method does not have a public implementation available.

## Conflicts of Interest
The authors declare no conflict of interest and are not associated with any of the evaluated commercial synthetic data providers.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
