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
      <a href="https://openreview.net/forum?id=ZfQofWYn6n">
        <img alt="Paper URL" src="https://img.shields.io/badge/OpenReview-ZfQofWYn6n-B31B1B.svg">
      </a>
      <a href="https://pypi.org/pypi/syntherela/">
        <img src="https://img.shields.io/pypi/pyversions/syntherela" alt="PyPI pyversions">
      </a>
      <a href="https://huggingface.co/spaces/SyntheRela/leaderboard">
        <img src="https://img.shields.io/badge/🤗-Leaderboard-yellow.svg" alt="Hugging Face Leaderboard">
      </a>
  </div>
</h2>

## About SyntheRela

SyntheRela is a comprehensive benchmark designed to evaluate and compare synthetic relational database generation methods. It provides a standardized framework for assessing both the fidelity and utility of synthetic data across multiple real-world databases. The benchmark includes novel evaluation metrics, particularly for relational data, and supports various open-source and commercial synthetic data generation methods.

SyntheRela is highly extensible, allowing users to benchmark on their own custom datasets and implement new evaluation metrics to suit specific use cases. 

Our research on SyntheRela is presented in the paper **"SyntheRela: A Benchmark For Synthetic Relational Database Generation"** at the ICLR 2025 Workshop "Will Synthetic Data Finally Solve the Data Access Problem?", available on [OpenReview](https://openreview.net/forum?id=ZfQofWYn6n).

We maintain a [public leaderboard on Hugging Face](https://huggingface.co/spaces/SyntheRela/leaderboard) where you can compare the performance of different synthetic data generation methods.

## Installation
To install only the benchmark package, run the following command:

```bash
pip install syntherela
```

## Replicating the paper's results

For detailed instructions on how to replicate the paper's results, please refer to [docs/REPLICATING_RESULTS.md](/docs/REPLICATING_RESULTS.md).

## Adding a new metric
The documentation for adding a new metric can be found in [docs/ADDING_A_METRIC.md](/docs/ADDING_A_METRIC.md).

## Synthetic Data Methods
### Open Source Methods
- SDV: [The Synthetic Data Vault](https://ieeexplore.ieee.org/document/7796926)
- RCTGAN: [Row Conditional-TGAN for Generating Synthetic Relational Databases](https://ieeexplore.ieee.org/abstract/document/10096001)
- REaLTabFormer: [Generating Realistic Relational and Tabular Data using Transformers](https://arxiv.org/abs/2302.02041)
- ClavaDDPM: [Multi-relational Data Synthesis with Cluster-guided Diffusion Models](https://arxiv.org/html/2405.17724v1)
- IRG: [Generating Synthetic Relational Databases using GANs](https://arxiv.org/abs/2312.15187)
- RGCLD: [Relational Data Generation with Graph Neural Networks and Latent Diffusion Models](https://openreview.net/forum?id=MNLR2NYN2Z#discussion)
- [Generating Realistic Synthetic Relational Data through Graph Variational Autoencoders](https://arxiv.org/abs/2211.16889)*
- [Generative Modeling of Complex Data](https://arxiv.org/abs/2202.02145)*
- BayesM2M & NeuralM2M: [Synthetic Data Generation of Many-to-Many Datasets via Random Graph Generation](https://iclr.cc/virtual/2023/poster/10982)*


\* Denotes the method does not have a public implementation available.

### Commercial Providers
A list of commercial synthetic relational data providers is available in [docs/SYNTHETIC_DATA_TOOLS.md](/docs/SYNTHETIC_DATA_TOOLS.md).

## Conflicts of Interest
The authors declare no conflict of interest and are not associated with any of the evaluated commercial synthetic data providers.

## Citation
If you use SyntheRela in your work, please cite our paper:
```
@inproceedings{
    iclrsyntheticdata2025syntherela,
    title={SyntheRela: A Benchmark For Synthetic Relational Database Generation},
    author={Martin Jurkovic and Valter Hudovernik and Erik {\v{S}}trumbelj},
    booktitle={Will Synthetic Data Finally Solve the Data Access Problem?},
    year={2025},
    url={https://openreview.net/forum?id=ZfQofWYn6n}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.