SyntheRela Documentation
========================

Welcome to SyntheRela's documentation! SyntheRela is a comprehensive benchmark designed to evaluate and compare synthetic relational database generation methods.

.. image:: https://raw.githubusercontent.com/martinjurkovic/syntherela/refs/heads/main/docs/SyntheRela.png
   :alt: SyntheRela Logo
   :align: center
   :height: 150px

About SyntheRela
----------------

SyntheRela provides a standardized framework for assessing both the fidelity and utility of synthetic data across multiple real-world databases. The benchmark includes novel evaluation metrics, particularly for relational data, and supports various open-source and commercial synthetic data generation methods.

The framework is highly extensible, allowing users to benchmark on their own custom datasets and implement new evaluation metrics to suit specific use cases.

Key Features
~~~~~~~~~~~~

* **Comprehensive Metrics**: Evaluate synthetic data at multiple levels (single column, single table, multi-table)
* **Extensible Framework**: Easy to add custom metrics and datasets
* **Standardized Evaluation**: Compare different synthetic data generation methods fairly
* **Real-world Benchmarks**: Includes multiple real-world databases for testing

Quick Start
-----------

Installation
~~~~~~~~~~~~

To install SyntheRela, simply run:

.. code-block:: bash

   pip install syntherela

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from syntherela.benchmark import Benchmark
   from syntherela.metrics.single_column.statistical import ChiSquareTest
   from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy

   # Create a benchmark instance
   benchmark = Benchmark(
       real_data_dir="path/to/real/data",
       synthetic_data_dir="path/to/synthetic/data",
       results_dir="path/to/results",
       benchmark_name="my_benchmark",
       single_column_metrics=[ChiSquareTest()],
       single_table_metrics=[MaximumMeanDiscrepancy()]
   )

   # Run the benchmark
   benchmark.run()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   guides/adding_metrics
   guides/replicating_results

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/benchmark
   api/metrics
   api/data
   api/metadata
   api/report
   api/visualisations

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   GitHub Repository <https://github.com/martinjurkovic/syntherela>
   Paper on OpenReview <https://openreview.net/forum?id=ZfQofWYn6n>
   Leaderboard <https://huggingface.co/spaces/SyntheRela/leaderboard>

Citation
--------

If you use SyntheRela in your work, please cite our paper:

.. code-block:: bibtex

   @inproceedings{
       iclrsyntheticdata2025syntherela,
       title={SyntheRela: A Benchmark For Synthetic Relational Database Generation},
       author={Martin Jurkovic and Valter Hudovernik and Erik {\v{S}}trumbelj},
       booktitle={Will Synthetic Data Finally Solve the Data Access Problem?},
       year={2025},
       url={https://openreview.net/forum?id=ZfQofWYn6n}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
