SyntheRela Documentation
========================

Welcome to SyntheRela's documentation! SyntheRela is a comprehensive benchmark designed to evaluate and compare synthetic relational database generation methods.

.. image:: SyntheRela.png
   :alt: SyntheRela Logo
   :align: center
   :height: 150px

About SyntheRela
----------------

SyntheRela provides a standardized framework for assessing both the fidelity and utility of synthetic data across multiple real-world databases. The benchmark includes novel evaluation metrics, particularly for relational data, and supports various open-source and commercial synthetic data generation methods.

The framework is highly extensible, allowing users to benchmark on their own custom datasets and implement new evaluation metrics to suit specific use cases.

Our research on SyntheRela is presented in the TMLR paper **"SyntheRela: A Benchmark For Synthetic Relational Database Generation"**, available on OpenReview.

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

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples
   leaderboard
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
   TMLR Paper (OpenReview) <https://openreview.net/forum?id=Mi8XioazWy>
   Leaderboard <https://huggingface.co/spaces/SyntheRela/leaderboard>

Citation
--------

If you use SyntheRela in your work, please cite our paper:

.. code-block:: bibtex

   @article{
       hudovernik2026syntherela,
       title={SyntheRela: A Benchmark For Synthetic Relational Database Generation},
       author={Valter Hudovernik and Martin Jurkovic and Erik {\v{S}}trumbelj},
       journal={Transactions on Machine Learning Research},
       issn={2835-8856},
       year={2026},
       url={https://openreview.net/forum?id=Mi8XioazWy},
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
