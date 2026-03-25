Quick Start Guide
=================

This guide will help you get started with SyntheRela.

Basic Usage
-----------

Here's a simple example of how to use SyntheRela to benchmark synthetic data:

.. code-block:: python

   from syntherela.benchmark import Benchmark
   from syntherela.metrics.single_column.statistical import ChiSquareTest
   from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy
   from syntherela.metrics.multi_table.statistical import (
       CardinalityShapeSimilarity,
   )
   from syntherela.metrics.multi_table.detection import AggregationDetection
   from xgboost import XGBClassifier

   # Create a benchmark instance
   benchmark = Benchmark(
       real_data_dir="path/to/real/data",
       synthetic_data_dir="path/to/synthetic/data",
       results_dir="path/to/results",
       benchmark_name="my_benchmark",
       single_column_metrics=[ChiSquareTest()],
       single_table_metrics=[MaximumMeanDiscrepancy()],
       multi_table_metrics=[
           CardinalityShapeSimilarity(),
           AggregationDetection(classifier_cls=XGBClassifier, random_state=42),
       ],
       datasets=["your_dataset_name"],
       methods=["your_method_name"],
   )

   # Run the benchmark
   benchmark.run()

Data Format
-----------

SyntheRela expects data in the following format:

* **Real data**: CSV files in the ``real_data_dir`` directory
* **Synthetic data**: CSV files in the ``synthetic_data_dir`` directory with matching names
* **Metadata**: A metadata file describing table relationships (optional)

Selecting Metrics
-----------------

SyntheRela provides various metrics at different levels:

Single Column Metrics
~~~~~~~~~~~~~~~~~~~~~

Evaluate individual columns:

.. code-block:: python

   from syntherela.metrics.single_column.statistical import (
       ChiSquareTest,
       KolmogorovSmirnovTest
   )
   from syntherela.metrics.single_column.distance import (
       JensenShannonDistance,
       HellingerDistance,
       WassersteinDistance
   )

   single_column_metrics = [
       ChiSquareTest(),
       KolmogorovSmirnovTest(),
       JensenShannonDistance(),
       HellingerDistance(),
       WassersteinDistance()
   ]

Single Table Metrics
~~~~~~~~~~~~~~~~~~~~

Evaluate tables as a whole:

.. code-block:: python

   from syntherela.metrics.single_table.distance import (
       MaximumMeanDiscrepancy,
       PairwiseCorrelationDifference
   )

   single_table_metrics = [
       MaximumMeanDiscrepancy(),
       PairwiseCorrelationDifference()
   ]

Multi Table Metrics
~~~~~~~~~~~~~~~~~~~

Evaluate relationships between tables:

.. code-block:: python

   from syntherela.metrics.multi_table.statistical import (
       CardinalityShapeSimilarity,
   )
   from syntherela.metrics.multi_table.detection import AggregationDetection
   from xgboost import XGBClassifier

   multi_table_metrics = [
       CardinalityShapeSimilarity(),
       AggregationDetection(classifier_cls=XGBClassifier, random_state=42),
   ]

Viewing Results
---------------

After running a benchmark, results are saved to the ``results_dir``:

* **JSON files**: Per-dataset/per-method metric outputs
* **In-memory dictionary**: Aggregated results in the benchmark object

.. code-block:: python

    # Access aggregated results
    results = benchmark.all_results

    # Access report object for a specific dataset/method
    report = benchmark.reports["your_dataset_name"]["your_method_name"]

Next Steps
----------

* Learn how to :doc:`add custom metrics <guides/adding_metrics>`
* Explore how to :doc:`replicate paper results <guides/replicating_results>`
* Check out the :doc:`API reference <api/benchmark>`
