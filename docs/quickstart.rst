Quick Start Guide
=================

This guide will help you get started with SyntheRela.

Basic Usage
-----------

Here's a simple example of how to use SyntheRela to benchmark synthetic data:

.. code-block:: python

   from syntherela import Benchmark
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

   from syntherela.metrics.multi_table import MultiTableMetric
   
   multi_table_metrics = [
       MultiTableMetric()
   ]

Viewing Results
---------------

After running a benchmark, results are saved to the ``results_dir``:

* **CSV files**: Detailed metric scores
* **Visualizations**: Plots comparing real and synthetic data
* **Report**: Summary report in HTML format

.. code-block:: python

   # Access the report
   report = benchmark.report
   
   # Get metric results
   results = report.get_results()

Next Steps
----------

* Learn how to :doc:`add custom metrics <guides/adding_metrics>`
* Explore how to :doc:`replicate paper results <guides/replicating_results>`
* Check out the :doc:`API reference <api/benchmark>`
