Examples
========

We provide example notebooks to help you get started with SyntheRela.

Evaluating the Rossmann Subsampled Dataset
------------------------------------------

The `examples/evaluate_rossmann_subsampled.ipynb <https://github.com/martinjurkovic/syntherela/blob/main/examples/evaluate_rossmann_subsampled.ipynb>`_ notebook provides a step-by-step guide to evaluating a subsampled version of the `Rossmann <https://www.kaggle.com/competitions/rossmann-store-sales>`_ dataset using various SyntheRela metrics.

It covers:

* Loading real and synthetic data
* Configuring the :class:`~syntherela.benchmark.Benchmark` with single-column, single-table, and multi-table metrics
* Running the evaluation pipeline
* Inspecting the results
