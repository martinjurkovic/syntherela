"""Maximum Mean Discrepancy (MMD) metric for single tables."""

import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sdmetrics.goal import Goal

from syntherela.metadata import drop_ids
from syntherela.metrics.base import DistanceBaseMetric, SingleTableMetric


class MaximumMeanDiscrepancy(DistanceBaseMetric, SingleTableMetric):
    """Maximum Mean Discrepancy metric for comparing single tables.

    This metric computes the Maximum Mean Discrepancy (MMD) between the distributions
    of real and synthetic data tables. It is applicable to tables containing numerical columns.

    The implementation is based on the work by Qian et al. (2023) in the Synthcity library.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    Attributes
    ----------
    name : str
        Name of the metric.
    goal : Goal
        Goal of the metric (minimize).
    min_value : float
        Minimum value of the metric (0.0).
    max_value : float
        Maximum value of the metric (infinity).

    References
    ----------
    .. [1] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
           A kernel two-sample test. Journal of Machine Learning Research, 13(1), 723-773.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MaximumMeanDiscrepancy"
        self.goal = Goal.MINIMIZE
        self.min_value = 0.0
        self.max_value = float("inf")

    @staticmethod
    def is_applicable(metadata):
        """Check if the table contains numerical column."""
        for column_name in metadata["columns"].keys():
            if metadata["columns"][column_name]["sdtype"] == "numerical":
                return True
        return False

    @staticmethod
    def compute(original_table, sythetic_table, metadata, kernel="linear", **kwargs):
        """Compute the Maximum Mean Discrepancy (MMD) between two tables.

        Code for MaximumMeanDiscrepancy metric modified from:
        Qian, Z., Cebere, B.-C., & van der Schaar, M. (2023).
        Synthcity: Facilitating innovative use cases of synthetic data in different data modalities.
        arXiv: https://arxiv.org/abs/2301.07573
        github: https://github.com/vanderschaarlab/synthcity/
        """
        orig = original_table.copy()
        synth = sythetic_table.copy()

        orig = drop_ids(orig, metadata)
        synth = drop_ids(synth, metadata)
        for col in orig.columns:
            if orig[col].dtype.name in ("object", "category"):
                orig.drop(col, axis=1, inplace=True)
                synth.drop(col, axis=1, inplace=True)
            elif "datetime" in str(orig[col].dtype):
                orig[col] = pd.to_numeric(orig[col])
                synth[col] = pd.to_numeric(synth[col])

        # standardize the values
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        combined = pd.concat([orig, synth])
        pipe.fit(combined)
        orig = pipe.transform(orig)
        synth = pipe.transform(synth)

        if kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta = orig.mean(axis=0) - synth.mean(axis=0)
            # delta = delta_df.values

            score = delta.dot(delta.T)
        elif kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(
                orig.reshape(len(original_table), -1),
                synth.reshape(len(original_table), -1),
                gamma,
            )
            YY = metrics.pairwise.rbf_kernel(
                synth.reshape(len(sythetic_table), -1),
                synth.reshape(len(sythetic_table), -1),
                gamma,
            )
            XY = metrics.pairwise.rbf_kernel(
                orig.reshape(len(original_table), -1),
                synth.reshape(len(sythetic_table), -1),
                gamma,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(
                orig.reshape(len(original_table), -1),
                orig.reshape(len(original_table), -1),
                degree,
                gamma,
                coef0,
            )
            YY = metrics.pairwise.polynomial_kernel(
                synth.numpy().reshape(len(sythetic_table), -1),
                synth.numpy().reshape(len(sythetic_table), -1),
                degree,
                gamma,
                coef0,
            )
            XY = metrics.pairwise.polynomial_kernel(
                orig.reshape(len(original_table), -1),
                synth.numpy().reshape(len(sythetic_table), -1),
                degree,
                gamma,
                coef0,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {kernel}")

        return score.astype(float)
