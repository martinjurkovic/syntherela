"""Detection metrics (C2ST) for single columns.

This module provides metrics for detecting synthetic data in single columns
by training classifiers to distinguish between real and synthetic marginal distributions.
"""

from syntherela.metrics.base import DetectionBaseMetric, SingleColumnMetric


class SingleColumnDetection(DetectionBaseMetric, SingleColumnMetric):
    """Detection metric for single columns.

    This class implements a detection metric that uses a classifier to distinguish
    between real and synthetic data at the column level. It is applicable to
    categorical, datetime, numerical, and boolean columns.

    Parameters
    ----------
    classifier_cls : class
        The classifier class to be used.
    classifier_args : dict, default={}
        Arguments to be passed to the classifier.
    random_state : int, optional
        Random state for reproducibility.
    folds : int, default=5
        Number of folds for cross-validation.
    **kwargs
        Additional keyword arguments to pass to the parent class.

    Attributes
    ----------
    name : str
        Name of the metric.
    classifiers : list
        List to store trained classifiers.
    models : list
        List to store trained models.

    """

    @staticmethod
    def is_applicable(column_type):
        """Check if the column type is applicable for this metric.

        Parameters
        ----------
        column_type : str
            The type of the column.

        Returns
        -------
        bool
            True if the metric is applicable to the column type, False otherwise.

        """
        return column_type in ["categorical", "datetime", "numerical", "boolean"]
