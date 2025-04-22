"""Detection metrics (C2ST) for single tables.

This module provides metrics for detecting synthetic data in single tables
by training classifiers to distinguish between real and synthetic data.
"""

from syntherela.metadata import drop_ids
from syntherela.metrics.base import DetectionBaseMetric, SingleTableMetric


class SingleTableDetection(DetectionBaseMetric, SingleTableMetric):
    """Detection metric (C2ST) for single tables.

    This class implements a detection metric that uses a classifier to distinguish
    between real and synthetic data at the table level. It prepares the data by
    removing ID columns before training the classifier.

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

    def prepare_data(self, real_data, synthetic_data, metadata, **kwargs):
        """Prepare the data for the classifier by removing ID columns.

        Parameters
        ----------
        real_data : pandas.DataFrame
            The real data table.
        synthetic_data : pandas.DataFrame
            The synthetic data table.
        metadata : dict
            Metadata dictionary for the table.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing:
            - X: The combined data with transformed features.
            - y: The labels for the real and synthetic data.

        """
        real_data = drop_ids(real_data, metadata)
        synthetic_data = drop_ids(synthetic_data, metadata)
        return super().prepare_data(real_data, synthetic_data)
