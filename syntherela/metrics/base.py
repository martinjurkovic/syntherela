"""Base classes for synthetic data evaluation metrics.

This module provides abstract base classes that define the common interface
and functionality for all metrics used in synthetic data evaluation.
"""

import re
import warnings
from typing import Union
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, norm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sdmetrics.goal import Goal
from sdmetrics.base import BaseMetric
# FIXME: We should implement our own BaseMetric class or
# we should be consistent with the sdmetrics API (run vs. compute)

from syntherela.utils import CustomHyperTransformer


class SingleColumnMetric(BaseMetric):
    """Base class for single column metrics.

    This class provides a foundation for metrics that evaluate the quality
    of synthetic data at the column level.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    """

    def __init__(self, **kwargs):
        """Initialize the SingleColumnMetric.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the parent class.

        """
        super().__init__(**kwargs)

    @staticmethod
    def is_constant(column: pd.Series):
        """Check if the column is constant.

        Parameters
        ----------
        column : pd.Series
            The column to check.

        Returns
        -------
        bool
            True if the column has only one unique value, False otherwise.

        """
        constant = column.nunique() == 1
        if constant:
            warnings.warn(f"Column {column.name} is constant.")
        return constant

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

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError()


class SingleTableMetric(BaseMetric):
    """Base class for single table metrics.

    This class provides a foundation for metrics that evaluate the quality
    of synthetic data at the table level.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    """

    def __init__(self, **kwargs):
        """Initialize the SingleTableMetric.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the parent class.

        """
        super().__init__(**kwargs)

    @staticmethod
    def is_applicable(metadata):
        """Check if the table contains at least one column that is not an id.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary for the table.

        Returns
        -------
        bool
            True if the table has at least one non-id column, False otherwise.

        """
        for column_name in metadata["columns"].keys():
            if metadata["columns"][column_name]["sdtype"] != "id":
                return True
        return False


class MultiTableMetric(BaseMetric):
    """Base class for multi table metrics.

    This class provides a foundation for metrics that evaluate the quality
    of synthetic data across multiple tables.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    """

    def __init__(self, **kwargs):
        """Initialize the MultiTableMetric.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the parent class.

        """
        super().__init__(**kwargs)


class StatisticalBaseMetric(BaseMetric):
    """Base class for statistical metrics.

    This class provides a foundation for metrics that use statistical tests
    to evaluate the quality of synthetic data.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to the parent class.

    """

    def __init__(self, **kwargs):
        """Initialize the StatisticalBaseMetric.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the parent class.

        """
        super().__init__(**kwargs)

    @staticmethod
    def validate(data):
        """Validate the input data.

        Parameters
        ----------
        data
            The data to validate.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError()

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """Compute the metric value between two samples.

        Parameters
        ----------
        real_data
            The values from the real dataset.
        synthetic_data
            The values from the synthetic dataset.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Union[float, tuple[float]]
            Metric output or outputs.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError()

    def run(self, real_data, synthetic_data, **kwargs):
        """Compute this metric.

        Parameters
        ----------
        real_data:
            The values from the real dataset.
        synthetic_data:
            The values from the synthetic dataset.

        Returns
        -------
            Union[float, tuple[float]]:
                Metric output or outputs.

        """
        self.validate(real_data)
        self.validate(synthetic_data)
        return self.compute(real_data, synthetic_data)


class DistanceBaseMetric(BaseMetric):
    """Base class for distance-based metrics.

    Attributes
    ----------
        alpha (float): Significance level for confidence intervals.

    Methods
    -------
        compute(real_data, synthetic_data, **kwargs):
            Compute the metric value between two samples. Must be implemented by subclasses.
        run(real_data, synthetic_data, **kwargs):
            Compute the reference and actual metric values.
        boostrap_metric_values(data1, data2, m=100, random_state=None, **kwargs):
            Compute the metric values for m bootstrap samples.
        bootstrap_metric_estimate(real_data, synthetic_data, m=1000, **kwargs):
            Compute the bootstrap mean and standard error estimates.
        bootstrap_reference_standard_conf_int(real_data, m=1000, alpha=0.05, **kwargs):
            Compute the standard CI on the original data using bootstrapping.

    """

    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """Compute the metric value between two samples."""
        raise NotImplementedError()

    def run(self, real_data, synthetic_data, **kwargs):
        """Compute the reference and actual metric values."""
        reference_mean, reference_variance, reference_standard_ci = (
            self.bootstrap_reference_standard_conf_int(
                real_data, alpha=self.alpha, **kwargs
            )
        )
        bootstrap_mean, bootstrap_se = self.bootstrap_metric_estimate(
            real_data, synthetic_data, **kwargs
        )
        value = self.compute(real_data, synthetic_data, **kwargs)
        return {
            "value": value,
            "reference_mean": reference_mean,
            "reference_variance": reference_variance,
            "reference_ci": reference_standard_ci,
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_se": bootstrap_se,
        }

    def boostrap_metric_values(self, data1, data2, m=100, random_state=None, **kwargs):
        """Compute the metric values for m bootstrap samples."""
        # get random_state from kwargs
        if random_state is None:
            random_state = 0
        values = []
        for i in range(m):
            sample1 = data1.sample(frac=1, replace=True, random_state=random_state + i)
            sample2 = data2.sample(
                frac=1, replace=True, random_state=random_state + i + 1
            )
            # compute the metric
            val = self.compute(sample1, sample2, **kwargs)
            values.append(val)
        return values

    def bootstrap_metric_estimate(self, real_data, synthetic_data, m=1000, **kwargs):
        """Compute the bootstrap mean and standard error estimates."""
        values = self.boostrap_metric_values(real_data, synthetic_data, m=m, **kwargs)
        return np.mean(values), np.std(values) / np.sqrt(m)

    def bootstrap_reference_standard_conf_int(
        self, real_data, m=1000, alpha=0.05, **kwargs
    ):
        """Compute the standard CI on the original data using bootstrapping."""
        values = self.boostrap_metric_values(real_data, real_data, m=m, **kwargs)
        m = len(values)
        mean = np.mean(values)
        bias_adjusted_variance = np.sqrt((1 / (m - 1)) * np.sum((values - mean) ** 2))

        if self.goal == Goal.MAXIMIZE:
            z_score = norm.ppf(alpha / 2)
            conf_int = (
                mean - z_score * np.sqrt(bias_adjusted_variance),
                self.max_value,
            )
        elif self.goal == Goal.MINIMIZE:
            z_score = norm.ppf(1 - alpha)
            conf_int = (0, mean + z_score * np.sqrt(bias_adjusted_variance))
        else:
            z_score = norm.ppf(1 - alpha / 2)
            conf_int = (
                mean - z_score * np.sqrt(bias_adjusted_variance),
                mean + z_score * np.sqrt(bias_adjusted_variance),
            )

        return mean, bias_adjusted_variance, conf_int


class DetectionBaseMetric(BaseMetric):
    """C2ST Base class.

    DetectionBaseMetric extends the BaseMetric class to provide methods for evaluating the
    performance of a classifier in distinguishing between real and synthetic data.
    It includes methods for preparing data, performing stratified k-fold cross-validation,
    computing the C2ST metric, generating bootstrap samples, estimating baseline performance,
    computing p-values using the binomial test, and plotting feature importance and partial dependence.

    Attributes
    ----------
    classifier_cls : class
        The classifier class to be used.
    classifier_args : dict
        Arguments to be passed to the classifier.
    random_state : int, optional
    folds : int
        Number of folds for cross-validation.
    classifiers : list
        List to store trained classifiers.
    models : list
        List to store trained models.
    name : str
        Name of the metric.

    Methods
    -------
    prepare_data(real_data, synthetic_data, **kwargs)
        Prepare the data for the classifier.
    stratified_kfold(X, y, save_models=False)
        Perform stratified k-fold cross-validation.
    compute(real_data, synthetic_data, metadata, **kwargs)
        Compute the C2ST metric.
    bootstrap_sample(real_data, random_state=None, metadata=None)
        Generate a bootstrap sample from the real data.
    baseline(real_data, metadata, m=1000, **kwargs)
        Estimate the metric using bootstrapping.
    binomial_test(x, n, p=0.5, alternative="greater")
        Compute the p-value of the metric using the binomial test.
    run(real_data, synthetic_data, metadata, **kwargs)
        Compute the C2ST metric.
    feature_importance(combine_categorical=False, combine_datetime=False)
        Return the feature importance scores for trained classifiers.
    plot_feature_importance(metadata, ax=None, combine_categorical=False, combine_datetime=False, lab_fontsize=30, fontsize=23)
        Plot the feature importance of the discriminator.
    plot_partial_dependence(feature, lab_fontsize=30, seed=0)
        Plot partial dependence for a given feature.

    """

    def __init__(
        self,
        classifier_cls,
        classifier_args={},
        random_state=None,
        folds=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
        self.folds = folds
        self.classifiers = []
        self.models = []
        self.name = f"{type(self).__name__}-{classifier_cls.__name__}"

    def prepare_data(
        self,
        real_data: Union[pd.DataFrame, pd.Series],
        synthetic_data: Union[pd.DataFrame, pd.Series],
        **kwargs,
    ):
        """Prepare the data for the classifier.

        Parameters
        ----------
        real_data:
            The values from the real dataset.
        synthetic_data:
            The values from the synthetic dataset.
        **kwargs:
            Additional keyword arguments.

        Returns
        -------
        X: pd.DataFrame
            The combined data with transformed features.
        y: np.ndarray
            The labels for the real and synthetic data.

        """
        if isinstance(real_data, pd.DataFrame):
            assert real_data.columns.equals(synthetic_data.columns), (
                "Columns of real and synthetic data do not match"
            )

        # sample the same number of rows from the real and synthetic data
        n = min(len(real_data), len(synthetic_data))
        real_data = real_data.sample(n, random_state=self.random_state)
        synthetic_data = synthetic_data.sample(
            n, random_state=self.random_state + 1 if self.random_state else None
        )

        ht = CustomHyperTransformer()
        combined_data = pd.concat([real_data, synthetic_data])
        ht.fit(combined_data)
        transformed_real_data = ht.transform(real_data.copy())
        transformed_synthetic_data = ht.transform(synthetic_data.copy())
        X = pd.concat([transformed_real_data, transformed_synthetic_data])
        y = np.hstack(
            [
                np.ones(len(transformed_real_data)),
                np.zeros(len(transformed_synthetic_data)),
            ]
        )
        # replace infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # drop constant columns
        if X.shape[1] > 1:
            X = X.loc[:, X.apply(lambda x: x.nunique() > 1)]
        return X, y

    def stratified_kfold(self, X, y, save_models=False):
        """Perform stratified k-fold cross-validation."""
        scores = []
        # Shuffle the data
        np.random.seed(self.random_state)
        idx = np.random.permutation(len(y))
        X = X.iloc[idx]
        y = y[idx]
        kf = StratifiedKFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            if self.random_state:
                np.random.seed(self.random_state + i)
            else:
                np.random.seed(i)
            model = Pipeline(
                [
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                    ("clf", self.classifier_cls(**self.classifier_args)),
                ]
            )
            model.fit(X.iloc[train_index], y[train_index])
            probs = model.predict_proba(X.iloc[test_index])
            y_pred = probs.argmax(axis=1)
            scores.extend(list((y[test_index] == y_pred).astype(int)))
            if save_models:
                self.classifiers.append(deepcopy(model["clf"]))
                self.models.append(model)
        return scores

    def compute(self, real_data, synthetic_data, metadata, **kwargs):
        """Compute the C2ST metric.

        Parameters
        ----------
        real_data:
            The values from the real dataset.
        synthetic_data:
            The values from the synthetic dataset.
        metadata:
            Metadata containing information about the tables / table / column.

        Returns
        -------
        dict:
            Metric output.

        """
        X, y = self.prepare_data(real_data, synthetic_data, metadata=metadata, **kwargs)
        # save the data for feature importance methods
        self.X = X
        self.y = y
        return self.stratified_kfold(X, y, save_models=True)

    @staticmethod
    def bootstrap_sample(real_data, random_state=None, metadata=None):
        """Generate a bootstrap sample from the real data."""
        return real_data.sample(frac=1, replace=True, random_state=random_state)

    def baseline(self, real_data, metadata, m=1000, **kwargs):
        """Estimate the metric using bootstrapping.

        Compute the baseline performance using bootstrapping and stratified k-fold cross-validation.

        Parameters
        ----------
        real_data: dict[str, pd.DataFrame]
            The real dataset to be used for bootstrapping.
        metadata: dict | Metadata
            Metadata information about the dataset.
        m: int
            The number of bootstrap samples to generate. Default is 1000.
        **kwargs:
            Additional keyword arguments to be passed to the prepare_data method.

        Returns
        -------
        tuple: A tuple containing the mean and standard error of the bootstrap accuracies.

        """
        bootstrap_scores = []
        for i in range(m):
            sample1 = self.bootstrap_sample(
                real_data, random_state=i, metadata=metadata
            )
            sample2 = self.bootstrap_sample(
                real_data, random_state=i + 1, metadata=metadata
            )
            X, y = self.prepare_data(sample1, sample2, metadata=metadata, **kwargs)
            scores = self.stratified_kfold(X, y)
            bootstrap_accuracy = np.mean(scores)
            bootstrap_scores.append(bootstrap_accuracy)
        return np.mean(bootstrap_scores), np.std(bootstrap_scores) / np.sqrt(m)

    @staticmethod
    def binomial_test(x, n, p=0.5, alternative="greater"):
        """Compute the p-value of the metric using the binomial test."""
        test = binomtest(x, n, p, alternative=alternative)
        return test.statistic, test.pvalue

    def run(self, real_data, synthetic_data, metadata, **kwargs):
        """Compute the C2ST metric.

        Parameters
        ----------
        real_data:
            The values from the real dataset.
        synthetic_data:
            The values from the synthetic dataset.
        metadata:
            Metadata containing information about the tables / table / column.

        Returns
        -------
            dict:
                Metric output.

        """
        scores = self.compute(real_data, synthetic_data, metadata=metadata, **kwargs)
        _, bin_test_p_val = self.binomial_test(
            sum(scores), len(scores), p=0.5, alternative="greater"
        )
        _, copying_p_val = self.binomial_test(
            sum(scores), len(scores), p=0.5, alternative="less"
        )
        standard_error = np.std(scores) / np.sqrt(len(scores))
        return {
            "accuracy": np.mean(scores),
            "SE": standard_error,
            "bin_test_p_val": np.round(bin_test_p_val, decimals=16),
            "copying_p_val": np.round(copying_p_val, decimals=16),
        }

    def feature_importance(self, combine_categorical=False, combine_datetime=False):
        """Return the feature importance scores for trained classifiers.

        Parameters
        ----------
        combine_categorical: bool
            If True, combine one-hot encoded categorical features into a single feature.
        combine_datetime: bool:
            If True, combine datetime features (Year, Month, Day, Hour, Minute, Second) into a single feature.

        Returns
        -------
        dict: A dictionary where keys are feature names and values are arrays of feature importance scores, sorted by the mean importance score in descending order.

        Raises
        ------
        ValueError: If no classifiers have been trained or if the classifier does not have a feature_importances_ attribute.

        """
        if not len(self.classifiers):
            raise ValueError("No classifiers have been trained.")
        if not hasattr(self.classifiers[0], "feature_importances_"):
            raise ValueError(
                "The classifier does not have a feature_importances_ attribute."
            )

        features = dict()
        feature_names = self.X.columns
        for model in self.classifiers:
            for feature, importance in zip(feature_names, model.feature_importances_):
                if feature not in features:
                    features[feature] = []
                features[feature].append(importance)

        features = {k: np.array(v) for k, v in features.items()}
        if combine_categorical:
            feature_names = dict()
            for feature in features.keys():
                # check if the feature is one-hot encoded
                if not re.search("_[0-9]+$", feature):
                    continue
                feature_name = "_".join(feature.split("_")[:-1])
                if feature_name not in feature_names:
                    feature_names[feature_name] = []
                feature_names[feature_name].append(feature)
            for feature_name, feature_group in feature_names.items():
                if len(feature_group) > 1:
                    features[feature_name] = np.concatenate(
                        [features[f] for f in feature_group]
                    )
                    for f in feature_group:
                        features.pop(f)

        if combine_datetime:
            feature_names = dict()
            for feature in features.keys():
                # check if the feature is one-hot encoded
                if not (
                    feature.endswith("_Year")
                    or feature.endswith("_Month")
                    or feature.endswith("_Day")
                    or feature.endswith("_Hour")
                    or feature.endswith("_Minute")
                    or feature.endswith("_Second")
                ):
                    continue
                feature_name = "_".join(feature.split("_")[:-1])
                if feature_name not in feature_names:
                    feature_names[feature_name] = []
                feature_names[feature_name].append(feature)
            for feature_name, feature_group in feature_names.items():
                if len(feature_group) > 1:
                    features[feature_name] = np.concatenate(
                        [features[f] for f in feature_group]
                    )
                    for f in feature_group:
                        features.pop(f)

        return dict(sorted(features.items(), key=lambda x: np.mean(x[1]), reverse=True))

    def plot_feature_importance(
        self,
        metadata,
        ax=None,
        combine_categorical=False,
        combine_datetime=False,
        lab_fontsize=30,
        fontsize=23,
    ):
        """Plot the feature importance of the discriminator.

        Parameters
        ----------
        metadata : dict or object
            Metadata containing information about the columns and their types.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        combine_categorical : bool, optional
            If True, combine categorical features. Default is False.
        combine_datetime : bool, optional
            If True, combine datetime features. Default is False.
        lab_fontsize : int, optional
            Font size for the x-axis label. Default is 30.
        fontsize : int, optional
            Font size for the y-axis tick labels. Default is 23.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.

        """
        features = self.feature_importance(
            combine_categorical=combine_categorical, combine_datetime=combine_datetime
        )

        def prettyify_feature_name(feature_name):
            split_name = feature_name.split("_")
            if len(split_name) > 1:
                return " ".join(
                    [
                        (
                            word.capitalize().replace("Nunique", "\#Unique")
                            if "id" not in word
                            else ""
                        )
                        for word in split_name
                    ]
                )
            return feature_name[0].upper() + feature_name[1:]

        def find_column_type(feature_name, column_info):
            for column, values in column_info.items():
                if values["sdtype"] == "id":
                    continue
                if column == feature_name:
                    return values["sdtype"]
                elif not combine_categorical and feature_name in column:
                    return values["sdtype"]
            return None

        def get_feature_type(feature_name, metadata):
            if (
                "_counts" in feature_name
                or "_mean" in feature_name
                or "_sum" in feature_name
                or "_nunique" in feature_name
            ):
                return "aggregate"

            feature_type = None
            if isinstance(metadata, dict):
                return find_column_type(feature_name, metadata["columns"])
            else:
                for table_data in metadata.to_dict()["tables"].values():
                    feature_type = find_column_type(feature_name, table_data["columns"])
                    if feature_type is not None:
                        break
            return feature_type

        colors = {
            "aggregate": "#d7191c",
            "numerical": "#fdae61",
            "datetime": "#e3d36b",
            "boolean": "#abd9e9",
            "categorical": "#2c7bb6",
        }

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        for i, (feature, importance) in enumerate(features.items()):
            feature_type = get_feature_type(feature, metadata)
            y = len(features) - i - 1
            scatter = ax.scatter(
                importance,
                np.ones(len(importance)) * y,
                s=20,
                alpha=0.6,
                c=colors[feature_type],
                label=feature_type.upper(),
            )
            color = scatter.get_facecolor()[0]

            se = np.std(importance) / np.sqrt(len(importance))
            ax.errorbar(
                np.mean(importance), y, xerr=se * 1.96, c=color, capsize=3, ls="None"
            )
            ax.scatter(np.mean(importance), y, s=120, marker="v", color=color)

        xlim = ax.get_xlim()
        ax.set_xlim(0, xlim[1])
        ax.set_yticks(range(len(features)))
        pretty_feature_names = [
            prettyify_feature_name(feature) for feature in features.keys()
        ][::-1]
        ax.set_yticklabels(pretty_feature_names)
        ax.set_xlabel("Feature importance", fontsize=lab_fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 2)
        ax.tick_params(axis="y", labelsize=fontsize)
        labels = ax.get_legend_handles_labels()
        unique_labels = {label: h for h, label in zip(*labels)}
        labels_handles = [*zip(*unique_labels.items())]
        legend = labels_handles[::-1]
        ax.legend(
            *legend,
            fontsize="x-small",
            loc="lower right",
            title="Feature type",
            markerscale=2,
        )

        return ax

    def plot_partial_dependence(self, feature, lab_fontsize=30, seed=0):
        """Plot partial dependence for a given feature.

        Compute the partial dependence for each model trained during cross-validation
        and plot the average partial dependence.

        Parameters
        ----------
        feature : str
            The feature for which to plot the partial dependence.
        lab_fontsize : int, optional, default=30
            Font size for the labels in the plot.
        seed : int, optional, default=0
            Random seed for reproducibility.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.

        """
        from matplotlib import rc

        rc("font", **{"family": "serif", "serif": ["Times"], "size": lab_fontsize})
        rc("text", usetex=True)
        from sklearn.inspection import PartialDependenceDisplay

        # TODO: move these functions to some utility module
        def prettyify_feature_name(feature_name):
            split_name = feature_name.split("_")
            if len(split_name) > 1:
                return " ".join(
                    [
                        (
                            word.capitalize().replace("Nunique", "\#Unique")
                            if "id" not in word
                            else ""
                        )
                        for word in split_name
                    ]
                )
            return feature_name[0].upper() + feature_name[1:]

        def get_average_pds(feature, seed=0, num_ice=30, subsample_avg=0.5):
            with plt.ioff():
                ys = []
                disps_ind = []
                for i, model in enumerate(self.models):
                    disp_ind = PartialDependenceDisplay.from_estimator(
                        model,
                        self.X.sample(num_ice, random_state=seed + i),
                        [feature],
                        kind="individual",
                        response_method="predict_proba",
                    )
                    np.random.seed(seed + i)
                    disp = PartialDependenceDisplay.from_estimator(
                        model,
                        self.X,
                        [feature],
                        kind="average",
                        response_method="predict_proba",
                        subsample=subsample_avg,
                        percentiles=(0, 1),
                    )
                    disps_ind.append(disp_ind)
                    y = disp.axes_[0][0].lines[0].get_ydata()
                    if i == 0:
                        ys.append(y)
                        x = disp.axes_[0][0].lines[0].get_xdata()

                    elif i > 0 and y.shape == ys[0].shape:
                        ys.append(y)
                    plt.clf()
            plt.close("all")
            return x, np.array(ys), disps_ind

        x, ys, disps = get_average_pds(feature, seed=seed)

        fig, ax = plt.subplots(figsize=(8, 6))
        disps[0].plot(ax=ax)
        for i in range(1, len(disps)):
            disps[i].plot(ax=disps[i - 1].axes_)

        ax = disps[0].axes_[0][0]
        y_mean = ys.mean(axis=0)
        y_se = ys.std(axis=0) / np.sqrt(ys.shape[0])

        ax.plot(x, y_mean, color="C0", label="Individual CEs")
        ax.plot(x, y_mean, color="C1", label="Average")
        ax.fill_between(
            x, y_mean - y_se, y_mean + y_se, alpha=0.4, color="C1", zorder=1, label="SE"
        )

        if all([x_.is_integer() for x_ in x]):
            ax.set_xticks(x)
            ax.set_xticklabels(x.astype(int))
        ax.set_xlabel(prettyify_feature_name(feature), fontsize=lab_fontsize)
        ax.set_ylabel("Partial dependence", fontsize=lab_fontsize)
        ax.legend(fontsize="xx-small", loc="lower left")

        return ax


def prepare_classifier_data(real_data, synthetic_data, **kwargs):
    """Prepare the data for the classifier.

    Parameters
    ----------
    real_data
        The values from the real dataset.
    synthetic_data
        The values from the synthetic dataset.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    X: pd.DataFrame
        The combined data with transformed features.
    y: np.ndarray
        The labels for the real and synthetic data.

    """
