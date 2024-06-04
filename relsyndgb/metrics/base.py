import re
import warnings
from copy import deepcopy
from typing import Union

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

from relsyndgb.utils import CustomHyperTransformer


class SingleColumnMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_constant(column: pd.Series):
        constant = column.nunique() == 1
        if constant:
            warnings.warn(f"Column {column.name} is constant.")
        return constant

    @staticmethod
    def is_applicable(column_type):
        raise NotImplementedError()


class SingleTableMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_applicable(metadata):
        """
        Check if the table contains at least one column that is not an id.
        """
        for column_name in metadata['columns'].keys():
            if metadata['columns'][column_name]['sdtype'] != 'id':
                return True
        return False
    

class StatisticalBaseMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def validate(data):
        raise NotImplementedError()

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """
        This method is used to compute the actual metric value between two samples.
        """
        raise NotImplementedError()
    
    def run(self, real_data, synthetic_data, **kwargs):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        self.validate(real_data)
        self.validate(synthetic_data)
        return self.compute(real_data, synthetic_data)


class DistanceBaseMetric(BaseMetric):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """
        This method is used to compute the actual metric value between two samples.
        """
        raise NotImplementedError()


    def run(self, real_data, synthetic_data, **kwargs):
        """ Compute the reference confidence intervals using bootstrap on the real data
        and compute the matric value on real vs synthetic data."""
        reference_mean, reference_variance, reference_standard_ci = self.bootstrap_reference_standard_conf_int(real_data, alpha=self.alpha, **kwargs)
        bootstrap_mean, bootstrap_se = self.bootstrap_metric_estimate(real_data, synthetic_data, **kwargs)
        value = self.compute(real_data, synthetic_data, **kwargs)
        return {'value': value, 
                'reference_mean': reference_mean,
                'reference_variance' : reference_variance,
                'reference_ci': reference_standard_ci, 
                'bootstrap_mean': bootstrap_mean, 
                'bootstrap_se': bootstrap_se}


    def boostrap_metric_values(self, data1, data2, m=100, random_state=None, **kwargs):
        # get random_state from kwargs
        if random_state is None:
            random_state = 0
        values = []
        for i in range(m):
            sample1 = data1.sample(frac=1, replace = True, random_state=random_state+i)
            sample2 = data2.sample(frac=1, replace = True, random_state=random_state+i+1)
            # compute the metric
            val = self.compute(sample1, sample2, **kwargs)
            values.append(val)
        return values


    def bootstrap_metric_estimate(self, real_data, synthetic_data, m=1000, **kwargs):
        """ Compute the bootstrap mean and standard error estimates.
        """
        values = self.boostrap_metric_values(real_data, synthetic_data, m=m, **kwargs)
        return np.mean(values), np.std(values) / np.sqrt(m)
    
    def bootstrap_reference_standard_conf_int(self, real_data, m=1000, alpha=0.05, **kwargs):
        """ Compute the standard confidence interval of the metric
            on the original data using the bootstrap method.
        """
        values = self.boostrap_metric_values(real_data, real_data, m=m, **kwargs)
        m = len(values)
        mean = np.mean(values)
        bias_adjusted_variance = np.sqrt((1/(m-1)) * np.sum((values - mean)**2))

        if self.goal == Goal.MAXIMIZE:
            z_score = norm.ppf(alpha/2)
            conf_int = (mean - z_score * np.sqrt(bias_adjusted_variance), self.max_value)
        elif self.goal == Goal.MINIMIZE:
            z_score = norm.ppf(1-alpha)
            conf_int = (0, mean + z_score * np.sqrt(bias_adjusted_variance))
        else:
            z_score = norm.ppf(1-alpha/2)
            conf_int = (mean - z_score * np.sqrt(bias_adjusted_variance), mean + z_score * np.sqrt(bias_adjusted_variance))
        
        return mean, bias_adjusted_variance, conf_int


class DetectionBaseMetric(BaseMetric):
    def __init__(self, classifier_cls, classifier_args = {}, random_state = None, folds=10, m=10, **kwargs):
        super().__init__(**kwargs)
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
        self.folds = folds
        self.m = m
        self.classifiers = []
        self.name = f"{type(self).__name__}-{classifier_cls.__name__}"


    def prepare_data(self, real_data: Union[pd.DataFrame, pd.Series], synthetic_data : Union[pd.DataFrame, pd.Series], **kwargs):
        if isinstance(real_data, pd.DataFrame):
            assert real_data.columns.equals(synthetic_data.columns), "Columns of real and synthetic data do not match"
        
        # sample the same number of rows from the real and synthetic data
        n = min(len(real_data), len(synthetic_data))
        real_data = real_data.sample(n, random_state=self.random_state)
        synthetic_data = synthetic_data.sample(n, random_state=self.random_state + 1 if self.random_state else None)

        ht = CustomHyperTransformer()
        combined_data = pd.concat([real_data, synthetic_data])
        ht.fit(combined_data)
        transformed_real_data = ht.transform(real_data.copy())
        transformed_synthetic_data = ht.transform(synthetic_data.copy())
        X = pd.concat([transformed_real_data, transformed_synthetic_data])
        y = np.hstack([
            np.ones(len(transformed_real_data)), np.zeros(len(transformed_synthetic_data))
        ])
        # replace infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # drop constant columns
        if X.shape[1] > 1:
            X = X.loc[:, X.apply(lambda x: x.nunique() > 1)]
        return X, y
    

    def stratified_kfold(self, X, y, save_models=False):
        scores = []
        # Shuffle the data
        np.random.seed(self.random_state)
        idx = np.random.permutation(len(y))
        X = X.iloc[idx]
        y = y[idx]
        kf = StratifiedKFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            np.random.seed(self.random_state + i)
            model = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('clf', self.classifier_cls(**self.classifier_args))
            ])
            model.fit(X.iloc[train_index], y[train_index])
            probs = model.predict_proba(X.iloc[test_index])
            y_pred = probs.argmax(axis=1)
            scores.extend(list((y[test_index] == y_pred).astype(int)))
            if save_models:
                self.classifiers.append(deepcopy(model['clf']))
        return scores
    

    def compute(self, real_data, synthetic_data, metadata, **kwargs):
        X, y = self.prepare_data(real_data, synthetic_data, metadata=metadata, **kwargs)
        # save the data for feature importance methods
        self.X = X
        self.y = y
        return self.stratified_kfold(X, y, save_models=True)
    
    @staticmethod
    def bootstrap_sample(real_data, random_state=None, metadata=None):
        return real_data.sample(frac=1, replace=True, random_state=random_state)
    
    
    def baseline(self, real_data, metadata, m=1000, **kwargs):
        bootstrap_scores = []
        for i in range(m):
            sample1 = self.bootstrap_sample(real_data, random_state=i, metadata=metadata)
            sample2 = self.bootstrap_sample(real_data, random_state=i+1, metadata=metadata)
            X, y = self.prepare_data(sample1, sample2, metadata=metadata, **kwargs)
            scores = self.stratified_kfold(X, y)
            bootstrap_accuracy = np.mean(scores)
            bootstrap_scores.append(bootstrap_accuracy)
        return np.mean(bootstrap_scores), np.std(bootstrap_scores) / np.sqrt(m)


    @staticmethod
    def binomial_test(x, n, p=0.5, alternative='greater'):
        """ Compute the p-value of the metric using the binomial test. """
        test = binomtest(x, n, p, alternative=alternative)
        return test.statistic, test.pvalue
    

    def run(self, real_data, synthetic_data, metadata, **kwargs):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        scores = self.compute(real_data, synthetic_data, metadata=metadata, **kwargs)
        _, bin_test_p_val = self.binomial_test(sum(scores), len(scores), p=0.5, alternative='greater')
        _, copying_p_val = self.binomial_test(sum(scores), len(scores), p=0.5, alternative='less')
        standard_error = np.std(scores) / np.sqrt(len(scores))
        return { "accuracy": np.mean(scores), "SE": standard_error, 
                "bin_test_p_val" : np.round(bin_test_p_val, decimals=16),
                "copying_p_val": np.round(copying_p_val, decimals=16)}
    

    def feature_importance(self, combine_categorical=False):
        if not len(self.classifiers):
            raise ValueError('No classifiers have been trained.')
        if not hasattr(self.classifiers[0], 'feature_importances_'):
            raise ValueError('The classifier does not have a feature_importances_ attribute.')
            
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
                if not re.search('_[0-9]+$', feature):
                    continue
                feature_name = '_'.join(feature.split('_')[:-1])
                if feature_name not in feature_names:
                    feature_names[feature_name] = []
                feature_names[feature_name].append(feature)
            for feature_name, feature_group in feature_names.items():
                if len(feature_group) > 1:
                    features[feature_name] = np.concatenate([features[f] for f in feature_group])
                    for f in feature_group:
                        features.pop(f)

        return dict(sorted(features.items(), key=lambda x: np.mean(x[1]), reverse=True))
    
    
    def plot_feature_importance(self, metadata, ax=None, combine_categorical=False):
        features = self.feature_importance(combine_categorical=combine_categorical)

        def find_column_type(feature_name, column_info):
            for column, values in column_info.items():
                if values['sdtype'] == 'id':
                    continue
                if column in feature_name:
                    return values['sdtype']
            return None

        def get_feature_type(feature_name, metadata):
            if '_counts' in feature_name or '_mean' in feature_name or '_sum' in feature_name:
                return 'aggregate'
            
            feature_type = None
            if isinstance(metadata, dict):
                return find_column_type(feature_name, metadata['columns'])
            else:
                for table_data in metadata.to_dict()['tables'].values():
                    feature_type = find_column_type(feature_name, table_data['columns'])
                    if feature_type is not None:
                        break
            return feature_type

        colors = {
                    'aggregate': '#d7191c',
                    'numerical': '#fdae61',
                    'datetime': '#e3d36b',
                    'boolean': '#abd9e9',
                    'categorical': '#2c7bb6',
                }
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        for i, (feature, importance) in enumerate(features.items()):
            feature_type = get_feature_type(feature, metadata)
            y = len(features) - i - 1
            scatter = ax.scatter(importance, np.ones(len(importance)) * y, s=20, alpha=0.6, c=colors[feature_type], label=feature_type.upper())
            color = scatter.get_facecolor()[0]
            
            se = np.std(importance) / np.sqrt(len(importance))
            ax.errorbar(np.mean(importance), y, xerr=se * 1.96, c=color, capsize=3, ls='None') 
            ax.scatter(np.mean(importance), y, s=120, marker='v', color=color)

        xlim = ax.get_xlim()
        ax.set_xlim(0, xlim[1])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(list(features.keys())[::-1])
        ax.set_xlabel('Feature importance')
        labels = ax.get_legend_handles_labels()
        unique_labels = {l:h for h,l in zip(*labels)}
        labels_handles = [*zip(*unique_labels.items())]
        legend = labels_handles[::-1]
        ax.legend(*legend)
    

    def partial_dependence(self):
        raise NotImplementedError()
    