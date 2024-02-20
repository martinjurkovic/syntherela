from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def compute(real_data, synthetic_data, **kwargs):
        """
        This method is used to compute the actual metric value between two samples.
        """
        raise NotImplementedError()

    def run(self, real_data, synthetic_data, **kwargs):
        """ Compute the reference confidence intervals using bootstrap on the real data
        and compute the matric value on real vs synthetic data."""
        reference_ci = self.bootstrap_reference_conf_int(real_data, **kwargs)
        boostrap_mean, bootstrap_se = self.bootstrap_metric_estimate(real_data, synthetic_data, **kwargs)
        value = self.compute(real_data, synthetic_data, **kwargs)
        return {'value': value, 'reference_ci': reference_ci, 'bootstrap_mean': boostrap_mean, 'bootstrap_se': bootstrap_se}
    
    def boostrap_metric_values(self, data1, data2, m=100, random_state=None, **kwargs):
        # get random_state from kwargs
        if random_state is None:
            random_state = 0
        values = []
        for i in range(m):
            # draw 2 samples with replacement of size 0.5 * n
            sample1 = data1.sample(frac=1, replace = True, random_state=random_state+i)
            sample2 = data2.sample(frac=1, replace = True, random_state=random_state+i+1)
            # compute the metric
            val = self.compute(sample1, sample2, **kwargs)
            values.append(val)
        return values


    def bootstrap_metric_estimate(self, real_data, synthetic_data, m=100, **kwargs):
        """ Compute the bootstrap mean and standard error estimates.
        """
        values = self.boostrap_metric_values(real_data, synthetic_data, m=m, **kwargs)
        return np.mean(values), np.std(values) / np.sqrt(m)
        

    def bootstrap_reference_conf_int(self, real_data, m=100, alpha=0.05, **kwargs):
        """ Compute the quantile confidence interval of the metric
            on the original data using the bootstrap method.
        """
        values = self.boostrap_metric_values(real_data, real_data, m=m, **kwargs)

        if self.goal == Goal.MAXIMIZE:
            return (np.quantile(values, q=0), np.quantile(values, q=1-alpha))
        elif self.goal == Goal.MINIMIZE:
            return (np.quantile(values, alpha), np.quantile(values, 1))
        else:
            return (np.quantile(values, alpha/2), np.quantile(values, 1-alpha/2))


class DetectionBaseMetric(BaseMetric):
    def __init__(self, classifier_cls, classifier_args = {}, random_state = None, folds=10, **kwargs):
        super().__init__(**kwargs)
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
        self.folds = folds
        self.classifiers = []


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
        transformed_real_data = ht.transform(real_data)
        transformed_synthetic_data = ht.transform(synthetic_data)
        X = pd.concat([transformed_real_data, transformed_synthetic_data])
        y = np.hstack([
            np.ones(len(transformed_real_data)), np.zeros(len(transformed_synthetic_data))
        ])
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan
        return X, y
    
    def compute(self, real_data, synthetic_data, metadata, **kwargs):

        

        X, y = self.prepare_data(real_data, synthetic_data, metadata=metadata)
        scores = []
        kf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(X, y):
            model = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('clf', self.classifier_cls(**self.classifier_args))
            ])
            model.fit(X.iloc[train_index], y[train_index])
            probs = model.predict_proba(X.iloc[test_index])
            y_pred = probs.argmax(axis=1)
            scores.extend(list((y[test_index] == y_pred).astype(int)))
            model['clf'].feature_names = X.columns.to_list()
            self.classifiers.append(deepcopy(model['clf']))
        return scores

    @staticmethod
    def binomial_test(x, n, p=0.5):
        """ Compute the p-value of the metric using the binomial test. """
        test = binomtest(x, n, p, alternative='greater')
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
        _, bin_test_p_val = self.binomial_test(sum(scores), len(scores))
        standard_error = np.std(scores) / np.sqrt(len(scores))
        return { "accuracy": np.mean(scores), "SE": standard_error, "bin_test_p_val" : np.round(bin_test_p_val, decimals=16) }
    

    def feature_importance(self, combine_categorical=False):
        if not len(self.classifiers):
            raise ValueError('No classifiers have been trained.')
        if not hasattr(self.classifiers[0], 'feature_importances_'):
            raise ValueError('The classifier does not have a feature_importances_ attribute.')
            
        features = dict()
        for model in self.classifiers:
            for feature, importance in zip(model.feature_names, model.feature_importances_):
                if feature not in features:
                    features[feature] = []
                features[feature].append(importance)

        features = {k: np.array(v) for k, v in features.items()}
        if combine_categorical:
            feature_names = dict()
            for feature in features.keys():
                # keep everything before last underscore
                if '_' not in feature:
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
            ax.errorbar(np.mean(importance), y, xerr=se * 1.96, c=color, capsize=3, fmt='*') 
            ax.scatter(np.mean(importance), y, s=120, marker='*', color=color)

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
    
