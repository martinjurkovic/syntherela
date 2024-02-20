from typing import Union

import numpy as np
import pandas as pd
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
        # TODO: bootstrap is only for uncertainty estimation, should also return SE?
        value = self.compute(real_data, synthetic_data, **kwargs)
        return {'value': value, 'reference_ci': reference_ci}


    def bootstrap_conf_int(self, real_data, synthetic_data, m=100, alpha=0.05, **kwargs):
        """ Compute the quantile confidence interval of the metric
        using the bootstrap method. 
        """
        values = []
        for _ in range(m):
            # draw 2 samples with replacement of size 0.5 * n
            sample1 = real_data.sample(frac=1, replace = True)
            sample2 = synthetic_data.sample(frac=1, replace = True)
            # compute the metric
            val = self.compute(sample1, sample2, **kwargs)
            values.append(val)

        if self.goal == Goal.MAXIMIZE:
            return (np.quantile(values, q=0), np.quantile(values, q=1-alpha))
        elif self.goal == Goal.MINIMIZE:
            return (np.quantile(values, alpha), np.quantile(values, 1))
        else:
            return (np.quantile(values, alpha/2), np.quantile(values, 1-alpha/2))
        

    def bootstrap_reference_conf_int(self, real_data, m=100, alpha=0.05, **kwargs):
        """ Compute the quantile confidence interval of the metric
        on the original data using subsampling to estimate the reference
        using the bootstrap method.
        """
        values = []
        for _ in range(m):
            sample1 = real_data.sample(frac=1, replace = True, random_state=m)
            sample2 = real_data.sample(frac=1, replace = True, random_state=m+1)
            # compute the metric
            val = self.compute(sample1, sample2, **kwargs)
            values.append(val)

        if self.goal == Goal.MAXIMIZE:
            return (np.quantile(values, q=0), np.quantile(values, q=1-alpha))
        elif self.goal == Goal.MINIMIZE:
            return (np.quantile(values, alpha), np.quantile(values, 1))
        else:
            return (np.quantile(values, alpha/2), np.quantile(values, 1-alpha/2))


class DetectionBaseMetric(BaseMetric):
    def __init__(self, classifier_cls, classifier_args = {}, random_state = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
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

        model = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('clf', self.classifier_cls(**self.classifier_args))
        ])

        X, y = self.prepare_data(real_data, synthetic_data, metadata=metadata)
        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(X, y):
            model.fit(X.iloc[train_index], y[train_index])
            probs = model.predict_proba(X.iloc[test_index])
            y_pred = probs.argmax(axis=1)
            scores.extend(list((y[test_index] == y_pred).astype(int)))
            model['clf'].feature_names = X.columns.to_list()
            self.classifiers.append(model['clf'])
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
    

    def feature_importance(self):
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

        return dict(sorted(features.items(), key=lambda x: np.mean(x[1]), reverse=True))
    
