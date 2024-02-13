import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sdmetrics.goal import Goal
from sdmetrics.base import BaseMetric
from sdmetrics.utils import HyperTransformer

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
        return value, reference_ci


    # TODO: m=2 for DEBUGGING, should be m=100
    def bootstrap_conf_int(self, real_data, synthetic_data, m=2, alpha=0.05, **kwargs):
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

        'Large sample confidence regions based on sub-samples under minimal assumptions. 
        Politis, D. N.; Romano, J. P. (1994). Annals of Statistics, 22, 2031-2050.9'
        """
        values = []
        for _ in range(m):
            # draw 2 samples with replacement of size 0.5 * n
            sample1 = real_data.sample(frac=0.5, replace = True)
            sample2 = real_data.sample(frac=0.5, replace = True)
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

    @staticmethod
    def prepare_data(real_data, synthetic_data, **kwargs):
        ht = HyperTransformer()
        combined_data = pd.concat([real_data, synthetic_data])
        ht.fit(combined_data)
        transformed_real_data = ht.transform(real_data).to_numpy()
        transformed_synthetic_data = ht.transform(synthetic_data).to_numpy()
        X = np.concatenate([transformed_real_data, transformed_synthetic_data])
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
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(X, y):
            model.fit(X[train_index], y[train_index])
            probs = model.predict_proba(X[test_index])
            y_pred = probs.argmax(axis=1)
            scores.extend(list((y[test_index] == y_pred).astype(int)))
        return scores

    @staticmethod
    def binomial_test(x, n):
        """ Compute the p-value of the metric using the binomial test. """
        test = binomtest(x, n, 0.5, alternative='greater')
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
        return { "accuracy": np.mean(scores), "SE": standard_error, "bin_test_p_val" : bin_test_p_val }