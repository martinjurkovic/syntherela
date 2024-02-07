import numpy as np
from scipy.stats import binomtest

from sdmetrics.goal import Goal
from sdmetrics.base import BaseMetric


class SingleColumnMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_applicable(column_type):
        raise NotImplementedError()
    

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def compute(real_data, synthetic_data):
        """
        This method is used to compute the actual metric value between two samples.
        """
        raise NotImplementedError()

    @staticmethod
    def binomial_test(x, n):
        """ Compute the p-value of the metric using the binomial test. """
        test = binomtest(x, n, 0.5, alternative='greater')
        return test.statistic, test.pvalue
    
    def run(self, real_data, synthetic_data):
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
        accuracy = self.compute(real_data, synthetic_data)
        n = len(real_data)
        return self.binomial_test(accuracy * n, n)