from sdmetrics.base import BaseMetric
import sdv

class StatisticalBaseMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DistanceBaseMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def bootstrap_95_conf_int(real_data):
        """ Compute the 95% confidence interval of the metric
        using the bootstrap method. 
        """
        raise NotImplementedError()

class DetectionBaseMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def binomial_test(real_data, synthetic_data):
        """ Compute the p-value of the metric using the binomial test. """
        raise NotImplementedError()