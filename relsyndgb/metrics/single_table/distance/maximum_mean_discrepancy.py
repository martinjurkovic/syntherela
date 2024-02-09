from relsyndgb.metrics.base import DistanceBaseMetric, SingleTableMetric
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sdmetrics.goal import Goal

class MaximumMeanDiscrepancy(DistanceBaseMetric, SingleTableMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MaximumMeanDiscrepancy"
        self.goal = Goal.MINIMIZE

    @staticmethod
    def compute(original_table, sythetic_table, metadata, kernel='linear', **kwargs):
        """
        Code for MaximumMeanDiscrepancy metric modified from:
        Qian, Z., Cebere, B.-C., & van der Schaar, M. (2023). 
        Synthcity: Facilitating innovative use cases of synthetic data in different data modalities. 
        arXiv: https://arxiv.org/abs/2301.07573
        github: https://github.com/vanderschaarlab/synthcity/
        """
        orig = original_table.copy()
        synth = sythetic_table.copy()

        orig.drop(metadata['primary_key'], axis=1, inplace=True)
        synth.drop(metadata['primary_key'], axis=1, inplace=True)
        for col in orig.columns:
            if orig[col].dtype.name in ("object", "category"):
                orig.drop(col, axis=1, inplace=True)
                synth.drop(col, axis=1, inplace=True)
            elif "datetime" in str(orig[col].dtype):
                orig[col] = orig[col].astype("int64")
                synth[col] = synth[col].astype("int64")

        # standardize the values
        pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ]
        )
        if len(orig.columns) == 0:
            return -1
        orig = pipe.fit_transform(orig)
        synth = pipe.transform(synth)
        
        if kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta = orig.mean(axis=0) - synth.mean(axis=0)
            #delta = delta_df.values

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
    