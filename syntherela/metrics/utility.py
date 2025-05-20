"""Single-table machine learning efficacy (utility) metric."""

from typing import Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sdmetrics.base import BaseMetric

from syntherela.metadata import drop_ids
from syntherela.utils import CustomHyperTransformer


class MachineLearningEfficacyMetric(BaseMetric):
    """Machine Learning Efficacy metric."""

    def __init__(
        self,
        target: Tuple[str, str],
        classifier_cls,
        classifier_args={},
        random_state=None,
        feature_engineering_function=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = target
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
        self.name = f"{type(self).__name__}-{classifier_cls.__name__}"
        self.feature_engineering_function = feature_engineering_function

    def prepare_data(self, X, ht=None, **kwargs):
        """Prepare the data for training."""
        if ht is None:
            ht = CustomHyperTransformer()
            ht.fit(X.copy())
        transformed_data = ht.transform(X.copy())
        # replace infinite values with NaN
        transformed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return transformed_data, ht

    def score(self, model, X, y):
        """Compute the score of the model."""
        # if classifier is a regressor, compute RMSE, else AUCROC
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.shape[1] > 2:
                # calculate AUCROC
                return roc_auc_score(y, probs, multi_class="ovr", average="macro")
            else:
                return roc_auc_score(y, probs[:, 1])
        else:
            # calculate RMSE
            y_pred = model.predict(X)
            return -np.sqrt(mean_squared_error(y, y_pred))

    def compute(self, X_train, y_train, X_test, y_test, m=100, **kwargs):
        """Compute the ML-E metric."""
        np.random.seed(self.random_state)
        model_full_train_set = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", self.classifier_cls(**self.classifier_args)),
            ]
        )
        model_full_train_set.fit(X_train, y_train)
        score_full_train_set = self.score(model_full_train_set, X_test, y_test)

        # Bootstrap to estimate the standard error
        scores = []
        models = []
        for bootstrap_idx in range(m):
            np.random.seed(self.random_state + bootstrap_idx)
            indices = np.random.choice(len(X_train), len(X_train), replace=m > 1)
            model = Pipeline(
                [
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                    ("clf", self.classifier_cls(**self.classifier_args)),
                ]
            )

            X_train_boot = X_train.iloc[indices]
            y_train_boot = y_train.iloc[indices]

            model.fit(X_train_boot, y_train_boot)
            score = self.score(model, X_test, y_test)
            scores.append(score)
            models.append(model)

        return (
            model_full_train_set,
            models,
            scores,
            score_full_train_set,
            np.std(scores) / np.sqrt(m),
        )

    def get_target_table(self, data, target, metadata):
        """Extract the target table and column from the data."""
        target_table, target_column, _ = target
        X, y = (
            data[target_table].drop(columns=target_column),
            data[target_table][target_column],
        )
        X = drop_ids(X, metadata.to_dict()["tables"][target_table])
        return X, y

    def run(
        self,
        real_data,
        synthetic_data,
        metadata,
        test_data,
        m=100,
        feature_importance=True,
        score_real=None,
        se_real=None,
        feature_importance_real=None,
        **kwargs,
    ):
        """Compute the ML-E using train-on-sythetic test-on-real approach."""
        if self.feature_engineering_function:
            X_real, y_real = self.feature_engineering_function(real_data, metadata)
            X_synthetic, y_synthetic = self.feature_engineering_function(
                synthetic_data, metadata
            )
            X_test, y_test = self.feature_engineering_function(test_data, metadata)
        else:
            X_real, y_real = self.get_target_table(real_data, self.target, metadata)
            X_synthetic, y_synthetic = self.get_target_table(
                synthetic_data, self.target, metadata
            )
            X_test, y_test = self.get_target_table(test_data, self.target, metadata)
        X_real = X_real[X_test.columns]
        X_synthetic = X_synthetic[X_test.columns]

        X_real, ht = self.prepare_data(X_real)
        X_synthetic, _ = self.prepare_data(X_synthetic, ht=ht)
        X_test, _ = self.prepare_data(X_test, ht=ht)

        # save the data for feature importance methods
        self.X_real = X_real
        self.y_real = y_real
        self.X_synthetic = X_synthetic
        self.y_synthetic = y_synthetic
        self.X_test = X_test
        self.y_test = y_test

        compute_real = score_real is None or se_real is None

        if compute_real:
            model_real, _, _, score_real, se_real = self.compute(
                X_real, y_real, X_test, y_test, m=m
            )
        (
            model_synthetic,
            models_synthetic,
            score_array_synthetic,
            score_synthetic,
            se_synthetic,
        ) = self.compute(X_synthetic, y_synthetic, X_test, y_test, m=m)
        # compute the baseline score
        difference = score_synthetic - score_real
        importances_syn = []
        if feature_importance:
            if compute_real:
                full_feature_importance_real, _ = self.feature_importance(model_real)
            else:
                full_feature_importance_real = feature_importance_real
            full_feature_importance_syn, feature_names = self.feature_importance(
                model_synthetic
            )
            for model_synthetic in models_synthetic:
                importance_syn, feature_names = self.feature_importance(model_synthetic)
                importances_syn.append(importance_syn)
        else:
            feature_names = []
            full_feature_importance_real = []
            full_feature_importance_syn = []

        return {
            "real_score": score_real,
            "real_score_se": se_real,
            "synthetic_score": score_synthetic,
            "synthetic_score_se": se_synthetic,
            "synthetic_score_array": score_array_synthetic,
            "difference": difference,
            "importance_synthetic": importances_syn,
            "feature_names": feature_names,
            "full_feature_importance_real": full_feature_importance_real,
            "full_feature_importance_synthetic": full_feature_importance_syn,
        }

    def feature_importance(self, model):
        """Extract feature importance from the trained model."""
        if hasattr(model["clf"], "feature_importances_"):
            importance = model["clf"].feature_importances_
        elif hasattr(model["clf"], "coef_"):
            importance = model["clf"].coef_
        else:
            raise NotImplementedError(
                f"Feature importance not supported for {type(model['clf'])}"
            )

        return importance, self.X_real.columns.tolist()
