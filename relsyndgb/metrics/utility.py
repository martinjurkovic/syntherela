from typing import Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score
from sdmetrics.base import BaseMetric

from relsyndgb.metadata import drop_ids
from relsyndgb.utils import CustomHyperTransformer


class MachineLearningEfficacyMetric(BaseMetric):
        
    def __init__(self, target: Tuple[str, str], classifier_cls, classifier_args = {}, random_state = None, feature_engineering_function = None, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.classifier_cls = classifier_cls
        self.classifier_args = classifier_args
        self.random_state = random_state
        self.name = f"{type(self).__name__}-{classifier_cls.__name__}"
        self.feature_engineering_function = feature_engineering_function


    def prepare_data(self, X, ht = None, **kwargs):
        if ht is None:
            ht = CustomHyperTransformer()
            ht.fit(X.copy())
        transformed_data = ht.transform(X.copy())
        # replace infinite values with NaN
        transformed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return transformed_data, ht


    def score(self, model, X, y):
        # if classifier is a regressor, compute RMSE, else AUCROC
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if probs.shape[1] > 2:
                # calculate AUCROC
                return roc_auc_score(y, probs, multi_class='ovr', average='macro')
            else:
                return roc_auc_score(y, probs[:, 1])
        else:
            # calculate RMSE
            y_pred = model.predict(X)
            return np.sqrt(mean_squared_error(y, y_pred))
        

    def compute(self, X_train, y_train, X_test, y_test, random_state=None, m=100, cv_args=None, **kwargs):
        np.random.seed(random_state)
        model = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('clf', self.classifier_cls(**self.classifier_args))
            ])
        
        if cv_args is not None:
            cv = GridSearchCV(model, **cv_args).fit(X_train, y_train)
            model.set_params(**cv.best_params_)

        model.fit(X_train, y_train)
        # bootstrap the test set
        scores = []
        for _ in range(m):
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_test_boot = X_test.iloc[indices]
            y_test_boot = y_test.iloc[indices]
            score = self.score(model, X_test_boot, y_test_boot)
            scores.append(score)

        return model, np.mean(scores), np.std(scores) / np.sqrt(m)
    

    def get_target_table(self, data, target, metadata):
        target_table, target_column = target
        X, y = data[target_table].drop(columns=target_column), data[target_table][target_column]
        X = drop_ids(X, metadata.to_dict()['tables'][target_table])
        return X, y


    def run(self, real_data, synthetic_data, metadata, test_data, cv_args=None, m=100, **kwargs):
        if self.feature_engineering_function:
            X_real, y_real = self.feature_engineering_function(real_data, metadata)
            X_synthetic, y_synthetic = self.feature_engineering_function(synthetic_data, metadata)
            X_test, y_test = self.feature_engineering_function(test_data, metadata)
        else:
            X_real, y_real = self.get_target_table(real_data, self.target, metadata)
            X_synthetic, y_synthetic = self.get_target_table(synthetic_data, self.target, metadata)
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

        self.model_real, scores_real, se_real = self.compute(X_real, y_real, X_test, y_test, cv_args=cv_args, random_state=self.random_state, m=m)
        self.model_synthetic, scores_synthetic, se_synthetic = self.compute(X_synthetic, y_synthetic, X_test, y_test, cv_args=cv_args, random_state=self.random_state + 1 if self.random_state else None, m=m)
        # compute the baseline score
        if metadata.get_table_meta(self.target[0])['columns'][self.target[1]]['sdtype'] == 'numerical':
            y_baseline = y_real.mean() * np.ones(len(y_test))
            baseline_score = np.sqrt(mean_squared_error(y_test, y_baseline))
            # positive difference means the synthetic data is better (lower RMSE)
            difference = scores_real - scores_synthetic
        else:
            # Accuracy baseline
            # y_baseline = np.ones(len(y_test)) * mode(y_real).mode
            # baseline_score = np.mean(y_test == y_baseline)
            # ROCAUC baseline
            baseline_score = 0.5
            # positive difference means the synthetic data is better (higher accuracy)
            difference = scores_synthetic - scores_real
        

        return {
            "real_score": scores_real, 
            "real_score_se": se_real,
            "synthetic_score": scores_synthetic,
            "synthetic_score_se": se_synthetic,
            "baseline_score": baseline_score,
            "difference": difference}
    

    def feature_importance(self):
        if hasattr(self.model_real['clf'], 'feature_importances_'):
            importance_real = self.model_real['clf'].feature_importances_
            importance_syn  = self.model_synthetic['clf'].feature_importances_
        elif hasattr(self.model_real['clf'], 'coef_'):
            importance_real = self.model_real['clf'].coef_
            importance_syn = self.model_synthetic['clf'].coef_
        else:
            raise NotImplementedError(f"Feature importance not supported for {type(self.model_real['clf'])}")
        order_real = np.argsort(importance_real)
        importance_real = importance_real[order_real]
        importance_syn = importance_syn[order_real]
        feature_names = self.X_real.columns[order_real]
        return importance_real, importance_syn, feature_names
    
    