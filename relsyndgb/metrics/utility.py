from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sdmetrics.base import BaseMetric

from relsyndgb.metadata import drop_ids
from relsyndgb.utils import CustomHyperTransformer


# TODO: move from base and add feature importance
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

    def compute(self, X_train, y_train, X_test, y_test, random_state=None, **kwargs):
        np.random.seed(random_state)
        model = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('clf', self.classifier_cls(**self.classifier_args))
            ])
        model.fit(X_train, y_train)

        # if classifier is a regressor, compute the mean squared error, else accuracy
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
            score = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
        else:
            # calculate RMSE
            y_pred = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))

        def plot(log=False):
            # draw perfect line
            plt.plot(y_test, y_test, label='perfect', color='r', linestyle='dashed', alpha=0.5)
            plt.scatter(y_test, y_pred, label='predicted', alpha=0.2)
            plt.xlabel('True')
            plt.ylabel('Predicted')
            if log:
                plt.yscale('log')
            plt.legend()
            plt.show()
        return score
    
    def get_target_table(self, data, target, metadata):
        target_table, target_column = target
        X, y = data[target_table].drop(columns=target_column), data[target_table][target_column]
        X = drop_ids(X, metadata.to_dict()['tables'][target_table])
        return X, y


    def run(self, real_data, synthetic_data, metadata, test_data, **kwargs):
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

        X_real, ht_real = self.prepare_data(X_real)
        X_test_real, _ = self.prepare_data(X_test, ht=ht_real)
        X_synthetic, ht_syn = self.prepare_data(X_synthetic)
        X_test_synthetic, _ = self.prepare_data(X_test, ht=ht_syn)

        # save the data for feature importance methods
        self.X_real = X_real
        self.y_real = y_real
        self.X_synthetic = X_synthetic
        self.y_synthetic = y_synthetic
        self.X_test_real = X_test_real
        self.X_test_synthetic = X_test_synthetic
        self.y_test = y_test

        scores_real = self.compute(X_real, y_real, X_test_real, y_test, random_state=self.random_state)
        scores_synthetic = self.compute(X_synthetic, y_synthetic, X_test_synthetic, y_test, random_state=self.random_state + 1 if self.random_state else None)
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
            "synthetic_score": scores_synthetic,
            "baseline_score": baseline_score,
            "difference": difference}

        
        