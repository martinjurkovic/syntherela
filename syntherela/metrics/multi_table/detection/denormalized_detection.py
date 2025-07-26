"""Detection metrics for denormalized multi-table data.

These metric are only intended as a base class for other metrics (Parent-Child detection) and should not be called on its own (see https://arxiv.org/abs/2410.03411)
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from syntherela.utils import CustomHyperTransformer
from syntherela.metadata import drop_ids, Metadata
from syntherela.data import denormalize_tables, make_column_names_unique
from syntherela.metrics.base import DetectionBaseMetric


class DenormalizedDetection(DetectionBaseMetric):
    """Detection on denormalized tables.

    This metric is only intended as a base class for other metrics
    and should not be called on its own (see https://arxiv.org/abs/2410.03411)
    """

    def prepare_data(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Metadata,
        parent_table: str,
    ):
        """Denormalize the tables and prepare the data for detection."""
        real_data_unique, synthetic_data_unique, metadata_unique = (
            make_column_names_unique(
                real_data.copy(),
                synthetic_data.copy(),
                deepcopy(metadata),
                validate=False,
            )
        )
        denormalized_real_data = denormalize_tables(real_data_unique, metadata_unique)
        denormalized_synthetic_data = denormalize_tables(
            synthetic_data_unique, metadata_unique
        )
        for table in metadata_unique.get_tables():
            table_metadata = metadata_unique.tables[table].to_dict()
            if table == parent_table:
                parent_id = metadata_unique.get_primary_key(table)
                real_ids = denormalized_real_data[parent_id]
                synthetic_ids = denormalized_synthetic_data[parent_id]
            denormalized_real_data = drop_ids(denormalized_real_data, table_metadata)
            denormalized_synthetic_data = drop_ids(
                denormalized_synthetic_data, table_metadata
            )

        n = min(denormalized_real_data.shape[0], denormalized_synthetic_data.shape[0])
        idx_real = np.random.choice(denormalized_real_data.index, n, replace=False)
        idx_synthetic = np.random.choice(
            denormalized_synthetic_data.index, n, replace=False
        )
        real_data = denormalized_real_data.loc[idx_real].reset_index(drop=True)
        synthetic_data = denormalized_synthetic_data.loc[idx_synthetic].reset_index(
            drop=True
        )
        real_ids = real_ids.loc[idx_real].reset_index(drop=True)
        synthetic_ids = synthetic_ids.loc[idx_synthetic].reset_index(drop=True)

        ht = CustomHyperTransformer()
        combined_data = pd.concat([real_data, synthetic_data])
        ht.fit(combined_data)
        transformed_real_data = ht.transform(real_data.copy())
        transformed_synthetic_data = ht.transform(synthetic_data.copy())

        unique_real_ids = np.unique(real_ids)
        unique_synthetic_ids = np.unique(synthetic_ids)
        ids_train_real = np.random.choice(
            unique_real_ids, len(unique_real_ids) // 2, replace=False
        )
        ids_train_synthetic = np.random.choice(
            unique_synthetic_ids, len(unique_synthetic_ids) // 2, replace=False
        )
        mask_train_real = real_ids.isin(ids_train_real).values
        mask_train_synthetic = synthetic_ids.isin(ids_train_synthetic).values

        X_train_real = transformed_real_data[mask_train_real]
        X_train_synthetic = transformed_synthetic_data[mask_train_synthetic]
        X_test_real = transformed_real_data[~mask_train_real]
        X_test_synthetic = transformed_synthetic_data[~mask_train_synthetic]

        X_train = pd.concat([X_train_real, X_train_synthetic])
        X_test = pd.concat([X_test_real, X_test_synthetic])
        y_train = np.hstack(
            [np.ones(len(X_train_real)), np.zeros(len(X_train_synthetic))]
        )
        y_test = np.hstack([np.ones(len(X_test_real)), np.zeros(len(X_test_synthetic))])
        return X_train, X_test, y_train, y_test

    def _fit_predict(self, X_train, y_train, X_test):
        model = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", self.classifier_cls(**self.classifier_args)),
            ]
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        return probs, model

    def compute(self, real_data, synthetic_data, metadata, **kwargs):
        """Compute the PC-C2ST metric based on a parent-level split.

        Parameters
        ----------
        real_data:
            The values from the denormalized real dataset.
        synthetic_data:
            The values from the denormalized synthetic dataset.
        metadata:
            Metadata containing information about the tables / table / column.

        Returns
        -------
        dict:
            Metric output.

        """
        X_train, X_test, y_train, y_test = self.prepare_data(
            real_data, synthetic_data, metadata=metadata, **kwargs
        )
        # save the data for feature importance methods
        self.X = pd.concat([X_train, X_test])
        self.y = np.hstack([y_train, y_test])
        scores = []
        probs1, model1 = self._fit_predict(X_train, y_train, X_test)
        y_pred1 = probs1.argmax(axis=1)
        scores.extend(list((y_test == y_pred1).astype(int)))
        probs2, model2 = self._fit_predict(X_test, y_test, X_train)
        y_pred2 = probs2.argmax(axis=1)
        scores.extend(list((y_train == y_pred2).astype(int)))
        self.classifiers.append(deepcopy(model1["clf"]))
        self.classifiers.append(deepcopy(model2["clf"]))
        return scores
