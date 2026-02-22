"""Parent-child (denormalization) detection metrics for multi-table data."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from syntherela.data import denormalize_tables, make_column_names_unique
from syntherela.metadata import Metadata, drop_ids
from syntherela.metrics.base import DetectionBaseMetric
from syntherela.utils import CustomHyperTransformer


class ParentChildDetection(DetectionBaseMetric):
    """Detection metric for parent-child relationships in multi-table datasets.

    This class implements a denormalization based detection metric that uses a
    classifier to distinguish between denormalization real and synthetic data
    across parent-child table pairs.

    Parameters
    ----------
    classifier_cls : class
        The classifier class to be used.
    classifier_args : dict, default={}
        Arguments to be passed to the classifier.
    random_state : int, optional
        Random state for reproducibility.
    folds : int, default=5
        Number of folds for cross-validation.
    **kwargs
        Additional keyword arguments to pass to the parent class.

    Attributes:
    ----------
    name : str
        Name of the metric.
    classifiers : list
        List to store trained classifiers.
    models : list
        List to store trained models.

    """

    @staticmethod
    def is_applicable(metadata: Metadata, table1: str, table2: str):
        """Check if the tables are applicable for this metric.

        This method checks if both tables contain at least one column that is
        not an ID and if the tables have a relationship with each other.

        Parameters
        ----------
        metadata : Metadata
            Metadata object containing information about the tables.
        table1 : str
            Name of the first table.
        table2 : str
            Name of the second table.

        Returns:
        -------
        bool
            True if the metric is applicable to the tables, False otherwise.

        """
        nonid1 = False
        table_metadata = metadata.tables[table1].to_dict()
        for column_name in table_metadata["columns"].keys():
            if table_metadata["columns"][column_name]["sdtype"] != "id":
                nonid1 = True
                break
        nonid2 = False
        table_metadata = metadata.tables[table2].to_dict()
        for column_name in table_metadata["columns"].keys():
            if table_metadata["columns"][column_name]["sdtype"] != "id":
                nonid2 = True
                break
        return nonid1 and nonid2

    def prepare_data(
        self,
        real_data,
        synthetic_data,
        metadata,
        parent_table,
        child_table,
        pair_metadata,
    ):
        """Prepare the data for the classifier by denormalizing PC table pairs.

        Parameters
        ----------
        real_data : dict
            Dictionary mapping table names to real data DataFrames.
        synthetic_data : dict
            Dictionary mapping table names to synthetic data DataFrames.
        metadata : Metadata
            Metadata object containing information about the tables.
        parent_table : str
            Name of the parent table.
        child_table : str
            Name of the child table.
        pair_metadata : Metadata
            Metadata object for the parent-child table pair.

        Returns:
        -------
        tuple
            A tuple containing:
            - X: The combined data with transformed features.
            - y: The labels for the real and synthetic data.

        """
        real_data_unique, synthetic_data_unique, metadata_unique = (
            make_column_names_unique(
                {
                    parent_table: real_data[parent_table],
                    child_table: real_data[child_table],
                },
                {
                    parent_table: synthetic_data[parent_table],
                    child_table: synthetic_data[child_table],
                },
                deepcopy(metadata),
                validate=False,
            )
        )
        denormalized_real_data = denormalize_tables(
            real_data_unique, metadata_unique
        )
        denormalized_synthetic_data = denormalize_tables(
            synthetic_data_unique, metadata_unique
        )
        for table in metadata_unique.get_tables():
            table_metadata = metadata_unique.tables[table].to_dict()
            if table == parent_table:
                parent_id = metadata_unique.get_primary_key(table)
                real_ids = denormalized_real_data[parent_id]
                synthetic_ids = denormalized_synthetic_data[parent_id]
            denormalized_real_data = drop_ids(
                denormalized_real_data, table_metadata
            )
            denormalized_synthetic_data = drop_ids(
                denormalized_synthetic_data, table_metadata
            )

            n = min(
                denormalized_real_data.shape[0],
                denormalized_synthetic_data.shape[0]
            )
            idx_real = np.random.choice(
                denormalized_real_data.index, n, replace=False
            )
            idx_synthetic = np.random.choice(
                denormalized_synthetic_data.index, n, replace=False
            )
            real_data = denormalized_real_data.loc[idx_real].reset_index(
                drop=True
            )
            synthetic_data = denormalized_synthetic_data.loc[
                idx_synthetic].reset_index(drop=True)
            real_ids = real_ids.loc[idx_real].reset_index(drop=True)
            synthetic_ids = synthetic_ids.loc[idx_synthetic].reset_index(
                drop=True
            )

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
                unique_synthetic_ids,
                len(unique_synthetic_ids) // 2,
                replace=False
            )
            mask_train_real = real_ids.isin(ids_train_real).values
            mask_train_syn = synthetic_ids.isin(ids_train_synthetic).values

            X_train_real = transformed_real_data[mask_train_real]
            X_train_synthetic = transformed_synthetic_data[mask_train_syn]
            X_test_real = transformed_real_data[~mask_train_real]
            X_test_synthetic = transformed_synthetic_data[~mask_train_syn]

            X_train = pd.concat([X_train_real, X_train_synthetic])
            X_test = pd.concat([X_test_real, X_test_synthetic])
            y_train = np.hstack(
                [np.ones(len(X_train_real)),
                 np.zeros(len(X_train_synthetic))]
            )
            y_test = np.hstack(
                [np.ones(len(X_test_real)),
                 np.zeros(len(X_test_synthetic))]
            )
        return X_train, X_test, y_train, y_test

    def run(
        self, real_data: dict, synthetic_data: dict, metadata: Metadata,
        **kwargs
    ):
        """Run the parent-child detection metric on all PC relationships.

        Parameters
        ----------
        real_data : dict
            Dictionary mapping table names to real data DataFrames.
        synthetic_data : dict
            Dictionary mapping table names to synthetic data DataFrames.
        metadata : Metadata
            Metadata object containing information about the tables.
        **kwargs
            Additional keyword arguments.

        Returns:
        -------
        dict
            Dictionary mapping relationship identifiers to metric results.

        """
        results = {}
        for relationship in metadata.relationships:
            child_table = relationship["child_table_name"]
            child_fk = relationship["child_foreign_key"]
            parent_table = relationship["parent_table_name"]
            if not self.is_applicable(metadata, parent_table, child_table):
                continue
            pair_meta = metadata.to_dict()
            for table in metadata.get_tables():
                if table != parent_table and table != child_table:
                    pair_meta["tables"].pop(table)
            pair_meta["relationships"] = [relationship]
            pair_metadata = Metadata.load_from_dict(pair_meta)
            results[f"{parent_table}_{child_table}_{child_fk}"] = super().run(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=metadata,
                parent_table=parent_table,
                child_table=child_table,
                pair_metadata=pair_metadata,
            )
        return results

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

        Returns:
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
