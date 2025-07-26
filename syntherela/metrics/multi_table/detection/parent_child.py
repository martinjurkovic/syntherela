"""Parent-child (denormalization) detection metrics for multi-table data."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from syntherela.metadata import Metadata, drop_ids
from syntherela.utils import CustomHyperTransformer
from syntherela.metrics.base import DetectionBaseMetric
from syntherela.data import denormalize_tables, make_column_names_unique


class ParentChildDetection(DetectionBaseMetric):
    """Detection metric for parent-child relationships in multi-table datasets.

    This class implements a denormalization based detection metric that uses a classifier
    to distinguish between denormalization real and synthetic data across parent-child table pairs.

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

    Attributes
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

        This method checks if both tables contain at least one column that is not an ID
        and if the tables have a relationship with each other.

        Parameters
        ----------
        metadata : Metadata
            Metadata object containing information about the tables.
        table1 : str
            Name of the first table.
        table2 : str
            Name of the second table.

        Returns
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
        """Prepare the data for the classifier by denormalizing the parent-child table pairs.

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

        Returns
        -------
        tuple
            A tuple containing:
            - X: The combined data with transformed features.
            - y: The labels for the real and synthetic data.

        """
        real_data_pair = {
            parent_table: real_data[parent_table],
            child_table: real_data[child_table],
        }
        synthetic_data_pair = {
            parent_table: synthetic_data[parent_table],
            child_table: synthetic_data[child_table],
        }
        return super().prepare_data(
            real_data_pair,
            synthetic_data_pair,
            pair_metadata,
            parent_table,
        )

    def run(self, real_data: dict, synthetic_data: dict, metadata: Metadata, **kwargs):
        """Run the parent-child detection metric on all parent-child relationships.

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

        Returns
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
