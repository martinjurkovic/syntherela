"""Parent-child (denormalization) detection metrics for multi-table data."""

from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection.denormalized_detection import (
    DenormalizedDetection,
)


class ParentChildDetection(DenormalizedDetection):
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
        return super().prepare_data(real_data_pair, synthetic_data_pair, pair_metadata)

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
