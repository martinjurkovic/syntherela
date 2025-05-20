"""Multi-table trends reports for synthetic data.

Based on https://github.com/weipang142857/ClavaDDPM/blob/main/gen_multi_report.py
"""

import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from syntherela.metadata import Metadata
from syntherela.data import make_column_names_unique
from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends

from .table_pairs import MultiTableTrendsReport


class PairTrendsReport(BaseReport):
    """Single table column pair trends report.

    This class extends BaseReport to evaluate trends between pairs of columns
    in a single table. It uses the ColumnPairTrends property to measure how well
    the synthetic data preserves relationships between columns.

    Attributes
    ----------
    _properties : dict
        Dictionary containing the properties to evaluate, with 'Column Pair Trends'
        as the key and a ColumnPairTrends instance as the value.

    """

    def __init__(self):
        super().__init__()
        self._properties = {"Column Pair Trends": ColumnPairTrends()}


def recursive_merge(dataframes: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Merge a list of dataframes using the given keys.

    This function recursively merges a list of dataframes using the specified keys.
    It starts with the last dataframe in the list and merges each preceding dataframe
    using the corresponding key pairs.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        List of pandas DataFrames to merge.
    keys : list[str]
        List of key pairs for merging. Each pair consists of (foreign_key, primary_key).

    Returns
    -------
    pd.DataFrame
        The merged DataFrame containing data from all input dataframes.

    """
    # Start with the top table, which is the last in the list if we are going top to bottom
    result_df = dataframes[-1]
    for i in range(
        len(dataframes) - 2, -1, -1
    ):  # Iterate backwards, excluding the last already used
        fk, pk = keys[i]
        result_df = pd.merge(
            left=result_df, right=dataframes[i], how="left", left_on=fk, right_on=pk
        )
    return result_df


def get_joint_table(
    long_path: list[str], tables: dict[pd.DataFrame], dataset_meta: Metadata
) -> tuple:
    """Denormalize the tables in the long path and return the joined table and metadata.

    This function joins multiple tables along a path of relationships defined in the metadata.
    It creates a denormalized view of the data by merging tables based on primary key and
    foreign key relationships, and removes columns from intermediate tables.

    Parameters
    ----------
    long_path : list[str]
        List of table names defining the path to join.
    tables : dict[pd.DataFrame]
        Dictionary mapping table names to pandas DataFrames.
    dataset_meta : Metadata
        Metadata object containing information about the tables and their relationships.

    Returns
    -------
    tuple
        A tuple containing:
        - The joined DataFrame with columns from intermediate tables removed
        - A SingleTableMetadata object for the joined table

    """
    path_tables = [tables[table] for table in long_path]
    path_keys = []
    for i in range(1, len(long_path)):
        parent = long_path[i - 1]
        child = long_path[i]
        pk = dataset_meta.get_primary_key(parent)
        fk = dataset_meta.get_foreign_keys(parent, child)[
            0
        ]  # ClavaDDPM assumes only 1 fk between tables
        path_keys.append((fk, pk))
    long_path_joined = recursive_merge(path_tables, path_keys)

    # Remove the in-between tables
    for i in range(1, len(long_path) - 1):
        in_between_table = long_path[i]
        single_table_meta = dataset_meta.get_table_meta(in_between_table)
        for column in single_table_meta["columns"].keys():
            if column in long_path_joined:
                long_path_joined.pop(column)

    final_tables = [long_path[0], long_path[-1]]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(long_path_joined)
    for table in final_tables:
        single_table_meta = dataset_meta.get_table_meta(table)
        for column, info in single_table_meta["columns"].items():
            if column in long_path_joined.columns:
                metadata.update_column(column, **info)

    return long_path_joined, metadata


def evaluate_long_path(
    real_joined: pd.DataFrame,
    syn_joined: pd.DataFrame,
    metadata: SingleTableMetadata,
    top_table_cols: list[str],
    bottom_table_cols: list[str],
    top_table: str,
    bottom_table: str,
    verbose: bool = True,
) -> dict:
    """Evaluate trends between columns in tables joined along a path of foreign keys.

    This function evaluates the correlation and contingency similarity between columns
    from the two tables joing along a path of foreign keys.

    Parameters
    ----------
    real_joined : pd.DataFrame
        The joined real data.
    syn_joined : pd.DataFrame
        The joined synthetic data.
    metadata : SingleTableMetadata
        Metadata for the joined table.
    top_table_cols : list[str]
        List of column names from the top table.
    bottom_table_cols : list[str]
        List of column names from the bottom table.
    top_table : str
        Name of the top table.
    bottom_table : str
        Name of the bottom table.
    verbose : bool, default=True
        Whether to print verbose output.

    Returns
    -------
    dict
        Dictionary mapping column pairs to trend scores.

    """
    quality = PairTrendsReport()
    quality.generate(real_joined, syn_joined, metadata.to_dict(), verbose=verbose)

    column_pair_quality = quality.get_details("Column Pair Trends")
    if "Error" in column_pair_quality.columns:
        errors = column_pair_quality["Error"]
        error_types = {str(e).split(":")[0] for e in errors if str(e) != "None"}
        warnings.warn(
            f"Found the following error types in the column pair trends: {error_types}"
        )
        mask = errors == errors  # Select rows with errors (not None)
        column_pair_quality.loc[mask.values, "Score"] = 0
        # set scores
    top_table_cols = set(top_table_cols)
    bottom_table_cols = set(bottom_table_cols)

    res = {}

    for _, row in column_pair_quality.iterrows():
        col_1 = row["Column 1"]
        col_2 = row["Column 2"]

        if (
            col_1 in top_table_cols
            and col_2 in bottom_table_cols
            or col_1 in bottom_table_cols
            and col_2 in top_table_cols
        ):
            res[f"{top_table} - {bottom_table} : {col_1} {col_2}"] = row["Score"]
    return res


def find_paths_with_length_greater_than_one(metadata: Metadata) -> list[str]:
    """Find paths in the database schema with length greater than one.

    This function uses depth-first search to find all paths in the database schema
    that contain at least two edges (three tables).

    Parameters
    ----------
    metadata : Metadata
        Metadata object containing information about the tables and their relationships.

    Returns
    -------
    list[str]
        List of paths with length greater than one, where each path is a list of table names.

    """
    # Build adjacency list while skipping edges that start with None
    graph = defaultdict(list)
    for relationship in metadata.relationships:
        parent = relationship["parent_table_name"]
        child = relationship["child_table_name"]
        if parent is not None:
            graph[parent].append(child)

    # This will store all paths with at least two edges
    results = []

    # Helper function to perform DFS
    def dfs(node, path):
        if len(path) > 2:  # path contains at least two edges
            results.append(path[:])  # copy of the current path
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()

    # Start DFS from each node that has children, making sure we're not modifying the graph
    starting_nodes = list(graph.keys())
    for node in starting_nodes:
        dfs(node, [node])

    return results


def get_long_range(
    real_tables: dict[pd.DataFrame],
    syn_tables: dict[pd.DataFrame],
    dataset_meta: Metadata,
    verbose: bool = True,
) -> dict:
    """Evaluate trends between columns in tables connected by long paths.

    This function identifies paths in the database schema with length greater than one
    and evaluates the similarity of trends between columns in tables at the ends of these paths.
    It compares how well the synthetic data preserves relationships between distant tables.

    Parameters
    ----------
    real_tables : dict[pd.DataFrame]
        Dictionary mapping table names to real data DataFrames.
    syn_tables : dict[pd.DataFrame]
        Dictionary mapping table names to synthetic data DataFrames.
    dataset_meta : Metadata
        Metadata object containing information about the tables and their relationships.
    verbose : bool, default=True
        Whether to print verbose output during evaluation.

    Returns
    -------
    dict
        Dictionary mapping hop counts to dictionaries of column pair scores.
        Each inner dictionary maps column pair identifiers to trend similarity scores.

    """
    long_paths = find_paths_with_length_greater_than_one(dataset_meta)
    res = {}

    tables_real, tables_syn, metadata = make_column_names_unique(
        deepcopy(real_tables),
        deepcopy(syn_tables),
        deepcopy(dataset_meta),
        validate=True,
    )
    for long_path in long_paths:
        hop = len(long_path) - 1
        real_joined, table_meta = get_joint_table(long_path, tables_real, metadata)
        syn_joined_1, _ = get_joint_table(long_path, tables_syn, metadata)
        top_table = long_path[0]
        bottom_table = long_path[-1]
        top_table_cols = list(metadata.tables[top_table].columns.keys())
        bottom_table_cols = list(metadata.tables[bottom_table].columns.keys())

        scores = evaluate_long_path(
            real_joined,
            syn_joined_1,
            table_meta,
            top_table_cols,
            bottom_table_cols,
            top_table,
            bottom_table,
            verbose=verbose,
        )
        if hop not in res:
            res[hop] = {}
        res[hop].update(scores)

    return res


def get_avg_long_range_scores(res: dict) -> tuple:
    """Calculate average scores and standard errors for long-range trends.

    This function computes the average score and standard error for each hop distance
    in the long-range trend results.

    Parameters
    ----------
    res : dict
        Dictionary mapping hop counts to dictionaries of column pair scores.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - avg_scores: Dictionary mapping hop counts to average scores
        - scores_se: Dictionary mapping hop counts to standard errors of scores

    """
    avg_scores = {}
    scores_se = {}
    for hop, scores in res.items():
        if len(scores) == 0:
            continue
        scores = np.array(list(scores.values()))
        avg_scores[hop] = scores.mean()
        scores_se[hop] = scores.std() / np.sqrt(len(scores))
    return avg_scores, scores_se


def multi_table_trends(
    tables: dict[pd.DataFrame],
    syn_tables: dict[pd.DataFrame],
    metadata: Metadata,
    verbose: bool = True,
) -> dict:
    """Evaluate trends between columns across tables.

    This function evaluates how well synthetic data preserves relationships between
    columns in different tables. It calculates scores for both direct relationships
    (one hop) and indirect relationships (multiple hops) between tables.

    Parameters
    ----------
    tables : dict[pd.DataFrame]
        Dictionary mapping table names to real data DataFrames.
    syn_tables : dict[pd.DataFrame]
        Dictionary mapping table names to synthetic data DataFrames.
    metadata : Metadata
        Metadata object containing information about the tables and their relationships.
    verbose : bool, default=True
        Whether to print verbose output during evaluation.

    Returns
    -------
    dict
        Dictionary containing evaluation results with the following keys:
        - hop_relation: Dictionary mapping hop counts to dictionaries of column pair scores
        - avg_scores: Dictionary mapping hop counts to average scores
        - scores_se: Dictionary mapping hop counts to standard errors of scores
        - all_avg_score: Overall average score across all hops
        - cardinality: Average cardinality score

    """
    for table in tables.keys():
        syn_tables[table] = syn_tables[table][tables[table].columns]

    hop_relation = get_long_range(tables, syn_tables, metadata, verbose=verbose)

    multi_report = MultiTableTrendsReport()
    multi_report.generate(tables, syn_tables, metadata.to_dict(), verbose)

    one_hop = multi_report.get_details("Intertable Trends").dropna(subset=["Score"])
    one_hop_dict = {}
    for _, row in one_hop.iterrows():
        one_hop_dict[
            f"{row['Parent Table']} - {row['Child Table']} : {row['Column 1']} {row['Column 2']}"
        ] = row["Score"]

    hop_relation[1] = one_hop_dict

    avg_scores, scores_se = get_avg_long_range_scores(hop_relation)

    # avg scores for all hops:
    all_avg_score = 0
    num_scores = 0
    for hop, score in hop_relation.items():
        all_avg_score += np.sum(list(score.values()))
        num_scores += len(score)

    all_avg_score /= num_scores

    if verbose:
        print("Long Range Scores:", avg_scores)
        print("All avg scores: ", all_avg_score)

    result = {}
    result["hop_relation"] = hop_relation
    result["avg_scores"] = avg_scores
    result["scores_se"] = scores_se
    result["all_avg_score"] = all_avg_score
    result["cardinality"] = multi_report.get_details("Cardinality")[
        "Score"
    ].values.mean()
    return result
