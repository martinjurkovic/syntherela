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

DF = pd.DataFrame


class PairTrendsReport(BaseReport):
    """Single table column pair trends report."""

    def __init__(self):
        super().__init__()
        self._properties = {"Column Pair Trends": ColumnPairTrends()}


def recursive_merge(dataframes: list[DF], keys: list[str]) -> DF:
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
    long_path: list[str], tables: dict[DF], dataset_meta: Metadata
) -> tuple:
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
    real_joined: DF,
    syn_joined: DF,
    metadata: SingleTableMetadata,
    top_table_cols: list[str],
    bottom_table_cols: list[str],
    top_table: str,
    bottom_table: str,
    verbose: bool = True,
) -> dict:
    quality = PairTrendsReport()
    quality.generate(real_joined, syn_joined, metadata.to_dict(), verbose=verbose)

    column_pair_quality = quality.get_details("Column Pair Trends")

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
    real_tables: dict[DF],
    syn_tables: dict[DF],
    dataset_meta: Metadata,
    verbose: bool = True,
) -> dict:
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
    tables: dict[DF],
    syn_tables: dict[DF],
    metadata: Metadata,
    verbose: bool = True,
) -> dict:
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
