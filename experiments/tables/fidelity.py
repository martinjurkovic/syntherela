import os
import json

from collections import defaultdict

import numpy as np
import pandas as pd


datasets = [
    'airbnb-simplified_subsampled',
    'rossmann_subsampled',
    'walmart_subsampled',
    "Berka_subsampled",
    "f1_subsampled",
    'imdb_MovieLens_v1',
    'Biodegradability_v1',
    'CORA_v1',

]


dataset_names = {
    "airbnb-simplified_subsampled": "Airbnb",
    "Berka_subsampled": "Berka",
    "Biodegradability_v1": "Biodegradability",
    "CORA_v1": "Cora",
    "imdb_MovieLens_v1": "IMDB",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "f1_subsampled": "F1",
}

methods = [
    'MOSTLYAI',
    'RGCLD',
    'CLAVADDPM',
    'RCTGAN',
    'REALTABFORMER',
    'SDV',
    'MARE',
]

model_names = {
    'CLAVADDPM': "ClavaDDPM",
    'RGCLD': "RGCLD",
    'MOSTLYAI': "TabularARGN",
    'RCTGAN': "RCTGAN",
    'REALTABFORMER': "REALTABF.",
    'SDV': "SDV",
    'MARE': "MARE",
}

defaultmethod = 'RCTGAN'
method_order = methods
method_names = [model_names[method] for method in method_order]

def add_row(df, results, dataset, metric=None):
    new_row = pd.DataFrame.from_dict({len(df):{model_names[methods[i]]: score for i, score in enumerate(results)}}, orient='index')
    new_row['Dataset'] = dataset
    new_row['Metric'] = metric
    return pd.concat([df, new_row])

def compute(dfs, idx, col, metric, factor=100.0):
    return metric([df.loc[idx, col] * factor for df in dfs])

def bold(s, math=False):
    if math:
        split = s.split('$')
        split[1] = '\\mathbf{' + split[1] + '}'
        return '$'.join(split)
        # return "$\\mathbf{" + s.replace('$', '') + "}$"
    return "\\textbf{" + s + "}"

def underline(s, math=False):
    if math:
        split = s.split('$')
        split[1] = '\\underline{' + split[1] + '}'
        return '$'.join(split)
    return "\\underline{" + s + "}"

def multirow(s, n_rows=2):
    return "\\multirow{" +str(n_rows) +"}{*}{" + s + "}"

def estimate_uncertainty(dfs, factor=100.0):
    master_df = dfs[0] + dfs[1] + dfs[2]
    master_df['Dataset'] = dfs[0]['Dataset']
    master_df['Metric'] = dfs[0]['Metric']

    for method in method_names:
        master_df[method] = master_df[method].astype(str)

    for i, row in master_df.iterrows():
        if (row[method_names] == 'nan').all():
            continue
        if (row['Metric'].startswith('C2ST') or
            row['Metric'].startswith('JS') or
            row['Metric'].startswith('Wass')):
            order = row[method_names].astype(float).fillna(float('inf')).values.argsort()
            maximize = False
        else:
            order = row[method_names].astype(float).fillna(-float('inf')).values.argsort()[::-1]
            maximize = True
        best_method = method_names[order[0]]
        for method in method_names:
            if master_df.loc[i, method] == 'nan':
                continue
            mean = compute(dfs, i, method, np.nanmean, factor=factor)
            std = compute(dfs, i, method, np.nanstd, factor=factor) / np.sqrt(len(dfs))
            if method == best_method:
                best_mean = mean
                best_std = std

            if std == 0:
                master_df.loc[i, method] = f"${mean:.2f}$"
            elif mean.round(2) == factor:
                master_df.loc[i, method] = f"${mean:.2f}$"
            elif std.round(2) == 0.0:
                master_df.loc[i, method] = f"${mean:.2f}$" + "{\\tiny $\\pm " + f"{std:.0e}$" + "}"
            else:
                master_df.loc[i, method] = f"${mean:.2f}$" + "{\\tiny $\\pm " + f"{std:.2f}$" + "}"
        master_df.loc[i, best_method] = bold(master_df.loc[i, best_method], math=True)
        values = row[method_names].astype(float) / len(dfs) * factor
        if maximize:
            within_std = best_mean - best_std
            competitors = values.index[values > within_std].to_list()
        else:
            within_std = best_mean + best_std
            competitors = values.index[values < within_std].to_list()
        for method in competitors:
            if method == best_method:
                continue
            else:
                master_df.loc[i, method] = underline(master_df.loc[i, method], math=True)
        master_df.loc[i] = master_df.loc[i].replace('nan', '-')
    return master_df


def get_latex_table(df, factor=100.0, bold_headers=True):
    if bold_headers:
        df.columns = [bold(col) for col in df.columns]
    format = 'c' * len(df.columns)
    df_latex = df.to_latex(column_format=format, index=False)
    df_latex = df_latex.replace('nan', '')
    df_latex = df_latex.replace(f'{factor:.2f}', f'\\approx {factor:.1f}')
    df_latex = df_latex.replace("\\\\\n\\multirow", "\\\\\n\\midrule\n\\multirow")
    df_latex = df_latex.replace("e-1", "\\text{e-}1")
    df_latex = df_latex.replace("e-0", "\\text{e-}")
    rows_ = df_latex.split("\n")
    rows = []
    for i, row in enumerate(rows_):
        if row.startswith("\\multirow"):
            num_rows = row.split("{")[1].split("}")[0]
            row = row.replace(" - ", "\\multirow{" + num_rows + "}{*}{-}")
        rows.append(row)
    df_latex = "\n".join(rows)
    df_latex = df_latex.replace(" - ", "")

    return df_latex


def create_single_column_df(results, single_column_results):
    DETECTION = "C2ST \\ \\ ($\\downarrow$)"
    SHAPES = "Shapes ($\\uparrow$)"
    df = pd.DataFrame(data=[['', ''] + [np.nan] * len(methods)], columns=['Dataset', 'Metric'] + [model_names[method] for method in methods])
    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        dataset_results = single_column_results[dataset_name]
        detection_scores = []
        shapes_scores = []
        for method in method_order:
            method_results = dataset_results[method]

            if len(method_results) == 0:
                detection_score = np.nan
                shapes_score = np.nan
            else:
                detection_score = np.mean(method_results)
                # print(dataset, method)
                trend_results = results[dataset][method]['single_column_metrics']['Trends']
                shapes_score = trend_results['shapes']['mean']
            detection_scores.append(detection_score)
            shapes_scores.append(shapes_score)
        n_rows = 2
        df = add_row(df, results=detection_scores, dataset=multirow(dataset_name, n_rows=n_rows), metric=DETECTION)
        df = add_row(df, results=shapes_scores, dataset='', metric=SHAPES)

    # Drop the first row
    return df[1:]


def create_single_table_df(results, single_table_results):
    DETECTION = "C2ST \\ ($\\downarrow$)"
    PAIRS = "Pairs ($\\uparrow$)"
    df = pd.DataFrame(data=[['', ''] + [np.nan] * len(methods)], columns=['Dataset', 'Metric'] + [model_names[method] for method in methods])
    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        dataset_results = single_table_results[dataset_name]
        detection_scores = []
        trend_scores = []
        for method in method_order:
            method_results = dataset_results[method]
            if len(method_results) == 0:
                detection_score = np.nan
                trend_score = np.nan
            else:
                detection_score = np.mean(method_results)
                if 'Trends' not in results[dataset][method]['single_table_metrics']:
                    trend_score = np.nan
                else:
                    trend_results = results[dataset][method]['single_table_metrics']['Trends']
                    trend_score = trend_results['pairs']['mean']
            detection_scores.append(detection_score)
            trend_scores.append(trend_score)
        n_rows = 2
        df = add_row(df, results=detection_scores, dataset=multirow(dataset_name, n_rows=n_rows), metric=DETECTION)
        df = add_row(df, results=trend_scores, dataset='', metric=PAIRS)

    # Drop the first row
    return df[1:]


def create_multi_table_df(results, multi_table_results):
    DETECTION = "C2ST-Agg ($\\downarrow$)"
    CARDINALITY = "Cardinality ($\\uparrow$)"
    df = pd.DataFrame(data=[['', ''] + [np.nan] * len(methods)], columns=['Dataset', 'Metric'] + [model_names[method] for method in methods])
    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        dataset_results = multi_table_results[dataset_name]
        detection_scores = []
        cardinality_scores = []
        k_hops = defaultdict(list)
        default_hops = results[dataset][defaultmethod]['multi_table_metrics']['Trends']['k_hop_similarity']
        for method in method_order:
            method_results = dataset_results[method]
            if len(method_results) == 0 or method == 'baseline':
                detection_score = np.nan
                cardinality = np.nan
                for hop, hop_results in default_hops.items():
                    k_hops[hop].append(np.nan)
            else:
                detection_score = np.mean(method_results)
                trend_results = results[dataset][method]['multi_table_metrics']['Trends']
                cardinality = trend_results['cardinality']
                hop_results = trend_results['k_hop_similarity']
                for hop, hop_results in hop_results.items():
                    k_hops[hop].append(hop_results['mean'])
            detection_scores.append(detection_score)
            cardinality_scores.append(cardinality)
        n_rows = 2 + len(k_hops)
        df = add_row(df, results=detection_scores, dataset=multirow(dataset_name, n_rows=n_rows), metric=DETECTION)
        df = add_row(df, results=cardinality_scores, dataset='', metric=CARDINALITY)
        for hop, scores in k_hops.items():
            df = add_row(df, results=scores, dataset='', metric=f"{hop}-HOP ($\\uparrow$)")


    # Drop the first row
    return df[1:]


def save_latex_table(df, filename, factor=100.0):
    latex_df = get_latex_table(df, factor=factor)
    with open(filename, 'w') as f:
        f.write(latex_df)


runs = [1, 2, 3]
results = {}
multi_table_results = {}
single_table_results = {}
single_column_results = {}

for run in runs:
    results[run] = {}
    run_results = results[run]
    for dataset in datasets:
        run_results[dataset] = {}
        for method in methods:
            try:
                with open(f'results/{run}/{dataset}_{method}_{run}_sample1.json') as f:
                    run_results[dataset][method] = json.load(f)
            except FileNotFoundError:
                continue


    multi_table_results[run] = {}
    single_table_results[run] = {}
    single_column_results[run] = {}
    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        multi_table_results[run].setdefault(dataset_name, {})
        single_table_results[run].setdefault(dataset_name, {})
        single_column_results[run].setdefault(dataset_name, {})
        multi_run = multi_table_results[run]
        single_run = single_table_results[run]
        column_run = single_column_results[run]
        for method in methods:
            multi_run[dataset_name].setdefault(method, [])
            single_run[dataset_name].setdefault(method, [])
            column_run[dataset_name].setdefault(method, [])
            if method not in results[run][dataset]:
                continue
            for table, single_results in results[run][dataset][method]['single_table_metrics']['SingleTableDetection-XGBClassifier'].items():
                multi_results = results[run][dataset][method]['multi_table_metrics']['AggregationDetection-XGBClassifier']
                if table in multi_results:
                    multi_run[dataset_name][method].append(multi_results[table]['accuracy'])
                single_run[dataset_name][method].append(single_results['accuracy'])
                for column, column_results in results[run][dataset][method]['single_column_metrics']['SingleColumnDetection-XGBClassifier'][table].items():
                    column_run[dataset_name][method].append(column_results['accuracy'])

os.makedirs('results/tables', exist_ok=True)
# Single-table
dfs = []
for run in runs:
    table_results = single_table_results[run]
    df = create_single_table_df(results[run], table_results)
    dfs.append(df)

df = estimate_uncertainty(dfs)
save_latex_table(df, 'results/tables/table2.tex')

# Multi-table
dfs = []
for run in runs:
    multi_results = multi_table_results[run]
    df = create_multi_table_df(results[run], multi_results)
    dfs.append(df)

df = estimate_uncertainty(dfs)
save_latex_table(df, 'results/tables/table3.tex')

# Single-column
dfs = []
for run in runs:
    col_results = single_column_results[run]
    df = create_single_column_df(results[run], col_results)
    dfs.append(df)
df = estimate_uncertainty(dfs)
save_latex_table(df, 'results/tables/table7.tex')
