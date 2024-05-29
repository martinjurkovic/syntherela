import json
import argparse
from datetime import datetime
from functools import partial
import os

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau, weightedtau
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from relsyndgb.metadata import Metadata
from relsyndgb.data import load_tables, remove_sdv_columns
from relsyndgb.metrics.utility import MachineLearningEfficacyMetric


## DATA LOADING
def load_rossmann(method):
    dataset_name = 'rossmann_subsampled' 
    run = "1"
    metadata = Metadata().load_from_json(f'{PROJECT_PATH}/data/downloads/{dataset_name}/metadata.json')

    tables = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name}/', metadata)
    tables_synthetic = load_tables(f'{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1', metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic, metadata = remove_sdv_columns(tables_synthetic, metadata, update_metadata=False)

    tables_test = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name.split("_")[0]}/', metadata)
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)
    # split the test data
    min_date = datetime.strptime('2014-10-01', '%Y-%m-%d')
    max_date = datetime.strptime('2014-11-01', '%Y-%m-%d')
    tables_test['historical'] = tables_test['historical'][(tables_test['historical']['Date'] >= min_date) & (tables_test['historical']['Date'] < max_date)]
    
    return tables, tables_synthetic, tables_test, metadata


def load_airbnb(method):
    dataset_name = 'airbnb-simplified_subsampled'
    run = "1"
    metadata = Metadata().load_from_json(f'{PROJECT_PATH}/data/downloads/{dataset_name}/metadata.json')

    tables = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name}/', metadata)
    tables_synthetic = load_tables(f'{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1', metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)
    for table in tables:
        if 'index' in tables_synthetic[table].columns:
            tables_synthetic[table].drop(columns=['index'], inplace=True)
    tables_synthetic, metadata = remove_sdv_columns(tables_synthetic, metadata, update_metadata=False)

    tables_test = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name.split("_")[0]}/', metadata)
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)

    # select users with at most 50 sessions
    sessions_count = tables_test['sessions'].reset_index().groupby('user_id').index.count()
    eligable_users = sessions_count[sessions_count <= 50].index 
    no_sessions = tables_test['users'][~tables_test['users']['id'].isin(tables_test['sessions']['user_id'])]['id']
    eligable_users = eligable_users.union(no_sessions)
    # select users that are not in the synthetic data
    eligable_users = eligable_users[~eligable_users.isin(tables['users']['id'])]

    selected_users = np.random.choice(eligable_users, 2000, replace=False)
    tables_test['users'] = tables_test['users'][tables_test['users']['id'].isin(selected_users)]
    tables_test['sessions'] = tables_test['sessions'][tables_test['sessions']['user_id'].isin(selected_users)]

    return tables, tables_synthetic, tables_test, metadata


def load_walmart(method):
    dataset_name = 'walmart_subsampled_12'
    run = "1"
    metadata = Metadata().load_from_json(f'{PROJECT_PATH}/data/downloads/{dataset_name}/metadata.json')

    tables = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name}/', metadata)
    tables_synthetic = load_tables(f'{PROJECT_PATH}/data/synthetic/{dataset_name}/{method}/{run}/sample1', metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic, metadata = remove_sdv_columns(tables_synthetic, metadata, update_metadata=False)

    # split the test data
    tables_test = load_tables(f'{PROJECT_PATH}/data/downloads/{dataset_name.split("_")[0]}/', metadata)
    tables_test, _ = remove_sdv_columns(tables_test, metadata, update_metadata=False)

    min_date = datetime.strptime('2012-01-01', '%Y-%m-%d')
    max_date = datetime.strptime('2012-02-01', '%Y-%m-%d')
    tables_test['depts'] = tables_test['depts'][(tables_test['depts']['Date'] >= min_date) & (tables_test['depts']['Date'] < max_date)]
    tables_test['features'] = tables_test['features'][(tables_test['features']['Date'] >= min_date) & (tables_test['features']['Date'] < max_date)]
    tables_test['features'].merge(tables_test['depts'], on=['Store', 'Date']).tail()

    return tables, tables_synthetic, tables_test, metadata

## DATA PREPARATION
def process_rossmann(tables, metadata):
    df = tables['historical'].merge(tables['store'], on='Store')
    numerical_columns = [] 
    for table in metadata.get_tables():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata['columns'].items():
            if column_info['sdtype'] == 'numerical':
                numerical_columns.append(column)
            elif column_info['sdtype'] == 'id':
                if column in df.columns:
                    df.drop(columns=[column], inplace=True)

    # drop the StateHoliday column as it is constant causing problems with standardization
    df.drop(columns=['StateHoliday'], inplace=True)
    # drop the dates due to subsampling
    df.drop(columns=['Date'], inplace=True)

    df[numerical_columns] = df[numerical_columns].fillna(0)
    y = df.pop('Customers')

    # remove missing / infinite y values
    mask = y.isna() | y.isin([np.inf, -np.inf])
    X = df[~mask]
    y = y[~mask]
    return X, y


def process_airbnb(tables, metadata, categories):
    # TODO: make use of other tables
    df = tables['users'].copy()
    df.drop(columns=['country_destination', 'date_first_booking'], inplace=True)  

    numerical_columns = [] 
    categorical_columns = []
    for table in metadata.get_tables():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata['columns'].items():
            if column_info['sdtype'] == 'numerical':
                if column in df.columns:
                    numerical_columns.append(column)
            elif column_info['sdtype'] == 'categorical':
                if column in df.columns:
                    categorical_columns.append(column)
                    df[column] = pd.Categorical(df[column], categories=categories[column])

    df[numerical_columns] = df[numerical_columns].fillna(0)
    
    y = tables['users'][tables['users']['id'].isin(df['id'])]['country_destination']
    X = df.drop(columns=['id'])
    
    # convert y to binary variable determining if the user booked a trip or not
    y = y != 'NDF'
        
    return X, y


def process_walmart(tables, metadata):
    df = tables['depts'].merge(tables['stores'], on='Store').merge(tables['features'], on=['Store', 'Date'], suffixes=('', '_y'))
    df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)
    df.drop(columns=['Dept'], inplace=True)
    categorical_columns = []
    for table in metadata.get_tables():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata['columns'].items():
            if column_info['sdtype'] == 'categorical' and column in df.columns:
                categorical_columns.append(column)

    # one-hot encode the categorical columns
    df = pd.get_dummies(df, columns=categorical_columns)
    # obtain average daily sales across all departments
    df = df.groupby(['Store', 'Date']).mean().reset_index(drop=True)
    
    y = df.pop('Weekly_Sales')

    # remove missing / infinite y values
    mask = y.isna() | y.isin([np.inf, -np.inf])
    X = df[~mask]
    y = y[~mask]
    return X, y


## UTILITY
def load_dataset(dataset_name, method):
    if dataset_name == 'rossmann':
        tables, tables_synthetic, tables_test, metadata = load_rossmann(method)
        feature_engineering_function = process_rossmann
        target=('historical', 'Customers', 'regression')
    elif dataset_name == 'airbnb':
        tables, tables_synthetic, tables_test, metadata = load_airbnb(method)
        categories = {}
        for _, table in tables_test.items():
            for column in table.columns:
                if table[column].dtype.name == 'category':
                    categories[column] = table[column].cat.categories.tolist()
        feature_engineering_function = partial(process_airbnb, categories=categories)
        # feature_engineering_function = None
        target=('users', 'country_destination', 'classification')
    elif dataset_name == 'walmart':
        tables, tables_synthetic, tables_test, metadata = load_walmart(method)
        feature_engineering_function = process_walmart
        target=('depts', 'Weekly_Sales', 'regression')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return tables, tables_synthetic, tables_test, metadata, feature_engineering_function, target


methods = [
    'SDV', 
    'RCTGAN',
    'REALTABFORMER',
    'MOSTLYAI',
    'GRETEL_ACTGAN',
    'GRETEL_LSTM'
]
    
datasets = [
    'rossmann',
    'airbnb',
    'walmart',
]

classifiers = {
    'regression': {
        'xgboost': xgb.XGBRegressor,
        'linear': LinearRegression,
        'random_forest': RandomForestRegressor,
        'decision_tree': DecisionTreeRegressor,
        'knn': KNeighborsRegressor,
        'svr': SVR,
        'mlp': MLPRegressor,
    },
    'classification': {
        'xgboost': xgb.XGBClassifier,
        'linear': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'decision_tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier,
        'svc': SVC,
        'gaussian_nb': GaussianNB,
        'mlp': MLPClassifier,
    }
}

cls_args = {
    'xgboost': {'random_state': None, 'seed': None},
    'linear': {},
    'random_forest': {'random_state': None},
    'decision_tree': {'random_state': None},
    'knn': {},
    'svr': {},
    'svc': {'random_state': None, 'probability': True},
    'gaussian_nb': {},
    'mlp': {'random_state': None},
}


feature_selection_models = {
    'regression': ['xgboost'],
    'classification': ['xgboost']
}

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_PATH = os.getenv("PROJECT_PATH")
    RESULTS_PATH = os.getenv("RESULTS_PATH")
    
    args = argparse.ArgumentParser()
    args.add_argument("--dataset-name", type=str, default="rossmann", choices=datasets, help="Dataset name to run the experiment on.")
    args.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    args.add_argument("--m", type=int, default=100, help="Number of bootstrap samples.")
    args = args.parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    m = args.m

    results = {}
    results[dataset_name] = {}
    for method in methods:
        print(f"Method: {method}, Dataset: {dataset_name}")
        results[dataset_name][method] = {}
        tables, tables_synthetic, tables_test, metadata, feature_engineering_function, target = load_dataset(dataset_name, method)
        task = target[2]
        for classifier, classifier_cls in classifiers[task].items():
            classifier_args_ = cls_args[classifier]
            if 'random_state' in classifier_args_:
                classifier_args_['random_state'] = seed
            if 'seed' in classifier_args_:
                classifier_args_['seed'] = seed

            ml_metric = MachineLearningEfficacyMetric(target=target, feature_engineering_function=feature_engineering_function,
                                                    classifier_cls=classifier_cls, random_state=seed, classifier_args=classifier_args_)
            

            result = ml_metric.run(tables, tables_synthetic, metadata, tables_test, m=m, feature_importance=classifier in feature_selection_models[task])
            print(f"Classifier: {classifier} real_score: {result['real_score'] :.3f} +- {result['real_score_se']:.3f}, synthetic_score: {result['synthetic_score']:.3f} +- {result['synthetic_score_se']:.3f}")
            importances_real = result.pop('importance_real', None)
            importances_syn = result.pop('importance_synthetic', None)
            results[dataset_name][method][classifier] = result
            if classifier in feature_selection_models[task]:
                feature_importances_spearman = []
                feature_importances_tau = []
                feature_importances_weighted = []
                for i in range(m):
                    importance_real = importances_real[i]
                    importance_syn = importances_syn[i]
                    real_rank = np.argsort(importance_real)
                    synthetic_rank = np.argsort(importance_syn)
                    features_spearman_rank = spearmanr(real_rank, synthetic_rank).statistic
                    features_tau_rank = kendalltau(real_rank, synthetic_rank).statistic
                    features_weighted_rank = weightedtau(real_rank, synthetic_rank, rank=None).statistic
                    feature_importances_spearman.append(features_spearman_rank)
                    feature_importances_tau.append(features_tau_rank)   
                    feature_importances_weighted.append(features_weighted_rank)

        
        # rank the classifiers
        real_classifier_rank = list(dict(sorted(results[dataset_name][method].items(), key=lambda x: x[1]['real_score'], reverse=True)).keys())
        synthetic_classifier_rank = list(dict(sorted(results[dataset_name][method].items(), key=lambda x: x[1]['synthetic_score'], reverse=True)).keys())
        print(f"Real data classifier rank: {real_classifier_rank}")
        print(f"Synthetic data classifier rank: {synthetic_classifier_rank}")

        classifier_rank_array_spearman = []
        classifier_rank_array_kendall = []
        classifier_rank_array_weighted = []
            
        for bootstrap_index in range(m):
            real_classifier_rank_boot = list(dict(sorted(results[dataset_name][method].items(), key=lambda x: x[1]['real_score_array'][bootstrap_index], reverse = True)).keys())
            synthetic_classifier_rank_boot = list(dict(sorted(results[dataset_name][method].items(), key=lambda x: x[1]['synthetic_score_array'][bootstrap_index], reverse = True)).keys())

            classifier_rank_array_spearman.append(spearmanr(real_classifier_rank_boot, synthetic_classifier_rank_boot).statistic)
            classifier_rank_array_kendall.append(kendalltau(real_classifier_rank_boot, synthetic_classifier_rank_boot).statistic)

            indexed_real_rank_boot = np.array([len(real_classifier_rank) - real_classifier_rank_boot.index(classifier) for classifier in real_classifier_rank_boot])
            indexed_synthetic_rank_boot = np.array([len(real_classifier_rank) - real_classifier_rank_boot.index(classifier) for classifier in synthetic_classifier_rank_boot])
            classifier_rank_array_weighted.append(weightedtau(indexed_real_rank_boot, indexed_synthetic_rank_boot, rank=None).statistic)

        spearman_rank = spearmanr(list(real_classifier_rank), list(synthetic_classifier_rank))
        results[dataset_name][method]['classifier_rank'] = spearman_rank.statistic
        results[dataset_name][method]['feature_importance_spearman_mean'] = np.mean(feature_importances_spearman)
        results[dataset_name][method]['feature_importance_spearman_se'] = np.std(feature_importances_spearman) / np.sqrt(m)
        results[dataset_name][method]['feature_importance_tau_mean'] = np.mean(feature_importances_tau)
        results[dataset_name][method]['feature_importance_tau_se'] = np.std(feature_importances_tau) / np.sqrt(m)
        results[dataset_name][method]['feature_importance_weighted_mean'] = np.mean(feature_importances_weighted)
        results[dataset_name][method]['feature_importance_weighted_se'] = np.std(feature_importances_weighted) / np.sqrt(m)
        results[dataset_name][method]['spearman_mean'] = np.mean(classifier_rank_array_spearman)
        results[dataset_name][method]['spearman_se'] = np.std(classifier_rank_array_spearman) / np.sqrt(m)
        results[dataset_name][method]['kendall_mean'] = np.mean(classifier_rank_array_kendall)
        results[dataset_name][method]['kendall_se'] = np.std(classifier_rank_array_kendall) / np.sqrt(m)
        results[dataset_name][method]['weighted_mean'] = np.mean(classifier_rank_array_weighted)
        results[dataset_name][method]['weighted_se'] = np.std(classifier_rank_array_weighted) / np.sqrt(m)

        print()
        print(f"Boot spearman: {np.mean(classifier_rank_array_spearman):.3f}+-{np.std(classifier_rank_array_spearman) / np.sqrt(m):.4f}")
        print(f"Boot kendall: {np.mean(classifier_rank_array_kendall):.3f}+-{np.std(classifier_rank_array_kendall) / np.sqrt(m):.4f}")
        print(f"Boot weighted: {np.mean(classifier_rank_array_weighted):.3f}+-{np.std(classifier_rank_array_weighted) / np.sqrt(m):.4f}")
        print(f"Spearman rank: {spearman_rank.statistic}")
        print(f"Feature importance spearman: {np.mean(feature_importances_spearman) :.3f}+-{np.std(feature_importances_spearman) / np.sqrt(m):.4f}")
        print(f"Feature importance tau: {np.mean(feature_importances_tau) :.3f}+-{np.std(feature_importances_tau) / np.sqrt(m):.4f}")
        print(f"Feature importance weighted: {np.mean(feature_importances_weighted) :.3f}+-{np.std(feature_importances_weighted) / np.sqrt(m):.4f}")
        print()
            
    with open(f'{RESULTS_PATH}/mle_{dataset_name}_{seed}.json', 'w') as f:
        json.dump(results, f, indent=4)
