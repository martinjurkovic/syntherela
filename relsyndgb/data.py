import os

import pandas as pd
from sdv.datasets.demo import get_available_demos, download_demo


def load_tables(data_path, metadata):
    tables = {}
    for file_name in os.listdir(data_path):
        if not file_name.endswith('.csv'):
            continue
        table_name = file_name.split('.')[0]
        dtypes = {}
        parse_dates = []
        for column, column_info in metadata.tables[table_name].columns.items():
            if column_info['sdtype'] == 'categorical':
                dtypes[column] = 'category'
            elif column_info['sdtype'] == 'boolean':
                dtypes[column] = 'bool'
            elif column_info['sdtype'] == 'datetime':
                parse_dates.append(column)
            # for ids and numerical values let pandas infer the type
        table = pd.read_csv(f'{data_path}/{file_name}', low_memory=False, dtype=dtypes, parse_dates=parse_dates)
        tables[table_name] = table
    return tables


def save_tables(tables, path):
    # create directory if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    for table_name, table in tables.items():
        table.to_csv(os.path.join(path, f"{table_name}.csv"), index=False)


def download_sdv_relational_datasets():
    sdv_relational_datasets = get_available_demos('multi_table')

    # iterate through the dataframe
    for dataset_name in sdv_relational_datasets.dataset_name:
        print(f'Downloading {dataset_name}...', end=' ')
        # TODO: data/downloads or data/original
        download_demo('multi_table', dataset_name, output_folder_name=f'data/downloads/{dataset_name}')
        print('Done.')