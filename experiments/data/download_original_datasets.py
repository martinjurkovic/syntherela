from sdv.datasets.demo import get_available_demos, download_demo

datasets = [
    'airbnb-simplified',
]

# iterate through the dataframe
for dataset_name in datasets:
    print(f'Downloading {dataset_name}...', end=' ')
    # TODO: data/downloads or data/original
    download_demo('multi_table', dataset_name, output_folder_name=f'data/downloads/{dataset_name}')
    print('Done.')