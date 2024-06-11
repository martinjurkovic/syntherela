import xgboost as xgb
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection import AggregationDetection
from syntherela.data import load_tables, remove_sdv_columns

sns.set_theme()
rc('font', **{'family': 'serif', 'serif': ['Times'], 'size':30})
rc('text', usetex=True)


def reproduce_figure(tables, tables_synthetic, metadata, dataset_name, figure_name):
    # Compute the metric
    xgb_cls = xgb.XGBClassifier
    xgb_args = {'seed': 0,}

    metric = AggregationDetection(classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42)


    for table in tables.keys():
        tables_synthetic[table] = tables_synthetic[table][tables[table].columns]

    if dataset_name == 'imdb_MovieLens_v1':
        target_table = 'movies'
    elif dataset_name == 'Biodegradability_v1':
        target_table = 'molecule'

    metric.run(
        tables,
        tables_synthetic,
        metadata = metadata,
        target_table=target_table,
    )

    # Plot the feature importance

    fig, ax = plt.subplots(figsize=(7, 7))
    metric.plot_feature_importance(metadata, ax = ax, combine_categorical=True)
    plt.savefig(f'results/figures/figure5{figure_name}.png', bbox_inches='tight', dpi=600)

## FIGURE 5 (a)
dataset_name = 'Biodegradability_v1' 
method = 'GRETEL_LSTM' 

metadata = Metadata().load_from_json(f'data/original/{dataset_name}/metadata.json')

tables = load_tables(f'data/original/{dataset_name}/', metadata)
tables_synthetic = load_tables(f'data/synthetic/{dataset_name}/{method}/1/sample1', metadata)

tables, metadata = remove_sdv_columns(tables, metadata)
tables_synthetic, metadata = remove_sdv_columns(tables_synthetic, metadata, update_metadata=False)

reproduce_figure(tables, tables_synthetic, metadata, dataset_name, 'a')

## FIGURE 5 (a)
dataset_name = 'imdb_MovieLens_v1'
method = 'GRETEL_ACTGAN'

metadata = Metadata().load_from_json(f'data/original/{dataset_name}/metadata.json')

tables = load_tables(f'data/original/{dataset_name}/', metadata)
tables_synthetic = load_tables(f'data/synthetic/{dataset_name}/{method}/1/sample1', metadata)

tables, metadata = remove_sdv_columns(tables, metadata)
tables_synthetic, metadata = remove_sdv_columns(tables_synthetic, metadata, update_metadata=False)

reproduce_figure(tables, tables_synthetic, metadata, dataset_name, 'b')