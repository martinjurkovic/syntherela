
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from relsyndgb.metadata import Metadata
from relsyndgb.data import load_tables, remove_sdv_columns

from relsyndgb.visualisations.utils import get_bins

sns.set_theme()
# set colorblind color palette
sns.set_palette("colorblind")


def visualize_marginals(real_data, synthetic_data, metadata):
    for table in metadata.get_tables():
        table_meta = metadata.get_table_meta(table)
        num_non_id_columns = len([column for column, column_info in table_meta['columns'].items() if column_info['sdtype'] != 'id'])
        
        # round num_non_id_columns to the next multiple of 3
        if num_non_id_columns >= 3:
            num_non_id_columns = num_non_id_columns + (3 - num_non_id_columns % 3) if num_non_id_columns % 3 != 0 else num_non_id_columns
        fig, axes = plt.subplots(max(num_non_id_columns // 3, 1), 3, figsize=(15, 5 * (max(num_non_id_columns // 3, 1))))
        i = 0
        for column, column_info in table_meta['columns'].items():
            if column_info['sdtype'] == 'id':
                continue
            
            if num_non_id_columns <= 3:
                ax = axes[i]
            else:
                ax = axes[i // 3, i % 3]
            data = pd.DataFrame(pd.concat([real_data[table][column], synthetic_data[table][column]], axis=0))
            data['Kind'] = ['Real'] * len(real_data[table]) + ['Synthetic'] * len(synthetic_data[table])
            if column_info['sdtype'] == 'categorical' or column_info['sdtype'] == 'boolean':
                data = data.astype('object')
                data.fillna('missing', inplace=True)
                sns.histplot(data=data, x=column, hue='Kind', multiple='dodge', stat='density', common_norm=False, legend=True, ax=ax)
                # rotate x-axis labels
                if len(data[column].unique()) > 16:
                    ax.tick_params("x", labelrotation=90)
            elif column_info['sdtype'] == 'numerical' or column_info['sdtype'] == 'datetime':
                data = data[data[column].notnull()]
                sns.kdeplot(data=data, x=column, hue='Kind', common_norm=False, fill=False, legend=True, ax=ax)
            ax.set_title(f'{table}.{column}')
            fig.tight_layout()
            fig.suptitle = f'{table}'
            i +=1
        if num_non_id_columns < 3:
            num_non_id_columns = 3
        for j in range(i, num_non_id_columns):
            if num_non_id_columns == 3:
                fig.delaxes(axes[j])
            else:
                fig.delaxes(axes[j // 3, j % 3])
    plt.show()

def visualize_bivariate_distributions(real_data, synthetic_data, metadata):
    for table in metadata.get_tables():
        table_meta = metadata.get_table_meta(table)
        non_id_columns = [column for column, column_info in table_meta['columns'].items() if column_info['sdtype'] != 'id']
        pairs = [(non_id_columns[i], non_id_columns[j]) for i in range(len(non_id_columns)) for j in range(i + 1, len(non_id_columns))]
        if len(pairs) == 0:
            continue
        
        fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 5 * len(pairs)))
        for i, pair in enumerate(pairs):
            if len(pairs) == 1:
                ax1 = axes[0]
                ax2 = axes[1]
            else:
                ax1 = axes[i, 0]
                ax2 = axes[i, 1]
            data_real = pd.DataFrame({pair[0]: real_data[table][pair[0]], pair[1]: real_data[table][pair[1]]})
            data_synthetic = pd.DataFrame({pair[0]: synthetic_data[table][pair[0]], pair[1]: synthetic_data[table][pair[1]]})
            binsx = get_bins(data_real[pair[0]])
            binsy = get_bins(data_real[pair[1]])
            sns.histplot(data=data_real, x=pair[0], y=pair[1], ax=ax1, bins=(binsx, binsy))
            ax1.set_title(f'Real')
            sns.histplot(data=data_synthetic, x=pair[0], y=pair[1], ax=ax2, bins=(binsx, binsy))
            ax2.set_title(f'Synthetic')
            if type(binsx) == int and binsx > 16 or (type(binsx) == np.ndarray and len(binsx) > 16):
                ax1.tick_params("x", labelrotation=90)
                ax2.tick_params("x", labelrotation=90)
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
        fig.tight_layout()

    plt.show()

def visualize_parent_child_bivariates(real_data, synthetic_data, metadata):
    for table in metadata.get_tables():
        pairs = []
        table_meta = metadata.get_table_meta(table)
        non_id_columns = [column for column, column_info in table_meta['columns'].items() if column_info['sdtype'] != 'id']
        for parent_table in metadata.get_parents(table):
            parent_table_meta = metadata.get_table_meta(parent_table)
            for column in parent_table_meta['columns']:
                if parent_table_meta['columns'][column]['sdtype'] == 'id':
                    continue
                for child_column in non_id_columns:
                    pairs.append(((parent_table, column), child_column))
        if len(pairs) == 0:
            continue

        fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 5 * len(pairs)))
        for i, pair in enumerate(pairs):
            if len(pairs) == 1:
                ax1 = axes[0]
                ax2 = axes[1]
            else:
                ax1 = axes[i, 0]
                ax2 = axes[i, 1]
            parent_table, parent_column = pair[0]
            data_real = pd.DataFrame({pair[0]: real_data[parent_table][parent_column], pair[1]: real_data[table][pair[1]]})
            data_synthetic = pd.DataFrame({pair[0]: synthetic_data[parent_table][parent_column], pair[1]: synthetic_data[table][pair[1]]})
            binsx = get_bins(data_real[pair[0]])
            binsy = get_bins(data_real[pair[1]])
            sns.histplot(data=data_real, x=pair[0], y=pair[1], ax=ax1, bins=(binsx, binsy))
            ax1.set_title(f'Real')
            ax1.set_xlabel(f'{parent_table}.{parent_column}')
            sns.histplot(data=data_synthetic, x=pair[0], y=pair[1], ax=ax2, bins=(binsx, binsy))
            ax2.set_title(f'Synthetic')
            ax2.set_xlabel(f'{parent_table}.{parent_column}')
            if type(binsx) == int and binsx > 16 or (type(binsx) == np.ndarray and len(binsx) > 16):
                ax1.tick_params("x", labelrotation=90)
                ax2.tick_params("x", labelrotation=90)
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
        fig.tight_layout()