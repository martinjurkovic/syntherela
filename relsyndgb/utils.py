import json

import numpy as np
import pandas as pd
from sdmetrics.utils import HyperTransformer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)
    

class CustomHyperTransformer(HyperTransformer):
    """
        CustomHyperTransformer overrides the transform method of HyperTransformer
        so that the feature names are preserved for one-hot-encoded columns.
    """
    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'{field}_{i}' for i in range(np.shape(out)[1])])
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isnull()
                data[field] = pd.to_datetime(data[field], errors='coerce')
                data[f'{field}_Year'] = data[field].dt.year
                data[f'{field}_Month'] = data[field].dt.month
                data[f'{field}_Day'] = data[field].dt.day
                data[f'{field}_Hour'] = data[field].dt.hour
                data[f'{field}_Minute'] = data[field].dt.minute
                data[f'{field}_Year'][nulls] = np.nan
                data[f'{field}_Month'][nulls] = np.nan
                data[f'{field}_Day'][nulls] = np.nan
                data[f'{field}_Hour'][nulls] = np.nan
                data[f'{field}_Minute'][nulls] = np.nan
                data = data.drop(columns=[field])

        return data