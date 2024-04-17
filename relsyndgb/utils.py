import json

import numpy as np
import pandas as pd
from sdmetrics.utils import HyperTransformer
from sklearn.preprocessing import OneHotEncoder

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
    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder()
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                dates = pd.to_datetime(data[field][~nulls], errors='coerce')
                has_hours = dates.dt.hour.sum() > 0
                has_minutes = dates.dt.minute.sum() > 0
                has_seconds = dates.dt.second.sum() > 0
                has_microseconds = dates.dt.microsecond.sum() > 0
                self.column_transforms[field] = {
                    'has_hours': has_hours,
                    'has_minutes': has_minutes,
                    'has_seconds': has_seconds,
                    'has_microseconds': has_microseconds
                }

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
                data.loc[nulls, f'{field}_Year'] = np.nan
                data.loc[nulls, f'{field}_Month'] = np.nan
                data.loc[nulls, f'{field}_Day'] = np.nan
                if transform_info['has_hours']:
                    data[f'{field}_Hour'] = data[field].dt.hour
                    data.loc[nulls, f'{field}_Hour'] = np.nan
                if transform_info['has_minutes']:
                    data[f'{field}_Minute'] = data[field].dt.minute
                    data.loc[nulls, f'{field}_Minute'] = np.nan
                if transform_info['has_seconds']:
                    data[f'{field}_Second'] = data[field].dt.second
                    data.loc[nulls, f'{field}_Second'] = np.nan
                if transform_info['has_microseconds']:
                    data[f'{field}_Microsecond'] = data[field].dt.microsecond
                    data.loc[nulls, f'{field}_Microsecond'] = np.nan
                data = data.drop(columns=[field])
        return data