"""Utility functions for the Syntherela package.

NpEncoder for saving results and CustomHyperTransformer for data preprocessing.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
from sdmetrics.utils import HyperTransformer
from sklearn.preprocessing import OneHotEncoder


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types.

    This encoder converts NumPy data types to their Python equivalents
    for proper JSON serialization.
    """

    def default(self, o: Any) -> Any:
        """Convert NumPy objects to Python types.

        Parameters
        ----------
        o: object
            The object to encode.

        Returns
        -------
        object
            The encoded object.

        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


class CustomHyperTransformer(HyperTransformer):
    """Custom HyperTransformer to preserve feature names.

    This class overrides the transform method of HyperTransformer
    so that the feature names are preserved for one-hot-encoded columns.
    """

    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Parameters
        ----------
        data: pandas.DataFrame
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
                numeric = pd.to_numeric(data[field], errors='coerce').astype(
                    float
                )
                self.column_transforms[field] = {
                    'mode': numeric.mode().iloc[0],
                }
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder(handle_unknown='ignore')
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
                    'has_microseconds': has_microseconds,
                }

    def transform(self, data):
        """Transform the data.

        Parameters
        ----------
        data: pandas.DataFrame
            The data to transform.

        Returns
        -------
        pandas.DataFrame
            The transformed data.

        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        out = data.copy()
        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                out[field] = out[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                out[field] = pd.to_numeric(out[field], errors='coerce')
                out[field] = out[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                encoded = (
                    transform_info['one_hot_encoder']
                    .transform(col_data)
                    .toarray()
                )
                cols: list[str] = [
                    f'{field}_{i}' for i in range(np.shape(encoded)[1])
                ]
                transformed = pd.DataFrame(encoded, columns=pd.Index(cols))
                out = out.drop(columns=[field])
                out = pd.concat([out, transformed.set_index(out.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isnull()
                out[field] = pd.to_datetime(data[field], errors='coerce')
                out[f'{field}_Year'] = out[field].dt.year
                out[f'{field}_Month'] = out[field].dt.month
                out[f'{field}_Day'] = out[field].dt.day
                out.loc[nulls, f'{field}_Year'] = np.nan
                out.loc[nulls, f'{field}_Month'] = np.nan
                out.loc[nulls, f'{field}_Day'] = np.nan
                if transform_info['has_hours']:
                    out[f'{field}_Hour'] = out[field].dt.hour
                    out.loc[nulls, f'{field}_Hour'] = np.nan
                if transform_info['has_minutes']:
                    out[f'{field}_Minute'] = out[field].dt.minute
                    out.loc[nulls, f'{field}_Minute'] = np.nan
                if transform_info['has_seconds']:
                    out[f'{field}_Second'] = out[field].dt.second
                    out.loc[nulls, f'{field}_Second'] = np.nan
                if transform_info['has_microseconds']:
                    out[f'{field}_Microsecond'] = out[field].dt.microsecond
                    out.loc[nulls, f'{field}_Microsecond'] = np.nan
                out = out.drop(columns=[field])
        return out
