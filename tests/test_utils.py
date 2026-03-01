import json

import numpy as np
import pandas as pd
from syntherela.utils import CustomHyperTransformer, NpEncoder


def test_np_encoder():
    data = {
        'np_array': np.array([1, 2, 3]),
        'np_float': np.float32(1.0),
        'np_int': np.int64(1),
        'np_bool': np.bool_(True),
    }

    data_str = json.dumps(data, sort_keys=True, indent=4, cls=NpEncoder)

    expected = '{\n    "np_array": [\n        1,\n        2,\n        3\n    ],\n    "np_bool": true,\n    "np_float": 1.0,\n    "np_int": 1\n}'  # noqa: E501

    assert data_str == expected


def test_hyper_transformer():
    df = pd.DataFrame(
        {
            'numbers': [1, 2, 3],
            'strings': ['a', 'b', 'c'],
            'bools': [True, False, True],
            'dates': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        }
    )

    ht = CustomHyperTransformer()
    X = ht.fit_transform(df)

    assert 'numbers' in X.columns
    assert 'strings_0' in X.columns
    assert 'strings_1' in X.columns
    assert 'strings_2' in X.columns
    assert 'bools' in X.columns
    assert 'dates_Year' in X.columns
    assert 'dates_Month' in X.columns
    assert 'dates_Day' in X.columns


def test_hyper_transformer_datetime_with_time_components():
    """CustomHyperTransformer expands datetime to Y/M/D/H/M/S."""
    df = pd.DataFrame(
        {
            'dt': pd.to_datetime(
                [
                    '2020-01-15 10:30:45',
                    '2021-06-20 14:22:00',
                    '2019-12-01 00:00:01',
                ]
            ),
        }
    )
    ht = CustomHyperTransformer()
    X = ht.fit_transform(df)
    assert 'dt_Year' in X.columns
    assert 'dt_Month' in X.columns
    assert 'dt_Day' in X.columns
    assert 'dt_Hour' in X.columns
    assert 'dt_Minute' in X.columns
    assert 'dt_Second' in X.columns
    assert 'dt_Second' in X.columns
    assert 'dt_Second' in X.columns
