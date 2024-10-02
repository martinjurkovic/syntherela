import json

import numpy as np
import pandas as pd

from syntherela.utils import NpEncoder, CustomHyperTransformer


def test_np_encoder():
    data = {
        "np_array": np.array([1, 2, 3]),
        "np_float": np.float32(1.0),
        "np_int": np.int64(1),
        "np_bool": np.bool_(True),
    }

    data_str = json.dumps(data, sort_keys=True, indent=4, cls=NpEncoder)

    expected = '{\n    "np_array": [\n        1,\n        2,\n        3\n    ],\n    "np_bool": true,\n    "np_float": 1.0,\n    "np_int": 1\n}'

    assert data_str == expected


def test_hyper_transformer():
    df = pd.DataFrame(
        {
            "numbers": [1, 2, 3],
            "strings": ["a", "b", "c"],
            "bools": [True, False, True],
            "dates": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )

    ht = CustomHyperTransformer()
    X = ht.fit_transform(df)

    assert "numbers" in X.columns
    assert "strings_0" in X.columns
    assert "strings_1" in X.columns
    assert "strings_2" in X.columns
    assert "bools" in X.columns
    assert "dates_Year" in X.columns
    assert "dates_Month" in X.columns
    assert "dates_Day" in X.columns
