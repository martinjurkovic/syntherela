import os
from shutil import rmtree

import numpy as np

from syntherela.data import (
    load_tables,
    save_tables,
    download_sdv_relational_datasets,
)
from data.data_generators import generate_real_data


def test_dataset_download():
    download_sdv_relational_datasets("tests/tmp", datasets=["Biodegradability_v1"])
    assert os.path.exists("tests/tmp/Biodegradability_v1")
    rmtree("tests/tmp")


test_dataset_download()


def test_loading_and_saving():
    tables, metadata = generate_real_data()
    save_tables(tables, path="tests/tmp/test_data")
    assert os.path.exists("tests/tmp/test_data")

    loaded_tables = load_tables("tests/tmp/test_data", metadata)
    metadata.validate_data(loaded_tables)
    for table_name in tables.keys():
        table_meta = metadata.get_table_meta(table_name, to_dict=False)
        for column in tables[table_name].columns:
            assert column in loaded_tables[table_name].columns
            if table_meta.columns[column]["sdtype"] == "numerical":
                assert np.isclose(
                    tables[table_name][column].values,
                    loaded_tables[table_name][column].values,
                    rtol=1e-16,
                ).all()
            else:
                assert (
                    tables[table_name][column] == loaded_tables[table_name][column]
                ).all()
    rmtree("tests/tmp")
