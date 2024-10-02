import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata

from data.data_generators import generate_real_data
from syntherela.metadata import convert_metadata_to_v0


def test_metadata():
    data, metadata = generate_real_data()
    metadata.validate()
    metadata.validate_data(data)

    assert metadata.get_tables() == list(data.keys())

    assert metadata.get_primary_key("table1") == "pk1"
    assert metadata.get_primary_key("table2") == "pk2"

    table1_meta_dict = metadata.get_table_meta("table1", to_dict=True)
    assert table1_meta_dict["primary_key"] == "pk1"
    for column in data["table1"].columns:
        assert column in table1_meta_dict["columns"]

    table2_meta = metadata.get_table_meta("table2", to_dict=False)
    assert table2_meta.primary_key == "pk2"
    assert type(table2_meta) is SingleTableMetadata
    for column in data["table2"].columns:
        assert column in table2_meta.columns

    children = metadata.get_children("table1")
    assert "table2" in children

    parents = metadata.get_parents("table2")
    assert "table1" in parents

    foreign_keys = metadata.get_foreign_keys("table1", "table2")
    assert "fk2" in foreign_keys

    root_tables = metadata.get_root_tables()
    assert "table1" in root_tables
    assert set(root_tables) == parents

    # Rename a column
    metadata.rename_column("table1", "normal", "new_normal")
    assert "new_normal" in metadata.get_table_meta("table1", to_dict=True)["columns"]

    metadata.rename_column("table1", "pk1", "pk")
    assert metadata.get_primary_key("table1") == "pk"

    metadata.rename_column("table2", "fk2", "new_fk2")
    assert "new_fk2" in metadata.get_foreign_keys("table1", "table2")


def test_metadata_conversion():
    data, metadata = generate_real_data()
    # TODO: add these datattypes in the generator
    metadata.add_column("table1", "date", sdtype="datetime", datetime_format="%Y-%m-%d")
    data["table1"]["date"] = pd.date_range(
        "2020-01-01", periods=len(data["table1"]), freq="D"
    )
    metadata.add_column("table2", "bool", sdtype="boolean")
    data["table2"]["bool"] = [True, False] * (len(data["table2"]) // 2)
    metadata.validate()
    metadata.validate_data(data)

    metadata_v0 = convert_metadata_to_v0(metadata)

    for table in metadata.get_tables():
        assert table in metadata_v0["tables"]
        for column in data[table].columns:
            assert column in metadata_v0["tables"][table]["fields"]
        pk = metadata.get_primary_key(table)
        assert pk == metadata_v0["tables"][table]["primary_key"]
