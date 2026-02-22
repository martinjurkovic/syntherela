import os
from syntherela.data import load_tables, save_tables
from syntherela.metadata import Metadata


PROJECT_PATH = __file__.split("experiments")[0]

path = os.path.join(PROJECT_PATH, "data/original/walmart/")
metadata_path = os.path.join(
    PROJECT_PATH, "data", "original", "walmart", "metadata.json"
)
metadata = Metadata.load_from_json(metadata_path)

tables = load_tables(path, metadata)

depts_df = tables["depts"]
features_df = tables["features"]

# keep only data data is between 2011-09-01 and 2011-10-01, column Date
features_df = features_df[
    (features_df["Date"] >= "2012-01-01") & (features_df["Date"] <= "2012-02-01")
]
depts_df = depts_df[
    (depts_df["Date"] >= "2012-01-01") & (depts_df["Date"] <= "2012-02-01")
]

tables["depts"] = depts_df
tables["features"] = features_df
save_path = os.path.join(PROJECT_PATH, "data/original/walmart_subsampled/")
save_tables(tables, save_path)
