import os

import pandas as pd
from syntherela.data import load_tables

from relbench.base import Database, Dataset, Table
from syntherela.metadata import Metadata


class RossmannDataset(Dataset):
    name = "rel-rossmann"
    val_timestamp = pd.Timestamp("2014-09-20")
    test_timestamp = pd.Timestamp("2014-10-01")

    from_timestamp = pd.Timestamp("2014-07-31")
    upto_timestamp = pd.Timestamp("2014-11-01")

    def make_db(self) -> Database:
        path = os.path.join("data", "original", "rossmann")
        # store = os.path.join(path, "store.csv")
        # historical = os.path.join(path, "historical.csv")
        metadata_path = os.path.join(path, "metadata.json")
        metadata = Metadata.load_from_json(metadata_path)

        tables = load_tables(path, metadata)

        store_df = tables["store"]
        historical_df = tables["historical"]

        # if max Date is smaller than self.upto_timestamp
        if historical_df["Date"].max() < self.test_timestamp:
            print("APPENDING TEST DATA")
            original_tables = load_tables(os.path.join("data", "rossmann"), metadata)
            # copy data from original store_df from dateself.upto_timestamp to historical_df
            original_historical_df = original_tables["historical"]
            # take data only from self.upto_timestamp onward
            original_historical_df = original_historical_df[original_historical_df["Date"] >= self.test_timestamp]
            historical_df = pd.concat([historical_df, original_historical_df])

        # store_df = pd.read_csv(store)
        # historical_df = pd.read_csv(historical)
        historical_df["Date"] = pd.to_datetime(historical_df["Date"], format="%Y-%m-%d")

        db = Database(
            table_dict={
                "store": Table(
                    df=store_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="Store",
                ),
                "historical": Table(
                    df=historical_df,
                    fkey_col_to_pkey_table={
                        "Store": "store",
                    },
                    pkey_col="Id",
                    time_col="Date",
                ),
            }
        )

        db = db.from_(self.from_timestamp)
        db = db.upto(self.upto_timestamp)

        return db
