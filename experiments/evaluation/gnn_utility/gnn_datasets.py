import os
from typing import Optional

import pandas as pd
from syntherela.data import load_tables

from relbench.base import Database, Dataset, Table
from syntherela.metadata import Metadata


class RossmannDataset(Dataset):
    name = "rossmann_subsampled"
    val_timestamp = pd.Timestamp("2014-09-20")
    test_timestamp = pd.Timestamp("2014-10-01")

    from_timestamp = pd.Timestamp("2014-07-31")
    upto_timestamp = pd.Timestamp("2014-11-01")

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        predict_column_task_config: dict = {},
        method: str = "ORIGINAL",
        run_id: int = 0,
    ):
        super().__init__(cache_dir, predict_column_task_config)
        self.method = method
        self.run_id = run_id

    def make_db(self) -> Database:
        data_type = "original" if self.method == "ORIGINAL" else "synthetic"
        path = os.path.join("data", data_type, "rossmann_subsampled")
        if self.method != "ORIGINAL":
            path = os.path.join(path, self.method, str(self.run_id), "sample1")
        # store = os.path.join(path, "store.csv")
        # historical = os.path.join(path, "historical.csv")
        metadata_path = os.path.join("data", "original", "rossmann_subsampled", "metadata.json")
        metadata = Metadata.load_from_json(metadata_path)

        tables = load_tables(path, metadata)

        store_df = tables["store"]
        historical_df = tables["historical"]

        # if max Date is smaller than self.upto_timestamp
        if historical_df["Date"].max() < self.test_timestamp:
            print("APPENDING TEST DATA")
            original_tables = load_tables(os.path.join("data", "original", "rossmann"), metadata)
            # copy data from original store_df from dateself.upto_timestamp to historical_df
            original_historical_df = original_tables["historical"]
            # take data only from self.upto_timestamp onward
            original_historical_df = original_historical_df[
                original_historical_df["Date"] >= self.test_timestamp
            ]
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


class WalmartDataset(Dataset):
    name = "walmart_subsampled"
    val_timestamp = pd.Timestamp("2011-09-26")
    test_timestamp = pd.Timestamp("2012-09-01")

    from_timestamp = pd.Timestamp("2011-09-01")
    upto_timestamp = pd.Timestamp("2012-10-01")

    def make_db(self) -> Database:
        path = os.path.join("data", "original", "walmart")
        metadata_path = os.path.join(path, "metadata.json")
        metadata = Metadata.load_from_json(metadata_path)

        tables = load_tables(path, metadata)

        depts_df = tables["depts"]
        stores_df = tables["stores"]
        features_df = tables["features"]

        depts_df = depts_df[depts_df["Date"] <= pd.Timestamp("2011-10-01")]

        if depts_df["Date"].max() < self.test_timestamp:
            print("APPENDING TEST DATA")
            original_tables = load_tables(
                os.path.join("data", "original", "walmart"), metadata
            )
            # copy data from original store_df from dateself.upto_timestamp to historical_df
            original_depts_df = original_tables["depts"]
            # take data only from self.upto_timestamp onward
            original_depts_df = original_depts_df[
                original_depts_df["Date"] >= self.test_timestamp
            ]
            depts_df = pd.concat([depts_df, original_depts_df])

        depts_df["Date"] = pd.to_datetime(depts_df["Date"], format="%Y-%m-%d")

        db = Database(
            table_dict={
                "depts": Table(
                    df=depts_df,
                    fkey_col_to_pkey_table={
                        "Store": "stores",
                    },
                    time_col="Date",
                ),
                "stores": Table(
                    df=stores_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="Store",
                ),
                "features": Table(
                    df=features_df,
                    fkey_col_to_pkey_table={
                        "Store": "stores",
                    },
                ),
            }
        )

        db = db.from_(self.from_timestamp)
        db = db.upto(self.upto_timestamp)

        return db

class F1Dataset(Dataset):
    name = "f1"
    val_timestamp = pd.Timestamp("2005-01-01")
    test_timestamp = pd.Timestamp("2010-01-01")

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        predict_column_task_config: dict = {},
        method: str = "ORIGINAL",
        run_id: int = 0,
    ):
        super().__init__(cache_dir, predict_column_task_config)
        self.method = method
        self.run_id = run_id

    def make_db(self) -> Database:
        data_type = "original" if self.method == "ORIGINAL" else "synthetic"
        path = os.path.join("data", data_type, "f1")
        if self.method != "ORIGINAL":
            path = os.path.join(path, self.method, str(self.run_id), "sample1")
        # store = os.path.join(path, "store.csv")
        # historical = os.path.join(path, "historical.csv")
        metadata_path = os.path.join("data", "original", "f1", "metadata.json")
        metadata = Metadata.load_from_json(metadata_path)

        tables = load_tables(path, metadata)

        # circuits = pd.read_csv(os.path.join(path, "circuits.csv"))
        # drivers = pd.read_csv(os.path.join(path, "drivers.csv"))
        # results = pd.read_csv(os.path.join(path, "results.csv"))
        # races = pd.read_csv(os.path.join(path, "races.csv"))
        # standings = pd.read_csv(os.path.join(path, "driver_standings.csv"))
        # constructors = pd.read_csv(os.path.join(path, "constructors.csv"))
        # constructor_results = pd.read_csv(os.path.join(path, "constructor_results.csv"))
        # constructor_standings = pd.read_csv(
        #     os.path.join(path, "constructor_standings.csv")
        # )
        # qualifying = pd.read_csv(os.path.join(path, "qualifying.csv"))

        circuits = tables["circuits"]
        drivers = tables["drivers"]
        results = tables["results"]
        races = tables["races"]
        standings = tables["standings"]
        constructors = tables["constructors"]
        constructor_results = tables["constructor_results"]
        constructor_standings = tables["constructor_standings"]
        qualifying = tables["qualifying"]

        # Remove columns that are irrelevant, leak time,
        # or have too many missing values

        # Drop the Wikipedia URL and some time columns with many missing values
        # races.drop(
        #     columns=[
        #         "url",
        #         "fp1_date",
        #         "fp1_time",
        #         "fp2_date",
        #         "fp2_time",
        #         "fp3_date",
        #         "fp3_time",
        #         "quali_date",
        #         "quali_time",
        #         "sprint_date",
        #         "sprint_time",
        #     ],
        #     inplace=True,
        # )

        # # Drop the Wikipedia URL as it is unique for each row
        # circuits.drop(
        #     columns=["url"],
        #     inplace=True,
        # )

        # # Drop the Wikipedia URL (unique) and number (803 / 857 are nulls)
        # drivers.drop(
        #     columns=["number", "url"],
        #     inplace=True,
        # )

        # # Drop the positionText, time, fastestLapTime and fastestLapSpeed
        # results.drop(
        #     columns=[
        #         "positionText",
        #         "time",
        #         "fastestLapTime",
        #         "fastestLapSpeed",
        #     ],
        #     inplace=True,
        # )

        # # Drop the positionText
        # standings.drop(
        #     columns=["positionText"],
        #     inplace=True,
        # )

        # # Drop the Wikipedia URL
        # constructors.drop(
        #     columns=["url"],
        #     inplace=True,
        # )

        # # Drop the positionText
        # constructor_standings.drop(
        #     columns=["positionText"],
        #     inplace=True,
        # )

        # # Drop the status as it only contains two categories, and
        # # only 17 rows have value 'D' (0.138%)
        # constructor_results.drop(
        #     columns=["status"],
        #     inplace=True,
        # )

        # # Drop the time in qualifying 1, 2, and 3
        # qualifying.drop(
        #     columns=["q1", "q2", "q3"],
        #     inplace=True,
        # )

        # # replase missing data and combine date and time columns
        # races["time"] = races["time"].replace(r"^\\N$", "00:00:00", regex=True)
        # races["date"] = races["date"] + " " + races["time"]
        # # Convert date column to pd.Timestamp
        # races["date"] = pd.to_datetime(races["date"])

        # # add time column to other tables
        # results = results.merge(races[["raceId", "date"]], on="raceId", how="left")
        # standings = standings.merge(races[["raceId", "date"]], on="raceId", how="left")
        # constructor_results = constructor_results.merge(
        #     races[["raceId", "date"]], on="raceId", how="left"
        # )
        # constructor_standings = constructor_standings.merge(
        #     races[["raceId", "date"]], on="raceId", how="left"
        # )

        # qualifying = qualifying.merge(
        #     races[["raceId", "date"]], on="raceId", how="left"
        # )

        # # Subtract a day from the date to account for the fact
        # # that the qualifying time is the day before the main race
        # qualifying["date"] = qualifying["date"] - pd.Timedelta(days=1)

        # # Replace "\N" with NaN in results tables
        # results = results.replace(r"^\\N$", np.nan, regex=True)

        # # Replace "\N" with NaN in circuits tables, especially
        # # for the column `alt` which has 3 rows of "\N"
        # circuits = circuits.replace(r"^\\N$", np.nan, regex=True)
        # # Convert alt from string to float
        # circuits["alt"] = circuits["alt"].astype(float)

        # # Convert non-numeric values to NaN in the specified column
        # results["rank"] = pd.to_numeric(results["rank"], errors="coerce")
        # results["number"] = pd.to_numeric(results["number"], errors="coerce")
        # results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
        # results["position"] = pd.to_numeric(results["position"], errors="coerce")
        # results["points"] = pd.to_numeric(results["points"], errors="coerce")
        # results["laps"] = pd.to_numeric(results["laps"], errors="coerce")
        # results["milliseconds"] = pd.to_numeric(
        #     results["milliseconds"], errors="coerce"
        # )
        # results["fastestLap"] = pd.to_numeric(results["fastestLap"], errors="coerce")

        # # Convert drivers date of birth to datetime
        # drivers["dob"] = pd.to_datetime(drivers["dob"])

        tables = {}

        tables["races"] = Table(
            df=pd.DataFrame(races),
            fkey_col_to_pkey_table={
                "circuitId": "circuits",
            },
            pkey_col="raceId",
            time_col="datetime",
        )

        tables["circuits"] = Table(
            df=pd.DataFrame(circuits),
            fkey_col_to_pkey_table={},
            pkey_col="circuitId",
            time_col=None,
        )

        tables["drivers"] = Table(
            df=pd.DataFrame(drivers),
            fkey_col_to_pkey_table={},
            pkey_col="driverId",
            time_col=None,
        )

        tables["results"] = Table(
            df=pd.DataFrame(results),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="resultId",
            time_col="date",
        )

        tables["standings"] = Table(
            df=pd.DataFrame(standings),
            fkey_col_to_pkey_table={"raceId": "races", "driverId": "drivers"},
            pkey_col="driverStandingsId",
            time_col="date",
        )

        tables["constructors"] = Table(
            df=pd.DataFrame(constructors),
            fkey_col_to_pkey_table={},
            pkey_col="constructorId",
            time_col=None,
        )

        tables["constructor_results"] = Table(
            df=pd.DataFrame(constructor_results),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorResultsId",
            time_col="date",
        )

        tables["constructor_standings"] = Table(
            df=pd.DataFrame(constructor_standings),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorStandingsId",
            time_col="date",
        )

        tables["qualifying"] = Table(
            df=pd.DataFrame(qualifying),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="qualifyId",
            time_col=None,
        )

        return Database(tables)

