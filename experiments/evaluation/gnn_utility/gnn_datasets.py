import os
from typing import Optional


import pandas as pd
from syntherela.typing import Tables
from syntherela.data import load_tables

from relbench.base import Database, Dataset, Table
from syntherela.metadata import Metadata


def append_test_set(
    tables_train: Tables, tables_test: Tables, metadata: Metadata
) -> Tables:
    tables = {}
    for table in tables_train.keys():
        id_columns = metadata.get_column_names(table, sdtype="id")
        # Add test and train prefix to the id columns
        for column in id_columns:
            tables_train[table][column] = tables_train[table][column].apply(
                lambda x: f"train_{x}"
            )
            tables_test[table][column] = tables_test[table][column].apply(
                lambda x: f"test_{x}"
            )
        # Add the concatenated dataframe to the tables dict
        tables[table] = pd.concat(
            [tables_train[table], tables_test[table]], ignore_index=True
        )
    return tables


def cut_off_set(
    tables_train: Tables,
    metadata: Metadata,
    test_timestamp: pd.Timestamp,
    before: bool = True,
) -> Tables:
    tables = {}
    for table in tables_train.keys():
        datetime_columns = metadata.get_column_names(table, sdtype="datetime")
        tables[table] = tables_train[table]
        for column in datetime_columns:
            if before:
                tables[table] = tables[table][(tables[table][column] < test_timestamp) | (tables[table][column].isna())]
            else:
                tables[table] = tables[table][(tables[table][column] >= test_timestamp) | (tables[table][column].isna())]
    return tables


def get_tables_and_metadata(
    dataset: str, method: str, run_id: int
) -> tuple[Tables, Metadata]:
    data_type = "original" if method == "ORIGINAL" else "synthetic"
    path = os.path.join("data", data_type, dataset)
    if method != "ORIGINAL":
        path = os.path.join(path, method, str(run_id), "sample1")

    metadata_path = os.path.join("data", "original", dataset, "metadata.json")
    metadata = Metadata.load_from_json(metadata_path)

    tables = load_tables(path, metadata)

    return tables, metadata


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
        tables, metadata = get_tables_and_metadata(self.name, self.method, self.run_id)

        tables = cut_off_set(tables, metadata, self.test_timestamp)

        # TEST TABLES
        tables_test = load_tables(
            os.path.join("data", "original", "rossmann"), metadata
        )
        tables_test = cut_off_set(tables_test, metadata, self.test_timestamp, False)

        tables = append_test_set(tables, tables_test, metadata)

        store_df = tables["store"]
        historical_df = tables["historical"]
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

class AirbnbDataset(Dataset):
    name = "airbnb-simplified_subsampled"
    val_timestamp = pd.Timestamp("2014-05-15")
    test_timestamp = pd.Timestamp("2014-06-01")

    from_timestamp = pd.Timestamp("2014-01-01")
    upto_timestamp = pd.Timestamp("2014-07-01")

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
        tables, metadata = get_tables_and_metadata(self.name, self.method, self.run_id)

        # TEST TABLES
        tables_test = load_tables(
            os.path.join("data", "original", self.name, "test"), metadata
        )

        cols_to_drop = ["date_first_booking"]

        # Remove country_destination and timestamp_first_active from metadata
        for col in cols_to_drop:
            del metadata.tables["users"].columns[col]

        tables["users"] = tables["users"].drop(columns=cols_to_drop)
        tables = cut_off_set(tables, metadata, self.from_timestamp, False)
        tables = cut_off_set(tables, metadata, self.test_timestamp)

        tables_test["users"] = tables_test["users"].drop(columns=cols_to_drop)

        tables_test = cut_off_set(tables_test, metadata, self.test_timestamp, False)
        tables_test = cut_off_set(tables_test, metadata, self.upto_timestamp, True)


        tables = append_test_set(tables, tables_test, metadata)

        users_df = tables["users"]
        sessions_df = tables["sessions"]

        db = Database(
            table_dict={
                "users": Table(
                    df=users_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="id",
                    time_col="date_account_created",
                ),
                "sessions": Table(
                    df=sessions_df,
                    fkey_col_to_pkey_table={
                        "user_id": "users",
                    },
                ),
            }
        )

        return db


class WalmartDataset(Dataset):
    name = "walmart_subsampled"
    val_timestamp = pd.Timestamp("2012-01-24")
    test_timestamp = pd.Timestamp("2012-02-01")

    from_timestamp = pd.Timestamp("2012-01-01")
    upto_timestamp = pd.Timestamp("2012-03-01")

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
        tables, metadata = get_tables_and_metadata(self.name, self.method, self.run_id)
        tables = cut_off_set(tables, metadata, self.test_timestamp)

        # TEST TABLES
        tables_test = load_tables(os.path.join("data", "original", "walmart"), metadata)
        tables_test = cut_off_set(tables_test, metadata, self.test_timestamp, False)
        tables_test = cut_off_set(tables_test, metadata, self.upto_timestamp, True)

        tables = append_test_set(tables, tables_test, metadata)

        depts_df = tables["depts"]
        stores_df = tables["stores"]
        features_df = tables["features"]

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
    name = "f1_subsampled"
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
        tables, metadata = get_tables_and_metadata(self.name, self.method, self.run_id)
        tables = cut_off_set(tables, metadata, self.test_timestamp)

        # TEST TABLES
        tables_test = load_tables(os.path.join("data", "original", "f1"), metadata)
        tables_test = cut_off_set(tables_test, metadata, self.test_timestamp, False)

        tables = append_test_set(tables, tables_test, metadata)

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


class Berka(Dataset):
    name = "Berka"
    val_timestamp = pd.Timestamp("1997-12-01")
    test_timestamp = pd.Timestamp("1998-01-01")

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
        path = os.path.join("data", data_type, "Berka_subsampled")
        test_path = os.path.join("data", "original", "Berka")
        if self.method != "ORIGINAL":
            path = os.path.join(path, self.method, str(self.run_id), "sample1")

        metadata_path = os.path.join(
            "data", "original", "Berka_subsampled", "metadata.json"
        )
        metadata = Metadata.load_from_json(metadata_path)

        tables = load_tables(path, metadata)
        tables_test = load_tables(test_path, metadata)

        tables = append_test_set(tables, tables_test, metadata)

        account = tables["account"]
        card = tables["card"]
        client = tables["client"]
        disp = tables["disp"]
        district = tables["district"]
        loan = tables["loan"]
        order = tables["order"]
        trans = tables["trans"]

        # Only look if the loan was good (A, C) or bad (B, D)
        def remap_status(x):
            if x == "C":
                return "A"
            elif x == "D":
                return "B"
            else:
                return x

        loan.status = loan.status.apply(remap_status)

        tables = {}

        tables["account"] = Table(
            df=pd.DataFrame(account),
            fkey_col_to_pkey_table={
                "district_id": "district",
            },
            pkey_col="account_id",
            time_col="date",
        )

        tables["card"] = Table(
            df=pd.DataFrame(card),
            fkey_col_to_pkey_table={
                "disp_id": "disp",
            },
            pkey_col="card_id",
            time_col="issued",
        )

        tables["client"] = Table(
            df=pd.DataFrame(client),
            fkey_col_to_pkey_table={
                "district_id": "district",
            },
            pkey_col="client_id",
            time_col=None,
        )

        tables["disp"] = Table(
            df=pd.DataFrame(disp),
            fkey_col_to_pkey_table={
                "client_id": "client",
                "account_id": "account",
            },
            pkey_col="disp_id",
            time_col=None,
        )

        tables["district"] = Table(
            df=pd.DataFrame(district),
            fkey_col_to_pkey_table={},
            pkey_col="district_id",
            time_col=None,
        )

        tables["loan"] = Table(
            df=pd.DataFrame(loan),
            fkey_col_to_pkey_table={
                "account_id": "account",
            },
            pkey_col="loan_id",
            time_col="date",
        )

        tables["order"] = Table(
            df=pd.DataFrame(order),
            fkey_col_to_pkey_table={
                "account_id": "account",
            },
            pkey_col="order_id",
            time_col=None,
        )

        tables["trans"] = Table(
            df=pd.DataFrame(trans),
            fkey_col_to_pkey_table={
                "account_id": "account",
            },
            pkey_col="trans_id",
            time_col="date",
        )

        return Database(tables)
