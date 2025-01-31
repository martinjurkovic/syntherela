import os
from syntherela.data import load_tables, save_tables
from syntherela.metadata import Metadata
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

project_path = os.getenv("PROJECT_PATH")

path = os.path.join(project_path, 'data/original/f1/')
metadata_path = os.path.join(project_path, "data", "original", "f1", "metadata.json")
metadata = Metadata.load_from_json(metadata_path)

tables = load_tables(path, metadata)

races_df = tables["races"]
standings_df = tables["standings"]
constructor_results_df = tables["constructor_results"]
constructor_standings_df = tables["constructor_standings"]
results_df = tables["results"]

test_timestamp = "2010-01-01 00:00:00"


# Convert test_timestamp to datetime
test_timestamp_dt = pd.to_datetime(test_timestamp, format='%Y-%m-%d %H:%M:%S')

# Filter dataframes
constructor_results_df = constructor_results_df[constructor_results_df['date'] < test_timestamp_dt]
constructor_standings_df = constructor_standings_df[constructor_standings_df['date'] < test_timestamp_dt]
races_df = races_df[races_df['datetime'] < test_timestamp_dt]
results_df = results_df[results_df['date'] < test_timestamp_dt]
standings_df = standings_df[standings_df['date'] < test_timestamp_dt]

# Save tables
tables["constructor_results"] = constructor_results_df
tables["constructor_standings"] = constructor_standings_df
tables["races"] = races_df
tables["results"] = results_df
tables["standings"] = standings_df
save_path = os.path.join(project_path, 'data/original/f1_subsampled/')
save_tables(tables, save_path, metadata)