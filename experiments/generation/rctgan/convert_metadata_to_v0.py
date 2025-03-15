import argparse
import os

from syntherela.metadata import Metadata, convert_and_save_metadata_v0

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="airbnb-simplified_subsampled")
args.add_argument("--real-data-path", type=str, default="data/original")
args = args.parse_args()

dataset_name = args.dataset_name
real_data_path = args.real_data_path


data_path = os.path.join(real_data_path, dataset_name)
metadata = Metadata.load_from_json(os.path.join(data_path, "metadata.json"))

convert_and_save_metadata_v0(
    metadata,
    data_path,
)
