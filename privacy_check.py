import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from syntherela.data import load_tables
from syntherela.metadata import Metadata
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Function to calculate distances in batches
def calculate_min_distances(syn_batch, data, batch_size_data):
    min_distances = torch.full(
        (syn_batch.size(0),), float("inf"), device=syn_batch.device
    )
    for start_idx in range(0, data.size(0), batch_size_data):
        end_idx = min(start_idx + batch_size_data, data.size(0))
        data_batch = data[start_idx:end_idx]
        distances = (syn_batch[:, None] - data_batch).abs().sum(dim=2)
        min_batch_distances, _ = distances.min(dim=1)
        min_distances = torch.min(min_distances, min_batch_distances)
    return min_distances


def transform_data(
    real_data: tuple[pd.DataFrame, pd.DataFrame],
    syn_data: tuple[pd.DataFrame, pd.DataFrame],
    test_data: tuple[pd.DataFrame, pd.DataFrame],
    num_scaler: MinMaxScaler | None = None,
    cat_encoder: OneHotEncoder | None = None,
):
    cat_real_data, num_real_data = real_data
    cat_syn_data, num_syn_data = syn_data
    cat_test_data, num_test_data = test_data

    if cat_encoder is not None:
        cat_real_data_oh = cat_encoder.transform(cat_real_data.to_numpy()).toarray()
        cat_syn_data_oh = cat_encoder.transform(cat_syn_data.to_numpy()).toarray()
        cat_test_data_oh = cat_encoder.transform(cat_test_data.to_numpy()).toarray()
    else:
        assert (
            cat_real_data.shape[1]
            == cat_syn_data.shape[1]
            == cat_test_data.shape[1]
            == 0
        )
        cat_real_data_oh = np.empty((cat_real_data.shape[0], 0))
        cat_syn_data_oh = np.empty((cat_syn_data.shape[0], 0))
        cat_test_data_oh = np.empty((cat_test_data.shape[0], 0))

    if num_scaler is not None:
        num_real_data_np = num_scaler.transform(num_real_data.fillna(0).to_numpy())
        num_syn_data_np = num_scaler.transform(num_syn_data.fillna(0).to_numpy())
        num_test_data_np = num_scaler.transform(num_test_data.fillna(0).to_numpy())

    real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
    syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
    test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)
    return real_data_np, syn_data_np, test_data_np


def eval_dcr(
    syn_data: pd.DataFrame,
    real_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metadata,
    dcr_batch_size=1000,
    device="cpu",
    save_path="",
    subsample=None,
):
    if subsample is not None:
        syn_data = syn_data.sample(n=subsample, random_state=42)

    num_columns = metadata.get_column_names(sdtype="numerical")
    cat_columns = metadata.get_column_names(
        sdtype="categorical"
    ) + metadata.get_column_names(sdtype="boolean")

    scaler = MinMaxScaler()
    scaler.fit(real_data[num_columns].fillna(0).to_numpy())
    cat_encoder = None
    if len(cat_columns) > 0:
        cat_encoder = OneHotEncoder(handle_unknown="ignore")
        cat_encoder.fit(real_data[cat_columns].to_numpy())

    num_real_data = real_data[num_columns]
    cat_real_data = real_data[cat_columns]
    num_syn_data = syn_data[num_columns]
    cat_syn_data = syn_data[cat_columns]
    num_test_data = test_data[num_columns]
    cat_test_data = test_data[cat_columns]

    real_data_np, syn_data_np, test_data_np = transform_data(
        (cat_real_data, num_real_data),
        (cat_syn_data, num_syn_data),
        (cat_test_data, num_test_data),
        num_scaler=scaler,
        cat_encoder=cat_encoder,
    )

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = dcr_batch_size

    for i in tqdm(
        range((syn_data_th.shape[0] // batch_size) + 1), desc="Calculating DCR"
    ):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i * batch_size : (i + 1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i * batch_size :]

        # Calculate distances for real and test data in smaller batches
        dcr_real = calculate_min_distances(batch_syn_data_th, real_data_th, batch_size)
        dcr_test = calculate_min_distances(batch_syn_data_th, test_data_th, batch_size)

        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)

    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)

    equal = (dcrs_real == dcrs_test) * 0.5
    per_sample_score = (dcrs_real < dcrs_test) * 1.0 + equal

    score = per_sample_score.mean()
    std = per_sample_score.float().std()

    # print("DCR Score, a value closer to 0.5 is better")
    print(f"DCR Score = {score} ± {std / np.sqrt(dcrs_real.shape[0])}")

    torch.save(dcrs_real.cpu(), f"{save_path}dcrs_real.pt")
    torch.save(dcrs_real.cpu(), f"{save_path}dcrs_real.pt")
    torch.save(dcrs_test.cpu(), f"{save_path}dcrs_test.pt")
    return score


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    os.makedirs("results/dcr", exist_ok=True)

    table = "sessions"
    method = "MOSTLYAI"
    print(f"Using device: {device}")
    data_path = "data/original/airbnb-simplified_subsampled"
    metadata = Metadata().load_from_json(f"{data_path}/metadata.json")

    tables_real = load_tables(data_path, metadata)
    tables_test = load_tables("data/original/airbnb-dcr", metadata)

    metadata.validate_data(tables_real)
    metadata.validate_data(tables_test)

    methods = [
        "MOSTLYAI",
        "RGCLD",
        "CLAVADDPM",
        "RCTGAN",
        "REALTABFORMER",
        "SDV",
    ]

    for method in methods:
        tables_syn = load_tables(
            f"data/synthetic/airbnb-simplified_subsampled/{method}/1/sample1", metadata
        )
        metadata.validate_data(tables_syn)
        for table in metadata.get_tables():
            print(f"Evaluating {table} with method {method}")
            syn_data = tables_syn[table].copy()
            real_data = tables_real[table].copy()
            test_data = tables_test[table].copy()
            eval_dcr(
                syn_data,
                real_data,
                test_data,
                metadata.get_table_meta(table, to_dict=False),
                device=device,
                save_path=f"results/dcr/{table}_{method}_",
                subsample=None,
            )
