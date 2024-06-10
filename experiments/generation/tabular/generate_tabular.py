import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from synthcity.plugins import Plugins
from syntherela.metadata import Metadata
from syntherela.data import load_tables, save_tables, remove_sdv_columns

Plugins(categories=["generic", "privacy"]).list()

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="airbnb-simplified_subsampled")
args.add_argument("--real-data-path", type=str, default="data/original")
args.add_argument("--synthetic-data-path", type=str, default="data/synthetic")
args.add_argument("--model-save-path", type=str, default="checkpoints")
args.add_argument("--run_id", type=str, default="1")
args = args.parse_args()
dataset_name = args.dataset_name
real_data_path = args.real_data_path
synthetic_data_path = args.synthetic_data_path
run_id = args.run_id

MODEL_NAMES = [
               'bayesian_network',
               'ddpm',
                'ctgan', 
               'tvae',
               'nflow',
               ]

SYNTHETIC_MODEL_PARAMS = {
    "marginal_distributions": {},
    "tvae": {
        "n_iter": 300,
        "lr": 0.0002,
        "decoder_n_layers_hidden": 4,
        "weight_decay": 0.001,
        "batch_size": 256,
        "n_units_embedding": 200,
        "decoder_n_units_hidden": 300,
        "decoder_nonlin": "elu",
        "decoder_dropout": 0.194325119117226,
        "encoder_n_layers_hidden": 1,
        "encoder_n_units_hidden": 450,
        "encoder_nonlin": "leaky_relu",
        "encoder_dropout": 0.04288563703094718,
    },
    "ctgan": {
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 50,
        "generator_nonlin": "tanh",
        "n_iter": 1000,
        "generator_dropout": 0.0574657940165757,
        "discriminator_n_layers_hidden": 4,
        "discriminator_n_units_hidden": 150,
        "discriminator_nonlin": "relu",
        "discriminator_n_iter": 3,
        "discriminator_dropout": 0.08727454632095322,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 500,
        "encoder_max_clusters": 14,
    },
    "bayesian_network": {
        "struct_learning_search_method": "hillclimb",
        "struct_learning_score": "bic",
    },
    "nflow": {
        "n_iter": 1000,
        "n_layers_hidden": 10,
        "n_units_hidden": 98,
        "dropout": 0.11496088236749386,
        "batch_norm": True,
        "lr": 0.0001,
        "linear_transform_type": "permutation",
        "base_transform_type": "rq-autoregressive",
        "batch_size": 512,
    },
    "ddpm": {
        "lr": 0.0009375080542687667,
        "batch_size": 2929,
        "num_timesteps": 998,
        "n_iter": 1051,
        "is_classification": False,
    },
}

metadata = Metadata().load_from_json(Path(real_data_path) / f'{dataset_name}/metadata.json')
real_data = load_tables(Path(real_data_path) / f'{dataset_name}', metadata)
real_data, metadata = remove_sdv_columns(real_data, metadata, validate=False)
metadata.validate_data(real_data)

for MODEL_NAME in MODEL_NAMES:
    synthetic_data = {}
    for table in real_data.keys():
        print(f"{MODEL_NAME} - {table}")
        try:
            X = real_data[table]
            X_orig = X.copy()
            syn_model = Plugins().get(MODEL_NAME, **SYNTHETIC_MODEL_PARAMS[MODEL_NAME])
            syn_model.strict = False

            # check if any column is constant
            constant_columns = X.columns[X.nunique() == 1]
            if len(constant_columns) > 0:
                X = X.drop(columns=constant_columns)


            numeric_columns = X.select_dtypes(include='number').columns
            if len(numeric_columns) > 0:
                imp = SimpleImputer(strategy='mean')
                X[numeric_columns] = pd.DataFrame(imp.fit_transform(X[numeric_columns]), columns=numeric_columns)
            
            syn_model.fit(X)
            synthetic_data[table] = syn_model.generate(len(X))
            synthetic_data[table] = synthetic_data[table].dataframe()

            # append comstant columns
            for column in constant_columns:
                synthetic_data[table][column] = X_orig[column]
        except Exception as e:
            print(f"Exception occured at {MODEL_NAME}-{table}: {e}")

    save_data_path = Path(synthetic_data_path) / dataset_name / MODEL_NAME / run_id / 'sample1'
    save_tables(synthetic_data, save_data_path)