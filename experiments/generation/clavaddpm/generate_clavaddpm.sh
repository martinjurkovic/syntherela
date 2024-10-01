#!/bin/bash

DATASET_NAME=$1
REAL_DATA_PATH=$2
SYNTHETIC_DATA_PATH=$3

# Preprocess the dataset and compare configurations
conda activate reproduce_benchmark
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --run-id 1
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --run-id 2
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --run-id 3

# Generate synthetic data using ClavADDPM
cd ClavaDDPM
conda activate clavaddpm

python complex_pipeline.py --config_path configs/"$DATASET_NAME"_run1.json
python complex_pipeline.py --config_path configs/"$DATASET_NAME"_run2.json
python complex_pipeline.py --config_path configs/"$DATASET_NAME"_run3.json

# Postprocess the synthetic data
cd ..
conda activate reproduce_benchmark
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --synthetic-data-path $SYNTHETIC_DATA_PATH --run-id 1
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --synthetic-data-path $SYNTHETIC_DATA_PATH --run-id 2
python postprocess_dataset.py $DATASET_NAME --real-data-path $REAL_DATA_PATH --synthetic-data-path $SYNTHETIC_DATA_PATH --run-id 3