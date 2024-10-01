#!/bin/bash

# Arrays to store datasets and their corresponding methods
DATASETS=(
    airbnb-simplified_subsampled
    rossmann_subsampled
    walmart_subsampled_12
    Biodegradability_v1
    imdb_MovieLens_v1
    CORA_v1
)

# List of run IDs
RUN_IDS=(
    1 
    2 
    3
)

# Loop over each dataset and run ID
for i in "${!DATASETS[@]}"
do
    DATASET=${DATASETS[$i]}
    for RUN_ID in "${RUN_IDS[@]}"
    do
        echo Evaluating dataset $DATASET, run-id $RUN_ID 
        # Run the Python script with the dataset, run ID, and methods
        python experiments/evaluation/utility.py --dataset-name $DATASET --run-id $RUN_ID $METHOD_ARGS
    done
done