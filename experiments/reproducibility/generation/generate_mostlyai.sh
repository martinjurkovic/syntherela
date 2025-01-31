#!/bin/bash

DATASETS=(
    airbnb-simplified_subsampled
    Biodegradability_v1
    CORA_v1
    imdb_MovieLens_v1
    rossmann_subsampled
    walmart_subsampled
    f1_subsampled
)

RUN_IDS=(
    1 
    2 
    3
)

for DATASET in ${DATASETS[@]}
do
    for RUN_ID in ${RUN_IDS[@]}
    do
        python experiments/generation/mostlyai/generate_mostlyai.py --dataset-name $DATASET --run-id $RUN_ID --api-key $1
        sleep 10
    done
done
