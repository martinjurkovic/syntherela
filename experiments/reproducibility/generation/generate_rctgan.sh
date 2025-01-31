#!/bin/bash
DATASETS=(
    airbnb-simplified_subsampled
    Biodegradability_v1
    CORA_v1
    imdb_MovieLens_v1
    rossmann_subsampled
    walmart_subsampled
    f1_subsampled
    Berka_subsampled
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
        python experiments/generation/rctgan/generate_rctgan.py --dataset-name $DATASET --run-id $RUN_ID
    done
done