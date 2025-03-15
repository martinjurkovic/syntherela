#!/bin/bash

# Arrays to store datasets and their corresponding methods
DATASETS=(
    airbnb-simplified_subsampled
    rossmann_subsampled
    walmart_subsampled
    Biodegradability_v1
    imdb_MovieLens_v1
    CORA_v1
)

METHODS_LIST=(
"SDV RCTGAN REALTABFORMER MOSTLYAI CLAVADDPM"
"SDV RCTGAN REALTABFORMER MOSTLYAI CLAVADDPM"
"SDV RCTGAN REALTABFORMER MOSTLYAI CLAVADDPM"
"SDV RCTGAN MOSTLYAI"
"RCTGAN MOSTLYAI CLAVADDPM"
"SDV RCTGAN MOSTLYAI"
)

RUN_IDS=(1 2 3)

# Loop over each dataset
for i in "${!DATASETS[@]}"
do
    DATASET=${DATASETS[$i]}
    METHODS=${METHODS_LIST[$i]}

    # Create a new tmux window for this dataset
    tmux new-window -d -n "$DATASET" "bash"
    PATH=/lfs/local/0/$USER/.pixi/bin:$PATH
    eval "$(/dfs/user/roed/st environ hook --lfs-home --mamba-root-prefix /lfs/local/0/$USER/env/micromamba)"
    conda activate syntherela_2

    for RUN_ID in "${RUN_IDS[@]}"
    do
        # Build the method arguments
        METHOD_ARGS=""
        for METHOD in $METHODS
        do
            METHOD_ARGS+="-m $METHOD "
        done

        echo "Evaluating dataset $DATASET, run-id $RUN_ID methods $METHOD_ARGS"

        # Send the command to the tmux window
        tmux send-keys -t "$DATASET" "python experiments/evaluation/benchmark.py --dataset-name $DATASET --run-id $RUN_ID $METHOD_ARGS" C-m
    done
done
