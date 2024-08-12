#!/bin/bash

# Check if the required parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <directory_path> <subset>"
    echo "Example: $0 /path/to/checkpoints devtest"
    exit 1
fi

# Directory path and subset
DIR_PATH=$1
SUBSET=$2

# Verify that the provided path is a valid directory
if [ ! -d "$DIR_PATH" ]; then
    echo "The specified path is not a valid directory."
    exit 1
fi

# Verify that the provided subset is either 'dev' or 'devtest'
if [[ "$SUBSET" != "dev" && "$SUBSET" != "devtest" ]]; then
    echo "The subset must be either 'dev' or 'devtest'."
    exit 1
fi

# List all checkpoint directories, sorted naturally
CHECKPOINT_DIRS=$(ls -d $DIR_PATH/checkpoint-* 2>/dev/null | sort -V)

# Check if there are no checkpoint directories
if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "No checkpoint directories found in the specified path."
    exit 0
fi

# Loop through each checkpoint directory
for CHECKPOINT_DIR in $CHECKPOINT_DIRS; do
    # Check if the results file for the specified subset exists
    if [ ! -f "$CHECKPOINT_DIR/beam10.$SUBSET.results" ]; then
        echo "Running decode command for $CHECKPOINT_DIR..."

        # Execute the command with the specified environment variable and parameters
        env CUDA_VISIBLE_DEVICES=1 python -m decode \
            --exp "$CHECKPOINT_DIR" \
            --decode_subset "$SUBSET" \
            --decode_beams 10 \
            --decode_batch_size 1 \
            --prompt basic \
            --model_name_or_path mistralai/Mistral-7B-v0.3
    else
        echo "Results already exist for $CHECKPOINT_DIR, skipping..."
    fi
done
