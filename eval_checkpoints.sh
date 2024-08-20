#!/bin/bash

# Check if the required parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <directory_path> <subset> [--decode_src_lang=<src_lang>] [--decode_tgt_lang=<tgt_lang>] [--cuda_devices=<devices>] [--model_name=<model_name>]"
    echo "Example: $0 /path/to/checkpoints devtest --decode_src_lang=eng_Latn --decode_tgt_lang=ukr_Cyrl --cuda_devices=0 --model_name=mistralai/Mistral-7B-v0.3"
    exit 1
fi

# Directory path and subset
DIR_PATH=$1
SUBSET=$2

# Default values for source and target languages, CUDA devices, and model name
DECODE_SRC_LANG="eng_Latn"
DECODE_TGT_LANG="ukr_Cyrl"
CUDA_DEVICES="1"  # Default value for CUDA_VISIBLE_DEVICES
MODEL_NAME="mistralai/Mistral-7B-v0.3"  # Default model name

# Process additional parameters for source, target languages, CUDA devices, and model name
for param in "$@"; do
    case $param in
        --decode_src_lang=*)
            DECODE_SRC_LANG="${param#*=}"
            ;;
        --decode_tgt_lang=*)
            DECODE_TGT_LANG="${param#*=}"
            ;;
        --cuda_devices=*)
            CUDA_DEVICES="${param#*=}"
            ;;
        --model_name=*)
            MODEL_NAME="${param#*=}"
            ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

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
CHECKPOINT_DIRS=$(ls -d $DIR_PATH/checkpoint-* 2>/dev/null | sort -V -r)

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
        python -m decode \
            --exp "$CHECKPOINT_DIR" \
            --decode_subset "$SUBSET" \
            --decode_beams 10 \
            --decode_batch_size 1 \
            --prompt basic \
            --model_name_or_path "$MODEL_NAME" \
            --decode_src_lang "$DECODE_SRC_LANG" \
            --decode_tgt_lang "$DECODE_TGT_LANG"
    else
        echo "Results already exist for $CHECKPOINT_DIR, skipping..."
    fi
done
