#!/bin/bash

# Check if the required parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <experiment_directory> <subset> [--decode_src_lang=<src_lang>] [--decode_tgt_lang=<tgt_lang>]"
    echo "Example: $0 /path/to/experiment_directory devtest --decode_src_lang=eng_Latn --decode_tgt_lang=ukr_Cyrl"
    exit 1
fi

# Directory path and subset
EXPERIMENT_DIR=$1
SUBSET=$2

# Default values for source and target languages
DECODE_SRC_LANG="eng_Latn"
DECODE_TGT_LANG="ukr_Cyrl"

# Process additional parameters for source and target languages
for param in "$@"; do
    case $param in
        --decode_src_lang=*)
            DECODE_SRC_LANG="${param#*=}"
            ;;
        --decode_tgt_lang=*)
            DECODE_TGT_LANG="${param#*=}"
            ;;
    esac
done

# Verify that the provided path is a valid directory
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "The specified path is not a valid directory."
    exit 1
fi

# Verify that the provided subset is either 'dev' or 'devtest'
if [[ "$SUBSET" != "dev" && "$SUBSET" != "devtest" ]]; then
    echo "The subset must be either 'dev' or 'devtest'."
    exit 1
fi

# Determine the result file name pattern based on the language parameters
if [[ "$DECODE_SRC_LANG" == "eng_Latn" && "$DECODE_TGT_LANG" == "ukr_Cyrl" ]]; then
    result_pattern="beam10.$SUBSET.results"
else
    result_pattern="beam10.$SUBSET.$DECODE_SRC_LANG-$DECODE_TGT_LANG.results"
fi

# Find all result files in subdirectories of the experiment directory and sort them naturally
result_files=$(find "$EXPERIMENT_DIR" -type f -name "$result_pattern" | sort -V)

# Check if any result files were found
if [ -z "$result_files" ]; then
    echo "No $result_pattern files found in the specified directory."
    exit 0
fi

# Initialize variables to track the highest score and corresponding checkpoint
max_score=-1
max_score_path=""

# CSV file to store the results
csv_file="$EXPERIMENT_DIR/$SUBSET.csv"
echo "Checkpoint,Score" > "$csv_file"

# Iterate over each result file
for file in $result_files; do
    # Extract the score from the JSON file
    score=$(jq '.score' "$file" 2>/dev/null)

    # Check if the score was successfully extracted
    if [ $? -eq 0 ]; then
        # Extract the checkpoint name from the path
        checkpoint_name=$(basename "$(dirname "$file")")
        
        echo "Checkpoint: $checkpoint_name, Score: $score"

        # Save the checkpoint name and score to the CSV file
        echo "$checkpoint_name,$score" >> "$csv_file"

        # Update the maximum score and path if the current score is higher
        if (( $(echo "$score > $max_score" | bc -l) )); then
            max_score=$score
            max_score_path=$checkpoint_name
        fi
    else
        echo "Failed to extract score from $file"
    fi
done

# Check if a valid maximum score was found
if [ "$max_score_path" != "" ]; then
    echo "Checkpoint with the highest score:"
    echo "Checkpoint: $max_score_path, Score: $max_score"
else
    echo "No valid scores found."
fi

echo "Results saved to $csv_file"
