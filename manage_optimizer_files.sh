#!/bin/bash

# Check if the required parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <directory_path> <checkpoint_number> [--dry-run]"
    echo "Example: $0 /path/to/checkpoints 50 --dry-run"
    exit 1
fi

# Directory path and checkpoint number
DIR_PATH=$1
CHECKPOINT_NUMBER=$2

# Dry run flag
DRY_RUN=false
if [ "$3" == "--dry-run" ]; then
    DRY_RUN=true
fi

# Verify that the provided path is a valid directory
if [ ! -d "$DIR_PATH" ]; then
    echo "The specified path is not a valid directory."
    exit 1
fi

# Collect all optimizer.pt files in the directory
optimizer_files=$(find "$DIR_PATH" -type f -name "optimizer.pt" | sort -V)

# Check if any optimizer.pt files were found
if [ -z "$optimizer_files" ]; then
    echo "No optimizer.pt files found in the specified directory."
    exit 0
fi

# Loop through each optimizer.pt file
for file in $optimizer_files; do
    # Extract the checkpoint number from the file path
    checkpoint_name=$(basename "$(dirname "$file")")
    checkpoint_num=$(echo "$checkpoint_name" | grep -oP '\d+')

    # Check if the checkpoint number is less than or equal to the specified number
    if [ "$checkpoint_num" -le "$CHECKPOINT_NUMBER" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Dry run: Would delete $file"
        else
            echo "Deleting $file"
            rm "$file"
        fi
    fi
done

echo "Operation completed."

