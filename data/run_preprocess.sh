#!/bin/bash

# Define paths to input/output files
ANNOTATION_PATH="/path/to/annotations.json"
SOURCE_DIR="/path/to/source/images"
OUTPUT_PATH="/path/to/output.json"
CSV_DIR="/path/to/csv/files"  # Optional, set to None if not needed
CSV_COLUMN_NAMES="['annotation_uuid', 'viewpoint_y']"
MERGE_COLS="['annotation_uuid', 'viewpoint_y']"

# Run the preprocess script
python3 preprocess.py \
    --annotation "$ANNOTATION_PATH" \
    --source "$SOURCE_DIR" \
    --output "$OUTPUT_PATH" \
    --csv "$CSV_DIR" \
    --csv_column_names "$CSV_COLUMN_NAMES" \
    --merge_cols "$MERGE_COLS"

