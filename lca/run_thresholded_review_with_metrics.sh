#!/bin/bash

# Run thresholded review algorithm with evaluation metrics displayed after each batch of 100 reviews
# The metrics will be shown in the console output if logging is configured properly

echo "=================================================="
echo "Running Thresholded Review with Batch Metrics"
echo "=================================================="
echo "Metrics will be displayed after each batch of 100 reviews:"
echo "- F1 Score"
echo "- Precision"
echo "- Recall"
echo "- Fraction Correct"
echo "=================================================="

# Run the clustering with save script
python3 run_clustering_with_save.py \
    --config ./configs/config_thresholded_review_GZCD.yaml \
    --save_dir /fs/ess/PAS2136/ggr_data/results/kate/GZCD/thresholded_review_clustering/

echo ""
echo "=================================================="
echo "Clustering completed!"
echo "Check the log file for detailed metrics from each batch."
echo "=================================================="