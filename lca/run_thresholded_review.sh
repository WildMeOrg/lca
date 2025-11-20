#!/bin/bash

# python3 run.py --config ./configs/config_beluga.yaml
# python3 run.py --config ./configs/config_grevyszebra.yaml
# python3 run.py --config ./configs/config_plainszebra.yaml
# python3 run.py --config ./configs/config_forestelephants.yaml
# python3 run.py --config ./configs/config_forestelephants_small.yaml
# python3 run.py --config ./configs/config_whaleshark.yaml
# python3 run.py --config ./configs/config_giraffe.yaml
# python3 run.py --config ./configs/config_lion.yaml
# python3 run.py --config ./configs/config_spermwhale.yaml
# ./configs/config_zebra.yaml 


# python3 run.py --config ./configs/config_whaleshark.yaml
# python3 run.py --config ./configs/lca_image.yaml -i

# python3 run.py --config ./configs/lca_hdbscan_drone.yaml -i

# python3 run_clustering_with_save.py --config ./configs/lca_hdbscan_drone_edit.yaml --output_prefix hdbscan --output_suffix clustered -i 

# python3 run_clustering_with_save.py \
#     --config ./configs/lca_hdbscan_drone_edit.yaml 

# Run with evaluation metrics printed after each batch of 100 reviews
# The thresholded_review_algorithm.py now automatically calculates and logs metrics after each batch

echo "=================================================="
echo "Running Thresholded Review with Batch Metrics"
echo "=================================================="
echo "Metrics will be logged after each batch of 100 reviews:"
echo "- F1 Score"
echo "- Precision"
echo "- Recall"
echo "- Fraction Correct"
echo "=================================================="
echo ""

# python3 run_clustering_with_save.py \
#     --config ./configs/config_thresholded_review_GZCD.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/GZCD/thresholded_review_clustering/
    

# python3 run_clustering_with_save.py \
#     --config ./configs/config_GZCD_manual_review.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/GZCD/manual_review_clustering/
 
# beluga
# python3 run_clustering_with_save.py \
#    --config ./configs/beluga/config_thresholded_review_beluga.yaml \
#    --save_dir /fs/ess/PAS2136/ggr_data/results/kate/beluga/thresholded_review_clustering/
  
# # forestelephants
# python3 run_clustering_with_save.py \
#     --config ./configs/forestelephants/config_thresholded_review_forestelephants.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/forestelephants/thresholded_review_clustering/

# # giraffe
# python3 run_clustering_with_save.py \
#     --config ./configs/giraffe/config_thresholded_review_giraffe.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/giraffe/thresholded_review_clustering/

# lion
# python3 run_clustering_with_save.py \
#     --config ./configs/lion/config_thresholded_review_lion.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/lion/thresholded_review_clustering/

# sperm whale
# python3 run_clustering_with_save.py \
#     --config ./configs/spermwhale/config_thresholded_review_spermwhale.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/spermwhale/thresholded_review_clustering/


# plains zebra
# python3 run_clustering_with_save.py \
#     --config ./configs/plainszebra/config_thresholded_review_plainszebra.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/plainszebra/thresholded_review_clustering/

# whaleshark
python3 run_clustering_with_save.py \
    --config ./configs/whaleshark/config_thresholded_review_whaleshark.yaml \
    --save_dir /fs/ess/PAS2136/ggr_data/results/kate/whaleshark/thresholded_review_clustering/
