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

# python3 run_clustering_with_save.py \
#     --config ./configs/config_GZCD_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/GZCD/hdbscan_clustering/
    

# python3 run_clustering_with_save.py \
#     --config ./configs/config_GZCD_manual_review.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/GZCD/manual_review_clustering/
 
# beluga
# python3 run_clustering_with_save.py \
#     --config ./configs/beluga/config_beluga_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/beluga/hdbscan_clustering/


# forestelephants
# python3 run_clustering_with_save.py \
#     --config ./configs/forestelephants/config_forestelephants_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/beluforestelephants/hdbscan_clustering/

# giraffe
# python3 run_clustering_with_save.py \
#     --config ./configs/giraffe/config_giraffe_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/giraffe/hdbscan_clustering/

# lion
# python3 run_clustering_with_save.py \
#     --config ./configs/lion/config_lion_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/lion/hdbscan_clustering/

# sperm whale
# python3 run_clustering_with_save.py \
#     --config ./configs/spermwhale/config_spermwhale_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/spermwhale/hdbscan_clustering/

# plains zebra
# python3 run_clustering_with_save.py \
#     --config ./configs/plainszebra/config_plainszebra_hdbscan.yaml \
#     --save_dir /fs/ess/PAS2136/ggr_data/results/kate/plainszebra/hdbscan_clustering/


# whaleshark
python3 run_clustering_with_save.py \
    --config ./configs/whaleshark/config_whaleshark_hdbscan.yaml \
    --save_dir /fs/ess/PAS2136/ggr_data/results/kate/whaleshark/hdbscan_clustering/