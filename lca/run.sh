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

python3 run_clustering_with_save.py \
    --config ./configs/lca_hdbscan_drone_edit.yaml 