import numpy as np
import hdbscan
from preprocess import preprocess_data
from embeddings import Embeddings
from tools import *
from init_logger import init_logger
import random
import os
import tempfile
import argparse
import shutil
import logging
import json

def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('hdbscan')
    # init params

    data_params = config['data']
    exp_name = config['exp_name']
    hdbscan_config = config.get('lca', {})

    embeddings, uuids = load_pickle(data_params['embedding_file'])

    # create db files
    temp_db = hdbscan_config.get('temp_db', False)
    
    if temp_db:
        logger.info(f"Using temp database...")
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(hdbscan_config['db_path'], exp_name)
        os.makedirs(db_path, exist_ok=True)

    def run_for_viewpoints(viewpoint_list, save_dir=str(db_path)):
        clustering_file = os.path.join(save_dir, "clustering.json")
        node2uuid_file = os.path.join(save_dir, "node2uuid_file.json")

        # preprocess data
        name_keys = data_params['name_keys']
        filter_key = '__'.join(name_keys)
        df = preprocess_data(data_params['annotation_file'], 
                            name_keys=name_keys,
                            convert_names_to_ids=True, 
                            viewpoint_list=viewpoint_list, 
                            n_filter_min=data_params['n_filter_min'], 
                            n_filter_max=data_params['n_filter_max'],
                            images_dir=data_params['images_dir'], 
                            embedding_uuids=uuids,
                            format='drone'
                        )
        
        print_intersect_stats(df, individual_key=filter_key)

        filtered_df = df[df['uuid_x'].isin(uuids)]
        filtered_embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
        node2uuid = {i: uuid for i, uuid in enumerate(filtered_df['uuid_x'])}

        # Apply hdbscan clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        labels = clusterer.fit_predict(filtered_embeddings)

        print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

        # Create clustering dict
        clustering = {}
        for node_id, label in enumerate(labels):
            lbl = f'cls{int(label)+1}' 
            if lbl not in clustering:
                clustering[lbl] = []
            clustering[lbl].append(node_id)

        print(clustering)

        # Save outputs
        write_json(clustering, clustering_file)
        write_json(node2uuid, node2uuid_file)

        logger.info(f"Saved clustering to {clustering_file} and node2uuid to {node2uuid_file}")

        return

    if data_params['separate_viewpoints']:
        for viewpoint in data_params['viewpoint_list']:
            print(f"Run for viewpoint {viewpoint}")
            save_dir = os.path.join(str(db_path), viewpoint)
            os.makedirs(save_dir, exist_ok=True)
            viewpoint_list = [viewpoint]
            run_for_viewpoints(viewpoint_list, save_dir)
    else:
        save_dir = str(db_path)
        os.makedirs(save_dir, exist_ok=True)
        run_for_viewpoints(data_params['viewpoint_list'], save_dir)

    if temp_db:
        shutil.rmtree(db_path)

    return

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()

if __name__ == '__main__':
    init_logger()
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    run(config)