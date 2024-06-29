import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
from tools import *
import random
import configparser
import os
from cluster_validator import ClusterValidator
import ga_driver

import argparse


def get_review(node_1, node_2, df, name_key, rate=0.98):

    is_similar = False
    if df.iloc[node_1][name_key] == df.iloc[node_2][name_key]:
        is_similar=True
    
    return is_similar if random.random() < rate else not is_similar

def call_get_reviews(df, name_key):
    def get_reviews(edge_nodes, get_quit=False):
        logger = logging.getLogger('lca')
        reviews = [(n0, n1, get_review(n0, n1, df, name_key)) for n0, n1 in edge_nodes]
        quit_lca = False
        if get_quit:
            return reviews, quit_lca
        logger.info(f'Reviews  {reviews} ')
        return reviews
    return get_reviews
    # return reviews, quit_lca



def save_probs_to_db(pos, neg, output_path, method='miewid'):
    dir_name = os.path.dirname(output_path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    data = {
        method: {
            "gt_positive_probs": [p for _, _, p in pos],
            "gt_negative_probs": [p for _, _, p in neg]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data

def call_verifier_alg(embeddings):
    def verifier_alg(edge_nodes):
        logger = logging.getLogger('lca')
        scores = [embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
        logger.info(f'Scores  {scores} ')
        return scores
    return verifier_alg

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()

def run(config):
    embeddings, labels, uuids = load_pickle(config['data']['embedding_file'])
    # print(uuids)
    name_keys = config['data']['name_keys']
    filter_key = '__'.join(name_keys)
    df = preprocess_data(config['data']['annotation_file'], 
                        name_keys= name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=config['data']['viewpoint_list'], 
                        n_filter_min=config['data']['n_filter_min'], 
                        n_filter_max=config['data']['n_filter_max'],
                        images_dir = config['data']['images_dir'], 
                        embedding_uuids = uuids
                    )
    
    print_intersect_stats(df, individual_key=filter_key)

    human_reviewer = call_get_reviews(df, filter_key)

    lca_config = config['lca']
    db_path = lca_config['db_path']

    os.makedirs(db_path, exist_ok=True)

    filtered_df = df[df['uuid_x'].isin(uuids)]

    ids = filtered_df.index.tolist()

    verifier_embeddings = Embeddings(embeddings, ids)
    verifier_edges = verifier_embeddings.get_edges()

    # generate wgtr calibration
    num_pos_needed = 50
    num_neg_needed = 50
    verifier_file =  os.path.join(db_path, "verifiers_probs.json")


    pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)

    wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)

    lca_config_file = lca_config['config_file']

    config_ini = configparser.ConfigParser()
    config_ini.read(lca_config_file )
    lca_params = generate_ga_params(config_ini)


    # create cluster validator

    gt_clustering, gt_node2cid = generate_gt_clusters(df, filter_key)


    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)

    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)



    #curate LCA

    human_reviews = []
    current_clustering={}
    verifier_name = lca_config['verifier_name']

    edge_db_file =  os.path.join(db_path, "quads.json")
    clustering_file = os.path.join(db_path, "clustering.json")

    verifier_alg = call_verifier_alg(verifier_embeddings)

    lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, current_clustering, lca_params)

    clusters = lca_object.curate(verifier_edges, human_reviews)

    cluster_data = {}


    for cluster in clusters[0]:
        # print( cluster.new_clustering)
        for k, vals in cluster.new_clustering.items():
            # print(k, vals)
            cluster_data[k] = list(vals)
        
    write_json(cluster_data, clustering_file)


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    run(config)