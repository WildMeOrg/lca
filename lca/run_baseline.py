import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
from tools import *
import random
import os
from baseline_clustering import baseline_clustering
from cluster_validator import ClusterValidator
import ga_driver
from init_logger import init_logger

import argparse



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



def run(config):

    # init params

    lca_config = config['lca']
    data_params = config['data']
    lca_params = generate_ga_params(lca_config)
    
    embeddings, uuids = load_pickle(data_params['embedding_file'])

    #create db files

    db_path = os.path.join(lca_config['db_path'], config['exp_name'])
    os.makedirs(db_path, exist_ok=True)

    verifier_file =  os.path.join(db_path, "verifiers_probs.json")
    edge_db_file =  os.path.join(db_path, "quads.json")
    clustering_file = os.path.join(db_path, "clustering.json")


    # preprocess data

    name_keys = data_params['name_keys']
    filter_key = '__'.join(name_keys)
    df = preprocess_data(data_params['annotation_file'], 
                        name_keys= name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=data_params['viewpoint_list'], 
                        n_filter_min=data_params['n_filter_min'], 
                        n_filter_max=data_params['n_filter_max'],
                        images_dir = data_params['images_dir'], 
                        embedding_uuids = uuids
                    )
    
    print_intersect_stats(df, individual_key=filter_key)

   

    # create embeddings verifier

    filtered_df = df[df['uuid_x'].isin(uuids)]
    ids = filtered_df.index.tolist()

    verifier_embeddings = Embeddings(embeddings, ids)
    verifier_edges = verifier_embeddings.get_edges()


    # generate wgtr calibration    

    num_pos_needed = lca_params['num_pos_needed']
    num_neg_needed = lca_params['num_neg_needed']
    prob_human_correct = lca_params['prob_human_correct']
    
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)

    pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)
    wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)

    

    # create cluster validator

    gt_clustering, gt_node2cid = generate_gt_clusters(df, filter_key)
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)


    # run baseline

    human_reviews = []
    results = []
    current_clustering={}
    cluster_data = {}
    verifier_name = lca_config['verifier_name']
    thrs = [0.35, 0.4, 0.45, 0.5]
    # verifier_alg = call_verifier_alg(verifier_embeddings)
    for thr in thrs:
        clustering, node2cid, num_human = baseline_clustering(list(gt_node2cid.keys()), verifier_edges, human_reviewer, thr)
    # logger = logging.getLogger('lca')
        print(f" Threshold: {thr}")
        result = cluster_validator.incremental_stats(num_human, clustering, node2cid, gt_clustering, gt_node2cid)
        results.append(result)

    return results

    # write_json(results, data_params['stats_file'])

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