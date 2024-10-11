import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
from tools import *
import random
import os
from baseline_clustering_topk import baseline_clustering_topk
from cluster_validator import ClusterValidator
import ga_driver
from init_logger import init_logger
import matplotlib.pyplot as plt

import argparse
import datetime




def run_baseline_topk(config):
    logger = logging.getLogger('lca')
    # init params

    lca_config = config['lca']
    data_params = config['data']
    # lca_params = generate_ga_params(lca_config)

    exp_name = config['exp_name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"tmp/logs/{exp_name}_baseline_topk.log"
    lca_config['logging']['log_file'] = log_file_name

    lca_params = generate_ga_params(lca_config)
    
    embeddings, uuids = load_pickle(data_params['embedding_file'])

    #create db files

    db_path = os.path.join(lca_config['db_path'], config['exp_name'])
    os.makedirs(db_path, exist_ok=True)


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
                        embedding_uuids = uuids,
                        format='old'
                    )
    
    print_intersect_stats(df, individual_key=filter_key)


     # create cluster validator
    filtered_df = df[df['uuid_x'].isin(uuids)]
    embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key)



    logger.info(f"Ground truth clustering: {gt_clustering}")
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)


    # create embeddings verifier
    print(len(node2uuid.keys()))
    print(len(embeddings))
    verifier_embeddings = Embeddings(embeddings, node2uuid, distance_power=lca_params['distance_power'])
    # verifier_edges = verifier_embeddings.get_edges()
    


    topk_results = verifier_embeddings.get_stats(filtered_df, filter_key)

    logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))
   

    # create human reviewer

    prob_human_correct = lca_params['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
    



    # run baseline

    results = []
    thr=1
    # topks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    # topks = range(0, 11, 1)
    topks = [5]


    for topk in topks:
        verifier_edges = verifier_embeddings.get_baseline_edges(topk=topk, distance_threshold=thr)

        print(len(verifier_edges))
        clustering, node2cid, num_human, G = baseline_clustering_topk(list(gt_node2cid.keys()), verifier_edges, human_reviewer, cluster_validator)
        print(f" Threshold: {thr}")
        if len(cluster_validator.gt_results) == 0:
            cluster_validator.trace_start_human(clustering, node2cid, G, num_human)
        else:
            cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, G)
        # result = cluster_validator.incremental_stats(num_human, clustering, node2cid, gt_clustering, gt_node2cid)
        for i in range(0, len(cluster_validator.gt_results)):
            cluster_validator.gt_results[i]['score_threshold'] = thr
            cluster_validator.gt_results[i]['topk'] = topk
            cluster_validator.r_results[i]['score_threshold'] = thr
            cluster_validator.r_results[i]['topk'] = topk

    # write_json(cluster_validator.gt_results, data_params['stats_file'])


    return cluster_validator.gt_results, cluster_validator.r_results, node2uuid

   

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

    run_baseline_topk(config)