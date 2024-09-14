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




def run_baseline_topk(config):
    # init params

    lca_config = config['lca']
    data_params = config['data']
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
                        embedding_uuids = uuids
                    )
    
    print_intersect_stats(df, individual_key=filter_key)


     # create cluster validator

    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(df, filter_key)
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)


    # create human reviewer

    prob_human_correct = lca_params['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)

   

    # create embeddings verifier

    filtered_df = df[df['uuid_x'].isin(uuids)]
    ids = filtered_df.index.tolist()
    embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
    verifier_embeddings = Embeddings(embeddings, ids)
    



    # run baseline

    results = []
    thr=0.4
    topks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]

    for topk in topks:
        verifier_edges = verifier_embeddings.get_baseline_edges(topk=topk, distance_threshold=thr)

        print(len(verifier_edges))
        clustering, node2cid, num_human = baseline_clustering_topk(list(gt_node2cid.keys()), verifier_edges, human_reviewer)
        print(f" Threshold: {thr}")
        result = cluster_validator.incremental_stats(num_human, clustering, node2cid, gt_clustering, gt_node2cid)
        result['score_threshold'] = thr
        result['topk'] = topk
        results.append(result)

    write_json(results, data_params['stats_file'])


    return results

   

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