import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
from tools import *
import random
import os
from cluster_validator import ClusterValidator
import ga_driver
from init_logger import init_logger
import tempfile
import argparse
import shutil
import datetime
from baseline_clustering import baseline_clustering




def save_probs_to_db(pos, neg, output_path, method='miewid'):
    dir_name = os.path.dirname(output_path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    data = {
        method: {
            "gt_positive_probs": [p.item() for _, _, p in pos],
            "gt_negative_probs": [p.item() for _, _, p in neg]
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

# def remove_outliers(pairs, std_mult=2):
#     scores = np.array([s for (_, _, s) in pairs])
#     filter = np.abs(scores - np.mean(scores)) < std_mult * np.std(scores)
#     return np.array(pairs)[filter], np.array(pairs)[np.logical_not(filter)]

def remove_outliers(pairs, sign=1, std_mult=2.5):
    scores = np.array([s for (_, _, s) in pairs])
    if sign < 0:
        filter = scores - np.mean(scores) > std_mult * np.std(scores)
    else:
        filter = np.mean(scores) - scores > std_mult * np.std(scores)
    # filter = np.abs(scores - np.mean(scores)) < std_mult * np.std(scores)
    return np.array(pairs)[np.logical_not(filter)], np.array(pairs)[filter]

def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('lca')
    # init params

    
    lca_config = config['lca']
    data_params = config['data']
    exp_name = config['exp_name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"tmp/logs/{exp_name}_baseline_threshold.log"
    lca_config['logging']['log_file'] = log_file_name


    lca_params = generate_ga_params(lca_config)
    
    embeddings, uuids = load_pickle(data_params['embedding_file'])

    #create db files
    
    temp_db = ('temp_db' in lca_config) and lca_config['temp_db']
    
    if temp_db:
        logger.info(f"Using temp database...")
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(lca_config['db_path'], config['exp_name'])
        os.makedirs(db_path, exist_ok=True)
        

    verifier_file =  os.path.join(str(db_path), "verifiers_probs.json")
    edge_db_file =  os.path.join(str(db_path), "quads.csv")
    clustering_file = os.path.join(str(db_path), "clustering.json")
    autosave_file = os.path.join(str(db_path), "autosave.json")

    lca_params['autosave_file'] = autosave_file


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
    verifier_edges = verifier_embeddings.get_edges()
    


    topk_results = verifier_embeddings.get_stats(filtered_df, filter_key)

    logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))
   

    # create human reviewer

    prob_human_correct = lca_params['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
    # run baseline

    human_reviews = []
    results = []
    current_clustering={}
    cluster_data = {}
    verifier_name = lca_config['verifier_name']
    # thrs = [0.1, 0.2, 0.35, 0.4, 0.45, 0.5]
    thrs = [0.5, 0.45, 0.4, 0.35, 0.2, 0.1]
    # verifier_alg = call_verifier_alg(verifier_embeddings)
    for thr in thrs:
        clustering, node2cid, num_human, G = baseline_clustering(list(gt_node2cid.keys()), verifier_edges, human_reviewer, thr)
        print(f" Threshold: {thr}")
        if num_human == 0:
            print("Initiate tracing")
            cluster_validator.trace_start_human(clustering, node2cid, G)            
        else:
            cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, G)
        
        cluster_validator.gt_results[-1]['score_threshold'] = thr
        cluster_validator.r_results[-1]['score_threshold'] = thr
        # result = cluster_validator.incremental_stats(num_human, clustering, node2cid, gt_clustering, gt_node2cid)
        # result['score_threshold'] = thr
        # results.append(result)
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

    run(config)