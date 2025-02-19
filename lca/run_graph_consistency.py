import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from embeddings_lightglue import LightglueEmbeddings
from binary_embeddings import BinaryEmbeddings
from synthetic_embeddings import SyntheticEmbeddings
from random_embeddings import RandomEmbeddings
import scores.kernel_density_scores as kernel_density_scores
# from synthetic_embeddings import SyntheticEmbeddings as Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth, generate_ground_truth_random, generate_calib_weights, generate_ground_truth_full_dataset
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
import networkx as nx

from graph_consistency import GraphConsistencyAlgorithm

from graph_algorithm import graph_algorithm


def call_verifier_alg(embeddings):
    def verifier_alg(edge_nodes):
        logger = logging.getLogger('lca')
        scores = [embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
        logger.info(f'Scores  {scores} ')
        return scores
    return verifier_alg


def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('lca')

    def simple_verifier(edge, low=0.4, high=0.6):
        """
        A primitive verifier that just labels the edge according to the ranker score.

        Args:
            edge : (n0, n1, score, ranker_name)
            low (float, optional): lower threshold, scores lower than that are considered negative. Defaults to 0.4.
            high (float, optional): higher threshold, scores higher than that are considered positive. Defaults to 0.6.

        Returns:
            (confidence, label)
        """
        score = edge[2]
        if score < low:
            return ((low - score)/low, 'negative')
        elif score < high:
            return (0.5, 'incomparable')
        else:
            return ((score - high)/(1 - high), 'positive')
        
    graph_consistency = GraphConsistencyAlgorithm(simple_verifier)

    lca_config = config['lca']
    data_params = config['data']
    exp_name = config['exp_name']
    species = config['species']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if lca_config['logging'].get("update_log_file", True):
        log_file_name = f"tmp/logs/{exp_name}_{timestamp}.log"
        lca_config['logging']['log_file'] = log_file_name

    verifier_name = lca_config['verifier_name']
    embeddings, uuids = load_pickle(data_params['embedding_file'])

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

    embeddings_dict = {
        'miewid': Embeddings(embeddings, node2uuid, distance_power=lca_config['distance_power']),
    }
    verifiers_dict = {ver_name: call_verifier_alg(embeddings_dict[ver_name]) for ver_name in embeddings_dict.keys()}

    verifier_embeddings = embeddings_dict[lca_config['verifier_name']]
        

    # Here we need to get initial edges from our ranker (MiewID, take the same code from LCA run.py)
    initial_edges = verifier_embeddings.get_edges()

    topk_results = verifier_embeddings.get_stats(filtered_df, filter_key)

    logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))


    top20_results = verifier_embeddings.get_top20_matches(filtered_df, filter_key)


    # Create human reviewer
    prob_human_correct = lca_config['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)

    PCCs, for_review = graph_consistency.step(initial_edges, [])

    while len(PCCs) > 0:
        human_reviews = human_reviewer(for_review)
        PCCs, for_review = graph_consistency.step([], human_reviews)
    
    # add logger prints to keep track of what is going on

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
       
