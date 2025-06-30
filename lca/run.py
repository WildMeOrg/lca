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
from functools import singledispatch

from graph_algorithm import graph_algorithm

def numpy_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
def save_probs_to_db(data, output_path):
    dir_name = os.path.dirname(output_path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=numpy_converter)

    return 

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


def process_verifier_edges(method, edges, human_reviewer, num_pos_needed, num_neg_needed, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg, quit = generate_ground_truth_random(edges, human_reviewer, num_pos_needed, num_neg_needed)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }, pos, neg


def process_edges(method, pos_edges, neg_edges, get_score, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg = generate_calib_weights(pos_edges, neg_edges, get_score)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    print(pos)
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }

    
def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('lca')
    # init params
    
    lca_config = config['lca']
    data_params = config['data']
    exp_name = config['exp_name']
    species = config['species']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if lca_config['logging'].get("update_log_file", True):
        log_file_name = f"tmp/logs/{exp_name}_{timestamp}.log"
        lca_config['logging']['log_file'] = log_file_name


    lca_params = generate_ga_params(lca_config)

    logger.info("START")
    
    verifier_name = lca_config['verifier_name']
    embeddings, uuids = load_pickle(data_params['embedding_file'])
    # uuids = [int(id) for id in uuids]
    # save_pickle((embeddings, uuids), data_params['embedding_file'])
    # a=1/0
    #create db files
    
    temp_db = lca_config.get('temp_db', False)
    
    if temp_db:
        logger.info(f"Using temp database...")
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(lca_config['db_path'], config['exp_name'])
        if ('clear_db' in lca_config) and lca_config['clear_db'] and os.path.exists(db_path):
            logger.info("Removing old database...")
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
    

    verifier_file =  os.path.join(str(db_path), "verifiers_probs.json")
    edge_db_file =  os.path.join(str(db_path), "quads.csv")
    clustering_file = os.path.join(str(db_path), "clustering.json")
    autosave_file = os.path.join(str(db_path), "autosave.json")
    edge_graph_file = os.path.join(str(db_path), "edge_graph_file.json")
    node2uuid_file = os.path.join(str(db_path), "node2uuid_file.json")

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

    # ids = [*range(51, 55), *range(56, 61), 347, 351]
    # for id in ids:
    #     uuid = node2uuid[id]
    #     # print(filtered_df[filtered_df["uuid_x"] == uuid])
    #     print(f"id: {id}, UUID: {uuid}, file_name: {list(filtered_df[filtered_df['uuid_x'] == uuid]['file_name'])}")

    # return
    logger.info(f"Ground truth clustering: {gt_clustering}")
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)

    

    # create embeddings verifier
    print(len(node2uuid.keys()))
    print(len(embeddings))
    

    # create human reviewer

    prob_human_correct = lca_params['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
    
    
    #curate LCA
    try:
        human_reviews = []
        current_clustering={}
        cluster_data = {}


        num_pos_needed = lca_params['num_pos_needed']
        num_neg_needed = lca_params['num_neg_needed']

        embeddings_dict = {
            'miewid': Embeddings(embeddings, node2uuid, distance_power=lca_params['distance_power']),
            'binary': BinaryEmbeddings(node2uuid, df, filter_key),
            'random': RandomEmbeddings(),
            'lightglue': LightglueEmbeddings(node2uuid, "lightglue_scores_superpoint.pickle")
        }
        verifiers_dict = {ver_name: call_verifier_alg(embeddings_dict[ver_name]) for ver_name in embeddings_dict.keys()}
        if verifier_name == 'synthetic' or 'synthetic' in lca_params['aug_names']:
            verifier_edges = embeddings_dict['miewid'].get_edges()
            
            pos, neg, quit = generate_ground_truth_full_dataset(verifier_edges, human_reviewer)
            scorer = kernel_density_scores.kernel_density_scores.create_from_samples(
                pos, neg
            ) 

            synthetic_embeddings = SyntheticEmbeddings(node2uuid, df, filter_key, lambda: scorer.density_pos.sample().item(), lambda: scorer.density_neg.sample().item())
            embeddings_dict['synthetic'] = synthetic_embeddings


        verifier_embeddings = embeddings_dict[lca_config['verifier_name']]
        
        verifier_alg = call_verifier_alg(verifier_embeddings)

        # verifier_embeddings = Embeddings(node2uuid, df, filter_key)
        verifier_edges = verifier_embeddings.get_edges()
        


        topk_results = verifier_embeddings.get_stats(filtered_df, filter_key)

        logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))


        top20_results = verifier_embeddings.get_top20_matches(filtered_df, filter_key)

        
        for uuid, top20 in top20_results.items():
            logger.info(f"ID: {uuid} | TOP-20: " + ", ".join([f"{k}: {v:.2f}" for (k, v) in top20]))
        
        if lca_config.get('clear_db', False) and os.path.exists(autosave_file):
            logger.info("Removing old autosave...")
            os.remove(autosave_file)
        if os.path.exists(autosave_file):
            wgtrs_calib_dict = load_json(verifier_file)
            autosave_object = load_json(autosave_file)
            current_clustering = autosave_object['clustering']
            cluster_ids_to_check = autosave_object['cluster_ids_to_check']
            lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)
            cluster_changes, is_finished = lca_object.curate([], [], cluster_ids_to_check)
        else:
            # generate wgtr calibration    

            # num_bins = 100
            # min_from_bin = 1
            # needed_total = 200#num_pos_needed + num_neg_needed
            # logger.info(f"Need total of {needed_total} reviews from {num_bins} bins with a minimum of {min_from_bin} samples from each bin")
            # pos, neg, quit = generate_wgtr_calibration_random_bins(verifier_edges, human_reviewer, needed_total, min_from_bin, num_bins=num_bins)
            human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
            
            # pos, pos_outliers = remove_outliers(pos, 1, 1.5)
            # neg, neg_outliers = remove_outliers(neg, -1, 1.5)
            # outliers = np.concatenate((pos_outliers, neg_outliers))
            # logger.info(f"Len before filtering: {len(verifier_edges)}")
            # verifier_edges = [edge for edge in verifier_edges if edge not in outliers]
            # logger.info(f"Len after filtering: {len(verifier_edges)}")
            
            if lca_config.get('verifier_file'):
                wgtrs_calib_dict = load_json(lca_config['verifier_file'])
            else:
                wgtrs_calib_dict = {}
                
                gt_weights, pos_edges, neg_edges = process_verifier_edges(verifier_name, verifier_edges, human_reviewer, num_pos_needed, num_neg_needed,  logger)
                wgtrs_calib_dict[verifier_name] = gt_weights
                
                for method in lca_params['aug_names']:
                    if method in {'human', verifier_name}:  # Skip 'human' and verifier itself
                        continue

                    if method not in embeddings_dict:
                        logger.warning(f"Embeddings for method {method} not found.")
                        continue

                    get_score = embeddings_dict[method].get_score
                    wgtrs_calib_dict[method] = process_edges(method, pos_edges, neg_edges, get_score, logger)
                    
                save_probs_to_db(wgtrs_calib_dict, verifier_file)

            wgtrs = ga_driver.generate_weighters(
                lca_params, wgtrs_calib_dict
            )
            # wgtr = wgtrs[0] 
            save_pickle(wgtrs, f"/ekaterina/work/src/lca/lca/tmp/wgtr_{exp_name}.pickle")

            # logger.info(f"positive edges to calibrate the weight function:")

            # for a0, a1, s in pos:
            #     logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtr.wgt(s)}")


            # logger.info(f"negative edges to calibrate the weight function:")

            # for a0, a1, s in neg:
            #     logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtr.wgt(s)}")

            logger.info(f"initial edges:")

            all_edges_plot = []
            if verifier_name in wgtrs.keys():
                for a0, a1, s in verifier_edges:
                    logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtrs[verifier_name].wgt(s)}")
                    all_edges_plot.append((a0, a1, s, wgtrs[verifier_name].wgt(s), gt_node2cid[int(a0)]== gt_node2cid[int(a1)]))
                write_json(all_edges_plot, f"/ekaterina/work/src/lca/lca/tmp/initial_edges_{exp_name}.json")
                get_histogram(all_edges_plot, wgtrs[verifier_name], species, timestamp, wgtrs_calib_dict[verifier_name])
            
            print(f"Printing log to {os.path.abspath(lca_config['logging']['log_file'])}")

            lca_object = curate_using_LCA(verifiers_dict, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)
            
            def save_iteration_graph(iteration):
                edge_graph = lca_object.db.edge_graph
                is_correct_edges = {(a0, a1): gt_node2cid[int(a0)]== gt_node2cid[int(a1)] for (a0, a1) in edge_graph.edges()} 
                
                nx.set_node_attributes(edge_graph, gt_node2cid, 'gt_cluster_id')
                nx.set_node_attributes(edge_graph, lca_object.db.node_to_cid, 'cluster_id')
                nx.set_edge_attributes(edge_graph, is_correct_edges , 'is_correct')
                edge_graph = nx.relabel_nodes(edge_graph, lambda x: str(x))

                edge_graph = nx.cytoscape_data(edge_graph)

                edge_graph_file = os.path.join(str(db_path), f"edge_graph_file_iter_{iteration}.json")
                write_json(edge_graph, edge_graph_file)
            graph_algorithm.save_iteration_graph = save_iteration_graph
            verifier_edges = [
                (n0, n1, s, verifier_name)
                for n0, n1, s in verifier_edges
            ]
            cluster_changes, is_finished = lca_object.curate(verifier_edges, human_reviews)
            embeddings_dict["binary"].final_print()
        
        

        write_json(lca_object.db.clustering, clustering_file)
        write_json(node2uuid, node2uuid_file)
        if is_finished and os.path.exists(autosave_file):
            os.remove(autosave_file)
    finally:
        if temp_db:
            shutil.rmtree(db_path)
        for handler in logger.handlers[:]:
            handler.flush()
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
       
