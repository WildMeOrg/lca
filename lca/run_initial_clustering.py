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
import pandas as pd



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
    log_file_name = f"tmp/logs/{exp_name}_{timestamp}.log"
    lca_config['logging']['log_file'] = log_file_name


    lca_params = generate_ga_params(lca_config)
    
    embeddings_query, uuids_query = load_pickle(data_params['embedding_file'])
    embeddings_db, uuids_db = load_pickle(data_params['db_embedding_file'])

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
    df_query = preprocess_data(data_params['annotation_file'], 
                        name_keys= name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=data_params['viewpoint_list'], 
                        n_filter_min=data_params['n_filter_min'], 
                        n_filter_max=data_params['n_filter_max'],
                        images_dir = data_params['images_dir'], 
                        embedding_uuids = uuids_query
                    )

    df_db = preprocess_data(data_params['db_annotation_file'], 
                        name_keys= name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=data_params['viewpoint_list'], 
                        n_filter_min=data_params['n_filter_min'], 
                        n_filter_max=data_params['n_filter_max'],
                        images_dir = data_params['images_dir'], 
                        embedding_uuids = uuids_db
                    )
    
    print_intersect_stats(df_query, individual_key=filter_key)
    print_intersect_stats(df_db, individual_key=filter_key)

   
   # create cluster validator
    filtered_df_query = df_query[df_query['uuid_x'].isin(uuids_query)]
    filtered_df_db = df_query[df_db['uuid_x'].isin(uuids_db)]
    filtered_df = pd.concat([filtered_df_query, filtered_df_db], ignore_index=True)
    embeddings_query = [embeddings_query[uuids_query.index(uuid)] for uuid in filtered_df_query['uuid_x']]
    embeddings_db = [embeddings_db[uuids_db.index(uuid)] for uuid in filtered_df_db['uuid_x']]
    embeddings = embeddings_query + embeddings_db
    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key)


    logger.info(f"Ground truth clustering: {gt_clustering}")
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)


    # create embeddings verifier
    print(len(node2uuid.keys()))
    print(len(embeddings_query))
    print(len(embeddings_db))
    verifier_embeddings = Embeddings(embeddings, node2uuid, distance_power=lca_params['distance_power'])
    db_edges = verifier_embeddings.get_edges(uuids_filter=set(uuids_db))
    query_edges = verifier_embeddings.get_edges(uuids_filter=set(uuids_query))
    


    # topk_results = verifier_embeddings.get_stats(filtered_df_query, filter_key, db=filtered_df_db)

    # logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))
   

    # create human reviewer

    prob_human_correct = lca_params['prob_human_correct']
        
    human_reviewer = call_get_reviews(df_query, filter_key, prob_human_correct)
    db_reviewer = call_get_reviews(df_db, filter_key, 1)
    

    #curate LCA
    try:
        human_reviews = []
        current_clustering={}
        cluster_data = {}
        verifier_name = lca_config['verifier_name']
        verifier_alg = call_verifier_alg(verifier_embeddings)

        if os.path.exists(autosave_file):
            wgtrs_calib_dict = load_json(verifier_file)
            autosave_object = load_json(autosave_file)
            current_clustering = autosave_object['clustering']
            cluster_ids_to_check = autosave_object['cluster_ids_to_check']
            lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)
            cluster_changes, is_finished = lca_object.curate([], [], cluster_ids_to_check)
        else:
            # generate wgtr calibration    

            reviews, _ = human_reviewer(db_edges)
            pos = [edge for (edge, (_, _, review)) in zip(db_edges, reviews) if review]
            neg = [edge for (edge, (_, _, review)) in zip(db_edges, reviews) if not review]

            logger.info(f"Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
            pos, pos_outliers = remove_outliers(pos, 1)
            neg, neg_outliers = remove_outliers(neg, -1)
            outliers = np.concatenate((pos_outliers, neg_outliers))
            logger.info(f"Len before filtering: {len(verifier_edges)}")
            verifier_edges = [edge for edge in verifier_edges if edge not in outliers]
            logger.info(f"Len after filtering: {len(verifier_edges)}")
            
            wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)

            wgtrs = ga_driver.generate_weighters(
                lca_params, wgtrs_calib_dict
            )
            wgtr = wgtrs[0] 
            save_pickle(wgtr, "/ekaterina/work/src/lca/lca/tmp/wgtr_extexp_uns.pickle")

            # logger.info(f"positive edges to calibrate the weight function:")

            # for a0, a1, s in pos:
            #     logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtr.wgt(s)}")


            # logger.info(f"negative edges to calibrate the weight function:")

            # for a0, a1, s in neg:
            #     logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtr.wgt(s)}")

            logger.info(f"initial edges:")

            all_edges_plot = []

            for a0, a1, s in verifier_edges:
                logger.info(f"a0: {a0}, a1: {a1}, s:{s}, w:{wgtr.wgt(s)}")
                all_edges_plot.append((a0, a1, s, wgtr.wgt(s), gt_node2cid[int(a0)]== gt_node2cid[int(a1)]))
            write_json(all_edges_plot, "/ekaterina/work/src/lca/lca/tmp/initial_edges_extexp_uns.json")
        
            lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)
            cluster_changes, is_finished = lca_object.curate(verifier_edges, human_reviews)

        write_json(lca_object.db.clustering, clustering_file)
        if is_finished and os.path.exists(autosave_file):
            os.remove(autosave_file)
    finally:
        if temp_db:
            shutil.rmtree(db_path)
    return cluster_validator.gt_results, node2uuid


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
       
