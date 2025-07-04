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
from human_db import human_db


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



def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('lca')
    # init params

    lca_config = config['lca']
    data_params = config['data']
    exp_name = config['exp_name']

    lca_params = generate_ga_params(lca_config)
    
    embeddings, uuids = load_pickle(data_params['embedding_file'])
    
    #create db files
    temp_db = lca_config['temp_db']
    
    if temp_db:
        logger.info(f"Using temp database...")
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(data_params['output_dir'], config['exp_name'])
        ui_db_path = data_params['ui_db_path']
        os.makedirs(db_path, exist_ok=True)


    verifier_file = lca_config['verifier_path']

    simulate_human = lca_params.get('simulate_human', True)
    
    use_human_reviews = True
    if "human" not in lca_params["aug_names"]:
        lca_params["aug_names"] = lca_params["aug_names"] + ["human"]
        use_human_reviews = False

    def run_for_viewpoints(viewpoint_list, save_dir=str(db_path)):
         # verifier_file =  os.path.join(str(db_path), "verifiers_probs.json")
        edge_db_file =  os.path.join(save_dir, "quads.csv")
        clustering_file = os.path.join(save_dir, "clustering.json")
        autosave_file = os.path.join(save_dir, "autosave.json")
        node2uuid_file = os.path.join(save_dir, "node2uuid_file.json")

        lca_params['autosave_file'] = autosave_file


        # preprocess data

        name_keys = data_params['name_keys']
        images_dir= data_params['images_dir']
        filter_key = '__'.join(name_keys)
        df = preprocess_data(data_params['annotation_file'], 
                            name_keys= name_keys,
                            convert_names_to_ids=True, 
                            viewpoint_list=viewpoint_list, 
                            n_filter_min=data_params['n_filter_min'], 
                            n_filter_max=data_params['n_filter_max'],
                            images_dir=images_dir, 
                            embedding_uuids = uuids,
                            format='drone'
                        )
        
        print_intersect_stats(df, individual_key=filter_key)

    
    # create cluster validator
        filtered_df = df[df['uuid_x'].isin(uuids)]
        filtered_embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
        gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key)
        write_json(node2uuid, node2uuid_file)
        cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
        ga_driver.set_validator_functions(cluster_validator.trace_start_human, cluster_validator.trace_iter_compare_to_gt)


        # create embeddings verifier
        # print(len(node2uuid.keys()))
        # print(len(filtered_embeddings))
        verifier_embeddings = Embeddings(filtered_embeddings, node2uuid, distance_power=lca_params['distance_power'])
        verifier_edges = verifier_embeddings.get_edges()

        # create human reviewer

        prob_human_correct = lca_params['prob_human_correct']
                
        if use_human_reviews:
            if simulate_human:
                human_reviewer = call_get_reviews(filtered_df, filter_key, prob_human_correct)
            else:
                human_reviewer = human_db(ui_db_path, filtered_df, node2uuid)
        else:
            human_reviewer = lambda _: ([], True)

        
        # human_reviewer.init_db(human_reviewer.db_path)
        

        #curate LCA
        try:
            human_reviews = []
            current_clustering={}
            cluster_data = {}
            verifier_name = lca_config['verifier_name']
            verifier_alg = call_verifier_alg(verifier_embeddings)
            

            
            # generate wgtr calibration    

            num_pos_needed = lca_params['num_pos_needed']
            num_neg_needed = lca_params['num_neg_needed']

            embeddings_dict = {
                'miewid': Embeddings(filtered_embeddings, node2uuid, distance_power=lca_params['distance_power']),
            }
            verifiers_dict = {ver_name: call_verifier_alg(embeddings_dict[ver_name]) for ver_name in embeddings_dict.keys()}



            if os.path.exists(verifier_file):
                print(f"verifier file exists at {verifier_file}")
                wgtrs_calib_dict = load_json(verifier_file)
            else:
                pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)
                wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)
        
            # lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)
            lca_object = curate_using_LCA(verifiers_dict, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, clustering_file, current_clustering, lca_params)

            verifier_edges = [
                (n0, n1, s, verifier_name)
                for n0, n1, s in verifier_edges
            ]
            print(f"Logging to {lca_config['logging']['log_file']}")
            cluster_changes, is_finished = lca_object.curate(verifier_edges, human_reviews)

            write_json(lca_object.db.clustering, clustering_file)

        finally:
            if temp_db:
                shutil.rmtree(db_path)
    

    if data_params['separate_viewpoints']:

        for viewpoint in data_params['viewpoint_list']:
            print(f"Run for viewpoint {viewpoint}")
            save_dir = os.path.join(str(db_path), viewpoint)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            viewpoint_list = [viewpoint]
            run_for_viewpoints(viewpoint_list, save_dir)
    else:
        save_dir = str(db_path)
        run_for_viewpoints(data_params['viewpoint_list'], save_dir)

       
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
       
