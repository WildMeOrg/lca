
import json
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
import pandas as pd
import random
import logging
import os
import configparser
from init_logger import init_logger, get_formatter

# init_logger()

gt_path = 'lca/tmp/zebra_gt.json'

def get_review(node_1, node_2, gt_path=gt_path, rate=0.98):
    with open(gt_path, 'r') as f:
        data = json.load(f)

    for row in data:
        if {row[0], row[1]} == {node_1, node_2}:
            return row[3] if random.random() < rate else not row[3]
    
    return False


def get_score(node_1, node_2, score_path=gt_path):
    with open(score_path, 'r') as f:
        data = json.load(f)

    for row in data:
        if {row[0], row[1]} == {node_1, node_2}:
            return row[2]
    
    return False

def human_reviewer(edge_nodes):
    reviews = [(n0, n1, get_review(n0, n1)) for n0, n1 in edge_nodes]
    quit_lca = False
    return reviews, quit_lca


def verifier_alg(edge_nodes):
    scores = [(n0, n1, get_score(n0, n1)) for n0, n1 in edge_nodes]
    return scores


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



def get_new_edges(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
        result = []
        
        for row in data:
            result.append([row[0], row[1], row[2]/100])
        
        # df = pd.DataFrame(result, columns=["annot_id_1", "annot_id_2", "weight", "aug_method"])
        
        return result
    

def get_new_reviews(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
        result = []
        
        for row in data:
            result.append([row[0], row[1], row[3]])
        
        # df = pd.DataFrame(result, columns=["annot_id_1", "annot_id_2", "weight", "aug_method"])
        
        return result
    

def generate_ga_params(config_ini):
    
    ga_params = dict()

    phc = float(config_ini['EDGE_WEIGHTS']['prob_human_correct'])
    assert 0 < phc < 1
    ga_params['prob_human_correct'] = phc
    s = config_ini['EDGE_WEIGHTS']['augmentation_names']
    ga_params['aug_names'] = s.strip().split()

    mult = float(config_ini['ITERATIONS']['min_delta_converge_multiplier'])
    ga_params['min_delta_converge_multiplier'] = mult

    s = float(config_ini['ITERATIONS']['min_delta_stability_ratio'])
    assert s > 1
    ga_params['min_delta_stability_ratio'] = s

    n = int(config_ini['ITERATIONS']['num_per_augmentation'])
    assert n >= 1
    ga_params['num_per_augmentation'] = n

    n = int(config_ini['ITERATIONS']['tries_before_edge_done'])
    assert n >= 1
    ga_params['tries_before_edge_done'] = n

    i = int(config_ini['ITERATIONS']['ga_iterations_before_return'])
    assert i >= 1
    ga_params['ga_iterations_before_return'] = i

    mw = int(config_ini['ITERATIONS']['ga_max_num_waiting'])
    assert mw >= 1
    ga_params['ga_max_num_waiting'] = mw

    ga_params['should_densify'] = config_ini['ITERATIONS'].getboolean(
        'should_densify', False
    )

    n = int(config_ini['ITERATIONS'].get('densify_min_edges', 1))
    assert n >= 1
    ga_params['densify_min_edges'] = n

    df = float(config_ini['ITERATIONS'].get('densify_frac', 0.0))
    assert 0 <= df <= 1
    ga_params['densify_frac'] = df

    log_level = config_ini['LOGGING']['log_level']
    ga_params['log_level'] = log_level
    log_file = config_ini['LOGGING']['log_file']
    ga_params['log_file'] = log_file

    ga_params['draw_iterations'] = config_ini['DRAWING'].getboolean('draw_iterations')
    ga_params['drawing_prefix'] = config_ini['DRAWING']['drawing_prefix']

    # logger = logging.getLogger('lca')
    # handler = logging.FileHandler(log_file, mode='w')
    # handler.setLevel(log_level)
    # handler.setFormatter(get_formatter())
    # logger.addHandler(handler)

    return ga_params
    

#add new edges to db
# def simulate(input_path, db_file, clustering_file):
#     db_object = db_interface_generic(db_file, clustering_file)
#     df_new_edges = get_new_edges(input_path)
#     # 
#     curate


input_path_initial = "lca/tmp/zebra_gt_initial.json"
input_path_additional = "lca/tmp/zebra_gt_add.json"
edge_db_file = "lca/tmp/db/quads.json"
clustering_file = "lca/tmp/db/clustering.json"
lca_config_file = "/home/kate/code/lca/lca/tmp/config.ini"

current_clustering = {}

verifier_name = "miewid"



# generate wgtr calibration
num_pos_needed = 50
num_neg_needed = 50
verifier_file =  "lca/tmp/db/verifiers_probs.json"

verifier_edges = get_new_edges(input_path_initial)

pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)

wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)

config_ini = configparser.ConfigParser()
config_ini.read(lca_config_file )
lca_config = generate_ga_params(config_ini)

verifier_results = get_new_edges(input_path_initial)
human_reviews = get_new_reviews(input_path_initial)



lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, current_clustering, lca_config)

clusters = lca_object.curate(verifier_results, human_reviews)

for cluster in clusters[0]:
    print(cluster.new_clustering)

print(len(clusters[0]))