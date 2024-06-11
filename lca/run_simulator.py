
import json
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
import pandas as pd
import random
import logging
import os
import configparser
from init_logger import init_logger, get_formatter
from tools import *
from meiwid_embeddings import MiewID_Embeddings

init_logger()

#set human reviewer

gt_path = 'lca/tmp/zebra_test_gt.json'

def get_review(node_1, node_2, gt_path=gt_path, rate=0.98):
    with open(gt_path, 'r') as f:
        data = json.load(f)

    is_similar = False
    for row in data:
        if {row[0], row[1]} == {node_1, node_2}:
            is_similar=True
    
    return is_similar if random.random() < rate else not is_similar

def human_reviewer(edge_nodes, get_quit=False):
    logger = logging.getLogger('lca')
    reviews = [(n0, n1, get_review(n0, n1)) for n0, n1 in edge_nodes]
    quit_lca = False
    if get_quit:
        return reviews, quit_lca
    logger.info(f'Reviews  {reviews} ')
    return reviews
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
    
   
    
#init paths

input_path = "lca/tmp/zebra_gt.json"
edge_db_file = "lca/tmp/db/quads.json"
clustering_file = "lca/tmp/db/clustering.json"
lca_config_file = "/home/kate/code/lca/lca/tmp/config.ini"
embedding_file = "/home/kate/code/lca/lca/tmp/zebra_embeddings.pickle"


#create verification algorithm

current_clustering = {}

verifier_name = "miewid"


embeddings, labels, uuids = load_pickle(embedding_file)
ids = [i for i,_ in enumerate(uuids)]
miewid_embeddings = MiewID_Embeddings(embeddings, ids)
verifier_edges = miewid_embeddings.get_edges()

def verifier_alg(edge_nodes):
    logger = logging.getLogger('lca')
    scores = [miewid_embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
    logger.info(f'Scores  {scores} ')
    return scores



# generate wgtr calibration
num_pos_needed = 50
num_neg_needed = 50
verifier_file =  "lca/tmp/db/verifiers_probs.json"


pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)

wgtrs_calib_dict = save_probs_to_db(pos, neg, verifier_file)

config_ini = configparser.ConfigParser()
config_ini.read(lca_config_file )
lca_config = generate_ga_params(config_ini)



#curate LCA

human_reviews = []

lca_object = curate_using_LCA(verifier_alg, verifier_name, human_reviewer, wgtrs_calib_dict, edge_db_file, current_clustering, lca_config)

clusters = lca_object.curate(verifier_edges, human_reviews)

cluster_data = {}


for cluster in clusters[0]:
    print( cluster.new_clustering)
    for k, vals in cluster.new_clustering.items():
        # print(k, vals)
        cluster_data[k] = list(vals)
    
save_json(cluster_data, clustering_file)

