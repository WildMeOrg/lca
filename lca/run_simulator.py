
import json
from curate_using_LCA import db_interface_generic, generate_wgtr_calibration_ground_truth
import pandas as pd
import random
import os

gt_path = 'lca/tmp/zebra_gt.json'

def get_review(node_1, node_2, gt_path=gt_path, rate=0.98):
    with open(gt_path, 'r') as f:
        data = json.load(f)

    for row in data:
        if {row[0], row[1]} == {node_1, node_2}:
            return row[3] if random.random() < rate else not row[3]
    
    return False

def human_reviewer(edge_nodes):
    reviews = [(n0, n1, get_review(n0, n1)) for n0, n1 in edge_nodes]
    quit_lca = False
    return reviews, quit_lca


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

    return 



def get_new_edges(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
        result = []
        
        for row in data:
            result.append([row[0], row[1], row[2], 'miewid'])
        
        df = pd.DataFrame(result, columns=["annot_id_1", "annot_id_2", "weight", "aug_method"])
        
        return df
    

#add new edges to db
def simulate(input_path, db_file, clustering_file):
    db_object = db_interface_generic(db_file, clustering_file)
    df_new_edges = get_new_edges(input_path)
    db_object.add_edges(df_new_edges)


input_path_initial = "lca/tmp/zebra_gt_initial.json"
input_path_additional = "lca/tmp/zebra_gt_add.json"
db_file = "lca/tmp/db/quads.json"
clustering_file = "lca/tmp/db/clustering.json"

#db doesn't exist
simulate(input_path_initial, db_file, clustering_file)

#add to existing db
simulate(input_path_additional, db_file, clustering_file)


# generate wgtr calibration
num_pos_needed = 50
num_neg_needed = 50
verifier_file =  "lca/tmp/db/verifiers_probs.json"

verifier_edges = get_new_edges(input_path_initial)

pos, neg, quit = generate_wgtr_calibration_ground_truth(verifier_edges, human_reviewer, num_pos_needed, num_neg_needed)

save_probs_to_db(pos, neg, verifier_file)