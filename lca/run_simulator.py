
import json
from curate_using_LCA import db_interface_generic
import pandas as pd


def get_new_edges(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
        result = []
        
        for row in data:
            result.append([row[0], row[1], row[2], 'miewid'])
        
        df = pd.DataFrame(result, columns=["annot_id_1", "annot_id_2", "weight", "aug_method"])
        
        return df
    

def simulate(input_path, db_file, clustering_file):
    db_object = db_interface_generic(db_file, clustering_file)
    df_new_edges = get_new_edges(input_path)
    db_object.add_edges(df_new_edges)


input_path = "/home/kate/code/lca/lca/tmp/zebra_gt_initial.json"
db_file = "/home/kate/code/lca/lca/tmp/db/quads.json"
clustering_file = "/home/kate/code/lca/lca/tmp/db/clustering.json"

a = simulate(input_path, db_file, clustering_file)

input_path = "/home/kate/code/lca/lca/tmp/zebra_gt_add.json"
db_file = "/home/kate/code/lca/lca/tmp/db/quads.json"
clustering_file = "/home/kate/code/lca/lca/tmp/db/clustering.json"

b = simulate(input_path, db_file, clustering_file)
