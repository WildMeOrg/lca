import pickle
import yaml
import json
import random
import logging
import cluster_tools as ct
from init_logger import get_formatter




def load_yaml(file_path):
    logger = logging.getLogger('lca')
    logger.info(f"Loading config from path: {file_path}")
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return config_dict



def get_config(file_path):

    config_dict = load_yaml(file_path)

    return config_dict


def get_review(node_1, node_2, df, name_key, rate=0.98):

    is_similar = False
    if df.iloc[node_1][name_key] == df.iloc[node_2][name_key]:
        is_similar=True
    
    return is_similar if random.random() < rate else not is_similar

def call_get_reviews(df, name_key, rate):
    def get_reviews(edge_nodes, rate=rate):
        logger = logging.getLogger('lca')
        reviews = [(n0, n1, get_review(n0, n1, df, name_key, rate)) for n0, n1 in edge_nodes]
        quit_lca = random.random() < 0.4
        # quit_lca = False
        return reviews, quit_lca
    return get_reviews
    # return reviews, quit_lca

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


class SetEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)
        

def load_json(path):
    with open(path, "r") as file:
        json_object = json.load(file)
    return json_object


def write_json(data, out_path):
    json_object = json.dumps(data, indent=4, cls=SetEncoder)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        


def load_pickle(file):
    with open(file, 'rb') as f_file:
        result = pickle.load(f_file)
    return result


def generate_ga_params(config_yaml):
    
    ga_params = dict()

    phc = float(config_yaml['edge_weights']['prob_human_correct'])
    assert 0 < phc <= 1
    ga_params['prob_human_correct'] = phc
    s = config_yaml['edge_weights']['augmentation_names']
    ga_params['aug_names'] = s.strip().split()

    ga_params['num_pos_needed'] = config_yaml['edge_weights']['num_pos_needed']
    ga_params['num_neg_needed'] = config_yaml['edge_weights']['num_neg_needed']


    mult = float(config_yaml['iterations']['min_delta_converge_multiplier'])
    ga_params['min_delta_converge_multiplier'] = mult

    s = float(config_yaml['iterations']['min_delta_stability_ratio'])
    assert s > 1
    ga_params['min_delta_stability_ratio'] = s

    n = int(config_yaml['iterations']['num_per_augmentation'])
    assert n >= 1
    ga_params['num_per_augmentation'] = n

    n = int(config_yaml['iterations']['tries_before_edge_done'])
    assert n >= 1
    ga_params['tries_before_edge_done'] = n

    i = int(config_yaml['iterations']['ga_iterations_before_return'])
    assert i >= 1
    ga_params['ga_iterations_before_return'] = i

    mw = int(config_yaml['iterations']['ga_max_num_waiting'])
    assert mw >= 1
    ga_params['ga_max_num_waiting'] = mw

    should_densify_str = str(config_yaml['iterations'].get('should_densify', False)).lower()
    ga_params['should_densify'] = should_densify_str == 'true'

    n = int(config_yaml['iterations'].get('densify_min_edges', 1))
    assert n >= 1
    ga_params['densify_min_edges'] = n

    df = float(config_yaml['iterations'].get('densify_frac', 0.0))
    assert 0 <= df <= 1
    ga_params['densify_frac'] = df

    log_level = config_yaml['logging']['log_level']
    ga_params['log_level'] = log_level
    log_file = config_yaml['logging']['log_file']
    ga_params['log_file'] = log_file

    draw_iterations_str = str(config_yaml['drawing'].get('draw_iterations', False)).lower()
    ga_params['draw_iterations'] = draw_iterations_str == 'true'
    ga_params['drawing_prefix'] = config_yaml['drawing']['drawing_prefix']

    logger = logging.getLogger('lca')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(log_level)
    handler.setFormatter(get_formatter())
    logger.addHandler(handler)

    return ga_params

def generate_gt_clusters(df, name_key):
    gt_node2cid = {}
    cids_unique = df[name_key].unique()
    cids = [f'ct{idx}' for idx in range(len(cids_unique))]
    gt_clustering = {cid: [] for cid in cids}
    ids = df.index.tolist()
    node2uuid = {}
    
    for i, row in df.iterrows():
        cid = cids[list(cids_unique).index(row[name_key])]
        gt_clustering[cid].append(i)
        gt_node2cid[i] = cid
        node2uuid[i] = row['uuid_x']

    gt_clustering = {
        cid: set(cluster) for cid, cluster in gt_clustering.items()
    }

    return gt_clustering, gt_node2cid, node2uuid


def print_intersect_stats(df,individual_key="name"):
    logger = logging.getLogger('lca')
    logger.info(f"** Dataset statistcs **")
    logger.info(f' - Counts: ')
    names = df[individual_key].unique()
    
    
    logger.info(f" ---- number of individuals: {len(names)}" )
    logger.info(f" ---- number of annotations: {len(df)}")
    avg_ratio = len(df) / len(names)
    logger.info(f" ---- average number of annotations per individual: {avg_ratio:.2f}")

    
