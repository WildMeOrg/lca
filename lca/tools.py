import pickle
import yaml
import json
import random
import logging
import cluster_tools as ct
from init_logger import get_formatter

import matplotlib.pyplot as plt
import json

import numpy as np

from weighter import weighter



def lazy(func):
    func.__val = None 
    def wrapper():
        computed = func.__val
        if computed:
            return computed
        computed = func()
        func.__val = computed
        return computed
    return wrapper

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
        # quit_lca = random.random() < 0.4
        quit_lca = False
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
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
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

def save_pickle(x, file):
    with open(file, 'wb') as f_file:
        result = pickle.dump(x, f_file)
    return result

def generate_ga_params(config):
    
    ga_params = dict()

    phc = float(config['edge_weights']['prob_human_correct'])
    assert 0 < phc <= 1
    ga_params['prob_human_correct'] = phc
    ga_params['simulate_human'] = config.get('edge_weights', True).get('simulate_human', True)
    s = config['edge_weights']['augmentation_names']
    ga_params['aug_names'] = s

    ga_params['num_pos_needed'] = config['edge_weights']['num_pos_needed']
    ga_params['num_neg_needed'] = config['edge_weights']['num_neg_needed']

    ga_params['distance_power'] = config['distance_power']

    
    ga_params['scorer'] = config['edge_weights']['scorer']

    mult = float(config['iterations']['min_delta_converge_multiplier'])
    ga_params['min_delta_converge_multiplier'] = mult

    s = float(config['iterations']['min_delta_stability_ratio'])
    assert s > 1
    ga_params['min_delta_stability_ratio'] = s

    n = int(config['iterations']['num_per_augmentation'])
    assert n >= 1
    ga_params['num_per_augmentation'] = n

    n = int(config['iterations']['tries_before_edge_done'])
    assert n >= 1
    ga_params['tries_before_edge_done'] = n

    ga_params['max_human_decisions'] = config['iterations']['max_human_decisions']

    i = int(config['iterations']['ga_iterations_before_return'])
    assert i >= 1
    ga_params['ga_iterations_before_return'] = i

    mw = int(config['iterations']['ga_max_num_waiting'])
    assert mw >= 1
    ga_params['ga_max_num_waiting'] = mw

    should_densify_str = str(config['iterations'].get('should_densify', False)).lower()
    ga_params['should_densify'] = should_densify_str == 'true'
    ga_params['densify_min_number_human'] = config['iterations']['densify_min_number_human']

    n = int(config['iterations'].get('densify_min_edges', 1))
    assert n >= 1
    ga_params['densify_min_edges'] = n

    df = float(config['iterations'].get('densify_frac', 0.0))
    assert 0 <= df <= 1
    ga_params['densify_frac'] = df

    log_level = config['logging']['log_level']
    ga_params['log_level'] = log_level
    log_file = config['logging']['log_file']
    ga_params['log_file'] = log_file

    draw_iterations_str = str(config['drawing'].get('draw_iterations', False)).lower()
    ga_params['draw_iterations'] = draw_iterations_str == 'true'
    ga_params['drawing_prefix'] = config['drawing']['drawing_prefix']

    

    if log_file is not None:
        logger = logging.getLogger('lca')
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

        file_mode = config['logging'].get("file_mode", 'w')
        
        handler = logging.FileHandler(log_file, mode=file_mode)
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

    annotation_counts = df[individual_key].value_counts()
    single_annotation_count = (annotation_counts == 1).sum()
    
    logger.info(f" ---- number of singletons: {single_annotation_count}")


def get_pos_neg_histogram(pos, neg, wgtr, species, timestamp, label):
    plt.figure()

    # print(pos)
    pos_weights = [wgtr.wgt(x)/10 for x in pos]
    neg_weights = [wgtr.wgt(x)/10 for x in neg]
 

    # plt.hist(pos_scores_filtered, bins=35, density=True, alpha=0.6, color='g')
    # plt.hist(neg_scores_filtered, bins=35, density=True, alpha=0.6, color='r')
    plt.hist(pos, bins=35, density=True, alpha=0.6, color='g')
    plt.hist(neg, bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_negative_probs'], bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_positive_probs'], bins=35, density=True, alpha=0.6, color='g')


    # plt.hist(ss, bins=35, density=True, alpha=0.6, color='b')

    xs = np.linspace(0, 1, 100)

    pos_ys = [wgtr.scorer.get_pos_neg(x)[0] for x in xs]
    neg_ys = [wgtr.scorer.get_pos_neg(x)[1] for x in xs]


    plt.plot(xs, pos_ys, color='g')
    plt.plot(xs, neg_ys, color='r')


    # wgtr = weighter.weighter(scorer, config["lca"]["edge_weights"]['prob_human_correct'])
    wgtr.max_weight = 10
    
    plt.plot(xs, [wgtr.wgt_smooth(x) for x in xs], 'k-')

    plt.xlabel("Score")
    plt.ylabel("Probability density function")
    plt.title(species)

    plt.savefig(f'/ekaterina/work/src/lca/lca/visualisations/{species}_{timestamp}_{label}.png')

def get_histogram(initial_edges, wgtr, species, timestamp, wgtrs_calib_dict):
    plt.figure()
    print(len(initial_edges))


    pos = [s for (a0, a1, s, w, c) in initial_edges if c]
    neg = [s for (a0, a1, s, w, c) in initial_edges if not c]

    ss = [s for (a0, a1, s, w, c) in initial_edges]

    pos_weights = [w/10 for (a0, a1, s, w, c) in initial_edges if c]
    neg_weights = [w/10 for (a0, a1, s, w, c) in initial_edges if not c]
 

    # plt.hist(pos_scores_filtered, bins=35, density=True, alpha=0.6, color='g')
    # plt.hist(neg_scores_filtered, bins=35, density=True, alpha=0.6, color='r')
    plt.hist(pos, bins=35, density=True, alpha=0.6, color='g')
    plt.hist(neg, bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_negative_probs'], bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_positive_probs'], bins=35, density=True, alpha=0.6, color='g')


    # plt.hist(ss, bins=35, density=True, alpha=0.6, color='b')

    xs = np.linspace(0, 1, 100)

    pos_ys = [wgtr.scorer.get_pos_neg(x)[0] for x in xs]
    neg_ys = [wgtr.scorer.get_pos_neg(x)[1] for x in xs]


    plt.plot(xs, pos_ys, color='g')
    plt.plot(xs, neg_ys, color='r')


    # wgtr = weighter.weighter(scorer, config["lca"]["edge_weights"]['prob_human_correct'])
    weighter_max = weighter.max_weight
    weighter.max_weight = 10
    
    plt.plot(xs, [wgtr.wgt_smooth(x) for x in xs], 'k-')

    weighter.max_weight = weighter_max

    plt.xlabel("Score")
    plt.ylabel("Probability density function")
    plt.title(species)

    plt.savefig(f'/ekaterina/work/src/lca/lca/visualisations/{species}__{timestamp}.png')


def connect_disconnected_clusters_strongest_node(G, node2cid):
    """
    Connects clusters by identifying the most "important" nodes (highest edge confidence)
    and linking disconnected clusters through them.

    Args:
        G (nx.Graph): The graph after final clustering.
        node2cid (dict): Mapping from node to cluster ID.

    Returns:
        G (nx.Graph): Modified graph with added edges between clusters.
        added_edges (list): List of newly added edges between clusters.
    """
    from collections import defaultdict
    logger = logging.getLogger('lca')
    cluster_to_nodes = defaultdict(list)
    node_max_conf = defaultdict(float)

    # Gather max confidence per node
    for u, v, data in G.edges(data=True):
        conf = data.get("confidence", 0)
        node_max_conf[u] = max(node_max_conf[u], conf)
        node_max_conf[v] = max(node_max_conf[v], conf)

    # Group nodes by cluster
    for node, cid in node2cid.items():
        cluster_to_nodes[cid].append(node)

    # Sort nodes in each cluster by descending edge confidence
    cluster_representatives = {
        cid: sorted(nodes, key=lambda x: -node_max_conf.get(x, 0))[0]
        for cid, nodes in cluster_to_nodes.items()
    }

    cluster_ids = list(cluster_representatives.keys())
    added_edges = set([])

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1, c2 = cluster_ids[i], cluster_ids[j]
            u, v = cluster_representatives[c1], cluster_representatives[c2]

            # Check if an edge already exists between any node pair from c1 and c2
            connected = any(G.has_edge(n1, n2) for n1 in cluster_to_nodes[c1] for n2 in cluster_to_nodes[c2])
            if not connected:
                # G.add_edge(u, v, label="added", confidence=0.0)
                # if G.has_edge(u, v):
                #     print("WTF")
                added_edges.add(tuple(sorted((u, v))))
                # print(f"Added edge between Cluster {c1} (node {u}) and Cluster {c2} (node {v})")
                # logger.info(f"Added edge between Cluster {c1} (node {u}) and Cluster {c2} (node {v})")
    logger.info(f"Missing {len(added_edges)} edges between clusters")
    return added_edges

