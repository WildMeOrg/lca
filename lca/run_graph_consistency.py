import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from embeddings_lightglue import LightglueEmbeddings
from binary_embeddings import BinaryEmbeddings
from synthetic_embeddings import SyntheticEmbeddings
from random_embeddings import RandomEmbeddings
from classifier import Classifier
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

from tools import connect_disconnected_clusters_strongest_node as connect_disconnected_clusters

from graph_consistency_old import GraphConsistencyAlgorithm

from graph_algorithm import graph_algorithm


def call_verifier_alg(embeddings):
    def verifier_alg(edge_nodes):
        logger = logging.getLogger('lca')
        scores = [embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
        logger.info(f'Scores  {scores} ')
        return scores
    return verifier_alg


def visualize_graph_with_labels(G):
    """
    Visualizes the graph with edge labels for "positive" and "negative" edges.

    Args:
        G (nx.Graph): The graph to visualize.
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Position layout

    # Extract edges based on labels
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == "positive"]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == "negative"]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", edgecolors="black")

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color="green", width=2, label="Positive Edges")
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color="red", width=2, label="Negative Edges")

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

    plt.title("Graph Visualization with Edge Labels")
    plt.legend()
    plt.savefig("graph_with_labels.png") 
    plt.show()


def save_graph_to_cytoscape(G, filename):
    """
    Saves the given NetworkX graph as a Cytoscape-compatible JSON file.

    Args:
        G (nx.Graph): The NetworkX graph.
        filename (str): The file path where the JSON will be saved.
    """
    # Ensure node labels are strings (Cytoscape expects string IDs)
    G = nx.relabel_nodes(G, lambda x: str(x))

    # Convert to Cytoscape format
    cytoscape_data = nx.cytoscape_data(G)

    # Save to file
    write_json(cytoscape_data, filename)

    print(f"Graph saved as Cytoscape JSON: {filename}")


def run(config):
    np.random.seed(42)
    random.seed(42)
    logger = logging.getLogger('lca')

    lca_config = config['lca']
    data_params = config['data']
    exp_name = config['exp_name']
    species = config['species']

    lca_config["flip_threshold"] = 0.5 #0.5
    lca_config["negative_threshold"] = 0.5 #0.5
    lca_config["densify_threshold"] = 10

    verifier_name = lca_config['verifier_name']


    def check_ground_truth(u, v, gt_node2cid):
        """
        Check if two nodes belong to the same cluster in ground truth.
        
        Args:
            u: first node
            v: second node
            gt_node2cid: dictionary mapping nodes to their ground truth cluster IDs
        
        Returns:
            bool: True if nodes are in same cluster, False otherwise
        """
        if u not in gt_node2cid or v not in gt_node2cid:
            return False
        return gt_node2cid[u] == gt_node2cid[v]



    def basic_verifier(edge, ranker_name, threshold=0.7):
        """
        Simulated verifier that uses ground truth with some error probability.
        
        Args:
            edge: tuple of (u, v, score)
            gt_node2cid: ground truth cluster assignments
            correct_prob: probability of making correct classification
        
        Returns:
            tuple: (confidence, label)
        """
        (u, v, score) = edge
        is_positive = score>threshold
        # confidence = score if is_positive else 1 - score
        max_range = min(1-threshold, threshold)
        confidence = min(np.abs(score - threshold)/max_range, 1)
        label = "positive" if is_positive else "negative"
        return (u, v, score, confidence, label, ranker_name)
    

    def simulated_verifier(edge, gt_node2cid, correct_prob=0.2):
        """
        Simulated verifier that uses ground truth with some error probability.
        
        Args:
            edge: tuple of (u, v, score, ranker_name)
            gt_node2cid: ground truth cluster assignments
            correct_prob: probability of making correct classification
        
        Returns:
            tuple: (confidence, label)
        """
        (u, v, score, ranker_name) = edge
        is_positive = check_ground_truth(u, v, gt_node2cid)
        confidence = score if is_positive else 1 - score
        full_correct_prob = correct_prob + (1-correct_prob)*confidence # if we want to correlate the correctness with the confidence even more
        if np.random.random() > full_correct_prob:
            is_positive = not is_positive
        return (confidence, "positive" if is_positive else "negative")
    

    # def configured_verifier(edge):
    #     return simulated_verifier(
    #         edge, 
    #         gt_node2cid,
    #         correct_prob=lca_config.get('classifier_correct_prob', 0.2)
    #     )

    def configured_verifier(edge):
        return basic_verifier(
            edge, 
            ranker_name = verifier_name,
            threshold=0.85,#0.7,
            # threshold=0.825,#0.7,
        )

    def human_verifier(edge):
        (u, v, human_label) = edge
        ranker_name = 'human'
        score = 1 if human_label else 0
        label = "positive" if human_label else "negative"
        return (u, v, score, confidence, label, ranker_name)
    

    def apply_verifier(edges, verifier):
        verified_edges = []
        for edge in edges:
            n0, n1, human_label = edge
            result = verifier(edge)
            verified_edges.append(result)
        return verified_edges

    

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if lca_config['logging'].get("update_log_file", True):
        log_file_name = f"tmp/logs/{exp_name}_{timestamp}.log"
        lca_config['logging']['log_file'] = log_file_name

    log_level = lca_config['logging']['log_level']
    log_file = lca_config['logging']['log_file']
    
    if log_file is not None:
        logger = logging.getLogger('lca')
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

        file_mode = lca_config['logging'].get("file_mode", 'w')
        
        handler = logging.FileHandler(log_file, mode=file_mode)
        handler.setLevel(log_level)
        handler.setFormatter(get_formatter())
        logger.addHandler(handler)

    
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
    
    print(f"Logging to {os.path.abspath(log_file)}", flush=True)

    # create cluster validator
    filtered_df = df[df['uuid_x'].isin(uuids)]
    embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key)



    logger.info(f"Ground truth clustering: {gt_clustering}")
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)

    embeddings_dict = {
        'miewid': Embeddings(embeddings, node2uuid, distance_power=lca_config['distance_power']),
    }

    verifier_embeddings = embeddings_dict[lca_config['verifier_name']]
    # uuids = verifier_embeddings.get_uuids()
    # embeds_dict = {"similarity_matrix":1-verifier_embeddings.get_distance_matrix(), 
    #                'animal_ids':[filtered_df[filtered_df['uuid_x'] == uuid][filter_key].squeeze() for uuid in uuids],
    #                'uuids':uuids,
    #                'embeddings':np.array(verifier_embeddings.embeddings)}

    # save_pickle(embeds_dict, "../gat/data.pickle")
    # raise "stop"

    # Here we need to get initial edges from our ranker (MiewID, take the same code from LCA run.py)
    initial_edges = list(verifier_embeddings.get_edges(target_edges=0, topk=10))
    
    initial_edges_verified = apply_verifier(initial_edges, configured_verifier)
    
    # print(initial_edges)

    classifier = Classifier(verifier_name, verifier_embeddings, configured_verifier)
    

    graph_consistency = GraphConsistencyAlgorithm(lca_config, classifier=classifier)

    topk_results = verifier_embeddings.get_stats(filtered_df, filter_key)

    logger.info(f"Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))

    # print(node2uuid.keys())

    # Create human reviewer
    prob_human_correct = lca_config['edge_weights']['prob_human_correct']
        
    human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)

    PCCs, for_review = graph_consistency.step(initial_edges_verified)

    
    logger.info(f"Finished step")

    num_human = 0
    iter = 0
    max_iter = 250000

    clustering, node2cid = graph_consistency.get_positive_clusters()
    cluster_validator.trace_start_human(clustering, node2cid, graph_consistency.G, num_human)
    
    human_review_step = 20
    
    while (len(PCCs) > 0 and iter < max_iter):
        human_reviews, _ = human_reviewer(for_review)
        confidence = lca_config["edge_weights"]["prob_human_correct"]
        human_reviews_verified = apply_verifier(human_reviews, human_verifier)
        
        num_human += len(human_reviews)

        logger.info(f"Received {len(human_reviews)} human reviews")
        logger.info(f"Iteration {iter}")
        PCCs, for_review = graph_consistency.step(human_reviews_verified)

        if num_human - cluster_validator.prev_num_human > human_review_step:
            clustering, node2cid = graph_consistency.get_positive_clusters()
            cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, graph_consistency.G)
        
        iter+=1

   
    # clustering, node2cid = graph_consistency.get_positive_clusters()

    # to_add_edges = connect_disconnected_clusters(graph_consistency.G, node2cid)

    # clustering, node2cid = graph_consistency.get_positive_clusters()
    # new_edges = [graph_consistency.classifier(edge) for edge in to_add_edges]
    # graph_consistency.add_new_edges(new_edges)

    # PCCs, for_review = graph_consistency.step(initial_edges_verified)
    # while (len(PCCs) > 0 and iter < max_iter):
    #     human_reviews = human_reviewer(for_review)
    #     confidence = lca_config["edge_weights"]["prob_human_correct"]
    #     human_reviews_verified = apply_verifier(human_reviews, human_verifier)
        
    #     num_human += len(human_reviews)

    #     logger.info(f"Received {len(human_reviews)} human reviews")
    #     logger.info(f"Iteration {iter}")
    #     PCCs, for_review = graph_consistency.step(human_reviews_verified)

    #     if num_human - cluster_validator.prev_num_human > human_review_step:
    #         clustering, node2cid = graph_consistency.get_positive_clusters()
    #         cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, graph_consistency.G)
        
    #     iter+=1


    cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, graph_consistency.G)

    db_path = os.path.join(lca_config['db_path'], config['exp_name'])
    if ('clear_db' in lca_config) and lca_config['clear_db'] and os.path.exists(db_path):
        logger.info("Removing old database...")
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)
    
    clustering_file = os.path.join(str(db_path), "consistency_clustering.json")
    node2uuid_file = os.path.join(str(db_path), "consistency_node2uuid_file.json")


    write_json(clustering, clustering_file)
    write_json(node2uuid, node2uuid_file)
    print(f"Saved output to {db_path}")
    # save_graph_to_cytoscape(graph_consistency.G, "graph_cytoscape.json")
    # print(f"Saved log to {os.path.abspath(log_file)}")
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
       
