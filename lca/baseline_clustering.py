# -*- coding: utf-8 -*-
import networkx as nx
import logging
import numpy as np
import cluster_tools as ct


logger = logging.getLogger('lca')



def baseline_clustering(all_nodes, verifier_edges, human_reviewer, threshold):

    """
    Perform baseline clustering of nodes using verifier edges and human review, with thresholding for edge strength.
    
    Parameters:
    -----------
    all_nodes : list
        A list of all nodes to be clustered.
    
    verifier_edges : list of tuples
        A list of tuples representing edges between nodes, where each tuple is (node1, node2, score).
        The score represents the strength of the edge between two nodes and is used for thresholding.
        
    human_reviewer : function
        A callable function that simulates or performs human review on a list of edges. It takes as input a list 
        of edges in the format [(node1, node2)] and returns a list of reviews in the format [(node1, node2, is_the_same)], 
        where is_the_same is a boolean indicating if the two nodes represent the same entity.
        
    threshold : float
        A float between 0 and 1, representing the threshold used to split strong edges from those requiring review.
        - Edges with a score higher than `high_threshold` are automatically added as strong edges.
        - Edges with scores between `low_threshold` and `high_threshold` are sent for human review.

    Returns:
    --------
    clustering : dict
        A dictionary where keys are cluster ids and values are sets of nodes that belong to the corresponding cluster.
        
    node2cid : dict
        A dictionary mapping each node to its corresponding cluster id.
        
    num_human : int
        The number of human verifications performed during the clustering process.
        
    Notes:
    ------
    - The function first splits the edges based on their score into strong edges (added automatically) and 
      those requiring human review.
    - Strong edges are added directly to the graph, while those requiring review are processed by the 
      `human_reviewer` function.
    - The function clusters nodes based on connected components of the graph, returning a node-to-cluster mapping.
    """

    scores = np.array([s for (n1, n2, s) in verifier_edges])
    max_score = np.max(scores)
    min_score = np.min(scores)
    range_score = max_score - min_score

    high_threshold = (1 - threshold) * range_score + min_score
    low_threshold = threshold * range_score + min_score


    new_G = nx.Graph()
    new_G.add_nodes_from(all_nodes)

    strong_edges = [(n1,n2) for (n1, n2, s) in verifier_edges if s > high_threshold]
    
    new_G.add_edges_from(strong_edges)

    for_review = [(n1,n2) for (n1, n2, s) in verifier_edges if s <= high_threshold and s > low_threshold ]
    
    num_human = len(for_review)
    reviews, quit = human_reviewer(for_review)

    new_edges = [(n1,n2) for (n1, n2, s) in reviews if s]
    new_G.add_edges_from(new_edges)

    idx = 0
    clustering = dict()
    for cc in nx.connected_components(new_G):
        clustering[idx] = cc
        idx += 1
    node2cid = ct.build_node_to_cluster_mapping(clustering)
    
    return clustering, node2cid, num_human




    
