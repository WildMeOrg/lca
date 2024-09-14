# -*- coding: utf-8 -*-
import networkx as nx
import logging

import cluster_tools as ct


logger = logging.getLogger('lca')



def baseline_clustering_topk(all_nodes, verifier_edges, human_reviewer):
    """
    Perform a baseline clustering of nodes using human review to verify the top-k highest scoring edges.

    Parameters:
    -----------
    all_nodes : list
        A list of nodes to be clustered.
        
    verifier_edges : list of tuples
        A list of tuples representing edges between nodes, where each tuple is (node1, node2, score).
        The score is used to sort and prioritize edges for human review.
        
    human_reviewer : function
        A callable that simulates or performs human review. It takes a list of edges as input and returns
        a list of reviews in the format [(node1, node2, is_the_same)] where is_the_same is a boolean indicating
        if the two nodes represent the same entity. 

    Returns:
    --------
    clustering : dict
        A dictionary where keys are cluster ids and values are sets of nodes that belong to the corresponding cluster.
        
    node2cid : dict
        A mapping of each node to its respective cluster id.
        
    num_human : int
        The number of human verifications performed during the clustering process.
    """


    num_human = 0


    new_G = nx.Graph()
    new_G.add_nodes_from(all_nodes)

    edges = []
    

    verifier_edges_sorted = sorted(verifier_edges, key=lambda x: -x[2])

    confirmed_nodes = set()

    for (n1, n2, s) in verifier_edges_sorted:
        if n1 > n2:
            n1, n2 = n2, n1

        if n1 in confirmed_nodes or n2 in confirmed_nodes:
            continue

        reviews, quit_flag = human_reviewer([(n1, n2)])
        
        if reviews and len(reviews) > 0:
            is_the_same = reviews[0][2]
            
            if is_the_same:
                edges.append((n1, n2))
                confirmed_nodes.add(n1)
        num_human += 1

    new_G.add_edges_from(edges)

    idx = 0
    clustering = dict()
    for cc in nx.connected_components(new_G):
        clustering[idx] = set(cc)
        idx += 1

    node2cid = ct.build_node_to_cluster_mapping(clustering)
    return clustering, node2cid, num_human


    
