# -*- coding: utf-8 -*-
import networkx as nx
import logging

import cluster_tools as ct


logger = logging.getLogger('lca')



def baseline_clustering(all_nodes, verifier_edges, human_reviewer, threshold):

    high_threshold = 1 - threshold
    low_threshold = threshold

    clustering = {f"ct{n}":set([n]) for n in all_nodes}
    node2cid = {n:f"ct{n}" for n in all_nodes}

    def connect_nodes(n1, n2):
        if node2cid[n1] != node2cid[n2]:
            cluster = node2cid[n1]
            c2nodes = clustering.pop(node2cid[n2])
            clustering[cluster] = clustering[cluster].union(c2nodes)
            for n in c2nodes:
                node2cid[n] = cluster

    for_review = []

    for (n1, n2, s) in verifier_edges:
        if s > high_threshold:
            connect_nodes(n1, n2)
        elif s > low_threshold:
            for_review.append((n1,n2))
    num_human = len(for_review)
    reviews, quit = human_reviewer(for_review)
    for (n1, n2, s) in reviews:
        if s:
            connect_nodes(n1, n2)
    return clustering, node2cid, num_human



    
