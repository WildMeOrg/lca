# -*- coding: utf-8 -*-

# import exp_scores as es
# import weighter
import logging


logger = logging.getLogger('lca')


class edge_generator(object):  # NOQA
    def __init__(self, db, wgtr):
        self.db = db
        self.wgtr = wgtr
        self.quit = False

        self.edge_requests = []  # triples (n0, n1, aug_name)
        self.edge_results = []  # quads (n0, n1, w, aug_name)
        self.nodes_to_remove = []
        self.active_clusters = {}

    def wgt_from_verifier(self, p, vn):
        if vn == 'zero':
            return 0
        else:
            return self.wgtr.wgt(p)

    def new_edges_from_verifier(self, verifier_quads, db_add=True):
        edge_quads = [
            (n0, n1, self.wgt_from_verifier(p, vn), vn)
            for n0, n1, p, vn in verifier_quads
        ]
        if db_add:
            self.db.add_edges(edge_quads)
        return edge_quads

    def new_edges_from_human(self, human_triples):
        edge_quads = [
            (n0, n1, self.wgtr.human_wgt(b), 'human') for n0, n1, b in human_triples
        ]
        self.db.add_edges(edge_quads)
        return edge_quads

    def edge_request_cb(self, req_list):
        self.edge_requests += req_list
        """
        Non-blocking
        Some MAGIC NOW HAPPENS to turn these into results, with calls to
        the appropriate verification algorithms or human decision manager.
        """
        self.edge_request_cb_async()

    def edge_result_cb(self, node_set=None):
        """
        Extract the edges (quads) from the results that are part of the weight list.
        """
        quads_remaining = []
        quads_to_return = []
        for quad in self.edge_results:
            n0, n1 = quad[0], quad[1]
            if node_set is None or (n0 in node_set and n1 in node_set):
                quads_to_return.append(quad)
            else:
                quads_remaining.append(quad)
        self.edge_results = quads_remaining
        return quads_to_return, self.quit
    
    def save_active_clusters_cb(self, ccpic_id, clustering):
        """
        Save active clusters
        """
        self.active_clusters[ccpic_id] = clustering
        return 

    def remove_nodes_cb(self, node_set):
        """
        For each node to be removed from the node_set, add it to list of nodes to be
        removed. Return this list.
        """
        to_remove = []
        for n in self.nodes_to_remove:
            if n in node_set:
                to_remove.append(n)
                self.db.remove_node(n)
        self.nodes_to_remove.clear()
        return to_remove

    def edge_request_cb_async(self):
        raise NotImplementedError()
