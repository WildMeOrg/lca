# -*- coding: utf-8 -*-
import networkx as nx
import logging

import cluster_tools as ct


logger = logging.getLogger('lca')


"""

1. For each edge, generate the simulation result as though it is
human review and add to a dictionary.

2. Sort the edges by abs weight.

3. Make the decision about human labeling

4. For each number of human decisions in the incremented list, form
the graph, do connected components labeling, and analyze the
structure.  Add to the accumulated results.

5. Generate final statistics and plots

"""


class baseline(object):  # NOQA
    def __init__(self, sim):
        self.sim = sim
        self.nodes = sim.G_orig.nodes
        # logger.info("\n========")
        # logger.info("In baseline.__init__:")
        edges = [e for e in sim.G_orig.edges.data('weight')]
        edges = [(min(e[0], e[1]), max(e[0], e[1]), e[2]) for e in edges]
        self.edges_by_abs_wgt = sorted(edges, key=lambda e: abs(e[2]))
        # logger.info("edges_by_abs_wgt:", self.edges_by_abs_wgt)
        prs = [(min(e[0], e[1]), max(e[0], e[1])) for e in edges]
        self.dict_human = {pr: sim.gen_human_wgt(pr) for pr in prs}
        # logger.info("dict_human:", self.dict_human)
        self.gt_results = []
        self.r_results = []

    def one_iteration(self, num_human):
        orig_edges = [e for e in self.edges_by_abs_wgt[num_human:] if e[2] > 0]
        human_prs = [(e[0], e[1]) for e in self.edges_by_abs_wgt[:num_human]]
        human_edges = [(pr[0], pr[1], self.dict_human[pr]) for pr in human_prs]
        human_edges = [e for e in human_edges if e[2] > 0]
        # logger.info("\n--------")
        # logger.info("orig_edges:", orig_edges)
        # logger.info("human_edges:", human_edges)
        edges = orig_edges + human_edges
        new_G = nx.Graph()
        new_G.add_nodes_from(self.nodes)
        new_G.add_weighted_edges_from(edges)

        idx = 0
        clustering = dict()
        for cc in nx.connected_components(new_G):
            # logger.info("idx =", idx, "cc =", list(cc))
            clustering[idx] = set(cc)
            idx += 1

        node2cid = ct.build_node_to_cluster_mapping(clustering)
        return clustering, node2cid

    def all_iterations(self, n_min, n_max, n_inc):
        for n in range(n_min, n_max + 1, n_inc):
            clustering, node2cid = self.one_iteration(n)
            result = self.sim.incremental_stats(
                n, clustering, node2cid, self.sim.gt_clustering, self.sim.gt_node2cid
            )
            self.gt_results.append(result)
            result = self.sim.incremental_stats(
                n, clustering, node2cid, self.sim.r_clustering, self.sim.r_node2cid
            )
            self.r_results.append(result)

    def generate_plots(self, out_prefix):
        out_name = out_prefix + '_gt.pdf'
        self.sim.plot_convergence(self.gt_results, out_name)
        out_name = out_prefix + '_reach_gt.pdf'
        self.sim.plot_convergence(self.r_results, out_name)
