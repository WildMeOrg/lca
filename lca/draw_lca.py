# -*- coding: utf-8 -*-
"""
Draw the LCA clustering and graph. This works effectively for up to
about a dozen nodes. After that the drawing gets too cluttered.

All nodes within the same cluster are shown in the same randomly
generated color.

Positive weight edges within a cluster are shown in solid black
Negative weight edges within a cluster are shown in solid red.
Positive weight edges between clusters are shown in dotted black.
Negative weight edges between clusters are shown in dotted red.

Weights are drawn on an edges.
"""

import logging
import math as m
import matplotlib.pyplot as plt
import networkx as nx
import random


logger = logging.getLogger('lca')


def random_hex_rgb():
    """
    Generate a string of three random hex values in the range 0..255
    (00 to FF).
    """
    s = '#'
    for _ in range(3):
        s += '{:0>2}'.format(hex(random.randint(0, 255))[2:])
    return s


def add_random_offset(pos, delta):
    """
    Add a random uniform offset in the range [-delta, delta] to the x
    and y values of each position.
    """
    for e in pos:
        dx = random.uniform(-delta, delta)
        dy = random.uniform(-delta, delta)
        pos[e][0] += dx
        pos[e][1] += dy


class draw_lca(object):  # NOQA
    def __init__(self, prefix, ext='.png', max_iterations=1000):
        self.prefix = prefix
        n = m.ceil(m.log10(max_iterations))
        self.iter_format = '{:0>%d}' % n
        self.ext = ext
        self.max_iterations = max_iterations
        self.prev_nodes = set()
        self.pos = None
        self.cluster_clr = dict()

    def draw_iteration(self, G, clustering, node2cid, iter):
        # If new nodes, recompute their positions. For now use
        # circular_layout with a random offset.
        ns = set(G)
        if not (ns == self.prev_nodes):
            self.pos = dict()
            self.prev_nodes = ns
            self.pos = nx.circular_layout(G)
            add_random_offset(self.pos, 0.1)

        #  Show the nodes in each cluster with a different color. Generate
        #  new random colors for each cluster, but remember for future
        #  iterations.
        for cid in clustering:
            if cid not in self.cluster_clr:
                self.cluster_clr[cid] = random_hex_rgb()
            nx.draw_networkx_nodes(
                G,
                self.pos,
                nodelist=clustering[cid],
                alpha=0.5,
                node_color=self.cluster_clr[cid],
            )

        # Split the edges into pos / neg and within and between clusters
        pos_in, neg_in, pos_out, neg_out = [], [], [], []
        weight_labels = nx.get_edge_attributes(G, 'weight')
        for e, w in weight_labels.items():
            n0, n1 = e
            in_comp = node2cid[n0] == node2cid[n1]
            if in_comp and w > 0:
                pos_in.append(e)
            elif in_comp:
                neg_in.append(e)
            elif w > 0:
                pos_out.append(e)
            else:
                neg_out.append(e)
        nx.draw_networkx_edges(
            G, self.pos, edgelist=pos_in, edge_color='k', style='solid'
        )
        nx.draw_networkx_edges(
            G, self.pos, edgelist=neg_in, edge_color='r', style='solid'
        )
        nx.draw_networkx_edges(
            G, self.pos, edgelist=pos_out, edge_color='k', style='dotted'
        )
        nx.draw_networkx_edges(
            G, self.pos, edgelist=neg_out, edge_color='r', style='dotted'
        )

        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=weight_labels)
        nx.draw_networkx_labels(G, self.pos)

        # Create file name
        iter_str = self.iter_format.format(iter)
        out_name = self.prefix + '_' + iter_str + self.ext
        plt.savefig(out_name)
        plt.clf()

        logger.debug('Wrote iteration graph ' + out_name)
        logger.info(out_name)
