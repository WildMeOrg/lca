# -*- coding: utf-8 -*-
import logging

from wbia_lca import cluster_tools as ct
from wbia_lca import edge_generator


logger = logging.getLogger('wbia_lca')


class edge_generator_sim(edge_generator.edge_generator):  # NOQA
    def __init__(
        self,
        db,
        wgtr,
        prob_quads=[],
        human_triples=[],
        gt_clusters=[],
        nodes_to_remove=[],
        delay_steps=0,
    ):
        super().__init__(db, wgtr)
        self.edge_dict = {
            (n0, n1, aug): wgtr.wgt(prob) for n0, n1, prob, aug in prob_quads
        }
        self.edge_dict.update(
            {(n0, n1, 'human'): wgtr.human_wgt(b) for n0, n1, b in human_triples}
        )
        self.gt_clustering = {cid: c for cid, c in enumerate(gt_clusters)}
        self.node_to_gt_cid = ct.build_node_to_cluster_mapping(self.gt_clustering)
        self.nodes_to_remove = nodes_to_remove
        self.delay_steps = delay_steps
        self.steps_remain = delay_steps

    def edge_request_cb_async(self):
        pass

    def edge_result_cb(self, node_set=None):
        """
        Extract the edges (quads) from the results that are part of the weight list.
        """
        if self.steps_remain > 0:
            # logger.info("skipping")
            self.steps_remain -= 1
            return []

        self.steps_remain = self.delay_steps

        edge_quads = []
        for tr in self.edge_requests:
            n0, n1, aug = tr
            if node_set is not None and (n0 not in node_set or n1 not in node_set):
                logger.warning(
                    'At least of node pair (%a,%a) is not in node set' % (n0, n1)
                )
                continue
            if tr in self.edge_dict:
                w = self.edge_dict[tr]
                logger.info(
                    'Adding hand-specified edge (%a, %a, %a, %a)' % (n0, n1, w, aug)
                )
                edge_quads.append((n0, n1, w, aug))
            elif n0 not in self.node_to_gt_cid or n1 not in self.node_to_gt_cid:
                logger.warning(
                    'Edge triple (%a, %a, %a) contains unknown node(s); returning wgt 0'
                    % (n0, n1, aug)
                )
                edge_quads.append((n0, n1, 0, aug))
            else:
                same_clustering = self.node_to_gt_cid[n0] == self.node_to_gt_cid[n1]
                if aug == 'human':
                    w = self.wgtr.human_random_wgt(same_clustering)
                else:
                    w = self.wgtr.random_wgt(same_clustering)
                logger.info(
                    'Adding auto-generated edge (%a, %a, %a, %a)' % (n0, n1, w, aug)
                )
                edge_quads.append((n0, n1, w, aug))

        self.edge_requests.clear()
        self.db.add_edges(edge_quads)
        return edge_quads

    def remove_nodes_cb(self, node_set):
        """
        For each node to be removed from the node_set, add it to list of nodes to be
        removed. Return this list.
        """
        to_remove = []
        if self.steps_remain == 0:
            for n in self.nodes_to_remove:
                if n in node_set:
                    to_remove.append(n)
                    self.db.remove_node(n)
            self.nodes_to_remove.clear()
        return to_remove
