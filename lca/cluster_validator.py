
import cluster_tools as ct
import json
import networkx as nx
import logging
import os

logger = logging.getLogger('lca')


class ClusterValidator(object):
    def __init__(self,
                 gt_clustering,
                 gt_node2cid):
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        self.prev_num_human = 0

        


        """
        Generate the "reachable" ground truth, the obtainable
        result given simulated failures to match that could disconnect
        a correct match.
        """
    

    def create_reachable(self, G):
        r_clustering = {}
        k = 0
        for cc in self.gt_clustering.values():
            H = G.subgraph(cc)
            prev_k = k
            for new_cc in nx.connected_components(H):
                r_clustering[k] = new_cc
                k += 1
            if k - prev_k > 1:
                logger.info('GT cluster %a split into %a ...' % (cc, k - prev_k))
                for i in range(prev_k, k):
                    logger.info('   %a' % r_clustering[i])
            else:
                logger.info('GT cluster %a is intact' % cc)
        r_node2cid = ct.build_node_to_cluster_mapping(r_clustering)

        return r_clustering, r_node2cid


    def trace_start_human(self, clustering, node2cid, G):
            """
            Beging to record information about the number of human decisions
            vs. the accuracy of the current clustering.  The comparison is
            made against both the ground truth clustering and the "reachable"
            ground truth clustering. For each new number of
            human decisions, we record (1) this number, (2) the number of
            ground truth clusters (fixed value), (3) the number of current
            clusters, (4) the fraction of current clusters that are
            exactly correct, (5) the precision and (6) the recall.  The
            same thing will be done for the "reachable" clusters.
            """
            info_text = 'Basic stats'
            result = self.incremental_stats(
                0, clustering, node2cid, self.gt_clustering, self.gt_node2cid, info_text
            )
            self.gt_results = [result]

            r_clustering, r_node2cid = self.create_reachable(G)
            info_text = 'Reachable stats'
            result = self.incremental_stats(
                0, clustering, node2cid, r_clustering, r_node2cid, info_text
            )
            self.r_results = [result]
            self.prev_num_human = 0

    def trace_iter_compare_to_gt(self, clustering, node2cid, num_human, G):
        if num_human <= self.prev_num_human:
            return
        info_text = 'Basic stats'
        result = self.incremental_stats(
            num_human, clustering, node2cid, self.gt_clustering, self.gt_node2cid, info_text
        )
        self.gt_results.append(result)
        r_clustering, r_node2cid = self.create_reachable(G)
        info_text = 'Reachable stats'
        result = self.incremental_stats(
            num_human, r_clustering, r_node2cid, self.gt_clustering, self.gt_node2cid, info_text
        )
        self.r_results.append(result)
        self.prev_num_human = num_human


    def incremental_stats(
            self, num_human, clustering, node2cid, true_clustering, true_node2cid, info_text="Incremental stats:"
        ):
        frac, prec, rec = ct.percent_and_PR(
            clustering, node2cid, true_clustering, true_node2cid
        )
        result = {
            'num human': num_human,
            'num clusters': len(clustering),
            'num true clusters': len(true_clustering),
            'frac correct': frac,
            'precision': prec,
            'recall': rec,
            'error_rate': 1- frac
        }

        logger.info(f'{info_text}: {json.dumps(result, indent=4)}')
        return result


    