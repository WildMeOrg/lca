
import cluster_tools as ct
import json
import networkx as nx
import logging

logger = logging.getLogger('lca')


class ClusterValidator(object):
    def __init__(self,
                 gt_clustering,
                 gt_node2cid):
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        self.G = nx.Graph()
        self.prev_num_human = 0


        """
        Generate the "reachable" ground truth, the obtainable
        result given simulated failures to match that could disconnect
        a correct match.
        """


        self.r_clustering = {}
        k = 0
        for cc in self.gt_clustering.values():
            H = self.G.subgraph(cc)
            prev_k = k
            for new_cc in nx.connected_components(H):
                self.r_clustering[k] = new_cc
                k += 1
            if k - prev_k > 1:
                logger.info('GT cluster %a split into %a ...' % (cc, k - prev_k))
                for i in range(prev_k, k):
                    logger.info('   %a' % self.r_clustering[i])
            else:
                logger.info('GT cluster %a is intact' % cc)
        self.r_node2cid = ct.build_node_to_cluster_mapping(self.r_clustering)

        print("clustering", self.r_clustering)


    def trace_start_human(self, clustering, node2cid):
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
            result = self.incremental_stats(
                0, clustering, node2cid, self.gt_clustering, self.gt_node2cid
            )
            self.gt_results = [result]
            # result = self.incremental_stats(
            #     0, clustering, node2cid, self.r_clustering, self.r_node2cid
            # )
            # self.r_results = [result]
            self.prev_num_human = 0

    def trace_iter_compare_to_gt(self, clustering, node2cid, num_human):
        if num_human <= self.prev_num_human:
            return
        result = self.incremental_stats(
            num_human, clustering, node2cid, self.gt_clustering, self.gt_node2cid
        )
        self.gt_results.append(result)
        # result = self.incremental_stats(
        #     num_human, clustering, node2cid, self.r_clustering, self.r_node2cid
        # )
        # self.r_results.append(result)
        self.prev_num_human = num_human


    def incremental_stats(
            self, num_human, clustering, node2cid, true_clustering, true_node2cid
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
        }

        logger.info('Incremental_stats: %s' % json.dumps(result, indent=4))
        return result


    