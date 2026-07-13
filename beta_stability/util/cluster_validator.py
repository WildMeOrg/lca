
import beta_stability.util.cluster_tools as ct
import json
import networkx as nx
import logging
import os
from beta_stability.util.tools import SetEncoder

logger = logging.getLogger("beta_stability")


class ClusterValidator(object):
    def __init__(self,
                 gt_clustering,
                 gt_node2cid,
                 metrics_file=None,
                 metrics_metadata=None):
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        self.prev_num_human = 0
        self.gt_results = []
        self.r_results = []
        self.metrics_file = metrics_file
        self.metrics_metadata = metrics_metadata or {}

        if self.metrics_file:
            metrics_dir = os.path.dirname(self.metrics_file)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            # Start each configured run with a fresh metrics file.
            open(self.metrics_file, 'w').close()

        


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
            # if k - prev_k > 1:
                # logger.info('GT cluster %a split into %a ...' % (cc, k - prev_k))
                # for i in range(prev_k, k):
                #     logger.info('   %a' % r_clustering[i])
            # else:
                # logger.info('GT cluster %a is intact' % cc)
        r_node2cid = ct.build_node_to_cluster_mapping(r_clustering)

        return r_clustering, r_node2cid


    def trace_start_human(self, clustering, node2cid, G, num_human=0):
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
                num_human, clustering, node2cid, self.gt_clustering, self.gt_node2cid, info_text
            )
            self.gt_results = [result]

            r_clustering, r_node2cid = self.create_reachable(G)
            info_text = 'Reachable stats'
            result = self.incremental_stats(
                num_human, clustering, node2cid, r_clustering, r_node2cid, info_text
            )
            self.r_results = [result]
            self.prev_num_human = num_human

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
            num_human, clustering, node2cid, r_clustering, r_node2cid, info_text
        )
        self.r_results.append(result)
        self.prev_num_human = num_human


    def incremental_stats(
            self, num_human, clustering, node2cid, true_clustering, true_node2cid, info_text="Incremental stats"
        ):
        frac, prec, rec, per_size, non_equal_clustering, f1 = ct.percent_and_PR(
            clustering, node2cid, true_clustering, true_node2cid
        )

        # Hungarian matching-based metrics (softer evaluation)
        hungarian = ct.hungarian_cluster_matching(clustering, true_clustering)

        result = {
            'num human': num_human,
            'num clusters': len(clustering),
            'num true clusters': len(true_clustering),

            # Hungarian matching metrics (cluster-level)
            'Hungarian precision': hungarian['precision'],
            'Hungarian recall': hungarian['recall'],
            'Hungarian f1 score': hungarian['f1'],
            # 'hungarian_tp': hungarian['tp'],
            # 'hungarian_fp': hungarian['fp'],
            # 'hungarian_fn': hungarian['fn'],

            'frac correct': frac,
            'precision': prec,
            'recall': rec,
            'error_rate': 1 - frac,
            'f1 score': f1,
            
            # 'per size': per_size,
            # 'non equal': non_equal_clustering,
            # 'current clustering': clustering
        }

        logger.info(f'{info_text}: {json.dumps(result, indent=4, cls=SetEncoder)}')
        if info_text == 'Basic stats':
            self._write_metrics_row(result)
        return result

    def _write_metrics_row(self, result):
        if not self.metrics_file:
            return

        row = {
            **self.metrics_metadata,
            'reviews': result['num human'],
            'h_f1': result['Hungarian f1 score'],
            'pcc_f1': result['f1 score'],
            'h_precision': result['Hungarian precision'],
            'h_recall': result['Hungarian recall'],
            'pcc_precision': result['precision'],
            'pcc_recall': result['recall'],
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(row, cls=SetEncoder) + '\n')


    
