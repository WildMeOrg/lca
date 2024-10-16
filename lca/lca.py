# -*- coding: utf-8 -*-
import logging
import math
import random as rnd
import networkx as nx

import cluster_tools as ct
import test_cluster_tools as tct


logger = logging.getLogger('lca')


"""
Additional testing and other notes:

1. Make sure that it handles single clusters in the cid list; fails if this
single cluster has only one node.

2. Change the to_clusters to make it a dictionary as well for
internal consistency.

3. Careful to avoid issues with having the operator values reversed.

4. Pass the graph as argument to the constructor

5. Add code to remember what list a candidate is on.

clustering is a mapping from a cluster id to a set of node ids.
"""

g_cluster_counter = 0


class LCA(object):
    def __init__(self, subG, clustering, cids, score):
        self.subgraph = subG  # Restricted to the clustering
        self.from_clusters = {c: clustering[c] for c in cids}
        self.from_cids_sorted = tuple(sorted(cids))
        self.__hash_value = hash(self.from_cids_sorted)
        self.from_n2c = ct.build_node_to_cluster_mapping(self.from_clusters)
        self.from_score = score
        self.to_clusters = None
        self.to_score = None
        self.to_n2c = None
        self.inconsistent = []

    def __hash__(self):
        return self.__hash_value

    def from_cids(self):
        return self.from_cids_sorted

    def from_node2cid(self):
        return self.from_n2c

    def nodes(self):
        return set.union(*self.from_clusters.values())

    def set_to_clusters(self, to_clusters, to_score):
        self.to_clusters = to_clusters
        self.to_score = to_score
        self.to_n2c = ct.build_node_to_cluster_mapping(self.to_clusters)
        self.inconsistent = []

    def delta_score(self):
        return self.to_score - self.from_score

    def get_inconsistent(self, num_to_return, is_futile_tester):
        # If the inconsistent list is empty, which can happen either
        # at the start if it is has been exhausted, then (re)generate
        # it. This function is quadratic in the number of nodes in a
        # cluster, but this should not be too much of a burden since
        # the clusters will tend to be small (even if an animal has
        # been seen frequently).
        if len(self.inconsistent) == 0:
            nodes = sorted(self.nodes())
            no_edge = []
            has_edge = []
            for i, m in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    n = nodes[j]

                    # Skip node pairs that are categorized the same
                    # between the two clusterings - either in the same
                    # clusters in both or in different clusters in
                    # both
                    same_in_from = self.from_n2c[m] == self.from_n2c[n]
                    same_in_to = self.to_n2c[m] == self.to_n2c[n]
                    if same_in_from == same_in_to:
                        continue
                    if is_futile_tester(m, n):
                        logger.info('Edge %a, %a skipped: too many tests' % (m, n))
                        continue

                    if n in self.subgraph[m]:  # edge exists
                        has_edge.append((m, n, self.subgraph[m][n]['weight']))
                    else:
                        no_edge.append((m, n))

            has_edge.sort(key=lambda e: abs(e[2]))
            self.inconsistent = [(m, n) for m, n, _ in has_edge] + no_edge

        #  Return the last num_to_return - the highest priority pairs
        new_len = max(0, len(self.inconsistent) - num_to_return)
        ret_edges = self.inconsistent[new_len:]
        self.inconsistent = self.inconsistent[:new_len]
        return ret_edges

    def get_score_change(self, delta_w, n0_cid, n1_cid):
        if n0_cid == n1_cid:
            return delta_w
        else:
            return -delta_w

    def is_singleton(self):
        return len(self.from_cids_sorted) == 1

    def add_edge(self, e):
        """
        Do not change weight here because the graph aliases the overall
        graph.  Assume the calling function makes this change.
        Also assume that e[0] < e[1]
        """
        n0, n1, wgt = e

        #  Update from score
        n0_cid = self.from_n2c[n0]
        n1_cid = self.from_n2c[n1]
        from_score_change = self.get_score_change(wgt, n0_cid, n1_cid)
        self.from_score += from_score_change

        """The to_clusters may not yet exist, which could occur if this LCA
        has just been created.  In this case, there is nothing more to
        do and we can safely return 0 for the to_delta score because
        this LCA will already be on the scoring queue and will be left there.
        """
        if self.to_n2c is None:
            return (from_score_change, 0)

        n0_cid = self.to_n2c[n0]
        n1_cid = self.to_n2c[n1]
        to_score_change = self.get_score_change(wgt, n0_cid, n1_cid)
        self.to_score += to_score_change

        # Remove the added edge from the inconsistent list. It will be
        # added back when the list is regenerated for the next round
        # of augmentation.
        try:
            self.inconsistent.remove((n0, n1))
        except ValueError:
            pass

        # Finally, return the score changes
        return (from_score_change, to_score_change)

    def densify_singleton(self, params):
        if len(self.from_cids_sorted) != 1:
            return []
        nodes = sorted(self.nodes())
        missing = []
        for i, m in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                n = nodes[j]
                if n not in self.subgraph[m]:
                    missing.append((m, n))

        if len(missing) == 0:
            return []

        max_ne = len(nodes) * (len(nodes) - 1) // 2
        curr_ne = max_ne - len(missing)
        targ_ne = max(
            params['densify_min_edges'], math.ceil(params['densify_frac'] * max_ne)
        )
        num_to_add = max(0, targ_ne - curr_ne)

        if num_to_add < len(missing):
            rnd.shuffle(missing)
            missing = missing[:num_to_add]
        return missing

    def pprint_short(self, initial_str='', stop_after_from=False):
        out_str = initial_str + 'From cid(s) and cluster(s):'
        for cid in sorted(self.from_clusters.keys()):
            out_str += '  %s: %a' % (cid, sorted(self.from_clusters[cid]))

        if logger.getEffectiveLevel() <= logging.DEBUG:
            check_score = ct.clustering_score(self.subgraph, self.from_n2c)
            if abs(check_score - self.from_score) > 1e-7:
                out_str += 'from score error: should be %d, but is %d' % (
                    check_score,
                    self.from_score,
                )
        if stop_after_from or self.to_clusters is None:
            logger.info(out_str)
            return

        out_str += '\nTo cluster(s):'
        for cid in sorted(self.to_clusters.keys()):
            out_str += '  %a' % sorted(self.to_clusters[cid])

        if logger.getEffectiveLevel() <= logging.DEBUG:
            check_score = ct.clustering_score(self.subgraph, self.to_n2c)
            if check_score != self.to_score:
                out_str += '\nto score error: should be %d, but is %d\n' % (
                    check_score,
                    self.to_score,
                )
        out_str += '\ndelta_score %a' % self.delta_score()

        logger.info(out_str)

    def pprint(self, stop_after_from=False):
        logger.info('from_n2c: %s' % (self.from_n2c,))
        logger.info('subgraph nodes %s' % (self.subgraph.nodes(),))
        check_score = ct.clustering_score(self.subgraph, self.from_n2c)
        logger.info(
            'from clusters (score = %a, checking %a):' % (self.from_score, check_score)
        )
        if self.from_score != check_score:
            logger.info('lca: SCORING ERROR in from')
        for cid in sorted(self.from_clusters.keys()):
            logger.info('    %s: %a' % (cid, self.from_clusters[cid]))
        if stop_after_from:
            return

        check_score = ct.clustering_score(self.subgraph, self.to_n2c)
        logger.info(
            'to clusters (score = %a, checking = %a):' % (self.to_score, check_score)
        )
        if self.to_score != check_score:
            logger.info('SCORING ERROR in to')
        for cid in sorted(self.to_clusters.keys()):
            logger.info('    %d: %a' % (cid, self.to_clusters[cid]))
        logger.info('score_difference %a' % self.delta_score())

        logger.info('inconsistent_pairs: %s' % (self.inconsistent,))


# ################################
# ######    Testing code    ######
# ################################


def build_example_LCA():
    G = tct.ex_graph_fig1()
    n2c_optimal = {
        'a': 0,
        'b': 0,
        'd': 0,
        'e': 0,
        'c': 1,
        'h': 2,
        'i': 2,
        'f': 3,
        'g': 3,
        'j': 3,
        'k': 3,
    }
    clustering_opt = ct.build_clustering(n2c_optimal)
    cid0 = 2
    cid1 = 3
    nodes_in_clusters = list(clustering_opt[2] | clustering_opt[3])
    subG = G.subgraph(nodes_in_clusters)

    score = ct.cid_list_score(subG, clustering_opt, n2c_optimal, [cid0, cid1])
    a = LCA(subG, clustering_opt, [cid0, cid1], score)

    to_clusters = {0: {'f', 'h', 'i', 'j'}, 1: {'g', 'k'}}
    subG = G.subgraph(nodes_in_clusters)
    to_node2cid = {n: cid for cid in range(len(to_clusters)) for n in to_clusters[cid]}
    to_score = ct.clustering_score(subG, to_node2cid)
    a.set_to_clusters(to_clusters, to_score)

    return a, G


def futile_tester_default(n0, n1):
    return False


def test_LCA_class():
    logger.info('===========================')
    logger.info('=====  test_LCA_class =====')
    logger.info('===========================')

    a, G = build_example_LCA()

    logger.info('a.from_cids should return [2, 3]; returns %s' % (sorted(a.from_cids()),))
    logger.info(
        "a.nodes should return ['f', 'g', 'h', 'i', 'j', 'k']; returns %s"
        % (
            sorted(
                a.nodes(),
            )
        )
    )

    logger.info('a.delta_score should be -18 and it is %s' % (a.delta_score(),))

    logger.info('Running pprint')
    a.pprint()

    logger.info('Running pprint_short')
    a.pprint_short()

    n = 3
    logger.info('-------')
    logger.info('1st call to get_inconsistent should return (f, g), (f, h), (h, j):')
    prs = a.get_inconsistent(n, futile_tester_default)
    logger.info(prs)

    logger.info('-------')
    logger.info('2nd call to get_inconsistent should return (g, j), (i, j), (j, k)')
    prs = a.get_inconsistent(n, futile_tester_default)
    logger.info(prs)

    logger.info('-------')
    logger.info('3rd call to get_inconsistent should return (f, i), (f, k)')
    prs = a.get_inconsistent(n, futile_tester_default)
    logger.info(prs)

    logger.info('-------')
    logger.info('4th call should regenerate and return (f, h), (h, j)')
    n = 2
    prs = a.get_inconsistent(n, futile_tester_default)
    logger.info(prs)
    logger.info(
        'At this point, the inconsistent list should be length 6. It is %a'
        % len(a.inconsistent),
    )
    logger.info(
        'The first pair on the inconsistent list should be (f, i) and is'
        + str(a.inconsistent[0]),
    )
    logger.info(
        'The last pair on the inconsistent list should be (f, g) and is'
        + str(a.inconsistent[-1]),
    )


def test_LCA_add_edge_method():
    logger.info('')
    logger.info('==============================')
    logger.info('====  test LCA.add_edge  =====')
    logger.info('==============================')

    a, G = build_example_LCA()
    a.pprint()

    logger.info('Changing an existing edge')
    change_edge = tuple(['i', 'j', 3])
    (from_change, to_change) = a.add_edge(change_edge)
    logger.info(
        f'Change edge: {change_edge} delta_wgt should be (-3, 3) '
        f'and is ({from_change}, {to_change})'
    )
    logger.info('a.delta_score should be -12 and it is %s' % (a.delta_score(),))
    G['i']['j']['weight'] += change_edge[2]

    logger.info('--------------')
    change_edge = tuple(['i', 'j', -3])
    (from_change, to_change) = a.add_edge(change_edge)
    logger.info(
        'Changing back by adding:' + f'{change_edge}' + ' delta_wgt should be (3, -3)'
        ' and is (%d, %d)' % (from_change, to_change),
    )
    logger.info('a.delta_score should be back to -18 and it is %s' % (a.delta_score(),))
    G['i']['j']['weight'] += change_edge[2]

    logger.info('--------------')
    logger.info('Adding a new edge')
    change_edge = tuple(['f', 'h', 4])
    (from_change, to_change) = a.add_edge(change_edge)
    logger.info(
        'Change edge:' + f'{change_edge}' + 'delta_wgt should be (-4, 4)'
        ' and is (%d, %d)' % (from_change, to_change),
    )
    logger.info('a.delta_score should be -10 and it is %s' % (a.delta_score(),))
    G.add_edge('f', 'h', weight=change_edge[2])

    logger.info('-------')
    logger.info('Adding a change to an existing, consistent edge')
    change_edge = tuple(['h', 'i', 9])
    (from_change, to_change) = a.add_edge(change_edge)
    logger.info(
        'Change edge:' + f'{change_edge}' + 'delta_wgt should be (9, 9)'
        ' and is (%d, %d)' % (from_change, to_change),
    )
    logger.info('a.delta_score should still be -10 and it is %s' % (a.delta_score(),))
    G['h']['i']['weight'] += change_edge[2]

    logger.info('-------')
    logger.info('Restarting tests for adding edges during augmentation')
    a, G = build_example_LCA()
    a.pprint()
    n = 2
    prs = a.get_inconsistent(n, futile_tester_default)
    logger.info(
        f'First check of get_inconsistent: prs should be [(f, h), (h,j)] and are {prs}'
    )
    logger.info(
        'Length of inconsistent_pairs should be 6 and is %s' % (len(a.inconsistent),)
    )

    n = 3
    delta_score_pre = a.delta_score()
    ce = tuple(['h', 'j', 10])
    (from_change, to_change) = a.add_edge(ce)
    logger.info(
        'Change edge:' + f'{ce}' + 'delta_wgt should be (-10, 10)'
        ' and is (%d, %d)' % (from_change, to_change),
    )
    logger.info(
        'a.delta_score be %d and it is %d' % (delta_score_pre + 20, a.delta_score())
    )
    G.add_edge(ce[0], ce[1], weight=ce[2])
    pr = ce[:2]
    logger.info(
        f'Pair {pr} should not be on the inconsistent list. Is it? {pr in a.inconsistent}'
    )

    ce = tuple(['f', 'k', -6])
    logger.info(
        'Before add_edge pr'
        + f'{ce[:2]}'
        + 'should be on a.inconsistent. Is it? '
        + f'{ce[:2] in a.inconsistent}',
    )
    delta_score_pre = a.delta_score()
    (from_change, to_change) = a.add_edge(ce)
    logger.info(
        'Change edge:' + f'{ce}' + 'delta_wgt should be (-6, 6)'
        ' and is (%d, %d)' % (from_change, to_change),
    )
    logger.info(
        'a.delta_score be %d and it is %d' % (delta_score_pre + 12, a.delta_score())
    )
    G.add_edge(ce[0], ce[1], weight=ce[2])
    logger.info(
        f'Pair {pr} should not be in the inconsistent list and result is'
        f'{str(ce[:2] in a.inconsistent)}',
    )


class futile_wrapper(object):  # NOQA
    def __init__(self, edge_counts, thresh):
        self.edge_counts = edge_counts
        self.count_thresh = thresh

    def is_futile_tester(self, m, n):
        pr = (m, n)
        if pr not in self.edge_counts:
            return False
        else:
            return self.edge_counts[pr] >= self.count_thresh


def test_futility_check():
    subG = nx.Graph()
    subG.add_weighted_edges_from([('a', 'b', 4), ('b', 'c', -1)])
    from_clustering = {9: set(['a', 'b', 'c'])}
    from_cids = [9]
    from_score = 5
    a = LCA(subG, from_clustering, from_cids, from_score)
    to_clustering = {0: set(['a', 'b']), 1: set(['c'])}
    to_score = 3
    a.set_to_clusters(to_clustering, to_score)

    edge_counts = {('a', 'b'): 1, ('b', 'c'): 3, ('a', 'c'): 4}
    futile_thresh = 4
    fw = futile_wrapper(edge_counts, futile_thresh)
    num_to_return = 3
    prs = a.get_inconsistent(num_to_return, fw.is_futile_tester)
    logger.info('***********')
    logger.info('Testing get_inconsistent with futility check on edges.')
    logger.info(
        'For simple three-node graph the non-futile, inconsistent pairs'
        "should be just [('b', 'c')] and is: %a " % prs,
    )


def test_densify_singleton():
    logger.info('***********')
    logger.info('Testing densify_singleton')
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 3),
            ('a', 'c', -2),
            ('a', 'd', 4),
            ('b', 'c', 5),
            ('b', 'd', -1),
            ('c', 'd', 8),
        ]
    )
    clustering = {0: set(['a', 'b', 'c', 'd'])}
    n2c = ct.build_node_to_cluster_mapping(clustering)
    cids = list(clustering.keys())
    score = ct.clustering_score(G, n2c)
    a = LCA(G, clustering, cids, score)
    params = {'densify_min_edges': 2, 'densify_frac': 1}
    to_add = a.densify_singleton(params)
    logger.info(
        'adding to complete graph should return length 0; ' f'result is: {len(to_add)}'
    )

    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 3),
            ('a', 'c', -2),
            ('a', 'd', 4),
            ('b', 'e', 5),
            ('c', 'e', -1),
            ('c', 'f', 8),
            ('d', 'f', 6),
        ]
    )

    clustering = {0: set(['a', 'b', 'c']), 1: set(['d', 'e', 'f'])}
    n2c = ct.build_node_to_cluster_mapping(clustering)
    cids = list(clustering.keys())
    score = ct.clustering_score(G, n2c)
    a = LCA(G, clustering, cids, score)
    params = {'densify_min_edges': 2, 'densify_frac': 1}
    to_add = a.densify_singleton(params)
    logger.info('result should be [] because not a singleton;' f' answer is {to_add}')

    clustering = {0: set(['a', 'b', 'c', 'd', 'e', 'f'])}
    n2c = ct.build_node_to_cluster_mapping(clustering)
    cids = list(clustering.keys())
    score = ct.clustering_score(G, n2c)
    a = LCA(G, clustering, cids, score)
    params = {'densify_min_edges': 2, 'densify_frac': 1}
    to_add = a.densify_singleton(params)
    logger.info(
        'parameter densify_frac = 1 should cause a returned length 8; '
        f'result is {len(to_add)}'
    )
    logger.info(f'here are the added edges: {to_add}')

    params['densify_min_edges'] = 10
    params['densify_frac'] = 0.5
    to_add = a.densify_singleton(params)
    logger.info(
        'parameter densify_min_edges '
        ' is %d should add length 3; result: %d'
        % (params['densify_min_edges'], len(to_add))
    )
    logger.info(f'here are the added edges: {to_add}')

    params['densify_min_edges'] = 5
    params['densify_frac'] = 0.75
    to_add = a.densify_singleton(params)
    logger.info(
        'parameter densify_frac %.2f should add length 5; result: %d'
        % (params['densify_frac'], len(to_add))
    )
    logger.info(f'here are the added edges: {to_add}')


if __name__ == '__main__':
    test_LCA_class()
    test_LCA_add_edge_method()
    test_futility_check()
    test_densify_singleton()
