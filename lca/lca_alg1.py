# -*- coding: utf-8 -*-
import networkx as nx
import logging

from wbia_lca import cluster_tools as ct
from wbia_lca import test_cluster_tools as tct


logger = logging.getLogger('wbia_lca')


def best_shift(n0, n1, G, clustering, node2cid, trace_on=False):
    c0_id, c1_id = node2cid[n0], node2cid[n1]
    c0, c1 = clustering[c0_id], clustering[c1_id]
    n01_nodes = list(c0 | c1)
    H = G.subgraph(n01_nodes)
    node2cid_H = {n: node2cid[n] for n in n01_nodes}
    delta_s = 0
    frontier = set()

    if trace_on:
        logger.info('===========================')
        logger.info('Running best_shift:  n0 = %s, n1 = %s' % (n0, n1))

    for m in H[n0]:
        w = H[n0][m]['weight']
        if node2cid_H[m] == c0_id:
            delta_s -= 2 * w
            frontier.add(m)
        else:
            delta_s += 2 * w
    shift_set = {n0}

    if trace_on:
        logger.info('Initial values:')

    while len(frontier) > 0 and len(shift_set) < len(c0) - 1:
        if trace_on:
            logger.info('delta_s: %s' % (delta_s,))
            logger.info('shift_set: %s' % (shift_set,))
            logger.info('frontier: %s' % (frontier,))
        best_delta = delta_s
        best_node = None
        for m in frontier:
            new_delta = delta_s
            for m1 in H[m]:
                if m1 in c1 or m1 in shift_set:
                    new_delta += 2 * H[m][m1]['weight']  # 2* since neg -> pos
                else:
                    new_delta -= 2 * H[m][m1]['weight']  # 2* since pos -> neg
            if trace_on:
                logger.info('m %s, new_delta %a' % (m, new_delta))
            if new_delta > best_delta:
                best_delta = new_delta
                best_node = m

        if trace_on:
            logger.info('.......')
        if best_node is None:
            if trace_on:
                logger.info('best_node is None')
            break

        frontier.remove(best_node)
        shift_set.add(best_node)
        delta_s = best_delta
        for m in H[best_node]:
            if node2cid_H[m] == c0_id and m not in shift_set:
                frontier.add(m)

    if trace_on:
        logger.info('============')
    return delta_s, shift_set


def lca_alg1(curr_G, stop_at_two=False, trace_on=False):
    if len(curr_G) == 0:
        return {}, 0
    elif len(curr_G) == 1:
        clustering = {0: set(curr_G.nodes())}
        return clustering, 0

    neg_edges, pos_edges = ct.get_weight_lists(curr_G, sort_positive=True)
    clustering = {c: {n} for c, n in enumerate(sorted(curr_G.nodes()))}
    node2cid = ct.build_node_to_cluster_mapping(clustering)

    G_prime = nx.Graph()
    G_prime.add_nodes_from(curr_G)
    G_prime.add_weighted_edges_from(neg_edges)
    score = ct.clustering_score(G_prime, node2cid)

    if trace_on:
        logger.info('====================')
        logger.info('====  lca_alg1  ====')
        logger.info('====================')
        ct.print_structures(G_prime, clustering, node2cid, score)

    for e in pos_edges:
        if trace_on:
            logger.info('=======================')
            logger.info('Start of next iteration')
            logger.info('=======================')
        if e[0] < e[1]:
            n0, n1 = e[0], e[1]
        else:
            n1, n0 = e[0], e[1]
        wgt = e[2]
        n0_cid, n1_cid = node2cid[n0], node2cid[n1]
        if trace_on:
            logger.info(
                'n0=%s, n1=%s, wgt=%a, n0_cid=%a, n1_cid=%a'
                % (n0, n1, wgt, n0_cid, n1_cid)
            )

        is_merge_allowed = not stop_at_two or len(clustering) > 2
        if trace_on:
            logger.info('is_merge_allowed %s' % (is_merge_allowed,))

        if n0_cid == n1_cid:
            if trace_on:
                logger.info('In the same cluster')
            score += wgt
        elif is_merge_allowed and not ct.has_edges_between_them(
            G_prime, clustering[n0_cid], clustering[n1_cid]
        ):
            if trace_on:
                logger.info('Merging disjoint clusters')
            sc_delta = ct.merge_clusters(n0_cid, n1_cid, G_prime, clustering, node2cid)
            assert sc_delta == 0
            score += sc_delta + wgt  # why might sc_delta be non-zero here???
        else:
            sc_merged = (
                ct.score_delta_after_merge(n0_cid, n1_cid, G_prime, clustering) + wgt
            )
            if trace_on:
                logger.info('sc_merged=%a' % sc_merged)
            sc_unmerged = -wgt
            if trace_on:
                logger.info('sc_unmerged=%a' % sc_unmerged)
            if len(clustering[n0_cid]) == 1 or len(clustering[n1_cid]) == 1:
                sc_n0_to_n1 = sc_n1_to_n0 = min(sc_merged, sc_unmerged) - 9999
                n0_to_move = n1_to_move = []
                if trace_on:
                    logger.info(
                        'not checking moving nodes because '
                        'at least one cluster is length 1'
                    )
            else:
                sc_n0_to_n1, n0_to_move = best_shift(
                    n0, n1, G_prime, clustering, node2cid, trace_on=trace_on
                )
                sc_n0_to_n1 += wgt
                if trace_on:
                    logger.info(
                        'sc_n0_to_n1=%a, n0_to_move=%a' % (sc_n0_to_n1, n0_to_move)
                    )
                sc_n1_to_n0, n1_to_move = best_shift(
                    n1, n0, G_prime, clustering, node2cid, trace_on=trace_on
                )
                sc_n1_to_n0 += wgt
                if trace_on:
                    logger.info(
                        'sc_n1_to_n0=%a, n1_to_move=%a' % (sc_n1_to_n0, n1_to_move)
                    )

            if is_merge_allowed and sc_merged >= max(
                sc_unmerged, sc_n0_to_n1, sc_n1_to_n0
            ):
                ct.merge_clusters(n0_cid, n1_cid, G_prime, clustering, node2cid)
                score += sc_merged
                if trace_on:
                    logger.info('Choose merge')
            elif sc_unmerged >= max(sc_n0_to_n1, sc_n1_to_n0):
                score += sc_unmerged
                if trace_on:
                    logger.info('Choose unmerged - unchanged')
            elif sc_n0_to_n1 >= sc_n1_to_n0:
                ct.shift_between_clusters(
                    n0_cid, n0_to_move, n1_cid, clustering, node2cid
                )
                score += sc_n0_to_n1
                if trace_on:
                    logger.info(
                        'Choose to shift from cluster %a to %a' % (n0_cid, n1_cid)
                    )
            else:
                ct.shift_between_clusters(
                    n1_cid, n1_to_move, n0_cid, clustering, node2cid
                )
                score += sc_n1_to_n0
                if trace_on:
                    logger.info(
                        'Choose to shift from cluster %a to %a' % (n1_cid, n0_cid)
                    )
        G_prime.add_weighted_edges_from([e])
        if trace_on:
            ct.print_structures(G_prime, clustering, node2cid, score)

    return clustering, score


def test_best_shift(trace_on=False):
    G = nx.Graph()

    logger.info('==================')
    logger.info('Testing best_shift')
    logger.info('==================')

    """
    For this test, leaving out ('c', 'e', 4), the edge to be added
    and leaving out ('d', 'e', 3), which should be added later.
    """
    G.add_weighted_edges_from(
        [
            ('a', 'b', 9),
            ('a', 'e', -2),
            ('b', 'c', -6),
            ('b', 'e', 5),
            ('b', 'f', -2),
            ('c', 'd', 7),
            ('d', 'f', -2),
            ('e', 'f', 6),
            ('d', 'g', -3),
            ('f', 'g', 4),
        ]
    )

    clustering = {0: {'a', 'b', 'e', 'f', 'g'}, 1: {'c', 'd'}}
    node2cid = ct.build_node_to_cluster_mapping(clustering)

    n0, n1 = 'e', 'c'  # from biggest set to smaller
    delta, to_move = best_shift(n0, n1, G, clustering, node2cid)
    exp_delta = -12
    exp_move = ['e', 'f', 'g']
    if exp_delta != delta or set(exp_move) != set(to_move):
        logger.info('Test 1 (larger to smaller): FAIL')
        logger.info('    delta %a, to_move %a' % (delta, sorted(to_move)))
        logger.info("    should be -12 and ['e', 'f', 'g']")
    else:
        logger.info('Test 1 (larger to smaller): success')

    n0, n1 = 'c', 'e'  # from biggest set to smaller
    delta, to_move = best_shift(n0, n1, G, clustering, node2cid)
    exp_delta = -26
    exp_move = ['c']
    if exp_delta != delta or set(exp_move) != set(to_move):
        logger.info('Test 2 (smaller to larger): FAIL')
        logger.info('delta %a, to_move %a' % (delta, sorted(to_move)))
        logger.info("should be -26 and ['c']")
    else:
        logger.info('Test 2 (smaller to larger): success')


def run_lca_alg1(G, expected_clustering, msg, stop_at_two=False, trace_on=False):
    node2cid = ct.build_node_to_cluster_mapping(expected_clustering)
    expected_score = ct.clustering_score(G, node2cid)
    clustering, score = lca_alg1(G, stop_at_two=stop_at_two, trace_on=trace_on)
    failed = False
    if not ct.same_clustering(clustering, expected_clustering):
        failed = True
        logger.info('%s FAILED' % (msg,))
    else:
        logger.info('%s success' % (msg,))

    if score != expected_score:
        failed = True
        logger.info('score %d, expected_score %d. FAILED' % (score, expected_score))

    if failed:
        logger.info('current structures with failure:')
        node2cid = ct.build_node_to_cluster_mapping(clustering)
        ct.print_structures(G, clustering, node2cid, score)


def test_overall(trace_on=False):
    logger.info('\n================\nTesting lca_alg1\n================')

    G = nx.Graph()
    expected_clustering = dict()
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (1) on empty graph:', trace_on=trace_on
    )

    G.add_nodes_from(['a'])
    expected_clustering = {0: {'a'}}
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (2) on single node graph:', trace_on=trace_on
    )

    G.add_nodes_from(['b'])
    expected_clustering = {0: {'a'}, 1: {'b'}}
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (3) on disjoint pair graph:', trace_on=trace_on
    )

    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 4)])
    expected_clustering = {0: {'a', 'b'}}
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (4) on one edge graph:', trace_on=trace_on
    )

    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', -6)])
    expected_clustering = {0: {'a'}, 1: {'b'}}
    run_lca_alg1(
        G,
        expected_clustering,
        'lca_alg1 (5) on one (negative) edge graph:',
        trace_on=trace_on,
    )

    G = nx.Graph()  # From Figure 2
    G.add_weighted_edges_from(
        [
            ('a', 'b', 9),
            ('a', 'e', -2),
            ('b', 'c', -6),
            ('b', 'e', 5),
            ('b', 'f', -2),
            ('c', 'd', 7),
            ('c', 'e', 4),
            ('d', 'e', 3),
            ('d', 'f', -1),
            ('e', 'f', 6),
        ]
    )
    expected_clustering = {0: {'a', 'b'}, 1: {'c', 'd', 'e', 'f'}}
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (6) on Figure 2 graph:', trace_on=trace_on
    )

    G = nx.Graph()  # Three component graph
    G.add_weighted_edges_from(
        [
            ('a', 'b', 1),
            ('a', 'd', 6),
            ('a', 'e', -8),
            ('a', 'f', -8),
            ('b', 'c', 1),
            ('b', 'd', -9),
            ('b', 'e', 4),
            ('b', 'f', -7),
            ('c', 'd', -8),
            ('c', 'e', -7),
            ('c', 'f', 5),
            ('d', 'e', 1),
            ('e', 'f', 1),
        ]
    )
    expected_clustering = {0: {'a', 'd'}, 1: {'b', 'e'}, 2: {'c', 'f'}}
    run_lca_alg1(
        G,
        expected_clustering,
        'lca_alg1 (7) on three-component graph:',
        trace_on=trace_on,
    )

    G = tct.ex_graph_fig1()
    expected_clustering = {
        0: {'a', 'b', 'd', 'e'},
        1: {'c'},
        2: {'f', 'g', 'j', 'k'},
        3: {'h', 'i'},
    }
    run_lca_alg1(
        G, expected_clustering, 'lca_alg1 (8) on Figure 1 graph:', trace_on=trace_on
    )


def test_no_final_merge(trace_on=False):
    logger.info('')
    logger.info(
        '=========================================\n'
        'Testing lca_alg1 with/without final merge\n'
        '========================================='
    )
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 8),
            ('a', 'd', -1),
            ('a', 'e', 2),
            ('b', 'c', 6),
            ('b', 'd', 3),
            ('b', 'e', 1),
            ('c', 'f', 1),
            ('d', 'e', 4),
            ('e', 'f', 5),
        ]
    )
    expected_clustering = {0: {'a', 'b', 'c'}, 1: {'d', 'e', 'f'}}
    run_lca_alg1(
        G,
        expected_clustering,
        '(1) No final merge allowed:',
        stop_at_two=True,
        trace_on=trace_on,
    )

    expected_clustering = {0: {'a', 'b', 'c', 'd', 'e', 'f'}}
    run_lca_alg1(
        G,
        expected_clustering,
        '(2) Final merge allowed:',
        stop_at_two=False,
        trace_on=trace_on,
    )


if __name__ == '__main__':
    trace_on = False
    test_best_shift(trace_on=trace_on)
    test_overall(trace_on=trace_on)
    test_no_final_merge(trace_on=trace_on)
