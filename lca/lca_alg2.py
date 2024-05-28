# -*- coding: utf-8 -*-
import networkx as nx
import logging

import lca_alg1 as a1
import cluster_tools as ct
import test_cluster_tools as tct


logger = logging.getLogger('lca')


def build_initial_from_constraints(G, in_same):
    """
    Form an initial clustering from the nodes in G such that any pair
    of nodes in the in_same list is in the same cluster and any node
    not in one of these pairs is in its own cluster.
    """
    # Use connected components to gather the in_same pairs
    tempG = nx.Graph()
    tempG.add_nodes_from(G.nodes())
    tempG.add_edges_from(in_same)
    same_sets = nx.connected_components(tempG)

    clustering = dict()
    for ci, s in enumerate(same_sets):
        clustering[ci] = s

    return clustering


def keep_separate(c0, c1, must_be_in_different):
    """
    Return true if the any pair of nodes in must_be_in_different
    are split across clusters c0 and c1.
    """
    for m, n in must_be_in_different:
        if (m in c0 and n in c1) or (m in c1 and n in c0):
            return True
    return False


def lca_alg1_constrained(curr_G, in_same=[], in_different=[], trace_on=False):
    """
    Use algorithm 1 to find the best clustering of the current
    subgraph subject to the constraints that all pairs of nodes from
    in_same must be in the same cluster and all pairs of nodes from
    in_different must be in different clusters.

    This does not check that the constraints from in_same and
    in_different can all be satisfied. In implementation the in_same
    constraints take precedence, but in use, one of the two in_same
    and in_different lists will be empty.
    """
    clustering = build_initial_from_constraints(curr_G, in_same)
    node2cid = ct.build_node_to_cluster_mapping(clustering)

    neg_edges, pos_edges = ct.get_weight_lists(curr_G, sort_positive=True)
    G_prime = nx.Graph()
    G_prime.add_nodes_from(curr_G)
    G_prime.add_weighted_edges_from(neg_edges)

    edges = [(p[0], p[1], curr_G[p[0]][p[1]]['weight']) for p in in_same]
    G_prime.add_weighted_edges_from(edges)
    score = ct.clustering_score(G_prime, node2cid)

    if trace_on:
        logger.info('=================================')
        logger.info('=====  lca_alg1_constrained  ====')
        logger.info('=================================')
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

        if (n0, n1) in in_same:
            if trace_on:
                logger.info('Skipping (%a, %a) because already in graph' % (n0, n1))
            continue

        wgt = e[2]
        n0_cid, n1_cid = node2cid[n0], node2cid[n1]
        if trace_on:
            logger.info(
                'n0=%s, n1=%s, wgt=%a, n0_cid=%a, n1_cid=%a'
                % (n0, n1, wgt, n0_cid, n1_cid)
            )

        if n0_cid == n1_cid:
            if trace_on:
                logger.info('Already in the same cluster')
            score += wgt

        elif keep_separate(clustering[n0_cid], clustering[n1_cid], in_different):
            if trace_on:
                logger.info('Must be kept separate')
            score -= wgt

        elif not ct.has_edges_between_them(
            G_prime, clustering[n0_cid], clustering[n1_cid]
        ):
            if trace_on:
                logger.info('Merging disjoint clusters')
            sc_delta = ct.merge_clusters(n0_cid, n1_cid, G_prime, clustering, node2cid)
            assert sc_delta == 0
            score += sc_delta + wgt

        else:
            sc_merged = (
                ct.score_delta_after_merge(n0_cid, n1_cid, G_prime, clustering) + wgt
            )
            if trace_on:
                logger.info('sc_merged=%a' % sc_merged)
            sc_unmerged = -wgt
            if trace_on:
                logger.info('sc_unmerged=%a' % sc_unmerged)

            if sc_merged > sc_unmerged:
                ct.merge_clusters(n0_cid, n1_cid, G_prime, clustering, node2cid)
                score += sc_merged
                if trace_on:
                    logger.info('Merging clusters with edges between')
            else:
                score += sc_unmerged
                if trace_on:
                    logger.info('No merge of clusters with edges between ')

        G_prime.add_weighted_edges_from([e])
        if trace_on:
            ct.print_structures(G_prime, clustering, node2cid, score)

    return clustering, score


def inconsistent_edges(G, clustering, node2cid):
    """
    Return each negatively-weighted edge that is inside a cluster, or
    positively-weighted edge that is between clusters.
    """
    inconsistent = []
    for m, n in G.edges():
        if m > n:
            m, n = n, m
        wgt = G[m][n]['weight']
        # logger.info(m, n, wgt)
        m_cid, n_cid = node2cid[m], node2cid[n]
        if (wgt < 0 and m_cid == n_cid) or (wgt > 0 and m_cid != n_cid):
            inconsistent.append((m, n, wgt))
    return inconsistent


def best_alternative_len2(G, clustering, node2cid):
    """Return the best alternative to the current clustering when G has
    exactly two nodes.
    """
    if len(clustering) == 2:
        alt_clustering = {0: set(G.nodes())}
    else:
        alt_clustering = {c: {n} for c, n in enumerate(G.nodes())}
    alt_node2cid = ct.build_node_to_cluster_mapping(alt_clustering)
    alt_score = ct.clustering_score(G, alt_node2cid)
    return alt_clustering, alt_score


def lca_alg2(G, clustering, node2cid, trace_on=False):
    """
    If it is a single cluster, then stop the original algorithm when
    there are two clusters.  Perhaps can run alternative multiple times

    If there are multiple clusterings, then one option is a merge, but
    add others based on inconsistency

    Don't allow len(G) <= 1 if it is two, and the
    nodes are disconnected, there is also no alternative.  If it is two,
    then split/merging vs. merging/splitting is the alternative.
    """
    assert len(G) >= 2

    if len(G) == 2:
        return best_alternative_len2(G, clustering, node2cid)

    """ Form the first estimate of the best alternative.  If there is just
        one cluster in the current (local) best clustering then rerun
        Alg1 constrained to stop at at most two.  Otherwise, just form
        a single clustering.
    """
    if len(clustering) == 1:
        best_clustering, best_score = a1.lca_alg1(G, stop_at_two=True)
        best_node2cid = ct.build_node_to_cluster_mapping(best_clustering)
    else:
        best_clustering = {0: set(G.nodes())}
        best_node2cid = {n: 0 for n in G.nodes()}
        best_score = ct.clustering_score(G, best_node2cid)

    if trace_on:
        logger.info(
            'In lca_alg2, before checking inconsistent\n'
            'best_clustering %a, best_score %d, checking %d'
            % (best_clustering, best_score, ct.clustering_score(G, best_node2cid))
        )

    inconsistent = inconsistent_edges(G, clustering, node2cid)
    inconsistent.sort(key=lambda e: abs(e[2]), reverse=True)
    if trace_on:
        logger.info('In lca_alg2: clustering %s' % (clustering,))
        logger.info('In lca_alg2: inconsistent edges %s' % (inconsistent,))
        logger.info('Starting inconsistent edge loop')

    for e in inconsistent:
        if trace_on:
            logger.info('e = %s' % (e,))
        if e[2] < 0:
            if trace_on:
                logger.info('Forcing edge into different clusters')
            new_clustering, new_score = lca_alg1_constrained(
                G, in_same=[], in_different=[(e[0], e[1])]
            )
        else:
            if trace_on:
                logger.info('Forcing edge into same cluster')
            new_clustering, new_score = lca_alg1_constrained(
                G, in_same=[(e[0], e[1])], in_different=[]
            )

        if trace_on:
            logger.info(
                'Best score returned by lca_alg1_constrained is %s' % (new_score,)
            )
            logger.info(
                'Checking',
                ct.clustering_score(G, ct.build_node_to_cluster_mapping(new_clustering)),
            )
        if new_score > best_score:
            if trace_on:
                logger.info('New best')
            best_score = new_score
            best_clustering = new_clustering

    return best_clustering, best_score


def test_build_initial():
    logger.info('==================\nTest build_initial\n==================')
    in_same = [
        ('b', 'a'),
        ('e', 'f'),
        ('g', 'h'),
        ('c', 'd'),
        ('e', 'c'),
        ('i', 'h'),
        ('c', 'b'),
        ('a', 'c'),
        ('g', 'i'),
    ]
    all_nodes = ['a', 'p', 'q', 'b', 'c', 'r', 'd', 'e', 'f', 'g', 'h', 'i', 's', 't']

    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    clustering = build_initial_from_constraints(G, in_same)
    exp_clustering = {
        0: {'a', 'b', 'c', 'd', 'e', 'f'},
        1: {'g', 'h', 'i'},
        2: {'p'},
        3: {'q'},
        4: {'r'},
        5: {'s'},
        6: {'t'},
    }
    if ct.same_clustering(clustering, exp_clustering, output_differences=True):
        logger.info('test_building_initial: success')
    else:
        logger.info('test_building_initial: FAIL')


def test_keep_separate():
    logger.info('\n==================\nTest keep_separate\n==================')
    c0 = {'a', 'b', 'c'}
    c1 = {'d', 'e', 'f', 'g'}
    must_be_in_different = [('b', 'd')]
    logger.info(
        'keep_separate(%a) should be True, is %a'
        % (must_be_in_different, keep_separate(c0, c1, must_be_in_different))
    )
    must_be_in_different = [('f', 'a')]
    logger.info(
        'keep_separate(%a) should be True, is %a'
        % (must_be_in_different, keep_separate(c0, c1, must_be_in_different))
    )
    must_be_in_different = [('f', 'g'), ('e', 'c')]
    logger.info(
        'keep_separate(%a) should be True, is %a'
        % (must_be_in_different, keep_separate(c0, c1, must_be_in_different))
    )
    must_be_in_different = [('a', 'b'), ('d', 'e')]
    logger.info(
        'keep_separate(%a) should be False, is %a'
        % (must_be_in_different, keep_separate(c0, c1, must_be_in_different))
    )
    must_be_in_different = [('a', 'q'), ('p', 'e')]
    logger.info(
        'keep_separate(%a) should be False, is %a'
        % (must_be_in_different, keep_separate(c0, c1, must_be_in_different))
    )


def test_lca_alg1_constrained():
    logger.info(
        '\n=========================\n'
        'Test lca_alg1_constrained\n'
        '========================='
    )
    G = tct.ex_graph_fig1()
    G['g']['j']['weight'] = -4  # a little larger than original to break a tie
    in_same = [('f', 'i')]
    in_different = [('d', 'e')]
    clustering, score = lca_alg1_constrained(G, in_same, in_different)
    node2cid = ct.build_node_to_cluster_mapping(clustering)
    correct_score = ct.clustering_score(G, node2cid)

    exp_clustering = {
        0: {'a', 'b', 'd'},
        1: {'f', 'g', 'h', 'i', 'k'},
        2: {'c'},
        3: {'e'},
        4: {'j'},
    }
    is_same = ct.same_clustering(clustering, exp_clustering, output_differences=True)
    if is_same:
        logger.info('constrained (d,e) different and (f,i) same: success')
    else:
        logger.info('constrained (d,e) different and (f,i) same: FAIL')

    if score != correct_score:
        logger.info('scoring error:  actual %a, correct %a' % (score, correct_score))
    else:
        logger.info('scoring correct:  actual %a, correct %a' % (score, correct_score))


def test_inconsistent_edges():
    logger.info(
        '\n=======================\n'
        'Test inconsistent_edges\n'
        '======================='
    )
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
    clustering = ct.build_clustering(n2c_optimal)

    inconsistent = inconsistent_edges(G, clustering, n2c_optimal)
    should_be = [
        ('a', 'e', -2),
        ('c', 'f', 2),
        ('d', 'h', 3),
        ('d', 'e', -1),
        ('f', 'i', 2),
        ('f', 'k', -3),
        ('g', 'j', -3),
    ]
    if set(should_be) == set(inconsistent):
        logger.info('Identify inconsistent edges: success')
    else:
        logger.info('Identify inconsistent edges: FAIL')


def run_lca_alg2(G, best_clustering, exp_alt_clustering, msg, trace_on=False):
    exp_alt_node2cid = ct.build_node_to_cluster_mapping(exp_alt_clustering)
    exp_alt_score = ct.clustering_score(G, exp_alt_node2cid)

    best_node2cid = ct.build_node_to_cluster_mapping(best_clustering)
    alt_clustering, alt_score = lca_alg2(
        G, best_clustering, best_node2cid, trace_on=trace_on
    )

    failed = False
    if not ct.same_clustering(alt_clustering, exp_alt_clustering):
        failed = True
        logger.info('%s FAILED' % (msg,))
    else:
        logger.info('%s success' % (msg,))

    if alt_score != exp_alt_score:
        failed = True
        logger.info('score %d, expected_score %d. FAILED' % (alt_score, exp_alt_score))

    if failed:
        logger.info('current structures with failure:')
        alt_node2cid = ct.build_node_to_cluster_mapping(alt_clustering)
        ct.print_structures(G, alt_clustering, alt_node2cid, alt_score)


def test_lca_alg2():
    logger.info('\n=============\nTest lca_alg2\n=============')

    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 5)])
    one_cluster = {0: set(['a', 'b'])}
    expected_alt = {0: {'a'}, 1: {'b'}}
    msg = '(1) Two node graph, initially all together:'
    run_lca_alg2(G, one_cluster, expected_alt, msg)

    two_clusters = {0: set('a'), 1: set('b')}
    expected_alt = {0: {'a', 'b'}}
    msg = '(2) Two node graph, start separate, so should be together:'
    run_lca_alg2(G, two_clusters, expected_alt, msg)

    G = nx.Graph()
    G.add_weighted_edges_from(
        [('a', 'b', -2), ('a', 'c', 8), ('a', 'd', -1), ('b', 'd', 3), ('c', 'd', 5)]
    )
    first_clustering = {0: {'a', 'c'}, 1: {'b', 'd'}}
    expected_alt = {0: {'a', 'b', 'c', 'd'}}
    msg = '(3) Testing generation of initial best from two pairs:'
    run_lca_alg2(G, first_clustering, expected_alt, msg)

    first_clustering = {0: {'a', 'b', 'c', 'd'}}
    expected_alt = {0: {'a', 'c', 'd'}, 1: {'b'}}
    msg = '(4) Testing generation of initial best from single cluster:'
    run_lca_alg2(G, first_clustering, expected_alt, msg)

    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 9),
            ('a', 'e', -2),
            ('b', 'c', -6),
            ('b', 'e', 5),
            ('b', 'f', -2),
            ('c', 'd', 7),
            ('c', 'e', 4),
            ('d', 'e', 4),
            ('d', 'f', -1),
            ('e', 'f', 6),
        ]
    )
    first_clustering = {0: {'a', 'b'}, 1: {'c', 'd', 'e', 'f'}}
    expected_alt = {0: {'a', 'b', 'c', 'd', 'e', 'f'}}
    msg = '(5) Testing two components merged into one as alternative:'
    run_lca_alg2(G, first_clustering, expected_alt, msg)

    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 5),
            ('a', 'c', 3),
            ('a', 'd', -8),
            ('b', 'c', -6),
            ('b', 'd', 4),
            ('c', 'd', 6),
        ]
    )
    first_clustering = {0: {'a', 'b'}, 1: {'c', 'd'}}
    expected_alt = {0: {'a', 'c'}, 1: {'b', 'd'}}
    msg = '(6) Testing two components reformed into two others:'
    run_lca_alg2(G, first_clustering, expected_alt, msg)

    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 8),
            ('a', 'c', 5),
            ('a', 'd', -2),
            ('b', 'd', 4),
            ('b', 'e', -5),
            ('c', 'd', 3),
            ('d', 'e', 6),
        ]
    )
    first_clustering = {0: {'a', 'b', 'c', 'd', 'e'}}
    expected_alt = {0: {'a', 'b', 'c'}, 1: {'d', 'e'}}
    msg = '(7) Testing one component split into two:'
    run_lca_alg2(G, first_clustering, expected_alt, msg)


if __name__ == '__main__':
    test_build_initial()
    test_keep_separate()
    test_lca_alg1_constrained()
    test_inconsistent_edges()
    test_lca_alg2()
