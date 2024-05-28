# -*- coding: utf-8 -*-
import networkx as nx
import cluster_tools as ct
import logging


logger = logging.getLogger('lca')


def ex_graph_fig1():
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ('a', 'b', 8),
            ('a', 'd', 4),
            ('a', 'e', -2),
            ('b', 'c', -1),
            ('b', 'e', 4),
            ('b', 'f', -4),
            ('b', 'i', -4),
            ('c', 'f', 2),
            ('c', 'g', -3),
            ('d', 'e', -1),
            ('d', 'h', 3),
            ('e', 'i', -5),
            ('f', 'g', 8),
            ('f', 'i', 2),
            ('f', 'j', 2),
            ('f', 'k', -3),
            ('g', 'j', -3),
            ('g', 'k', 7),
            ('h', 'i', 6),
            ('i', 'j', -4),
            ('j', 'k', 5),
        ]
    )
    return G


def ex_graph_fig4():
    G = nx.Graph()
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
    return G


"""
def ex_graph_fig5():
    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 9), ('a', 'e', -2),
                               ('b', 'c', -6), ('b', 'e', 5), ('b', 'f', -2),
                               ('c', 'd', 8), ('c', 'e', 4),
                               ('d', 'e', 2), ('d', 'f', -1),
                               ('e', 'f', 6)])
    return G
"""


def test_build_clustering_and_mapping():
    logger.info('==================')
    logger.info('Testing build_clustering')
    empty_n2c = {}
    empty_clustering = ct.build_clustering(empty_n2c)
    logger.info(
        'Empty node 2 cluster mapping should produce empty clustering %s'
        % (empty_clustering,)
    )

    # G = ex_graph_fig1()
    n2c_optimal = {
        'a': '0',
        'b': '0',
        'd': '0',
        'e': '0',
        'c': '1',
        'h': '2',
        'i': '2',
        'f': '3',
        'g': '3',
        'j': '3',
        'k': '3',
    }

    clustering = ct.build_clustering(n2c_optimal)
    logger.info(
        "Cluster 0 should be ['a', 'b', 'd', 'e']. It is %s" % (sorted(clustering['0']),)
    )
    logger.info("Cluster 1 should be ['c']. It is %s" % (sorted(clustering['1']),))
    logger.info("Cluster 2 should be ['h', 'i']. It is %s" % (sorted(clustering['2']),))
    logger.info(
        "Cluster 3 should be ['f', 'g', 'j', 'k']. It is %s" % (sorted(clustering['3']),)
    ),

    logger.info('==================')
    logger.info('Testing build_node_to_cluster_mapping')
    empty_clustering = {}
    empty_n2c = ct.build_node_to_cluster_mapping(empty_clustering)
    logger.info(
        'Empty clustering should produce empty node-to-cluster mapping %s' % (empty_n2c,)
    )

    n2c_rebuilt = ct.build_node_to_cluster_mapping(clustering)
    logger.info(
        'After rebuilding the node2cid mapping should be the same.  Is it? %s'
        % (n2c_optimal == n2c_rebuilt,)
    )


def test_build_clustering_from_clusters():
    logger.info('================================')
    logger.info('test_build_clustering_from_clusters')
    clist = [['h', 'i', 'j'], ['k', 'm'], ['p']]
    n = len(clist)
    cids = list(ct.cids_from_range(n))
    clustering = ct.build_clustering_from_clusters(cids, clist)
    logger.info('Returned clustering:')
    logger.info(clustering)
    correct = len(clustering) == 3
    logger.info('Correct number of clusters %s' % (correct,))
    correct = (
        set(clist[0]) == clustering[cids[0]]
        and set(clist[1]) == clustering[cids[1]]
        and set(clist[2]) == clustering[cids[2]]
    )
    logger.info('Clusters are correct: %s' % (correct,))

    #  Catching error from repeated entry
    clist = [['h', 'i', 'j'], ['k', 'm'], ['p', 'p']]
    n = len(clist)
    try:
        clustering = ct.build_clustering_from_clusters(ct.cids_from_range(n), clist)
    except AssertionError:
        logger.info('Caught error from having repeated entry in one cluster')

    #  Catching error from intersecting lists
    clist = [['h', 'i', 'k'], ['k', 'm'], ['p', 'q']]
    n = len(clist)
    try:
        clustering = ct.build_clustering_from_clusters(ct.cids_from_range(n), clist)
    except AssertionError:
        logger.info('Caught error from having intersecting lists')


def test_cluster_scoring_and_weights():
    G = ex_graph_fig1()

    logger.info('=====================')
    logger.info('Testing cid_list_score')
    cids = list(ct.cids_from_range(4))
    n2c_random = {
        'a': cids[0],
        'b': cids[0],
        'f': cids[0],
        'c': cids[1],
        'g': cids[1],
        'd': cids[2],
        'e': cids[2],
        'i': cids[2],
        'h': cids[3],
        'j': cids[3],
        'k': cids[3],
    }
    clustering_random = ct.build_clustering(n2c_random)
    score = ct.cid_list_score(
        G, clustering_random, n2c_random, [cids[0], cids[2], cids[3]]
    )
    logger.info('Score between clusters [c0, c2, c3] should be -5 and is %s' % (score,))

    logger.info('=====================')
    logger.info('Testing clustering_score')
    """ First clustering:  all together """
    n2c_single_cluster = {n: 'c0' for n in G.nodes}
    logger.info(
        'Score with all together should be 21.  Score = %s'
        % (ct.clustering_score(G, n2c_single_cluster),)
    )

    """ Second clustering:  all separate """
    n2c_all_separate = {n: 'c' + str(i) for i, n in enumerate(G.nodes)}
    logger.info(
        'Score with all together should be -21.  Score = %s'
        % (ct.clustering_score(G, n2c_all_separate),)
    )

    """ Third clustering: optimal, by hand """
    cids = list(ct.cids_from_range(4))
    n2c_optimal = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[3],
        'k': cids[3],
    }
    logger.info(
        'Optimal score should be 49. Score = %s' % (ct.clustering_score(G, n2c_optimal),)
    )

    negatives, positives = ct.get_weight_lists(G, sort_positive=True)
    logger.info('Length of negatives should be 10.  It is %s' % (len(negatives),))
    logger.info('Length of positives should be 11.  It is %s' % (len(positives),))
    logger.info('0th positive should be 8.  It is %s' % (positives[0],))
    logger.info('Last positive should be 2.  It is %s' % (positives[-1],))


def test_has_edges_between():
    G = ex_graph_fig1()
    c0 = {'a', 'd'}
    c1 = {'c', 'f'}
    c2 = {'b', 'i', 'j'}
    logger.info('========================')
    logger.info('Testing has_edges_between_them')
    res01 = ct.has_edges_between_them(G, c0, c1)
    logger.info('c0 to c1 should be False. is %s' % (res01,))
    res02 = ct.has_edges_between_them(G, c0, c2)
    logger.info('c0 to c2 should be True. is %s' % (res02,))
    res12 = ct.has_edges_between_them(G, c1, c2)
    logger.info('c1 to c2 should be True. is %s' % (res12,))
    res10 = ct.has_edges_between_them(G, c1, c0)
    logger.info('c1 to c0 should be False. is %s' % (res10,))


def test_merge():
    logger.info('===========================')
    logger.info('test_merge')
    G = ex_graph_fig1()
    cids = list(ct.cids_from_range(4))
    logger.info(cids)
    n2c_optimal = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[3],
        'k': cids[3],
    }
    clustering = ct.build_clustering(n2c_optimal)

    logger.info('-------------')
    logger.info('score_delta_after_merge')
    delta = ct.score_delta_after_merge(cids[2], cids[3], G, clustering)
    logger.info('possible merge of 2, 3; delta should be -4, and is %s' % (delta,))

    logger.info('-------------')
    logger.info('merge_clusters')
    score_before = ct.clustering_score(G, n2c_optimal)
    delta = ct.merge_clusters(cids[0], cids[2], G, clustering, n2c_optimal)
    score_after = ct.clustering_score(G, n2c_optimal)
    logger.info(
        'delta = %s should be %s'
        % (
            delta,
            score_after - score_before,
        )
    )
    logger.info('---')
    for c in clustering:
        logger.info('%s: %s' % (c, clustering[c]))
    logger.info('---')
    for n in G.nodes:
        logger.info('%s: %s' % (n, n2c_optimal[n]))

    logger.info('--------')
    logger.info('Retesting merge with order of clusters reversed')
    n2c_optimal = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[3],
        'k': cids[3],
    }
    clustering = ct.build_clustering(n2c_optimal)

    logger.info('-------------')
    logger.info('score_delta_after_merge')
    delta = ct.score_delta_after_merge(cids[3], cids[2], G, clustering)
    logger.info('possible merge of 3, 2; delta should be -4, and is %s' % (delta,))

    logger.info('-------------')
    logger.info('merge_clusters')
    score_before = ct.clustering_score(G, n2c_optimal)
    delta = ct.merge_clusters(cids[2], cids[0], G, clustering, n2c_optimal)
    score_after = ct.clustering_score(G, n2c_optimal)
    logger.info(
        'delta = %s should be %s'
        % (
            delta,
            score_after - score_before,
        )
    )
    logger.info('---')
    for c in clustering:
        logger.info('%s: %s' % (c, clustering[c]))
    logger.info('---')
    for n in G.nodes:
        logger.info('%s: %s' % (n, n2c_optimal[n]))


def test_shift_between_clusters():
    logger.info('===========================')
    logger.info('test_shift_between_clusters')
    cids = list(ct.cids_from_range(4))
    n2c_optimal = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[3],
        'k': cids[3],
    }
    clustering = ct.build_clustering(n2c_optimal)

    n0_cid, n1_cid = cids[3], cids[2]
    n0_nodes_to_move = {'f', 'j'}
    logger.info('Shifting from cluster %s to %s:' % (n0_cid, n1_cid))
    logger.info('Nodes to move: %s' % (sorted(n0_nodes_to_move),))
    logger.info('Cluster %s: %s' % (n0_cid, sorted(clustering[n0_cid])))
    logger.info('Cluster %s: %s' % (n1_cid, sorted(clustering[n1_cid])))

    ct.shift_between_clusters(n0_cid, n0_nodes_to_move, n1_cid, clustering, n2c_optimal)
    logger.info('After shift, cluster %s: %s' % (n0_cid, sorted(clustering[n0_cid])))
    logger.info('After shift, cluster %s: %s' % (n1_cid, sorted(clustering[n1_cid])))
    logger.info("n2c['f'] = %s" % (n2c_optimal['f'],))
    logger.info("n2c['j'] = %s" % (n2c_optimal['j'],))
    logger.info("n2c['h'] = %s" % (n2c_optimal['h'],))
    logger.info("n2c['i'] = %s" % (n2c_optimal['i'],))
    logger.info("n2c['g'] = %s" % (n2c_optimal['g'],))
    logger.info("n2c['k'] = %s" % (n2c_optimal['k'],))


def test_replace_clusters():
    logger.info('===========================')
    logger.info('test replace_clusters')
    cids = list(ct.cids_from_range(8))
    n2c = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[4],
        'k': cids[4],
    }
    clustering = ct.build_clustering(n2c)
    old_cids = [cids[2], cids[4]]
    added_clusters = {cids[5]: set(['j']), cids[7]: set(['h', 'i', 'k'])}
    ct.replace_clusters(old_cids, added_clusters, clustering, n2c)
    logger.info(
        'Cluster ids, should be c0, c1, c3, c5, c7.  Are: %s' % (list(clustering.keys()),)
    )
    logger.info("clustering[c5] should be {'j'}!! and is %s" % (clustering[cids[5]],))
    logger.info(
        "clustering[c7] should be {'h', 'i', 'k'} and is %s" % (clustering[cids[7]],)
    )
    logger.info("n2c['h'] should be c7 and is %s" % (n2c['h'],))
    logger.info("n2c['j'] should be c5 and is %s" % (n2c['j'],))


def test_form_connected_cluster_pairs():
    logger.info('=================================')
    logger.info('test form_connected_cluster_pairs')
    G = ex_graph_fig1()
    cids = list(ct.cids_from_range(5))
    n2c = {
        'a': cids[0],
        'b': cids[0],
        'd': cids[0],
        'e': cids[0],
        'c': cids[1],
        'h': cids[2],
        'i': cids[2],
        'f': cids[3],
        'g': cids[3],
        'j': cids[4],
        'k': cids[4],
    }
    clustering = ct.build_clustering(n2c)

    cid_pairs = ct.form_connected_cluster_pairs(G, clustering, n2c)
    logger.info('form_connected_cluster_pairs(G, clustering, n2c)')
    logger.info('result:  %s' % (cid_pairs,))
    logger.info(
        'expecting: %s'
        % (
            [
                (cids[0], cids[1]),
                (cids[0], cids[2]),
                (cids[0], cids[3]),
                (cids[1], cids[3]),
                (cids[2], cids[3]),
                (cids[2], cids[4]),
                (cids[3], cids[4]),
            ],
        )
    )

    new_cids = [cids[1], cids[4]]
    cid_pairs = ct.form_connected_cluster_pairs(G, clustering, n2c, new_cids)
    logger.info('form_connected_cluster_pairs(G, clustering, n2c, new_cids)')
    logger.info('result:  %s' % (cid_pairs,))
    logger.info(
        'expecting: %s'
        % (
            [
                (cids[0], cids[1]),
                (cids[1], cids[3]),
                (cids[2], cids[4]),
                (cids[3], cids[4]),
            ],
        )
    )


def test_same_clustering():
    """"""
    cids = list(ct.cids_from_range(99))

    clustering0 = {
        cids[0]: {'a', 'b'},
        cids[3]: {'c'},
        cids[4]: {'d', 'e'},
        cids[6]: {'f', 'g', 'h'},
        cids[8]: {'i', 'j', 'k', 'l', 'm'},
    }
    clustering1 = {
        cids[6]: {'d', 'e'},
        cids[8]: {'c'},
        cids[16]: {'f', 'g', 'h'},
        cids[19]: {'i', 'k', 'l', 'm', 'j'},
        cids[20]: {'b', 'a'},
    }
    clustering2 = {
        cids[6]: {'d', 'c', 'e'},
        cids[16]: {'f', 'g', 'h'},
        cids[22]: {'i', 'j', 'k', 'l', 'm'},
        cids[25]: {'b', 'a'},
    }

    logger.info('====================')
    logger.info('test_same_clustering')
    logger.info('first test should generate no output and then return True')
    logger.info(ct.same_clustering(clustering0, clustering1, True))
    logger.info('second test should generate no output and then return False')
    logger.info(ct.same_clustering(clustering0, clustering2, False))
    logger.info('third test should generate mismatch output and then return False')
    logger.info('Expected:')
    logger.info("['c'] not in 2nd")
    logger.info("['d', 'e'] not in 2nd")
    logger.info("['c', 'd', 'e'] not in 1st")
    result = ct.same_clustering(clustering0, clustering2, True)
    logger.info('It returned %s' % (result,))


def test_extract_subclustering():
    clustering = {
        '0': {'a', 'b', 'c'},
        '1': {'d', 'e', 'f', 'g'},
        '2': {'h', 'i', 'j'},
        '3': {'k'},
    }
    nodes = ['k', 'a', 'f', 'e', 'b', 'c']
    new_c = ct.extract_subclustering(nodes, clustering)

    logger.info('====================')
    logger.info('test_extract_subclustering')
    logger.info('length of new should be 3, and it is: %d' % len(new_c))
    logger.info("new cluster 0 should have {'a', 'b', 'c'}: %a" % new_c['0'])
    logger.info("new cluster 1 should have {'e', 'f'}: %a" % new_c['1'])
    logger.info("new cluster 1 should have {'k'}: %a" % new_c['3'])


def test_comparisons():
    """"""
    cids = list(ct.cids_from_range(99))
    gt = {
        cids[0]: {'a', 'b'},
        cids[3]: {'c'},
        cids[4]: {'d', 'e'},
        cids[6]: {'f', 'g', 'h'},
        cids[8]: {'i', 'j', 'k', 'l', 'm'},
        cids[10]: {'o'},
        cids[13]: {'p', 'q'},
        cids[15]: {'r', 's', 't'},
        cids[16]: {'u', 'v', 'w'},
        cids[19]: {'y', 'z', 'aa'},
    }
    gt_n2c = ct.build_node_to_cluster_mapping(gt)

    est = {
        cids[25]: {'y', 'z', 'aa'},
        cids[29]: {'u', 'v'},
        cids[31]: {'w', 'r', 's', 't'},
        cids[37]: {'p'},
        cids[41]: {'q', 'o', 'm'},
        cids[43]: {'i', 'j', 'k', 'l'},
        cids[47]: {'a', 'b'},
        cids[53]: {'c'},
        cids[59]: {'d', 'e'},
        cids[61]: {'f', 'g', 'h'},
    }
    est_n2c = ct.build_node_to_cluster_mapping(est)

    logger.info('================')
    logger.info('test_comparisons')
    logger.info('ct.compare_by_lengths')

    ct.compare_by_lengths(est, est_n2c, gt)

    logger.info(
        'Output for this example should be:\n'
        '1, 2, 1, 0.50, 0.667\n'
        '2, 3, 2, 0.67, 0.833\n'
        '3, 4, 2, 0.50, 0.854\n'
        '5, 1, 0, 0.00, 0.800'
    )

    logger.info('------')
    logger.info('ct.pairwise_eval')
    # result = ct.compare_to_ground_truth(est, est_n2c, gt, gt_n2c)
    result = ct.percent_and_PR(est, est_n2c, gt, gt_n2c)
    logger.info('Result is [%1.3f, %1.3f, %1.3f]' % tuple(result))
    num_clusters = len(est)
    num_correct = 5
    tp, fp, fn = 18, 6, 7
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    logger.info(
        'Should be [%1.3f, %1.3f, %1.3f]'
        % (num_correct / num_clusters, precision, recall)
    )


def test_count_equal():
    """"""
    cids = list(ct.cids_from_range(99))
    gt = {
        cids[0]: {'a', 'b'},
        cids[3]: {'c'},
        cids[4]: {'d', 'e'},
        cids[6]: {'f', 'g', 'h'},
        cids[8]: {'i', 'j', 'k', 'l', 'm'},
        cids[10]: {'o'},
        cids[13]: {'p', 'q'},
        cids[15]: {'r', 's', 't'},
        cids[16]: {'u', 'v', 'w'},
        cids[19]: {'y', 'z', 'aa'},
    }

    est = {
        cids[25]: {'y', 'z', 'aa'},
        cids[29]: {'u', 'v'},
        cids[31]: {'w', 'r', 's', 't'},
        cids[37]: {'p'},
        cids[41]: {'q', 'o', 'm'},
        cids[43]: {'i', 'j', 'k', 'l'},
        cids[47]: {'a', 'b'},
        cids[53]: {'c'},
        cids[59]: {'d', 'e'},
        cids[61]: {'f', 'g', 'h'},
    }

    est_n2c = ct.build_node_to_cluster_mapping(est)
    n = ct.count_equal_clustering(gt, est, est_n2c)
    logger.info('test_count_equal: should be 5 and is %s' % (n,))


if __name__ == '__main__':
    test_build_clustering_and_mapping()
    test_build_clustering_from_clusters()
    test_cluster_scoring_and_weights()
    test_has_edges_between()
    test_merge()
    test_shift_between_clusters()
    test_replace_clusters()
    test_form_connected_cluster_pairs()
    test_comparisons()
    test_same_clustering()
    test_extract_subclustering()
    test_count_equal()
