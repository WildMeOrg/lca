# -*- coding: utf-8 -*-

import networkx as nx
import logging

import cluster_tools as ct


logger = logging.getLogger('lca')


"""
Generates the comparison of the clusterings before and after the run of
the graph algorithm, producing the main interpretation of the graph
algorithm result.

Given two clustering dictionaries and their node-to-cluster id
mapping, the main function here produces a list of cluster_change
objects.  Each contains:

old_clustering - sub-dictionary (of the old clustering) involved in a
change; the sub-dictionary is a mapping from cluster id (cid) to a set
of graph nodes (annotations).

old_n2c - old node to cluster mapping

new_clustering - sub-dictionary (of the new clustering) involved in a
change; this has the same structure as the old clustering; however,
the cluster ids are completely meaningless and don't preserve the old
ones - this has to be managed when they are committed to the database

new_n2c - new node to cluster mapping

query_nodes - new nodes added during to the old_clustering

removed_nodes - old nodes that might have been eliminated (this is
rare)

change_type - type of change

There are several ways to think about a cluster change object. Perhaps
the easiest is to say that each is a minimal sub-clustering of old and
new clustering covering the same set of graph nodes. One caveat to
this definition is that it largely ignores query nodes that have been
added at the start of this run of the id process and other nodes
that might (rarely) have been deleted

The key definition is the type of change. This dictates what might
happen in an interaction with people.


Unchanged:  A new cluster is the same as an old cluster

New: A new cluster is formed only from new nodes

Extended: A new cluster contains nodes from a single old cluster plus
new annotations

Removed: All existing nodes from an old cluster have been
removed. This should be relatively rare.

Merge: A new cluster is formed from a combination of all nodes
in two or more previous clusters plus zero or more new nodes

Split: A new cluster is formed from a proper subset of nodes from
exactly one old cluster, plus, perhaps, new nodes:

Merge/split: A new cluster is formed from a combination of nodes from at
least two old clusters, where at least one of them contains a proper
subset of nodes from a previous cluster. New nodes may be added.
"""


class clustering_change(object):  # NOQA
    def __init__(self, old_clustering, new_clustering):
        self.old_clustering = old_clustering
        self.new_clustering = new_clustering

        old_nodes = set()
        if len(old_clustering) > 0:
            old_nodes = set.union(*old_clustering.values())

        new_nodes = set()
        if len(new_clustering) > 0:
            new_nodes = set.union(*new_clustering.values())

        self.query_nodes = new_nodes - old_nodes
        self.removed_nodes = old_nodes - new_nodes

        if len(self.old_clustering) == 0:
            self.change_type = 'New'
            assert len(self.new_clustering) == 1

        elif len(self.new_clustering) == 0:
            self.change_type = 'Removed'
            assert len(self.old_clustering) == 1

        elif len(self.old_clustering) == 1 and len(self.new_clustering) == 1:
            if len(self.query_nodes) == 0:
                self.change_type = 'Unchanged'
            else:
                self.change_type = 'Extension'

        elif len(self.old_clustering) == 1:
            self.change_type = 'Split'

        elif len(self.new_clustering) == 1:
            self.change_type = 'Merge'

        else:
            self.change_type = 'Merge/Split'

    def log_change(self):
        logger.info('Old clustering %a' % self.old_clustering)
        logger.info('New clustering %a' % self.new_clustering)
        logger.info('Query nodes %a' % self.query_nodes)
        if len(self.removed_nodes) > 0:
            logger.info('Removed nodes %a' % self.query_nodes)
        logger.info('Change type %a' % self.change_type)

    def print_it(self):
        logger.info('old_clustering %s' % (self.old_clustering,))
        logger.info('new_clustering %s' % (self.new_clustering,))
        logger.info('query_nodes %s' % (self.query_nodes,))
        logger.info('removed_nodes %s' % (self.removed_nodes,))
        logger.info('change_type %s' % (self.change_type,))

    def serialize(self):
        data = {
            'old_clustering': self.old_clustering,
            'new_clustering': self.new_clustering,
            'query_nodes': self.query_nodes,
            'removed_nodes': self.removed_nodes,
            'change_type': self.change_type,
        }
        return data


def bipartite_cc(from_visited, from_nbrs, to_visited, to_nbrs, from_nodes):
    """
    Recursive BFS labeling of the reachable nodes of a bipartite
    graph starting from the from_nodes set.
    """
    to_nodes = set()
    for v in from_nodes:
        for t in from_nbrs[v]:
            if not to_visited[t]:
                to_visited[t] = True
                to_nodes.add(t)
    if len(to_nodes) == 0:
        return from_nodes, set()
    else:
        to_rec, from_rec = bipartite_cc(
            to_visited, to_nbrs, from_visited, from_nbrs, to_nodes
        )
        return from_rec | from_nodes, to_rec


def find_changes(old_clustering, old_n2c, new_clustering, new_n2c):
    """
    Main function to find changes between an old and new clustering as
    described above.

    1. Form a bipartite graph between the ids in the old clustering and
    the ids in the new clustering. Edges are generated between old
    and new clusters that intersect.
    """
    old_nbrs = dict()
    for oc, nodes in old_clustering.items():
        # logger.info(oc, nodes)
        old_nbrs[oc] = {new_n2c[n] for n in nodes if n in new_n2c}
    # logger.info(old_nbrs)
    new_nbrs = dict()
    for nc, nodes in new_clustering.items():
        new_nbrs[nc] = {old_n2c[n] for n in nodes if n in old_n2c}
    # logger.info(new_nbrs)

    """
    2. Perform bipartite connected components labeling, preserving old
    and new in separate sets. Generate a change from each component.
    """
    clustering_changes = []
    old_visited = {oc: False for oc in old_clustering}
    new_visited = {nc: False for nc in new_clustering}
    for o, v in old_visited.items():
        if v:
            continue
        old_set, new_set = bipartite_cc(
            old_visited, old_nbrs, new_visited, new_nbrs, set([o])
        )
        old_sub_clustering = {oc: old_clustering[oc] for oc in old_set}
        new_sub_clustering = {nc: new_clustering[nc] for nc in new_set}
        cc = clustering_change(old_sub_clustering, new_sub_clustering)
        clustering_changes.append(cc)

    """
    3. Handle the special case of a new cluster, which is detected
    when there is no old cluster corresponding to a particular new
    one.
    """
    for n, v in new_visited.items():
        if not v:
            old_sub_clustering = dict()
            new_sub_clustering = {n: new_clustering[n]}
            cc = clustering_change(old_sub_clustering, new_sub_clustering)
            assert cc.change_type == 'New'
            clustering_changes.append(cc)

    return clustering_changes


def write_changes_to_log(s, edges):
    logger.info(s)
    if len(edges) == 0:
        logger.info('   -none-')
    else:
        for e in edges:
            logger.info('   %s' % str(e))


def compare_to_other_clustering(new_cl, new_n2c, other_cl, G):
    logger.info('===================================')
    logger.info('Analyzing difference with other clustering')
    nodes = sorted(new_n2c.keys())
    other_cl = ct.extract_subclustering(nodes, other_cl)
    other_n2c = ct.build_node_to_cluster_mapping(other_cl)
    differences = find_changes(
        other_cl,
        other_n2c,
        new_cl,
        new_n2c,
    )

    for cc in differences:
        if cc.change_type == 'Unchanged':
            continue
        logger.info('...')
        c_internal = []  # edges internal to a cluster in both
        c_between = []  # edges between clusters in both
        i_new_internal = []  # edges internal in new clustering, between in other
        i_new_between = []  # edges between in new clustering, internal in other
        new_score = ct.cid_list_score(
            G, cc.new_clustering, new_n2c, list(cc.new_clustering.keys())
        )
        logger.info('New score %d, new clustering %a' % (new_score, cc.new_clustering))
        other_score = ct.cid_list_score(
            G, cc.old_clustering, other_n2c, list(cc.old_clustering.keys())
        )
        logger.info(
            'Other score %d, other clustering %a' % (other_score, cc.old_clustering)
        )

        cc_nodes = sorted(set.union(*cc.new_clustering.values()))
        for i, ni in enumerate(cc_nodes):
            for j in range(i + 1, len(cc_nodes)):
                nj = cc_nodes[j]
                if nj not in G[ni]:  # no edge
                    continue

                e = (ni, nj, G[ni][nj]['weight'])
                same_in_new = new_n2c[ni] == new_n2c[nj]
                same_in_other = other_n2c[ni] == other_n2c[nj]
                if same_in_new and same_in_other:
                    c_internal.append(e)
                elif not same_in_new and not same_in_other:
                    c_between.append(e)
                elif same_in_new and not same_in_other:
                    i_new_internal.append(e)
                else:
                    i_new_between.append(e)
        write_changes_to_log('Edges internal to a cluster in both', c_internal)
        write_changes_to_log('Edges between clusters in both', c_between)
        write_changes_to_log(
            'Edges internal in new clustering, between in other', i_new_internal
        )
        write_changes_to_log(
            'Edges between in new clustering, internal in other', i_new_between
        )


# ==============================


def test_bipartite_cc():
    """
    Test the bipartite CC algorithm above. Nodes labeled with a letter are
    considered on the side called 'old' while nodes labeled with a digit
    are considered 'new'.  old_nbrs has the outgoing edges from the old partition,
    while new_nbrs has the outgoing edges for the new partition. One could
    easily be generated from the other, of course.
    """
    old_visited = {
        'a': False,
        'b': False,
        'c': False,
        'd': False,
        'e': False,
        'f': False,
        'g': False,
    }
    old_nbrs = {
        'a': set([0]),
        'b': set([0]),
        'c': set([2, 3]),
        'd': set([3]),
        'e': set([3, 4]),
        'f': set([5]),
        'g': set(),
    }
    new_visited = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
    new_nbrs = {
        0: set(['a', 'b']),
        1: set(),
        2: set(['c']),
        3: set(['c', 'd', 'e']),
        4: set(['e']),
        5: set(['f']),
    }

    assert old_visited.keys() == old_nbrs.keys()
    assert new_visited.keys() == new_nbrs.keys()

    old_sets = []
    new_sets = []

    logger.info('\ntest_bipartite_cc:')
    for o, v in old_visited.items():
        if not v:
            old_set, new_set = bipartite_cc(
                old_visited, old_nbrs, new_visited, new_nbrs, set([o])
            )
            old_sets.append(old_set)
            new_sets.append(new_set)

    exp_old_sets = [set(['a', 'b']), set(['c', 'd', 'e']), set(['f']), set(['g'])]
    exp_new_sets = [set([0]), set([2, 3, 4]), set([5]), set([])]

    for i in range(len(old_sets)):
        logger.info('.........')
        logger.info('Component: %s' % (i,))
        logger.info(
            '%d:old %a, expected old %a, correct %a'
            % (i, old_sets[i], exp_old_sets[i], old_sets[i] == exp_old_sets[i])
        )
        logger.info(
            '%d:new %a, expected new %a, correct %a'
            % (i, new_sets[i], exp_new_sets[i], new_sets[i] == exp_new_sets[i])
        )


def test_find_changes():
    logger.info('\ntest_find_changes:')
    old_clustering = {
        0: set(['e']),
        1: set(['f', 'g']),
        2: set(['h', 'i']),
        3: set(['j', 'k']),
        4: set(['l']),
        5: set(['m', 'n', 'o', 'p']),
        6: set(['q']),
        7: set(['r', 's']),
        8: set(['t', 'u']),
    }
    old_n2c = ct.build_node_to_cluster_mapping(old_clustering)
    new_clustering = {
        100: set(['a', 'b']),
        101: set(['f', 'g']),
        102: set(['h', 'c']),
        103: set(['j', 'k', 'l', 'd']),
        104: set(['m']),
        105: set(['n']),
        106: set(['o', 'p']),
        107: set(['q', 'r']),
        108: set(['s', 't', 'u', 'x', 'y']),
    }
    new_n2c = ct.build_node_to_cluster_mapping(new_clustering)

    correct_types = [
        'Removed',
        'Unchanged',
        'Extension',
        'Merge',
        'Split',
        'Merge/Split',
        'New',
    ]
    changes = find_changes(old_clustering, old_n2c, new_clustering, new_n2c)
    for c, t in zip(changes, correct_types):
        logger.info('..........')
        c.print_it()
        logger.info('Correct change type? %s' % (t == c.change_type,))


def test_compare_to_other_clustering():

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

    new_cl = {
        '0': set(['a', 'b', 'd', 'e']),
        '1': set(['c']),
        '2': set(['h', 'i']),
        '3': set(['f', 'g', 'j', 'k']),
    }
    new_n2c = ct.build_node_to_cluster_mapping(new_cl)
    other_cl = {
        '5': set(['a', 'b', 'd', 'h']),
        '6': set(['f', 'g', 'k']),
        '7': set(['j']),
        '8': set(['c']),
        '9': set(['e', 'i']),
        '10': set(['m', 'n']),
    }
    compare_to_other_clustering(new_cl, new_n2c, other_cl, G)


if __name__ == '__main__':
    test_bipartite_cc()
    test_find_changes()
    test_compare_to_other_clustering()
