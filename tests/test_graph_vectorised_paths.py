"""The graph's vectorised sweeps, against the obvious implementations.

Several steps of the algorithm used to walk networkx adjacency dictionaries in
Python, which cost around three microseconds per edge visit and so tens of
seconds per pass on the larger datasets. Each was replaced by a vectorised read
of the :class:`EdgeTable`, and each rewrite shipped with a correctness check
inside the method it replaced, gated on a ``BETA_SELFCHECK`` environment
variable. There were nine of them, 255 lines, and they ran only when somebody
remembered to set the variable.

They live here instead. Every ``reference_*`` function below is the slow
implementation that the fast path replaced, and the claim in each case is not
merely that the values agree but that the *order* does, because edge insertion
order decides how networkx breaks ties between equal-weight MST edges, and the
candidate order decides which pairs are sent for review.
"""

import random

import networkx as nx
import pytest

from beta_stability.stability import candidates
from beta_stability.stability.candidates import StabilityCandidate
from beta_stability.stability.graph import EdgeLabel, StabilityGraph

ALPHA = 0.5


def graph_with(positive, negative, positive_inactive=()):
    """A graph holding exactly these labelled edges."""
    graph = StabilityGraph()
    for label, edges in ((EdgeLabel.POSITIVE, positive),
                         (EdgeLabel.NEGATIVE, negative),
                         (EdgeLabel.POSITIVE_INACTIVE, positive_inactive)):
        for (u, v), confidence in dict(edges).items():
            graph.add_edge(u, v, label, confidence)
    return graph


@pytest.fixture(params=range(6))
def graph(request):
    """A random labelled graph, dense enough to give PCCs of several nodes."""
    rng = random.Random(request.param)
    positive, negative, inactive = {}, {}, {}
    for _ in range(140):
        u, v = rng.sample(range(32), 2)
        edge = (min(u, v), max(u, v))
        if edge in positive or edge in negative or edge in inactive:
            continue
        roll = rng.random()
        target = positive if roll < 0.4 else (negative if roll < 0.85 else inactive)
        target[edge] = round(rng.uniform(0.05, 1.0), 3)
    return graph_with(positive, negative, inactive)


def pcc_of_node(graph):
    """Node to PCC id, built from the PCCs themselves rather than borrowed."""
    return {node: pcc_id for pcc_id, pcc in enumerate(graph.pccs()) for node in pcc}


def pcc_pairs_by_label(graph, label):
    """Per cross-PCC pair, the strongest edge of this label, in discovery order."""
    pcc_of = pcc_of_node(graph)
    best = {}
    for u, v, attr in graph.G.edges(data=True):
        edge = attr.get('data')
        if edge is None or edge.label != label:
            continue
        pcc_u = pcc_of.get(u)
        pcc_v = pcc_of.get(v)
        if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
            continue
        pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
        if pair not in best or edge.confidence > best[pair][0]:
            best[pair] = (edge.confidence, (u, v))
    return best


def intra_pcc_edges_by_pair_scan(graph):
    """Per PCC, its negative and positive edges, found by trying every pair."""
    result = {}
    for pcc_id, pcc in enumerate(graph.pccs()):
        if len(pcc) < 2:
            continue
        members = sorted(pcc)
        negatives, positives = [], []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                if not graph.G.has_edge(u, v):
                    continue
                edge = graph.G[u][v].get('data')
                if edge is None:
                    continue
                if edge.label == EdgeLabel.NEGATIVE:
                    negatives.append((u, v, edge.confidence))
                elif edge.label == EdgeLabel.POSITIVE:
                    positives.append((u, v, edge.confidence))
        if negatives or positives:
            result[pcc_id] = (negatives, positives)
    return result


def mst_of(edges):
    """The maximum spanning tree networkx builds from these weighted edges."""
    reference = nx.Graph()
    reference.add_edges_from(edges)
    return nx.maximum_spanning_tree(reference, weight='weight')


# --------------------------------------------------------------------------
# PCCs and the MST forest
# --------------------------------------------------------------------------

def test_pccs_are_the_positive_subgraphs_components(graph):
    positive = [(u, v) for u, v, attr in graph.G.edges(data=True)
                if attr.get('data') and attr['data'].label == EdgeLabel.POSITIVE]
    reference = nx.Graph()
    reference.add_nodes_from(graph.G.nodes())
    reference.add_edges_from(positive)

    expected = [list(c) for c in nx.connected_components(reference)]
    assert [list(p) for p in graph.get_pccs()] == expected


def test_the_mst_forest_is_the_maximum_spanning_tree_of_the_positives(graph):
    produced = graph.mst_forest()
    reference = nx.Graph()
    reference.add_nodes_from(graph.G.nodes())
    reference.add_edges_from((u, v, {'weight': attr['data'].confidence})
                             for u, v, attr in graph.G.edges(data=True)
                             if attr.get('data') and attr['data'].label == EdgeLabel.POSITIVE)
    expected = nx.maximum_spanning_tree(reference, weight='weight')
    assert list(produced.edges(data='weight')) == list(expected.edges(data='weight'))


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------

def reference_per_pcc_internal_stability(graph):
    """(anchor, size, minimum internal stability) per PCC, by trying every pair."""
    result = []
    for pcc in graph.pccs():
        members = sorted(pcc)
        anchor = members[0]
        if len(members) < 2:
            result.append((anchor, len(members), None))
            continue
        negatives, positives = [], []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                if not graph.G.has_edge(u, v):
                    continue
                edge = graph.G[u][v].get('data')
                if edge and edge.label == EdgeLabel.NEGATIVE:
                    negatives.append((u, v, edge.confidence))
                elif edge and edge.label == EdgeLabel.POSITIVE:
                    positives.append((u, v, edge.confidence))
        if not negatives or not positives:
            result.append((anchor, len(members), None))
            continue
        tree = mst_of((u, v, {'weight': c}) for u, v, c in positives)
        best = float('inf')
        for u, v, negative_confidence in negatives:
            if not (tree.has_node(u) and tree.has_node(v)):
                continue
            try:
                path = nx.shortest_path(tree, u, v)
            except nx.NetworkXNoPath:
                continue
            if len(path) < 2:
                continue
            weakest = min(tree[path[k]][path[k + 1]]['weight'] for k in range(len(path) - 1))
            best = min(best, weakest - negative_confidence)
        result.append((anchor, len(members), best if best != float('inf') else None))
    return result


def test_per_pcc_internal_stability_matches_the_all_pairs_computation(graph):
    produced = graph.per_pcc_internal_stability()
    expected = reference_per_pcc_internal_stability(graph)

    assert len(produced) == len(expected)
    for (anchor, size, stability), (ref_anchor, ref_size, ref_stability) in zip(
            produced, expected, strict=True):
        assert (anchor, size) == (ref_anchor, ref_size)
        if ref_stability is None:
            assert stability is None
        else:
            assert stability == pytest.approx(ref_stability, abs=1e-9)


def test_external_stabilities_match_the_adjacency_scan(graph):
    negatives = pcc_pairs_by_label(graph, EdgeLabel.NEGATIVE)
    inactive = pcc_pairs_by_label(graph, EdgeLabel.POSITIVE_INACTIVE)
    expected = sorted(confidence - inactive.get(pair, (0.0, None))[0]
                      for pair, (confidence, _) in negatives.items())

    produced = sorted(graph.external_stabilities())
    assert len(produced) == len(expected)
    for value, reference in zip(produced, expected, strict=True):
        assert value == pytest.approx(reference, abs=1e-9)


def test_graph_stats_minimum_internal_stability_matches(graph):
    """get_graph_stats used to compute this a second way, by its own path walk."""
    expected = min((stability for _, _, stability
                    in reference_per_pcc_internal_stability(graph)
                    if stability is not None), default=float('inf'))
    produced = graph.get_graph_stats()['min_internal_stability']
    if expected == float('inf'):
        assert produced == float('inf')
    else:
        assert produced == pytest.approx(expected, abs=1e-9)


def test_graph_stats_minimum_external_stability_matches(graph):
    negatives = pcc_pairs_by_label(graph, EdgeLabel.NEGATIVE)
    inactive = pcc_pairs_by_label(graph, EdgeLabel.POSITIVE_INACTIVE)
    expected = min((confidence - inactive.get(pair, (0.0, None))[0]
                    for pair, (confidence, _) in negatives.items()), default=float('inf'))

    assert graph.get_graph_stats()['min_external_stability'] == pytest.approx(expected, abs=1e-9)


def test_intra_pcc_edges_match_the_pair_scan(graph):
    assert graph.intra_pcc_edges() == intra_pcc_edges_by_pair_scan(graph)


# --------------------------------------------------------------------------
# the sweep order that decides MST tie-breaking
# --------------------------------------------------------------------------

def reference_sweep_order(graph):
    """Per PCC, the positive and negative edges in historical adjacency order.

    make_zero_stable built each PCC's subgraph with
    ``for a in pcc_set: for b in graph[a]`` keeping b > a, and networkx breaks
    ties between equal-weight MST edges by insertion order, so this order is
    part of the result.
    """
    expected = {}
    for pcc_id, pcc in enumerate(graph.pccs()):
        positives, negatives = [], []
        for a in pcc:
            for b, attr in graph.G[a].items():
                if b <= a or b not in pcc:
                    continue
                edge = attr.get('data')
                if edge is None:
                    continue
                if edge.label == EdgeLabel.POSITIVE:
                    positives.append((a, b, edge.confidence))
                elif edge.label == EdgeLabel.NEGATIVE:
                    negatives.append((a, b, edge.confidence))
        if positives or negatives:
            expected[pcc_id] = (positives, negatives)
    return expected


def test_the_sweep_order_reproduces_the_historical_adjacency_walk(graph):
    produced = graph._intra_edges_sweep_order()
    expected = reference_sweep_order(graph)

    assert set(produced) == set(expected)
    for pcc_id, (pos_lo, pos_hi, pos_conf, neg_lo, neg_hi, neg_conf) in produced.items():
        positives, negatives = expected[pcc_id]
        assert list(zip(pos_lo.tolist(), pos_hi.tolist(), strict=True)) == \
            [(a, b) for a, b, _ in positives]
        assert list(zip(neg_lo.tolist(), neg_hi.tolist(), strict=True)) == \
            [(a, b) for a, b, _ in negatives]
        for produced_conf, (_, _, reference_conf) in zip(pos_conf.tolist(), positives,
                                                         strict=True):
            assert produced_conf == pytest.approx(reference_conf)
        for produced_conf, (_, _, reference_conf) in zip(neg_conf.tolist(), negatives,
                                                         strict=True):
            assert produced_conf == pytest.approx(reference_conf)


# --------------------------------------------------------------------------
# the candidate pools
# --------------------------------------------------------------------------

def reference_internal_pool(graph, alpha, include_weak_positive):
    """The internal pool, built from each PCC's MST with one path search per pair."""
    pool = []
    for pcc_id, pcc in enumerate(graph.pccs()):
        if len(pcc) < 2:
            continue
        tree = graph.mst_forest().subgraph(pcc).copy()
        if tree.number_of_edges() == 0:
            continue

        size = len(pcc)
        parent, order, stack = {}, [], [(next(iter(pcc)), None)]
        while stack:
            node, node_parent = stack.pop()
            parent[node] = node_parent
            order.append(node)
            stack.extend((other, node) for other in tree.neighbors(node) if other != node_parent)
        subtree = dict.fromkeys(order, 1)
        for node in reversed(order):
            if parent[node] is not None:
                subtree[parent[node]] += subtree[node]

        def impact_of(edge, parent=parent, subtree=subtree, size=size):
            """How many nodes the smaller side holds if this tree edge is cut."""
            first, second = edge
            child = second if parent.get(second) == first else (
                first if parent.get(first) == second else second)
            held = subtree.get(child, 1)
            return float(min(held, size - held))

        if include_weak_positive:
            weakest, weakest_confidence = None, float('inf')
            for u, v, attr in tree.edges(data=True):
                confidence = attr.get('weight', 1.0)
                if confidence < weakest_confidence:
                    weakest_confidence, weakest = confidence, (u, v)
            if weakest is not None and weakest_confidence < alpha:
                pool.append(StabilityCandidate(
                    candidate_type="WEAK_POSITIVE", stability=weakest_confidence,
                    review_edge=weakest, structural_impact=max(impact_of(weakest), 1.0),
                    node_pair=weakest, pcc_id=pcc_id))

        negatives = [(u, v, graph.G[u][v]['data'].confidence)
                     for u, v, _ in intra_pcc_edges_by_pair_scan(graph).get(pcc_id, ([], []))[0]]
        if not negatives:
            continue
        paths = dict(nx.all_pairs_shortest_path(tree))
        for u, v, negative_confidence in negatives:
            path = paths.get(u, {}).get(v)
            if path is None or len(path) < 2:
                continue
            weakest_confidence, weakest = float('inf'), None
            for k in range(len(path) - 1):
                confidence = tree[path[k]][path[k + 1]]['weight']
                if confidence < weakest_confidence:
                    weakest_confidence, weakest = confidence, (path[k], path[k + 1])
            stability = weakest_confidence - negative_confidence
            if stability < alpha:
                pool.append(StabilityCandidate(
                    candidate_type="INTERNAL", stability=stability, review_edge=(u, v),
                    structural_impact=max(impact_of(weakest) if weakest else 1.0, 1.0),
                    node_pair=(u, v), pcc_id=pcc_id))
    # The production pool is generation order, stably sorted by stability.
    return sorted(pool, key=lambda candidate: candidate.stability)


@pytest.mark.parametrize('include_weak_positive', [False, True])
def test_the_internal_pool_matches_the_per_pair_path_search(graph, include_weak_positive):
    """Order matters as much as content: it decides which pairs get review seats."""
    pools = candidates.pools(graph, alpha=ALPHA,
                             include_weak_positive=include_weak_positive)
    expected = reference_internal_pool(graph, ALPHA, include_weak_positive)
    assert pools['internal'] == expected


def test_the_cross_pcc_maxima_agree_including_their_order(graph):
    """The pair order drives candidate order, so the keys must match in sequence."""
    produced_negative, produced_inactive = graph._cross_pcc_pair_maxima()
    expected_negative = pcc_pairs_by_label(graph, EdgeLabel.NEGATIVE)
    expected_inactive = pcc_pairs_by_label(graph, EdgeLabel.POSITIVE_INACTIVE)

    assert list(produced_negative) == list(expected_negative)
    assert list(produced_inactive) == list(expected_inactive)
    for pair, (confidence, _) in expected_negative.items():
        assert produced_negative[pair] == pytest.approx(confidence)
    for pair, (confidence, _) in expected_inactive.items():
        assert produced_inactive[pair] == pytest.approx(confidence)
