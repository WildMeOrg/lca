"""PCC separation strength, against the obvious implementation.

``StabilityGraph.pcc_separation_strength`` sums, for each PCC, the strongest
negative confidence to every other PCC it has a negative edge to. It used to be
a walk over each node's adjacency, and was replaced by a vectorised read of the
graph's :class:`EdgeTable` because the walk cost around 94 seconds per outer
iteration. That rewrite shipped with its own correctness check inside the
algorithm, gated on a ``BETA_SELFCHECK`` environment variable, which meant it
ran only when somebody remembered to set it. The check lives here instead:
:func:`reference_strength` is the walk it replaced.

The table is maintained incrementally as edges are added and relabelled, so
``verify_edge_table`` rebuilding it from the graph and finding the same thing is
part of the same claim.
"""

import random

import pytest

from beta_stability.stability.graph import EdgeLabel, StabilityGraph


def build(positive, negative):
    """A graph holding exactly these labelled edges."""
    graph = StabilityGraph()
    for (u, v), confidence in positive.items():
        graph.add_edge(u, v, EdgeLabel.POSITIVE, confidence)
    for (u, v), confidence in negative.items():
        graph.add_edge(u, v, EdgeLabel.NEGATIVE, confidence)
    return graph


def reference_strength(graph):
    """The per-adjacency walk the vectorised version replaced."""
    node_to_pcc = {node: pcc_id for pcc_id, pcc in enumerate(graph.pccs())
                   for node in pcc}

    strongest = {}
    for pcc_id, pcc in enumerate(graph.pccs()):
        for node in pcc:
            for neighbour in graph.G.neighbors(node):
                neighbour_pcc = node_to_pcc.get(neighbour)
                if neighbour_pcc is None or neighbour_pcc == pcc_id:
                    continue
                edge = graph.G[node][neighbour].get('data')
                if edge and edge.label == EdgeLabel.NEGATIVE:
                    pair = (min(pcc_id, neighbour_pcc), max(pcc_id, neighbour_pcc))
                    strongest[pair] = max(strongest.get(pair, 0.0), edge.confidence)

    result = {}
    for (a, b), confidence in strongest.items():
        result[a] = result.get(a, 0.0) + confidence
        result[b] = result.get(b, 0.0) + confidence
    return result


def by_representative(graph, strength, nodes):
    """Re-key a strength map by a node of each PCC, since PCC ids are assigned."""
    pcc_of = {node: pcc_id for pcc_id, pcc in enumerate(graph.pccs()) for node in pcc}
    return {node: strength.get(pcc_of[node], 0.0) for node in nodes}


def test_each_pcc_pair_contributes_its_strongest_negative():
    """Three PCCs: A--B have two negatives, A--C one, B--C none."""
    graph = build(positive={(0, 1): 0.9, (0, 6): 0.9, (2, 3): 0.9, (4, 5): 0.9},
                      negative={(0, 2): 0.4, (1, 3): 0.7,   # A--B, strongest 0.7
                                (0, 4): 0.5,                # A--C
                                (1, 6): 0.8})               # inside A, not a separation

    strength = by_representative(graph,
                                 graph.pcc_separation_strength(),
                                 nodes=[0, 2, 4])
    assert strength[0] == pytest.approx(1.2), 'A is separated from both B and C'
    assert strength[2] == pytest.approx(0.7), 'B only from A, by the stronger edge'
    assert strength[4] == pytest.approx(0.5), 'C only from A'


def test_a_graph_with_no_crossing_negatives_has_no_strength():
    graph = build(positive={(0, 1): 0.9, (2, 3): 0.9}, negative={})
    assert graph.pcc_separation_strength() == {}


def test_an_empty_graph_has_no_strength():
    graph = build(positive={}, negative={})
    assert graph.pcc_separation_strength() == {}


@pytest.mark.parametrize('seed', range(8))
def test_the_vectorised_strength_matches_the_adjacency_walk(seed):
    """The check that used to live behind BETA_SELFCHECK, on random graphs."""
    rng = random.Random(seed)
    nodes = range(40)
    positive, negative = {}, {}
    for _ in range(120):
        u, v = rng.sample(sorted(nodes), 2)
        edge = (min(u, v), max(u, v))
        if edge in positive or edge in negative:
            continue
        target = positive if rng.random() < 0.35 else negative
        target[edge] = round(rng.uniform(0.05, 1.0), 3)

    graph = build(positive, negative)
    vectorised = graph.pcc_separation_strength()
    reference = reference_strength(graph)

    assert set(vectorised) == set(reference)
    for pcc_id, strength in reference.items():
        assert vectorised[pcc_id] == pytest.approx(strength, abs=1e-9)


@pytest.mark.parametrize('seed', range(4))
def test_the_incremental_edge_mirror_matches_a_rebuild(seed):
    """Relabelling an edge must keep the columnar mirror in step with the graph."""
    rng = random.Random(seed)
    graph = build(positive={(0, 1): 0.9, (2, 3): 0.9}, negative={(1, 2): 0.6})

    for _ in range(50):
        u, v = rng.sample(range(8), 2)
        label = rng.choice([EdgeLabel.POSITIVE, EdgeLabel.NEGATIVE,
                            EdgeLabel.POSITIVE_INACTIVE])
        graph.add_edge(u, v, label, round(rng.uniform(0.1, 1.0), 3))
        graph.pcc_separation_strength()

    assert graph.verify_edge_table(), 'the edge table drifted from the graph'
