"""Step 3: choosing which annotation pairs to put to a reviewer.

A candidate is a pair whose stability is below beta, together with what it
would cost to be wrong about it. There are three kinds, and the paper's
ablation measures each one:

* **internal**, an intra-PCC negative edge whose endpoints the positive edges
  nonetheless hold together, which review can split;
* **weak positive**, the weakest edge of a PCC's maximum spanning tree, offered
  because a PCC held together only by weak positives is unstable with no
  negative edge to flag it;
* **external**, a pair of PCCs whose separation is not confidently drawn, which
  review can merge. On the paper's datasets these take between 80 and 97 per
  cent of the review seats.

:func:`pools` ranks them and returns them; it selects nothing. Which candidates
reach the reviewer, under what budget and in what priority order, is the
algorithm's decision, in :mod:`beta_stability.stability.algorithm`.

What a candidate carries beyond its stability is ``structural_impact``: how
many annotations would change hands if the review went the other way, which is
the number of nodes on the smaller side of the edge within its PCC.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from beta_stability.stability.graph import EdgeLabel

logger = logging.getLogger("beta_stability")


@dataclass
class StabilityCandidate:
    """A candidate for human review."""
    candidate_type: str  # "INTERNAL" or "EXTERNAL"
    stability: float
    review_edge: Tuple[int, int]  # Edge to present to human
    structural_impact: float = 1.0  # min(subtree_left, subtree_right) for splits, min(|pcc_a|, |pcc_b|) for merges
    # For INTERNAL: the unstable pair
    node_pair: Optional[Tuple[int, int]] = None
    pcc_id: Optional[int] = None
    # For EXTERNAL: the two PCCs
    pcc_pair: Optional[Tuple[int, int]] = None


class RootedTree:
    """A PCC's maximum spanning tree, rooted, with what each cut would cost.

    Rooting it once per PCC gives both the path between any two of its nodes,
    walked from the parent pointers, and the size of the smaller side of any
    edge in it, which is a candidate's structural impact.
    """

    def __init__(self, tree, root):
        self.tree = tree
        self.size = tree.number_of_nodes()
        self.parent = {}
        self.depth = {root: 0}

        order, stack = [], [(root, None)]
        while stack:
            node, parent = stack.pop()
            self.parent[node] = parent
            order.append(node)
            stack.extend((other, node) for other in tree.neighbors(node) if other != parent)
        for node in order:
            if self.parent[node] is not None:
                self.depth[node] = self.depth[self.parent[node]] + 1

        self.subtree_size = dict.fromkeys(order, 1)
        for node in reversed(order):
            if self.parent[node] is not None:
                self.subtree_size[self.parent[node]] += self.subtree_size[node]

    def holds(self, node):
        return node in self.depth

    def path(self, start, end):
        """The unique path between two nodes, by climbing to their ancestor."""
        up_from_start, up_from_end = [start], [end]
        left, right = start, end
        while self.depth[left] > self.depth[right]:
            left = self.parent[left]
            up_from_start.append(left)
        while self.depth[right] > self.depth[left]:
            right = self.parent[right]
            up_from_end.append(right)
        while left != right:
            left = self.parent[left]
            up_from_start.append(left)
            right = self.parent[right]
            up_from_end.append(right)
        return up_from_start + up_from_end[-2::-1]

    def weakest_edge_on(self, path):
        """The weakest edge of a path, and its confidence."""
        weakest, confidence = None, float('inf')
        for first, second in itertools.pairwise(path):
            weight = self.tree[first][second]['weight']
            if weight < confidence:
                weakest, confidence = (first, second), weight
        return weakest, confidence

    def weakest_edge(self):
        """The weakest edge of the whole tree, which is its lowest stability."""
        weakest, confidence = None, float('inf')
        for first, second, attr in self.tree.edges(data=True):
            weight = attr.get('weight', 1.0)
            if weight < confidence:
                weakest, confidence = (first, second), weight
        return weakest, confidence

    def cut_impact(self, edge):
        """How many annotations change hands if this edge is cut."""
        if edge is None:
            return 1.0
        first, second = edge
        child = second if self.parent.get(second) == first else (
            first if self.parent.get(first) == second else second)
        held = self.subtree_size.get(child, 1)
        return max(float(min(held, self.size - held)), 1.0)


def strongest_crossing_edge(graph, label):
    """Per PCC pair, the strongest edge of this label that crosses between them.

    Returns {(first_pcc, second_pcc): (confidence, edge)}. Two orderings here
    are load-bearing and both come from the edge table being built in the
    graph's own edge order: within a pair the first edge to reach the maximum
    confidence wins, since that edge is the one sent for review, and the pairs
    themselves are keyed in order of first appearance, since that decides the
    order candidates are generated in.
    """
    table, pcc_u, pcc_v, _ = graph.edge_pcc_ids()
    if len(table) == 0:
        return {}
    crossing = (pcc_u >= 0) & (pcc_v >= 0) & (pcc_u != pcc_v) & table.with_label(label)
    rows = np.flatnonzero(crossing)
    if rows.size == 0:
        return {}

    low = np.minimum(pcc_u, pcc_v)
    high = np.maximum(pcc_u, pcc_v)
    pair_key = (low * (int(high.max()) + 1) + high)[rows]
    confidence = table.confidence[rows]

    # strongest edge per pair, earliest row breaking ties
    by_strength = np.lexsort((rows, -confidence, pair_key))
    _, strongest = np.unique(pair_key[by_strength], return_index=True)
    best_rows = rows[by_strength[strongest]]

    # and each pair's first appearance, which is the order to return them in
    by_position = np.lexsort((rows, pair_key))
    _, earliest = np.unique(pair_key[by_position], return_index=True)
    first_rows = rows[by_position[earliest]]

    return {(int(low[row]), int(high[row])):
            (float(table.confidence[row]), (int(table.u[row]), int(table.v[row])))
            for row in best_rows[np.argsort(first_rows, kind='stable')]}


def pools(graph, alpha, include_weak_positive=False, review_positive_inactive=False):
    """Step 3: the candidates whose stability is below alpha, ranked.

    The caller composes the review batch from these pools, applying its own
    priority order, per-PCC conflict rule and budget cap.

    Args:
        alpha: stability threshold; candidates at or above it are dropped.
        include_weak_positive: also offer each PCC's weakest positive MST
            edge, which lets a PCC held together only by weak positives be
            split even with no internal negative to flag it.
        review_positive_inactive: also offer the strongest cross-PCC
            positive-inactive edge when that is the side making the pair
            unstable, which is the merge dual of the internal selector.

    Returns:
        Dict with keys 'internal' and 'external', each a list of
        StabilityCandidate, most unstable first.
    """
    pccs = graph.pccs()
    forest = graph.mst_forest()
    # Every PCC's intra-PCC edges in one pass, in ascending (u, v) order, which
    # is the order the maximum spanning tree below needs.
    intra_pcc = graph.intra_pcc_edges()

    # Internal candidates come from a pass over each PCC, external ones from a
    # single scan of the edges that cross between PCCs.
    internal_pool: List[StabilityCandidate] = []
    external_pool: List[StabilityCandidate] = []

    for pcc_id, pcc in enumerate(pccs):
        if len(pcc) < 2:
            continue

        mst = forest.subgraph(pcc).copy()
        if mst.number_of_edges() == 0:
            continue
        rooted = RootedTree(mst, next(iter(pcc)))

        # A pair with no negative edge between them has stability equal to
        # the strength of their maximum strength path, so a PCC held
        # together only by weak positives is unstable with nothing to flag
        # it. Offer its weakest MST edge, which is its lowest internal
        # stability. The negative-driven candidates below are more
        # unstable, so they win the seat where a PCC has both.
        if include_weak_positive:
            weakest, confidence = rooted.weakest_edge()
            if weakest is not None and confidence < alpha:
                internal_pool.append(StabilityCandidate(
                    candidate_type="WEAK_POSITIVE",
                    stability=confidence,
                    review_edge=weakest,
                    structural_impact=rooted.cut_impact(weakest),
                    node_pair=weakest,
                    pcc_id=pcc_id,
                ))

        # The stability of a negative edge's endpoints is the strength of the
        # path holding them together, minus the negative's own confidence.
        negative_edges = intra_pcc.get(pcc_id, ([], []))[0]
        if not negative_edges:
            continue

        for u, v, negative_confidence in negative_edges:
            if not (rooted.holds(u) and rooted.holds(v)):
                continue
            path = rooted.path(u, v)
            if len(path) < 2:
                continue
            weakest, strength = rooted.weakest_edge_on(path)
            stability = strength - negative_confidence
            if stability < alpha:
                internal_pool.append(StabilityCandidate(
                    candidate_type="INTERNAL",
                    stability=stability,
                    review_edge=(u, v),
                    structural_impact=rooted.cut_impact(weakest),
                    node_pair=(u, v),
                    pcc_id=pcc_id,
                ))
    # External candidates: a PCC pair with a crossing negative edge has an
    # external stability whether or not it also has a positive-inactive one,
    # whose confidence is then taken as zero.
    strongest_negative = strongest_crossing_edge(graph, EdgeLabel.NEGATIVE)
    strongest_inactive = strongest_crossing_edge(graph, EdgeLabel.POSITIVE_INACTIVE)
    # A PCC pair is a candidate when its external stability is below alpha.
    # What gets reviewed is the decisive side: normally the strongest
    # crossing negative, but when review_positive_inactive is on and the
    # strongest positive-inactive rivals it, that edge instead, since
    # confirming it reactivates the edge and merges the two PCCs. Pairs
    # joined only by a positive-inactive edge are reachable only then, and
    # are appended after the negative-driven ones.
    pairs = list(strongest_negative)
    if review_positive_inactive:
        pairs += [pair for pair in strongest_inactive if pair not in strongest_negative]

    for pcc_pair in pairs:
        negative = strongest_negative.get(pcc_pair)
        inactive = strongest_inactive.get(pcc_pair)
        negative_confidence = negative[0] if negative else 0.0
        inactive_confidence = inactive[0] if inactive else 0.0
        stability = negative_confidence - inactive_confidence
        if stability >= alpha:
            continue

        if (review_positive_inactive and inactive is not None
                and inactive_confidence >= negative_confidence):
            review_edge = inactive[1]
        elif negative is not None:
            review_edge = negative[1]
        else:
            continue
        first_pcc, second_pcc = pcc_pair
        impact = float(min(len(pccs[first_pcc]), len(pccs[second_pcc])))
        external_pool.append(StabilityCandidate(
            candidate_type="EXTERNAL",
            stability=stability,
            review_edge=review_edge,
            structural_impact=max(impact, 1.0),
            pcc_pair=pcc_pair
        ))

    # Most unstable first in both pools, so the caller can interleave them.
    # The internal pool mixes negative-driven and weak-positive candidates, and
    # this is what makes a PCC's negative-driven one take its seat.
    internal_pool.sort(key=lambda candidate: candidate.stability)
    external_pool.sort(key=lambda candidate: candidate.stability)
    logger.info(f"Generated candidate pools: {len(internal_pool)} internal/split, "
                f"{len(external_pool)} external/merge")

    return {'internal': internal_pool, 'external': external_pool}
