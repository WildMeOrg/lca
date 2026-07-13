"""
Stability Graph Module for Beta Stability (stability-driven) algorithm.

Implements the graph structure and stability computations as defined in:
"LCA V2 Formulation" by Charles Stewart, January 2026

Key definitions from the PDF:
- Edge labels: positive, positive-inactive, negative, incomparable
- PCCs: Connected components using ONLY positive edges (not positive-inactive)
- MSP: Maximum strength path = path with highest minimum edge confidence
- Internal stability(u,v) = MSP(u,v) if no negative edge, else MSP(u,v) - neg_conf
- External stability(A,B) = max_neg_conf - max_pos_inactive_conf
- alpha-stable: min(internal, external) >= alpha

CRITICAL: Only positive edges can be deactivated (become positive-inactive).
Negative edges are NEVER deactivated.
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from beta_stability.util.tools import order_edge

logger = logging.getLogger("beta_stability")


class EdgeLabel(Enum):
    """Edge label types."""
    POSITIVE = "positive"
    POSITIVE_INACTIVE = "positive-inactive"
    NEGATIVE = "negative"


@dataclass
class EdgeData:
    """Edge attributes."""
    label: EdgeLabel
    confidence: float  # [0, 1]
    score: float = 0.0  # Original embedding score
    ranker: str = ""  # Source of classification
    deactivator: Optional[Tuple[int, int]] = None  # Edge that caused deactivation (for positive-inactive)


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


class StabilityGraph:
    """
    Graph structure for Beta Stability stability-driven clustering.

    Per PDF spec:
    - Only POSITIVE edges can be deactivated (become positive-inactive)
    - Negative edges are NEVER deactivated
    - PCCs use only active positive edges
    """

    def __init__(self):
        self.G = nx.Graph()
        self._pcc_cache_valid = False
        self._pccs: List[Set[int]] = []
        self._node_to_pcc: Dict[int, int] = {}

        # MST forest - built explicitly when needed, no caching
        self._mst_forest: Optional[nx.Graph] = None

        # Incremental edge label counts (avoids iterating all edges)
        self._edge_counts = {'positive': 0, 'positive_inactive': 0, 'negative': 0}

        # Track positive-inactive edges for fast external stability computation
        self._pos_inactive_edges: Set[Tuple[int, int]] = set()

    def add_node(self, node_id: int):
        if node_id not in self.G:
            self.G.add_node(node_id)
            self._pcc_cache_valid = False

    def add_edge(self, u: int, v: int, label: EdgeLabel, confidence: float,
                 score: float = 0.0, ranker: str = ""):
        """Add or update an edge."""
        self.add_node(u)
        self.add_node(v)
        confidence = np.clip(confidence, 0, 1)

        if self.G.has_edge(u, v):
            old_data = self.G[u][v].get('data')
            if old_data:
                # Decrement old label count
                self._decrement_edge_count(old_data.label)
                # If label or confidence changed significantly, may need to reactivate
                if old_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    if label == EdgeLabel.POSITIVE:
                        # Reactivating - clear deactivator
                        new_data = EdgeData(label=label, confidence=confidence,
                                          score=score, ranker=ranker, deactivator=None)
                    else:
                        new_data = EdgeData(label=label, confidence=confidence,
                                          score=score, ranker=ranker)
                else:
                    new_data = EdgeData(label=label, confidence=confidence,
                                      score=score, ranker=ranker)
                self.G[u][v]['data'] = new_data
            else:
                new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
                self.G[u][v]['data'] = new_data
        else:
            new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
            self.G.add_edge(u, v, data=new_data)
        # Increment new label count and maintain pos-inactive tracking
        self._increment_edge_count(label)
        key = (min(u, v), max(u, v))
        if label == EdgeLabel.POSITIVE_INACTIVE:
            self._pos_inactive_edges.add(key)
        else:
            self._pos_inactive_edges.discard(key)
        self._pcc_cache_valid = False

    def get_edge(self, u: int, v: int) -> Optional[EdgeData]:
        if self.G.has_edge(u, v):
            return self.G[u][v].get('data')
        return None

    def has_edge(self, u: int, v: int) -> bool:
        return self.G.has_edge(u, v)

    def get_confidence(self, u: int, v: int) -> float:
        if self.G.has_edge(u, v):
            data = self.G[u][v].get('data')
            return data.confidence if data else 0.0
        return 0.0

    def _increment_edge_count(self, label: EdgeLabel):
        if label == EdgeLabel.POSITIVE:
            self._edge_counts['positive'] += 1
        elif label == EdgeLabel.POSITIVE_INACTIVE:
            self._edge_counts['positive_inactive'] += 1
        elif label == EdgeLabel.NEGATIVE:
            self._edge_counts['negative'] += 1

    def _decrement_edge_count(self, label: EdgeLabel):
        if label == EdgeLabel.POSITIVE:
            self._edge_counts['positive'] -= 1
        elif label == EdgeLabel.POSITIVE_INACTIVE:
            self._edge_counts['positive_inactive'] -= 1
        elif label == EdgeLabel.NEGATIVE:
            self._edge_counts['negative'] -= 1

    def deactivate_positive(self, edge: Tuple[int, int], deactivator: Tuple[int, int] = None):
        """
        Deactivate a POSITIVE edge (make it positive-inactive).
        Per PDF: Only positive edges can be deactivated. Negative edges are never deactivated.
        """
        u, v = edge
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v].get('data')
        if not edge_data:
            return

        # Only deactivate POSITIVE edges
        if edge_data.label != EdgeLabel.POSITIVE:
            logger.warning(f"Cannot deactivate non-positive edge ({u}, {v}) with label {edge_data.label}")
            return

        self._edge_counts['positive'] -= 1
        self._edge_counts['positive_inactive'] += 1
        edge_data.label = EdgeLabel.POSITIVE_INACTIVE
        edge_data.deactivator = deactivator
        self._pos_inactive_edges.add((min(u, v), max(u, v)))
        self._pcc_cache_valid = False

    def _invalidate_cache(self):
        """Invalidate PCC cache. MST is rebuilt explicitly when needed."""
        self._pcc_cache_valid = False

    def _ensure_pcc_cache(self):
        if not self._pcc_cache_valid:
            self._compute_pccs()
            self._pcc_cache_valid = True

    def _compute_pccs(self):
        """
        Compute PCCs using ONLY positive edges (not positive-inactive).
        Per PDF: "PCCs (positive edges only, not positive-inactive) correspond to individuals"
        """
        positive_edges = [
            (u, v) for u, v, data in self.G.edges(data=True)
            if data.get('data') and data['data'].label == EdgeLabel.POSITIVE
        ]

        positive_graph = nx.Graph()
        positive_graph.add_nodes_from(self.G.nodes())
        positive_graph.add_edges_from(positive_edges)

        components = list(nx.connected_components(positive_graph))
        self._pccs = [set(comp) for comp in components]
        self._node_to_pcc = {}
        for pcc_id, pcc in enumerate(self._pccs):
            for node in pcc:
                self._node_to_pcc[node] = pcc_id

    def _build_mst_forest(self):
        """
        Build MST forest for all active positive edges.
        Called explicitly at the start of phases that need MST.
        """
        positive_edges = []
        for u, v, data in self.G.edges(data=True):
            edge_data = data.get('data')
            if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                positive_edges.append((u, v, {'weight': edge_data.confidence}))

        self._mst_forest = nx.Graph()
        self._mst_forest.add_nodes_from(self.G.nodes())
        self._mst_forest.add_edges_from(positive_edges)

        # Build MST forest (one MST per connected component)
        self._mst_forest = nx.maximum_spanning_tree(self._mst_forest, weight='weight')

    def sparsify_pccs(self) -> int:
        """
        Sparsify each PCC to its Maximum Spanning Tree.

        For each PCC, keeps only the MST edges as positive and deactivates
        all other positive edges (-> positive-inactive). This ensures PCCs
        are tree-structured, so make_zero_stable converges in one pass.

        Should be called once after initial graph construction (before Phase 0).

        Returns: Number of edges deactivated.
        """
        self._ensure_pcc_cache()

        # Build MST of positive edges
        self._build_mst_forest()
        mst_edges = set()
        for u, v in self._mst_forest.edges():
            mst_edges.add((min(u, v), max(u, v)))

        # Deactivate all positive edges NOT in the MST
        deactivations = 0
        for u, v, data in list(self.G.edges(data=True)):
            edge_data = data.get('data')
            if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                key = (min(u, v), max(u, v))
                if key not in mst_edges:
                    edge_data.label = EdgeLabel.POSITIVE_INACTIVE
                    edge_data.deactivator = None
                    self._pos_inactive_edges.add(key)
                    deactivations += 1

        if deactivations > 0:
            self._edge_counts['positive'] -= deactivations
            self._edge_counts['positive_inactive'] += deactivations
            self._pcc_cache_valid = False

        logger.info(f"Sparsified PCCs: kept {len(mst_edges)} MST edges, "
                    f"deactivated {deactivations} non-MST positive edges")

        return deactivations

    def get_pccs(self) -> List[Set[int]]:
        self._ensure_pcc_cache()
        return self._pccs.copy()

    def get_node_pcc(self, node: int) -> Optional[int]:
        self._ensure_pcc_cache()
        return self._node_to_pcc.get(node)

    def nodes_in_same_pcc(self, u: int, v: int) -> bool:
        self._ensure_pcc_cache()
        pcc_u = self._node_to_pcc.get(u)
        pcc_v = self._node_to_pcc.get(v)
        return pcc_u is not None and pcc_u == pcc_v

    def get_msp_strength(self, u: int, v: int) -> Optional[Tuple[float, List[int], Tuple[int, int]]]:
        """
        Get MSP (Maximum Strength Path) between two nodes in same PCC.

        Returns: (strength, path, min_edge) or None if not in same PCC.

        Per PDF: "The maximum strength path (MSP) between two vertices in same PCC
        is the maximum strength simple path."

        Computed via Maximum Spanning Tree - the path in MST is the widest path.
        """
        self._ensure_pcc_cache()

        pcc_u = self._node_to_pcc.get(u)
        pcc_v = self._node_to_pcc.get(v)

        if pcc_u is None or pcc_u != pcc_v:
            return None

        if u == v:
            return (float('inf'), [u], (u, u))

        # Build MST for this PCC using positive edges
        pcc = self._pccs[pcc_u]
        pcc_graph = nx.Graph()

        for a in pcc:
            for b in self.G.neighbors(a):
                if b in pcc and a < b:
                    edge_data = self.G[a][b].get('data')
                    if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                        pcc_graph.add_edge(a, b, weight=edge_data.confidence)

        if pcc_graph.number_of_edges() == 0:
            return None

        mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

        if not mst.has_node(u) or not mst.has_node(v):
            return None

        try:
            path = nx.shortest_path(mst, u, v)
        except nx.NetworkXNoPath:
            return None

        if len(path) < 2:
            return None

        # Find minimum confidence edge on path
        min_conf = float('inf')
        min_edge = None
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            conf = mst[a][b]['weight']
            if conf < min_conf:
                min_conf = conf
                min_edge = (a, b)

        return (min_conf, path, min_edge)

    def compute_internal_stability(self, u: int, v: int) -> Optional[float]:
        """
        Compute internal stability for vertex pair (u, v) in same PCC.

        Per PDF:
        - MSP(u, v) if there is no negative edge between u and v
        - MSP(u, v) - conf(u, v) if there is a negative edge
        """
        msp_result = self.get_msp_strength(u, v)
        if msp_result is None:
            return None

        msp_strength = msp_result[0]

        # Check for negative edge between u and v
        edge_data = self.get_edge(u, v)
        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
            return msp_strength - edge_data.confidence

        return msp_strength

    def compute_external_stability(self, pcc_a: int, pcc_b: int) -> Optional[float]:
        """
        Compute external stability between two PCCs.

        Per PDF: "The stability of any pair of PCCs is the maximum confidence of any
        negative edges joining the PCCs, minus the maximum confidence of any
        positive-inactive edges joining them."

        Returns None if no negative edge exists between the PCCs.
        """
        self._ensure_pcc_cache()

        if pcc_a >= len(self._pccs) or pcc_b >= len(self._pccs):
            return None

        nodes_a = self._pccs[pcc_a]
        nodes_b = self._pccs[pcc_b]

        max_neg_conf = None
        max_pos_inactive_conf = 0.0

        for u in nodes_a:
            for v in self.G.neighbors(u):
                if v not in nodes_b:
                    continue

                edge_data = self.G[u][v].get('data')
                if edge_data is None:
                    continue

                if edge_data.label == EdgeLabel.NEGATIVE:
                    if max_neg_conf is None:
                        max_neg_conf = edge_data.confidence
                    else:
                        max_neg_conf = max(max_neg_conf, edge_data.confidence)
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    max_pos_inactive_conf = max(max_pos_inactive_conf, edge_data.confidence)

        if max_neg_conf is None:
            return None  # No negative edge = external stability not defined

        return max_neg_conf - max_pos_inactive_conf

    def find_unstable_internal_pairs(self, alpha: float = 0.0) -> List[Tuple[int, int, float, Tuple[int, int]]]:
        """
        Find all vertex pairs within PCCs with stability < alpha.

        Returns: List of (u, v, stability, min_edge_on_msp)
        """
        self._ensure_pcc_cache()
        unstable = []

        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Collect negative edges within this PCC first
            negative_edges = []
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                            negative_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Build MST ONCE for this PCC
            pcc_graph = nx.Graph()
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                            pcc_graph.add_edge(u, v, weight=edge_data.confidence)

            if pcc_graph.number_of_edges() == 0:
                continue

            mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

            # Check each negative edge pair using the cached MST
            for u, v, neg_conf in negative_edges:
                if not mst.has_node(u) or not mst.has_node(v):
                    continue

                try:
                    path = nx.shortest_path(mst, u, v)
                except nx.NetworkXNoPath:
                    continue

                if len(path) < 2:
                    continue

                # Find minimum edge on path (MSP strength)
                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                # stability = MSP - neg_conf
                stability = min_conf - neg_conf
                if stability < alpha:
                    unstable.append((u, v, stability, min_edge))

        return unstable

    def find_unstable_external_pairs(self, alpha: float = 0.0) -> List[Tuple[int, int, float, Tuple[int, int]]]:
        """
        Find all PCC pairs with external stability < alpha.

        Returns: List of (pcc_a, pcc_b, stability, highest_neg_edge)
        """
        self._ensure_pcc_cache()
        unstable = []

        for pcc_a in range(len(self._pccs)):
            for pcc_b in range(pcc_a + 1, len(self._pccs)):
                stability = self.compute_external_stability(pcc_a, pcc_b)

                if stability is not None and stability < alpha:
                    # Find highest confidence negative edge
                    max_neg_edge = None
                    max_neg_conf = -float('inf')

                    for u in self._pccs[pcc_a]:
                        for v in self.G.neighbors(u):
                            if v not in self._pccs[pcc_b]:
                                continue
                            edge_data = self.G[u][v].get('data')
                            if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                                if edge_data.confidence > max_neg_conf:
                                    max_neg_conf = edge_data.confidence
                                    max_neg_edge = (u, v)

                    if max_neg_edge:
                        unstable.append((pcc_a, pcc_b, stability, max_neg_edge))

        return unstable

    def make_zero_stable(self, alpha: float = 0.0,
                         max_deactivations: int = -1) -> int:
        """
        Make the graph alpha-stable by deactivating positive edges.

        Per PDF: "The initial goal is to make positive-inactive assignments that
        will make the graph 0-stable. This is done entirely without human input.
        Note that we do not need to explicitly inactivate negative edges."

        Args:
            alpha: Stability threshold. Default 0.0 for strict 0-stability.
                   Negative values (e.g., -0.1) allow small instabilities,
                   resulting in less aggressive fragmentation.
            max_deactivations: -1 (default) = unlimited. Positive value caps the
                   total number of deactivations per call. Used during active
                   review to throttle the cascade: a single INTERNAL flip can
                   trigger 100+ deactivations in an over-merged giant cluster,
                   fragmenting it faster than the merge selectors can recover,
                   causing a strict H-F1 decrease. Limiting cascade size to a
                   small N (e.g. 5) spreads the fragmentation across batches,
                   so H-F1 evolves gradually instead of dropping in spikes.

        For each pair (u,v) with stability < alpha:
        - Deactivate the min edge on the MSP path to split the PCC

        Returns: Number of edges deactivated.

        OPTIMIZATION: Build MST forest once, maintain incrementally within this function.
        """
        deactivations = 0

        # Build MST forest once at the start
        self._build_mst_forest()

        while True:
            # Recompute PCCs after each round of deactivations
            self._pcc_cache_valid = False
            self._ensure_pcc_cache()

            # Process each PCC that has unstable pairs
            made_progress = False

            for pcc_id, pcc in enumerate(self._pccs):
                if len(pcc) < 2:
                    continue

                # Collect negative edges within this PCC using pair iteration
                # (O(N^2) with N = PCC size, much faster than iterating all neighbors)
                pcc_list = sorted(pcc)
                negative_edges = []
                for i in range(len(pcc_list)):
                    for j in range(i + 1, len(pcc_list)):
                        u, v = pcc_list[i], pcc_list[j]
                        if self.G.has_edge(u, v):
                            edge_data = self.G[u][v].get('data')
                            if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                                negative_edges.append((u, v, edge_data.confidence))

                if not negative_edges:
                    continue

                # Extract MST for this PCC from global forest (no rebuild needed!)
                mst = self._mst_forest.subgraph(pcc).copy()
                pcc_deactivations = 0

                # Process all unstable pairs in this PCC using the same MST
                # Keep processing until no more unstable pairs or MST becomes disconnected
                pcc_changed = True
                while pcc_changed:
                    pcc_changed = False

                    # Precompute all paths in MST once per iteration (avoids repeated BFS)
                    all_paths = dict(nx.all_pairs_shortest_path(mst))

                    # Find unstable pairs using precomputed paths
                    unstable_pairs = []
                    for u, v, neg_conf in negative_edges:
                        if not mst.has_node(u) or not mst.has_node(v):
                            continue

                        # Look up precomputed path (O(1) instead of BFS)
                        path = all_paths.get(u, {}).get(v)
                        if path is None:
                            continue  # Pair already separated

                        if len(path) < 2:
                            continue

                        # Find minimum edge on path
                        min_conf = float('inf')
                        min_edge = None
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i + 1]
                            conf = mst[a][b]['weight']
                            if conf < min_conf:
                                min_conf = conf
                                min_edge = (a, b)

                        stability = min_conf - neg_conf
                        if stability < alpha:
                            unstable_pairs.append((u, v, stability, min_edge, neg_conf))

                    if not unstable_pairs:
                        break

                    # Batching optimization: Cut multiple edges at once
                    # Count which edges appear on paths of unstable pairs (greedy hitting set)
                    edge_hit_count = {}
                    edge_to_pairs = {}

                    for u, v, stability, min_edge, neg_conf in unstable_pairs:
                        # Count the min edge for this pair
                        if min_edge not in edge_hit_count:
                            edge_hit_count[min_edge] = 0
                            edge_to_pairs[min_edge] = []
                        edge_hit_count[min_edge] += 1
                        edge_to_pairs[min_edge].append((u, v))

                    # Sort edges by how many pairs they fix (descending)
                    edges_to_cut = sorted(edge_hit_count.keys(),
                                         key=lambda e: edge_hit_count[e],
                                         reverse=True)

                    # Deactivate edges in batch (greedy - most impactful first)
                    for edge in edges_to_cut:
                        if not mst.has_edge(edge[0], edge[1]):
                            continue  # Already removed by previous cut in this batch

                        # Deactivate this edge
                        self.deactivate_positive(edge, deactivator=edge_to_pairs[edge][0])
                        deactivations += 1
                        pcc_deactivations += 1
                        made_progress = True
                        pcc_changed = True

                        # Remove from local MST and global forest
                        mst.remove_edge(edge[0], edge[1])
                        if self._mst_forest.has_edge(edge[0], edge[1]):
                            self._mst_forest.remove_edge(edge[0], edge[1])

                # Log summary for this PCC
                if pcc_deactivations > 0:
                    logger.info(f"PCC (size {len(pcc)}): deactivated {pcc_deactivations} edges for {alpha}-stability")

            # If no progress was made in any PCC, we're done
            if not made_progress:
                break

        return deactivations

    def _get_msp_for_pair(self, u: int, v: int, pcc_id: int) -> Optional[Tuple[float, Tuple[int, int]]]:
        """Get MSP strength and min edge for a specific pair. Returns (strength, min_edge)."""
        pcc = self._pccs[pcc_id]

        # Build MST for this PCC only
        pcc_graph = nx.Graph()
        for a in pcc:
            for b in self.G.neighbors(a):
                if b in pcc and a < b:
                    edge_data = self.G[a][b].get('data')
                    if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                        pcc_graph.add_edge(a, b, weight=edge_data.confidence)

        if pcc_graph.number_of_edges() == 0:
            return None

        mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

        if not mst.has_node(u) or not mst.has_node(v):
            return None

        try:
            path = nx.shortest_path(mst, u, v)
        except nx.NetworkXNoPath:
            return None

        if len(path) < 2:
            return None

        # Find minimum edge on path
        min_conf = float('inf')
        min_edge = None
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            conf = mst[a][b]['weight']
            if conf < min_conf:
                min_conf = conf
                min_edge = (a, b)

        return (min_conf, min_edge)

    def get_review_candidates(self, alpha: float) -> List[StabilityCandidate]:
        """
        Get candidates for human review with stability < alpha.

        Per PDF algorithm steps 1-3:
        1. Compute internal stability for each pair in each PCC
        2. Compute external stability for each PCC pair with negative edges
        3. Order by increasing stability, select first k pairs

        OPTIMIZATION: Build MST forest once at start.
        """
        self._ensure_pcc_cache()

        # Build MST forest once for all internal stability calculations
        self._build_mst_forest()

        candidates = []

        # Internal candidates
        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Collect negative edges within this PCC
            negative_edges = []
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                            negative_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Extract MST for this PCC from the forest
            mst = self._mst_forest.subgraph(pcc).copy()
            if mst.number_of_edges() == 0:
                continue

            # Precompute all paths in this PCC's MST (avoids repeated BFS)
            all_paths = dict(nx.all_pairs_shortest_path(mst))

            # Check each negative edge pair using precomputed paths
            for u, v, neg_conf in negative_edges:
                # Look up precomputed path (O(1) instead of BFS)
                path = all_paths.get(u, {}).get(v)
                if path is None:
                    continue

                if len(path) < 2:
                    continue

                # Find minimum edge on path (MSP strength)
                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                # stability = MSP - neg_conf
                stability = min_conf - neg_conf
                if stability < alpha:
                    candidates.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),
                        node_pair=(u, v),
                        pcc_id=pcc_id
                    ))

        # External candidates - optimized: pre-collect cross-PCC negative edges
        # Build map of PCC pairs with negative edges and their max negative edge
        pcc_pair_neg_edges: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        for u in self.G.nodes():
            pcc_u = self._node_to_pcc.get(u)
            if pcc_u is None:
                continue

            for v in self.G.neighbors(u):
                pcc_v = self._node_to_pcc.get(v)
                if pcc_v is None or pcc_u == pcc_v:
                    continue

                # Normalize PCC pair order
                pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                edge_data = self.G[u][v].get('data')
                if not edge_data:
                    continue

                if edge_data.label == EdgeLabel.NEGATIVE:
                    current = pcc_pair_neg_edges.get(pcc_pair)
                    if current is None or edge_data.confidence > current[0]:
                        pcc_pair_neg_edges[pcc_pair] = (edge_data.confidence, (u, v))
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    current = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
                    pcc_pair_pos_inactive[pcc_pair] = max(current, edge_data.confidence)

        # Now compute external stability only for PCC pairs with negative edges
        for pcc_pair, (max_neg_conf, max_neg_edge) in pcc_pair_neg_edges.items():
            max_pos_inactive = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stability = max_neg_conf - max_pos_inactive

            if stability < alpha:
                candidates.append(StabilityCandidate(
                    candidate_type="EXTERNAL",
                    stability=stability,
                    review_edge=max_neg_edge,
                    pcc_pair=pcc_pair
                ))

        # Sort by stability ascending (most unstable first)
        candidates.sort(key=lambda c: c.stability)
        return candidates

    def _build_likely_fn_pos_graph(self, method: str, human_boost: float):
        """Return (nodes, node_to_idx, sparse A) for the positive subgraph
        the candidate selector operates on.

        Weight scheme for `*_weighted` methods: uses the edge's `score`
        field directly. Because `apply_human_review` saturates score to 1.0
        for human-confirmed positives, those edges naturally dominate the
        algorithm-classified ones (typical score ~0.6-0.8). No special-case
        ranker lookup needed.
        """
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)
        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            if ed.label not in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            if method.endswith('weighted'):
                score = float(getattr(ed, 'score', 0.5) or 0.0)
                # Re-center on the "no info" baseline (0.5 in pipeline space)
                # so that human-confirmed positives (score=1.0 -> weight=0.5)
                # dominate algorithm-classified positives (score~0.65 ->
                # weight~0.15). Below-baseline scores get weight 0 (effectively
                # excluded), avoiding spurious weak-positive links.
                w = max(score - 0.5, 0.0)
                if w <= 0:
                    continue
            else:
                w = 1.0
            if i < j:
                rows.append(i); cols.append(j); data.append(w)
        if not rows:
            return all_nodes, node_to_idx, None
        all_rows = np.array(rows + cols, dtype=np.int64)
        all_cols = np.array(cols + rows, dtype=np.int64)
        all_data = np.array(data + data, dtype=np.float64)
        A = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _build_likely_fn_confidence_signed_graph(self):
        """Signed adjacency keyed on the edge's existing `confidence` field.

        `confidence` is already the right quantity: ThresholdBasedClassifier
        defines it as |score - threshold| / max_range — so it accounts for
        the actual classifier threshold and the data-specific score range
        without any special-case baseline. After human review it saturates
        toward 1.0 in apply_human_review (or is set to 1.0 directly in
        apply_ground_truth_review). The label provides the sign.

        Weight rule:
            positive / positive-inactive  →  +confidence
            negative                      →  −confidence

        Verified-positive and verified-negative edges both end up at
        magnitude 1.0; algorithm-classified edges contribute proportionally
        to their distance from the classifier's decision boundary.
        """
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)

        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            conf = float(getattr(ed, 'confidence', 0.0) or 0.0)
            if conf <= 0:
                continue
            if ed.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                w = conf
            elif ed.label == EdgeLabel.NEGATIVE:
                w = -conf
            else:
                continue
            rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
        if not rows:
            return all_nodes, node_to_idx, None
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _build_likely_fn_signed_graph(self, human_boost: float, neg_repulsion: float):
        """Signed adjacency: positives weighted by `score - 0.5` (so human-
        confirmed pull strongly, algorithm-classified pull weakly); negatives
        identified by score == 0.0 (human-confirmed negatives, saturated by
        apply_human_review) get weight `-neg_repulsion`. Algorithm-classified
        negatives (score > 0) are excluded since they're not high-confidence
        enough to use as repulsion signal."""
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)
        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            score = float(getattr(ed, 'score', 0.5) or 0.0)
            if ed.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                w = max(score - 0.5, 0.0)
                if w <= 0:
                    continue
            elif ed.label == EdgeLabel.NEGATIVE and score == 0.0:
                # Human-confirmed negative (apply_human_review saturated score
                # to 0.0). Algorithm-classified negatives have score > 0 and
                # are excluded — they are too noisy to use as repulsion.
                w = -neg_repulsion
            else:
                continue
            rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
        if not rows:
            return all_nodes, node_to_idx, None
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _compute_node_heat_features(self, A, signed: bool, n_eigs: int, heat_t: float):
        """Return phi_weighted ([N, k]) such that K_t(i, j) = phi_w[i] · phi_w[j]."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        N = A.shape[0]
        if signed:
            abs_degrees = np.array(np.abs(A).sum(axis=1)).flatten()
            D_bar = sp.diags(abs_degrees)
            L = D_bar - A
        else:
            degrees = np.array(A.sum(axis=1)).flatten()
            d_inv_sqrt = np.zeros(N, dtype=np.float64)
            nz = degrees > 0
            d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            L = sp.eye(N, format='csr') - (D_inv_sqrt @ A @ D_inv_sqrt)
        k = min(n_eigs, max(N - 1, 1))
        try:
            lambdas, phi = eigsh(L, k=k, which='SM', tol=1e-6, maxiter=N * 10)
        except Exception as e:
            logger.warning(f"Heat-kernel eigsh failed ({e}); using dense fallback")
            lambdas_full, phi_full = np.linalg.eigh(L.toarray())
            lambdas, phi = lambdas_full[:k], phi_full[:, :k]
        order = np.argsort(lambdas)
        lambdas = np.clip(lambdas[order], 0.0, None)
        phi = phi[:, order]
        weights = np.exp(-heat_t * lambdas)
        return phi * np.sqrt(np.maximum(weights, 0.0))

    def _compute_node_ppr(self, A, alpha: float):
        """Return full PPR matrix (dense, N×N)."""
        import scipy.sparse as sp
        N = A.shape[0]
        # PPR requires non-negative transition probabilities. For weighted
        # graphs A is already non-negative; we don't call this for signed graphs.
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp.diags(1.0 / row_sums)
        P = D_inv @ A
        M = sp.eye(N, format='csr') - (1 - alpha) * P.T
        try:
            return alpha * np.linalg.inv(M.toarray())
        except np.linalg.LinAlgError:
            return alpha * np.linalg.pinv(M.toarray())

    def select_likely_fn_candidates(
        self,
        verified_edges: Optional[Set[Tuple[int, int]]] = None,
        max_count: int = 100,
        min_score: float = 0.0,
        exclude_edges: Optional[Set[Tuple[int, int]]] = None,
        n_eigs: int = 64,
        heat_kernel_t: float = 10.0,
        method: str = 'heat_kernel',
        human_boost: float = 20.0,
        ppr_alpha: float = 0.15,
        neg_repulsion: float = 5.0,
    ) -> List[Tuple[int, int, float]]:
        """Generate heat-kernel-ranked false-negative candidates for human review.

        For every active cross-PCC negative edge (u, v) we compute the heat
        kernel similarity over the *ever-positive* graph (active positive
        edges + positive-inactive edges Phase 0 deactivated):

            K_t(u, v) = Σ_k exp(-t · λ_k) · φ_k(u) · φ_k(v)

        where {λ_k, φ_k} are the lowest eigenvalues / eigenvectors of the
        normalized symmetric Laplacian L_sym = I - D^{-1/2} A D^{-1/2} of the
        ever-positive graph. High K_t(u, v) means u and v are in the same
        diffusion neighborhood — strong evidence they are actually the same
        individual whose connecting edges Phase 0 deactivated.

        Why ever-positive (active+inactive) and not just active: Phase 0
        deactivates edges to resolve internal contradictions, which is exactly
        the information needed to identify FNs. Using the ever-positive graph
        recovers that hidden signal. (Validated in
        live metric_tracker runs have since shown spectral/heat-kernel
        metrics collapse to ~0 FN precision after the first few batches.)

        Runs in PARALLEL with `generate_candidate_pools`; the stability
        mechanism is untouched.

        Args:
            verified_edges: edges already presented to humans; skipped.
            max_count: maximum number of candidates to return.
            min_score: only return candidates with heat-kernel strictly above this.
            exclude_edges: additional edges to skip (e.g. already in batch).
            n_eigs: number of Laplacian eigenvectors to compute (cost ~N*k).
            heat_kernel_t: diffusion time. Larger t = more global similarity.

        Returns: list of (u, v, score) tuples sorted by heat-kernel descending,
            length up to max_count. Empty list if max_count <= 0.
        """
        if max_count <= 0:
            return []
        try:
            import scipy.sparse  # noqa
        except ImportError:
            logger.error("scipy is required for graph-spectral candidate selection")
            return []

        _verified = verified_edges or set()
        _exclude = exclude_edges or set()
        self._ensure_pcc_cache()

        all_nodes = sorted(self.G.nodes())
        if len(all_nodes) < 2:
            return []

        # Build the appropriate positive (or signed) graph and a per-node
        # scoring function score_fn(i, j) -> heat/PPR/etc.
        if method in ('heat_kernel', 'heat_kernel_weighted'):
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(method, human_boost)
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=False,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method in ('heat_kernel_unnorm', 'heat_kernel_unnorm_weighted'):
            # Unnormalized Laplacian (D - A). Empirically much more
            # discriminative than the normalized version for FN detection —
            # preserves the hub/community structure that maps to individuals
            # instead of rescaling it away by sqrt(degree). At iter 0 of the
            # GZCD megadescriptor run this hit 32/50 vs the normalized 12/50.
            build_method = ('heat_kernel_weighted'
                            if method == 'heat_kernel_unnorm_weighted'
                            else 'heat_kernel')
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(build_method, human_boost)
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method in ('ppr', 'ppr_weighted'):
            # PPR needs a positive graph; reuse the same builder.
            heat_method = 'heat_kernel_weighted' if method == 'ppr_weighted' else 'heat_kernel'
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(heat_method, human_boost)
            if A is None:
                return []
            ppr = self._compute_node_ppr(A, alpha=ppr_alpha)
            score_fn = lambda i, j: float(0.5 * (ppr[i, j] + ppr[j, i]))
        elif method == 'human_signed':
            all_nodes, node_to_idx, A = self._build_likely_fn_signed_graph(
                human_boost=human_boost, neg_repulsion=neg_repulsion
            )
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method == 'confidence_signed':
            # Unnormalized signed Laplacian using confidence as the natural
            # weight: |conf| handles threshold/score-range automatically, and
            # the sign comes from the edge label. Human-reviewed edges of
            # both polarities saturate to |1.0| — both positives and
            # negatives become primary signal.
            all_nodes, node_to_idx, A = self._build_likely_fn_confidence_signed_graph()
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Score each active cross-PCC negative edge.
        scored: List[Tuple[float, int, int, float]] = []  # (-heat, u, v, raw_score)
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None or ed.label != EdgeLabel.NEGATIVE:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key in _verified or edge_key in _exclude:
                continue
            pcc_u = self._node_to_pcc.get(u)
            pcc_v = self._node_to_pcc.get(v)
            if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                continue
            iu = node_to_idx.get(u)
            iv = node_to_idx.get(v)
            if iu is None or iv is None:
                continue
            metric = score_fn(iu, iv)
            if metric <= min_score:
                continue
            raw_score = float(ed.score) if ed.score is not None else 0.0
            # negate so ascending sort picks largest metric first
            scored.append((-metric, u, v, raw_score))

        if not scored:
            return []

        scored.sort(key=lambda t: t[0])
        top = scored[:max_count]
        return [(u, v, s) for _, u, v, s in top]

    def generate_candidate_pools(
        self,
        alpha: float,
        verified_edges: Optional[Set[Tuple[int, int]]] = None,
        unverified_threshold: float = 0.0,
        pcc_separation_strength: Optional[Dict[int, float]] = None,
    ) -> Dict[str, List["StabilityCandidate"]]:
        """Generate sorted candidate pools (no batch cap, no selection).

        The caller composes the final review batch by drawing from pools in
        whatever priority order it wants, applying its own per-PCC conflict
        rules and budget cap.

        Args:
            alpha: stability threshold (candidates with stability >= alpha
                are filtered out).
            verified_edges: edges already human-reviewed — excluded from pools.
            unverified_threshold: confidence cutoff for the unverified pool
                (0 disables it).
            pcc_separation_strength: NIS n_hat analog for scoring
                UNVERIFIED_NEG (lower = more isolated = higher priority).

        Returns:
            Dict with keys 'internal', 'external', 'unverified', each a
            sorted list of StabilityCandidate (highest priority first).
        """
        self._ensure_pcc_cache()

        # Build MST forest once for internal candidate generation
        self._build_mst_forest()

        # Per-pool accumulators. Internal (intra-PCC negative edges driving
        # instability) and unverified-pos (intra-PCC low-confidence positive
        # MST edges) come from the same per-PCC pass; external and
        # unverified-neg come from the global cross-PCC edge scan.
        internal_pool: List[StabilityCandidate] = []
        external_pool: List[StabilityCandidate] = []
        unverified_pool: List[StabilityCandidate] = []
        _verified = verified_edges or set()

        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Extract MST for this PCC from the forest
            mst = self._mst_forest.subgraph(pcc).copy()
            if mst.number_of_edges() == 0:
                continue

            # Root MST and compute subtree sizes for structural impact (O(N) per PCC)
            n_pcc = len(pcc)
            mst_root = next(iter(pcc))
            mst_parent = {}
            mst_order = []
            stack = [(mst_root, None)]
            while stack:
                node, par = stack.pop()
                mst_parent[node] = par
                mst_order.append(node)
                for nbr in mst.neighbors(node):
                    if nbr != par:
                        stack.append((nbr, node))
            mst_subtree_size = {node: 1 for node in mst_order}
            for node in reversed(mst_order):
                if mst_parent[node] is not None:
                    mst_subtree_size[mst_parent[node]] += mst_subtree_size[node]

            # Collect unverified MST edges below threshold
            if unverified_threshold > 0:
                for a, b, attr in mst.edges(data=True):
                    edge_key = (min(a, b), max(a, b))
                    if edge_key in _verified:
                        continue
                    conf = attr.get('weight', 1.0)
                    if conf < unverified_threshold:
                        # Structural impact: size of smaller subtree if this edge were cut
                        if mst_parent.get(b) == a:
                            child = b
                        elif mst_parent.get(a) == b:
                            child = a
                        else:
                            child = b
                        child_size = mst_subtree_size.get(child, 1)
                        impact = float(min(child_size, n_pcc - child_size))
                        unverified_pool.append(StabilityCandidate(
                            candidate_type="UNVERIFIED",
                            stability=conf,  # use confidence as sort key (lower = higher priority)
                            review_edge=(a, b),
                            structural_impact=max(impact, 1.0),
                            node_pair=(a, b),
                            pcc_id=pcc_id
                        ))

            # Find intra-PCC negative edges for internal instability candidates
            pcc_list = sorted(pcc)
            negative_edges = []
            for i in range(len(pcc_list)):
                for j in range(i + 1, len(pcc_list)):
                    u, v = pcc_list[i], pcc_list[j]
                    if self.G.has_edge(u, v):
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                            negative_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Precompute all paths in this PCC's MST (avoids repeated BFS)
            all_paths = dict(nx.all_pairs_shortest_path(mst))

            for u, v, neg_conf in negative_edges:
                # Look up precomputed path (O(1) instead of BFS)
                path = all_paths.get(u, {}).get(v)
                if path is None:
                    continue

                if len(path) < 2:
                    continue

                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                stability = min_conf - neg_conf
                if stability < alpha:
                    # Compute structural impact: size of smaller subtree if weakest edge is cut
                    impact = 1.0
                    if min_edge is not None:
                        a, b = min_edge
                        if mst_parent.get(b) == a:
                            child = b
                        elif mst_parent.get(a) == b:
                            child = a
                        else:
                            child = b
                        child_size = mst_subtree_size.get(child, 1)
                        impact = float(min(child_size, n_pcc - child_size))

                    internal_pool.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),
                        structural_impact=max(impact, 1.0),
                        node_pair=(u, v),
                        pcc_id=pcc_id
                    ))

        # Step 2: Generate external candidates.
        # Per PDF: external_stability = max_neg - max_pos_inactive for ANY
        # PCC pair with a negative edge (not just those with pos-inactive edges).
        pcc_pair_max_neg: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            pcc_u = self._node_to_pcc.get(u)
            pcc_v = self._node_to_pcc.get(v)
            if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                continue
            pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
            if ed.label == EdgeLabel.NEGATIVE:
                current = pcc_pair_max_neg.get(pcc_pair)
                if current is None or ed.confidence > current[0]:
                    pcc_pair_max_neg[pcc_pair] = (ed.confidence, (u, v))
            elif ed.label == EdgeLabel.POSITIVE_INACTIVE:
                current = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
                pcc_pair_pos_inactive[pcc_pair] = max(current, ed.confidence)

        # Step 3: Build external candidates
        # Skip candidates whose review edge is already verified (exhausted)
        for pcc_pair, (max_neg_conf, max_neg_edge) in pcc_pair_max_neg.items():
            edge_key = (min(max_neg_edge[0], max_neg_edge[1]), max(max_neg_edge[0], max_neg_edge[1]))
            if edge_key in _verified:
                continue
            max_pos_inactive = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stability = max_neg_conf - max_pos_inactive

            if stability < alpha:
                pcc_a_id, pcc_b_id = pcc_pair
                impact = float(min(len(self._pccs[pcc_a_id]), len(self._pccs[pcc_b_id])))
                external_pool.append(StabilityCandidate(
                    candidate_type="EXTERNAL",
                    stability=stability,
                    review_edge=max_neg_edge,
                    structural_impact=max(impact, 1.0),
                    pcc_pair=pcc_pair
                ))

        # Step 3a: Generate unverified negative candidates (potential merges)
        # Low-confidence negative edges between PCCs that haven't been human-reviewed
        # NIS-style: if pcc_separation_strength is provided, score by PCC isolation
        # (min separation strength of the two PCCs) rather than individual edge confidence.
        # This biases review toward edges connecting weakly-separated PCCs — the ones
        # most likely to be incorrect negatives hiding real merges.
        if unverified_threshold > 0:
            for u, v, attr in self.G.edges(data=True):
                edge_data = attr.get('data')
                if edge_data is None or edge_data.label != EdgeLabel.NEGATIVE:
                    continue
                edge_key = (min(u, v), max(u, v))
                if edge_key in _verified:
                    continue
                if edge_data.confidence >= unverified_threshold:
                    continue
                # Must be cross-PCC
                pcc_u = self._node_to_pcc.get(u)
                pcc_v = self._node_to_pcc.get(v)
                if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                    continue
                impact = float(min(len(self._pccs[pcc_u]), len(self._pccs[pcc_v])))

                # NIS-style scoring: use min PCC separation strength (lower = more isolated)
                if pcc_separation_strength is not None:
                    score_key = min(
                        pcc_separation_strength.get(pcc_u, 0.0),
                        pcc_separation_strength.get(pcc_v, 0.0)
                    )
                else:
                    score_key = edge_data.confidence

                unverified_pool.append(StabilityCandidate(
                    candidate_type="UNVERIFIED_NEG",
                    stability=score_key,  # lower = higher priority (more isolated or less confident)
                    review_edge=(u, v),
                    structural_impact=max(impact, 1.0),
                    pcc_pair=(min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                ))

        # Sort each instability pool by combined stability + structural impact + confidence.
        # Lower sort key = higher priority. Internal and external are sorted with
        # the same blend (consistent semantics across pools); the caller may
        # interleave them in whatever priority order it chooses.
        def _get_review_edge_confidence(candidate):
            u, v = candidate.review_edge
            ed = self.get_edge(u, v)
            return ed.confidence if ed else 0.5

        def _sort_instability(pool: List[StabilityCandidate]):
            pool.sort(key=lambda c: c.stability)

        _sort_instability(internal_pool)
        _sort_instability(external_pool)
        # Unverified candidates: weakest-confidence (or most-isolated) first.
        unverified_pool.sort(key=lambda c: c.stability)

        n_unverified_pos = sum(1 for c in unverified_pool if c.candidate_type == "UNVERIFIED")
        n_unverified_neg = sum(1 for c in unverified_pool if c.candidate_type == "UNVERIFIED_NEG")
        logger.info(
            f"Generated candidate pools: "
            f"{len(internal_pool)} internal/split, "
            f"{len(external_pool)} external/merge, "
            f"{len(unverified_pool)} unverified "
            f"({n_unverified_pos} pos/split, {n_unverified_neg} neg/merge)"
        )

        return {
            'internal': internal_pool,
            'external': external_pool,
            'unverified': unverified_pool,
        }

    def apply_human_review(self, u: int, v: int, human_agrees: bool, ch: float):
        """
        Apply human review result to an edge.

        Per PDF step 4:
        (a) Agree: add ch to edge confidence
        (b) Disagree: subtract ch; if negative, flip label and confidence
        """
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v]['data']

        old_label = edge_data.label
        edge_data.ranker = 'human'

        if human_agrees:
            edge_data.confidence = min(1.0, edge_data.confidence + ch)
            # Re-activate positive-inactive edges when human confirms they're positive
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                edge_data.label = EdgeLabel.POSITIVE
        else:
            edge_data.confidence -= ch
            if edge_data.confidence < 0:
                # Flip label
                if edge_data.label == EdgeLabel.POSITIVE:
                    edge_data.label = EdgeLabel.NEGATIVE
                elif edge_data.label == EdgeLabel.NEGATIVE:
                    edge_data.label = EdgeLabel.POSITIVE
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    edge_data.label = EdgeLabel.NEGATIVE
                edge_data.confidence = abs(edge_data.confidence)

        # Update edge counts and pos-inactive tracking if label changed
        if edge_data.label != old_label:
            self._decrement_edge_count(old_label)
            self._increment_edge_count(edge_data.label)
            key = (min(u, v), max(u, v))
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.add(key)
            elif old_label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.discard(key)

        # Saturate the edge score to reflect the human verdict directly.
        # Algorithm-classified positives have score ~0.6-0.8; pushing human-
        # confirmed positives to 1.0 (or negatives to 0.0) lets any downstream
        # consumer of `edge_data.score` (histograms, candidate scoring,
        # spectral metrics) respect human input without special-case ranker
        # lookups.
        if edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
            edge_data.score = 1.0
        elif edge_data.label == EdgeLabel.NEGATIVE:
            edge_data.score = 0.0

        self._invalidate_cache()

    def apply_ground_truth_review(self, u: int, v: int, is_positive: bool):
        """
        Apply a ground-truth human review: directly set the edge label.

        Unlike apply_human_review which uses confidence arithmetic,
        this sets the label definitively with confidence 1.0.
        Used during VST verification where human reviews are authoritative.
        """
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v]['data']
        old_label = edge_data.label

        new_label = EdgeLabel.POSITIVE if is_positive else EdgeLabel.NEGATIVE
        edge_data.label = new_label
        edge_data.confidence = 1.0
        edge_data.ranker = 'human'
        # Saturate score to match the authoritative GT label.
        edge_data.score = 1.0 if is_positive else 0.0

        if edge_data.label != old_label:
            self._decrement_edge_count(old_label)
            self._increment_edge_count(edge_data.label)
            key = (min(u, v), max(u, v))
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.add(key)
            elif old_label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.discard(key)

        self._invalidate_cache()

    def get_clustering(self) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
        """Get current clustering as dictionaries."""
        self._ensure_pcc_cache()
        cluster_dict = {pcc_id: pcc for pcc_id, pcc in enumerate(self._pccs)}
        node2cid = {node: pcc_id for pcc_id, pcc in enumerate(self._pccs) for node in pcc}
        return cluster_dict, node2cid

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph state.

        Optimized to avoid iterating all edges (which can be 10M+).
        - Edge counts: maintained incrementally
        - Internal stability: iterates PCC node pairs (O(N^2) per PCC, N small)
        - External stability: iterates only positive-inactive edges (O(small) vs O(10M))
        """
        self._ensure_pcc_cache()

        pcc_sizes = [len(pcc) for pcc in self._pccs]

        # Compute min internal stability using PCC pair iteration (not neighbor iteration)
        min_internal = float('inf')
        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Find intra-PCC edges by iterating node pairs (O(N^2) with N = PCC size)
            pcc_list = sorted(pcc)
            negative_edges = []
            positive_edges = []
            for i in range(len(pcc_list)):
                for j in range(i + 1, len(pcc_list)):
                    u, v = pcc_list[i], pcc_list[j]
                    if self.G.has_edge(u, v):
                        edge_data = self.G[u][v].get('data')
                        if edge_data:
                            if edge_data.label == EdgeLabel.NEGATIVE:
                                negative_edges.append((u, v, edge_data.confidence))
                            elif edge_data.label == EdgeLabel.POSITIVE:
                                positive_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Build MST from positive edges within this PCC
            pcc_graph = nx.Graph()
            for u, v, conf in positive_edges:
                pcc_graph.add_edge(u, v, weight=conf)

            if pcc_graph.number_of_edges() == 0:
                continue

            mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

            for u, v, neg_conf in negative_edges:
                if not mst.has_node(u) or not mst.has_node(v):
                    continue
                try:
                    path = nx.shortest_path(mst, u, v)
                except nx.NetworkXNoPath:
                    continue
                if len(path) < 2:
                    continue

                msp_strength = min(mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                stab = msp_strength - neg_conf
                if stab < min_internal:
                    min_internal = stab

        # Compute min external stability.
        # Per PDF: external_stability(A,B) = max_neg_conf - max_pos_inactive_conf
        # Even PCC pairs with NO positive-inactive edges have finite external stability
        # equal to their max negative confidence (since max_pos_inactive = 0).
        min_external = float('inf')
        pcc_pair_max_neg: Dict[Tuple[int, int], float] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            pcc_u = self._node_to_pcc.get(u)
            pcc_v = self._node_to_pcc.get(v)
            if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                continue
            pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
            if ed.label == EdgeLabel.NEGATIVE:
                current = pcc_pair_max_neg.get(pcc_pair, 0.0)
                pcc_pair_max_neg[pcc_pair] = max(current, ed.confidence)
            elif ed.label == EdgeLabel.POSITIVE_INACTIVE:
                current = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
                pcc_pair_pos_inactive[pcc_pair] = max(current, ed.confidence)

        for pcc_pair, max_neg in pcc_pair_max_neg.items():
            max_pi = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stab = max_neg - max_pi
            if stab < min_external:
                min_external = stab

        return {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'edge_counts': dict(self._edge_counts),
            'num_pccs': len(self._pccs),
            'pcc_sizes': pcc_sizes,
            'min_pcc_size': min(pcc_sizes) if pcc_sizes else 0,
            'max_pcc_size': max(pcc_sizes) if pcc_sizes else 0,
            'min_internal_stability': min_internal,
            'min_external_stability': min_external,
        }

    def get_mst_cache_stats(self) -> Dict[str, Any]:
        """Get MST forest statistics (no caching - built explicitly when needed)."""
        return {
            'mst_forest_edges': self._mst_forest.number_of_edges() if self._mst_forest else 0
        }

    def densify_component(self, component: Set[int], classifier_manager,
                         max_edges: int = 2000, prioritize_negatives: bool = False,
                         on_positive_added=None) -> int:
        """
        Add missing edges within a component.

        Args:
            component: Set of node IDs in the component
            classifier_manager: Classifier to use for edge classification
            max_edges: Maximum number of edges to add
            prioritize_negatives: If True, sort by score ascending (adds likely negatives first).
                                  If False, sort by score descending (adds likely positives first).
                                  Default False reduces aggressive fragmentation.
            on_positive_added: Optional callback(u, v) invoked for each newly-added
                               POSITIVE edge — lets the caller (e.g. the algorithm)
                               react to accumulating positives.
        """
        first_classifier = classifier_manager.algo_classifiers[0] if classifier_manager.algo_classifiers else None
        if first_classifier is None:
            return 0

        embeddings, _ = classifier_manager.classifier_units[first_classifier]

        nodes = list(component)
        missing_edges = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n0, n1 = nodes[i], nodes[j]
                if not self.G.has_edge(n0, n1):
                    score = embeddings.get_score(n0, n1)
                    missing_edges.append((n0, n1, score))

        # Sort by score: ascending (negatives first) or descending (positives first)
        missing_edges.sort(key=lambda x: x[2], reverse=not prioritize_negatives)
        if len(missing_edges) > max_edges:
            missing_edges = missing_edges[:max_edges]

        added = 0
        for n0, n1, score in missing_edges:
            edge = classifier_manager.classify_edge(n0, n1, first_classifier)
            _, _, score, confidence, label, ranker = edge
            edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
            self.add_edge(n0, n1, edge_label, confidence, score, ranker)
            added += 1
            if edge_label == EdgeLabel.POSITIVE and on_positive_added is not None:
                on_positive_added(n0, n1)

        return added
