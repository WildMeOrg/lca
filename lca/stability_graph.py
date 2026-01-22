"""
Stability Graph Module for LCA v2 (stability-driven) algorithm.

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
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from tools import order_edge

logger = logging.getLogger('lca')


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
    # For INTERNAL: the unstable pair
    node_pair: Optional[Tuple[int, int]] = None
    pcc_id: Optional[int] = None
    # For EXTERNAL: the two PCCs
    pcc_pair: Optional[Tuple[int, int]] = None


class StabilityGraph:
    """
    Graph structure for LCA v2 stability-driven clustering.

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
                self.G[u][v]['data'] = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
        else:
            self.G.add_edge(u, v, data=EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker))
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

        edge_data.label = EdgeLabel.POSITIVE_INACTIVE
        edge_data.deactivator = deactivator
        self._pcc_cache_valid = False

    def reactivate(self, u: int, v: int):
        """Reactivate a positive-inactive edge back to positive."""
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v].get('data')
        if edge_data and edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
            edge_data.label = EdgeLabel.POSITIVE
            edge_data.deactivator = None
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

    def make_zero_stable(self, alpha: float = 0.0) -> int:
        """
        Make the graph alpha-stable by deactivating positive edges.

        Per PDF: "The initial goal is to make positive-inactive assignments that
        will make the graph 0-stable. This is done entirely without human input.
        Note that we do not need to explicitly inactivate negative edges."

        Args:
            alpha: Stability threshold. Default 0.0 for strict 0-stability.
                   Negative values (e.g., -0.1) allow small instabilities,
                   resulting in less aggressive fragmentation.

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
                    # Per PDF 3(a): prefer negative edge for review
                    candidates.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),  # The negative edge
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

    def select_non_conflicting_candidates_lazy(self, alpha: float, max_batch_size: int = 500) -> List[Tuple[int, int, float]]:
        """
        Lazily generate and select non-conflicting candidates for parallel review.
        Stops early once PCCs are saturated to avoid wasting time on candidates we won't use.

        Args:
            alpha: Target stability threshold
            max_batch_size: Maximum number of edges to select (prevents exhausting review budget)

        Returns: List of (u, v, score) tuples for selected edges
        """
        self._ensure_pcc_cache()

        # Build MST forest once for internal candidate generation
        self._build_mst_forest()

        # Step 1: Generate all internal candidates
        internal_candidates = []
        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

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

            for u, v, neg_conf in negative_edges:
                # Look up precomputed path (O(1) instead of BFS)
                path = all_paths.get(u, {}).get(v)
                if path is None:
                    continue

                if len(path) < 2:
                    continue

                min_conf = float('inf')
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf

                stability = min_conf - neg_conf
                if stability < alpha:
                    internal_candidates.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),
                        node_pair=(u, v),
                        pcc_id=pcc_id
                    ))

        # Step 2: Generate external candidate data (PCC pair -> stability, edge)
        # Build this once, then filter by occupied PCCs during selection
        pcc_pair_neg_edges: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        for u, v, edge_data in self.G.edges(data='data'):
            if not edge_data:
                continue

            pcc_u = self._node_to_pcc.get(u)
            pcc_v = self._node_to_pcc.get(v)

            if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                continue

            pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))

            if edge_data.label == EdgeLabel.NEGATIVE:
                current = pcc_pair_neg_edges.get(pcc_pair)
                if current is None or edge_data.confidence > current[0]:
                    pcc_pair_neg_edges[pcc_pair] = (edge_data.confidence, (u, v))
            elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                current = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
                pcc_pair_pos_inactive[pcc_pair] = max(current, edge_data.confidence)

        # Step 3: Merge internal and external candidates, sort by stability
        all_candidates = list(internal_candidates)

        for pcc_pair, (max_neg_conf, max_neg_edge) in pcc_pair_neg_edges.items():
            max_pos_inactive = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stability = max_neg_conf - max_pos_inactive

            if stability < alpha:
                all_candidates.append(StabilityCandidate(
                    candidate_type="EXTERNAL",
                    stability=stability,
                    review_edge=max_neg_edge,
                    pcc_pair=pcc_pair
                ))

        all_candidates.sort(key=lambda c: c.stability)

        logger.info(f"Generated {len(all_candidates)} candidates ({len(internal_candidates)} internal, {len(all_candidates)-len(internal_candidates)} external)")

        # Step 4: Select candidates with differentiated constraints
        # INTERNAL: max 1 per PCC (conflicting stability signals) - select ALL
        # EXTERNAL: no per-PCC limit, but cap total external edges for iterative refinement
        batch = []
        internal_pccs_used = set()
        num_internal = 0
        num_external = 0

        for candidate in all_candidates:
            if candidate.candidate_type == "INTERNAL":
                # Strict: only 1 internal edge per PCC (select all internal)
                if candidate.pcc_id in internal_pccs_used:
                    continue
                internal_pccs_used.add(candidate.pcc_id)
                num_internal += 1

                u, v = candidate.review_edge
                edge_data = self.get_edge(u, v)
                score = edge_data.score if edge_data else 0.0
                batch.append((u, v, score))
            else:
                # External edges: limit total count to prevent exhausting review budget
                if num_external >= max_batch_size:
                    continue
                num_external += 1

                u, v = candidate.review_edge
                edge_data = self.get_edge(u, v)
                score = edge_data.score if edge_data else 0.0
                batch.append((u, v, score))

        logger.info(f"Selected {len(batch)} edges ({num_internal} internal [max 1/PCC], {num_external} external [max {max_batch_size}])")
        return batch

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

        if human_agrees:
            edge_data.confidence = min(1.0, edge_data.confidence + ch)
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

        self._invalidate_cache()

    def get_clustering(self) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
        """Get current clustering as dictionaries."""
        self._ensure_pcc_cache()
        cluster_dict = {pcc_id: pcc for pcc_id, pcc in enumerate(self._pccs)}
        node2cid = {node: pcc_id for pcc_id, pcc in enumerate(self._pccs) for node in pcc}
        return cluster_dict, node2cid

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph state."""
        self._ensure_pcc_cache()

        edge_counts = {'positive': 0, 'positive_inactive': 0, 'negative': 0}
        for _, _, data in self.G.edges(data=True):
            edge_data = data.get('data')
            if edge_data:
                if edge_data.label == EdgeLabel.POSITIVE:
                    edge_counts['positive'] += 1
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    edge_counts['positive_inactive'] += 1
                elif edge_data.label == EdgeLabel.NEGATIVE:
                    edge_counts['negative'] += 1

        pcc_sizes = [len(pcc) for pcc in self._pccs]

        # Compute min internal stability - only for pairs with negative edges
        min_internal = float('inf')
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

            # Build MST once for this PCC
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

            for u, v, neg_conf in negative_edges:
                if not mst.has_node(u) or not mst.has_node(v):
                    continue
                try:
                    path = nx.shortest_path(mst, u, v)
                except nx.NetworkXNoPath:
                    continue
                if len(path) < 2:
                    continue

                # Find MSP strength
                msp_strength = min(mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                stab = msp_strength - neg_conf
                if stab < min_internal:
                    min_internal = stab

        # Compute min external stability - only for PCC pairs with negative edges
        min_external = float('inf')
        pcc_pair_stats: Dict[Tuple[int, int], Tuple[float, float]] = {}  # (max_neg, max_pos_inactive)

        for u in self.G.nodes():
            pcc_u = self._node_to_pcc.get(u)
            if pcc_u is None:
                continue

            for v in self.G.neighbors(u):
                pcc_v = self._node_to_pcc.get(v)
                if pcc_v is None or pcc_u == pcc_v:
                    continue

                pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                edge_data = self.G[u][v].get('data')
                if not edge_data:
                    continue

                current = pcc_pair_stats.get(pcc_pair, (None, 0.0))
                max_neg, max_pos_inactive = current

                if edge_data.label == EdgeLabel.NEGATIVE:
                    if max_neg is None or edge_data.confidence > max_neg:
                        max_neg = edge_data.confidence
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    max_pos_inactive = max(max_pos_inactive, edge_data.confidence)

                pcc_pair_stats[pcc_pair] = (max_neg, max_pos_inactive)

        for pcc_pair, (max_neg, max_pos_inactive) in pcc_pair_stats.items():
            if max_neg is not None:
                stab = max_neg - max_pos_inactive
                if stab < min_external:
                    min_external = stab

        return {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'edge_counts': edge_counts,
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
                         max_edges: int = 2000, prioritize_negatives: bool = False) -> int:
        """
        Add missing edges within a component.

        Args:
            component: Set of node IDs in the component
            classifier_manager: Classifier to use for edge classification
            max_edges: Maximum number of edges to add
            prioritize_negatives: If True, sort by score ascending (adds likely negatives first).
                                  If False, sort by score descending (adds likely positives first).
                                  Default False reduces aggressive fragmentation.
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

        return added
