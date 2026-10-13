"""The identification graph, and the stability measured on it.

Every edge carries a label, one of positive, positive-inactive or negative, and
a confidence. A PCC is a connected component of the positive edges alone, and
stands for one individual. The maximum strength path between two vertices is
the path whose weakest edge is strongest, and the definitions built on it are:

- internal stability of a pair in one PCC: the strength of their maximum
  strength path, minus the confidence of the negative edge between them if
  there is one;
- external stability of two PCCs: the strongest negative edge joining them,
  minus the strongest positive-inactive edge joining them;
- alpha-stable: both are at least alpha.

Only positive edges are ever deactivated, becoming positive-inactive. Negative
edges are never deactivated, which is what makes a cut irreversible by this
module and settleable only by review.

The sweeps over these definitions are vectorised through :class:`EdgeTable`,
and :mod:`tests.test_graph_vectorised_paths` holds the obvious implementation
of each one as the check that the fast path agrees.
"""

import networkx as nx
import numpy as np
from array import array
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("beta_stability")


def _build_krt(mst, idx):
    """Kruskal reconstruction tree over `mst` (edges added in DECREASING weight).
    Internal node = one merge, carrying that merge's weight and edge. Bottleneck
    of any pair = weight at their LCA, so all queries reduce to one LCA pass."""
    n = len(idx)
    edges = sorted(mst.edges(data='weight'), key=lambda t: t[2], reverse=True)
    M = 2 * n
    parent = np.full(M, -1, dtype=np.int64)
    wt = np.zeros(M, dtype=np.float64)
    eu = np.full(M, -1, dtype=np.int64)
    ev = np.full(M, -1, dtype=np.int64)
    uf = list(range(n))
    top = list(range(n))

    def find(x):
        r = x
        while uf[r] != r:
            r = uf[r]
        while uf[x] != r:
            uf[x], x = r, uf[x]
        return r

    nxt = n
    for a, b, w in edges:
        ra, rb = find(idx[a]), find(idx[b])
        if ra == rb:
            continue
        p = nxt
        nxt += 1
        parent[top[ra]] = p
        parent[top[rb]] = p
        wt[p] = w
        eu[p] = a
        ev[p] = b
        uf[ra] = rb
        top[rb] = p
    K = nxt
    par = parent[:K]
    depth = np.zeros(K, dtype=np.int64)
    root = np.arange(K)
    for v in range(K - 1, -1, -1):
        p = par[v]
        if p >= 0:
            depth[v] = depth[p] + 1
            root[v] = root[p]
    LOG = max(1, int(np.ceil(np.log2(max(K, 2)))) + 1)
    up = np.empty((LOG, K), dtype=np.int64)
    up[0] = np.where(par >= 0, par, np.arange(K))
    for k in range(1, LOG):
        up[k] = up[k - 1][up[k - 1]]
    return wt, eu, ev, depth, root, up, LOG


def _lca_of_connected_pairs(krt, nu, nv):
    """For each pair joined by the tree, the reconstruction node that is their LCA.

    The merge recorded at that node is the weakest edge on the pair's path, so
    its weight is the bottleneck and its edge is the one a cut must deactivate.
    Pairs in different components are dropped; `qi` indexes those kept.
    """
    _, _, _, depth, root, up, levels = krt
    qi = np.flatnonzero((root[nu] == root[nv]) & (nu != nv))
    if qi.size == 0:
        return qi, qi
    deep = nu[qi].copy()
    shallow = nv[qi].copy()
    swap = depth[deep] < depth[shallow]
    deep[swap], shallow[swap] = shallow[swap].copy(), deep[swap].copy()

    # lift the deeper node to the shallower one's depth, then both together
    difference = depth[deep] - depth[shallow]
    for level in range(levels):
        selected = ((difference >> level) & 1).astype(bool)
        if selected.any():
            deep[selected] = up[level][deep[selected]]
    met = deep == shallow
    for level in range(levels - 1, -1, -1):
        up_deep, up_shallow = up[level][deep], up[level][shallow]
        apart = (~met) & (up_deep != up_shallow)
        if apart.any():
            deep[apart] = up_deep[apart]
            shallow[apart] = up_shallow[apart]
    return qi, np.where(met, deep, up[0][deep])


def _bottleneck_lca(mst, idx, nu, nv):
    """The weakest edge weight on the tree path, for many pairs at once.

    One reconstruction tree and one vectorised ascent answer every query. A
    tree path is unique, so these are the same values a per-pair path walk
    would give.
    """
    krt = _build_krt(mst, idx)
    qi, lca = _lca_of_connected_pairs(krt, nu, nv)
    if qi.size == 0:
        return qi, np.zeros(0, dtype=np.float64)
    return qi, krt[0][lca]


def _edges_to_cut(mst, idx, inv, nu, nv, nconf, alpha):
    """The MST edges to deactivate this round, for every pair below alpha.

    Returns {(a, b): (u, v)} in original node ids: the edge to cut, keyed with
    the lower id first, against the first negative pair in `negatives` order
    that demands the cut. `inv` maps a tree index back to its original id.
    """
    krt = _build_krt(mst, idx)
    wt, eu, ev = krt[0], krt[1], krt[2]
    qi, lca = _lca_of_connected_pairs(krt, nu, nv)
    if qi.size == 0:
        return {}

    need = (wt[lca] - nconf[qi]) < alpha
    sel = qi[need]
    if sel.size == 0:
        return {}
    ea, eb = eu[lca[need]], ev[lca[need]]
    lo = np.minimum(ea, eb)
    hi = np.maximum(ea, eb)
    order = np.argsort(sel, kind='stable')        # first-in-negatives-order wins
    lo, hi, sel = lo[order], hi[order], sel[order]
    key = lo.astype(np.int64) * (int(max(hi.max(), lo.max())) + 1) + hi
    _, first = np.unique(key, return_index=True)
    pu = inv[nu[sel[first]]]
    pv = inv[nv[sel[first]]]
    return {(int(lo[i]), int(hi[i])): (int(x), int(y))
            for i, x, y in zip(first, pu, pv, strict=True)}


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
_LABEL_CODE = {EdgeLabel.POSITIVE: 0, EdgeLabel.POSITIVE_INACTIVE: 1, EdgeLabel.NEGATIVE: 2}


class EdgeTable:
    """The graph's edges held as columns, so a sweep over them is vectorised.

    Several steps of the algorithm need every edge of a given label, or every
    edge crossing between two PCCs. Asking networkx means walking adjacency
    dictionaries in Python, which costs around three microseconds per edge
    visit, or roughly ninety seconds for one pass over the 14.5 million edges
    of the largest dataset, and the steps that need it run once per iteration.
    Holding the same edges as numpy columns turns each of those passes into a
    handful of array operations.

    The table therefore duplicates what the graph already knows, which is the
    one thing to keep in mind when changing this file: every write to an edge's
    label or confidence must go through :meth:`update`, and a new edge
    invalidates the table entirely because the columns would have to grow.
    :meth:`matches` rebuilds from the graph and compares, which is how a write
    site that forgot to call :meth:`update` is caught.
    """

    def __init__(self, u, v, label, confidence, row_of):
        self.u = u                      #: first endpoint, as given by the graph
        self.v = v                      #: second endpoint
        self.label = label              #: label codes; select with with_label()
        self.confidence = confidence
        self._row_of = row_of           #: {(low, high): row index}
        self._adjacency_rank = None

    @classmethod
    def build(cls, graph):
        """One O(E) pass over the graph."""
        rows = [(u, v, _LABEL_CODE[attr['data'].label], float(attr['data'].confidence))
                for u, v, attr in graph.edges(data=True) if attr.get('data') is not None]
        u = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        v = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
        label = np.fromiter((r[2] for r in rows), dtype=np.int8, count=len(rows))
        confidence = np.fromiter((r[3] for r in rows), dtype=np.float64, count=len(rows))
        row_of = {(a, b) if a <= b else (b, a): i for i, (a, b, _, _) in enumerate(rows)}
        return cls(u, v, label, confidence, row_of)

    def __len__(self):
        return int(self.u.size)

    def update(self, u, v, edge_data):
        """Write one edge's label and confidence back.

        Returns False when the edge is not in the table, which means it is new
        and the table has to be rebuilt rather than patched.
        """
        row = self._row_of.get((u, v) if u <= v else (v, u))
        if row is None:
            return False
        self.label[row] = _LABEL_CODE[edge_data.label]
        self.confidence[row] = float(edge_data.confidence)
        return True

    def with_label(self, label):
        """Mask of the rows carrying ``label``, which is an :class:`EdgeLabel`."""
        return self.label == _LABEL_CODE[label]

    def endpoints(self, mask=None):
        """``(u, v, confidence)`` for the selected rows, in table order."""
        if mask is None:
            return self.u, self.v, self.confidence
        rows = np.flatnonzero(mask)
        return self.u[rows], self.v[rows], self.confidence[rows]

    def ordered_endpoints(self):
        """``(low, high)`` per row, which is how a cross-PCC pair is keyed."""
        return np.minimum(self.u, self.v), np.maximum(self.u, self.v)

    def adjacency_rank(self, graph):
        """Per row, the position of the higher endpoint in the lower's adjacency.

        ``make_zero_stable`` historically built each PCC's subgraph by sweeping
        ``for a in pcc_set: for b in graph[a]`` and keeping b > a, so its edge
        insertion order, which networkx's MST uses to break equal-weight ties,
        is (position of a in the PCC set, position of b in a's adjacency). This
        is the second key, for every edge in one pass. Graph structure does not
        change during a run, since adding an edge drops the table, so it is
        built once.
        """
        if self._adjacency_rank is not None:
            return self._adjacency_rank
        lows, highs, ranks = array('q'), array('q'), array('q')
        for node, neighbours in graph._adj.items():
            for rank, other in enumerate(neighbours):
                if node < other:
                    lows.append(node)
                    highs.append(other)
                    ranks.append(rank)
        lows = np.frombuffer(lows, dtype=np.int64)
        highs = np.frombuffer(highs, dtype=np.int64)
        ranks = np.frombuffer(ranks, dtype=np.int64)

        low, high = self.ordered_endpoints()
        stride = int(max(high.max() if high.size else 0,
                         highs.max() if highs.size else 0)) + 1
        order = np.argsort(lows * stride + highs, kind='stable')
        adjacency_keys = (lows * stride + highs)[order]
        wanted = low * stride + high
        position = np.searchsorted(adjacency_keys, wanted)
        if position.size and not np.array_equal(
                adjacency_keys[np.minimum(position, adjacency_keys.size - 1)], wanted):
            raise AssertionError('adjacency rank alignment failed')
        self._adjacency_rank = ranks[order][position]
        return self._adjacency_rank

    def matches(self, graph):
        """True when a fresh build from ``graph`` agrees with this table."""
        fresh = EdgeTable.build(graph)
        if len(fresh) != len(self):
            return False
        for key, row in fresh._row_of.items():
            mine = self._row_of.get(key)
            if (mine is None or fresh.label[row] != self.label[mine]
                    or abs(fresh.confidence[row] - self.confidence[mine]) > 1e-12):
                return False
        return True


class StabilityGraph:
    """The identification graph: labelled edges, PCCs, and stability."""

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
        # The edges as columns, built on first use and kept in step by
        # _touch_edge at every label or confidence write. See EdgeTable.
        self._edges: Optional[EdgeTable] = None

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
                self._touch_edge(u, v, new_data)
            else:
                new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
                self.G[u][v]['data'] = new_data
                self._touch_edge(u, v, new_data)
        else:
            new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
            self.G.add_edge(u, v, data=new_data)
            self._drop_edge_table()          # a new edge, so the columns must be rebuilt
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

    # ------------------------------------------------------------------ arrays
    def edge_pcc_ids(self):
        """The edge columns, plus the PCC id of each endpoint, -1 where there is none.

        Every per-iteration sweep starts here: select rows by label, then ask
        whether the two endpoints sit in the same PCC or in different ones.
        """
        self._ensure_pcc_cache()
        table = self.edges_table()
        if len(table) == 0:
            empty = np.zeros(0, dtype=np.int64)
            return table, empty, empty, np.zeros(0, dtype=np.int64)
        size = int(max(table.u.max(), table.v.max())) + 1
        pcc_of = np.full(size, -1, dtype=np.int64)
        for node, pcc_id in self._node_to_pcc.items():
            if 0 <= node < size:
                pcc_of[node] = pcc_id
        return table, pcc_of[table.u], pcc_of[table.v], pcc_of

    def edges_table(self):
        """The edges as columns, built on first use."""
        if self._edges is None:
            self._edges = EdgeTable.build(self.G)
        return self._edges

    def _drop_edge_table(self):
        """Forget the columns, because the set of edges itself changed."""
        self._edges = None

    def _touch_edge(self, u, v, edge_data):
        """Write one edge's label and confidence through to the columns."""
        if self._edges is not None and not self._edges.update(u, v, edge_data):
            self._drop_edge_table()             # a new edge; rebuild on next use

    def verify_edge_table(self):
        """True when the columns still agree with the graph.

        The columns duplicate the graph, so a write site that forgot to call
        _touch_edge would leave them stale. This is what catches that.
        """
        return self._edges is None or self._edges.matches(self.G)

    def deactivate_positive(self, edge: Tuple[int, int],
                            deactivator: Optional[Tuple[int, int]] = None):
        """Deactivate a positive edge, making it positive-inactive.

        ``deactivator`` records the negative edge that caused the cut.
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
        self._touch_edge(u, v, edge_data)
        self._pos_inactive_edges.add((min(u, v), max(u, v)))
        self._pcc_cache_valid = False


    def _invalidate_cache(self):
        """Invalidate PCC cache. MST is rebuilt explicitly when needed."""
        self._pcc_cache_valid = False

    def _positive_edges_scan_order(self):
        """(u, v, conf) arrays of POSITIVE edges in exact G.edges(data=True) order."""
        table = self.edges_table()
        return table.endpoints(table.with_label(EdgeLabel.POSITIVE))

    def _ensure_pcc_cache(self):
        if not self._pcc_cache_valid:
            self._compute_pccs()
            self._pcc_cache_valid = True

    def _compute_pccs(self):
        """The connected components of the positive edges alone.

        Positive-inactive edges do not join a component; that is what
        deactivating one means. Runs after every label change.
        """
        # The edges come from the table in G.edges order, so component ids and
        # the iteration order within a component are stable across calls.
        _pu, _pv, _ = self._positive_edges_scan_order()
        positive_edges = list(zip(_pu.tolist(), _pv.tolist(), strict=True))

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
        # Same mirror-backed positive edge list as _compute_pccs (identical order,
        # orientation and weights), so the forest and its tie-breaking are unchanged.
        # This full scan ran at the start of every candidate-generation call.
        _pu, _pv, _pc = self._positive_edges_scan_order()
        positive_edges = [(u, v, {'weight': c})
                          for u, v, c in zip(_pu.tolist(), _pv.tolist(), _pc.tolist(),
                                             strict=True)]

        self._mst_forest = nx.Graph()
        self._mst_forest.add_nodes_from(self.G.nodes())
        self._mst_forest.add_edges_from(positive_edges)

        # Build MST forest (one MST per connected component)
        self._mst_forest = nx.maximum_spanning_tree(self._mst_forest, weight='weight')
    def pcc_separation_strength(self) -> Dict[int, float]:
        """Per PCC, how confidently it is separated from its neighbours.

        For each PCC, the sum over the other PCCs it has a negative edge to of
        the strongest such edge. A high sum means every boundary is confidently
        drawn, a low one that the PCC is weakly separated from its neighbours.
        Reported at the end of a run; see tracking.log_graph_stats.
        """
        self._ensure_pcc_cache()
        table, _, _, pcc_of = self.edge_pcc_ids()
        if len(table) == 0:
            return {}

        negative = table.with_label(EdgeLabel.NEGATIVE)
        pu = pcc_of[table.u[negative]]
        pv = pcc_of[table.v[negative]]
        conf = table.confidence[negative]
        keep = (pu >= 0) & (pv >= 0) & (pu != pv)
        if not keep.any():
            return {}
        lo = np.minimum(pu[keep], pv[keep])
        hi = np.maximum(pu[keep], pv[keep])
        conf = conf[keep]

        # Group the crossing negatives by PCC pair and keep the strongest of each.
        pair_key = lo * (int(hi.max()) + 1) + hi
        unique_keys, inverse = np.unique(pair_key, return_inverse=True)
        strongest = np.zeros(unique_keys.size, dtype=np.float64)
        np.maximum.at(strongest, inverse, conf)

        # One representative row per pair, to recover its two PCC ids.
        first = np.zeros(unique_keys.size, dtype=np.int64)
        first[inverse[::-1]] = np.arange(inverse.size)[::-1]

        strength: Dict[int, float] = defaultdict(float)
        for a_id, b_id, value in zip(lo[first].tolist(), hi[first].tolist(),
                                     strongest.tolist(), strict=True):
            strength[a_id] += value
            strength[b_id] += value
        return dict(strength)

    def pccs(self):
        """The PCCs, recomputed if an edge label has changed since the last call."""
        self._ensure_pcc_cache()
        return self._pccs

    def mst_forest(self):
        """The maximum spanning forest of the active positive edges.

        Rebuilt on every call, since a cut changes it. Callers that need it
        more than once bind it to a name.
        """
        self._build_mst_forest()
        return self._mst_forest

    def get_pccs(self) -> List[Set[int]]:
        self._ensure_pcc_cache()
        return self._pccs.copy()

    def per_pcc_internal_stability(self):
        """Per-PCC internal stability, for tracking over time.

        Returns a list of (anchor_node, size, internal_stability), one per PCC.
        anchor_node = min node id (a stable-ish identity across iterations).
        internal_stability = min over within-PCC negative pairs of
        (MSP - neg_conf); None if the PCC has no internal negative (nothing can
        destabilize it). Negative value = unstable (a cut is pending).
        """
        self._ensure_pcc_cache()
        # One vectorised bottleneck pass per PCC, rather than a path search per
        # negative edge. This runs on every Phase 0 iteration and throughout
        # review, on PCCs of several thousand nodes.
        intra = self.intra_pcc_edges()
        out = []
        for pcc_id, pcc in enumerate(self._pccs):
            pcc_list = sorted(pcc)
            anchor = pcc_list[0]
            if len(pcc_list) < 2:
                out.append((anchor, len(pcc_list), None))
                continue
            neg, pos = intra.get(pcc_id, ([], []))
            if not neg or not pos:
                out.append((anchor, len(pcc_list), None))
                continue
            pg = nx.Graph()
            for u, v, c in pos:
                pg.add_edge(u, v, weight=c)
            mst = nx.maximum_spanning_tree(pg, weight='weight')
            nodes = list(mst.nodes())
            nidx = {n: i for i, n in enumerate(nodes)}
            keep = [(u, v, nc) for (u, v, nc) in neg if u in nidx and v in nidx]
            if not keep:
                out.append((anchor, len(pcc_list), None))
                continue
            nu = np.fromiter((nidx[e[0]] for e in keep), dtype=np.int64, count=len(keep))
            nv = np.fromiter((nidx[e[1]] for e in keep), dtype=np.int64, count=len(keep))
            nc = np.fromiter((e[2] for e in keep), dtype=np.float64, count=len(keep))
            qi, w = _bottleneck_lca(mst, nidx, nu, nv)
            if qi.size == 0:
                out.append((anchor, len(pcc_list), None))
                continue
            min_stab = float(np.min(w - nc[qi]))
            out.append((anchor, len(pcc_list), min_stab))
        return out

    def intra_pcc_edges(self):
        """{pcc_id: (negatives, positives)} for intra-PCC edges, each list of
        (u, v, confidence) in ascending (u, v) order -- the same order the old
        node-pair loop produced, so MSTs built from `positives` break ties the
        same way. One vectorised pass over the edge mirror."""
        res: Dict[int, Tuple[list, list]] = {}
        table, pu, pv, _ = self.edge_pcc_ids()
        if len(table) == 0:
            return res
        lab = table.label
        neg_code = _LABEL_CODE[EdgeLabel.NEGATIVE]
        pos_code = _LABEL_CODE[EdgeLabel.POSITIVE]
        m = (pu >= 0) & (pu == pv) & ((lab == neg_code) | (lab == pos_code))
        sel = np.flatnonzero(m)
        if sel.size == 0:
            return res
        lo = np.minimum(table.u[sel], table.v[sel])
        hi = np.maximum(table.u[sel], table.v[sel])
        order = np.lexsort((hi, lo, pu[sel]))
        sel = sel[order]
        lo = lo[order]
        hi = hi[order]
        p = pu[sel]
        c = table.confidence[sel]
        labels = lab[sel]
        for i in range(sel.size):
            b = res.get(int(p[i]))
            if b is None:
                b = ([], [])
                res[int(p[i])] = b
            b[0 if labels[i] == neg_code else 1].append((int(lo[i]), int(hi[i]), float(c[i])))
        return res

    def external_stabilities(self):
        """External stability for every PCC pair that has a crossing negative.

        How much support there is for keeping the two apart, with the same sign
        convention as internal stability: at or above zero is supported, below
        it is unstable. A flat list, read as a distribution, so the pair each
        value belongs to is not reported.
        """
        self._ensure_pcc_cache()
        max_neg, max_pi = self._cross_pcc_pair_maxima()
        return [max_neg[pair] - max_pi.get(pair, 0.0) for pair in max_neg]

    def _cross_pcc_pair_maxima(self):
        """(max negative conf, max positive-inactive conf) per cross-PCC pair."""
        table, pu, pv, _ = self.edge_pcc_ids()
        if len(table) == 0:
            return {}, {}
        cross = (pu >= 0) & (pv >= 0) & (pu != pv)
        lo = np.minimum(pu, pv)
        hi = np.maximum(pu, pv)
        K = int(hi.max()) + 1 if hi.size else 1
        conf = table.confidence

        def pair_max(mask):
            i = np.flatnonzero(mask)
            if i.size == 0:
                return {}
            k = lo[i] * K + hi[i]
            uk, iv = np.unique(k, return_inverse=True)
            mx = np.zeros(uk.size, dtype=np.float64)
            np.maximum.at(mx, iv, conf[i])
            return {(int(x // K), int(x % K)): float(y)
                    for x, y in zip(uk, mx, strict=True)}

        return (pair_max(cross & table.with_label(EdgeLabel.NEGATIVE)),
                pair_max(cross & table.with_label(EdgeLabel.POSITIVE_INACTIVE)))

    def compute_external_stability(self, pcc_a: int, pcc_b: int) -> Optional[float]:
        """External stability of two PCCs.

        The strongest negative edge joining them, minus the strongest
        positive-inactive edge joining them.

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
                            if (edge_data and edge_data.label == EdgeLabel.NEGATIVE
                                    and edge_data.confidence > max_neg_conf):
                                max_neg_conf = edge_data.confidence
                                max_neg_edge = (u, v)

                    if max_neg_edge:
                        unstable.append((pcc_a, pcc_b, stability, max_neg_edge))

        return unstable

    def _intra_edges_sweep_order(self):
        """Per PCC id: (pos_lo, pos_hi, pos_conf, neg_lo, neg_hi, neg_conf) arrays in
        the exact order the historical adjacency sweep in make_zero_stable emitted
        them: PCC by PCC, then position of lo in the PCC set, then position of hi in
        G._adj[lo]."""
        self._ensure_pcc_cache()
        out = {}
        table = self.edges_table()
        if len(table) == 0:
            return out
        n = int(max(table.u.max(), table.v.max())) + 1
        pcc_of = np.full(n, -1, dtype=np.int64)
        setpos = np.zeros(n, dtype=np.int64)
        for pid, pcc in enumerate(self._pccs):
            for i, nd in enumerate(pcc):
                if 0 <= nd < n:
                    pcc_of[nd] = pid
                    setpos[nd] = i
        pu = pcc_of[table.u]
        pv = pcc_of[table.v]
        lab = table.label
        pos_code = _LABEL_CODE[EdgeLabel.POSITIVE]
        neg_code = _LABEL_CODE[EdgeLabel.NEGATIVE]
        sel = np.flatnonzero((pu >= 0) & (pu == pv) & ((lab == pos_code) | (lab == neg_code)))
        if sel.size == 0:
            return out
        lo = np.minimum(table.u[sel], table.v[sel])
        hi = np.maximum(table.u[sel], table.v[sel])
        order = np.lexsort((table.adjacency_rank(self.G)[sel], setpos[lo], pu[sel]))
        sel = sel[order]
        lo = lo[order]
        hi = hi[order]
        pp = pu[sel]
        ll = lab[sel]
        cc = table.confidence[sel]
        starts = np.flatnonzero(np.r_[True, pp[1:] != pp[:-1]])
        ends = np.r_[starts[1:], pp.size]
        for st, en in zip(starts.tolist(), ends.tolist(), strict=True):
            gl = ll[st:en]
            ip = gl == pos_code
            ineg = gl == neg_code
            out[int(pp[st])] = (lo[st:en][ip], hi[st:en][ip], cc[st:en][ip],
                                lo[st:en][ineg], hi[st:en][ineg], cc[st:en][ineg])
        return out

    def make_zero_stable(self, alpha: float = 0.0) -> int:
        """Step 2: deactivate positive edges until the graph is alpha-stable.

        Needs no human input. Negative edges are left alone, since a negative
        edge between two PCCs is already satisfied by their separation.

        Args:
            alpha: Stability threshold. Default 0.0 for strict 0-stability.
                   Negative values (e.g., -0.1) allow small instabilities,
                   resulting in less aggressive fragmentation.

        For each within-PCC negative pair (u,v) with stability < alpha, one
        step of Step 2 inactivates the weakest positive edge on MSP(u,v). The
        PCC's maximum spanning tree is RECOMPUTED after each cut and the loop
        repeats until every internal negative pair is at least alpha-stable --
        per the paper, "once a negative starts to cause the removal of positive
        edges, it continues until v_i and v_j are in different PCCs". Recomputing
        the tree is what lets non-tree positive edges be reconsidered, so a
        densely-connected pair is actually separated instead of leaving a severed
        spanning-tree path while other edges still join it.

        Returns: Number of edges deactivated.

        Cost: no global MST forest is built. Only PCCs that contain a negative
        edge do any work, each on its own (small) positive subgraph; a PCC that
        is already 0-stable costs a single local tree build.
        """
        deactivations = 0

        self._pcc_cache_valid = False
        self._ensure_pcc_cache()

        # Group every PCC's edges once, up front. A cut inside one PCC only
        # relabels that PCC's own edges, and nothing in the loop below refreshes
        # the PCC cache, so the grouping stays correct throughout.
        _groups = self._intra_edges_sweep_order()

        for _pid, pcc in enumerate(list(self._pccs)):
            if len(pcc) < 2:
                continue
            pcc_set = pcc if isinstance(pcc, set) else set(pcc)
            _g = _groups.get(_pid)

            P = nx.Graph()
            P.add_nodes_from(pcc_set)
            if _g is not None and _g[0].size:
                P.add_edges_from((a, b, {'weight': c}) for a, b, c in
                                 zip(_g[0].tolist(), _g[1].tolist(), _g[2].tolist(),
                                     strict=True))
            if _g is None or _g[3].size == 0:
                continue

            pcc_deactivations = 0
            # Query arrays are built ONCE per PCC and reused by every cutting
            # round: only P's EDGES change as we cut, never its node set.
            _nodes = list(P.nodes())
            _idx = {n: i for i, n in enumerate(_nodes)}
            _inv = np.array(_nodes, dtype=np.int64)
            _nu = _g[3]
            _nv = _g[4]
            _nc = _g[5]
            _map = np.zeros(int(max(_inv.max(), _nu.max(), _nv.max())) + 1, dtype=np.int64)
            _map[_inv] = np.arange(len(_nodes), dtype=np.int64)
            _nu = _map[_nu]
            _nv = _map[_nv]
            while P.number_of_edges() > 0:
                mst = nx.maximum_spanning_tree(P, weight='weight')
                # Cut the weakest edge of every pair below alpha in one round,
                # then rebuild the tree. Cutting them one at a time would mean
                # one MST rebuild per cut, and an over-merged PCC needs
                # thousands of them.
                to_cut = _edges_to_cut(mst, _idx, _inv, _nu, _nv, _nc, alpha)

                if not to_cut:
                    break  # every negative pair is now >= alpha-stable

                for (a, b), deact in to_cut.items():
                    if P.has_edge(a, b):
                        self.deactivate_positive((a, b), deactivator=deact)
                        P.remove_edge(a, b)
                        deactivations += 1
                        pcc_deactivations += 1

            if pcc_deactivations > 0:
                logger.info(f"PCC (size {len(pcc)}): deactivated "
                            f"{pcc_deactivations} edges for {alpha}-stability")
        self._pcc_cache_valid = False
        return deactivations

    def apply_human_review(self, u: int, v: int, human_agrees: bool, ch: float,
                           cap_confidence: bool = True):
        """Step 4: apply one review verdict to its edge.

        A verdict agreeing with the current label adds ``ch`` to the
        confidence, one disagreeing subtracts it, and a confidence driven below
        zero flips the label.

        With ``cap_confidence`` the accumulated confidence is clamped at 1.0.
        Without it, confirmations accumulate without bound, so a single
        conflicting review cannot flip a well established edge.
        """
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v]['data']

        old_label = edge_data.label
        edge_data.ranker = 'human'

        if human_agrees:
            new_conf = edge_data.confidence + ch
            edge_data.confidence = min(1.0, new_conf) if cap_confidence else new_conf
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
        self._touch_edge(u, v, edge_data)

        # Update edge counts and pos-inactive tracking if label changed
        if edge_data.label != old_label:
            self._decrement_edge_count(old_label)
            self._increment_edge_count(edge_data.label)
            key = (min(u, v), max(u, v))
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.add(key)
            elif old_label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.discard(key)

        # Saturate the score, so that anything reading edge_data.score sees the
        # human verdict without having to check the ranker. A classifier puts a
        # positive at roughly 0.6 to 0.8; a confirmed one goes to 1.0.
        if edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
            edge_data.score = 1.0
        elif edge_data.label == EdgeLabel.NEGATIVE:
            edge_data.score = 0.0

        self._invalidate_cache()

    def get_clustering(self) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
        """Get current clustering as dictionaries."""
        self._ensure_pcc_cache()
        cluster_dict = dict(enumerate(self._pccs))
        node2cid = {node: pcc_id for pcc_id, pcc in enumerate(self._pccs) for node in pcc}
        return cluster_dict, node2cid

    def clustering_with_view(self):
        """Clustering plus a read-only graph view, WITHOUT copying the edge set.

        A full `get_clustering()` materialises a fresh networkx graph holding
        every edge (14,569,752 on whale shark 2023) with five attributes each,
        about 80 s, and validation called it at every point -- some 74 min of a
        5 h run. The validator only hands the graph to `create_reachable`,
        which uses `subgraph` and `connected_components`, so connectivity is
        all it needs. The edge filter mirrors `get_clustering()`, which also
        skips edges carrying no data, so the reachable clustering is unchanged.
        """
        cluster_dict, node2cid = self.get_clustering()
        view = nx.subgraph_view(
            self.G, filter_edge=lambda u, v: self.G[u][v].get('data') is not None)
        return cluster_dict, node2cid, view

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph state.

        Optimized to avoid iterating all edges (which can be 10M+).
        - Edge counts: maintained incrementally
        - Internal stability: iterates PCC node pairs (O(N^2) per PCC, N small)
        - External stability: iterates only positive-inactive edges (O(small) vs O(10M))
        """
        self._ensure_pcc_cache()

        pcc_sizes = [len(pcc) for pcc in self._pccs]

        # The minimum over PCCs of each one's own minimum, which is what
        # per_pcc_internal_stability already computes. A PCC with no internal
        # negative edge reports None, since nothing can destabilise it.
        min_internal = min((stability for _, _, stability
                            in self.per_pcc_internal_stability()
                            if stability is not None), default=float('inf'))

        # A PCC pair with no positive-inactive edge still has a finite external
        # stability, equal to its strongest negative confidence.
        min_external = float('inf')
        pcc_pair_max_neg: Dict[Tuple[int, int], float] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        # Cross-PCC maxima from the mirror (shared with external_stabilities()).
        pcc_pair_max_neg, pcc_pair_pos_inactive = self._cross_pcc_pair_maxima()

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

