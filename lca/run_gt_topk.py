"""
GT Top-K Graph reference clustering.

Algorithm:
  1. Compute pairwise cosine similarities from embeddings
  2. For each node, find its top-K most similar neighbors (K=10 by default)
  3. These connections form the edges of the graph (fixed edge set)
  4. Use GT to label each edge: same individual -> positive, different -> negative
  5. Connected components of positive edges = predicted clusters (PCCs)
  6. Report clustering metrics

Unlike GT PCC (which sweeps a similarity threshold), this method fixes the
graph topology via top-K neighbors, then relies on GT to label edges.
The number of edges is determined entirely by K and the number of nodes.

Usage:
  python3 run_gt_topk.py --dataset GZCD
  python3 run_gt_topk.py --dataset GZCD --topk 10
  python3 run_gt_topk.py --dataset GZCD --sweep_k
  python3 run_gt_topk.py --dataset beluga --topk 15
"""

import argparse
import os
import sys
import time
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

# Import shared utilities
from run_nis_reference import DATASET_CONFIGS, load_and_filter, hungarian_metrics


# ---------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------

_log_file = None

def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


# ---------------------------------------------------------------
# Core GT Top-K algorithm
# ---------------------------------------------------------------

def build_topk_edges(sim_matrix, topk):
    """
    Build a fixed edge set by connecting each node to its top-K
    most similar neighbors (by cosine similarity).

    Returns:
        edges: set of (i, j) tuples with i < j (undirected, deduplicated)
        similarities: dict mapping (i, j) -> cosine similarity
    """
    n = sim_matrix.shape[0]
    edges = set()
    similarities = {}

    for i in range(n):
        # Zero out self-similarity to avoid self-loops
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf

        # Top-K neighbors by similarity
        topk_indices = np.argpartition(sims, -topk)[-topk:]

        for j in topk_indices:
            edge = (min(i, j), max(i, j))
            if edge not in edges:
                edges.add(edge)
                similarities[edge] = sim_matrix[i, j]

    return edges, similarities


def gt_topk_clustering(sim_matrix, gt_labels, topk):
    """
    Build GT Top-K clustering.

    1. For each node, find top-K most similar neighbors -> fixed edge set
    2. Label each edge using GT: same individual -> positive, different -> negative
    3. Connected components of positive edges = predicted clusters

    Returns:
        pred_labels: np array of cluster assignments
        stats: dict with edge counts and clustering info
    """
    n = len(gt_labels)
    gt_arr = np.asarray(gt_labels)

    # Step 1: Build top-K graph
    edges, similarities = build_topk_edges(sim_matrix, topk)

    # Step 2: Label edges with GT
    positive_edges = []
    negative_edges = []
    pos_sims = []
    neg_sims = []

    for (i, j) in edges:
        sim = similarities[(i, j)]
        if gt_arr[i] == gt_arr[j]:
            positive_edges.append((i, j))
            pos_sims.append(sim)
        else:
            negative_edges.append((i, j))
            neg_sims.append(sim)

    # Count total same-individual pairs (for recall computation)
    triu_i, triu_j = np.triu_indices(n, k=1)
    same_individual = gt_arr[triu_i] == gt_arr[triu_j]
    total_positive_pairs = int(same_individual.sum())

    # How many same-individual pairs are NOT in the edge set?
    edge_set = set(edges)
    missed_positive = 0
    for idx in range(len(triu_i)):
        if same_individual[idx]:
            pair = (int(triu_i[idx]), int(triu_j[idx]))
            if pair not in edge_set:
                missed_positive += 1

    # Step 3: Build graph from positive edges, find connected components
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(positive_edges)

    pred_labels = np.full(n, -1, dtype=int)
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            pred_labels[node] = cid
    num_clusters = cid + 1 if n > 0 else 0

    stats = {
        'topk': topk,
        'total_edges': len(edges),
        'num_positive_edges': len(positive_edges),
        'num_negative_edges': len(negative_edges),
        'num_missed_positive': missed_positive,
        'num_total_positive_pairs': total_positive_pairs,
        'num_clusters': num_clusters,
        'total_gt_reviews': len(edges),
    }

    if pos_sims:
        stats['pos_sim_min'] = float(np.min(pos_sims))
        stats['pos_sim_mean'] = float(np.mean(pos_sims))
        stats['pos_sim_max'] = float(np.max(pos_sims))
    if neg_sims:
        stats['neg_sim_min'] = float(np.min(neg_sims))
        stats['neg_sim_mean'] = float(np.mean(neg_sims))
        stats['neg_sim_max'] = float(np.max(neg_sims))

    return pred_labels, stats


# ---------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------

def pairwise_metrics(pred_labels, gt_labels):
    """Compute pairwise TP/FP/FN/precision/recall/F1."""
    n = len(gt_labels)
    gt_same = defaultdict(set)
    pred_same = defaultdict(set)
    for i in range(n):
        gt_same[gt_labels[i]].add(i)
        pred_same[pred_labels[i]].add(i)

    tp_gt = sum(len(s) * (len(s) - 1) // 2 for s in gt_same.values())
    tp_pred = sum(len(s) * (len(s) - 1) // 2 for s in pred_same.values())

    tp = 0
    for gt_c in gt_same.values():
        counts = defaultdict(int)
        for node in gt_c:
            counts[pred_labels[node]] += 1
        tp += sum(c * (c - 1) // 2 for c in counts.values())

    fp = tp_pred - tp
    fn = tp_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)

    return {'p_precision': precision, 'p_recall': recall, 'p_f1': f1}


def compute_all_metrics(pred_labels, gt_labels, true_K):
    """Compute all evaluation metrics for a clustering result."""
    h = hungarian_metrics(pred_labels, gt_labels)
    p = pairwise_metrics(pred_labels, gt_labels)
    ari = adjusted_rand_score(gt_labels, pred_labels)
    num_clusters = len(set(pred_labels))
    k_ratio = num_clusters / max(1, true_K)

    return {
        'h_f1': h['h_f1'],
        'h_precision': h['h_precision'],
        'h_recall': h['h_recall'],
        'p_f1': p['p_f1'],
        'p_precision': p['p_precision'],
        'p_recall': p['p_recall'],
        'ari': ari,
        'num_clusters': num_clusters,
        'true_K': true_K,
        'k_ratio': k_ratio,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GT Top-K Graph reference clustering")
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of nearest neighbors per node (default: 10)')
    parser.add_argument('--sweep_k', action='store_true',
                        help='Sweep K values from 1 to 50')
    parser.add_argument('--k_values', type=int, nargs='+', default=None,
                        help='Specific K values to sweep (default: 1..50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Setup logging
    global _log_file
    dataset = args.dataset
    log_dir = f"tmp/{dataset}/output"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gt_topk.log")
    _log_file = open(log_path, 'w')

    log(f"{'='*70}")
    log(f"GT Top-K Graph Reference Clustering: {dataset}")
    log(f"{'='*70}")

    # Load data
    log(f"\nLoading {dataset}...")
    t0 = time.time()
    embeddings, gt_labels, true_K = load_and_filter(dataset)
    n = len(embeddings)
    log(f"  Loaded in {time.time()-t0:.1f}s: {n} nodes, {true_K} true clusters")

    # Compute pairwise cosine similarity
    log(f"\nComputing {n}x{n} cosine similarity matrix...")
    t0 = time.time()
    sim_matrix = cosine_similarity(embeddings)
    log(f"  Done in {time.time()-t0:.1f}s")

    # Summary of similarity distribution
    triu_i, triu_j = np.triu_indices(n, k=1)
    all_sims = sim_matrix[triu_i, triu_j]
    same_mask = gt_labels[triu_i] == gt_labels[triu_j]
    log(f"\nSimilarity statistics:")
    log(f"  All pairs:     min={all_sims.min():.4f}  mean={all_sims.mean():.4f}  max={all_sims.max():.4f}")
    if same_mask.sum() > 0:
        log(f"  Same-indiv:    min={all_sims[same_mask].min():.4f}  mean={all_sims[same_mask].mean():.4f}  max={all_sims[same_mask].max():.4f}")
    if (~same_mask).sum() > 0:
        log(f"  Diff-indiv:    min={all_sims[~same_mask].min():.4f}  mean={all_sims[~same_mask].mean():.4f}  max={all_sims[~same_mask].max():.4f}")
    log(f"  Total same-individual pairs: {same_mask.sum()}")
    log(f"  Total diff-individual pairs: {(~same_mask).sum()}")

    if args.sweep_k:
        # Sweep mode
        if args.k_values:
            k_values = sorted(args.k_values)
        else:
            k_values = list(range(1, 51))

        log(f"\n{'='*70}")
        log(f"Top-K sweep: K = {k_values[0]}..{k_values[-1]} ({len(k_values)} values)")
        log(f"{'='*70}")

        header = (f"{'K':>5s} | {'Edges':>8s} | {'Pos_Edges':>10s} | {'Neg_Edges':>10s} | "
                  f"{'Missed_Pos':>10s} | {'GT_Reviews':>10s} | "
                  f"{'K_pred':>7s} | {'K_true':>7s} | {'K_ratio':>7s} | "
                  f"{'P_F1':>6s} | {'P_Prec':>6s} | {'P_Rec':>6s} | "
                  f"{'H_F1':>6s} | {'H_Prec':>6s} | {'H_Rec':>6s} | "
                  f"{'ARI':>6s}")
        log(f"\n{header}")
        log("-" * len(header))

        best_hf1 = -1
        best_k = 0
        best_row = ""

        for k in k_values:
            pred_labels, stats = gt_topk_clustering(sim_matrix, gt_labels, k)
            m = compute_all_metrics(pred_labels, gt_labels, true_K)

            row = (f"K={k:>4d} | "
                   f"Edges={stats['total_edges']:>8d} | "
                   f"Pos_Edges={stats['num_positive_edges']:>10d} | "
                   f"Neg_Edges={stats['num_negative_edges']:>10d} | "
                   f"Missed_Pos={stats['num_missed_positive']:>10d} | "
                   f"GT_Reviews={stats['total_gt_reviews']:>10d} | "
                   f"K_pred={m['num_clusters']:>7d} | "
                   f"K_true={m['true_K']:>7d} | "
                   f"K_ratio={m['k_ratio']:>7.3f} | "
                   f"P_F1={m['p_f1']:>6.4f} | "
                   f"P_Prec={m['p_precision']:>6.4f} | "
                   f"P_Rec={m['p_recall']:>6.4f} | "
                   f"H_F1={m['h_f1']:>6.4f} | "
                   f"H_Prec={m['h_precision']:>6.4f} | "
                   f"H_Rec={m['h_recall']:>6.4f} | "
                   f"ARI={m['ari']:>6.4f}")
            log(row)

            if m['h_f1'] > best_hf1:
                best_hf1 = m['h_f1']
                best_k = k
                best_row = row

        log(f"\n{'='*70}")
        log(f"BEST RESULT (by Hungarian F1):")
        log(f"  K = {best_k}")
        log(f"  {best_row}")
        log(f"{'='*70}")

    else:
        # Single K mode
        topk = args.topk
        log(f"\nRunning GT Top-K with K = {topk}")
        t0 = time.time()

        pred_labels, stats = gt_topk_clustering(sim_matrix, gt_labels, topk)
        m = compute_all_metrics(pred_labels, gt_labels, true_K)
        elapsed = time.time() - t0

        log(f"\n{'='*70}")
        log(f"GT Top-K Results (K={topk})")
        log(f"{'='*70}")
        log(f"  Graph construction + clustering in {elapsed:.1f}s")
        log(f"")
        log(f"  Graph topology:")
        log(f"    Nodes:                 {n:,}")
        log(f"    Total edges (fixed):   {stats['total_edges']:,}  (each node -> top-{topk} neighbors)")
        log(f"    Avg degree:            {2 * stats['total_edges'] / n:.1f}")
        log(f"")
        log(f"  GT edge labeling:")
        log(f"    Positive edges (GT):   {stats['num_positive_edges']:,}  (same individual)")
        log(f"    Negative edges (GT):   {stats['num_negative_edges']:,}  (different individual)")
        log(f"    Total GT reviews:      {stats['total_gt_reviews']:,}")
        log(f"    Missed positive pairs: {stats['num_missed_positive']:,} / {stats['num_total_positive_pairs']:,}")
        pos_recall = 1 - stats['num_missed_positive'] / max(1, stats['num_total_positive_pairs'])
        log(f"    Positive recall:       {pos_recall:.4f}")
        log(f"")
        if 'pos_sim_mean' in stats:
            log(f"  Edge similarity distribution:")
            log(f"    Positive edges:  min={stats['pos_sim_min']:.4f}  mean={stats['pos_sim_mean']:.4f}  max={stats['pos_sim_max']:.4f}")
        if 'neg_sim_mean' in stats:
            log(f"    Negative edges:  min={stats['neg_sim_min']:.4f}  mean={stats['neg_sim_mean']:.4f}  max={stats['neg_sim_max']:.4f}")
        log(f"")
        log(f"  Clustering results:")
        log(f"    Predicted clusters:    {m['num_clusters']}")
        log(f"    True clusters:         {m['true_K']}")
        log(f"    Cluster count ratio:   {m['k_ratio']:.4f}")
        log(f"")
        log(f"  Hungarian F1:          {m['h_f1']:.4f}")
        log(f"  Hungarian Precision:   {m['h_precision']:.4f}")
        log(f"  Hungarian Recall:      {m['h_recall']:.4f}")
        log(f"")
        log(f"  Pairwise F1:           {m['p_f1']:.4f}")
        log(f"  Pairwise Precision:    {m['p_precision']:.4f}")
        log(f"  Pairwise Recall:       {m['p_recall']:.4f}")
        log(f"")
        log(f"  ARI:                   {m['ari']:.4f}")
        log(f"{'='*70}")

    log(f"\nLog saved to: {log_path}")
    _log_file.close()


if __name__ == '__main__':
    main()
