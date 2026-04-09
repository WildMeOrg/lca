"""
GT PCC (Ground Truth Positive Connected Components) reference clustering.

Upper-bound clustering algorithm: given the MIEWID embedding graph, what is
the best clustering achievable if every edge is perfectly labeled by GT?

Algorithm:
  1. Compute pairwise cosine similarities from embeddings
  2. At a given threshold, create candidate edges (sim > threshold)
  3. Use GT to label each edge: same individual → positive, different → discard
  4. Connected components of positive edges = predicted clusters (PCCs)

Key property: precision is always 1.0 (no false positives possible with GT).
The only error is missed positive edges — same-individual pairs below threshold.
This directly measures embedding quality / recall.

Usage:
  python3 run_gt_pcc.py --dataset beluga --sweep
  python3 run_gt_pcc.py --dataset beluga --sweep --num_steps 100
  python3 run_gt_pcc.py --dataset beluga --threshold 0.72
  python3 run_gt_pcc.py --dataset beluga --max_flips 5000
  python3 run_gt_pcc.py --dataset GZCD --sweep
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

# Import shared utilities from run_nis_reference
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
# Core GT PCC algorithm
# ---------------------------------------------------------------

def gt_pcc_clustering(sim_matrix, gt_labels, threshold):
    """
    Build GT PCC clustering at a given similarity threshold.

    For each pair where cosine_similarity > threshold:
      - If same individual (GT) → positive edge (keep)
      - If different individual → negative edge (discard)
    Connected components of positive edges = predicted clusters.

    Returns:
        pred_labels: np array of cluster assignments
        stats: dict with edge counts and cluster info
    """
    n = len(gt_labels)
    gt_arr = np.asarray(gt_labels)

    # Upper triangle indices (all unique pairs)
    triu_i, triu_j = np.triu_indices(n, k=1)

    # Candidate edges: similarity above threshold
    sims = sim_matrix[triu_i, triu_j]
    candidate_mask = sims > threshold

    # Among candidates, which are same individual?
    same_individual = gt_arr[triu_i] == gt_arr[triu_j]
    positive_mask = candidate_mask & same_individual

    # Count missed positives: same individual but below threshold
    missed_mask = ~candidate_mask & same_individual

    # Build graph from positive edges only
    G = nx.Graph()
    G.add_nodes_from(range(n))
    pos_i = triu_i[positive_mask]
    pos_j = triu_j[positive_mask]
    G.add_edges_from(zip(pos_i.tolist(), pos_j.tolist()))

    # Extract connected components → cluster labels
    pred_labels = np.full(n, -1, dtype=int)
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            pred_labels[node] = cid
    num_clusters = cid + 1 if n > 0 else 0

    stats = {
        'total_pairs': int(len(triu_i)),
        'num_candidate_edges': int(candidate_mask.sum()),
        'num_positive_edges': int(positive_mask.sum()),
        'num_negative_edges': int((candidate_mask & ~same_individual).sum()),
        'num_missed_positive': int(missed_mask.sum()),
        'num_total_positive_pairs': int(same_individual.sum()),
        'num_clusters': num_clusters,
    }

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
        description="GT PCC reference clustering (upper-bound)")
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Single similarity threshold (default: sweep mode)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep thresholds from 0.0 to 0.99')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of threshold steps for sweep (default: 50)')
    parser.add_argument('--max_flips', type=int, default=None,
                        help='Max GT flips budget. Sweeps thresholds and finds '
                             'the best H_F1 with Total_Flips <= budget.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Default to sweep if neither threshold nor sweep nor max_flips specified
    if args.threshold is None and not args.sweep and args.max_flips is None:
        args.sweep = True
    # max_flips implies sweep
    if args.max_flips is not None:
        args.sweep = True

    # Setup logging
    global _log_file
    dataset = args.dataset
    log_dir = f"tmp/{dataset}/output"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gt_pcc_sweep.log")
    _log_file = open(log_path, 'w')

    log(f"{'='*70}")
    log(f"GT PCC Reference Clustering: {dataset}")
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

    if args.sweep:
        # Sweep mode
        thresholds = np.linspace(0.0, 0.99, args.num_steps)
        # Ensure 0.0 is included (all-pairs baseline)
        thresholds = sorted(set([0.0] + list(thresholds)))

        log(f"\n{'='*70}")
        log(f"Threshold sweep: {len(thresholds)} steps")
        log(f"{'='*70}")

        header = (f"{'Threshold':>10s} | {'Pos_Edges':>10s} | {'Neg_Edges':>10s} | {'Missed_Pos':>10s} | {'Total_Flips':>11s} | "
                  f"{'K_pred':>7s} | {'K_true':>7s} | {'K_ratio':>7s} | "
                  f"{'P_F1':>6s} | {'P_Prec':>6s} | {'P_Rec':>6s} | "
                  f"{'H_F1':>6s} | {'H_Prec':>6s} | {'H_Rec':>6s} | "
                  f"{'ARI':>6s}")
        log(f"\n{header}")
        log("-" * len(header))

        best_hf1 = -1
        best_threshold = 0
        best_row = ""

        # Collect all results for budget-constrained analysis
        sweep_results = []

        for thresh in thresholds:
            pred_labels, stats = gt_pcc_clustering(sim_matrix, gt_labels, thresh)
            m = compute_all_metrics(pred_labels, gt_labels, true_K)

            total_flips = stats['num_negative_edges'] + stats['num_missed_positive']
            row = (f"Threshold={thresh:.4f} | "
                   f"Pos_Edges={stats['num_positive_edges']:>10d} | "
                   f"Neg_Edges={stats['num_negative_edges']:>10d} | "
                   f"Missed_Pos={stats['num_missed_positive']:>10d} | "
                   f"Total_Flips={total_flips:>11d} | "
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

            sweep_results.append({
                'threshold': thresh,
                'total_flips': total_flips,
                'h_f1': m['h_f1'],
                'row': row,
            })

            if m['h_f1'] > best_hf1:
                best_hf1 = m['h_f1']
                best_threshold = thresh
                best_row = row

        log(f"\n{'='*70}")
        log(f"BEST RESULT (by Hungarian F1, unconstrained):")
        log(f"  Threshold = {best_threshold:.4f}")
        log(f"  {best_row}")
        log(f"{'='*70}")

        # Budget-constrained analysis
        if args.max_flips is not None:
            budget = args.max_flips
            eligible = [r for r in sweep_results if r['total_flips'] <= budget]
            log(f"\n{'='*70}")
            log(f"BUDGET-CONSTRAINED RESULT (max_flips={budget:,}):")
            if eligible:
                best_budget = max(eligible, key=lambda r: r['h_f1'])
                log(f"  {len(eligible)}/{len(sweep_results)} thresholds within budget")
                log(f"  Best threshold = {best_budget['threshold']:.4f}  (Total_Flips={best_budget['total_flips']:,})")
                log(f"  {best_budget['row']}")
            else:
                # Find the threshold with the minimum flips
                min_flips_r = min(sweep_results, key=lambda r: r['total_flips'])
                log(f"  NO threshold within budget!")
                log(f"  Minimum Total_Flips across all thresholds: {min_flips_r['total_flips']:,} "
                    f"(at threshold={min_flips_r['threshold']:.4f})")
                log(f"  Increase --max_flips to at least {min_flips_r['total_flips']} or use a finer sweep.")
            log(f"{'='*70}")

    else:
        # Single threshold mode
        thresh = args.threshold
        log(f"\nRunning GT PCC at threshold = {thresh:.4f}")

        pred_labels, stats = gt_pcc_clustering(sim_matrix, gt_labels, thresh)
        m = compute_all_metrics(pred_labels, gt_labels, true_K)

        log(f"\n{'='*70}")
        log(f"GT PCC Results (threshold={thresh:.4f})")
        log(f"{'='*70}")
        total_flips = stats['num_negative_edges'] + stats['num_missed_positive']
        log(f"  Candidate edges:       {stats['num_candidate_edges']:,}")
        log(f"  Positive edges (GT):   {stats['num_positive_edges']:,}")
        log(f"  Negative edges (GT):   {stats['num_negative_edges']:,}  (MIEWID said similar, GT says different)")
        log(f"  Missed positive pairs: {stats['num_missed_positive']:,} / {stats['num_total_positive_pairs']:,}  (MIEWID said dissimilar, GT says same)")
        log(f"  Total edge flips:      {total_flips:,}  (corrections needed by GT oracle)")
        log(f"  Positive recall:       {1 - stats['num_missed_positive']/max(1,stats['num_total_positive_pairs']):.4f}")
        log(f"")
        log(f"  Predicted clusters:    {m['num_clusters']}")
        log(f"  True clusters:         {m['true_K']}")
        log(f"  Cluster count ratio:   {m['k_ratio']:.4f}")
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
