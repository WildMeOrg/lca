"""
Standalone reference NIS (Nested Importance Sampling) estimator.

Runs the EXACT algorithm from:
  "Human-in-the-Loop Visual Re-ID for Population Size Estimation"
  (Perez et al., ECCV 2024, arXiv:2312.05287)
  https://github.com/cvl-umass/counting-clusters

This loads our data (pickle embeddings + JSON annotations), applies the
same filtering as the LCA pipeline, then calls nested_is() directly
with ground-truth oracle (no batching, no LCA framework overhead).

Usage:
  python3 run_nis_reference.py --dataset beluga
  python3 run_nis_reference.py --dataset GZCD
  python3 run_nis_reference.py --dataset beluga --runs 20 --N_v 50 --N_n 100

  # Sweep mode: run NIS at multiple budget levels (0 to 5000)
  python3 -u run_nis_reference.py --dataset beluga --sweep
  python3 -u run_nis_reference.py --dataset beluga --sweep --budget_max 5000 --budget_steps 10
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import numpy.core as _nc
# Compat shim: pickles saved with numpy 2.x reference numpy._core
sys.modules.setdefault('numpy._core', _nc)
sys.modules.setdefault('numpy._core.multiarray', _nc.multiarray)
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize


# ---------------------------------------------------------------
# Logging helper: prints to stdout AND writes to log file
# ---------------------------------------------------------------

_log_file = None

def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


# ---------------------------------------------------------------
# Reference NIS implementation (verbatim from Perez et al.)
# with GT query counting added around lookups
# ---------------------------------------------------------------

def nested_is(gt_s_ij, s_ij, N_v, N_n, n_hat=None, ci=False):
    """
    Estimate number of categories using nested importance sampling.
    Exact copy of reference: github.com/cvl-umass/counting-clusters

    Returns additional stats dict with GT query counts.
    """
    OBSERVATIONS = len(gt_s_ij)

    if not n_hat:
        n_hat = []
        for i in range(OBSERVATIONS):
            n_hat.append(np.sum(s_ij[i]))

    # Proposal distribution Q(u) ∝ 1/n_hat(u)
    Q = (1 / np.array(n_hat)) / np.sum((1 / np.array(n_hat)))
    sampled_vertices = list(np.random.choice(
        list(range(OBSERVATIONS)), N_v, p=Q, replace=True))

    sampled_neighbors = []
    q_all = []
    for v_i in sampled_vertices:
        q = (s_ij[v_i]) / np.sum((s_ij[v_i]))
        q_all.append(q)
        sampled_neighbors.append(
            [v_i] + list(np.random.choice(
                list(range(len(s_ij[v_i]))), N_n - 1, p=q, replace=True)))

    # --- Count GT queries ---
    gt_lookups = 0          # total gt_s_ij[v_i][v_j] accesses
    self_lookups = 0        # lookups where v_i == v_j (self-comparison)
    positive_lookups = 0    # lookups where gt_s_ij == 1 (same individual)
    unique_pairs = set()    # unique (v_i, v_j) pairs queried

    # Estimate K
    sum_cc = 0
    for i, v_i in enumerate(sampled_vertices):
        sum_n_bar = 0
        for j, v_j in enumerate(sampled_neighbors[i]):
            gt_val = gt_s_ij[v_i][v_j]
            gt_lookups += 1
            if v_i == v_j:
                self_lookups += 1
            if gt_val == 1.0:
                positive_lookups += 1
            unique_pairs.add((min(v_i, v_j), max(v_i, v_j)))
            sum_n_bar += gt_val / q_all[i][v_j]
        n_bar = sum_n_bar / len(sampled_neighbors[i])
        sum_cc += (1 / n_bar) * (1 / Q[v_i])
    f_hat = sum_cc / len(sampled_vertices)

    stats = {
        'gt_lookups': gt_lookups,
        'self_lookups': self_lookups,
        'non_self_lookups': gt_lookups - self_lookups,
        'positive_lookups': positive_lookups,
        'unique_pairs': len(unique_pairs),
        'unique_vertices': len(set(sampled_vertices)),
    }

    if not ci:
        return f_hat, n_hat, stats
    else:
        # CI recomputes same values — no new GT information used
        w_ci = 0
        for i, v_i in enumerate(sampled_vertices):
            sum_n_bar = 0
            for v_j in sampled_neighbors[i]:
                sum_n_bar += gt_s_ij[v_i][v_j] / q_all[i][v_j]
            n_bar = sum_n_bar / len(sampled_neighbors[i])
            w_ci += ((1 / n_bar) * (1 / Q[v_i]) - f_hat) ** 2
        var_hat = w_ci / N_v
        ci_val = 1.96 * (np.sqrt(var_hat / N_v))
        return f_hat, ci_val, n_hat, stats


# ---------------------------------------------------------------
# Data loading (replicates LCA pipeline filtering)
# ---------------------------------------------------------------

DATASET_CONFIGS = {
    'beluga': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/beluga/annotations_beluga.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/beluga/embeddings_beluga.pickle',
        'name_keys': ['name'],
        'id_key': 'uuid',
        'viewpoint_list': ['up'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
    'GZCD': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/image_data/GZCD/annotations/reid_census_region.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/image_data/GZCD/annotations/pipeline_steps_out/miew_id_step5/miewid_census_region.pickle',
        'name_keys': ['individual_id'],
        'id_key': 'uuid',
        'viewpoint_list': ['right', 'backright', 'downright', 'frontright'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
    'giraffe': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/giraffe/annotations_giraffe.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/giraffe/embeddings_giraffe.pickle',
        'name_keys': ['name'],
        'id_key': 'uuid',
        'viewpoint_list': ['right'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
    'lion': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/lion/annotations_lion.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/lion/embeddings_lion.pickle',
        'name_keys': ['name'],
        'id_key': 'uuid',
        'viewpoint_list': None,
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
    'forestelephants': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/forestelephants/annotations_forestelephants.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/forestelephants/embeddings_forestelephants.pickle',
        'name_keys': ['individual_uuid'],
        'id_key': 'uuid',
        'viewpoint_list': ['right'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'standard',
    },
    'plainszebra': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/plainszebra/annotations_plainszebra.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/plainszebra/embeddings_plainszebra.pickle',
        'name_keys': ['name'],
        'id_key': 'uuid',
        'viewpoint_list': ['left'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
    'whaleshark': {
        'annotation_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/whaleshark/annotations_whaleshark.json',
        'embedding_file': '/fs/ess/PAS2136/ggr_data/kate/data_embeddings/whaleshark/embeddings_whaleshark.pickle',
        'name_keys': ['name'],
        'id_key': 'uuid',
        'viewpoint_list': ['left'],
        'n_filter_min': 1,
        'n_filter_max': 100,
        'format': 'drone',
    },
}


def load_and_filter(dataset_name):
    """Load embeddings and annotations, apply same filtering as LCA pipeline."""
    cfg = DATASET_CONFIGS[dataset_name]

    # Load embeddings pickle: (embedding_array, uuid_list_or_dict)
    with open(cfg['embedding_file'], 'rb') as f:
        embeddings_raw, uuids_raw = pickle.load(f)

    if isinstance(uuids_raw, dict):
        uuid_list = list(uuids_raw.keys())
    else:
        uuid_list = list(uuids_raw)

    # Load annotations JSON
    with open(cfg['annotation_file'], 'r') as f:
        data = json.load(f)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images']).drop_duplicates(subset=['uuid'])

    if cfg['format'] == 'standard':
        dfn = pd.DataFrame(data['individuals'])
        dfc = pd.DataFrame(data['categories'])

    # Merge annotations with images
    if 'image_uuid' in dfa.columns and 'uuid' in dfi.columns:
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid',
                       suffixes=('', '_y'))
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id',
                       suffixes=('', '_y'))

    if cfg['format'] == 'standard':
        df = df.merge(dfn, left_on='individual_uuid', right_on='uuid',
                      suffixes=('', '_y'))
        df = df.merge(dfc, left_on='category_id', right_on='id',
                      suffixes=('', '_y'))

    id_key = cfg['id_key']
    name_keys = cfg['name_keys']
    filter_key = '__'.join(name_keys)

    # Create composite name key
    df[filter_key] = df[name_keys].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Filter 1: only annotations with embeddings
    df = df[df[id_key].isin(uuid_list)]
    log(f"  After UUID filter: {len(df)} annotations")

    # Filter 2: viewpoint
    if cfg['viewpoint_list']:
        for vp_val in [cfg['viewpoint_list']]:
            def matches(value):
                if pd.isna(value):
                    return False
                return str(value) in [str(v) for v in vp_val]
            df = df[df['viewpoint'].apply(matches)]
    log(f"  After viewpoint filter: {len(df)} annotations")

    # Filter 3: min count per individual
    if cfg['n_filter_min']:
        df = df.groupby(filter_key).filter(
            lambda g: len(g) >= cfg['n_filter_min'])
    log(f"  After min filter: {len(df)} annotations")

    # Filter 4: max count per individual
    if cfg['n_filter_max']:
        df = df.groupby(filter_key, as_index=False).apply(
            lambda g: g.sample(frac=1, random_state=0).head(
                cfg['n_filter_max'])).droplevel(level=0)
    log(f"  After max filter: {len(df)} annotations")

    df = df.reset_index(drop=True)

    # Extract filtered embeddings
    filtered_uuids = df[id_key].tolist()
    filtered_embeddings = np.array([
        embeddings_raw[uuid_list.index(uuid)] for uuid in filtered_uuids])

    # Ground truth labels
    le = LabelEncoder()
    gt_labels = le.fit_transform(df[filter_key].values)
    true_K = len(le.classes_)

    log(f"  Dataset: {len(filtered_uuids)} nodes, {true_K} true clusters")
    sizes = np.bincount(gt_labels)
    log(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.1f}, singletons={np.sum(sizes == 1)}")

    return filtered_embeddings, gt_labels, true_K


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def hungarian_metrics(pred_labels, gt_labels):
    """
    Hungarian (optimal assignment) cluster matching.
    Same as cluster_tools.hungarian_cluster_matching:
    1. Build Jaccard similarity matrix between GT and predicted clusters
    2. Run Hungarian algorithm for optimal one-to-one assignment
    3. Count matched/unmatched clusters as TP/FP/FN
    """
    # Build cluster sets
    gt_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    for i, (g, p) in enumerate(zip(gt_labels, pred_labels)):
        gt_clusters[g].add(i)
        pred_clusters[p].add(i)

    gt_ids = list(gt_clusters.keys())
    pred_ids = list(pred_clusters.keys())
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0 or n_pred == 0:
        return {'h_precision': 0, 'h_recall': 0, 'h_f1': 0}

    # Jaccard similarity matrix
    jaccard = np.zeros((n_gt, n_pred))
    for i, gid in enumerate(gt_ids):
        gs = gt_clusters[gid]
        for j, pid in enumerate(pred_ids):
            ps = pred_clusters[pid]
            inter = len(gs & ps)
            union = len(gs | ps)
            jaccard[i, j] = inter / union if union > 0 else 0

    # Hungarian assignment (minimize negative Jaccard)
    row_ind, col_ind = linear_sum_assignment(-jaccard)

    # Count valid matches (Jaccard > 0)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if jaccard[r, c] > 0)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'h_precision': precision, 'h_recall': recall, 'h_f1': f1}


def evaluate_kmeans(embeddings, gt_labels, k, true_K, n_init=10):
    """Run k-means and compute pairwise + Hungarian metrics."""
    k = max(1, min(len(embeddings), round(k)))
    norm_emb = normalize(embeddings, norm='l2')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    pred_labels = kmeans.fit_predict(norm_emb)

    ari = adjusted_rand_score(gt_labels, pred_labels)

    # Pairwise precision/recall/f1
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

    total_pairs = n * (n - 1) // 2
    fp = tp_pred - tp
    fn = tp_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)

    # Hungarian metrics
    h = hungarian_metrics(pred_labels, gt_labels)

    return {
        'k_used': k,
        'ari': ari,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'h_precision': h['h_precision'],
        'h_recall': h['h_recall'],
        'h_f1': h['h_f1'],
        'n_clusters': len(set(pred_labels)),
    }


def find_optimal_ratio(gt_s_synth, s_ij, n_hat, n_nodes, budget, K_synth,
                       n_trials=50):
    """Find optimal N_n/N_v ratio by simulating NIS with synthetic oracle.

    Creates ~20 candidate (N_v, N_n) allocations for the given budget,
    runs NIS n_trials times for each, and returns the ratio that minimizes
    MSE of K_hat vs K_synth.
    """
    # Generate candidate N_v values (log-spaced for wide coverage)
    max_nv = min(budget // 2, n_nodes)
    if max_nv < 2:
        return 7.0  # fallback
    nv_candidates = sorted(set(
        max(2, int(x))
        for x in np.logspace(np.log10(2), np.log10(max_nv), 20)
    ))

    best_mse = float('inf')
    best_ratio = 7.0
    results = []

    for nv in nv_candidates:
        nn = max(2, budget // nv)
        if nn > n_nodes:
            nn = n_nodes
        k_hats = []
        for t in range(n_trials):
            np.random.seed(100000 + t)
            f_hat, _, _, _ = nested_is(
                gt_s_synth, s_ij, nv, nn, n_hat=n_hat, ci=True)
            k_hats.append(f_hat)
        k_arr = np.array(k_hats)
        mse = float(np.mean((k_arr - K_synth) ** 2))
        ratio = nn / max(1, nv)
        results.append((nv, nn, ratio, mse, k_arr.mean(), k_arr.std()))
        if mse < best_mse:
            best_mse = mse
            best_ratio = ratio

    return best_ratio, results


def budget_to_nv_nn(budget, n_nodes, ratio=7.0):
    """Convert total GT lookup budget to (N_v, N_n) pair.

    N_v = sqrt(budget / ratio), N_n = budget / N_v.
    The ratio should be computed by find_optimal_ratio().
    """
    if budget <= 0:
        return 0, 0
    N_v = max(1, int(np.sqrt(budget / ratio)))
    N_v = min(N_v, n_nodes, budget)
    N_n = max(2, budget // max(1, N_v))
    N_n = min(N_n, n_nodes)
    return N_v, N_n


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description='Reference NIS estimator on LCA data')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--N_v', type=int, default=50,
                        help='Number of sampled vertices')
    parser.add_argument('--N_n', type=int, default=100,
                        help='Number of neighbors per vertex (incl. self)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save log output (default: '
                             'tmp/<dataset>/output/nis_reference.log)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep GT budgets from 0 to budget_max')
    parser.add_argument('--budget_max', type=int, default=5000,
                        help='Max total GT lookups for sweep (default: 5000)')
    parser.add_argument('--budget_steps', type=int, default=10,
                        help='Number of evenly-spaced budget levels (default: 10)')
    args = parser.parse_args()

    # Setup log file
    if args.log_file is None:
        log_dir = os.path.join('tmp', args.dataset, 'output')
        os.makedirs(log_dir, exist_ok=True)
        suffix = 'nis_reference_sweep.log' if args.sweep else 'nis_reference.log'
        args.log_file = os.path.join(log_dir, suffix)
    else:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    _log_file = open(args.log_file, 'w')

    log(f"=== Reference NIS on {args.dataset} ===")
    log(f"N_v={args.N_v}, N_n={args.N_n}, runs={args.runs}, seed={args.seed}")
    log(f"Log file: {args.log_file}")
    log()

    # Load data
    log("Loading data...")
    embeddings, gt_labels, true_K = load_and_filter(args.dataset)
    n = len(embeddings)
    log()

    # Adjust N_v, N_n for dataset size
    N_v = min(args.N_v, n)
    N_n = min(args.N_n, n)
    log(f"N_v={N_v}, N_n={N_n}")
    log(f"True K = {true_K}")
    log()

    # Explain GT query budget
    log("--- GT query budget ---")
    log(f"Per run:")
    log(f"  Total GT lookups:     N_v * N_n = {N_v} * {N_n} = {N_v * N_n}")
    log(f"  Self lookups (j=0):   N_v * 1   = {N_v}  (always gt=1, answer is known)")
    log(f"  Non-self lookups:     N_v * (N_n-1) = {N_v} * {N_n - 1} = {N_v * (N_n - 1)}")
    log(f"  (non-self may include duplicates from with-replacement sampling)")
    log()

    # Build similarity matrix
    log("Computing similarity matrix...")
    t0 = time.time()
    s_ij = cosine_similarity(embeddings)
    s_ij[s_ij < 0] = 0
    log(f"  s_ij computed in {time.time() - t0:.1f}s, shape={s_ij.shape}")

    # Build ground truth similarity matrix
    log("Building ground truth matrix...")
    t0 = time.time()
    gt_labels_arr = np.array(gt_labels)
    gt_s_ij = (gt_labels_arr[:, None] == gt_labels_arr[None, :]).astype(np.float64)
    log(f"  gt_s_ij computed in {time.time() - t0:.1f}s")

    # Precompute n_hat and K_hat_0 (zero-review estimate)
    n_hat = list(np.sum(s_ij, axis=1))
    n_hat_arr = np.array(n_hat)
    K_hat_0 = float(np.sum(1.0 / n_hat_arr))
    K_hat_0 = max(1.0, min(float(n), K_hat_0))
    log(f"  K_hat_0 = {K_hat_0:.2f} (from embeddings, 0 reviews)")

    # Find optimal N_n/N_v ratio via synthetic oracle simulation
    log("\nFinding optimal ratio via synthetic oracle simulation...")
    t0 = time.time()
    K_synth = max(1, round(K_hat_0))
    synth_labels = KMeans(
        n_clusters=K_synth, random_state=42, n_init=10
    ).fit_predict(normalize(embeddings.reshape(n, -1), norm='l2'))
    gt_s_synth = (synth_labels[:, None] == synth_labels[None, :]).astype(
        np.float64)
    K_synth_actual = len(set(synth_labels))
    budget_ref = args.budget_max if args.sweep else N_v * N_n
    optimal_ratio, ratio_results = find_optimal_ratio(
        gt_s_synth, s_ij, n_hat, n, budget_ref, K_synth_actual)
    log(f"  Synthetic K = {K_synth_actual} (k-means with k={K_synth})")
    log(f"  Reference budget = {budget_ref}")
    log(f"  Optimal ratio = {optimal_ratio:.2f} "
        f"(computed in {time.time() - t0:.1f}s)")
    log(f"  {'N_v':>5} | {'N_n':>5} | {'ratio':>6} | {'MSE':>10} | "
        f"{'K_hat mean':>10} | {'K_hat std':>10}")
    for nv, nn, r, mse, mean, std in ratio_results:
        marker = " <-- best" if abs(r - optimal_ratio) < 0.01 else ""
        log(f"  {nv:>5} | {nn:>5} | {r:>6.2f} | {mse:>10.1f} | "
            f"{mean:>10.2f} | {std:>10.2f}{marker}")

    # ---------------------------------------------------------------
    # Sweep mode: run NIS at multiple budget levels
    # ---------------------------------------------------------------
    if args.sweep:
        budgets = np.linspace(0, args.budget_max,
                              args.budget_steps + 1).astype(int).tolist()
        budgets = sorted(set(budgets))

        log(f"\n=== Budget sweep: {len(budgets)} levels, "
            f"{args.runs} runs each ===")
        log(f"Budgets: {budgets}")
        log()

        def pm(a):
            """Format mean±std."""
            return f"{a.mean():.3f}±{a.std():.3f}"

        # Table header
        log(f"{'Budget':>6} | {'N_v':>4} | {'N_n':>4} | {'Non-self':>8} | "
            f"{'K_hat':>14} | {'K/K_true':>8} | "
            f"{'F1':>14} | {'ARI':>14} | "
            f"{'H_F1':>14} | {'H_Prec':>14} | {'H_Rec':>14}")
        log("-" * 140)

        for budget in budgets:
            if budget == 0:
                # Singleton baseline: each node is its own cluster
                pred = np.arange(n)
                ari_val = adjusted_rand_score(gt_labels, pred)
                h = hungarian_metrics(pred, gt_labels)
                log(f"{0:>6} | {'--':>4} | {'--':>4} | {0:>8} | "
                    f"{'singletons':>14} | {n / true_K:>8.3f} | "
                    f"{'0.000':>14} | {ari_val:>14.3f} | "
                    f"{h['h_f1']:>14.3f} | {h['h_precision']:>14.3f} | "
                    f"{h['h_recall']:>14.3f}")
                continue

            N_v_b, N_n_b = budget_to_nv_nn(budget, n, ratio=optimal_ratio)
            actual = N_v_b * N_n_b
            non_self = N_v_b * (N_n_b - 1)

            k_hats_b, metrics_b = [], []
            for run_i in range(args.runs):
                np.random.seed(args.seed + run_i)
                f_hat, ci_val, _, _ = nested_is(
                    gt_s_ij, s_ij, N_v_b, N_n_b, n_hat=n_hat, ci=True)
                k_hats_b.append(f_hat)
                m = evaluate_kmeans(
                    embeddings, gt_labels, f_hat, true_K, n_init=3)
                metrics_b.append(m)

            k_arr = np.array(k_hats_b)
            f1s = np.array([m['f1'] for m in metrics_b])
            aris = np.array([m['ari'] for m in metrics_b])
            hf1 = np.array([m['h_f1'] for m in metrics_b])
            hp = np.array([m['h_precision'] for m in metrics_b])
            hr = np.array([m['h_recall'] for m in metrics_b])

            log(f"{actual:>6} | {N_v_b:>4} | {N_n_b:>4} | {non_self:>8} | "
                f"{k_arr.mean():>7.1f}±{k_arr.std():<5.1f} | "
                f"{k_arr.mean() / true_K:>8.3f} | "
                f"{pm(f1s):>14} | {pm(aris):>14} | "
                f"{pm(hf1):>14} | {pm(hp):>14} | {pm(hr):>14}")

        log("-" * 140)

        # Oracle baseline
        log(f"\n--- Oracle baseline: k-means with true K={true_K} ---")
        np.random.seed(args.seed)
        oracle = evaluate_kmeans(embeddings, gt_labels, true_K, true_K)
        log(f"  Pairwise:   F1={oracle['f1']:.3f}  "
            f"Prec={oracle['precision']:.3f}  "
            f"Rec={oracle['recall']:.3f}  "
            f"ARI={oracle['ari']:.3f}")
        log(f"  Hungarian:  F1={oracle['h_f1']:.3f}  "
            f"Prec={oracle['h_precision']:.3f}  "
            f"Rec={oracle['h_recall']:.3f}")

        log(f"\nLog saved to: {args.log_file}")
        _log_file.close()
        return

    # Run (single-budget mode)
    log(f"\nRunning {args.runs} independent NIS estimations...")
    log("-" * 130)
    log(f"{'Run':>4} | {'K_hat':>8} | {'CI':>8} | {'k_used':>6} | "
        f"{'ARI':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | "
        f"{'H_Prec':>6} | {'H_Rec':>6} | {'H_F1':>6} | "
        f"{'GT_look':>7} | {'Non-self':>8} | {'Uniq':>5} | {'Pos':>5}")
    log("-" * 130)

    k_hats = []
    cis = []
    metrics_all = []
    stats_all = []

    for run in range(args.runs):
        np.random.seed(args.seed + run)

        f_hat, ci_val, _, stats = nested_is(
            gt_s_ij, s_ij, N_v, N_n, n_hat=n_hat, ci=True)

        k_hats.append(f_hat)
        cis.append(ci_val)
        stats_all.append(stats)

        metrics = evaluate_kmeans(embeddings, gt_labels, f_hat, true_K, n_init=3)
        metrics_all.append(metrics)

        log(f"{run + 1:>4} | {f_hat:>8.1f} | {ci_val:>8.1f} | "
            f"{metrics['k_used']:>6} | {metrics['ari']:>6.3f} | "
            f"{metrics['precision']:>6.3f} | {metrics['recall']:>6.3f} | "
            f"{metrics['f1']:>6.3f} | "
            f"{metrics['h_precision']:>6.3f} | {metrics['h_recall']:>6.3f} | "
            f"{metrics['h_f1']:>6.3f} | "
            f"{stats['gt_lookups']:>7} | "
            f"{stats['non_self_lookups']:>8} | "
            f"{stats['unique_pairs']:>5} | "
            f"{stats['positive_lookups']:>5}")

    # Summary
    log("-" * 130)
    k_hats = np.array(k_hats)
    cis = np.array(cis)
    f1s = np.array([m['f1'] for m in metrics_all])
    aris = np.array([m['ari'] for m in metrics_all])
    precs = np.array([m['precision'] for m in metrics_all])
    recs = np.array([m['recall'] for m in metrics_all])
    h_f1s = np.array([m['h_f1'] for m in metrics_all])
    h_precs = np.array([m['h_precision'] for m in metrics_all])
    h_recs = np.array([m['h_recall'] for m in metrics_all])

    log(f"\nSummary over {args.runs} runs:")
    log(f"  True K:     {true_K}")
    log(f"  K_hat:      {k_hats.mean():.1f} +/- {k_hats.std():.1f} "
        f"(range: [{k_hats.min():.1f}, {k_hats.max():.1f}])")
    log(f"  CI (mean):  +/- {cis.mean():.1f}")
    log(f"  K_hat/K:    {k_hats.mean() / true_K:.3f}")
    log(f"  Pairwise metrics:")
    log(f"    F1:         {f1s.mean():.3f} +/- {f1s.std():.3f}")
    log(f"    ARI:        {aris.mean():.3f} +/- {aris.std():.3f}")
    log(f"    Precision:  {precs.mean():.3f} +/- {precs.std():.3f}")
    log(f"    Recall:     {recs.mean():.3f} +/- {recs.std():.3f}")
    log(f"  Hungarian metrics:")
    log(f"    H_F1:       {h_f1s.mean():.3f} +/- {h_f1s.std():.3f}")
    log(f"    H_Prec:     {h_precs.mean():.3f} +/- {h_precs.std():.3f}")
    log(f"    H_Recall:   {h_recs.mean():.3f} +/- {h_recs.std():.3f}")

    # GT usage summary
    log()
    log("--- GT usage summary (per run) ---")
    avg_stats = {k: np.mean([s[k] for s in stats_all]) for k in stats_all[0]}
    log(f"  Total GT lookups:      {avg_stats['gt_lookups']:.0f}")
    log(f"    Self (v_i == v_j):   {avg_stats['self_lookups']:.0f}  "
        f"(always gt=1, trivial)")
    log(f"    Non-self:            {avg_stats['non_self_lookups']:.0f}  "
        f"(actual 'human questions')")
    log(f"  Unique (i,j) pairs:    {avg_stats['unique_pairs']:.0f}  "
        f"(deduplicated, both directions)")
    log(f"  Positive (gt=1):       {avg_stats['positive_lookups']:.0f}  "
        f"({100*avg_stats['positive_lookups']/avg_stats['gt_lookups']:.1f}% of lookups)")
    log(f"  Unique vertices:       {avg_stats['unique_vertices']:.0f}  "
        f"(out of {N_v} sampled, with replacement)")
    log()
    log(f"  In the reference code, each gt_s_ij[v_i][v_j] lookup = 1 oracle query.")
    log(f"  The algorithm queries GT exactly N_v * N_n = {N_v * N_n} times per run.")
    log(f"  Of these, N_v = {N_v} are self-comparisons (j=0, answer known a priori).")
    log(f"  The remaining {N_v * (N_n - 1)} are the 'human review' equivalent.")

    # Oracle baseline
    log(f"\n--- Oracle baseline: k-means with true K={true_K} ---")
    np.random.seed(args.seed)
    oracle_metrics = evaluate_kmeans(embeddings, gt_labels, true_K, true_K)
    log(f"  Pairwise:   F1={oracle_metrics['f1']:.3f}  "
        f"Prec={oracle_metrics['precision']:.3f}  "
        f"Rec={oracle_metrics['recall']:.3f}  "
        f"ARI={oracle_metrics['ari']:.3f}")
    log(f"  Hungarian:  F1={oracle_metrics['h_f1']:.3f}  "
        f"Prec={oracle_metrics['h_precision']:.3f}  "
        f"Rec={oracle_metrics['h_recall']:.3f}")

    log(f"\nLog saved to: {args.log_file}")
    _log_file.close()


if __name__ == '__main__':
    main()
