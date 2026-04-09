"""
Spectral analysis of the PCC-level meta-graph after Phase 0.

Builds a k×k similarity matrix W where W_ij = aggregate embedding score
between PCC_i and PCC_j. Then runs spectral clustering on W and compares
to ground truth.

The insight: if two PCCs are sightings of the same animal, their row
vectors in W should be correlated — both score high against other sightings
of that animal and low against everything else — even when W_ij itself
is below the classifier threshold.

Usage:
    python spectral_metagraph.py --config ./configs/beluga/config_beluga_stability.yaml
"""

import argparse
import logging
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from init_logger import init_logger
from tools import get_config
from algorithm_preparation import create_algorithm
from run import get_initial_edges, get_human_responses

logger = logging.getLogger('lca')


def run_phase0(algorithm, common_data, config):
    """Run the algorithm through Phase 0 only, then stop."""
    initial_edges = get_initial_edges(common_data, config)
    requested_edges = algorithm.step(initial_edges)

    max_outer = config.get('algorithm', {}).get('max_outer_iterations', 10000)
    iteration = 0

    while not algorithm.is_finished() and iteration < max_outer:
        if hasattr(algorithm, 'phase') and algorithm.phase != "PHASE0":
            logger.info(f"Phase 0 complete after {iteration} outer iterations. Phase: {algorithm.phase}")
            break

        if requested_edges:
            human_responses, quit = get_human_responses(requested_edges, common_data)
            if quit:
                break
            requested_edges = algorithm.step(human_responses)
        else:
            requested_edges = algorithm.step([])

        iteration += 1

    logger.info(f"Phase 0 finished. PCCs: {len(algorithm.graph.get_pccs())}")
    return algorithm


def build_metagraph(algorithm, common_data):
    """Build the k×k meta-graph similarity matrix."""
    pccs = algorithm.graph.get_pccs()
    k = len(pccs)
    gt_node2cid = common_data['cluster_validator'].gt_node2cid

    # Build node -> PCC index mapping
    node_to_pcc = {}
    for pcc_idx, pcc in enumerate(pccs):
        for node in pcc:
            node_to_pcc[node] = pcc_idx

    # Build GT PCC label: majority GT cluster ID for each PCC
    pcc_gt_label = []
    for pcc in pccs:
        gt_counts = defaultdict(int)
        for node in pcc:
            gt = gt_node2cid.get(node)
            if gt is not None:
                gt_counts[gt] += 1
        if gt_counts:
            pcc_gt_label.append(max(gt_counts, key=gt_counts.get))
        else:
            pcc_gt_label.append(-1)

    # Ground truth: which PCC pairs should merge?
    gt_merge_pairs = set()
    for i in range(k):
        for j in range(i + 1, k):
            if pcc_gt_label[i] == pcc_gt_label[j] and pcc_gt_label[i] != -1:
                gt_merge_pairs.add((i, j))

    # Ensure _all_edges_sorted is computed
    if algorithm._all_edges_sorted is None:
        first_classifier = algorithm.classifier_manager.algo_classifiers[0]
        embeddings, _ = algorithm.classifier_manager.classifier_units[first_classifier]
        algorithm._initialize_sorted_edges(embeddings)

    # Build W: multiple aggregation strategies
    # We'll compute sum and count, then derive mean and max
    pair_sum = np.zeros((k, k))
    pair_count = np.zeros((k, k))
    pair_max = np.full((k, k), -np.inf)

    for n0, n1, score in algorithm._all_edges_sorted:
        p0 = node_to_pcc.get(n0)
        p1 = node_to_pcc.get(n1)
        if p0 is None or p1 is None or p0 == p1:
            continue

        pair_sum[p0, p1] += score
        pair_sum[p1, p0] += score
        pair_count[p0, p1] += 1
        pair_count[p1, p0] += 1
        if score > pair_max[p0, p1]:
            pair_max[p0, p1] = score
            pair_max[p1, p0] = score

    # Mean matrix
    W_mean = np.zeros((k, k))
    nonzero = pair_count > 0
    W_mean[nonzero] = pair_sum[nonzero] / pair_count[nonzero]

    # Max matrix
    W_max = pair_max.copy()
    W_max[W_max == -np.inf] = 0.0

    # Number of GT clusters
    unique_gt = set(pcc_gt_label) - {-1}
    n_gt_clusters = len(unique_gt)

    print(f"\n{'='*60}")
    print(f"Meta-graph: {k} PCCs, {n_gt_clusters} GT clusters")
    print(f"GT merge pairs: {len(gt_merge_pairs)}")
    print(f"W_mean range: [{W_mean[W_mean > 0].min():.4f}, {W_mean.max():.4f}]")
    print(f"W_max range: [{W_max[W_max > 0].min():.4f}, {W_max.max():.4f}]")
    print(f"{'='*60}\n")

    return {
        'W_mean': W_mean,
        'W_max': W_max,
        'pccs': pccs,
        'pcc_gt_label': pcc_gt_label,
        'gt_merge_pairs': gt_merge_pairs,
        'n_gt_clusters': n_gt_clusters,
        'k': k,
        'node_to_pcc': node_to_pcc,
    }


def spectral_analysis(data, output_dir):
    """Run spectral clustering on the meta-graph and evaluate."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    k = data['k']
    pcc_gt_label = data['pcc_gt_label']
    n_gt = data['n_gt_clusters']
    gt_merge_pairs = data['gt_merge_pairs']

    results = {}

    for name, W in [('W_mean', data['W_mean']), ('W_max', data['W_max'])]:
        print(f"\n--- Spectral clustering on {name} ---")

        # Make sure diagonal is 0 for affinity (self-similarity is not informative)
        W_sym = W.copy()
        np.fill_diagonal(W_sym, 0)

        # Shift to non-negative if needed (some scores could be negative)
        if W_sym.min() < 0:
            W_sym = W_sym - W_sym.min()

        # Eigendecomposition of the affinity matrix
        # Use the normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
        D = np.diag(W_sym.sum(axis=1))
        D_inv_sqrt = np.zeros_like(D)
        nonzero = np.diag(D) > 0
        D_inv_sqrt[nonzero, nonzero] = 1.0 / np.sqrt(np.diag(D)[nonzero])
        L_norm = np.eye(k) - D_inv_sqrt @ W_sym @ D_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Sort by eigenvalue (ascending — smallest eigenvalues = clusters)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print(f"  First 20 eigenvalues: {eigenvalues[:20].round(4)}")

        # Eigengap: largest gap suggests number of clusters
        gaps = np.diff(eigenvalues[:min(50, k)])
        best_gap_idx = np.argmax(gaps) + 1  # +1 because gap after eigenvalue i
        print(f"  Largest eigengap at k={best_gap_idx} "
              f"(gap={gaps[best_gap_idx-1]:.4f})")

        # Try spectral clustering with GT number of clusters
        for n_clusters in [n_gt, best_gap_idx]:
            if n_clusters < 2 or n_clusters >= k:
                continue

            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    assign_labels='kmeans',
                    random_state=42,
                    n_init=20,
                )
                labels = sc.fit_predict(W_sym)

                # Compare to GT
                ari = adjusted_rand_score(pcc_gt_label, labels)
                nmi = normalized_mutual_info_score(pcc_gt_label, labels)

                # Count how many GT merge pairs are in the same spectral cluster
                merge_found = 0
                merge_missed = 0
                for i, j in gt_merge_pairs:
                    if labels[i] == labels[j]:
                        merge_found += 1
                    else:
                        merge_missed += 1

                # Count false merges (non-GT pairs in same spectral cluster)
                false_merges = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        if labels[i] == labels[j] and (i, j) not in gt_merge_pairs:
                            false_merges += 1

                n_spectral = len(set(labels))
                print(f"\n  n_clusters={n_clusters} (actual: {n_spectral}):")
                print(f"    ARI={ari:.4f}, NMI={nmi:.4f}")
                print(f"    GT merges found: {merge_found}/{len(gt_merge_pairs)} "
                      f"({100*merge_found/max(1,len(gt_merge_pairs)):.1f}%)")
                print(f"    GT merges missed: {merge_missed}")
                print(f"    False merges: {false_merges}")

                results[f'{name}_k{n_clusters}'] = {
                    'ari': ari, 'nmi': nmi,
                    'merge_found': merge_found,
                    'merge_missed': merge_missed,
                    'false_merges': false_merges,
                    'labels': labels,
                }
            except Exception as e:
                print(f"  n_clusters={n_clusters}: FAILED - {e}")

        # --- Row correlation analysis ---
        # For each GT merge pair, compute cosine similarity of their rows
        # Compare to non-merge pairs
        print(f"\n  Row correlation analysis ({name}):")
        merge_cosines = []
        nonmerge_cosines = []

        norms = np.linalg.norm(W_sym, axis=1, keepdims=True)
        norms[norms == 0] = 1
        W_normed = W_sym / norms

        for i in range(k):
            for j in range(i + 1, k):
                cos = np.dot(W_normed[i], W_normed[j])
                if (i, j) in gt_merge_pairs:
                    merge_cosines.append(cos)
                else:
                    nonmerge_cosines.append(cos)

        merge_cosines = np.array(merge_cosines)
        nonmerge_cosines = np.array(nonmerge_cosines)

        if len(merge_cosines) > 0:
            print(f"    Merge pairs row cosine: "
                  f"mean={merge_cosines.mean():.4f}, "
                  f"std={merge_cosines.std():.4f}, "
                  f"median={np.median(merge_cosines):.4f}")
        if len(nonmerge_cosines) > 0:
            print(f"    Non-merge pairs row cosine: "
                  f"mean={nonmerge_cosines.mean():.4f}, "
                  f"std={nonmerge_cosines.std():.4f}, "
                  f"median={np.median(nonmerge_cosines):.4f}")
        if len(merge_cosines) > 0 and len(nonmerge_cosines) > 0:
            # Separation
            sep = (merge_cosines.mean() - nonmerge_cosines.mean()) / \
                  (nonmerge_cosines.std() + 1e-8)
            print(f"    Separation (merge_mean - nonmerge_mean) / nonmerge_std = {sep:.2f}")

        # Plot histogram of row cosines
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(nonmerge_cosines) > 0:
            ax.hist(nonmerge_cosines, bins=80, alpha=0.5, color='red',
                    label=f'Non-merge ({len(nonmerge_cosines)})', density=True)
        if len(merge_cosines) > 0:
            ax.hist(merge_cosines, bins=40, alpha=0.5, color='green',
                    label=f'Merge ({len(merge_cosines)})', density=True)
        ax.set_xlabel('Row Cosine Similarity', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title(f'PCC Row Correlation ({name}): Merge vs Non-merge', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, f'metagraph_row_cosine_{name}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot: {path}")

    # --- Eigenvalue plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (name, W) in enumerate([('W_mean', data['W_mean']),
                                      ('W_max', data['W_max'])]):
        W_sym = W.copy()
        np.fill_diagonal(W_sym, 0)
        if W_sym.min() < 0:
            W_sym = W_sym - W_sym.min()
        D = np.diag(W_sym.sum(axis=1))
        D_inv_sqrt = np.zeros_like(D)
        nonzero_d = np.diag(D) > 0
        D_inv_sqrt[nonzero_d, nonzero_d] = 1.0 / np.sqrt(np.diag(D)[nonzero_d])
        L_norm = np.eye(k) - D_inv_sqrt @ W_sym @ D_inv_sqrt
        evals = np.linalg.eigvalsh(L_norm)

        ax = axes[idx]
        n_show = min(50, len(evals))
        ax.plot(range(n_show), sorted(evals)[:n_show], 'b.-')
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(f'Laplacian Eigenvalues ({name})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    path = os.path.join(output_dir, 'metagraph_eigenvalues.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved eigenvalue plot: {path}")

    return results


def main():
    init_logger()
    parser = argparse.ArgumentParser(
        description="Spectral analysis of PCC meta-graph after Phase 0")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    args = parser.parse_args()

    config = get_config(args.config)

    # Create algorithm
    algorithm, common_data = create_algorithm(config)

    # Run Phase 0 only
    logger.info("Running Phase 0...")
    algorithm = run_phase0(algorithm, common_data, config)

    # Build meta-graph
    logger.info("Building meta-graph...")
    data = build_metagraph(algorithm, common_data)

    # Spectral analysis
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = spectral_analysis(data, output_dir)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for key, res in results.items():
        print(f"  {key}: ARI={res['ari']:.4f}, NMI={res['nmi']:.4f}, "
              f"merges={res['merge_found']}/{res['merge_found']+res['merge_missed']}, "
              f"false={res['false_merges']}")


if __name__ == '__main__':
    main()
