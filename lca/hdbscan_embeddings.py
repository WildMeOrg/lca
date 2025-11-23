import numpy as np
import hdbscan
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_chunked
import time
import functools
from tools import kth_diag_indices
import logging

logger = logging.getLogger('lca')


class HDBSCANEmbeddings(object):
    """
    Embeddings class that uses HDBSCAN clustering probabilities for scoring.
    Follows the same interface as Embeddings class but scores are based on
    HDBSCAN's cluster membership probabilities instead of raw distances.
    """

    def __init__(self, embeddings, ids, distance_power=1, print_func=print, min_cluster_size=2, metric='euclidean'):
        self.embeddings = np.array(embeddings)
        self.uuids = ids
        self.ids = list(ids.keys())
        self.distance_power = distance_power
        self.print_func = print_func
        self.min_cluster_size = min_cluster_size
        self.metric = metric

        # Train HDBSCAN clusterer with prediction_data=True to enable soft clustering
        self.print_func(f"Training HDBSCAN with min_cluster_size={self.min_cluster_size}, metric={self.metric}")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            cluster_selection_method='eom',
            prediction_data=True,
            gen_min_span_tree=True,
            alpha=1.0
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        self.print_func(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Precompute soft cluster memberships for all points
        self._precompute_memberships()

    def _precompute_memberships(self):
        """Precompute membership vectors for all points."""
        self.membership_vectors = []
        if hasattr(self.clusterer, 'prediction_data_'):
            try:
                for i in range(len(self.embeddings)):
                    membership = hdbscan.membership_vector(self.clusterer, self.embeddings[i:i+1])[0]
                    self.membership_vectors.append(membership)
            except:
                # Fall back to hard clustering
                n_clusters = len(set(self.labels[self.labels != -1]))
                for i in range(len(self.embeddings)):
                    membership = np.zeros(n_clusters)
                    if self.labels[i] != -1:
                        membership[self.labels[i]] = 1.0
                    self.membership_vectors.append(membership)
        else:
            # Hard clustering only
            n_clusters = len(set(self.labels[self.labels != -1]))
            for i in range(len(self.embeddings)):
                membership = np.zeros(n_clusters)
                if self.labels[i] != -1:
                    membership[self.labels[i]] = 1.0
                self.membership_vectors.append(membership)

    def _reduce_func(self, distmat, start):
        """Convert distance matrix to HDBSCAN probability scores."""
        # Convert distances to membership probability scores
        scores = np.zeros_like(distmat)

        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                node_i = i + start
                node_j = j

                if node_i < len(self.membership_vectors) and node_j < len(self.membership_vectors):
                    # Calculate membership overlap
                    overlap = np.dot(self.membership_vectors[node_i], self.membership_vectors[node_j])
                    scores[i, j] = 1 - overlap  # Convert to distance (0 = same cluster, 1 = different)
                else:
                    scores[i, j] = 1.0  # Maximum distance for out-of-bounds

        # Set diagonal to infinity
        rng = np.arange(min(scores.shape[0], scores.shape[1] - start))
        if len(rng) > 0 and start < scores.shape[1]:
            scores[rng, rng + start] = np.inf

        return scores

    @functools.cache
    def _calculate_distance_matrix(self, flags=None):
        """Calculate distance matrix using HDBSCAN membership probabilities."""
        if flags is None:
            embeddings = self.embeddings
            ids = self.ids
        else:
            embeddings = [self.embeddings[i] for (i, f) in enumerate(flags) if f]
            ids = [self.ids[i] for (i, f) in enumerate(flags) if f]

        # Use pairwise_distances_chunked with our custom reduce function
        chunks = pairwise_distances_chunked(
            embeddings,
            metric=self.metric,
            reduce_func=self._reduce_func,
            n_jobs=-1
        )
        return list(chunks), ids

    def get_score(self, id1, id2):
        """Get HDBSCAN membership probability score between two nodes."""
        idx1 = self.ids.index(id1)
        idx2 = self.ids.index(id2)

        if idx1 < len(self.membership_vectors) and idx2 < len(self.membership_vectors):
            # Calculate membership overlap as score
            overlap = np.dot(self.membership_vectors[idx1], self.membership_vectors[idx2])
            return float(overlap)

        # Check if same cluster (hard clustering fallback)
        if self.labels[idx1] == self.labels[idx2] and self.labels[idx1] != -1:
            return 1.0
        else:
            return 0.0

    def get_embeddings_score(self, embedding1, embedding2):
        """Get score between two embeddings (requires finding their indices)."""
        # This is less efficient - try to use get_score with IDs instead
        idx1 = None
        idx2 = None

        for i, emb in enumerate(self.embeddings):
            if np.allclose(emb, embedding1):
                idx1 = i
            if np.allclose(emb, embedding2):
                idx2 = i
            if idx1 is not None and idx2 is not None:
                break

        if idx1 is not None and idx2 is not None:
            overlap = np.dot(self.membership_vectors[idx1], self.membership_vectors[idx2])
            return float(overlap)

        return 0.0

    def sign_power(self, x, power):
        return np.sign(x) * np.power(np.abs(x), power)

    def get_score_from_cosine_distance(self, cosine_dist):
        """Convert cosine distance to score (kept for compatibility)."""
        return 1 - self.sign_power(cosine_dist, self.distance_power) * 0.5

    def get_topk_acc(self, labels_q, labels_db, dists, topk):
        return sum(self.get_topk_hits(labels_q, labels_db, dists, topk)) / len(labels_q)

    def get_topk_hits(self, labels_q, labels_db, dists, topk):
        indices = np.argsort(dists, axis=1)
        top_labels = np.array(labels_db)[indices[:, :topk]]
        hits = (top_labels.T == labels_q).T
        return np.sum(hits[:, :topk+1], axis=1) > 0

    def get_top_ks(self, q_pids, distmat, ks=[1, 3, 5, 10]):
        return [(k, self.get_topk_acc(q_pids, q_pids, distmat, k)) for k in ks]

    def get_stats(self, df, filter_key, id_key='uuid'):
        start_time = time.time()
        self.print_func("Calculating distances...")
        self.print_func(f"{len(self.embeddings)}/{len(self.ids)}")

        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)

        labels = [df.loc[df[id_key] == self.uuids[id], filter_key].values[0] for id in ids]
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1, 3, 5, 10])

        return top1, top3, top5, top10

    def get_all_scores(self):
        """Get all pairwise HDBSCAN membership probability scores."""
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=1)
        scores = [1-distmat[y, x] for y,x in zip(all_inds_y, all_inds_x)]
        return scores

    def get_distance_matrix(self):
        chunks, ids = self.get_distmat_chunks()
        distmat = np.stack(list(chunks), axis=0)
        return distmat

    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]

    def get_distmat_chunks(self, uuids_filter=None):
        if uuids_filter is not None:
            flags = tuple(self.uuids[id] in uuids_filter for id in self.ids)
        else:
            flags = None

        return self._calculate_distance_matrix(flags)

    def get_edges(self, topk=5, target_edges=10000, target_proportion=None, uuids_filter=None):
        """Get edges based on HDBSCAN cluster membership probabilities."""
        self.print_func("Calculating HDBSCAN-based edges...")
        start_time = time.time()
        chunks, ids = self.get_distmat_chunks(uuids_filter=uuids_filter)
        self.print_func(f"Calculate chunks time: {time.time() - start_time:.6f} seconds")
        start_time = time.time()

        result = []
        start = 0
        embeds_num = len(ids)
        total_edges = (embeds_num * embeds_num - embeds_num) / 2
        if target_proportion is None:
            target_proportion = np.clip(target_edges / max(1, total_edges), 0, 1)
        else:
            target_edges = int(total_edges * target_proportion)

        self.print_func(f"Target: {target_edges}/{total_edges}")

        for distmat in chunks:
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk

            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)

            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]

            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            diag_y, diag_x = kth_diag_indices(distmat, start)

            filtered = np.logical_or(sorted_dists, selected_dists)
            filtered[diag_y, diag_x] = False
            inds_y, inds_x = np.nonzero(filtered)

            result.extend([
                (*sorted([ids[ind1+start], ids[ind2]]), 1-distmat[ind1, ind2])
                for (ind1, ind2) in zip(inds_y, inds_x)
            ])

            self.print_func(f"Chunk result: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += filtered.shape[0]

        self.print_func(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        self.print_func(f"{len(set(result))}")
        return set(result)