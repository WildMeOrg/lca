import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import functools

class Embeddings(object):
    def __init__(self, embeddings, ids, distance_power=1):
        self.embeddings = embeddings
        self.uuids = ids
        self.ids = list(ids.keys())
        self.distance_power = distance_power

    def _reduce_func(self, distmat, start):
        """Common distance matrix reduction function."""
        distmat = 1 - self.get_score_from_cosine_distance(distmat)
        rng = np.arange(distmat.shape[0])
        distmat[rng, rng + start] = np.inf
        return distmat
    
    
    def _calculate_distance_matrix(self, embeddings=None, ids=None):
        """Calculate distance matrix using chunked computation."""
        if embeddings is None:
            embeddings = self.embeddings
        if ids is None:
            ids = self.ids
            
        chunks = pairwise_distances_chunked(
            embeddings,
            metric='cosine',
            reduce_func=self._reduce_func,
            n_jobs=-1
        )
        return chunks, ids

    def get_score(self, id1, id2):
        embedding1 = self.embeddings[self.ids.index(id1)]
        embedding2 = self.embeddings[self.ids.index(id2)]
        return self.get_embeddings_score(embedding1, embedding2)
    
    def get_embeddings_score(self, embedding1, embedding2):
        return self.get_score_from_cosine_distance(cosine(embedding1, embedding2))
    
    def sign_power(self, x, power):
        return np.sign(x) * np.power(np.abs(x), power)
    
    def get_score_from_cosine_distance(self, cosine_dist):
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
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        
        labels = [df.loc[df[id_key] == self.uuids[id], filter_key].values[0] for id in ids]
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1, 3, 5, 10])
        
        return top1, top3, top5, top10
    
    @functools.cache
    def get_all_scores(self):
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=1)
        scores = [1-distmat[y, x] for y,x in zip(all_inds_y, all_inds_x)]
        return scores

    def get_top20_matches(self, df, filter_key):
        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        
        top20_results = {}
        for i, row in enumerate(distmat):
            top20_indices = np.argsort(row)[:20]
            top20_scores = row[top20_indices]
            top20_uuids = [ids[idx] for idx in top20_indices]
            top20_results[ids[i]] = [(uuid, 1-score) for uuid, score in zip(top20_uuids, top20_scores)]
        
        print(f"Finished calculating distances in {time.time() - start_time:.2f} seconds.")
        return top20_results

    def get_distance_matrix(self):
        chunks, ids = self.get_distmat_chunks()
        distmat = np.stack(list(chunks), axis=0)
        return distmat
    
    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]
    
    def get_distmat_chunks(self, uuids_filter=None):
        if uuids_filter is not None:
            embeddings = [emb for emb, id in zip(self.embeddings, self.ids) if self.uuids[id] in uuids_filter]
            ids = [id for id in self.ids if self.uuids[id] in uuids_filter]
        else:
            embeddings = self.embeddings
            ids = self.ids
            
        return self._calculate_distance_matrix(embeddings, ids)

    def get_edges(self, topk=5, target_edges=10000, target_proportion=None, uuids_filter=None):
        start_time = time.time()
        print("Calculating distances...")
        
        chunks, ids = self.get_distmat_chunks(uuids_filter=uuids_filter)
        result = []
        start = 0
        
        if target_proportion is None:
            embeds_num = len(ids)
            total_edges = (embeds_num * embeds_num - embeds_num) / 2
            target_proportion = np.clip(target_edges / total_edges, 0, 1)
        else:
            target_edges = int(total_edges * target_proportion)
        
        print(f"Target: {target_edges}/{total_edges}")
        
        for distmat in chunks:
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk
            
            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)
            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]
            
            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            filtered = np.logical_or(sorted_dists, selected_dists)
            inds_y, inds_x = np.nonzero(filtered)
            
            result.extend([
                (*sorted([ids[ind1+start], ids[ind2]]), 1-distmat[ind1, ind2]) 
                for (ind1, ind2) in zip(inds_y, inds_x)
            ])
            
            print(f"Chunk result: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += filtered.shape[0]
            
        print(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        print(f"{len(set(result))}")
        return set(result)

    def get_baseline_edges(self, topk=10, distance_threshold=0.5):
        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks, ids = self.get_distmat_chunks()
        result = []
        start = 0
        
        for distmat in chunks:
            sorted_indices = np.argsort(distmat, axis=1)
            
            sorted_dists_mask = np.zeros_like(distmat, dtype=bool)
            for i in range(distmat.shape[0]):
                topk_indices = sorted_indices[i, :topk]
                valid_topk_indices = topk_indices[distmat[i, topk_indices] <= distance_threshold]
                sorted_dists_mask[i, valid_topk_indices] = True
            
            inds_y, inds_x = np.nonzero(sorted_dists_mask)
            result.extend([
                (*sorted([ids[ind1 + start], ids[ind2]]), 1 - distmat[ind1, ind2]) 
                for (ind1, ind2) in zip(inds_y, inds_x)
            ])
            
            print(f"Chunk processed in: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += sorted_dists_mask.shape[0]
        
        print(f"Calculated distances in: {time.time() - start_time:.6f} seconds")
        return set(result)