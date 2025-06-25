import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
import sklearn


class Embeddings(object):
    def __init__(self,
                 embeddings,
                 ids,
                 distance_power=1):
        self.embeddings = embeddings
        self.uuids = ids
        self.ids = list(ids.keys())
        self.distance_power = distance_power


    def get_score(self, id1, id2):
        embedding1 = self.embeddings[self.ids.index(id1)]
        embedding2 = self.embeddings[self.ids.index(id2)]
        return self.get_embeddings_score(embedding1, embedding2)
    
    def get_embeddings_score(self, embedding1, embedding2):
        return self.get_score_from_cosine_distance(cosine(embedding1, embedding2))
    
    def sign_power(self, x, power):
        return np.sign(x) * np.power(np.abs(x), power)
    
    def get_score_from_cosine_distance(self, cosine_dist):
        return 1 - self.sign_power(cosine_dist, self.distance_power)*0.5
        # return np.power(np.maximum(0, 1 - cosine_dist), self.distance_power

    def get_topk_acc(self, labels_q, labels_db, dists, topk):
        return sum(self.get_topk_hits(labels_q, labels_db, dists, topk)) / len(labels_q)


    def get_topk_hits(self,labels_q, labels_db, dists, topk):
        indices = np.argsort(dists, axis=1)
        top_labels = np.array(labels_db)[indices[:, :topk]]
        hits = (top_labels.T == labels_q).T
        return np.sum(hits[:, :topk+1], axis=1) > 0

    def get_top_ks(self,q_pids, distmat, ks=[1,3,5,10]):
        return [(k, self.get_topk_acc(q_pids, q_pids, distmat, k)) for k in ks]


    def get_stats(self, df, filter_key):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            # print(np.min(distmat), np.max(distmat))
            # raise Exception("Sorry")
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat
        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        chunks = pairwise_distances_chunked(
            self.embeddings, #     self.embeddings/norm(self.embeddings, axis=1).reshape((-1,1)), 
            metric='cosine', #     metric=self.get_norm_embeddings_score,
            reduce_func=reduce_func, n_jobs=-1)#, working_memory=4)
        
        distmat = np.concatenate(list(chunks), axis=0)
        labels = [df.loc[df['uuid_x'] == self.uuids[id], filter_key].values[0] for id in self.ids]
        
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1,3,5,10])


        return top1, top3, top5, top10


    def get_top20_matches(self, df, filter_key):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            # print(np.min(distmat), np.max(distmat))
            # raise Exception("Sorry")
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat

        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        chunks = pairwise_distances_chunked(
            self.embeddings, #     self.embeddings/norm(self.embeddings, axis=1).reshape((-1,1)), 
            metric='cosine', #     metric=self.get_norm_embeddings_score,
            reduce_func=reduce_func, n_jobs=-1)#, working_memory=4)
        
        distmat = np.concatenate(list(chunks), axis=0)
        labels = [df.loc[df['uuid_x'] == self.uuids[id], filter_key].values[0] for id in self.ids]

        top20_results = {}

        for i, row in enumerate(distmat):
            top20_indices = np.argsort(row)[:20]  # Get the top 20 closest indices
            top20_scores = row[top20_indices]  # Get their corresponding scores

            # Fetch the actual UUIDs corresponding to the indices
            top20_uuids =[self.ids[idx] for idx in top20_indices]

            # Build the dictionary for each UUID with top 20 matches and their scores
            top20_results[self.ids[i]] = [(uuid, 1-score) for uuid, score in zip(top20_uuids, top20_scores)]
        
        print(f"Finished calculating distances in {time.time() - start_time:.2f} seconds.")
        
        return top20_results

    def get_distance_matrix(self):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            # print(np.min(distmat), np.max(distmat))
            # raise Exception("Sorry")
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat
        
        embeddings = self.embeddings
        ids = self.ids

        print("Calculating distances...")
        # print(f"{len(self.embeddings)}/{len(self.ids)}")
        chunks = pairwise_distances_chunked(
            embeddings, #     self.embeddings/norm(self.embeddings, axis=1).reshape((-1,1)), 
            metric='cosine', #     metric=self.get_norm_embeddings_score,
            reduce_func=reduce_func, n_jobs=-1)#, working_memory=4)
        distmat = np.stack(list(chunks), axis=0)
        return distmat
    
    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]
    
    def get_edges(self, topk=5, target_edges=10000, uuids_filter=None):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            # print(np.min(distmat), np.max(distmat))
            # raise Exception("Sorry")
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat
        start_time = time.time()


        if uuids_filter is not None:
            embeddings = [emb for emb, id in zip(self.embeddings, self.uuids.keys()) if self.uuids[id] in uuids_filter]
            ids = [id for id in self.uuids.keys() if self.uuids[id] in uuids_filter]
        else:
            embeddings = self.embeddings
            ids = self.ids



        print("Calculating distances...")
        # print(f"{len(self.embeddings)}/{len(self.ids)}")
        chunks = pairwise_distances_chunked(
            embeddings, #     self.embeddings/norm(self.embeddings, axis=1).reshape((-1,1)), 
            metric='cosine', #     metric=self.get_norm_embeddings_score,
            reduce_func=reduce_func, n_jobs=-1)#, working_memory=4)
        result = []
        start = 0
        embeds_num = len(embeddings)
        total_edges = (embeds_num * embeds_num - embeds_num)/2
        target_proportion = np.clip(target_edges/total_edges, 0, 1)
        print(f"Target: {target_edges}/{total_edges}")
        for distmat in chunks:
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk
            # thresholded_dists = np.triu(distmat <= np.quantile(distmat.flatten(), distance_threshold), start)
            # thresholded_dists = np.triu(distmat <= distance_threshold, start)
            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)
            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len*target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]
            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            filtered = np.logical_or(sorted_dists, selected_dists)
            inds_y, inds_x = np.nonzero(filtered)
            result.extend([(*sorted([ids[ind1+start], ids[ind2]]), 1-distmat[ind1, ind2]) for (ind1, ind2) in zip(inds_y, inds_x)])
            print(f"Chunk result: {time.time() - start_time:.6f} seconds, Total estimate: {len(embeddings) * (time.time() - start_time)/(60 * len(distmat)):.6f} minutes")
            start_time = time.time()
            start += filtered.shape[0]
            # print(f"Max dist: {np.max(distmat[np.where(distmat != np.inf)])}")
        print(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        print(f"{len(set(result))}")
        return set(result)


    def get_baseline_edges(self, topk=10, distance_threshold=0.5):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat

        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks = pairwise_distances_chunked(
            self.embeddings,
            metric='cosine',
            reduce_func=reduce_func,
            n_jobs=-1
        )
        
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
            result.extend([(*sorted([self.ids[ind1 + start], self.ids[ind2]]), 1 - distmat[ind1, ind2]) for (ind1, ind2) in zip(inds_y, inds_x)])
            
            print(f"Chunk processed in: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += sorted_dists_mask.shape[0]
        
        print(f"Calculated distances in: {time.time() - start_time:.6f} seconds")
        
        return set(result)



