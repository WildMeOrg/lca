import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
import sklearn


class Embeddings(object):
    def __init__(self,
                 embeddings,
                 ids,
                 distance_power=1):
        self.embeddings = embeddings
        self.ids = ids
        self.distance_power = distance_power


    def get_score(self, id1, id2):
        embedding1 = self.embeddings[self.ids.index(id1)]
        embedding2 = self.embeddings[self.ids.index(id2)]
        return self.get_embeddings_score(embedding1, embedding2)
    
    def get_embeddings_score(self, embedding1, embedding2):
        return self.get_score_from_cosine_distance(cosine(embedding1, embedding2))
    
    def get_score_from_cosine_distance(self, cosine_dist):
        return np.power(1 - cosine_dist*0.5, self.distance_power)


    
    def get_edges(self, topk=5, distance_threshold=0.25):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            # print(np.min(distmat), np.max(distmat))
            # raise Exception("Sorry")
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat
        start_time = time.time()
        print("Calculating distances...")

        chunks = pairwise_distances_chunked(
            self.embeddings, #     self.embeddings/norm(self.embeddings, axis=1).reshape((-1,1)), 
            metric='cosine', #     metric=self.get_norm_embeddings_score,
            reduce_func=reduce_func, n_jobs=-1)#, working_memory=4)
        result = []
        start = 0
        for distmat in chunks:
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk
            thresholded_dists = np.triu(distmat <= distance_threshold, start)
            filtered = np.logical_or(sorted_dists, thresholded_dists)
            inds_y, inds_x = np.nonzero(filtered)
            print(filtered.shape)
            print(len(self.ids))
            result.extend([(*sorted([self.ids[ind1+start], self.ids[ind2]]), 1-distmat[ind1, ind2]) for (ind1, ind2) in zip(inds_y, inds_x)])
            print(f"Chunk result: {time.time() - start_time:.6f} seconds, Total estimate: {len(self.embeddings) * (time.time() - start_time)/(60 * len(distmat)):.6f} minutes")
            start_time = time.time()
            start += filtered.shape[0]
        print(f"Calculated dsitances: {time.time() - start_time:.6f} seconds")
        return set(result)


