import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from sklearn.mixture import GaussianMixture


class SyntheticEmbeddings(object):
    def __init__(self,
                 ids, df, 
                 filter_key,
                 sample_positive,
                 sample_negative):
        self.uuids = ids
        self.ids = list(ids.keys())
        self.labels = [df.loc[df['uuid_x'] == self.uuids[id], filter_key].values[0] for id in self.ids]
        self.generate_positive_score = sample_positive
        self.generate_negative_score = sample_negative

        self.scores = self.generate_scores()

    
    def generate_scores(self):
        self.scores = {}
        for i, id1 in enumerate(self.ids):
            for j, id2 in enumerate(self.ids):
                if i < j:  
                    negative = self.labels[id1] != self.labels[id2]
                    self.scores[(id1, id2)] = -1
                    while self.scores[(id1, id2)] < 0 or self.scores[(id1, id2)] > 1:
                        if negative:
                            self.scores[(id1, id2)] = self.generate_negative_score()
                        else:
                            self.scores[(id1, id2)] = self.generate_positive_score()
                    # if self.scores[(id1, id2)] > 1:
                    #     print(f"{not negative}: {self.scores[(id1, id2)]}")
                elif i == j:
                    self.scores[(id1, id2)] = 1  
        return self.scores
        



    def get_score(self, id1, id2):
        if hasattr(id1, "__len__"):
            id1 = id1[0]
        if hasattr(id2, "__len__"):
            id2 = id2[0]
        if id1 > id2:
            id1, id2 = id2, id1
        return self.scores.get((id1, id2), None)
        

    
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
        # raise NotImplemented()
        distmat = -pairwise_distances(
            np.array(self.ids).reshape(-1, 1), #     
            metric=self.get_score)
        
        labels = [df.loc[df['uuid_x'] == self.uuids[id], filter_key].values[0] for id in self.ids]
        
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1,3,5,10])


        return top1, top3, top5, top10


    def get_top20_matches(self, df, filter_key):
        # raise NotImplemented()
        distmat = -pairwise_distances(
            np.array(self.ids).reshape(-1, 1), #     
            metric=self.get_score)
        labels = [df.loc[df['uuid_x'] == self.uuids[id], filter_key].values[0] for id in self.ids]

        top20_results = {}

        for i, row in enumerate(distmat):
            top20_indices = np.argsort(row)[:20]  # Get the top 20 closest indices
            top20_scores = row[top20_indices]  # Get their corresponding scores

            # Fetch the actual UUIDs corresponding to the indices
            top20_uuids =[self.ids[idx] for idx in top20_indices]

            # Build the dictionary for each UUID with top 20 matches and their scores
            top20_results[self.ids[i]] = [(uuid, 1-score) for uuid, score in zip(top20_uuids, top20_scores)]
        
        # print(f"Finished calculating distances in {time.time() - start_time:.2f} seconds.")
        
        return top20_results


    
    def get_edges(self, topk=5, target_edges=10000, uuids_filter=None):
    
        start_time = time.time()

        # Filter IDs if uuids_filter is provided
        if uuids_filter is not None:
            ids = [id for id in self.ids if self.uuids[id] in uuids_filter]
        else:
            ids = self.ids

        print("Calculating edges using scores...")
        result = []
        num_ids = len(ids)
        total_edges = (num_ids * (num_ids - 1)) / 2
        target_proportion = np.clip(target_edges / total_edges, 0, 1)

    
        for i, id1 in enumerate(ids):
            scores = [(id2, self.scores[(id1, id2)] if id1 < id2 else self.scores[(id2, id1)]) for id2 in ids if id1 != id2]

        
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            topk_scores = scores[:topk]

    
            remaining_edges = int(target_proportion * len(scores))
            additional_edges = np.random.choice(len(scores), remaining_edges, replace=False)
            additional_scores = [scores[idx] for idx in additional_edges]

            combined_scores = set(topk_scores + additional_scores)
            for id2, score in combined_scores:
                result.append((*sorted([id1, id2]), score))

            print(f"Processed {i + 1}/{num_ids} IDs, elapsed time: {time.time() - start_time:.2f} seconds")

        print(f"Generated {len(result)} edges")
        return set(result)

        # raise NotImplemented()

    def get_baseline_edges(self, topk=10, distance_threshold=0.5):
        raise NotImplemented()



