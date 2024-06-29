import numpy as np
from numpy.linalg import norm 


class Embeddings(object):
    def __init__(self,
                 embeddings,
                 ids):
        self.embeddings = embeddings
        self.ids = ids


    def get_score(self, id1, id2):
        embedding1 = self.embeddings[self.ids.index(id1)]
        embedding2 = self.embeddings[self.ids.index(id2)]
        return max(self.cosine_similarity(embedding1, embedding2), 0)


    def cosine_similarity(self, input1, input2):
        return np.dot(input1, input2)/(norm(input1)*norm(input2))
    
    def get_edges(self, topk=5,distance_threshold = 0.5 ):
        result=[]
        added_pairs = set()
       

        for i, id1 in enumerate(self.ids):
            distances = [1-self.get_score(id1, id2) for id2 in self.ids]
            
            sorted_indices = np.argsort(distances)
            
            # top_5_indices = sorted_indices[1:6]
            
            for j, idx in enumerate(sorted_indices):
                another_id = self.ids[idx]
                distance = distances[idx]
                
                if id1 != another_id and (distance <= distance_threshold or j <= topk):
                    pair = tuple(sorted((id1, another_id)))
                    if pair not in added_pairs:
                        added_pairs.add(pair)
                        result.append([pair[0], pair[1], 1-distance])
        return result


