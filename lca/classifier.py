
from classifier_base import ClassifierBase

class Classifier(ClassifierBase):
    def __init__(self, name, embeddings, classify):
        self.name = name
        self.embeddings = embeddings
        self.classify = classify

        

    def __call__(self, edge):
        score = self.embeddings.get_score(edge[0], edge[1])
        return self.classify((*edge, score))