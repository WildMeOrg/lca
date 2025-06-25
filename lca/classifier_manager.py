import numpy as np

class ClassifierManager(object):  # NOQA
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def find_next_classifier(self, classifier_name):
        # Given a classifier_name, find its index in self.classifiers 
        # and return the next classifier from the list
        raise NotImplemented()

    def get_edge_weights(self, edges):
        """_summary_

        Args:
            edges ([(n0, n1, classifier_name)...]): an iterable of edges of the form (n0, n1, classifier_name)
            excluded_classifiers (list, optional): a list of classifiers to exclude. 
                                                 Intented use is to either use all classifiers ([]), 
                                                 or exclude human (["human"]). Defaults to [].

        Returns:
            [(n0, n1, score, confidence, label, ranker_name)...]: verified edges. 
                The edges that would be verified asynchronously (e.g. by a human) 
                are added to the corresponding classifiers and will be collected later
        """
        result = []
        for (n0, n1, classifier_name) in edges:
            classifier = self.find_next_classifier(classifier_name)
            # Check if we want to avoid some classifier (for example, we don't want human reviews)
            
            verified_edge = classifier((n0, n1, classifier_name))
            # classifier either outputs verified edge in the format:
            # (n0, n1, score, confidence, label, ranker_name)
            
            result.append(verified_edge)
        return result


