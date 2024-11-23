# -*- coding: utf-8 -*-
import logging
import math as m
import random
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import LogisticRegression
from tools import save_pickle

logger = logging.getLogger('lca')


class logistic_scores(object):  # NOQA
    """Model the verification scores as exponential distribution
    representations of two histograms, truncated to the domain [0,1].
    For any given score, two histogram values are produced, one for the
    positive (correct) matches and one for the negative (incorrect)
    matches. The histogram for the positive matches is represented by
    an exponential distribution truncated to the domain [0,1] and
    reversed so the peak is at 1.0.  The histogram for the negative
    matches is represented by a different truncated exponential
    distribution, together with a ratio of the expected number of
    negative to positive matches.
    """

    def __init__(self, prior, model):
        """Construct the object from the three main parameters"""
        self.np_ratio = prior
        self.model = model

    @classmethod
    def create_from_error_frac(cls, error_frac, np_ratio, create_from_pdf=True):
        raise NotImplemented()

    @classmethod
    def create_from_samples(cls, pos_samples, neg_samples):
        """Create an exp_scores object from histogram of scores
        samples from the verification algorithm on positive and
        negative samples.  It is VERY important that the relative
        number of positive and negative samples reasonably represents
        the distribution of samples fed into the verification
        algorithm.
        """
        # save_pickle({"pos":pos_samples, "neg":neg_samples}, "../samples.pickle")
        logger.info('fitting logistic regression to ground truth samples')
        X = np.concatenate((pos_samples, neg_samples))
        y = np.concatenate((np.ones(len(pos_samples)), np.zeros(len(neg_samples))))
        model = LogisticRegression(class_weight="balanced")
        model.fit(X.reshape(-1, 1), y)
        prior = len(pos_samples) / (len(pos_samples) + len(neg_samples))
        prior = 0.5
        
        logger.info('estimate of a prior: %.3f' % prior)

        return cls(prior, model)

    def get_pos_neg(self, score):
        """
        Get the positive and negative histogram values for a
        score.
        """
        hp = self.model.predict_proba([[score]])[0, 1]
        hn = self.model.predict_proba([[score]])[0, 0]
        return hp, hn

    def random_pos_neg(self):
        """Generate a random entry from the histograms. First decide
        is the match will be sample from the positive or negative
        distributions and then sample from the histograms.
        """
        is_match_correct = random.random() > self.np_ratio / (self.np_ratio + 1)
        s = self.random_score(is_match_correct)
        return self.get_pos_neg(s), is_match_correct

    def random_score(self, is_match_correct):
        """Generate a random score (not histogram entry) from the
        truncated exponential distributions depending on whether the
        match is correct or not.  This only returns a score.
        """
        raise NotImplemented()
        return s
    
    def raw_wgt_(self, s):
        """Return a continuous-valued weight from the score, using
        the scorer to get positive and negative histogram values.
        """
        # logger.info(f"Input score: {s}")
        ratio = self.get_prob(s)
        # logger.info(f"Probability: {ratio}")
        eps = 1e-8
        wgt = m.log(ratio/(1-ratio + eps))
        # logger.info(f"Weight: {wgt}")
        return wgt#**3

    def get_prob(self, s):
        # logger.info(f"Score: {s}, pos: {pos}, neg: {neg}, prior: {self.prior}, prob: {prob}")
        return self.model.predict_proba([[s]])[0, 1]

