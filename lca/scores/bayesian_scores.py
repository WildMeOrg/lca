# -*- coding: utf-8 -*-
import logging
import math as m
import random
import scipy.stats as stats
import numpy as np
from scipy.stats import gaussian_kde
from tools import save_pickle

logger = logging.getLogger('lca')


class bayesian_scores(object):  
    """Model the verification scores using Bayesian probability.
    For a given score, two likelihoods are produced: one for the
    positive (correct) matches and one for the negative (incorrect)
    matches. These likelihoods are estimated from the score distributions
    of positive and negative matches. A Bayesian posterior probability
    is then computed to determine the likelihood of a correct match.
    """

    def __init__(self, pos_kde, neg_kde, prior_pos=0.5):
        """Initialize with KDE models for positive and negative samples and a prior probability."""
        self.pos_kde = pos_kde
        self.neg_kde = neg_kde
        self.prior_pos = prior_pos  # Prior probability of a positive match

    @classmethod
    def create_from_samples(cls, pos_samples, neg_samples):
        """Create a BayesianScores object from histograms of scores
        from the verification algorithm on positive and negative samples.
        """
        # save_pickle({"pos": pos_samples, "neg": neg_samples}, "../samples.pickle")
        logger.info('Fitting KDE models to positive and negative samples')
        
        # Fit KDEs to the positive and negative samples
        pos_kde = gaussian_kde(pos_samples)
        neg_kde = gaussian_kde(neg_samples)
        
        # Assuming a prior probability (e.g., 50% chance for a positive match)
        prior_pos = len(pos_samples) / (len(pos_samples) + len(neg_samples))
        
        return cls(pos_kde, neg_kde, prior_pos)

    def get_pos_neg(self, score):
        """
        Get the positive and negative posterior probabilities for a score.
        """
        # Compute likelihoods for positive and negative samples
        likelihood_pos = self.pos_kde(score)
        likelihood_neg = self.neg_kde(score)
        
        # Apply Bayes' rule to compute the posterior probability of a positive match
        posterior_pos = (likelihood_pos * self.prior_pos) / (
            likelihood_pos * self.prior_pos + likelihood_neg * (1 - self.prior_pos) + 1e-8)
        posterior_neg = 1 - posterior_pos
        
        return posterior_pos, posterior_neg

    def random_pos_neg(self):
        """Generate a random entry from the distributions. First, decide
        if the match will be sampled from the positive or negative
        distributions, then sample from the distributions.
        """
        is_match_correct = random.random() < self.prior_pos
        s = self.random_score(is_match_correct)
        return self.get_pos_neg(s), is_match_correct

    def random_score(self, is_match_correct):
        """Generate a random score based on whether the match is correct or not."""
        if is_match_correct:
            return self.pos_kde.resample(1)[0]
        else:
            return self.neg_kde.resample(1)[0]

    def raw_wgt_(self, s):
        """Return a continuous-valued weight from the score, using
        Bayesian posterior probability for positive matches.
        """
        ratio = self.get_prob(s)
        eps = 1e-8
        wgt = m.log(max(ratio, eps) / max(1 - ratio, eps))
        return wgt

    def get_prob(self, s):
        """Get the Bayesian posterior probability of a positive match."""
        posterior_pos, _ = self.get_pos_neg(s)
        return np.clip(posterior_pos, 0, 1)

