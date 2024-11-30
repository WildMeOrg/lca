# -*- coding: utf-8 -*-
import logging
import math as m
import random
import scipy.stats as stats
import numpy as np

from tools import save_pickle

logger = logging.getLogger('lca')


class extexp_scores(object):  # NOQA
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

    def __init__(self, prior, extexp_pos, extexp_neg):
        """Construct the object from the three main parameters"""
        self.prior = prior
        self.extexp_pos = extexp_pos
        self.extexp_neg = extexp_neg

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
        save_pickle({"pos":pos_samples, "neg":neg_samples}, "../samples.pickle")
        logger.info('creating exp_scores from ground truth sample distributions')
        prior = len(pos_samples) / (len(pos_samples) + len(neg_samples))
        pos_dist = extexp.create_from_samples(pos_samples, is_positive=True)
        neg_dist = extexp.create_from_samples(neg_samples, is_positive=False)
        logger.info('estimate of a prior: %.3f' % prior)
        logger.info(f'positive distribution: {pos_dist}')
        logger.info(f'negative distribution: {neg_dist}')

        return cls(prior, pos_dist, neg_dist)

    def get_pos_neg(self, score):
        """
        Get the positive and negative histogram values for a
        score.
        """
        hp = self.extexp_pos.pdf(score)
        hn = self.extexp_neg.pdf(score)
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
        if is_match_correct:
            s = self.extexp_pos.sample()
        else:
            s = self.extexp_neg.sample()
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
        return wgt

    def get_prob(self, s):
        pos, neg = self.get_pos_neg(s)
        prob = (self.prior * pos) / np.maximum(self.prior * pos + (1-self.prior) * neg, 1e-8)
        # logger.info(f"Score: {s}, pos: {pos}, neg: {neg}, prior: {self.prior}, prob: {prob}")
        return prob


class extexp(object):
    def __init__(self, loc, scale, is_positive):
        self.loc = loc
        self.scale = scale
        self.is_positive = is_positive
        self.extexp_dist = stats.expon(scale=scale, loc=loc)
    
    def check_reverse(self, x):
        """
        The long tail should point in the opposite direction. 
        For the scores, if the score is high for the positive 
        distribution, the tail should extend to the negative 
        direction and vice versa. This is accomplished by 
        negating the values because gamma distribution tail is 
        pointing forward
        """
        return np.maximum(self.loc, -x if self.is_positive else x)

    def pdf(self, x):
        return stats.expon.pdf(self.check_reverse(x), scale=self.scale, loc=self.loc)
    
    def cdf(self, x):
        return stats.expon.cdf(self.check_reverse(x), scale=self.scale, loc=self.loc)
    
    def sample(self):
        return self.check_reverse(stats.expon.rvs(scale=self.scale, loc=self.loc))
    
    def mean(self):
        return self.check_reverse(stats.expon.mean(scale=self.scale, loc=self.loc))
    
    def __repr__(self):
        return f"Truncated exponential distribution(loc={self.loc}, scale={self.scale}, is_positive={self.is_positive})"

    @classmethod
    def create_from_samples(cls, samples, is_positive):
        if is_positive:
            samples = -np.array(samples)
        loc, scale = stats.expon.fit(samples)
        return cls(loc, scale, is_positive)