# -*- coding: utf-8 -*-
import logging
import math as m
import random
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from tools import save_pickle

logger = logging.getLogger('lca')

class kernel_density_scores(object):  # NOQA
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

    def __init__(self, prior, density_pos, density_neg):
        """Construct the object from the three main parameters"""
        self.prior = prior
        self.density_pos = density_pos
        self.density_neg = density_neg


    def sample_from_density(self, density, num_samples=1):
        """
        Sample from a given KDEUnivariate density.
        """
        # Generate points in the range of the density
        x_min = density.support.min()
        x_max = density.support.max()

        # Create a fine grid of points
        x_grid = np.linspace(x_min, x_max, 1000)
        
        # Evaluate the density on this grid
        pdf = density.evaluate(x_grid)
        
        # Normalize the PDF
        pdf /= pdf.sum()
        
        # Use inverse transform sampling
        return np.random.choice(x_grid, size=num_samples, p=pdf)


    @classmethod
    def create_from_samples(cls, pos_samples, neg_samples):
        """Create an exp_scores object from histogram of scores
        samples from the verification algorithm on positive and
        negative samples.
        """
        save_pickle({"pos": pos_samples, "neg": neg_samples}, "../samples.pickle")
        logger.info('creating kernel_density_scores from ground truth sample distributions')
        prior = len(pos_samples) / (len(pos_samples) + len(neg_samples))
        prior = 0.5
        bandwidth = 0.05

        # Flatten the samples to ensure they are 1D
        pos_samples = np.array(pos_samples).flatten()
        neg_samples = np.array(neg_samples).flatten()

        # Create KDE for positive samples
        pos_density = KDEUnivariate(pos_samples)
        pos_density.fit(kernel='gau', bw=bandwidth, fft=True)

        # Create KDE for negative samples
        neg_density = KDEUnivariate(neg_samples)
        neg_density.fit(kernel='gau', bw=bandwidth, fft=True)

        logger.info('estimate of a prior: %.3f' % prior)
        return cls(prior, pos_density, neg_density)
    
    def normal_kernel(self, x):
        return np.exp(-(x-0.4)**2/(2*(0.05)**2))

    def get_pos_neg(self, score):
        """
        Get the positive and negative histogram values for a
        score.
        """
        hp = self.density_pos.evaluate(score)
        hn = self.density_neg.evaluate(score)
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
        """Generate a random score from the truncated exponential
        distributions depending on whether the match is correct or not.
        """
        if is_match_correct:
            return self.sample_from_density(self.density_pos)[0]
        else:
            return self.sample_from_density(self.density_neg)[0]

    def raw_wgt_(self, s):
        """Return a continuous-valued weight from the score, using
        the scorer to get positive and negative histogram values.
        """
        ratio = self.get_prob(s)
        eps = 1e-8
        wgt = m.log(ratio / (1 - ratio + eps))
        return wgt

    def get_prob(self, s):
        pos, neg = self.get_pos_neg(s)
        prob = (self.prior * pos) / np.maximum(self.prior * pos + (1 - self.prior) * neg, 1e-8)
        return prob
