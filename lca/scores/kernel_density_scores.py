# -*- coding: utf-8 -*-
import logging
import math as m
import random
import scipy.stats as stats
import numpy as np
from sklearn.neighbors import KernelDensity
from tools import save_pickle

logger = logging.getLogger('lca')


# class kernel_density_scores(object):  # NOQA
#     """Model the verification scores as exponential distribution
#     representations of two histograms, truncated to the domain [0,1].
#     For any given score, two histogram values are produced, one for the
#     positive (correct) matches and one for the negative (incorrect)
#     matches. The histogram for the positive matches is represented by
#     an exponential distribution truncated to the domain [0,1] and
#     reversed so the peak is at 1.0.  The histogram for the negative
#     matches is represented by a different truncated exponential
#     distribution, together with a ratio of the expected number of
#     negative to positive matches.
#     """

#     def __init__(self, prior, density_pos, density_neg):
#         """Construct the object from the three main parameters"""
#         self.prior = prior
#         self.density_pos = density_pos
#         self.density_neg = density_neg

#     @classmethod
#     def create_from_error_frac(cls, error_frac, np_ratio, create_from_pdf=True):
#         raise NotImplemented()

#     @classmethod
#     def create_from_samples(cls, pos_samples, neg_samples):
#         """Create an exp_scores object from histogram of scores
#         samples from the verification algorithm on positive and
#         negative samples.  It is VERY important that the relative
#         number of positive and negative samples reasonably represents
#         the distribution of samples fed into the verification
#         algorithm.
#         """
#         save_pickle({"pos":pos_samples, "neg":neg_samples}, "../samples.pickle")
#         logger.info('creating exp_scores from ground truth sample distributions')
#         prior = len(pos_samples) / (len(pos_samples) + len(neg_samples))
#         prior = 1 - prior
#         prior = 0.5
#         pos_density = CustomKernelDensity(np.array(pos_samples))
#         neg_density = CustomKernelDensity(np.array(neg_samples))
#         logger.info('estimate of a prior: %.3f' % prior)

#         return cls(prior, pos_density, neg_density)

#     def get_pos_neg(self, score):
#         """
#         Get the positive and negative histogram values for a
#         score.
#         """
#         hp = self.density_pos.density(score)
#         hn = self.density_neg.density(score)
#         return hp, hn

#     def random_pos_neg(self):
#         """Generate a random entry from the histograms. First decide
#         is the match will be sample from the positive or negative
#         distributions and then sample from the histograms.
#         """
#         is_match_correct = random.random() > self.np_ratio / (self.np_ratio + 1)
#         s = self.random_score(is_match_correct)
#         return self.get_pos_neg(s), is_match_correct

#     def random_score(self, is_match_correct):
#         """Generate a random score (not histogram entry) from the
#         truncated exponential distributions depending on whether the
#         match is correct or not.  This only returns a score.
#         """
#         if is_match_correct:
#             s = self.density_pos.sample()
#         else:
#             s = self.density_neg.sample()
#         return s
    
#     def raw_wgt_(self, s):
#         """Return a continuous-valued weight from the score, using
#         the scorer to get positive and negative histogram values.
#         """
#         # logger.info(f"Input score: {s}")
#         ratio = self.get_prob(s)
#         # logger.info(f"Probability: {ratio}")
#         eps = 1e-8
#         wgt = m.log(ratio/(1-ratio + eps))
#         # logger.info(f"Weight: {wgt}")
#         return wgt

#     def get_prob(self, s):
#         pos, neg = self.get_pos_neg(s)
#         prob = (self.prior * pos) / np.maximum(self.prior * pos + (1-self.prior) * neg, 1e-8)
#         # logger.info(f"Score: {s}, pos: {pos}, neg: {neg}, prior: {self.prior}, prob: {prob}")
#         # eps = 1e-8
#         # prob = pos / (neg + pos + eps)
#         # print(prob)
#         return prob

# # def normal_kernel(x, m, sigma):
# #     return np.exp(-(x-m)**2/(2*sigma**2))/np.sqrt(2 * np.pi)
# def normal_kernel(x):
#     return np.exp(-x**2/2)/np.sqrt(2 * np.pi)

# # def normal_kernel(x, m, sigma):
# #     if x < sigma + m and x > m - sigma:
# #         return x
# #     else:
# #         return 0

# class CustomKernelDensity(object):
#     def __init__(self, vals, bandwidth=0.03, bins=35, filter_mult=1.5, filter_threshold=0.1):
#         hist, bin_edges = np.histogram(vals, bins=bins, density=True)
#         self.bandwidth= bandwidth
#         self.hist = hist
#         self.xs = (bin_edges[:-1] + bin_edges[1:])/2
#         # for i, (v, x) in enumerate(zip(self.hist, self.xs)):
#         #     tst = np.sum(self.hist * normal_kernel((self.xs-x)/(filter_mult*self.bandwidth))) - np.sum(self.hist * normal_kernel((self.xs-v)/(2*self.bandwidth)))
#         #     if tst < filter_threshold:
#         #         self.hist[i] = 0
            

#     def density(self, score):
#         # r = 0

#         # for v, x in zip(self.hist, self.xs):
#         #     r = r + v * normal_kernel((x-score)/self.bandwidth)
#         #     # if x >= e1 and x < e2:
#         #     #     return v
#         # r = r/(len(self.hist) * self.bandwidth)

#         r = np.sum(self.hist * normal_kernel((self.xs-score)/self.bandwidth)) /(len(self.hist) * self.bandwidth)
#         return r
        

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
        prior = 0.5
        bandwidth = 0.05
        pos_density = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(pos_samples).reshape(-1, 1))
        neg_density = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(neg_samples).reshape(-1, 1))
        logger.info('estimate of a prior: %.3f' % prior)

        return cls(prior, pos_density, neg_density)

    def normal_kernel(self, x):
        return np.exp(-(x-0.4)**2/(2*(0.05)**2))

    def get_pos_neg(self, score):
        """
        Get the positive and negative histogram values for a
        score.
        """
        hp = np.exp(self.density_pos.score_samples([[score]]))
        hn = np.exp(self.density_neg.score_samples([[score]]))
        # if score > 0.5:
        #     hn *= self.normal_kernel(score)
        # else:
        #     hp *= self.normal_kernel(score)
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
            s = self.density_pos.sample()
        else:
            s = self.density_neg.sample()
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