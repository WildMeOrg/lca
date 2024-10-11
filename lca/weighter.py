# -*- coding: utf-8 -*-
import logging
import math as m
import random
import numpy as np
import exp_scores as es

#  Need to add weight according to verifier names!!!

logger = logging.getLogger('lca')


class weighter(object):  # NOQA
    """Object whose primary function is to generate a signed edge
    weight from a verification score. This verification score is
    converted to a value from histograms of positive and negative
    scores and thence to a weight.  The weights are always integer.
    The human_prob of decision correctness is used to scale and bound
    the weights. Specifically, the human_prob -> max_weight and
    nothing can be outside the range -max_weight to max_weight
    """

    def __init__(self, scorer, human_prob=0.98):
        self.scorer = scorer
        self.human_prob = human_prob
        self.incomparable_weight = 0
        self.max_weight = 100  # should not change
        self.max_raw_weight = self.scorer.raw_wgt_(human_prob)
        logger.info(
            'Built weighter with human_prob %1.2f and max_weight %d'
            % (self.human_prob, self.max_weight)
        )

    def wgt(self, score):
        """Given a verification score produce a (scalar) weight"""
        w0 = self.scorer.raw_wgt_(score)
        w = self.scale_and_trunc_(w0)
        return int(w)
    
    def wgt_smooth(self, score):
        """Given a verification score produce a (scalar) weight"""
        w0 = self.scorer.raw_wgt_(score)
        w0 = np.clip(w0, -self.max_raw_weight, self.max_raw_weight)
        # logger.info(f"Clipped score: {w0}")
        w = w0 / self.max_raw_weight * self.max_weight
        return w

    def human_wgt(self, is_marked_correct):
        """Given a human decision, produce a weight"""
        if is_marked_correct is None:
            return self.incomparable_weight

        if is_marked_correct:
            return self.max_weight
        else:
            return -self.max_weight

    def random_wgt(self, is_match_correct):
        """Generate a random weight depending on whether or not the
        match is correct.
        """
        s = self.scorer.random_score(is_match_correct)
        return self.wgt(s)

    def human_random_wgt(self, is_marked_correct):
        """
        Return random weight for human decision.  Combine the correctness
        of the match with a random flip as to whether the human makes
        a correct decision.
        """
        p = random.random()
        correct_decision = p <= self.human_prob
        if (correct_decision and is_marked_correct) or (
            not correct_decision and not is_marked_correct
        ):
            return self.max_weight
        else:
            return -self.max_weight

    
    def scale_and_trunc_(self, w0):
        """Map the weight into an integer in the range -max_weight,
        ... max_weight.
        """
        # logger.info(f"Input score: {w0}")
        w0 = np.clip(w0, -self.max_raw_weight, self.max_raw_weight)
        # logger.info(f"Clipped score: {w0}")
        w = round(w0 / self.max_raw_weight * self.max_weight)
        # logger.info(f"Final score: {w}")
        return w


def test_weighter():
    error_frac = 0.15
    neg_pos_ratio = 5.0
    scorer = es.exp_scores.create_from_error_frac(error_frac, neg_pos_ratio)

    human_prob = 0.98
    s2w = weighter(scorer, human_prob)

    logger.info('\nSampling of weights:')
    n = 100
    for i in range(n + 1):
        s = i / n
        logger.info('score: %4.2f, wgt %d' % (s, s2w.wgt(s)))

    """
    n = 25
    logger.info("\nGenerating", n, "positives")
    wgts = sorted([ew.random_wgt(is_match_correct=True) for i in range(n)])
    for w in wgts:
        logger.info(w, end=' ')
    logger.info()

    logger.info("\nGenerating", n, "negatives")
    wgts = sorted([ew.random_wgt(is_match_correct=False) for i in range(n)])
    for w in wgts:
        logger.info(w, end=' ')
    logger.info()

    marked_correct = True
    logger.info("\nPositive weight from human = %d" % ew.human_wgt(marked_correct))

    marked_correct = False
    logger.info("Negative weight from human = %d" % ew.human_wgt(marked_correct))

    is_match_correct = True
    num_samples = 1000
    num_errors = 0
    for i in range(num_samples):
        wgt = ew.human_random_wgt(is_match_correct)
        if wgt != ew.max_weight:
            num_errors += 1
    logger.info("Positive human_random_wgt: num errors %d, num samples %d, pct errors %1.2f"
          % (num_errors, num_samples, 100*num_errors/num_samples))

    """


if __name__ == '__main__':
    test_weighter()
