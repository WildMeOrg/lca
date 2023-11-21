# -*- coding: utf-8 -*-
import logging
import math as m
import random


logger = logging.getLogger('wbia_lca')


class exp_scores(object):  # NOQA
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

    def __init__(self, np_ratio, pos_lambda, neg_lambda):
        """Construct the object from the three main parameters"""
        self.np_ratio = np_ratio
        self.trunc_exp_pos = truncated_exponential(pos_lambda)
        self.trunc_exp_neg = truncated_exponential(neg_lambda)

    @classmethod
    def create_from_error_frac(cls, error_frac, np_ratio, create_from_pdf=True):
        """Create an exp_scores object from a model of the expected fraction
        of scores that are in error (negative when they should be
        positive), along with the np_ratio, as above.
        """
        assert 0 <= error_frac < 0.5
        assert np_ratio >= 1.0
        pos_lambda = find_lambda_cdf(1.0, error_frac)

        if create_from_pdf:
            neg_lambda = find_lambda_pdf(np_ratio, pos_lambda)
        else:
            neg_lambda = find_lambda_cdf(np_ratio, error_frac)

        # Debugging output
        trunc_exp_pos = truncated_exponential(pos_lambda)
        logger.info('creating exp_scores from error fractions')
        logger.info('error fraction %1.3f' % error_frac)
        logger.info('positive error rate %1.3f' % (1 - trunc_exp_pos.cdf(0.5)))
        trunc_exp_neg = truncated_exponential(neg_lambda)
        logger.info(
            'negative error rate %1.3f' % (np_ratio * (1 - trunc_exp_neg.cdf(0.5)))
        )

        return cls(np_ratio, pos_lambda, neg_lambda)

    @classmethod
    def create_from_samples(cls, pos_samples, neg_samples):
        """Create an exp_scores object from histogram of scores
        samples from the verification algorithm on positive and
        negative samples.  It is VERY important that the relative
        number of positive and negative samples reasonably represents
        the distribution of samples fed into the verification
        algorithm.
        """
        logger.info('creating exp_scores from ground truth sample distributions')
        np_ratio = len(neg_samples) / len(pos_samples)
        pos_lambda = find_lambda_from_samples(pos_samples, is_positive=True)
        neg_lambda = find_lambda_from_samples(neg_samples, is_positive=False)
        logger.info('negative positive ratio: %.3f' % np_ratio)
        logger.info('positive lambda for expoential: %.3f' % pos_lambda)
        logger.info('negative lambda for expoential: %.3f' % neg_lambda)

        return cls(np_ratio, pos_lambda, neg_lambda)

    def get_pos_neg(self, score):
        """
        Get the positive and negative histogram values for a
        score.
        """
        hp = self.trunc_exp_pos.pdf(1 - score)
        hn = self.np_ratio * self.trunc_exp_neg.pdf(score)
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
            s = 1 - self.trunc_exp_pos.sample()
        else:
            s = self.trunc_exp_neg.sample()
        return s


class truncated_exponential(object):  # NOQA
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.normalize = 1 - m.exp(-self.lmbda)

    def pdf(self, x):
        assert 0 <= x <= 1
        return self.lmbda * m.exp(-self.lmbda * x) / self.normalize

    def cdf(self, x):
        assert 0 <= x <= 1
        return (1 - m.exp(-self.lmbda * x)) / self.normalize

    def sample(self):
        p = random.random()
        x = -m.log(1 - self.normalize * p) / self.lmbda
        return x

    def mean(self):
        mu = 1 / self.lmbda - m.exp(-self.lmbda) / self.normalize
        return mu


def find_lambda_cdf(np_ratio, error_frac):
    """
    Find the parameter lambda such that when we form a truncated
    exponential distribution using lambda then the expected
    error_fraction of values above 0.5 in the distribution times the
    np_ratio equals the given error_frac.  When np_ratio==1 this is
    simply returns the value of lambda such that the cdf =
    (1-error_frac).
    """
    allowed_error = error_frac / np_ratio
    min_beta = 0.001
    max_beta = 0.999
    delta_cutoff = 0.0001
    while max_beta - min_beta >= delta_cutoff:
        beta = 0.5 * (min_beta + max_beta)
        te0 = truncated_exponential(1 / beta)
        tmp_error = 1 - te0.cdf(0.5)
        if tmp_error < allowed_error:
            min_beta = beta
        else:
            max_beta = beta

    beta = 0.5 * (min_beta + max_beta)
    return 1 / beta


def find_lambda_pdf(np_ratio, lambda_p):
    """
    Find the parameter lambda such that at 0.5

       np_ratio * pdf(0.5, lambda) = pdf(0.5, lambda_p)

    where the pdfs are from the truncated exponential.
    """
    te0 = truncated_exponential(lambda_p)
    target_pdf = te0.pdf(0.5)
    min_beta = 0.001
    max_beta = 0.999
    delta_cutoff = 0.000001
    while max_beta - min_beta >= delta_cutoff:
        beta = 0.5 * (min_beta + max_beta)
        te1 = truncated_exponential(1 / beta)
        scaled_pdf = np_ratio * te1.pdf(0.5)
        if scaled_pdf < target_pdf:
            min_beta = beta
        else:
            max_beta = beta

    beta = 0.5 * (min_beta + max_beta)
    lambda_n = 1 / beta
    return lambda_n


def find_lambda_from_samples(samples, is_positive=True):
    if is_positive:
        samples = [1 - s for s in samples]
    pop_mean = sum(samples) / len(samples)
    min_beta = 0.001
    max_beta = 0.999
    delta_cutoff = 0.000001
    while max_beta - min_beta >= delta_cutoff:
        beta = 0.5 * (min_beta + max_beta)
        te = truncated_exponential(1 / beta)
        mean = te.mean()
        if mean < pop_mean:
            min_beta = beta
        else:
            max_beta = beta

    beta = 0.5 * (min_beta + max_beta)
    lmbda = 1 / beta
    return lmbda


"""  test functions  """


def test_truncated_exponential():
    lmbda = 1 / 0.3
    te = truncated_exponential(lmbda)
    n = 20
    for i in range(n + 1):
        x = i / n
        p = te.pdf(x)
        c = te.cdf(x)
        logger.info('%4.2f: %5.3f %5.3f' % (x, p, c))

    n = 10000
    s = 0
    for i in range(n):
        s += te.sample()
    logger.info('Average from sample %.4f' % (s / n))

    # Numerically integrate:
    integral = 0
    n = 1000
    delta = 1 / n
    for i in range(n):
        x = (i + 0.5) / n
        integral += te.pdf(x)
    integral *= delta
    logger.info('PDF integrates to %1.4f' % integral)


def test_find_lambda():
    pairs = [(1.0, 0.2), (6.5, 0.3)]
    for r, err in pairs:
        """
        Find the positive lambda based on the error rate by simply
        using.
        """
        logger.info('----------')
        logger.info('test find_lambda_cdf ')
        logger.info(
            'r = %s err = %s'
            % (
                r,
                err,
            )
        )
        lmbda_pos = find_lambda_cdf(1, err)  # find the positive match lambda
        lmbda_neg = find_lambda_cdf(r, err)  # find the negative match lambda
        logger.info('lmbda_pos = %s' % (lmbda_pos,))
        te_pos = truncated_exponential(lmbda_pos)
        te_neg = truncated_exponential(lmbda_neg)
        logger.info('goal: positive frac below 0.5 prob = %s' % (err,))
        logger.info(
            'estimated: negative frac above 0.5 = %s' % (r * (1 - te_neg.cdf(0.5)),)
        )

        logger.info('----------')
        lmbda_neg = find_lambda_pdf(r, lmbda_pos)
        logger.info('test find_lambda_pdf ')
        logger.info(
            'r = %s lmbda_pos = %s'
            % (
                r,
                lmbda_pos,
            )
        )
        logger.info('lmbda_neg = %s' % (lmbda_neg,))
        te_neg = truncated_exponential(lmbda_neg)
        logger.info('goal: pdf of positive at 0.5 = %.4f' % te_pos.pdf(0.5))
        logger.info('est: scaled pdf of negative at 0.5 = %.4f' % (r * te_neg.pdf(0.5),))
        logger.info('')


def test_find_lambda_from_samples():
    beta = 0.225
    lmbda = 1 / beta
    te = truncated_exponential(lmbda)

    logger.info('-----------\ntest_find_lambda_from_samples')
    n = 100000
    samples = [te.sample() for i in range(n)]
    s_mean = sum(samples) / len(samples)
    logger.info('from %d samples mean is %1.5f' % (n, s_mean))
    logger.info('population mean is %1.5f' % (te.mean()))
    est_lmbda = find_lambda_from_samples(samples, is_positive=False)
    logger.info('target lmbda %.4f, estimated %.4f' % (lmbda, est_lmbda))


def test_create_from_error_frac():
    np_ratio = 6.5
    error_frac = 0.3
    corr_lambda_pos = 1.6946
    corr_lambda_neg = 8.1818
    logger.info('------------\ntest_create_from_error_frac\n')
    score_obj = exp_scores.create_from_error_frac(
        error_frac, np_ratio, create_from_pdf=True
    )
    logger.info(
        "corr_lambda_pos %.4f, object's lambda %.4f"
        % (corr_lambda_pos, score_obj.trunc_exp_pos.lmbda)
    )
    logger.info(
        "corr_lambda_neg %.4f, object's lambda %.4f"
        % (corr_lambda_neg, score_obj.trunc_exp_neg.lmbda)
    )
    logger.info(
        "corr np_ratio %.2f, object's np_ratio %.2f" % (np_ratio, score_obj.np_ratio)
    )

    """  See if we can recreate the error fraction on positives from sampling
    """
    n = 100000
    num_match = num_pos_err = num_neg_err = 0
    for _ in range(n):
        (pos, neg), is_correct = score_obj.random_pos_neg()
        if is_correct:
            num_match += 1
            if pos < neg:
                num_pos_err += 1
            else:
                pass
        elif neg < pos:
            num_neg_err += 1

    exp_match_frac = 1 / (np_ratio + 1)
    logger.info('expected match frac %.4f, actual %.4f' % (exp_match_frac, num_match / n))
    exp_pos_error = round(n * exp_match_frac * error_frac)
    logger.info('expected num positive (and negative) errors: %s' % (exp_pos_error,))
    logger.info('actual num positive errors: %s' % (num_pos_err,))
    logger.info('actual num negative errors: %s' % (num_neg_err,))
    logger.info('pos pdf at 0.5: %.4f' % (score_obj.trunc_exp_pos.pdf(0.5)))
    logger.info(
        'np_ratio * neg pdf at 0.5: %.4f' % (np_ratio * score_obj.trunc_exp_neg.pdf(0.5))
    )
    cdf_at_half = score_obj.trunc_exp_neg.cdf(0.5)
    exp_neg_mistakes_from_cdf = round((1 - exp_match_frac) * n * (1 - cdf_at_half))
    logger.info('cdf_at_half (negative) = %.4f' % cdf_at_half)
    logger.info('exp_neg_mistakes_from_cid = %s' % (exp_neg_mistakes_from_cdf,))


def test_create_from_samples():
    logger.info('------------\ntest_create_from_error_from_samples\n')
    np_ratio = 5.0
    n_pos = 100000
    n_neg = int(n_pos * np_ratio)

    pos_beta = 0.24
    pos_lambda = 1 / pos_beta
    te_pos = truncated_exponential(pos_lambda)
    pos_samples = [1 - te_pos.sample() for _ in range(n_pos)]

    neg_beta = 0.15
    neg_lambda = 1 / neg_beta
    te_neg = truncated_exponential(neg_lambda)
    neg_samples = [te_neg.sample() for _ in range(n_neg)]

    score_obj = exp_scores.create_from_samples(pos_samples, neg_samples)
    logger.info(
        "pos_lambda %.4f, object's pos_lambda %.4f"
        % (pos_lambda, score_obj.trunc_exp_pos.lmbda)
    )
    logger.info(
        "neg_lambda %.4f, object's neg_lambda %.4f"
        % (neg_lambda, score_obj.trunc_exp_neg.lmbda)
    )
    logger.info("np_ratio %.2f, object's np_ratio %.2f" % (np_ratio, score_obj.np_ratio))


if __name__ == '__main__':
    test_truncated_exponential()
    # test_find_np_ratio()
    test_find_lambda()
    test_find_lambda_from_samples()
    test_create_from_error_frac()
    test_create_from_samples()
