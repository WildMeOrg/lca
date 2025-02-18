from scipy.stats import gaussian_kde, rv_continuous
import numpy as np
from numpy import (atleast_2d,cov,pi)
from scipy import linalg

class KDEDist(rv_continuous):
    
    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde
    
    def _pdf(self, x):
        return self._kde.pdf(x)
    
    def _cdf(self, x):
        return self._kde.integrate_box_1d(0, x)

class TruncatedKDE(rv_continuous):
    def __init__(self, kde, a=0, b=1, *args, **kwargs):
        super().__init__(a=a, b=b, *args, **kwargs)
        self.kde = kde
    def _pdf(self, x):
        g = 0 if ((x < self.a) or (x > self.b)) else self.kde.pdf(x)
        return g / (self.kde.cdf(self.b) - self.kde.cdf(self.a))
    def _cdf(self, x):
        lower = self.kde.cdf(self.a)
        return (self.kde.cdf(x) - lower)/(self.kde.cdf(self.b) - lower)

def truncated_kde(samples, lower=0, upper=1):
    density = gaussian_kde(samples)
    # return density
    kde_distribution = KDEDist(density)
    return TruncatedKDE(kde_distribution, lower, upper)

class kernel_density_scores:
    def __init__(self, prior, density_pos, density_neg):
        self.prior = prior
        self.density_pos = density_pos
        self.density_neg = density_neg

    @classmethod
    def create_from_samples(cls, pos_samples, neg_samples):
        """Create KDEs using scipy's gaussian_kde."""
        prior = len(pos_samples) / (len(pos_samples) + len(neg_samples))
        prior = 0.5
        bandwidth = 0.05

        # Create KDEs for positive and negative samples
        pos_density = truncated_kde(np.array(pos_samples).reshape((1,-1)))#, bw_method=bandwidth)
        neg_density = truncated_kde(np.array(neg_samples).reshape((1,-1)))#, bw_method=bandwidth)

        return cls(prior, pos_density, neg_density)

    def get_pos_neg(self, score):
        """Get positive and negative histogram values for a score."""
        hp = self.density_pos.pdf(score)
        hn = self.density_neg.pdf(score)
        return hp, hn

    def rejection_sampling(self, density):
        

    def random_score(self, is_match_correct, sz=None):
        """Generate a random score."""
        if is_match_correct:
            return self.density_pos.rvs(size=sz)  # Single scalar value
        else:
            return self.density_neg.rvs(size=sz)
