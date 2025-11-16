"""
Robust GMM-based threshold detection for imbalanced binary classification.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=-1, keepdims=True)
    return x_max.squeeze() + np.log(np.sum(np.exp(x - x_max), axis=-1))


def _fit_gmm(X, K=2, max_iter=100, tol=1e-6, reg=1e-6,
             entropy_alpha=1.0, n_init=1, verbose=False, print_func=print):
    """Fit K-component GMM with entropy regularization."""
    n = len(X)
    best_ll = -np.inf
    best_params = None

    for run in range(n_init):
        # Initialize: use percentiles for first run, random for others
        if run == 0:
            percentiles = np.linspace(0, 100, K + 2)[1:-1]
            mu = np.percentile(X, percentiles)
        else:
            indices = np.random.choice(n, size=K, replace=False)
            mu = np.sort(X[indices])

        sigma2 = np.full(K, np.var(X) / K)
        pi = np.full(K, 1.0 / K)

        # EM iterations
        ll_old = -np.inf
        for _ in range(max_iter):
            # E-step: compute responsibilities
            log_resp = np.array([stats.norm.logpdf(X, mu[k], np.sqrt(sigma2[k]))
                                 for k in range(K)]).T
            log_resp += np.log(pi)
            ll = np.sum(_logsumexp(log_resp))

            gamma = np.exp(log_resp - _logsumexp(log_resp)[:, None])

            # Check convergence
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll

            # M-step: update parameters
            N_k = np.sum(gamma, axis=0)

            # Entropy regularization on mixing weights
            if entropy_alpha == 1.0:
                pi = N_k / n
            else:
                beta = 1.0 / entropy_alpha
                pi = N_k ** beta
                pi /= pi.sum()

            mu = np.sum(gamma * X[:, None], axis=0) / N_k
            sigma2 = np.sum(gamma * (X[:, None] - mu)**2, axis=0) / N_k + reg

        if ll > best_ll:
            best_ll = ll
            best_params = (pi.copy(), mu.copy(), sigma2.copy())

    return best_params


def _compute_f1(threshold, pi, mu, sigma2):
    """Compute predicted F1 score at threshold (K=2 only)."""
    cdf = [stats.norm.cdf(threshold, mu[k], np.sqrt(sigma2[k])) for k in range(2)]

    TP = pi[1] * (1 - cdf[1])
    FP = pi[0] * (1 - cdf[0])
    FN = pi[1] * cdf[1]

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * precision * recall / (precision + recall)


def _find_threshold(pi, mu, sigma2):
    """Find threshold maximizing predicted F1 score."""
    mu_min, mu_max = mu.min(), mu.max()
    range_width = mu_max - mu_min

    result = minimize_scalar(
        lambda t: -_compute_f1(t, pi, mu, sigma2),
        bounds=(mu_min - 0.5 * range_width, mu_max + 0.5 * range_width),
        method='bounded'
    )

    return result.x, -result.fun


def _plot_threshold(scores, pi, mu, sigma2, threshold, plot_path):
    """Plot histogram with fitted distributions and threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(scores, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black')

    # Generate x values for plotting distributions
    x_min, x_max = scores.min(), scores.max()
    x_range = x_max - x_min
    x = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 1000)

    # Plot left distribution (red) and right distribution (green)
    for k in range(2):
        pdf = stats.norm.pdf(x, mu[k], np.sqrt(sigma2[k]))
        color = 'red' if k == 0 else 'green'
        label = f'Component {k+1} (μ={mu[k]:.3f}, σ={np.sqrt(sigma2[k]):.3f}, π={pi[k]:.3f})'
        ax.plot(x, pdf, color=color, linewidth=2, label=label)

    # Plot threshold as vertical line
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.4f}')

    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('GMM-based Threshold Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_threshold(scores, entropy_alpha=0.9, n_init=1, verbose=False, print_func=print, plot_path=None):
    """
    Find optimal classification threshold from unlabeled scores.

    Fits 2-component GMM with entropy regularization to handle extreme imbalance.
    Automatically detects and handles left-tail distributions.

    Parameters
    ----------
    scores : array-like
        Classifier scores (combined positive and negative samples)
    entropy_alpha : float, default=0.9
        Entropy regularization: < 1 favors sparse mixtures (bulk + tail)
    n_init : int, default=1
        Number of random initializations
    verbose : bool, default=False
        Print fitting progress
    plot_path : str, optional
        Path to save histogram plot with fitted distributions and threshold

    Returns
    -------
    threshold : float
        Optimal decision threshold
    predicted_f1 : float
        Predicted F1 score at threshold
    """
    scores = np.asarray(scores).ravel()

    # Fit K=2 GMM
    pi, mu, sigma2 = _fit_gmm(scores, K=2, entropy_alpha=entropy_alpha,
                               n_init=n_init, verbose=verbose, print_func=print_func)

    # Safety check: if small component is LEFT of large component,
    # we have a left tail → refit with K=3 and use last 2 components
    if pi[0] < pi[1] and mu[0] < mu[1]:
        if verbose:
            print_func("⚠ Detected left tail, refitting with K=3...")

        pi3, mu3, sigma2_3 = _fit_gmm(scores, K=3, entropy_alpha=0.1,
                                       n_init=max(5, n_init), verbose=verbose, print_func=print_func)

        # Use components 2 and 3 (ignore left tail)
        pi = pi3[1:3]
        pi /= pi.sum()
        mu = mu3[1:3]
        sigma2 = sigma2_3[1:3]

    # Find optimal threshold
    threshold, predicted_f1 = _find_threshold(pi, mu, sigma2)
    print_func(f"✅ Optimal threshold: {threshold:.4f}, predicted F1: {predicted_f1:.4f}")

    # Generate plot if requested
    if plot_path is not None:
        _plot_threshold(scores, pi, mu, sigma2, threshold, plot_path)
        print_func(f"📊 Saved threshold plot to: {plot_path}")

    return threshold
