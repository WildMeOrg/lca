"""Utility functions for threshold detection algorithms."""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture


def fit_robust_gaussian(data: np.ndarray, sigma_threshold: float = 3.0, max_iterations: int = 10) -> tuple[float, float]:
    """
    Robust Gaussian fitting using iterative σ-clipping.
    
    Returns:
        (mean, std) of the fitted Gaussian
    """
    current_data = data.copy()
    
    for iteration in range(max_iterations):
        mu, sigma = stats.norm.fit(current_data)
        
        # Identify inliers within sigma_threshold
        distances = np.abs(current_data - mu) / sigma
        inliers = distances <= sigma_threshold
        
        # Check convergence
        if np.all(inliers):
            break
            
        # Remove outliers for next iteration
        current_data = current_data[inliers]
        
        # Safety check - don't remove too much data
        if len(current_data) < len(data) * 0.3:
            break
    
    # Final fit
    return stats.norm.fit(current_data)


def fit_gmm(data: np.ndarray, n_components: int = 2, max_iter: int = 200, 
            random_state: int = 42) -> dict:
    """
    Fit Gaussian Mixture Model to data.
    
    Returns:
        Dictionary with success status and fitted parameters
    """
    try:
        gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type='full',
            max_iter=max_iter, 
            random_state=random_state, 
            init_params='kmeans'
        )
        gmm.fit(data.reshape(-1, 1))
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_
        
        # Sort by mean
        sort_idx = np.argsort(means)
        
        return {
            'success': True,
            'gmm': gmm,
            'means': means[sort_idx],
            'stds': stds[sort_idx],
            'weights': weights[sort_idx],
            'converged': gmm.converged_
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def check_gmm_separability(mean1: float, std1: float, weight1: float,
                          mean2: float, std2: float, weight2: float,
                          failure_threshold: float = 5.0) -> tuple[bool, dict]:
    """
    Check if GMM components are well-separated.
    
    Returns:
        (success, info_dict) where info_dict contains separability metrics
    """
    # Identify smaller component by weight
    if weight1 < weight2:
        smaller_mean, smaller_std = mean1, std1
        larger_mean = mean2
    else:
        smaller_mean, smaller_std = mean2, std2
        larger_mean = mean1
    
    # Check separability using smaller component's spread
    separation_distance = abs(smaller_mean - larger_mean)
    required_separation = failure_threshold * smaller_std
    
    success = separation_distance >= required_separation
    
    return success, {
        'separation_distance': separation_distance,
        'required_separation': required_separation,
        'smaller_std': smaller_std,
        'separability_ratio': separation_distance / required_separation,
        'reason': f"separation {separation_distance:.3f} < {required_separation:.3f}" if not success else ""
    }


def extract_tail_samples(data: np.ndarray, mean: float, std: float, 
                        sigma_threshold: float = 3.0) -> np.ndarray:
    """Extract samples beyond sigma_threshold standard deviations from mean."""
    tail_threshold = mean + sigma_threshold * std
    return data[data > tail_threshold], tail_threshold


def calculate_residuals(tail_samples: np.ndarray, main_mean: float, main_std: float,
                       tail_threshold: float, bins: int = 50) -> tuple:
    """
    Calculate residuals between observed and expected frequencies in tail.
    
    Returns:
        (bin_centers, residuals, expected_hist, observed_hist)
    """
    # Create histogram
    hist, bin_edges = np.histogram(tail_samples, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate expected frequencies
    tail_prob = 1 - stats.norm.cdf(tail_threshold, main_mean, main_std)
    
    bin_probs = []
    for i in range(len(bin_edges)-1):
        prob = stats.norm.cdf(bin_edges[i+1], main_mean, main_std) - \
               stats.norm.cdf(bin_edges[i], main_mean, main_std)
        conditional_prob = prob / tail_prob if tail_prob > 0 else 0
        bin_probs.append(conditional_prob)
    
    expected_hist = np.array(bin_probs) * len(tail_samples)
    residuals = hist - expected_hist
    
    return bin_centers, residuals, expected_hist, hist


def find_residual_zero_crossing(residuals: np.ndarray, bin_centers: np.ndarray,
                               smoothing_sigma: float, bin_width: float) -> tuple[float | None, np.ndarray]:
    """
    Find zero-crossing point in smoothed residuals.
    
    Returns:
        (crossing_value, smoothed_residuals) where crossing_value is None if not found
    """
    # Convert sigma from data units to bin units
    sigma_in_bins = smoothing_sigma / bin_width
    
    # Smooth residuals
    smoothed_residuals = gaussian_filter1d(residuals, sigma=sigma_in_bins, mode='nearest')
    
    # Find negative to positive transitions
    sign_changes = np.diff(np.sign(smoothed_residuals))
    crossing_indices = np.where(sign_changes > 0)[0]
    
    if len(crossing_indices) == 0:
        return None, smoothed_residuals
    
    # Use leftmost crossing
    return bin_centers[crossing_indices[0]], smoothed_residuals


def calculate_threshold_percentile(mean: float, std: float, percentile: float) -> float:
    """Calculate threshold at given percentile of a normal distribution."""
    return stats.norm.ppf(percentile, loc=mean, scale=std)