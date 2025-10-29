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
    return bin_centers[crossing_indices[-1]], smoothed_residuals


def calculate_threshold_percentile(mean: float, std: float, percentile: float) -> float:
    """Calculate threshold at given percentile of a normal distribution."""
    return stats.norm.ppf(percentile, loc=mean, scale=std)


def evaluate_gaussian_pdf(x: float, mean: float, std: float) -> float:
    """Evaluate Gaussian PDF at a given point."""
    return stats.norm.pdf(x, loc=mean, scale=std)


def evaluate_mixture_pdf(x: float | np.ndarray, means: np.ndarray, stds: np.ndarray, 
                        weights: np.ndarray) -> float | np.ndarray:
    """
    Evaluate 2-component Gaussian mixture PDF at given point(s).
    
    Parameters:
    -----------
    x : float or array
        Point(s) at which to evaluate PDF
    means : array of shape (2,)
        Means of the two components
    stds : array of shape (2,)
        Standard deviations of the two components
    weights : array of shape (2,)
        Weights of the two components
        
    Returns:
    --------
    float or array
        PDF value(s) at x
    """
    pdf1 = weights[0] * stats.norm.pdf(x, loc=means[0], scale=stds[0])
    pdf2 = weights[1] * stats.norm.pdf(x, loc=means[1], scale=stds[1])
    return pdf1 + pdf2


def find_mixture_mode(means: np.ndarray, stds: np.ndarray, weights: np.ndarray,
                     n_points: int = 1000) -> float:
    """
    Find mode of 2-component Gaussian mixture using grid search.
    
    Parameters:
    -----------
    means : array of shape (2,)
        Means of the two components
    stds : array of shape (2,)
        Standard deviations of the two components
    weights : array of shape (2,)
        Weights of the two components
    n_points : int
        Number of points for grid search
        
    Returns:
    --------
    float
        Location of mixture mode
    """
    # Define search range: from min_mean - 3*max_std to max_mean + 3*max_std
    x_min = min(means) - 3 * max(stds)
    x_max = max(means) + 3 * max(stds)
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, n_points)
    
    # Evaluate mixture PDF
    pdf_values = evaluate_mixture_pdf(x_grid, means, stds, weights)
    
    # Find maximum
    max_idx = np.argmax(pdf_values)
    return x_grid[max_idx]


def find_largest_component_by_mode(means: np.ndarray, stds: np.ndarray, 
                                  weights: np.ndarray) -> int:
    """
    Find component with largest weighted mode value.
    
    For a Gaussian, the mode is at the mean, so we evaluate w_i * N(μ_i|μ_i,σ_i).
    
    Returns:
    --------
    int
        Index of component with largest weighted mode (0 or 1)
    """
    # For each component, evaluate weighted PDF at its mode (mean)
    weighted_mode_1 = weights[0] * evaluate_gaussian_pdf(means[0], means[0], stds[0])
    weighted_mode_2 = weights[1] * evaluate_gaussian_pdf(means[1], means[1], stds[1])
    
    return 0 if weighted_mode_1 > weighted_mode_2 else 1


def check_mode_similarity(mode1: float, mode2: float, reference_std: float,
                         tolerance_factor: float = 0.1) -> bool:
    """
    Check if two modes are similar within tolerance.
    
    Parameters:
    -----------
    mode1, mode2 : float
        Modes to compare
    reference_std : float
        Reference standard deviation for scaling tolerance
    tolerance_factor : float
        Tolerance as fraction of reference_std
        
    Returns:
    --------
    bool
        True if modes are similar within tolerance
    """
    tolerance = tolerance_factor * reference_std
    return abs(mode1 - mode2) < tolerance


def fit_mirrored_gamma(data: np.ndarray, zero_density_point: float = None,
                      regularization_strength: float = 1.0) -> tuple[float, float, float, float]:
    """
    Fit mirrored Gamma distribution to data with left tail.

    The Gamma distribution naturally has a right tail, so we mirror the data
    to model a left tail distribution. Optionally applies regularization to
    enforce near-zero density at a specified point.

    Parameters:
    -----------
    data : array
        Data values to fit (should have left tail)
    zero_density_point : float, optional
        Point where density should be near zero (e.g., negative distribution mode)
        If None, uses standard unregularized fitting
    regularization_strength : float
        Strength of regularization penalty (higher = stronger constraint)
        Only used if zero_density_point is provided

    Returns:
    --------
    tuple
        (shape, loc, scale, mirror_point) parameters of mirrored Gamma
    """
    # Find mirror point with small buffer for numerical stability
    mirror_point = np.max(data) + 1e-6

    # Mirror the data
    mirrored_data = mirror_point - data

    if zero_density_point is None:
        # Standard unregularized fitting
        shape, _, scale = stats.gamma.fit(mirrored_data, floc=0)
        loc = 0
    else:
        # Regularized fitting with zero-density constraint
        from scipy.optimize import minimize

        # Initial estimate from standard fit
        initial_shape, _, initial_scale = stats.gamma.fit(mirrored_data, floc=0)

        # Transform zero_density_point to mirrored space
        mirrored_zero_point = mirror_point - zero_density_point

        def objective(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return 1e10  # Invalid parameters

            # Negative log-likelihood
            try:
                nll = -np.sum(stats.gamma.logpdf(mirrored_data, shape, loc=0, scale=scale))
            except:
                return 1e10

            # Regularization: penalize density at zero_density_point
            # Only apply penalty if point is in valid range
            if mirrored_zero_point > 0:
                density = stats.gamma.pdf(mirrored_zero_point, shape, loc=0, scale=scale)
                # Use exponential penalty for stronger effect when density is high
                # Also scale by data size to balance with likelihood
                penalty = regularization_strength * len(mirrored_data) * (density ** 4)
            else:
                penalty = 0  # Point is outside distribution support

            return nll + penalty

        # Debug: Check initial density at zero point
        initial_density = stats.gamma.pdf(mirrored_zero_point, initial_shape, loc=0, scale=initial_scale)

        # Optimize
        result = minimize(
            objective,
            x0=[initial_shape, initial_scale],
            method='L-BFGS-B',
            bounds=[(0.1, 100), (0.001, 10)]  # Reasonable bounds for shape and scale
        )

        if result.success:
            shape, scale = result.x
            loc = 0

            # Debug: Check final density at zero point
            final_density = stats.gamma.pdf(mirrored_zero_point, shape, loc=0, scale=scale)
            print(f"    DEBUG: Regularization - initial density at {zero_density_point:.3f}: {initial_density:.6f}, "
                  f"final: {final_density:.6f} (mirrored point: {mirrored_zero_point:.3f})")
        else:
            # Fall back to unregularized fit if optimization fails
            print(f"    WARNING: Regularized optimization failed, using unregularized fit")
            shape, scale = initial_shape, initial_scale
            loc = 0

    return shape, loc, scale, mirror_point


def evaluate_mirrored_gamma_pdf(x: float | np.ndarray, shape: float, loc: float, 
                               scale: float, mirror_point: float) -> float | np.ndarray:
    """
    Evaluate mirrored Gamma PDF at given points.
    
    Parameters:
    -----------
    x : float or array
        Point(s) at which to evaluate PDF
    shape : float
        Shape parameter of Gamma distribution
    loc : float
        Location parameter of Gamma distribution
    scale : float
        Scale parameter of Gamma distribution
    mirror_point : float
        Mirror point used for transformation
        
    Returns:
    --------
    float or array
        PDF value(s) at x
    """
    # Transform x to mirrored space and evaluate
    return stats.gamma.pdf(mirror_point - x, shape, loc=loc, scale=scale)


def calculate_mirrored_gamma_percentile(shape: float, loc: float, scale: float,
                                       mirror_point: float, percentile: float) -> float:
    """
    Calculate percentile of mirrored Gamma distribution.
    
    Parameters:
    -----------
    shape : float
        Shape parameter of Gamma distribution
    loc : float
        Location parameter of Gamma distribution
    scale : float
        Scale parameter of Gamma distribution
    mirror_point : float
        Mirror point used for transformation
    percentile : float
        Percentile to calculate (between 0 and 1)
        
    Returns:
    --------
    float
        Value at the given percentile
    """
    # For mirrored distribution, low percentiles correspond to high percentiles in original
    return mirror_point - stats.gamma.ppf(1 - percentile, shape, loc=loc, scale=scale)


def fit_gamma_from_gaussian_percentile(data: np.ndarray, gaussian_mean: float, 
                                      gaussian_std: float, percentile: float = 0.9) -> tuple:
    """
    Fit Gamma distribution to data below a Gaussian percentile threshold.
    
    Parameters:
    -----------
    data : array
        Data values to fit
    gaussian_mean : float
        Mean of the Gaussian distribution
    gaussian_std : float
        Standard deviation of the Gaussian distribution
    percentile : float
        Percentile of Gaussian to use as threshold (default 0.9)
        
    Returns:
    --------
    tuple
        (shape, loc, scale, threshold_value, n_samples_used)
    """
    # Calculate threshold from Gaussian percentile
    threshold = stats.norm.ppf(percentile, loc=gaussian_mean, scale=gaussian_std)
    
    # Filter data below threshold
    filtered_data = data[data <= threshold]
    
    if len(filtered_data) < 10:
        raise ValueError(f"Insufficient samples ({len(filtered_data)}) below threshold")
    
    # Fit Gamma distribution with loc fixed below minimum
    # Need sufficient buffer for MLE requirements
    data_min = np.min(filtered_data)
    data_range = np.max(filtered_data) - data_min
    loc = data_min - 0.01 * data_range  # 1% of data range as buffer
    shape, _, scale = stats.gamma.fit(filtered_data, floc=loc)
    
    return shape, loc, scale, threshold, len(filtered_data)


def calculate_residuals_gamma(tail_samples: np.ndarray, shape: float, loc: float, 
                             scale: float, tail_threshold: float, bins: int = 50) -> tuple:
    """
    Calculate residuals between observed and expected frequencies in tail using Gamma distribution.
    
    Parameters:
    -----------
    tail_samples : array
        Samples in the tail region
    shape : float
        Shape parameter of Gamma distribution
    loc : float
        Location parameter of Gamma distribution
    scale : float
        Scale parameter of Gamma distribution
    tail_threshold : float
        Threshold defining the tail region
    bins : int
        Number of histogram bins
        
    Returns:
    --------
    tuple
        (bin_centers, residuals, expected_hist, observed_hist)
    """
    # Create histogram
    hist, bin_edges = np.histogram(tail_samples, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate expected frequencies using Gamma distribution
    tail_prob = 1 - stats.gamma.cdf(tail_threshold, shape, loc=loc, scale=scale)
    
    if tail_prob <= 0:
        # Handle edge case where tail_threshold is beyond Gamma support
        expected_hist = np.zeros_like(hist, dtype=float)
        residuals = hist.astype(float)
        return bin_centers, residuals, expected_hist, hist
    
    bin_probs = []
    for i in range(len(bin_edges)-1):
        prob = stats.gamma.cdf(bin_edges[i+1], shape, loc=loc, scale=scale) - \
               stats.gamma.cdf(bin_edges[i], shape, loc=loc, scale=scale)
        conditional_prob = prob / tail_prob
        bin_probs.append(conditional_prob)
    
    expected_hist = np.array(bin_probs) * len(tail_samples)
    residuals = hist - expected_hist
    
    return bin_centers, residuals, expected_hist, hist


def extract_tail_samples_gamma(data: np.ndarray, shape: float, loc: float, 
                              scale: float, sigma_threshold: float = 3.0) -> tuple:
    """
    Extract samples beyond sigma_threshold standard deviations from Gamma distribution.
    
    Parameters:
    -----------
    data : array
        Data values
    shape : float
        Shape parameter of Gamma distribution
    loc : float
        Location parameter of Gamma distribution
    scale : float
        Scale parameter of Gamma distribution
    sigma_threshold : float
        Number of standard deviations for tail threshold
        
    Returns:
    --------
    tuple
        (tail_samples, tail_threshold)
    """
    # Calculate mean and std of Gamma distribution
    gamma_mean = loc + shape * scale
    gamma_std = np.sqrt(shape) * scale
    
    # Calculate threshold
    tail_threshold = gamma_mean + sigma_threshold * gamma_std
    
    # Extract tail samples
    tail_samples = data[data > tail_threshold]
    
    return tail_samples, tail_threshold