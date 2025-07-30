"""Robust threshold detection using hybrid GMM and tail analysis."""

import numpy as np
from scipy import stats

try:
    from .threshold_utils import (
        fit_robust_gaussian,
        fit_gmm,
        check_gmm_separability,
        extract_tail_samples,
        calculate_residuals,
        find_residual_zero_crossing,
        calculate_threshold_percentile
    )
    from .threshold_visualization import create_diagnostic_plot
except ImportError:
    # Fallback to absolute imports for testing
    from threshold_utils import (
        fit_robust_gaussian,
        fit_gmm,
        check_gmm_separability,
        extract_tail_samples,
        calculate_residuals,
        find_residual_zero_crossing,
        calculate_threshold_percentile
    )
    from threshold_visualization import create_diagnostic_plot


def find_robust_threshold(data, bins=500, threshold_fraction=0.15, 
                         failure_threshold=2.0, fallback_percentile=85,
                         em_max_iter=200, em_random_state=42,
                         tail_sigma_threshold=3.0, residual_smoothing_sigma=0.01,
                         debug_plots=False, plot_path="dist.png", 
                         print_func=print):
    """
    Robust threshold detection following a clear step-by-step algorithm:
    
    1. Fit GMM to data
    2. Determine main distribution:
       - If GMM separable: use bigger GMM component directly
       - If not separable: robustly fit to all data
    3. Extract tail samples using main distribution
    4. Find zero crossing in residuals
    5. Fit target distribution to samples beyond zero-crossing
    6. Calculate threshold using percentile of target distribution
    
    Parameters:
    -----------
    data : array-like
        Input data values
    bins : int
        Number of histogram bins for visualization
    threshold_fraction : float
        Percentile of target distribution for final threshold (0.15 = 15th percentile)
    failure_threshold : float
        GMM separability threshold (peaks must be failure_threshold×σ_small apart)
    fallback_percentile : int
        Percentile to use if algorithm fails completely
    em_max_iter : int
        Maximum EM iterations
    em_random_state : int
        Random seed for EM
    tail_sigma_threshold : float
        Number of standard deviations beyond main peak to define tail region
    residual_smoothing_sigma : float
        Gaussian smoothing sigma in data units for residual analysis
    debug_plots : bool
        Whether to save diagnostic plots
    plot_path : str
        Path to save diagnostic plots
    print_func : callable
        Print function (use logger if needed)
        
    Returns:
    --------
    float
        Threshold at specified percentile of target distribution
    """
    data = np.array(data).flatten()
    print_func(f"Analyzing {len(data)} data points")
    
    # Step 1: Fit GMM
    gmm_result = fit_gmm(data, n_components=2, max_iter=em_max_iter, 
                         random_state=em_random_state)
    
    # Step 2: Determine main distribution based on GMM separability
    main_mean, main_std, fit_info = fit_main_distribution(
        data, gmm_result, failure_threshold, print_func
    )
    
    # Steps 3-6: Analyze tail to find target distribution and threshold
    result = analyze_tail(
        data, main_mean, main_std, tail_sigma_threshold, 
        residual_smoothing_sigma, threshold_fraction, 
        fallback_percentile, print_func
    )
    
    # Add fit info to result
    result.update(fit_info)
    
    if debug_plots:
        create_diagnostic_plot(data, result, bins, plot_path)
    
    threshold = result['threshold']
    print_func(f"Final threshold (p{threshold_fraction:.2f}): {threshold:.4f}")
    return threshold


def fit_main_distribution(data, gmm_result, failure_threshold, print_func):
    """
    Fit main distribution based on GMM results.
    
    Returns:
        (main_mean, main_std, fit_info) where fit_info contains method details
    """
    if not gmm_result['success']:
        # GMM failed, use robust fitting on all data
        print_func("  GMM fitting failed, using robust Gaussian fit")
        main_mean, main_std = fit_robust_gaussian(data)
        print_func(f"  Main distribution: mean={main_mean:.3f}, std={main_std:.3f}")
        
        return main_mean, main_std, {
            'method': 'robust',
            'gmm_success': False
        }
    
    # GMM succeeded, check separability
    means = gmm_result['means']
    stds = gmm_result['stds'] 
    weights = gmm_result['weights']
    
    print_func(f"  GMM components: [{means[0]:.3f}±{stds[0]:.3f}, w={weights[0]:.2f}] "
               f"[{means[1]:.3f}±{stds[1]:.3f}, w={weights[1]:.2f}]")
    
    # Check separability
    is_separable, sep_info = check_gmm_separability(
        means[0], stds[0], weights[0],
        means[1], stds[1], weights[1],
        failure_threshold
    )
    
    print_func(f"  Separability: distance={sep_info['separation_distance']:.3f} "
               f"{'>==' if is_separable else '<'} {sep_info['required_separation']:.3f} "
               f"({failure_threshold}×σ_small)")
    
    if not is_separable:
        # Not separable, use robust fitting on all data
        print_func("  GMM not separable, using robust Gaussian fit")
        main_mean, main_std = fit_robust_gaussian(data)
        print_func(f"  Main distribution: mean={main_mean:.3f}, std={main_std:.3f}")
        
        return main_mean, main_std, {
            'method': 'robust',
            'gmm_success': True,
            'gmm_separable': False,
            'gmm': gmm_result,
            'separability_info': sep_info
        }
    
    # GMM is separable, use bigger component directly
    print_func("  GMM separable, using bigger component as main distribution")
    
    # Identify bigger component by weight
    bigger_idx = 0 if weights[0] > weights[1] else 1
    main_mean = means[bigger_idx]
    main_std = stds[bigger_idx]
    
    print_func(f"  Main distribution (GMM component {bigger_idx+1}): "
               f"mean={main_mean:.3f}, std={main_std:.3f}")
    
    return main_mean, main_std, {
        'method': 'gmm_component',
        'gmm_success': True,
        'gmm_separable': True,
        'gmm': gmm_result,
        'separability_info': sep_info,
        'component_used': bigger_idx,
        'means': means,
        'stds': stds,
        'weights': weights
    }


def analyze_tail(data, main_mean, main_std, tail_sigma_threshold,
                residual_smoothing_sigma, threshold_fraction, 
                fallback_percentile, print_func):
    """
    Analyze tail of distribution to find target component and threshold.
    
    Steps:
    3. Extract tail samples using main distribution parameters
    4. Find zero crossing in residuals
    5. Fit target distribution to samples beyond zero-crossing
    6. Calculate threshold using percentile of target distribution
    
    Returns:
        dict with threshold and analysis details
    """
    # Step 3: Extract tail samples
    tail_samples, tail_threshold = extract_tail_samples(
        data, main_mean, main_std, tail_sigma_threshold
    )
    print_func(f"  Tail samples beyond {tail_threshold:.3f} ({tail_sigma_threshold}σ): "
               f"{len(tail_samples)} ({len(tail_samples)/len(data):.1%})")
    
    if len(tail_samples) < 20:
        print_func("  Insufficient tail samples, using fallback percentile")
        return {
            'threshold': np.percentile(data, fallback_percentile),
            'main_mean': main_mean,
            'main_std': main_std,
            'tail_analysis_success': False
        }
    
    # Step 4: Calculate residuals and find zero crossing
    bins = min(50, len(tail_samples)//5)
    bin_centers, residuals, expected_hist, observed_hist = calculate_residuals(
        tail_samples, main_mean, main_std, tail_threshold, bins
    )
    
    print_func(f"    Expected hist: total={np.sum(expected_hist):.1f}, "
               f"max={np.max(expected_hist):.3f}")
    print_func(f"    Observed hist: total={np.sum(observed_hist):.1f}, "
               f"max={np.max(observed_hist):.1f}")
    print_func(f"    Residuals: max={np.max(residuals):.1f}, "
               f"min={np.min(residuals):.1f}")
    
    # Find zero-crossing
    bin_width = bin_centers[1] - bin_centers[0]
    cutoff_value, smoothed_residuals = find_residual_zero_crossing(
        residuals, bin_centers, residual_smoothing_sigma, bin_width
    )
    
    if cutoff_value is None:
        print_func("    No zero-crossing found, using tail threshold")
        cutoff_value = tail_threshold
        target_samples = tail_samples
    else:
        print_func(f"    Found zero-crossing at {cutoff_value:.3f}")
        target_samples = tail_samples[tail_samples > cutoff_value]
        
        if len(target_samples) < 10:
            print_func(f"    Insufficient samples beyond cutoff ({len(target_samples)}), "
                      "using all tail samples")
            target_samples = tail_samples
            cutoff_value = tail_threshold
    
    # Step 5: Fit target distribution
    target_mean, target_std = stats.norm.fit(target_samples)
    print_func(f"    Target distribution: mean={target_mean:.3f}, std={target_std:.3f}")
    print_func(f"    Using {len(target_samples)} samples beyond cutoff={cutoff_value:.3f}")
    
    # Step 6: Calculate threshold
    threshold = calculate_threshold_percentile(target_mean, target_std, threshold_fraction)
    
    return {
        'threshold': threshold,
        'main_mean': main_mean,
        'main_std': main_std,
        'tail_threshold': tail_threshold,
        'tail_samples': len(tail_samples),
        'target_distribution': {'mean': target_mean, 'std': target_std},
        'second_component': {'mean': target_mean, 'std': target_std},  # For compatibility
        'bin_centers': bin_centers,
        'residuals': residuals,
        'smoothed_residuals': smoothed_residuals,
        'cutoff_value': cutoff_value,
        'threshold_fraction': threshold_fraction,
        'tail_analysis_success': True,
        'zero_crossing_found': cutoff_value != tail_threshold
    }