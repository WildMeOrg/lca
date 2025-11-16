"""Robust threshold detection using hybrid GMM and tail analysis."""

import numpy as np
from scipy import stats

try:
    from .threshold_utils import (
        fit_robust_gaussian,
        fit_gmm,
        extract_tail_samples,
        calculate_residuals,
        find_residual_zero_crossing,
        calculate_threshold_percentile,
        find_mixture_mode,
        find_largest_component_by_mode,
        check_mode_similarity,
        fit_mirrored_gamma,
        calculate_mirrored_gamma_percentile,
        fit_gamma_from_gaussian_percentile,
        calculate_residuals_gamma,
        extract_tail_samples_gamma
    )
    from .threshold_visualization import create_diagnostic_plot
except ImportError:
    # Fallback to absolute imports for testing
    from threshold_utils import (
        fit_robust_gaussian,
        fit_gmm,
        extract_tail_samples,
        calculate_residuals,
        find_residual_zero_crossing,
        calculate_threshold_percentile,
        find_mixture_mode,
        find_largest_component_by_mode,
        check_mode_similarity,
        fit_mirrored_gamma,
        calculate_mirrored_gamma_percentile,
        fit_gamma_from_gaussian_percentile,
        calculate_residuals_gamma,
        extract_tail_samples_gamma
    )
    from threshold_visualization import create_diagnostic_plot


def find_robust_threshold(data, bins=500, threshold_fraction=0.15,
                         fallback_percentile=85,
                         em_max_iter=200, em_random_state=42,
                         tail_sigma_threshold=3, residual_smoothing_sigma=0.01,
                         mode_tolerance_factor=0.1,
                         main_percentile=0.999,
                         regularization_strength=10.0,
                         debug_plots=False, plot_path="dist.png",
                         print_func=print):
    """
    Robust threshold detection following a clear step-by-step algorithm:
    
    1. Fit GMM to data
    2. Determine main distribution:
       - Find mode of mixture PDF and component with largest weighted mode
       - If mixture mode ≈ largest component mode: use that component
       - Otherwise: robustly fit to all data
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
    mode_tolerance_factor : float
        Tolerance for mode comparison as fraction of standard deviation
    main_percentile : float
        Percentile of Gaussian main distribution to use for Gamma fitting (0.9 = 90%)
    regularization_strength : float
        Strength of regularization for zero-density constraint in target distribution fitting
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
    if data.size == 0:
        return 0.5
    # Step 1: Fit GMM
    gmm_result = fit_gmm(data, n_components=2, max_iter=em_max_iter, 
                         random_state=em_random_state)
    
    # Step 2: Determine main distribution based on GMM separability
    main_mean, main_std, fit_info = fit_main_distribution(
        data, gmm_result, mode_tolerance_factor, print_func
    )
    
    # Steps 3-6: Analyze tail to find target distribution and threshold
    result = analyze_tail(
        data, main_mean, main_std, tail_sigma_threshold,
        residual_smoothing_sigma, threshold_fraction,
        fallback_percentile, main_percentile, regularization_strength, print_func
    )
    
    # Add fit info to result
    result.update(fit_info)
    
    if debug_plots:
        create_diagnostic_plot(data, result, bins, plot_path)
        print_func(f"Diagnostic plot saved to {plot_path}")
    
    threshold = result['threshold']
    print_func(f"Final threshold (p{threshold_fraction:.2f}): {threshold:.4f}")
    return threshold


def fit_main_distribution(data, gmm_result, mode_tolerance_factor, print_func):
    """
    Fit main distribution based on GMM results using mode comparison.
    
    New algorithm:
    1. Find mode of mixture PDF
    2. Find component with largest weighted mode value
    3. Compare mixture mode with mode of largest component
    4. If modes are similar, use largest component; otherwise use robust fit
    
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
    
    # GMM succeeded, extract parameters
    means = gmm_result['means']
    stds = gmm_result['stds'] 
    weights = gmm_result['weights']
    
    print_func(f"  GMM components: [{means[0]:.3f}±{stds[0]:.3f}, w={weights[0]:.2f}] "
               f"[{means[1]:.3f}±{stds[1]:.3f}, w={weights[1]:.2f}]")
    
    # Find mode of mixture PDF
    mixture_mode = find_mixture_mode(means, stds, weights)
    print_func(f"  Mixture PDF mode: {mixture_mode:.3f}")
    
    # Find component with largest weighted mode value
    largest_idx = find_largest_component_by_mode(means, stds, weights)
    component_mode = means[largest_idx]  # For Gaussian, mode = mean
    
    # Calculate weighted mode values for reporting
    weighted_mode_0 = weights[0] * stats.norm.pdf(means[0], loc=means[0], scale=stds[0])
    weighted_mode_1 = weights[1] * stats.norm.pdf(means[1], loc=means[1], scale=stds[1])
    
    print_func(f"  Component weighted modes: [0]={weighted_mode_0:.4f}, [1]={weighted_mode_1:.4f}")
    print_func(f"  Largest component: {largest_idx+1} (mode={component_mode:.3f})")
    
    # Check if modes are similar
    reference_std = stds[largest_idx]
    modes_similar = check_mode_similarity(mixture_mode, component_mode, 
                                         reference_std, mode_tolerance_factor)
    
    tolerance = mode_tolerance_factor * reference_std
    print_func(f"  Mode comparison: |{mixture_mode:.3f} - {component_mode:.3f}| = "
               f"{abs(mixture_mode - component_mode):.3f} {'<' if modes_similar else '>='} "
               f"{tolerance:.3f} ({mode_tolerance_factor}×σ)")
    
    if modes_similar:
        # Modes are similar, use largest component
        print_func("  Modes are similar, using largest component as main distribution")
        main_mean = means[largest_idx]
        main_std = stds[largest_idx]
        
        return main_mean, main_std, {
            'method': 'gmm_component',
            'gmm_success': True,
            'gmm': gmm_result,
            'mixture_mode': mixture_mode,
            'component_used': largest_idx,
            'modes_similar': True,
            'mode_distance': abs(mixture_mode - component_mode),
            'mode_tolerance': tolerance,
            'weighted_modes': [weighted_mode_0, weighted_mode_1],
            'means': means,
            'stds': stds,
            'weights': weights
        }
    else:
        # Modes are not similar, use robust fitting
        print_func("  Modes are not similar, using robust Gaussian fit")
        main_mean, main_std = fit_robust_gaussian(data)
        print_func(f"  Main distribution: mean={main_mean:.3f}, std={main_std:.3f}")
        
        return main_mean, main_std, {
            'method': 'robust',
            'gmm_success': True,
            'gmm': gmm_result,
            'mixture_mode': mixture_mode,
            'largest_component': largest_idx,
            'modes_similar': False,
            'mode_distance': abs(mixture_mode - component_mode),
            'mode_tolerance': tolerance,
            'weighted_modes': [weighted_mode_0, weighted_mode_1],
            'means': means,
            'stds': stds,
            'weights': weights
        }


def analyze_tail(data, main_mean, main_std, tail_sigma_threshold,
                residual_smoothing_sigma, threshold_fraction,
                fallback_percentile, main_percentile, regularization_strength, print_func):
    """
    Analyze tail of distribution to find target component and threshold.
    
    Steps:
    3. Fit Gamma distribution to main distribution data
    4. Extract tail samples using Gamma distribution parameters
    5. Find zero crossing in residuals using Gamma
    6. Fit target distribution to samples beyond zero-crossing
    7. Calculate threshold using percentile of target distribution
    
    Returns:
        dict with threshold and analysis details
    """
    # NEW STEP: Fit Gamma to main distribution
    try:
        gamma_shape, gamma_loc, gamma_scale, gamma_threshold, n_gamma_samples = \
            fit_gamma_from_gaussian_percentile(data, main_mean, main_std, main_percentile)
        
        print_func(f"  Gamma fit to {main_percentile*100:.0f}% of main: "
                   f"shape={gamma_shape:.3f}, loc={gamma_loc:.3f}, scale={gamma_scale:.3f}")
        print_func(f"    Using {n_gamma_samples} samples below {gamma_threshold:.3f}")
        
        # Calculate equivalent mean and std
        gamma_mean = gamma_loc + gamma_shape * gamma_scale
        gamma_std = np.sqrt(gamma_shape) * gamma_scale
        print_func(f"    Gamma equivalent: mean={gamma_mean:.3f}, std={gamma_std:.3f}")
        
        gamma_fit_success = True
    except Exception as e:
        print_func(f"  WARNING: Gamma fitting failed: {e}")
        print_func("  Falling back to Gaussian for tail analysis")
        gamma_fit_success = False
        # Use Gaussian parameters as fallback
        gamma_shape = gamma_loc = gamma_scale = None
        gamma_mean = main_mean
        gamma_std = main_std
    
    # Step 3: Extract tail samples using appropriate distribution
    if gamma_fit_success:
        tail_samples, tail_threshold = extract_tail_samples_gamma(
            data, gamma_shape, gamma_loc, gamma_scale, tail_sigma_threshold
        )
    else:
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
            'tail_analysis_success': False,
            'gamma_fit_success': False
        }
    
    # Step 4: Calculate residuals and find zero crossing
    bins = min(50, len(tail_samples)//5)
    
    if gamma_fit_success:
        bin_centers, residuals, expected_hist, observed_hist = calculate_residuals_gamma(
            tail_samples, gamma_shape, gamma_loc, gamma_scale, tail_threshold, bins
        )
    else:
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
    
    # Step 5: Fit target distribution (mirrored Gamma)
    # Pass main_mean as the point where we want near-zero density
    shape, loc, scale, mirror_point = fit_mirrored_gamma(
        target_samples,
        zero_density_point=main_mean,
        regularization_strength=regularization_strength
    )

    # Calculate mean and std for compatibility
    gamma_mean = loc + shape * scale  # Gamma mean in mirrored space
    gamma_var = shape * scale**2      # Gamma variance
    target_mean = mirror_point - gamma_mean  # Mean in original space
    target_std = np.sqrt(gamma_var)  # Std is same

    print_func(f"    Target distribution (mirrored Gamma): shape={shape:.3f}, loc={loc:.3f}, "
               f"scale={scale:.3f}, mirror={mirror_point:.3f}")
    print_func(f"    Regularized with zero-density at main mode={main_mean:.3f}, strength={regularization_strength:.1f}")
    print_func(f"    Equivalent mean={target_mean:.3f}, std={target_std:.3f}")
    print_func(f"    Using {len(target_samples)} samples beyond cutoff={cutoff_value:.3f}")
    
    # Step 6: Calculate threshold
    threshold = calculate_mirrored_gamma_percentile(shape, loc, scale, mirror_point, threshold_fraction)
    
    result_dict = {
        'threshold': threshold,
        'main_mean': main_mean,
        'main_std': main_std,
        'tail_sigma_threshold': tail_sigma_threshold,
        'tail_threshold': tail_threshold,
        'tail_samples': len(tail_samples),
        'target_distribution': {
            'mean': target_mean, 
            'std': target_std,
            'shape': shape,
            'loc': loc,
            'scale': scale,
            'mirror_point': mirror_point
        },
        'second_component': {'mean': target_mean, 'std': target_std},  # For compatibility
        'bin_centers': bin_centers,
        'residuals': residuals,
        'smoothed_residuals': smoothed_residuals,
        'cutoff_value': cutoff_value,
        'threshold_fraction': threshold_fraction,
        'tail_analysis_success': True,
        'zero_crossing_found': cutoff_value != tail_threshold
    }
    
    # Add Gamma fit information if successful
    if gamma_fit_success:
        result_dict['main_distribution_gamma'] = {
            'shape': gamma_shape,
            'loc': gamma_loc,
            'scale': gamma_scale,
            'percentile_used': main_percentile,
            'threshold': gamma_threshold,
            'n_samples': n_gamma_samples,
            'mean': gamma_mean,
            'std': gamma_std
        }
        result_dict['gamma_fit_success'] = True
    else:
        result_dict['gamma_fit_success'] = False
    
    return result_dict