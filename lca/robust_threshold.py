import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt

def find_robust_threshold(data, bins=500, threshold_fraction=0.15, 
                         failure_threshold=2.0, fallback_percentile=85,
                         em_max_iter=200, em_random_state=42,
                         tail_sigma_threshold=3.0, residual_smoothing_sigma=0.01,
                         debug_plots=False, plot_path="dist.png", 
                         print_func=print):
    """
    Robust threshold detection using hybrid EM + direct tail fitting approach.
    
    Parameters:
    -----------
    data : array-like
        Input data values
    bins : int
        Number of histogram bins
    threshold_fraction : float
        Percentile of second peak Gaussian (0.5 = center, 0.25 = left, 0.75 = right)
    failure_threshold : float
        EM separability threshold (peaks must be failure_threshold×σ_small apart)
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
        Threshold at specified percentile of second peak distribution
    """
    
    data = np.array(data).flatten()
    print_func(f"Analyzing {len(data)} data points")
    
    # Try EM first
    em_result = _try_em_method(data, threshold_fraction, failure_threshold, 
                              em_max_iter, em_random_state, print_func)
    
    if em_result['success']:
        print_func("EM succeeded")
        method_used, result = 'EM', em_result
        if debug_plots:
            _plot_em_results(data, em_result, bins, plot_path)
    else:
        print_func("EM failed, using direct tail fitting")
        result = _direct_tail_method(data, threshold_fraction, fallback_percentile, 
                                   tail_sigma_threshold, residual_smoothing_sigma, print_func)
        method_used = 'Direct Tail'
        if debug_plots:
            _plot_tail_results(data, result, bins, plot_path)
    
    threshold = result['threshold']
    print_func(f"Final result ({method_used}, p{threshold_fraction:.2f}): {threshold:.4f}")
    return threshold


def _try_em_method(data, threshold_fraction, failure_threshold, max_iter, random_state, print_func):
    """Try EM mixture model and detect if it succeeded"""
    
    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full', 
                             max_iter=max_iter, random_state=random_state, init_params='kmeans')
        gmm.fit(data.reshape(-1, 1))
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_
        
        sort_idx = np.argsort(means)
        mean1, mean2 = means[sort_idx]
        std1, std2 = stds[sort_idx]
        weight1, weight2 = weights[sort_idx]
        
        print_func(f"  Components: [{mean1:.3f}±{std1:.3f}, w={weight1:.2f}] [{mean2:.3f}±{std2:.3f}, w={weight2:.2f}]")
        
        # Check EM success using separability criterion
        success, separability_info = _check_em_success(mean1, std1, weight1, mean2, std2, weight2, 
                                                       failure_threshold, print_func)
        
        if success:
            threshold = stats.norm.ppf(threshold_fraction, loc=mean2, scale=std2)
            print_func(f"  Threshold: p{threshold_fraction:.2f} of second peak = {threshold:.4f}")
            
            return {
                'success': True, 'threshold': threshold, 'gmm': gmm,
                'means': [mean1, mean2], 'stds': [std1, std2], 'weights': [weight1, weight2],
                'separability_info': separability_info, 'threshold_fraction': threshold_fraction
            }
        else:
            print_func(f"  EM failed: {separability_info['reason']}")
            return {'success': False}
            
    except Exception as e:
        print_func(f"  EM fitting error: {e}")
        return {'success': False}


def _check_em_success(mean1, std1, weight1, mean2, std2, weight2, failure_threshold, print_func):
    """Check if EM succeeded using peak separability criterion"""
    
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
    
    print_func(f"  Separability: distance={separation_distance:.3f} {'>==' if success else '<'} {required_separation:.3f} ({failure_threshold}×σ_small)")
    
    return success, {
        'separation_distance': separation_distance,
        'required_separation': required_separation,
        'smaller_std': smaller_std,
        'separability_ratio': separation_distance / required_separation,
        'reason': f"separation {separation_distance:.3f} < {required_separation:.3f}" if not success else ""
    }


def _direct_tail_method(data, threshold_fraction, fallback_percentile, 
                       tail_sigma_threshold, residual_smoothing_sigma, print_func):
    """Direct tail fitting: robust main fit + residual analysis on tail samples"""
    
    # Stage 1: Robust Gaussian fit to identify main component
    main_fit = _fit_robust_gaussian(data, print_func)
    if main_fit is None:
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}
    
    main_mean, main_std = main_fit
    
    # Stage 2: Extract tail samples using configurable threshold
    tail_threshold = main_mean + tail_sigma_threshold * main_std
    tail_samples = data[data > tail_threshold]
    
    print_func(f"  Main component: mean={main_mean:.3f}, std={main_std:.3f}")
    print_func(f"  Tail samples beyond {tail_threshold:.3f} ({tail_sigma_threshold}σ): {len(tail_samples)} ({len(tail_samples)/len(data):.1%})")
    
    if len(tail_samples) < 20:
        print_func("  Insufficient tail samples")
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}
    
    # Stage 3: Calculate residuals for tail samples
    try:
        residual_result = _find_threshold_via_residuals(tail_samples, main_mean, main_std, 
                                                       tail_threshold, threshold_fraction, 
                                                       residual_smoothing_sigma, print_func)
        if residual_result is None:
            # Fallback to simple tail fitting
            tail_mean, tail_std = stats.norm.fit(tail_samples)
            threshold = stats.norm.ppf(threshold_fraction, loc=tail_mean, scale=tail_std)
            print_func(f"  Fallback: simple tail fit: mean={tail_mean:.3f}, std={tail_std:.3f}")
            
            return {
                'threshold': threshold, 'success': True, 
                'main_mean': main_mean, 'main_std': main_std,
                'tail_threshold': tail_threshold, 'tail_samples': len(tail_samples),
                'sigma_threshold': tail_sigma_threshold, 'crossing_success': False,
                'second_component': {'mean': tail_mean, 'std': tail_std},
                'threshold_fraction': threshold_fraction
            }
        else:
            threshold, residuals, smoothed_residuals, cutoff_value, second_component = residual_result
            print_func(f"  Residual analysis: second component at {second_component['mean']:.3f}±{second_component['std']:.3f}")
        
        print_func(f"  Threshold: p{threshold_fraction:.2f} of second component = {threshold:.4f}")
        
        return {
            'threshold': threshold, 'success': True, 
            'main_mean': main_mean, 'main_std': main_std,
            'tail_threshold': tail_threshold, 'tail_samples': len(tail_samples),
            'sigma_threshold': tail_sigma_threshold, 'crossing_success': residual_result is not None,
            'tail_samples_data': tail_samples, 'residuals': residuals, 'smoothed_residuals': smoothed_residuals,
            'cutoff_value': cutoff_value, 'second_component': second_component, 'threshold_fraction': threshold_fraction
        }
        
    except Exception as e:
        print_func(f"  Tail residual fitting failed: {e}")
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}


def _find_threshold_via_residuals(tail_samples, main_mean, main_std, tail_threshold, 
                                 threshold_fraction, residual_smoothing_sigma, print_func):
    """Calculate residuals for tail samples and find cutoff using smoothed zero-crossing"""
    
    # Create histogram of tail samples
    bins = min(50, len(tail_samples)//5)
    hist, bin_edges = np.histogram(tail_samples, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate conditional probabilities given we're in the tail region
    # P(bin | in tail) = P(bin AND in tail) / P(in tail)
    tail_prob = 1 - stats.norm.cdf(tail_threshold, main_mean, main_std)
    
    bin_probs = []
    for i in range(len(bin_edges)-1):
        # Probability of landing in bin [bin_edges[i], bin_edges[i+1]]
        prob = stats.norm.cdf(bin_edges[i+1], main_mean, main_std) - stats.norm.cdf(bin_edges[i], main_mean, main_std)
        # Convert to conditional probability given in tail
        conditional_prob = prob / tail_prob if tail_prob > 0 else 0
        bin_probs.append(conditional_prob)
    
    expected_hist = np.array(bin_probs) * len(tail_samples)
    
    print_func(f"    Expected hist: total={np.sum(expected_hist):.1f}, max={np.max(expected_hist):.3f}")
    print_func(f"    Observed hist: total={np.sum(hist):.1f}, max={np.max(hist):.1f}")
    
    # Calculate residuals
    residuals = hist - expected_hist
    
    print_func(f"    Residuals: max={np.max(residuals):.1f}, min={np.min(residuals):.1f}")
    
    # Smooth residuals using Gaussian filter
    # Convert sigma from data units to bin units
    sigma_in_bins = residual_smoothing_sigma / bin_width
    
    from scipy.ndimage import gaussian_filter1d
    smoothed_residuals = gaussian_filter1d(residuals, sigma=sigma_in_bins, mode='nearest')
    
    # Find zero-crossing points in smoothed residuals
    # Look for points where residual changes from negative to positive
    sign_changes = np.diff(np.sign(smoothed_residuals))
    crossing_indices = np.where(sign_changes > 0)[0]  # Negative to positive transitions
    
    if len(crossing_indices) == 0:
        print_func("    No zero-crossing found in smoothed residuals, using original tail threshold")
        cutoff_value = tail_threshold
    else:
        # Use leftmost crossing
        cutoff_idx = crossing_indices[0]
        cutoff_value = bin_centers[cutoff_idx]
        print_func(f"    Found {len(crossing_indices)} crossing(s), using leftmost at {cutoff_value:.3f}")
    
    # Select samples beyond cutoff
    clean_tail_samples = tail_samples[tail_samples > cutoff_value]
    
    if len(clean_tail_samples) < 10:
        print_func(f"    Insufficient samples beyond cutoff ({len(clean_tail_samples)}), using all tail samples")
        clean_tail_samples = tail_samples
        cutoff_value = tail_threshold
    
    # Fit single Gaussian to clean tail samples
    try:
        second_mean, second_std = stats.norm.fit(clean_tail_samples)
        second_component = {
            'mean': second_mean,
            'std': second_std
        }
        
        threshold = stats.norm.ppf(threshold_fraction, loc=second_mean, scale=second_std)
        
        print_func(f"    Fitted second component: mean={second_mean:.3f}, std={second_std:.3f}")
        print_func(f"    Using {len(clean_tail_samples)} samples beyond cutoff={cutoff_value:.3f}")
        
        return threshold, residuals, smoothed_residuals, cutoff_value, second_component
        
    except Exception as e:
        print_func(f"    Tail fitting error: {e}")
        return None


def _fit_robust_gaussian(data, print_func):
    """Robust Gaussian fitting using iterative σ-clipping"""
    
    current_data = data.copy()
    iteration = 0
    max_iterations = 10
    sigma_threshold = 3.0
    
    while iteration < max_iterations:
        # Fit Gaussian to current data
        mu, sigma = stats.norm.fit(current_data)
        
        # Identify outliers beyond sigma_threshold
        distances = np.abs(current_data - mu) / sigma
        inliers = distances <= sigma_threshold
        
        # Check convergence (no more outliers removed)
        if np.all(inliers):
            break
            
        # Remove outliers for next iteration
        current_data = current_data[inliers]
        iteration += 1
        
        # Safety check - don't remove too much data
        if len(current_data) < len(data) * 0.3:
            print_func(f"  σ-clipping removed too much data, stopping at iteration {iteration}")
            break
    
    # Final fit
    mu_final, sigma_final = stats.norm.fit(current_data)
    print_func(f"  Robust Gaussian: mean={mu_final:.3f}, std={sigma_final:.3f} (iterations: {iteration})")
    
    return mu_final, sigma_final


def _plot_em_results(data, em_result, bins, plot_path):
    """Plot EM results"""
    
    plt.figure(figsize=(12, 8))
    
    mean1, mean2 = em_result['means']
    std1, std2 = em_result['stds']
    weight1, weight2 = em_result['weights']
    threshold = em_result['threshold']
    separability_info = em_result['separability_info']
    threshold_fraction = em_result['threshold_fraction']
    
    x_plot = np.linspace(data.min(), data.max(), 1000)
    
    # Plot 1: Mixture fit
    plt.subplot(2, 2, 1)
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='lightblue', label='Data')
    
    comp1_pdf = weight1 * stats.norm.pdf(x_plot, mean1, std1)
    comp2_pdf = weight2 * stats.norm.pdf(x_plot, mean2, std2)
    mixture_pdf = comp1_pdf + comp2_pdf
    
    plt.plot(x_plot, comp1_pdf, 'r--', alpha=0.8, label='Component 1')
    plt.plot(x_plot, comp2_pdf, 'g--', alpha=0.8, label='Component 2')
    plt.plot(x_plot, mixture_pdf, 'b-', linewidth=2, label='Mixture')
    plt.axvline(threshold, color='purple', linewidth=2, label=f'Threshold (p{threshold_fraction:.2f})')
    plt.legend()
    plt.title('EM SUCCESS: Mixture Model')
    
    # Plot 2: Separability analysis
    plt.subplot(2, 2, 2)
    
    # Show the two peaks and their separation
    x_range = np.linspace(data.min(), data.max(), 1000)
    comp1_pdf = weight1 * stats.norm.pdf(x_range, mean1, std1)
    comp2_pdf = weight2 * stats.norm.pdf(x_range, mean2, std2)
    
    plt.plot(x_range, comp1_pdf, 'r-', linewidth=2, label=f'Component 1 (w={weight1:.2f})')
    plt.plot(x_range, comp2_pdf, 'g-', linewidth=2, label=f'Component 2 (w={weight2:.2f})')
    
    # Show separation distance
    plt.axvline(mean1, color='red', linestyle='--', alpha=0.7)
    plt.axvline(mean2, color='green', linestyle='--', alpha=0.7)
    plt.plot([mean1, mean2], [0, 0], 'k-', linewidth=3, alpha=0.7)
    
    # Add separability info
    sep_distance = separability_info['separation_distance']
    req_separation = separability_info['required_separation']
    plt.text(0.05, 0.95, f'Separation: {sep_distance:.3f}', transform=plt.gca().transAxes, va='top')
    plt.text(0.05, 0.85, f'Required: {req_separation:.3f}', transform=plt.gca().transAxes, va='top')
    plt.text(0.05, 0.75, f'Ratio: {separability_info["separability_ratio"]:.2f}', transform=plt.gca().transAxes, va='top')
    
    plt.legend()
    plt.title('Separability Analysis')
    
    # Plot 3: Component responsibilities
    plt.subplot(2, 2, 3)
    responsibilities = em_result['gmm'].predict_proba(x_plot.reshape(-1, 1))
    sort_idx = np.argsort(em_result['means'])
    plt.plot(x_plot, responsibilities[:, sort_idx[0]], 'r-', label='P(Component 1)')
    plt.plot(x_plot, responsibilities[:, sort_idx[1]], 'g-', label='P(Component 2)')
    plt.axvline(threshold, color='purple', linewidth=2, label='Threshold')
    plt.legend()
    plt.title('Component Responsibilities')
    
    # Plot 4: Diagnostics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, 'EM SUCCESS', fontweight='bold', color='green', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Converged: {em_result["gmm"].converged_}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Method: p{threshold_fraction:.2f} of peak 2', transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Threshold: {threshold:.4f}', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Diagnostics')
    
    plt.tight_layout()
    plt.savefig(plot_path)


def _plot_tail_results(data, result, bins, plot_path):
    """Plot direct tail fitting with GMM results"""
    
    plt.figure(figsize=(12, 8))
    
    main_mean = result['main_mean']
    main_std = result['main_std']
    tail_threshold = result['tail_threshold']
    threshold = result['threshold']
    threshold_fraction = result['threshold_fraction']
    second_component = result['second_component']
    sigma_threshold = result['sigma_threshold']
    
    x_plot = np.linspace(data.min(), data.max(), 1000)
    
    # Plot 1: Data with main component fit
    plt.subplot(2, 2, 1)
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='lightblue', label='Data')
    
    main_pdf = stats.norm.pdf(x_plot, main_mean, main_std)
    plt.plot(x_plot, main_pdf, 'r-', linewidth=2, label='Main component')
    plt.axvline(main_mean, color='red', linestyle='--', alpha=0.7, label='Main peak')
    plt.axvline(tail_threshold, color='orange', linestyle=':', alpha=0.7, label=f'{sigma_threshold}σ boundary')
    plt.legend()
    plt.title('DIRECT TAIL: Main Component')
    
    # Plot 2: Tail samples, residuals, and smoothed residuals
    plt.subplot(2, 2, 2)
    tail_threshold = result['tail_threshold']
    
    if 'tail_samples_data' in result:
        tail_samples = result['tail_samples_data']
        bins_tail = min(50, len(tail_samples)//5)
        
        # Show original tail samples histogram
        hist_counts, bin_edges = np.histogram(tail_samples, bins=bins_tail)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.hist(tail_samples, bins=bins_tail, density=False, alpha=0.4, color='lightblue', label='Tail samples')
        
        # Show residuals if available
        if 'residuals' in result:
            residuals = result['residuals']
            
            # Plot original residuals
            plt.plot(bin_centers, residuals, 'r-', linewidth=2, alpha=0.7, label='Residuals')
            plt.axhline(0, color='black', linestyle='-', alpha=0.3)
            
            # Plot smoothed residuals if available
            if 'smoothed_residuals' in result:
                smoothed_residuals = result['smoothed_residuals']
                plt.plot(bin_centers, smoothed_residuals, 'b-', linewidth=2, label='Smoothed residuals')
                
                # Mark zero-crossing/cutoff point
                if 'cutoff_value' in result:
                    cutoff_value = result['cutoff_value']
                    plt.axvline(cutoff_value, color='orange', linewidth=2, linestyle='--', 
                               label=f'Cutoff ({cutoff_value:.3f})')
            
            # Show fitted second component
            if 'second_component' in result:
                second_comp = result['second_component']
                x_range = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
                
                # Scale the PDF to match histogram counts
                bin_width = bin_edges[1] - bin_edges[0]
                # Find samples beyond cutoff for scaling
                cutoff_val = result.get('cutoff_value', tail_threshold)
                n_samples_used = np.sum(tail_samples > cutoff_val)
                
                second_pdf = stats.norm.pdf(x_range, second_comp['mean'], second_comp['std'])
                second_pdf_scaled = second_pdf * n_samples_used * bin_width
                
                plt.plot(x_range, second_pdf_scaled, 'g-', linewidth=2, label='Fitted component')
                plt.axvline(second_comp['mean'], color='green', linestyle='--', alpha=0.7, 
                           label=f"Second peak ({second_comp['mean']:.3f})")
        
        # Mark threshold
        threshold = result['threshold']
        threshold_fraction = result['threshold_fraction']
        plt.axvline(threshold, color='purple', linewidth=3, label=f'Threshold (p{threshold_fraction:.2f})')
    
    # Position legend outside plot area to avoid covering data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title('Residual Analysis & Zero-Crossing')
    
    # Plot 3: Combined view
    plt.subplot(2, 2, 3)
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='lightblue', label='Data')
    
    main_pdf = stats.norm.pdf(x_plot, main_mean, main_std)
    plt.plot(x_plot, main_pdf, 'r-', linewidth=2, alpha=0.8, label='Main component')
    
    # Show second component in context
    second_pdf = stats.norm.pdf(x_plot, second_component['mean'], second_component['std'])
    # Scale second component by approximate weight
    tail_weight = len(tail_samples) / len(data)
    # Use simple tail proportion since we're no longer using GMM
    second_pdf_scaled = second_pdf * tail_weight * 0.5  # rough estimate
    
    plt.plot(x_plot, second_pdf_scaled, 'g-', linewidth=2, alpha=0.8, label='Second component (scaled)')
    
    plt.axvline(threshold, color='purple', linewidth=2, label=f'Threshold (p{threshold_fraction:.2f})')
    plt.legend()
    plt.title('Combined Model with Residual Analysis')
    
    # Plot 4: Diagnostics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, 'DIRECT TAIL SUCCESS', fontweight='bold', color='blue', transform=plt.gca().transAxes)
    plt.text(0.1, 0.75, f'Main: μ={main_mean:.3f}, σ={main_std:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Second: μ={second_component["mean"]:.3f}, σ={second_component["std"]:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.45, f'Tail samples ({sigma_threshold}σ): {result["tail_samples"]} ({result["tail_samples"]/len(data):.1%})', transform=plt.gca().transAxes)
    
    # Add residual analysis info
    if result.get('crossing_success', False):
        plt.text(0.1, 0.3, f'Zero-crossing: SUCCESS', transform=plt.gca().transAxes)
        if 'cutoff_value' in result:
            cutoff_samples = np.sum(result['tail_samples_data'] > result['cutoff_value'])
            plt.text(0.1, 0.15, f'Cutoff: {result["cutoff_value"]:.3f} ({cutoff_samples} samples)', transform=plt.gca().transAxes)
    else:
        plt.text(0.1, 0.3, f'Zero-crossing: FAILED (simple fit)', transform=plt.gca().transAxes)
    plt.text(0.1, 0.0, f'Threshold: {threshold:.4f}', transform=plt.gca().transAxes)
    
    plt.axis('off')
    plt.title('Diagnostics')
    
    plt.tight_layout()
    plt.savefig(plot_path)