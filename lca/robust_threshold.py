import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt

def find_robust_threshold(data, bins=500, threshold_fraction=0.15, 
                         failure_threshold=0.3, fallback_percentile=85,
                         em_max_iter=200, em_random_state=42,
                         debug_plots=True, plot_path="dist.png", 
                         print_func=print):
    """
    Robust threshold detection using hybrid EM + residual approach.
    
    Parameters:
    -----------
    data : array-like
        Input data values
    bins : int
        Number of histogram bins
    threshold_fraction : float
        Percentile of second peak Gaussian (0.5 = center, 0.25 = left, 0.75 = right)
    failure_threshold : float
        EM failure detection threshold (intersection PDF ratio)
    fallback_percentile : int
        Percentile to use if algorithm fails completely
    em_max_iter : int
        Maximum EM iterations
    em_random_state : int
        Random seed for EM
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
        print_func("EM failed, using residual method")
        result = _residual_method(data, threshold_fraction, bins, 
                                 fallback_percentile, em_max_iter, em_random_state, print_func)
        method_used = 'Residual'
        if debug_plots:
            _plot_residual_results(data, result, bins, plot_path)
    
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
        
        # Check EM success using intersection point PDF
        success, intersection_info = _check_em_success(mean1, std1, weight1, mean2, std2, weight2, 
                                                      failure_threshold, print_func)
        
        if success:
            threshold = stats.norm.ppf(threshold_fraction, loc=mean2, scale=std2)
            print_func(f"  Threshold: p{threshold_fraction:.2f} of second peak = {threshold:.4f}")
            
            return {
                'success': True, 'threshold': threshold, 'gmm': gmm,
                'means': [mean1, mean2], 'stds': [std1, std2], 'weights': [weight1, weight2],
                'intersection_info': intersection_info, 'threshold_fraction': threshold_fraction
            }
        else:
            print_func(f"  EM failed: {intersection_info['reason']}")
            return {'success': False}
            
    except Exception as e:
        print_func(f"  EM fitting error: {e}")
        return {'success': False}


def _check_em_success(mean1, std1, weight1, mean2, std2, weight2, failure_threshold, print_func):
    """Check if EM succeeded using intersection point PDF criterion"""
    
    x_range = np.linspace(mean1, mean2, 1000)
    pdf1 = weight1 * stats.norm.pdf(x_range, mean1, std1)
    pdf2 = weight2 * stats.norm.pdf(x_range, mean2, std2)
    mixture_pdf = pdf1 + pdf2
    
    intersection_idx = np.argmin(mixture_pdf)
    intersection_point = x_range[intersection_idx]
    intersection_pdf = mixture_pdf[intersection_idx]
    
    max_pdf = max(weight1 * stats.norm.pdf(mean1, mean1, std1),
                  weight2 * stats.norm.pdf(mean2, mean2, std2))
    
    failure_ratio = intersection_pdf / max_pdf
    success = failure_ratio <= failure_threshold
    
    print_func(f"  Intersection analysis: ratio={failure_ratio:.3f} {'<=' if success else '>'} {failure_threshold}")
    
    return success, {
        'intersection_point': intersection_point, 'intersection_pdf': intersection_pdf,
        'max_pdf': max_pdf, 'failure_ratio': failure_ratio, 'failure_threshold': failure_threshold,
        'reason': f"failure_ratio {failure_ratio:.3f} > {failure_threshold}" if not success else ""
    }


def _residual_method(data, threshold_fraction, bins, fallback_percentile, max_iter, random_state, print_func):
    """Residual-based method: single fit + EM on residuals"""
    
    # Try Gamma first, fallback to Gaussian
    single_fit, hist, bin_centers, expected_hist = _fit_single_distribution(data, bins, print_func)
    
    if single_fit is None:
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}
    
    # Calculate residuals and shift to non-negative
    residuals = hist - expected_hist
    residual_shift = np.min(residuals)
    shifted_residuals = residuals - residual_shift
    
    print_func(f"  Residuals: max_pos={np.max(residuals[residuals > 0]):.1f}, shift={residual_shift:.1f}")
    
    if np.max(shifted_residuals) < 1:
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}
    
    # Apply EM to shifted residuals
    second_peak, residual_gmm, em_success = _fit_residuals_em(shifted_residuals, bin_centers, 
                                                             max_iter, random_state, print_func)
    
    if second_peak is None:
        return {'threshold': np.percentile(data, fallback_percentile), 'success': False}
    
    # Calculate threshold as percentile of second peak
    threshold = stats.norm.ppf(threshold_fraction, loc=second_peak['mean'], scale=second_peak['std'])
    print_func(f"  Threshold: p{threshold_fraction:.2f} of second peak = {threshold:.4f}")
    
    return {
        'threshold': threshold, 'success': True, 'single_fit': single_fit,
        'second_peak': second_peak, 'hist': hist, 'bin_centers': bin_centers,
        'expected_hist': expected_hist, 'residuals': residuals, 'shifted_residuals': shifted_residuals,
        'residual_shift': residual_shift, 'residual_em_success': em_success,
        'residual_gmm': residual_gmm, 'threshold_fraction': threshold_fraction
    }


def _fit_single_distribution(data, bins, print_func):
    """Fit single Gamma or Gaussian to data"""
    
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Try Gamma first
    try:
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        expected_hist = len(data) * bin_width * stats.gamma.pdf(bin_centers, shape, loc=loc, scale=scale)
        single_fit = {'distribution': 'gamma', 'shape': shape, 'loc': loc, 'scale': scale}
        print_func(f"  Gamma fit: shape={shape:.2f}, scale={scale:.3f}")
        return single_fit, hist, bin_centers, expected_hist
    except Exception:
        pass
    
    # Fallback to Gaussian
    try:
        mu, sigma = stats.norm.fit(data)
        expected_hist = len(data) * bin_width * stats.norm.pdf(bin_centers, mu, sigma)
        single_fit = {'distribution': 'gaussian', 'mean': mu, 'std': sigma}
        print_func(f"  Gaussian fit: mean={mu:.3f}, std={sigma:.3f}")
        return single_fit, hist, bin_centers, expected_hist
    except Exception as e:
        print_func(f"  Single distribution fitting failed: {e}")
        return None, None, None, None


def _fit_residuals_em(shifted_residuals, bin_centers, max_iter, random_state, print_func):
    """Apply EM to residuals to find second peak"""
    
    # Create data points from shifted residuals
    residual_data = []
    for center, count in zip(bin_centers, shifted_residuals):
        if count > 0:
            residual_data.extend([center] * int(max(1, count)))
    
    residual_data = np.array(residual_data)
    if len(residual_data) < 10:
        print_func("  Insufficient residual data")
        return None, None, False
    
    # Try 2-component EM first
    for n_components in [2, 1]:
        try:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                max_iter=max_iter, random_state=random_state)
            gmm.fit(residual_data.reshape(-1, 1))
            
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            weights = gmm.weights_
            
            # Select rightmost component
            rightmost_idx = np.argmax(means)
            second_peak = {
                'mean': means[rightmost_idx],
                'std': stds[rightmost_idx], 
                'weight': weights[rightmost_idx]
            }
            
            print_func(f"  Residual EM ({n_components}D): second peak at {second_peak['mean']:.3f}±{second_peak['std']:.3f}")
            return second_peak, gmm, True
            
        except Exception:
            continue
    
    # Final fallback: weighted mean
    try:
        sig_mask = shifted_residuals > np.max(shifted_residuals) * 0.1
        if np.any(sig_mask):
            centers = bin_centers[sig_mask]
            weights = shifted_residuals[sig_mask]
            mean_second = np.average(centers, weights=weights)
            var_second = np.average((centers - mean_second)**2, weights=weights)
            
            second_peak = {'mean': mean_second, 'std': np.sqrt(var_second), 'weight': 1.0}
            print_func(f"  Fallback: weighted mean at {mean_second:.3f}")
            return second_peak, None, False
    except Exception:
        pass
    
    return None, None, False


def _plot_em_results(data, em_result, bins, plot_path):
    """Plot EM results"""
    
    plt.figure(figsize=(12, 8))
    
    mean1, mean2 = em_result['means']
    std1, std2 = em_result['stds']
    weight1, weight2 = em_result['weights']
    threshold = em_result['threshold']
    intersection_info = em_result['intersection_info']
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
    
    # Plot 2: Intersection analysis
    plt.subplot(2, 2, 2)
    x_range = np.linspace(mean1, mean2, 1000)
    pdf1 = weight1 * stats.norm.pdf(x_range, mean1, std1)
    pdf2 = weight2 * stats.norm.pdf(x_range, mean2, std2)
    mixture_pdf_range = pdf1 + pdf2
    
    plt.plot(x_range, mixture_pdf_range, 'b-', linewidth=2, label='Mixture PDF')
    plt.plot(x_range, pdf1, 'r--', alpha=0.8, label='Component 1')
    plt.plot(x_range, pdf2, 'g--', alpha=0.8, label='Component 2')
    plt.axvline(intersection_info['intersection_point'], color='orange', linewidth=2)
    plt.scatter([intersection_info['intersection_point']], [intersection_info['intersection_pdf']], 
                color='red', s=50, zorder=5)
    plt.text(intersection_info['intersection_point'], intersection_info['intersection_pdf'] + 0.01,
             f'Ratio: {intersection_info["failure_ratio"]:.3f}', ha='center', va='bottom')
    plt.legend()
    plt.title('Intersection Analysis')
    
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


def _plot_residual_results(data, result, bins, plot_path):
    """Plot residual method results"""
    
    plt.figure(figsize=(12, 8))
    
    single_fit = result['single_fit']
    second_peak = result['second_peak']
    threshold = result['threshold']
    hist = result['hist']
    bin_centers = result['bin_centers']
    expected_hist = result['expected_hist']
    residuals = result['residuals']
    shifted_residuals = result['shifted_residuals']
    residual_shift = result['residual_shift']
    residual_em_success = result['residual_em_success']
    residual_gmm = result['residual_gmm']
    threshold_fraction = result['threshold_fraction']
    
    # Plot 1: Single distribution fit
    plt.subplot(2, 2, 1)
    plt.hist(data, bins=bins, density=False, alpha=0.7, color='lightblue', label='Data')
    plt.plot(bin_centers, expected_hist, 'r-', linewidth=2, label=f'Single {single_fit["distribution"].title()}')
    
    if single_fit['distribution'] == 'gamma':
        main_peak_pos = max(0, (single_fit['shape'] - 1) * single_fit['scale'] + single_fit['loc'])
    else:
        main_peak_pos = single_fit['mean']
    
    plt.axvline(main_peak_pos, color='red', linestyle='--', alpha=0.7, label='Main peak')
    plt.legend()
    plt.title(f'RESIDUAL METHOD: {single_fit["distribution"].title()} Fit')
    
    # Plot 2: Residuals with EM components
    plt.subplot(2, 2, 2)
    positive_residuals = np.maximum(0, residuals)
    plt.plot(bin_centers, residuals, 'b-', alpha=0.7, label='All residuals')
    plt.plot(bin_centers, positive_residuals, 'r-', label='Positive residuals')
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    # Show EM components if available
    if residual_em_success and residual_gmm is not None:
        x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        total_points = len(result.get('residual_data_points', []))
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.001
        
        if residual_gmm.n_components == 1:
            comp_mean = residual_gmm.means_.flatten()[0]
            comp_std = np.sqrt(residual_gmm.covariances_).flatten()[0]
            scale_factor = total_points * bin_width
            comp_pdf = scale_factor * stats.norm.pdf(x_plot, comp_mean, comp_std) + residual_shift
            plt.plot(x_plot, comp_pdf, 'g--', alpha=0.8, label='EM component')
        else:
            means = residual_gmm.means_.flatten()
            stds = np.sqrt(residual_gmm.covariances_).flatten()
            weights = residual_gmm.weights_
            rightmost_idx = np.argmax(means)
            
            for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
                scale_factor = weight * total_points * bin_width
                comp_pdf = scale_factor * stats.norm.pdf(x_plot, mean, std) + residual_shift
                
                if i == rightmost_idx:
                    plt.plot(x_plot, comp_pdf, 'g-', linewidth=2, alpha=0.9, label='Selected component')
                else:
                    plt.plot(x_plot, comp_pdf, 'gray', linestyle='--', alpha=0.5, label='Other component')
    
    plt.axvline(second_peak['mean'], color='green', linestyle=':', label='Second peak')
    plt.legend()
    plt.title('Residuals + EM Analysis')
    
    # Plot 3: Combined model (simplified)
    plt.subplot(2, 2, 3)
    plt.hist(data, bins=bins, density=False, alpha=0.7, color='lightblue', label='Data')
    plt.plot(bin_centers, expected_hist, 'r-', alpha=0.8, label=f'{single_fit["distribution"].title()} fit')
    plt.axvline(second_peak['mean'], color='green', linestyle=':', alpha=0.7, label='Second peak')
    plt.axvline(threshold, color='purple', linewidth=2, label=f'Threshold (p{threshold_fraction:.2f})')
    plt.legend()
    plt.title('Combined Model')
    
    # Plot 4: Final result
    plt.subplot(2, 2, 4)
    plt.hist(data, bins=bins, density=False, alpha=0.7, color='lightblue', label='Data')
    plt.axvline(main_peak_pos, color='red', linestyle=':', alpha=0.7, label='Main peak')
    plt.axvline(second_peak['mean'], color='green', linestyle=':', alpha=0.7, label='Second peak')
    plt.axvline(threshold, color='purple', linewidth=2, label=f'Threshold (p{threshold_fraction:.2f})')
    
    plt.text(0.02, 0.98, 'RESIDUAL METHOD', fontweight='bold', color='blue', transform=plt.gca().transAxes, va='top')
    plt.text(0.02, 0.85, f'Single: {single_fit["distribution"].upper()}', transform=plt.gca().transAxes, va='top')
    plt.text(0.02, 0.72, f'EM: {"SUCCESS" if residual_em_success else "FALLBACK"}', transform=plt.gca().transAxes, va='top')
    plt.text(0.02, 0.59, f'Threshold: {threshold:.4f}', transform=plt.gca().transAxes, va='top')
    
    plt.legend()
    plt.title('Final Result')
    
    plt.tight_layout()
    plt.savefig(plot_path)