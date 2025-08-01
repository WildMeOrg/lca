"""Comprehensive visualization functions for threshold detection algorithm."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

try:
    from .threshold_utils import evaluate_mirrored_gamma_pdf
except ImportError:
    from threshold_utils import evaluate_mirrored_gamma_pdf


def create_diagnostic_plot(data: np.ndarray, result: dict, bins: int = 500, 
                          plot_path: str = "diagnostic.png") -> None:
    """
    Create comprehensive diagnostic plot showing all algorithm steps.
    
    Layout:
    - Plot 1: GMM fit and separability
    - Plot 2: Main distribution fit with cutoffs
    - Plot 3: Detailed tail analysis
    - Plot 4: Overall summary
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: GMM fit and separability
    plot_gmm_analysis(axes[0, 0], data, result, bins)
    
    # Plot 2: Main distribution fit
    plot_main_distribution(axes[0, 1], data, result, bins)
    
    # Plot 3: Tail analysis
    plot_tail_analysis(axes[1, 0], data, result)
    
    # Plot 4: Overall summary
    plot_overall_summary(axes[1, 1], data, result, bins)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def plot_gmm_analysis(ax, data: np.ndarray, result: dict, bins: int):
    """Plot 1: GMM fit and separability analysis."""
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='lightblue', label='Data')
    
    x_plot = np.linspace(data.min(), data.max(), 1000)
    
    if result.get('gmm_success', False):
        gmm = result.get('gmm', {})
        if 'means' in result:
            means = result['means']
            stds = result['stds']
            weights = result['weights']
            
            # Plot individual components
            for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
                comp_pdf = weight * stats.norm.pdf(x_plot, mean, std)
                ax.plot(x_plot, comp_pdf, '--', linewidth=2, 
                       label=f'Comp {i+1}: μ={mean:.2f}, σ={std:.2f}, w={weight:.2f}')
            
            # Plot mixture
            mixture_pdf = sum(w * stats.norm.pdf(x_plot, m, s) 
                            for m, s, w in zip(means, stds, weights))
            ax.plot(x_plot, mixture_pdf, 'k-', linewidth=2, alpha=0.8, label='GMM Mixture')
            
            # Add separability info
            if 'separability_info' in result:
                sep_info = result['separability_info']
                is_separable = result.get('gmm_separable', False)
                
                ax.text(0.05, 0.95, f"Separability: {'PASS' if is_separable else 'FAIL'}", 
                       transform=ax.transAxes, va='top', fontweight='bold',
                       color='green' if is_separable else 'red',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.text(0.05, 0.85, f"Distance: {sep_info['separation_distance']:.3f}", 
                       transform=ax.transAxes, va='top')
                ax.text(0.05, 0.78, f"Required: {sep_info['required_separation']:.3f}", 
                       transform=ax.transAxes, va='top')
    else:
        ax.text(0.5, 0.5, 'GMM Failed', transform=ax.transAxes, 
               ha='center', va='center', fontsize=14, color='red')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Step 1: GMM Fit and Separability')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)


def plot_main_distribution(ax, data: np.ndarray, result: dict, bins: int):
    """Plot 2: Main distribution fit with cutoff thresholds."""
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='lightblue', label='Data')
    
    x_plot = np.linspace(data.min(), data.max(), 1000)
    
    # Plot main distribution
    main_mean = result['main_mean']
    main_std = result['main_std']
    main_pdf = stats.norm.pdf(x_plot, main_mean, main_std)
    
    method = result.get('method', 'unknown')
    if method == 'gmm_component':
        component_idx = result.get('component_used', 0)
        label = f'Main (GMM component {component_idx+1}): μ={main_mean:.2f}, σ={main_std:.2f}'
    elif method == 'gmm_refined':
        label = f'Main (GMM refined): μ={main_mean:.2f}, σ={main_std:.2f}'
    else:
        label = f'Main (robust fit): μ={main_mean:.2f}, σ={main_std:.2f}'
    
    ax.plot(x_plot, main_pdf, 'r-', linewidth=2.5, label=label)
    
    # Plot Gamma distribution if available
    if 'main_distribution_gamma' in result and result.get('gamma_fit_success', False):
        gamma_info = result['main_distribution_gamma']
        gamma_shape = gamma_info['shape']
        gamma_loc = gamma_info['loc']
        gamma_scale = gamma_info['scale']
        
        # Plot Gamma PDF
        gamma_pdf = stats.gamma.pdf(x_plot, gamma_shape, loc=gamma_loc, scale=gamma_scale)
        ax.plot(x_plot, gamma_pdf, 'g--', linewidth=2.5, 
                label=f'Main (Gamma): α={gamma_shape:.2f}, β={gamma_scale:.2f}')
        
        # Show Gamma percentile cutoff
        gamma_threshold = gamma_info['threshold']
        percentile_used = gamma_info['percentile_used']
        ax.axvline(gamma_threshold, color='darkgreen', linestyle='-.', linewidth=2,
                  label=f'Gamma {percentile_used*100:.0f}% cutoff: {gamma_threshold:.2f}')
    
    # Show refinement cutoff if GMM was used (for backward compatibility)
    if method == 'gmm_refined' and 'refinement_percentile' in result:
        percentile = result['refinement_percentile']
        # Get bigger component parameters
        weights = result.get('weights', [])
        means = result.get('means', [])
        stds = result.get('stds', [])
        if len(weights) > 0:
            bigger_idx = 0 if weights[0] > weights[1] else 1
            bigger_mean = means[bigger_idx]
            bigger_std = stds[bigger_idx]
            cutoff = stats.norm.ppf(percentile, bigger_mean, bigger_std)
            ax.axvline(cutoff, color='green', linestyle='--', linewidth=2,
                      label=f'{percentile*100:.0f}th percentile cutoff')
    
    # Show tail cutoff
    if 'tail_threshold' in result:
        tail_threshold = result['tail_threshold']
        tail_sigma_threshold = result['tail_sigma_threshold']
        ax.axvline(tail_threshold, color='orange', linestyle=':', linewidth=2,
                  label=f'Tail cutoff ({tail_sigma_threshold:.2f}σ): {tail_threshold:.2f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Step 2: Main Distribution Fit')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)


def plot_tail_analysis(ax, data: np.ndarray, result: dict):
    """Plot 3: Detailed tail analysis with residuals."""
    if not result.get('tail_analysis_success', False):
        ax.text(0.5, 0.5, 'Insufficient tail samples', transform=ax.transAxes, 
               ha='center', va='center', fontsize=14, color='red')
        ax.set_title('Step 3-5: Tail Analysis')
        return
    
    # Create twin axes for residuals
    ax2 = ax.twinx()
    
    # Get tail data
    tail_threshold = result.get('tail_threshold', 0)
    tail_samples = data[data > tail_threshold]
    
    if 'bin_centers' in result and 'residuals' in result:
        bin_centers = result['bin_centers']
        residuals = result['residuals']
        
        # Plot tail histogram
        bins_tail = min(50, len(tail_samples)//5)
        n, bins, patches = ax.hist(tail_samples, bins=bins_tail, alpha=0.5, 
                                   color='lightblue', label='Tail samples')
        
        # Plot residuals
        ax2.plot(bin_centers, residuals, 'r-', linewidth=2, alpha=0.7, label='Residuals')
        
        # Plot smoothed residuals if available
        if 'smoothed_residuals' in result:
            smoothed_residuals = result['smoothed_residuals']
            ax2.plot(bin_centers, smoothed_residuals, 'b-', linewidth=2.5, 
                    label='Smoothed residuals')
        
        # Mark zero line
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Mark cutoff/zero-crossing
        if 'cutoff_value' in result:
            cutoff = result['cutoff_value']
            ax.axvline(cutoff, color='orange', linewidth=2, linestyle='--',
                      label=f'Zero-crossing: {cutoff:.2f}')
    
    # Plot target distribution
    if 'target_distribution' in result:
        target = result['target_distribution']
        x_range = np.linspace(tail_samples.min(), tail_samples.max(), 200)
        
        # Scale to match histogram
        if 'cutoff_value' in result:
            samples_used = np.sum(tail_samples > result['cutoff_value'])
        else:
            samples_used = len(tail_samples)
        
        if len(tail_samples) > 0:
            hist_vals, hist_bins = np.histogram(tail_samples, bins=bins_tail)
            bin_width = hist_bins[1] - hist_bins[0]
            scale = samples_used * bin_width
            
            # Check if mirrored Gamma parameters are available
            if 'shape' in target:
                # Use mirrored Gamma
                target_pdf = evaluate_mirrored_gamma_pdf(x_range, target['shape'], 
                                                        target['loc'], target['scale'], 
                                                        target['mirror_point']) * scale
                ax.plot(x_range, target_pdf, 'g-', linewidth=2.5, 
                       label=f"Target (Gamma): α={target['shape']:.2f}, β={target['scale']:.2f}")
            else:
                # Fallback to normal distribution
                target_pdf = stats.norm.pdf(x_range, target['mean'], target['std']) * scale
                ax.plot(x_range, target_pdf, 'g-', linewidth=2.5, 
                       label=f"Target: μ={target['mean']:.2f}, σ={target['std']:.2f}")
            
            # Mark target peak
            ax.axvline(target['mean'], color='green', linestyle=':', linewidth=2,
                      label='Target peak')
    
    # Mark final threshold
    if 'threshold' in result:
        ax.axvline(result['threshold'], color='purple', linewidth=3,
                  label=f"Threshold: {result['threshold']:.2f}")
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency', color='blue')
    ax2.set_ylabel('Residual', color='red')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')
    
    ax.set_title('Steps 3-5: Tail Analysis')
    ax.grid(True, alpha=0.3)


def plot_overall_summary(ax, data: np.ndarray, result: dict, bins: int):
    """Plot 4: Overall summary with all components."""
    # Main histogram
    ax.hist(data, bins=bins, density=True, alpha=0.5, color='lightblue', label='Data')
    
    x_plot = np.linspace(data.min(), data.max(), 1000)
    
    # Plot main distribution
    main_mean = result['main_mean']
    main_std = result['main_std']
    main_pdf = stats.norm.pdf(x_plot, main_mean, main_std)
    
    # Scale by weight if using GMM component
    if result.get('method') == 'gmm_component' and 'weights' in result:
        weights = result['weights']
        component_idx = result.get('component_used', 0)
        main_weight = weights[component_idx]
        main_pdf = main_pdf * main_weight
        label = f'Main distribution (w={main_weight:.2f})'
    else:
        label = 'Main distribution'
    
    ax.plot(x_plot, main_pdf, 'r-', linewidth=2, label=label)
    
    # Plot main Gamma distribution if available
    if 'main_distribution_gamma' in result and result.get('gamma_fit_success', False):
        gamma_info = result['main_distribution_gamma']
        gamma_shape = gamma_info['shape']
        gamma_loc = gamma_info['loc']
        gamma_scale = gamma_info['scale']
        
        # Plot Gamma PDF
        gamma_pdf = stats.gamma.pdf(x_plot, gamma_shape, loc=gamma_loc, scale=gamma_scale)
        
        # Scale by weight if using GMM component
        if result.get('method') == 'gmm_component' and 'weights' in result:
            gamma_pdf = gamma_pdf * main_weight
            
        ax.plot(x_plot, gamma_pdf, 'r--', linewidth=2, alpha=0.7,
                label='Main Gamma distribution')
    
    # Plot target distribution (scaled by approximate weight)
    if 'target_distribution' in result and result.get('tail_analysis_success', False):
        target = result['target_distribution']
        # Check if mirrored Gamma parameters are available
        if 'shape' in target:
            # Use mirrored Gamma
            target_pdf = evaluate_mirrored_gamma_pdf(x_plot, target['shape'], 
                                                    target['loc'], target['scale'], 
                                                    target['mirror_point'])
        else:
            # Fallback to normal distribution
            target_pdf = stats.norm.pdf(x_plot, target['mean'], target['std'])
        
        # Estimate weight from tail samples
        tail_weight = result.get('tail_samples', 0) / len(data)
        target_pdf_scaled = target_pdf * tail_weight * 0.5  # Rough scaling
        
        ax.plot(x_plot, target_pdf_scaled, 'g-', linewidth=2, 
               label='Target distribution (scaled)')
    
    # Mark threshold
    if 'threshold' in result:
        ax.axvline(result['threshold'], color='purple', linewidth=3,
                  label=f'Final threshold: {result["threshold"]:.3f}')
    
    # Add summary text
    method = result.get('method', 'unknown')
    method_text = {
        'gmm_component': 'GMM component',
        'gmm_refined': 'GMM (refined)',
        'robust': 'Robust fit',
        'unknown': 'Unknown'
    }.get(method, method)
    
    summary_text = f"Method: {method_text}\n"
    summary_text += f"Main: μ={main_mean:.3f}, σ={main_std:.3f}\n"
    
    # Add main Gamma info if available
    if 'main_distribution_gamma' in result and result.get('gamma_fit_success', False):
        gamma_info = result['main_distribution_gamma']
        summary_text += f"Main (Gamma): α={gamma_info['shape']:.3f}, β={gamma_info['scale']:.3f}\n"
    
    if 'target_distribution' in result:
        target = result['target_distribution']
        if 'shape' in target:
            summary_text += f"Target (Gamma): α={target['shape']:.3f}, β={target['scale']:.3f}\n"
            summary_text += f"        μ={target['mean']:.3f}, σ={target['std']:.3f}\n"
        else:
            summary_text += f"Target: μ={target['mean']:.3f}, σ={target['std']:.3f}\n"
    
    summary_text += f"Threshold: {result['threshold']:.3f}"
    
    if 'zero_crossing_found' in result:
        summary_text += f"\nZero-crossing: {'Yes' if result['zero_crossing_found'] else 'No'}"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10, family='monospace')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Overall Summary')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)


# Additional utility visualization functions

def plot_algorithm_flow(result: dict, save_path: str = "algorithm_flow.png"):
    """Create a flowchart-style visualization of algorithm decisions."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Create flowchart showing algorithm decisions
    y_pos = 0.9
    
    # Step 1: GMM
    gmm_success = result.get('gmm_success', False)
    color = 'green' if gmm_success else 'red'
    ax.text(0.5, y_pos, '1. Fit GMM', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    y_pos -= 0.15
    
    # Step 2: Main distribution
    if gmm_success and result.get('gmm_separable', False):
        method_text = '2. GMM separable → Refine with percentile cutoff'
        color = 'green'
    else:
        method_text = '2. GMM failed/not separable → Robust fit'
        color = 'orange'
    
    ax.text(0.5, y_pos, method_text, ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Continue with remaining steps...
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()