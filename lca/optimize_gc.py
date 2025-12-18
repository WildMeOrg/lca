#!/usr/bin/env python3
"""
Hyperparameter optimization for GC clustering using Optuna.

Usage:
    python optimize_gc.py --config path/to/config.yaml --n_trials 50
    python optimize_gc.py --config path/to/config.yaml --n_trials 100 --output_dir ./optimization_results
"""

import argparse
import copy
import json
import os
import signal
import tempfile
import shutil

import optuna
import yaml

from init_logger import init_logger
from run_clustering_with_save import run_clustering_with_save


class TrialTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise TrialTimeout("Trial exceeded time limit")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_objective(base_config, metric='f1', output_dir=None, trial_timeout=None):
    """Create Optuna objective function for GC hyperparameter optimization."""

    def objective(trial):
        config = copy.deepcopy(base_config)

        # Sample the 5 GC parameters
        p = trial.suggest_float('p', 0.01, 0.99)
        distance_power = trial.suggest_float('distance_power', 0.5, 2.0)
        initial_topk = trial.suggest_int('initial_topk', 2, 50)
        max_densify_edges = trial.suggest_int('max_densify_edges', 50, 500)
        theta = trial.suggest_float('theta', 0.0, 1.0)

        # Apply parameters to config
        config['edge_weights']['classifier_thresholds']['miewid'] = f'auto({p})'
        config['algorithm']['distance_power'] = distance_power
        config['algorithm']['initial_topk'] = initial_topk
        config['gc']['max_densify_edges'] = max_densify_edges
        config['gc']['theta'] = theta

        # Create trial output directory
        if output_dir:
            trial_output = os.path.join(output_dir, 'trials', f'trial_{trial.number}')
            os.makedirs(trial_output, exist_ok=True)
            cleanup = False
        else:
            trial_output = tempfile.mkdtemp(prefix=f"gc_trial_{trial.number}_")
            cleanup = True

        config['data']['output_path'] = trial_output

        # Update logging paths to use trial directory
        config['logging']['log_file'] = os.path.join(trial_output, 'trial.log')
        config['logging']['update_log_file'] = False  # Prevent timestamp being appended
        config['logging']['file_mode'] = 'w'  # Overwrite for each trial
        if 'auto_threshold_plot_path' in config['logging']:
            config['logging']['auto_threshold_plot_path'] = os.path.join(trial_output, 'hist.png')

        try:
            # Set timeout if specified
            if trial_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(trial_timeout)

            metrics = run_clustering_with_save(config)
            score = metrics.get(metric, 0.0) if metrics else 0.0

            # Save trial params and score
            if output_dir:
                with open(os.path.join(trial_output, 'params.json'), 'w') as f:
                    json.dump({'params': trial.params, metric: score}, f, indent=2)

            return score
        except TrialTimeout:
            print(f"Trial {trial.number} timed out after {trial_timeout}s")
            return 0.0
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
        finally:
            if trial_timeout:
                signal.alarm(0)  # Cancel the alarm
            if cleanup:
                shutil.rmtree(trial_output, ignore_errors=True)

    return objective


def save_results(study, output_dir, metric):
    """Save optimization results to output directory."""
    # Save best params
    best_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'metric': metric,
    }
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_results, f, indent=2)

    # Save all trials summary
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
        })
    with open(os.path.join(output_dir, 'all_trials.json'), 'w') as f:
        json.dump(trials_data, f, indent=2)

    # Save config snippet for best params
    config_snippet = f"""algorithm:
  initial_topk: {study.best_params['initial_topk']}
  distance_power: {study.best_params['distance_power']}
edge_weights:
  classifier_thresholds:
    miewid: auto({study.best_params['p']})
gc:
  theta: {study.best_params['theta']}
  max_densify_edges: {study.best_params['max_densify_edges']}
"""
    with open(os.path.join(output_dir, 'best_config_snippet.yaml'), 'w') as f:
        f.write(config_snippet)


def main():
    init_logger()

    parser = argparse.ArgumentParser(description="Optimize GC clustering hyperparameters")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--metric', type=str, default='f1',
                       choices=['f1', 'precision', 'recall', 'frac_correct'])
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save all trial outputs and results')
    parser.add_argument('--study_name', type=str, default='gc_optimization')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--trial_timeout', type=int, default=None,
                       help='Timeout per trial in seconds')
    args = parser.parse_args()

    base_config = load_config(args.config)

    # Setup output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(args.output_dir, 'study.db')}"
    else:
        storage = None

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,
    )

    objective = create_objective(base_config, args.metric, args.output_dir, args.trial_timeout)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save results
    if args.output_dir:
        save_results(study, args.output_dir, args.metric)
        print(f"\nResults saved to: {args.output_dir}")

    # Print results
    print("\n" + "="*50)
    print(f"Best {args.metric}: {study.best_value:.4f}")
    print("="*50)
    print("\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print("\nConfig snippet:")
    print(f"""
algorithm:
  initial_topk: {study.best_params['initial_topk']}
  distance_power: {study.best_params['distance_power']}
edge_weights:
  classifier_thresholds:
    miewid: auto({study.best_params['p']})
gc:
  theta: {study.best_params['theta']}
  max_densify_edges: {study.best_params['max_densify_edges']}
""")


if __name__ == '__main__':
    main()
