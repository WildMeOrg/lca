#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

# Define species mapping - mapping directory names to proper species names
species_mapping = {
    'beluga': 'Beluga whale',
    'forestelephants': 'Forest elephant',
    'giraffe': 'Giraffe',
    'lion': 'Lion',
    'plainszebra': 'Plains zebra',
    'spermwhale': 'Sperm whale',
    'whaleshark': 'Whale shark'
}

# Base paths
config_base = "/users/PAS2136/nepove/code/lca/lca/configs"
data_base = "/fs/ess/PAS2136/ggr_data/kate/data_embeddings"

def generate_hdbscan_config(species_dir, species_name):
    """Generate HDBSCAN config for a species"""
    config = {
        'exp_name': species_dir,
        'species': species_name,
        'algorithm': {
            'initial_topk': 10,
            'target_edges': 0,
            'target_proportion': 1.0
        },
        'algorithm_type': 'hdbscan',
        'hdbscan': {
            'min_cluster_size': 2,
            'min_samples': None,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'verifier_name': 'miewid',
            'theta': 0.1,
            'validation_step': 20
        },
        'data': {
            'viewpoint_list': ['right'],
            'name_keys': ['individual_id'],
            'id_key': 'uuid',
            'n_filter_min': 1,
            'n_filter_max': 100,
            'annotation_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/annotations_{species_dir}.json',
            'embedding_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/embeddings_{species_dir}.pickle',
            'output_path': f'/fs/ess/PAS2136/ggr_data/results/kate/{species_dir}/hdbscan_clustering/',
            'separate_viewpoints': False
        },
        'edge_weights': {
            'classifier_thresholds': {
                'miewid': 'auto',
                'metadata(miewid)': 'miewid'
            },
            'prob_human_correct': 1,
            'verifier_names': 'metadata(miewid) simulated_human'
        },
        'logging': {
            'log_level': 'INFO',
            'log_file': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/output/hdbscan_output.log',
            'update_log_file': True,
            'file_mode': 'w'
        }
    }
    return config

def generate_gc_config(species_dir, species_name):
    """Generate GC (graph clustering) config for a species"""
    config = {
        'exp_name': species_dir,
        'species': species_name,
        'algorithm': {
            'initial_topk': 10,
            'target_edges': 0
        },
        'algorithm_type': 'gc',
        'hdbscan': {
            'min_cluster_size': 2,
            'min_samples': None,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'verifier_name': 'miewid'
        },
        'data': {
            'viewpoint_list': ['right'],
            'name_keys': ['individual_id'],
            'id_key': 'uuid',
            'n_filter_min': 1,
            'n_filter_max': 100,
            'annotation_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/annotations_{species_dir}.json',
            'embedding_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/embeddings_{species_dir}.pickle',
            'output_path': f'/fs/ess/PAS2136/ggr_data/results/kate/{species_dir}/gc_clustering/',
            'separate_viewpoints': False
        },
        'edge_weights': {
            'classifier_thresholds': {
                'miewid': 'auto(0.01)',
                'metadata(miewid)': 'miewid'
            },
            'prob_human_correct': 1,
            'verifier_names': 'miewid simulated_human'
        },
        'logging': {
            'log_level': 'INFO',
            'log_file': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/output/gc_output_theta_005_metadata.log',
            'update_log_file': True,
            'file_mode': 'w',
            'auto_threshold_plot_path': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/hist/gc_hist.png'
        },
        'gc': {
            'verifier_name': 'miewid',
            'theta': 0.05,
            'validation_step': 20
        }
    }
    return config

def generate_manual_review_config(species_dir, species_name):
    """Generate manual review config for a species"""
    config = {
        'exp_name': f'{species_dir}_manual_review',
        'species': species_name,
        'algorithm_type': 'manual_review',
        'manual_review': {
            'topk': 10,
            'threshold': None,
            'clustering_method': 'connected_components',
            'review_batch_size': 100,
            'review_all_at_once': False,
            'max_reviews': 5000
        },
        'algorithm': {
            'initial_topk': 0,
            'target_edges': 0,
            'target_proportion': 0
        },
        'data': {
            'viewpoint_list': ['right', 'left'],
            'name_keys': ['individual_id'],
            'id_key': 'uuid',
            'n_filter_min': 1,
            'n_filter_max': 100,
            'annotation_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/annotations_{species_dir}.json',
            'embedding_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/embeddings_{species_dir}.pickle',
            'output_path': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/manual_review_clustering/',
            'separate_viewpoints': False
        },
        'edge_weights': {
            'prob_human_correct': 1.0,
            'verifier_names': 'miewid simulated_human'
        },
        'logging': {
            'log_level': 'INFO',
            'log_file': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/output/manual_review_output.log',
            'update_log_file': True,
            'file_mode': 'w'
        },
        'gc': {
            'verifier_name': 'miewid',
            'distance_power': 1
        }
    }
    return config

def generate_thresholded_review_config(species_dir, species_name):
    """Generate thresholded review config for a species"""
    config = {
        'exp_name': f'{species_dir}_thresholded_review',
        'species': species_name,
        'algorithm_type': 'thresholded_review',
        'thresholded_review': {
            'low_threshold': 0.85,
            'high_threshold': 0.9,
            'topk': 10,
            'max_reviews': 5000,
            'clustering_method': 'connected_components',
            'review_batch_size': 100
        },
        'algorithm': {
            'initial_topk': 0,
            'target_edges': 0,
            'target_proportion': 0
        },
        'data': {
            'viewpoint_list': ['right', 'left'],
            'name_keys': ['individual_id'],
            'id_key': 'uuid',
            'n_filter_min': 1,
            'n_filter_max': 100,
            'annotation_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/annotations_{species_dir}.json',
            'embedding_file': f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species_dir}/embeddings_{species_dir}.pickle',
            'output_path': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/thresholded_output/',
            'separate_viewpoints': False
        },
        'edge_weights': {
            'prob_human_correct': 1.0,
            'verifier_names': 'miewid simulated_human'
        },
        'logging': {
            'log_level': 'INFO',
            'log_file': f'/users/PAS2136/nepove/code/lca/lca/tmp/{species_dir}/thresholded_review_output.log',
            'update_log_file': True,
            'file_mode': 'w'
        },
        'gc': {
            'verifier_name': 'miewid',
            'distance_power': 1
        }
    }
    return config

def write_yaml_with_comments(config, filepath):
    """Write YAML with preserved formatting and comments"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        if 'manual_review' in config:
            f.write("# Manual review algorithm configuration\n")
        elif 'thresholded_review' in config:
            f.write("# Thresholded review algorithm configuration\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    """Generate configs for all species"""

    for species_dir, species_name in species_mapping.items():
        print(f"Generating configs for {species_name} ({species_dir})...")

        # Generate each config type
        configs = [
            ('config_{}_hdbscan.yaml'.format(species_dir), generate_hdbscan_config(species_dir, species_name)),
            ('config_{}.yaml'.format(species_dir), generate_gc_config(species_dir, species_name)),
            ('config_manual_review_{}.yaml'.format(species_dir), generate_manual_review_config(species_dir, species_name)),
            ('config_thresholded_review_{}.yaml'.format(species_dir), generate_thresholded_review_config(species_dir, species_name))
        ]

        # Write each config
        for filename, config in configs:
            filepath = os.path.join(config_base, species_dir, filename)
            write_yaml_with_comments(config, filepath)
            print(f"  Created: {filepath}")

        print(f"Completed configs for {species_name}\n")

if __name__ == "__main__":
    main()
    print("All species configs generated successfully!")