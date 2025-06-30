"""
Drone use case - simple wrapper around basic run functionality.
"""

import argparse
import os
import tempfile
import shutil
from init_logger import init_logger
from tools import get_config
from run import run

def run_for_viewpoints(config, viewpoint_list, save_dir):
    """Run algorithm for specific viewpoints."""
    # Simple config modification
    config_copy = config.copy()
    config_copy['data'] = config_copy['data'].copy()
    config_copy['data']['viewpoint_list'] = viewpoint_list
    
    # Override save directory for LCA if needed (backwards compatible)
    algorithm_type = config_copy.get('algorithm_type', 'lca')
    if algorithm_type == 'lca':
        if 'lca' not in config_copy:
            config_copy['lca'] = {}
        config_copy['lca'] = config_copy['lca'].copy()
        config_copy['lca']['db_path'] = save_dir
    
    # Use the basic run function
    return run(config_copy)


def main(config):
    """Main drone execution logic."""
    data_params = config['data']
    algorithm_config = config.get('lca', config.get('gc', {}))
    
    # Setup output directory
    temp_db = algorithm_config.get('temp_db', False)
    
    if temp_db:
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(data_params['output_dir'], config['exp_name'])
        os.makedirs(db_path, exist_ok=True)
    
    try:
        if data_params.get('separate_viewpoints', False):
            # Run separately for each viewpoint
            for viewpoint in data_params['viewpoint_list']:
                print(f"Running for viewpoint: {viewpoint}")
                save_dir = os.path.join(db_path, viewpoint)
                os.makedirs(save_dir, exist_ok=True)
                run_for_viewpoints(config, [viewpoint], save_dir)
        else:
            # Run for all viewpoints together
            run_for_viewpoints(config, data_params['viewpoint_list'], db_path)
    
    finally:
        if temp_db:
            shutil.rmtree(db_path, ignore_errors=True)


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser(description="Run clustering algorithm for drone data.")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    config = get_config(args.config)
    main(config)