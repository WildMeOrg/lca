"""
Simple, truly algorithm-agnostic run script.
Core execution logic that works for any algorithm with step() interface.
"""

import argparse
import logging
from init_logger import init_logger
from tools import get_config, write_json
from algorithm_preparation import create_algorithm
import os

logger = logging.getLogger('lca')


def get_initial_edges(common_data, config):
    """Get initial edges in unified format."""
    verifier_name = common_data['verifier_name']
    verifier_embeddings = common_data['embeddings_dict'][verifier_name]
    
    # Get algorithm parameters (with backwards compatibility)
    # New: algorithm section, Old: no specific location (use defaults)
    algorithm_params = config.get('algorithm', {})
    print(algorithm_params)
    target_edges = algorithm_params.get('target_edges', 10000)
    initial_topk = algorithm_params.get('initial_topk', 10)
    
    # Get raw edges
    raw_edges = list(verifier_embeddings.get_edges(target_edges=target_edges, topk=initial_topk))
    raw_edges = [(*n, verifier_name) for n in raw_edges]

    return raw_edges


def get_human_responses(requested_edges, common_data):
    """Get human responses in unified format."""
    human_reviewer = common_data['human_reviewer']
    human_reviews, quit = human_reviewer(requested_edges)
    
    # Convert to unified format - use 'human' for all human responses for simplicity
    unified_responses = []
    for n0, n1, decision in human_reviews:
        score = 1.0 if decision else 0.0
        unified_responses.append((n0, n1, score, 'human'))
    
    return unified_responses, quit


def run(config):
    """
    Core algorithm execution - truly algorithm agnostic.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: Algorithm results
    """
    logger.info(f"Starting {config.get('algorithm_type', 'lca').upper()} algorithm")
    
    # Create algorithm
    algorithm, common_data = create_algorithm(config)
    
    # Check if we need to handle temp database cleanup for LCA
    algorithm_type = config.get('algorithm_type', 'lca')
    lca_config = config.get('lca', {})
    temp_db = lca_config.get('temp_db', False) if algorithm_type == 'lca' else False
    
    try:
        # Get initial edges
        initial_edges = get_initial_edges(common_data, config)
        logger.info(f"Starting with {len(initial_edges)} initial edges")
        # Start algorithm
        requested_edges = algorithm.step(initial_edges)
        
        # Main loop - algorithm handles all its own logic
        while requested_edges and not algorithm.is_finished():
            human_responses, quit = get_human_responses(requested_edges, common_data)
            
            if quit:
                logger.info("Human reviewer requested to quit")
                break
                
            requested_edges = algorithm.step(human_responses)
        
        logger.info("Algorithm execution completed")
        algorithm.show_stats()
        
        # Return results
        result = algorithm.get_clustering()

        (clustering_dict, node2cid_dict, G) = result
        output_path = common_data.get('output_path', 'tmp')
        # if 'timestamp' in common_data:
        #     output_path = output_path + '_' + common_data['timestamp']
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Saving results to {output_path}")
        write_json(clustering_dict, os.path.join(output_path, 'clustering.json'))
        write_json(node2cid_dict, os.path.join(output_path, 'node2cid.json'))
        write_json(common_data['node2uuid'], os.path.join(output_path, 'node2uuid_file.json'))
        write_json(G, os.path.join(output_path, 'graph.json'))
        return result
    
    finally:
        # Cleanup temp database if needed
        if temp_db and algorithm_type == 'lca':
            import tempfile
            import shutil
            # Get the temp path from the LCA instance
            if hasattr(algorithm, 'lca_config') and 'autosave_file' in algorithm.lca_config:
                autosave_path = algorithm.lca_config['autosave_file']
                temp_db_path = os.path.dirname(autosave_path)
                if temp_db_path.startswith(tempfile.gettempdir()):
                    logger.info(f"Cleaning up temp database: {temp_db_path}")
                    shutil.rmtree(temp_db_path, ignore_errors=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run clustering algorithm.")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--interactive', '-i', 
                   action='store_true', 
                   help='Enable interactive mode')
    return parser.parse_args()

 

if __name__ == '__main__':
    init_logger()
    args = parse_args()
    if args.interactive:
        print(f"Working with config {os.path.abspath(args.config)}")
        input("Press Enter to continue...")
    config = get_config(args.config)
    run(config)