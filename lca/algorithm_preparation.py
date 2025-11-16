"""
Unified algorithm preparation module.
Handles all common setup logic for both LCA and Graph Consistency algorithms.
"""

import types
import numpy as np
import random
import os
import logging
import datetime
import tempfile
import shutil
import re
from pathlib import Path


from negative_only_embeddings import NegativeOnlyEmbeddings
from preprocess import preprocess_data
from embeddings import Embeddings
from embeddings_lightglue import LightglueEmbeddings
from binary_embeddings import BinaryEmbeddings
from synthetic_embeddings import SyntheticEmbeddings
from random_embeddings import RandomEmbeddings
import scores.kernel_density_scores as kernel_density_scores
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth, generate_ground_truth_random, generate_calib_weights, generate_ground_truth_full_dataset
from tools import *
from cluster_validator import ClusterValidator
from graph_consistency import GraphConsistencyAlgorithm
from classifier import Classifier
import ga_driver
from classifier_system import ClassifierManager, WeighterBasedClassifier, ThresholdBasedClassifier
from metadata_verifier import MetadataEmbeddings
from tracking_id_verifier import TrackingIdEmbeddings
from robust_threshold import find_robust_threshold
from robust_gmm_threshold import find_threshold as robust_gmm_find_threshold
from hdbscan_algorithm import HDBSCANAlgorithm
from manual_review_algorithm import ManualReviewAlgorithm
from thresholded_review_algorithm import ThresholdedReviewAlgorithm

logger = logging.getLogger('lca')

def parse_verifier_names(verifier_names):
    """
    Parse verifier names with metadata syntax.
    
    Examples:
    - "metadata(miewid) lightglue human" → metadata wraps miewid explicitly
    - "metadata miewid human" → metadata wraps next verifier (miewid) implicitly
    
    Returns:
    - parsed_verifiers: List of (verifier_name, base_embeddings_name) tuples
    """
    parsed_verifiers = []
    
    meta_names = ['metadata', 'tracking', 'negative_only']

    i = 0
    while i < len(verifier_names):
        current = verifier_names[i]
        meta = False
        for meta_name in meta_names:
            if current.startswith(meta_name + '(') and current.endswith(')'):
                # Explicit: metadata(miewid)
                # Remove "metadata(" prefix and ")" suffix
                base_name = current.removeprefix(meta_name + '(').removesuffix(')')
                if not base_name:
                    raise ValueError(f"Empty base verifier in {meta_name}() syntax")
                parsed_verifiers.append((meta_name, base_name))
                meta = True
                break
            elif current == meta_name:
                # Implicit: metadata uses next verifier as base
                if i + 1 >= len(verifier_names):
                    raise ValueError(f"{meta_name} verifier needs a base verifier")
                base_name = verifier_names[i + 1]
                parsed_verifiers.append((meta_name, base_name))
                i += 1
                meta = True
                break
        if not meta:
            parsed_verifiers.append((current, None))
        i += 1
    
    return parsed_verifiers

def prepare_common(config):
    """
    Handle all shared preparation tasks common to both algorithms.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Common data needed by both algorithms
    """
    np.random.seed(42)
    random.seed(42)
    
    data_params = config['data']
    images_dir = data_params.get('images_dir', None)
    if images_dir == "None":
        images_dir = None

    data_params['images_dir'] = images_dir
    algorithm_config = config.get('lca', config.get('gc', {}))  # Fallback to available config
    
    # 1. Load and preprocess data
    logger.info("Loading embeddings and preprocessing data...")
    embeddings, uuids = load_pickle(data_params['embedding_file'])
    
    name_keys = data_params['name_keys']
    filter_key = '__'.join(name_keys)
    
    # Support different preprocessing formats
    format_type = data_params.get('format', 'drone')  # 'old' or 'drone'
    id_key = data_params.get('id_key', 'uuid')
    
    # Create field_filters from any {field}_list entries in config
    field_filters = {}
    for key in data_params:
        if key.endswith('_list'):
            field_name = key[:-5]  # Remove '_list' suffix
            field_filters[field_name] = data_params[key]
    
    df = preprocess_data(
        data_params['annotation_file'], 
        name_keys=name_keys,
        convert_names_to_ids=True, 
        n_filter_min=data_params['n_filter_min'], 
        n_filter_max=data_params['n_filter_max'],
        images_dir=data_params['images_dir'], 
        embedding_uuids=uuids,
        id_key=id_key,
        format=format_type,
        field_filters=field_filters,
        print_func=logger.info
    )
    
    print_intersect_stats(df, individual_key=filter_key)
    
    # 2. Setup ground truth and validation
    logger.info("Setting up ground truth and validation...")
    filtered_df = df[df[id_key].isin(uuids)]
    
    # Keep both filtered and unfiltered embeddings
    filtered_embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df[id_key]]
    # Create node2uuid mapping for ALL data (for threshold calculation)
    all_node2uuid = {i: uuid for i, uuid in enumerate(uuids)}
    
    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key, id_key)
    
    cluster_validator = ClusterValidator(gt_clustering, gt_node2cid)
    ga_driver.set_validator_functions(
        cluster_validator.trace_start_human, 
        cluster_validator.trace_iter_compare_to_gt
    )
    
    # 3. Create embeddings dictionary (filtered for algorithm use)
    logger.info("Setting up embeddings...")
    distance_power = algorithm_config.get('distance_power', 1)
    embeddings_dict = {
        'miewid': lazy(lambda: Embeddings(filtered_embeddings, node2uuid, distance_power=distance_power, print_func=logger.info)),
        'binary': lazy(lambda: BinaryEmbeddings(node2uuid, df, filter_key)),
        'random': lazy(lambda: RandomEmbeddings()),
        'lightglue': lazy(lambda: LightglueEmbeddings(node2uuid, "lightglue_scores_superpoint.pickle")) # TODO: get the correct path
    }
    
    # Create unfiltered embeddings for threshold calculation (when needed)
    unfiltered_embeddings_dict = {
        'miewid': lazy(lambda: Embeddings(embeddings, all_node2uuid, distance_power=distance_power, print_func=logger.info))
    }
    
    # 4. Setup human reviewer based on augmentation names (with backwards compatibility)
    logger.info("Setting up human reviewer...")
    
    # Backwards compatibility: edge_weights can be top-level or inside algorithm config
    edge_weights = config.get('edge_weights', algorithm_config.get('edge_weights', {}))
    algorithm_config['scorer'] = edge_weights.get('scorer', 'kde')
    algorithm_config['prob_human_correct'] = edge_weights.get('prob_human_correct', 0.98)
    
    prob_human_correct = edge_weights.get('prob_human_correct', 0.98)
    # aug_names = edge_weights.get('augmentation_names', 'miewid human').split()
    verifier_names_str = edge_weights.get('verifier_names', edge_weights.get('augmentation_names', 'miewid human'))
    aug_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str
    parsed_verifiers = parse_verifier_names(aug_names)

    # Backwards compatibility: handle old "human" + simulate_human flag
    simulate_human = algorithm_config.get('simulate_human', True)
    
    # Determine human reviewer type
    human_reviewer = None
    
    # New format: specific human types in aug_names
    for aug_name in aug_names:
        if aug_name == 'simulated_human':
            human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
            break
        elif aug_name == 'ui_human':
            ui_db_path = data_params.get('ui_db_path')
            if ui_db_path:
                from human_db import human_db
                human_reviewer = human_db(ui_db_path, filtered_df, node2uuid)
            else:
                logger.warning("ui_human specified but no ui_db_path provided, falling back to simulated")
                human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
            break
        elif aug_name == 'no_human':
            logger.info("no_human - running without human reviews")
            human_reviewer = lambda _: ([], True)
            break
    
    # Backwards compatibility: handle old "human" in aug_names (from existing configs)
    if human_reviewer is None and 'human' in aug_names:
        if simulate_human:
            human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
        else:
            # Old non-simulated case - try UI database
            ui_db_path = data_params.get('ui_db_path')
            if ui_db_path:
                from human_db import human_db
                human_reviewer = human_db(ui_db_path, filtered_df, node2uuid)
            else:
                human_reviewer = call_get_reviews(df, filter_key, prob_human_correct)
    
    # Final fallback: default to simulated if no human type found
    if human_reviewer is None:
        logger.info("No human reviewer type specified, defaulting to no_human")
        human_reviewer = lambda _: ([], True)
    
    # 5. Setup synthetic embeddings if needed
    verifier_name = algorithm_config.get('verifier_name', 'miewid')
    
    if verifier_name == 'synthetic' or 'synthetic' in aug_names:
        logger.info("Setting up synthetic embeddings...")
        verifier_edges = embeddings_dict['miewid']().get_edges()
        pos, neg, quit = generate_ground_truth_full_dataset(verifier_edges, human_reviewer)
        scorer = kernel_density_scores.kernel_density_scores.create_from_samples(pos, neg) 
        synthetic_embeddings = SyntheticEmbeddings(
            node2uuid, df, filter_key, 
            lambda: scorer.density_pos.sample().item(), 
            lambda: scorer.density_neg.sample().item()
        )
        embeddings_dict['synthetic'] = lazy(lambda : synthetic_embeddings)
    
    for name, base_name in parsed_verifiers:
        if name == 'metadata':
            # Create metadata wrapper
            base_embeddings = embeddings_dict[base_name]()
            embeddings_dict[f'metadata({base_name})'] = lazy(lambda: MetadataEmbeddings(
                base_embeddings, df, node2uuid)
            )
        elif name == 'tracking':
            # Create tracking ID wrapper
            base_embeddings = embeddings_dict[base_name]()
            embeddings_dict[f'tracking({base_name})'] = lazy(lambda: TrackingIdEmbeddings.from_embeddings(
                base_embeddings, df, node2uuid, id_key, tracking_key='tracking_id', multiplier=1)
            )
        elif name == 'negative_only':
            # Create tracking ID wrapper
            base_embeddings = embeddings_dict[base_name]()
            embeddings_dict[f'negative_only({base_name})'] = lazy(lambda: NegativeOnlyEmbeddings.from_embeddings(
                base_embeddings, df, node2uuid, id_key, class_key='tracking_id', multiplier=1)
            )

    logger.info("Computing and logging verifier performance statistics...")
    primary_verifier_embeddings = embeddings_dict[verifier_name]()
    
    try:
        # Log top-k accuracy statistics  
        topk_results = primary_verifier_embeddings.get_stats(filtered_df, filter_key, id_key)
        logger.info(f"Top-k Accuracy Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))

        # Log detailed top-20 matches for each individual
        # top20_results = primary_verifier_embeddings.get_top20_matches(filtered_df, filter_key)
        # for uuid, top20 in top20_results.items():
        #     logger.info(f"ID: {uuid} | TOP-20: " + ", ".join([f"{k}: {v:.2f}" for (k, v) in top20]))
            
    except Exception as e:
        logger.warning(f"Failed to compute verifier statistics: {e}")
        logger.info("Continuing without statistics logging")

    for aug_name in aug_names:
        if 'human' not in aug_name:
            embeddings_dict[aug_name] = embeddings_dict[aug_name]()
    if verifier_name not in aug_names:
        embeddings_dict[verifier_name] = embeddings_dict[verifier_name]()
    
    algorithm_type = config.get('algorithm_type', 'gc')

    weighters = {}
    weighters_calibration = None
    
    edge_weights = config.get('edge_weights', config.get('lca', {}).get('edge_weights', {}))
    weight_thresholds = edge_weights.get('weight_thresholds', {})  # NEW

    if algorithm_type == 'lca' or weight_thresholds:
        # 6. Generate weighter calibration
        logger.info("Generating weighter calibration...")
        weighters_calibration = generate_weighter_calibration(
            embeddings_dict, human_reviewer, edge_weights, verifier_name, aug_names, logger, config
        )
        
        # 7. Create weighters
        weighters = ga_driver.generate_weighters(algorithm_config, weighters_calibration)

    if "histogram_path" in config.get("logging", {}):
        histogram_path = config["logging"]["histogram_path"]
        plt.figure()

        scores = primary_verifier_embeddings.get_all_scores()
        plt.hist(scores, bins=500, density=True, alpha=0.6, color='g')

        if verifier_name in weighters:
            xs = np.linspace(0, 1, 100)
            wgtr = weighters[verifier_name]
            pos_ys = [wgtr.scorer.get_pos_neg(x)[0] for x in xs]
            neg_ys = [wgtr.scorer.get_pos_neg(x)[1] for x in xs]

            plt.plot(xs, pos_ys, color='g')
            plt.plot(xs, neg_ys, color='r')

            # wgtr = weighter.weighter(scorer, config["lca"]["edge_weights"]['prob_human_correct'])
            wgtr.max_weight = 10
            
            plt.plot(xs, [wgtr.wgt_smooth(x) for x in xs], 'k-')

        plt.xlabel("Score")
        plt.ylabel("Probability density function")
        plt.title(config.get("species", ""))

        plt.savefig(histogram_path)
        plt.close()

    if 'output_path' in data_params:
        output_path = data_params['output_path']
        os.makedirs(output_path, exist_ok=True)
    elif 'lca' in config: #  and algorithm_type == 'lca'
        db_path = config['lca'].get('db_path', 'tmp')
        exp_name = config.get('exp_name', 'default')
        output_path = os.path.join(db_path, exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = 'tmp'

    algorithm_params = config.get('algorithm', {})
    target_edges = algorithm_params.get('target_edges', 0)
    target_proportion = algorithm_params.get('target_proportion', None)
    initial_topk = algorithm_params.get('initial_topk', 10)


    # 9. Return common data
    return {
        'embeddings_dict': embeddings_dict,
        'unfiltered_embeddings_dict': unfiltered_embeddings_dict,
        'gt_clustering': gt_clustering,
        'gt_node2cid': gt_node2cid,
        'node2uuid': node2uuid,
        'cluster_validator': cluster_validator,
        'human_reviewer': human_reviewer,
        'weighters': weighters,
        'weighters_calibration': weighters_calibration,
        'filtered_df': filtered_df,
        'df': df,
        'filter_key': filter_key,
        'verifier_name': verifier_name,
        'output_path': output_path,
        'target_edges': target_edges,
        'initial_topk': initial_topk,
        'target_proportion': target_proportion
    }


def generate_weighter_calibration(embeddings_dict, human_reviewer, edge_weights, verifier_name, aug_names, logger, config=None):
    """
    Generate calibration data for weighters with caching support.
    
    Args:
        config: Full configuration dict for accessing cache settings and db_path
        
    Returns:
        dict: Calibration data for all embedding methods
    """
    # Determine cache file location
    cache_file = None
    if 'verifier_file' in edge_weights:
        # Use specified cache file
        cache_file = edge_weights['verifier_file']
        logger.info(f"Using specified cache file: {cache_file}")
    elif config:
        # Use default cache location in database
        algorithm_type = config.get('algorithm_type', 'gc')
        if 'lca' in config: #  and algorithm_type == 'lca'
            db_path = config['lca'].get('db_path', 'tmp')
            exp_name = config.get('exp_name', 'default')
            cache_dir = os.path.join(db_path, exp_name)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "verifiers_probs.json")
            logger.info(f"Using default cache file: {cache_file}")
    logger.info(os.path.exists(cache_file))
    # Try to load cached calibration
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached weighter calibration from {cache_file}")
        try:
            return load_json(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            logger.info("Proceeding with fresh calibration computation")
    
    # Compute calibration (existing logic)
    logger.info("Computing weighter calibration...")
    num_pos_needed = edge_weights.get('num_pos_needed', 300)
    num_neg_needed = edge_weights.get('num_neg_needed', 50)
    
    wgtrs_calib_dict = {}
    
    # Get verifier edges for calibration
    verifier_edges = embeddings_dict[verifier_name].get_edges()
    
    # Generate ground truth for main verifier
    logger.info(f"Generating calibration data for {verifier_name}...")
    gt_weights, pos_edges, neg_edges = process_verifier_edges(
        verifier_name, verifier_edges, human_reviewer, num_pos_needed, num_neg_needed, logger
    )
    wgtrs_calib_dict[verifier_name] = gt_weights
    
    # Generate calibration for other methods (excluding human reviewer types)
    human_types = {'simulated_human', 'ui_human', 'no_human', 'human'}  # Include old 'human' for compatibility
    for method in aug_names:
        if method in human_types or method == verifier_name:
            continue
            
        if method not in embeddings_dict:
            logger.warning(f"Embeddings for method {method} not found.")
            continue
            
        logger.info(f"Generating calibration data for {method}...")
        get_score = embeddings_dict[method].get_score
        wgtrs_calib_dict[method] = process_edges(method, pos_edges, neg_edges, get_score, logger)
    
    # Save to cache
    if cache_file:
        try:
            # Use the same save function as the old system
            def save_probs_to_db(data, output_path):
                dir_name = os.path.dirname(output_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                write_json(data, output_path)
            
            save_probs_to_db(wgtrs_calib_dict, cache_file)
            logger.info(f"Saved weighter calibration to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    return wgtrs_calib_dict


def process_verifier_edges(method, edges, human_reviewer, num_pos_needed, num_neg_needed, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg, quit = generate_ground_truth_random(edges, human_reviewer, num_pos_needed, num_neg_needed)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }, pos, neg


def process_edges(method, pos_edges, neg_edges, get_score, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg = generate_calib_weights(pos_edges, neg_edges, get_score)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }


def setup_logging(config):
    """
    Setup logging configuration with backwards compatibility.
    """
    # exp_name = config.get('exp_name', "")
    algorithm_config = config.get('lca', config.get('gc', {}))
    
    # Backwards compatibility: logging can be top-level or inside algorithm config
    logging_config = config.get('logging', algorithm_config.get('logging', {}))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    log_level = logging_config.get('log_level', 'INFO')
    log_file = logging_config.get('log_file')
    
    if log_file is not None:
        if logging_config.get("update_log_file", True):
            log_file_name = Path(log_file)
            log_file_name = log_file_name.with_name(log_file_name.stem + f'_{timestamp}' + log_file_name.suffix)
            logging_config['log_file'] = log_file_name

        logger = logging.getLogger('lca')
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

        file_mode = logging_config.get("file_mode", 'w')
        handler = logging.FileHandler(log_file, mode=file_mode)
        handler.setLevel(log_level)
        handler.setFormatter(get_formatter())
        logger.addHandler(handler)
        
        # print(f"Logging to {os.path.abspath(log_file)}")
    return log_file, timestamp


def prepare_lca(common_data, config):
    """
    LCA-specific preparation with backwards compatibility.
    
    Args:
        common_data: Common data from prepare_common()
        config: Configuration dictionary
        
    Returns:
        curate_using_LCA: Configured LCA instance
    """
    lca_config = config['lca'].copy()  # Don't modify original
    exp_name = config['exp_name']
    
    # Backwards compatibility: if edge_weights or logging are at top level, use them
    if 'edge_weights' in config:
        lca_config['edge_weights'] = config['edge_weights']
    if 'logging' in config:
        lca_config['logging'] = config['logging']
    
    # Generate ga_params in exactly the same format as generate_ga_params()
    # This ensures LCA gets parameters in the expected format
    ga_params = generate_lca_params(lca_config)
    
    # Setup database
    temp_db = lca_config.get('temp_db', False)
    if temp_db:
        logger.info(f"Using temp database...")
        db_path = tempfile.mkdtemp()
    else:
        db_path = os.path.join(lca_config.get('db_path', 'tmp'), exp_name)
        if ('clear_db' in lca_config) and lca_config['clear_db'] and os.path.exists(db_path):
            logger.info("Removing old database...")
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
    
    edge_db_file = os.path.join(str(db_path), "quads.csv")
    clustering_file = os.path.join(str(db_path), "clustering.json")
    autosave_file = os.path.join(str(db_path), "autosave.json")
    node2uuid_file = os.path.join(str(db_path), "node2uuid_file.json")
    
    ga_params['autosave_file'] = autosave_file
    
    # Create verifiers dictionary
    verifiers_dict = {
        ver_name: call_verifier_alg(common_data['embeddings_dict'][ver_name]) 
        for ver_name in common_data['embeddings_dict'].keys()
    }
    
    algorithm_params = config.get('algorithm', {})
    tries_before_edge_done = algorithm_params.get('tries_before_edge_done', 
                                                 lca_config.get('iterations', {}).get('tries_before_edge_done', 4))
    
    ga_params['tries_before_edge_done'] = tries_before_edge_done

    # Create LCA instance with ga_params in the exact expected format
    lca_instance = curate_using_LCA(
        verifiers_dict,
        common_data['verifier_name'],
        common_data['human_reviewer'],
        common_data['weighters_calibration'],
        edge_db_file,
        clustering_file,
        {},  # current_clustering (empty)
        ga_params  # Use ga_params instead of lca_config
    )
    
    return lca_instance


def generate_lca_params(lca_config):
    """
    Generate LCA parameters in exactly the same format as generate_ga_params().
    This ensures compatibility with existing LCA code.
    """
    ga_params = dict()

    # Handle backwards compatibility for edge_weights location
    edge_weights = lca_config.get('edge_weights', {})
    
    phc = float(edge_weights.get('prob_human_correct', 0.98))
    assert 0 < phc <= 1
    ga_params['prob_human_correct'] = phc
    
    # Handle new human reviewer format with backwards compatibility
    aug_names_str = edge_weights.get('augmentation_names', 'miewid human')
    if isinstance(aug_names_str, str):
        aug_names = aug_names_str.split()
    else:
        aug_names = aug_names_str
    
    # Backwards compatibility: convert new human types to old simulate_human flag
    if 'simulated_human' in aug_names:
        ga_params['simulate_human'] = True
    elif 'ui_human' in aug_names:
        ga_params['simulate_human'] = False
    elif 'human' in aug_names:
        # Old format - check if simulate_human exists in edge_weights
        ga_params['simulate_human'] = edge_weights.get('simulate_human', True)
    else:
        ga_params['simulate_human'] = True
        
    ga_params['aug_names'] = aug_names

    ga_params['num_pos_needed'] = edge_weights.get('num_pos_needed', 300)
    ga_params['num_neg_needed'] = edge_weights.get('num_neg_needed', 50)

    ga_params['distance_power'] = lca_config.get('distance_power', 2)
    ga_params['scorer'] = edge_weights.get('scorer', 'kde')

    # Iterations parameters
    iterations = lca_config.get('iterations', {})
    ga_params['min_delta_converge_multiplier'] = float(iterations.get('min_delta_converge_multiplier', 0.95))
    ga_params['min_delta_stability_ratio'] = float(iterations.get('min_delta_stability_ratio', 4))
    ga_params['num_per_augmentation'] = int(iterations.get('num_per_augmentation', 2))
    ga_params['tries_before_edge_done'] = int(iterations.get('tries_before_edge_done', 4))
    ga_params['max_human_decisions'] = iterations.get('max_human_decisions', 5000)
    ga_params['ga_iterations_before_return'] = int(iterations.get('ga_iterations_before_return', 10))
    ga_params['ga_max_num_waiting'] = int(iterations.get('ga_max_num_waiting', 50))
    
    should_densify_str = str(iterations.get('should_densify', True)).lower()
    ga_params['should_densify'] = should_densify_str == 'true'
    ga_params['densify_min_number_human'] = iterations.get('densify_min_number_human', 10)
    ga_params['densify_min_edges'] = int(iterations.get('densify_min_edges', 10))
    ga_params['densify_frac'] = float(iterations.get('densify_frac', 0.5))

    # Logging parameters
    logging_config = lca_config.get('logging', {})
    ga_params['log_level'] = logging_config.get('log_level', 'INFO')
    ga_params['log_file'] = logging_config.get('log_file')

    # Drawing parameters
    drawing_config = lca_config.get('drawing', {})
    draw_iterations_str = str(drawing_config.get('draw_iterations', False)).lower()
    ga_params['draw_iterations'] = draw_iterations_str == 'true'
    ga_params['drawing_prefix'] = drawing_config.get('drawing_prefix', 'drawing_lca')

    return ga_params


def prepare_gc(common_data, config):
    """
    Graph Consistency preparation - always uses ClassifierManager.
    Supports both weighter-based and threshold-based classifiers.
    """
    gc_config = config.get('gc', {})
    edge_weights = config.get('edge_weights', config.get('lca', {}).get('edge_weights', {}))
    
    gc_config['theta'] = gc_config.get('theta', config.get('gc', 0.1).get('theta', 0.1))
    logger.info(f"THETA {gc_config['theta']}")
    print(f"THETA {gc_config['theta']}")
    gc_config['prob_human_correct'] = common_data.get('prob_human_correct', 0.98)
    gc_config['max_densify_edges'] = gc_config.get('max_densify_edges', 200)

    # Backwards compatibility: check for both verifier_names and augmentation_names
    verifier_names_str = edge_weights.get('verifier_names', edge_weights.get('augmentation_names', 'miewid human'))
    verifier_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str
    
    # Get threshold configuration if specified
    classifier_thresholds = edge_weights.get('classifier_thresholds', {})
    weight_thresholds = edge_weights.get('weight_thresholds', {})  # NEW

    algorithm_params = config.get("algorithm", {})
    lca_config = config.get("lca", {})
    tries_before_edge_done = algorithm_params.get('tries_before_edge_done', 
                                                 lca_config.get('iterations', {}).get('tries_before_edge_done', 4))
    gc_config["tries_before_edge_done"] = tries_before_edge_done
    # Create classifier units from existing common data
    classifier_units = {}

    
    do_robust_plot = "auto_threshold_plot_path" in config.get("logging", {}) 
    robust_plot_path = config.get("logging", {}).get("auto_threshold_plot_path", "dist.png")

    # Check if threshold is specified for this classifier
    for name in classifier_thresholds:
        if name not in common_data['embeddings_dict']:
            continue  # Skip if no embeddings for this classifier
        embeddings = common_data['embeddings_dict'][name]
        if isinstance(embeddings, types.FunctionType):
            embeddings = embeddings()
        
        # Use threshold-based classifier
        threshold = classifier_thresholds[name]
        if isinstance(threshold, str):
            if "auto" in threshold:
                # Parse threshold_fraction from auto string if provided
                match = re.match(r'auto\((\d+\.?\d*)\)', threshold)
                if match:
                    threshold_fraction = float(match.group(1))
                else:
                    threshold_fraction = 0.15  # default
                
                # Use unfiltered embeddings for threshold calculation if available
                unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
                if name in unfiltered_embeddings_dict:
                    thresh_embeddings = unfiltered_embeddings_dict[name]
                    if isinstance(thresh_embeddings, types.FunctionType):
                        thresh_embeddings = thresh_embeddings()
                    logger.info(f"Using unfiltered embeddings for auto threshold calculation for {name}")
                else:
                    thresh_embeddings = embeddings
                    logger.warning(f"No unfiltered embeddings available for {name}, using filtered embeddings for threshold")
                if hasattr(embeddings, 'get_base'):
                    thresh_embeddings = embeddings.get_base()
                # threshold = find_robust_threshold(
                #     np.array(thresh_embeddings.get_all_scores()), 
                #     threshold_fraction=threshold_fraction,
                #     print_func=logger.info, 
                #     debug_plots=do_robust_plot, 
                #     plot_path=robust_plot_path
                # )
                threshold = robust_gmm_find_threshold(
                    np.array(thresh_embeddings.get_all_scores()), 
                    verbose=True, 
                    print_func=logger.info,
                    plot_path=robust_plot_path
                )
            elif threshold in classifier_units:
                # Use existing threshold value
                classifier = classifier_units[threshold]
                logger.info(f"Using existing threshold-based classifier for {name} with threshold from {threshold}")
                classifier_units[name] = classifier
                continue
        
        classifier = ThresholdBasedClassifier(threshold)
        logger.info(f"Created threshold-based classifier for {name} with threshold {threshold}")
        classifier_units[name] = (embeddings, classifier)

    for name in verifier_names:
        if 'human' in name:
            continue  # Handled by human_reviewer
        if name not in classifier_units:
            embeddings = common_data['embeddings_dict'].get(name)
            
            # Use unfiltered embeddings for auto threshold calculation if available
            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
                logger.info(f"Using unfiltered embeddings for auto threshold calculation for {name}")
            else:
                thresh_embeddings = embeddings
                logger.warning(f"No unfiltered embeddings available for {name}, using filtered embeddings for threshold")
            
            # Fallback to default threshold
            # threshold = find_robust_threshold(np.array(thresh_embeddings.get_all_scores()), print_func=logger.info, debug_plots=do_robust_plot, plot_path=robust_plot_path)
            threshold = robust_gmm_find_threshold(
                    np.array(thresh_embeddings.get_all_scores()), 
                    verbose=True, 
                    print_func=logger.info,
                    plot_path=robust_plot_path,
                )
            classifier = ThresholdBasedClassifier(threshold)
            logger.warning(f"No weighter or threshold for {name}, using default auto threshold {threshold}")
            
            classifier_units[name] = (embeddings, classifier)
    
    # Create classifier manager (handles algorithmic classifiers only)
    classifier_manager = ClassifierManager(
        verifier_names=verifier_names,
        classifier_units=classifier_units
    )
    
    # Create GC instance
    gc_instance = GraphConsistencyAlgorithm(gc_config, 
                                            classifier_manager=classifier_manager, 
                                            cluster_validator=common_data['cluster_validator'])
    
    return gc_instance


def create_algorithm(config):
    """
    Factory function to create the appropriate algorithm instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (algorithm_instance, common_data)
    """
    # Setup logging first
    log_file, timestamp = setup_logging(config)
    if log_file is not None:
        print(f"Logging to {os.path.abspath(log_file)}")
    
    # Prepare common data
    common_data = prepare_common(config)
    common_data['timestamp'] = timestamp
    
    # Create algorithm based on config
    algorithm_type = config.get('algorithm_type', 'gc')  # Default to GC

    if algorithm_type == 'lca':
        algorithm = prepare_lca(common_data, config)
    elif algorithm_type == 'gc':
        algorithm = prepare_gc(common_data, config)
    elif algorithm_type == 'hdbscan':
        algorithm = prepare_hdbscan(common_data, config)
    elif algorithm_type == 'manual_review':
        algorithm = prepare_manual_review(common_data, config)
    elif algorithm_type == 'thresholded_review':
        algorithm = prepare_thresholded_review(common_data, config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    logger.info(f"Created {algorithm_type.upper()} algorithm instance")


    return algorithm, common_data


def prepare_hdbscan(common_data, config):
    """
    Prepare HDBSCAN algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        HDBSCANAlgorithm: Configured HDBSCAN algorithm instance
    """
    # Get all nodes from node2uuid mapping
    all_nodes = list(common_data['node2uuid'].keys())

    # Get embeddings dictionary
    embeddings_dict = common_data['embeddings_dict']

    # Create and return HDBSCAN instance
    hdbscan_instance = HDBSCANAlgorithm(config, all_nodes, embeddings_dict)

    return hdbscan_instance


def prepare_manual_review(common_data, config):
    """
    Prepare Manual Review algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        ManualReviewAlgorithm: Configured Manual Review algorithm instance
    """
    # Get manual_review config section if it exists, otherwise use defaults
    manual_config = config.get('manual_review', {})

    # Create and return Manual Review instance
    manual_instance = ManualReviewAlgorithm(manual_config, common_data)
    return manual_instance


def prepare_thresholded_review(common_data, config):
    """
    Prepare Thresholded Review algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        ThresholdedReviewAlgorithm: Configured Thresholded Review algorithm instance
    """
    # Get thresholded_review config section if it exists, otherwise use defaults
    thresholded_config = config.get('thresholded_review', {})

    # Create and return Thresholded Review instance
    thresholded_instance = ThresholdedReviewAlgorithm(thresholded_config, common_data)
    return thresholded_instance


def call_verifier_alg(embeddings):
    """Helper function to create verifier algorithm from embeddings."""
    def verifier_alg(edge_nodes):
        logger = logging.getLogger('lca')
        scores = [embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
        logger.info(f'Scores  {scores} ')
        return scores
    return verifier_alg

