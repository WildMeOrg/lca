import numpy as np
from preprocess import preprocess_data
from embeddings import Embeddings
from curate_using_LCA import curate_using_LCA, generate_wgtr_calibration_ground_truth
from tools import load_pickle, print_intersect_stats, get_config
from cluster_validator import ClusterValidator
import ga_driver
from init_logger import init_logger
import argparse
from sklearn.metrics import pairwise_distances_chunked


def get_score_from_cosine_distance(cosine_dist):
    """Convert cosine distance to a score."""
    return 1 - cosine_dist * 0.5


def get_topk_acc(labels_q, labels_db, dists, topk):
    """Compute top-k accuracy for label queries."""
    return sum(get_topk_hits(labels_q, labels_db, dists, topk)) / len(labels_q)


def get_topk_hits(labels_q, labels_db, dists, topk):
    """Return whether the correct label is in the top-k closest predictions."""
    indices = np.argsort(dists, axis=1)
    top_labels = np.array(labels_db)[indices[:, :topk]]
    hits = (top_labels.T == labels_q).T
    return np.sum(hits[:, :topk+1], axis=1) > 0


def get_top_ks(q_pids, distmat, ks=[1, 3, 5, 10]):
    """Calculate top-k accuracies for the given distance matrix."""
    return [(k, get_topk_acc(q_pids, q_pids, distmat, k)) for k in ks]


def calculate_distances(embeddings, ids, uuids, reduce_func):
    """Calculate pairwise distances and apply a reduction function."""
    print(f"Calculating distances for {len(embeddings)} embeddings and {len(ids)} IDs...")

    chunks = pairwise_distances_chunked(
        embeddings,
        metric='cosine',
        reduce_func=reduce_func,
        n_jobs=-1
    )
    return np.concatenate(list(chunks), axis=0)


def prepare_reduce_func():
    """Prepare the distance reduction function that applies cosine distance transformation."""
    def reduce_func(distmat, start):
        distmat = 1 - get_score_from_cosine_distance(distmat)
        np.fill_diagonal(distmat, np.inf)
        return distmat
    return reduce_func


def get_stats(df, filter_key, embeddings, id_key='uuid'):
    """Compute statistics based on distance matrix and top-k accuracy."""
    uuids = {i: row[id_key] for i, row in df.iterrows()}
    ids = list(uuids.keys())

    reduce_func = prepare_reduce_func()

    # Calculate distance matrix
    distmat = calculate_distances(embeddings, ids, uuids, reduce_func)

    # Map labels and compute top-k accuracies
    labels = [df.loc[df[id_key] == uuids[id], filter_key].values[0] for id in ids]
    return get_top_ks(labels, distmat, ks=[1, 3, 5, 10])


def run(config):
    """Main pipeline function to process embeddings and calculate statistics."""
    data_params = config['data']
    
    # Load embeddings and UUIDs
    embeddings, uuids = load_pickle(data_params['embedding_file'])

    # Preprocess data
    name_keys = data_params['name_keys']
    filter_key = '__'.join(name_keys)
    
    df = preprocess_data(
        data_params['annotation_file'],
        name_keys=name_keys,
        convert_names_to_ids=True,
        viewpoint_list=data_params['viewpoint_list'],
        n_filter_min=data_params['n_filter_min'],
        n_filter_max=data_params['n_filter_max'],
        images_dir=data_params['images_dir'],
        embedding_uuids=uuids
    )

    print_intersect_stats(df, individual_key=filter_key)

    # Filter dataframe by UUIDs
    filtered_df = df[df['uuid_x'].isin(uuids)]
    print('     ', len(filtered_df), 'annotations remain after filtering by the provided embeddings')
    filtered_embeddings = [embeddings[uuids.index(uuid)] for uuid in filtered_df['uuid_x']]
    print('     ', len(filtered_embeddings), 'embeddings remain after filtering by the provided annotations')

    # Compute statistics
    topk_results = get_stats(filtered_df, filter_key, filtered_embeddings)

    print(f"Statistics: {', '.join([f'top-{k}: {100*v:.2f}%' for (k, v) in topk_results])}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()


if __name__ == '__main__':
    init_logger()
    args = parse_args()
    config = get_config(args.config)
    run(config)
