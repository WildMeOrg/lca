"""Every algorithm type runs end to end and produces a usable clustering.

The five baselines produce the comparison numbers in the paper and nothing
checked that they still run, let alone that they still agree with themselves.
This is that check: for each `algorithm_type` the factory accepts, build it
from a config, run it on the generated dataset, and assert that the result is
a complete clustering of the right shape.

These are not accuracy tests. The dataset is twelve well separated individuals,
so a method that recovers them is doing the easiest possible job and a method
that does not is broken rather than merely weaker. What the assertions pin is
that the path from config through preparation to the written annotation file
works for every type, and that two runs of one config agree.
"""

import pytest

from beta_stability import config as config_module
from beta_stability import run
from tests.conftest import INDIVIDUALS, PER_INDIVIDUAL, read_json

# What each type needs beyond the shared config. The review baselines build
# their own candidate list from their own `topk`, so they must not be handed
# the shared initial edges; `initial_topk: 0` is how their configs say that.
SETTINGS = {
    'stability': {
        'algorithm': {'initial_topk': 10},
        'stability': {'target_alpha': 0.5, 'max_human_reviews': 100,
                      'edges_per_review_batch': 20, 'validation_step': 20},
    },
    'manual_review': {
        'algorithm': {'initial_topk': 0},
        'manual_review': {'clustering_method': 'connected_components', 'max_reviews': 100,
                          'review_batch_size': 20, 'threshold': None, 'topk': 5},
    },
    'thresholded_review': {
        'algorithm': {'initial_topk': 0},
        'thresholded_review': {'low_threshold': 0.65, 'high_threshold': 0.9, 'topk': 5,
                               'max_reviews': 100, 'review_batch_size': 20,
                               'clustering_method': 'connected_components'},
    },
    'hdbscan': {
        'algorithm': {'initial_topk': 10},
        'hdbscan': {'min_cluster_size': 2, 'min_samples': None, 'metric': 'euclidean',
                    'cluster_selection_method': 'eom', 'verifier_name': 'miewid'},
    },
    'np3_aas': {
        'algorithm': {'initial_topk': 10, 'initial_botk': 0},
        'np3_aas': {'dbscan_eps': 'auto', 'dbscan_min_samples': 2, 'dbscan_metric': 'cosine',
                    's_min': 0.3, 'k_max': 5, 'epsilon': 0.6, 'max_human_reviews': 100,
                    'edges_per_review_batch': 20, 'validation_step': 20,
                    'verifier_name': 'miewid'},
    },
    'nis': {
        'algorithm': {'initial_topk': 10, 'initial_botk': 0},
        'nis': {'n_sampled_vertices': 10, 'm_neighbors_per_vertex': 20,
                'max_human_reviews': 100, 'edges_per_review_batch': 20,
                'validation_step': 20, 'verifier_name': 'miewid'},
    },
}


def build_config(dataset, algorithm_type, output_name):
    root, anno_file, embedding_file = dataset
    output = root / output_name
    base = {
        'algorithm_type': algorithm_type,
        'seed': 0,
        'exp_name': f'baseline_{algorithm_type}',
        'species': 'test',
        'data': {
            'annotation_file': str(anno_file),
            'embedding_file': str(embedding_file),
            'name_keys': ['individual_id'],
            'n_filter_min': 1,
            'n_filter_max': 100,
            'output_path': str(output),
        },
        'edge_weights': {'verifier_names': 'miewid simulated_human',
                         'prob_human_correct': 1.0},
        'logging': {'log_file': str(output / 'run.log'), 'file_mode': 'w'},
    }
    for section, values in SETTINGS[algorithm_type].items():
        base[section] = {**base.get(section, {}), **values}
    return config_module.load_mapping(base)


def clustering_of(annots_path):
    """{uuid: cluster id} from a written annotation file."""
    annotations = read_json(annots_path)['annotations']
    return {a['uuid']: a['cluster_id'] for a in annotations}


@pytest.fixture(scope='module', params=sorted(SETTINGS))
def completed(request, dataset):
    """One run per algorithm type, shared by the assertions below."""
    algorithm_type = request.param
    path = run.run_from_config(build_config(dataset, algorithm_type, algorithm_type))
    return algorithm_type, path


def test_every_algorithm_type_produces_a_complete_clustering(completed):
    algorithm_type, annots_path = completed
    assignment = clustering_of(annots_path)

    assert len(assignment) == INDIVIDUALS * PER_INDIVIDUAL, 'every annotation is written back'
    unassigned = [uuid for uuid, cluster in assignment.items() if cluster is None]
    assert not unassigned, f'{algorithm_type} left {len(unassigned)} annotations unclustered'
    assert len(set(assignment.values())) > 1, f'{algorithm_type} put everything in one cluster'


def test_no_algorithm_type_mixes_individuals_under_a_perfect_reviewer(completed):
    """On data this separable, a cluster holding two individuals is a defect."""
    algorithm_type, annots_path = completed
    by_cluster = {}
    for a in read_json(annots_path)['annotations']:
        by_cluster.setdefault(a['cluster_id'], set()).add(a['individual_id'])
    impure = {c: sorted(ids) for c, ids in by_cluster.items() if len(ids) > 1}
    assert not impure, f'{algorithm_type} mixed individuals: {impure}'


def test_every_algorithm_type_is_deterministic(dataset, completed):
    """Two runs of one config must agree, or no result from it means anything."""
    algorithm_type, first_path = completed
    again = run.run_from_config(build_config(dataset, algorithm_type, f'{algorithm_type}_again'))
    assert clustering_of(again) == clustering_of(first_path)
