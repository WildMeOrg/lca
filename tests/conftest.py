"""A small, separable dataset that every end-to-end test runs on.

Twelve individuals with five annotations each, embedded around orthonormal
per-individual centres with a fixed seed. Same-individual similarity is at least
0.69 and different-individual at most 0.23, so the data is separable by
construction and a test measures the pipeline rather than the difficulty of the
data. Generated rather than committed, so there is no binary in the repository.
"""

import json
import pickle

import numpy as np
import pytest

INDIVIDUALS, PER_INDIVIDUAL, DIM, NOISE = 12, 5, 64, 0.5


@pytest.fixture(scope='session')
def dataset(tmp_path_factory):
    """Write an annotation file and matching embeddings, deterministically."""
    root = tmp_path_factory.mktemp('dataset')
    rng = np.random.default_rng(0)
    # orthonormal centres, so two different individuals are near-orthogonal
    centres = np.linalg.qr(rng.normal(size=(DIM, INDIVIDUALS)))[0].T
    annotations, images, embeddings, ids = [], [], [], []
    for individual in range(INDIVIDUALS):
        for sighting in range(PER_INDIVIDUAL):
            uuid, image = f'u{individual:02d}_{sighting}', f'img{individual:02d}_{sighting}'
            # noise scaled by 1/sqrt(DIM) so its norm does not grow with dimension
            vector = centres[individual] + NOISE * rng.normal(size=DIM) / np.sqrt(DIM)
            vector /= np.linalg.norm(vector)
            annotations.append({'uuid': uuid, 'image_uuid': image, 'viewpoint': 'left',
                                'individual_id': f'ind{individual:02d}', 'species': 'test'})
            images.append({'uuid': image, 'file_name': f'{image}.jpg'})
            embeddings.append(vector)
            ids.append(uuid)

    anno_file = root / 'annots.json'
    anno_file.write_text(json.dumps({'categories': [{'id': 1, 'name': 'test'}],
                                     'images': images, 'annotations': annotations}))
    embedding_file = root / 'embeds.pkl'
    with open(embedding_file, 'wb') as f:
        pickle.dump((np.asarray(embeddings, dtype=np.float32), ids), f)
    return root, anno_file, embedding_file


def read_json(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope='session')
def dataset_without_identities(dataset, tmp_path_factory):
    """The same annotations with the identity field removed: a production run."""
    anno_file, embedding_file = dataset[1], dataset[2]
    stripped = tmp_path_factory.mktemp('no_identities')
    data = read_json(anno_file)
    for annotation in data['annotations']:
        annotation.pop('individual_id', None)
    target = stripped / 'annots.json'
    target.write_text(json.dumps(data))
    return stripped, target, embedding_file
