"""The annotation file a run writes is a contract.

Downstream curation reads that file, so its filename, its structure and its
field names are pinned here: the input file comes back unchanged with one field
added per annotation, and per-field runs are renumbered into a single file.
"""

import json
import os
import pathlib

import pytest

from beta_stability import config as config_module
from beta_stability.data import results

@pytest.fixture
def dataset(tmp_path):
    """A small annotation file with two viewpoints and a nested structure."""
    annotations = [
        {'uuid': 'a', 'viewpoint': 'left', 'individual_id': 1, 'bbox': [0, 0, 1, 1]},
        {'uuid': 'b', 'viewpoint': 'left', 'individual_id': 1, 'bbox': [1, 1, 2, 2]},
        {'uuid': 'c', 'viewpoint': 'right', 'individual_id': 2, 'bbox': [2, 2, 3, 3]},
        {'uuid': 'd', 'viewpoint': 'right', 'individual_id': 3, 'bbox': [3, 3, 4, 4]},
        {'uuid': 'e', 'viewpoint': 'right', 'individual_id': 3, 'bbox': [4, 4, 5, 5]},
    ]
    anno_file = tmp_path / 'annots.json'
    anno_file.write_text(json.dumps({
        'categories': [{'id': 1, 'name': 'zebra'}],
        'images': [{'uuid': 'img1'}],
        'annotations': annotations,
    }))
    return anno_file


def _config(anno_file, tmp_path, **output):
    """A config as the loader produces it: declared defaults plus these values."""
    return config_module.load_mapping({
        'data': {'annotation_file': str(anno_file), 'output_path': str(tmp_path / 'out')},
        'output': output,
    })


# --------------------------------------------------------------------------
# the output contract
# --------------------------------------------------------------------------

CLUSTERING = {'0': [0, 1], '1': [2], '2': [3, 4]}
NODE2UUID = {'0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e'}


@pytest.mark.parametrize(('output', 'filename'), [
    ({}, 'beta_stability_annots.json'),
    ({'prefix': 'gzcd'}, 'gzcd.json'),
    ({'suffix': 'run1'}, 'run1.json'),
    ({'prefix': 'gzcd', 'suffix': 'run1'}, 'gzcd_run1.json'),
])
def test_the_file_is_named_from_the_prefix_and_suffix(dataset, tmp_path, output, filename):
    mapping, _ = results.uuid_to_cluster_map(CLUSTERING, NODE2UUID)
    out = tmp_path / 'out'
    out.mkdir()
    written = results.write_annotations(_config(dataset, tmp_path, **output), mapping, str(out))
    assert pathlib.Path(written).name == filename
    assert sorted(p.name for p in out.iterdir()) == [filename]


def test_the_file_is_the_input_with_a_cluster_id_added(dataset, tmp_path):
    original = json.loads(dataset.read_text())
    mapping, _ = results.uuid_to_cluster_map(CLUSTERING, NODE2UUID)
    out = tmp_path / 'out'
    out.mkdir()
    written = results.write_annotations(_config(dataset, tmp_path), mapping, str(out))
    body = json.loads(pathlib.Path(written).read_text())

    assert list(body) == ['categories', 'images', 'annotations']
    assert body['categories'] == original['categories']
    assert body['images'] == original['images']
    for before, after in zip(original['annotations'], body['annotations'], strict=True):
        assert list(after) == [*before, 'cluster_id'], 'cluster_id is appended last'
        assert {key: after[key] for key in before} == before, 'input fields are untouched'
    assert {a['uuid']: a['cluster_id'] for a in body['annotations']} == \
        {'a': '0', 'b': '0', 'c': '1', 'd': '2', 'e': '2'}


def test_field_separated_runs_are_renumbered_into_one_file(dataset, tmp_path):
    """Each per-field run numbers its clusters from zero, so combining offsets them."""
    per_field = {
        'viewpoint-left': ({'0': [0, 1]}, {'0': 'a', '1': 'b'}),
        'viewpoint-right': ({'0': [0], '1': [1, 2]}, {'0': 'c', '1': 'd', '2': 'e'}),
    }
    mapping, offset = {}, 0
    for name in sorted(per_field):
        clustering, node2uuid = per_field[name]
        combo, offset = results.uuid_to_cluster_map(clustering, node2uuid, offset)
        mapping.update(combo)
    out = tmp_path / 'out'
    out.mkdir()
    written = results.write_annotations(_config(dataset, tmp_path), mapping, str(out))
    body = json.loads(pathlib.Path(written).read_text())

    assert offset == 3, 'three clusters across the two field runs'
    assert pathlib.Path(written).name == 'beta_stability_annots.json'
    assert {a['uuid']: a['cluster_id'] for a in body['annotations']} == \
        {'a': '0', 'b': '0', 'c': '1', 'd': '2', 'e': '2'}


def test_unclustered_annotations_get_a_null_cluster_id(dataset, tmp_path):
    config = _config(dataset, tmp_path)
    mapping, _ = results.uuid_to_cluster_map({'0': [0]}, {'0': 'a'})
    written = results.write_annotations(config, mapping, str(tmp_path))
    annotations = json.loads(pathlib.Path(written).read_text())['annotations']
    assert [a['cluster_id'] for a in annotations] == ['0', None, None, None, None]
    assert all('bbox' in a for a in annotations), 'original fields must survive'


def test_field_filter_selects_by_substring(dataset, tmp_path):
    config = _config(dataset, tmp_path)
    mapping, _ = results.uuid_to_cluster_map({'0': [0, 1]}, {'0': 'a', '1': 'b'})
    written = results.write_annotations(config, mapping, str(tmp_path),
                                    field_filters={'viewpoint': 'left'})
    annotations = json.loads(pathlib.Path(written).read_text())['annotations']
    assert [a['uuid'] for a in annotations] == ['a', 'b']
    assert pathlib.Path(written).name == 'viewpoint-left.json'


# --------------------------------------------------------------------------
# config handling
# --------------------------------------------------------------------------


def test_run_tag_redirects_every_output_path(tmp_path):
    config = {
        'data': {'output_path': str(tmp_path / 'out')},
        'logging': {'log_file': str(tmp_path / 'logs' / 'run.log'),
                    'auto_threshold_plot_path': str(tmp_path / 'p' / 'thr.png')},
        'stability': {'stability_track_path': str(tmp_path / 's' / 'track.json')},
    }
    config_module.apply_run_tag(config, seed=7, prob_human_correct=0.95)
    assert config['seed'] == 7
    assert config['edge_weights']['prob_human_correct'] == 0.95
    assert config['run']['tag'] == os.path.join('phc95', 'seed7')
    for value in (config['data']['output_path'], config['logging']['log_file'],
                  config['logging']['auto_threshold_plot_path'],
                  config['stability']['stability_track_path']):
        assert 'phc95' in value and 'seed7' in value
    assert pathlib.Path(config['data']['output_path']).is_dir(), 'output dir is created'


def test_no_sweep_leaves_paths_alone():
    config = {'data': {'output_path': 'out'}, 'logging': {'log_file': 'run.log'}}
    tagged = config_module.apply_run_tag(dict(config))
    assert tagged['data']['output_path'] == 'out'
    assert tagged['logging']['log_file'] == 'run.log'
    assert tagged['run']['tag'] == ''


def test_output_filenames_come_from_the_config(dataset, tmp_path):
    cfg = _config(dataset, tmp_path)
    cfg['output']['clustering_file'] = 'clusters.json'
    cfg['output']['graph_file'] = 'g.json'
    out = tmp_path / 'graphout'
    out.mkdir()
    results.write_graph(cfg, {'0': [0]}, {'0': '0'}, {'0': 'a'}, {}, str(out))
    assert sorted(p.name for p in out.iterdir()) == \
        ['clusters.json', 'g.json', 'node2cid.json', 'node2uuid_file.json']
