"""The single runner must write exactly what the superseded runners wrote.

The annotation file produced by a run is consumed by the analysis scripts, so
its structure, its field names and its filename are a contract. These tests
pin that contract by comparing the current implementation against the
superseded one, which is kept in ``beta_stability/old/superseded`` for exactly
this purpose.
"""

import importlib.util
import json
import os
import pathlib
import sys

import pytest

from beta_stability import config as config_module
from beta_stability.data import results

SUPERSEDED = pathlib.Path(__file__).resolve().parents[1] / 'beta_stability' / 'old' / 'superseded'


def _load_superseded(name):
    """Import a superseded module by path, without putting it on the package."""
    path = SUPERSEDED / f'{name}.py'
    if not path.exists():
        pytest.skip(f'{path} is not present')
    spec = importlib.util.spec_from_file_location(f'_superseded_{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

@pytest.mark.parametrize('output', [
    {},
    {'prefix': 'gzcd'},
    {'suffix': 'run1'},
    {'prefix': 'gzcd', 'suffix': 'run1'},
])
def test_annotation_output_matches_superseded(dataset, tmp_path, output):
    """Same clustering in, byte-identical annotation file out."""
    old = _load_superseded('save_clustering_results')
    clustering = {'0': [0, 1], '1': [2], '2': [3, 4]}
    node2uuid = {'0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e'}

    # the superseded writer reads the graph files off disk
    old_in = tmp_path / 'old_in'
    old_in.mkdir()
    (old_in / 'clustering.json').write_text(json.dumps(clustering))
    (old_in / 'node2uuid_file.json').write_text(json.dumps(node2uuid))
    old_out = tmp_path / 'old_out'
    old_out.mkdir()
    old.save_clustering_results(str(old_in), str(dataset), str(old_out),
                                output.get('prefix'), output.get('suffix'),
                                field_filters=None, uuid_key='uuid', output_key='cluster_id')

    # the current writer is handed the same clustering in memory
    new_out = tmp_path / 'new_out'
    new_out.mkdir()
    config = _config(dataset, tmp_path, **output)
    mapping, _ = results.uuid_to_cluster_map(clustering, node2uuid)
    written = results.write_annotations(config, mapping, str(new_out))

    old_files = sorted(p.name for p in old_out.iterdir())
    assert old_files == sorted(p.name for p in new_out.iterdir()), 'filename differs'
    assert json.loads(pathlib.Path(written).read_text()) == \
        json.loads((old_out / old_files[0]).read_text())


def test_field_separated_ids_are_unique_and_offset_like_superseded(dataset, tmp_path):
    """Per-field runs are renumbered into one combined file, as before."""
    old = _load_superseded('save_clustering_results')
    per_field = {
        'viewpoint-left': ({'0': [0, 1]}, {'0': 'a', '1': 'b'}),
        'viewpoint-right': ({'0': [0], '1': [1, 2]}, {'0': 'c', '1': 'd', '2': 'e'}),
    }
    base = tmp_path / 'base'
    for name, (clustering, node2uuid) in per_field.items():
        d = base / name
        d.mkdir(parents=True)
        (d / 'clustering.json').write_text(json.dumps(clustering))
        (d / 'node2uuid_file.json').write_text(json.dumps(node2uuid))

    old_out = tmp_path / 'old_out'
    old_out.mkdir()
    old.combine_field_separated_results(str(base), str(dataset), str(old_out), None, None,
                                        ['viewpoint'], uuid_key='uuid', output_key='cluster_id')

    mapping, offset = {}, 0
    for name in sorted(per_field):                      # the superseded code globs sorted()
        clustering, node2uuid = per_field[name]
        combo, offset = results.uuid_to_cluster_map(clustering, node2uuid, offset)
        mapping.update(combo)
    new_out = tmp_path / 'new_out'
    new_out.mkdir()
    written = results.write_annotations(_config(dataset, tmp_path), mapping, str(new_out))

    assert offset == 3, 'three clusters across the two field runs'
    assert json.loads(pathlib.Path(written).read_text()) == \
        json.loads((old_out / 'beta_stability_annots.json').read_text())


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
