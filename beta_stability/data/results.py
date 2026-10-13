"""Writing out what a run produced.

Two artefacts per run, and both are contracts with code outside this package:

* the graph files, read by the analysis scripts that build the paper's tables;
* the annotation file, a copy of the input annotations with a cluster id added
  to each one, read by the downstream curation tooling.

Their names come from the config, defaulting to the values in
:data:`beta_stability.config.DEFAULTS`.
"""

import json
import logging
import os

from beta_stability.data.io import write_json

logger = logging.getLogger("beta_stability")


def write_graph(config, clustering, node2cid, node2uuid, graph, output_path):
    """Write the per-run graph files."""
    output = config['output']
    for key, payload in (('clustering_file', clustering),
                         ('node2cid_file', node2cid),
                         ('node2uuid_file', node2uuid),
                         ('graph_file', graph)):
        write_json(payload, os.path.join(output_path, output[key]))


def uuid_to_cluster_map(clustering, node2uuid, offset=0):
    """Map each annotation UUID to its cluster id, renumbered from ``offset``.

    Field-separated runs are independent, so their cluster ids collide. The
    offset makes them unique across the combined output. Returns the mapping
    and the next free id.
    """
    # The algorithm keys node2uuid by int; the same map read back from JSON is
    # keyed by str. Normalise so either source works.
    by_node = {str(node): uuid for node, uuid in node2uuid.items()}

    mapping = {}
    for nodes in clustering.values():
        new_id = str(offset)
        offset += 1
        for node in nodes:
            uuid = by_node.get(str(node))
            if uuid:
                mapping[uuid] = new_id
    return mapping, offset


def annots_filename(config, field_filters=None):
    """Build the output filename from the configured prefix and suffix."""
    output = config['output']
    parts = [p for p in (output['prefix'],) if p]
    if field_filters:
        parts.append('_'.join(f"{k}-{v}" for k, v in field_filters.items()))
    if output['suffix']:
        parts.append(output['suffix'])
    return '_'.join(parts) + '.json' if parts else output['annotations_name']


def write_annotations(config, uuid_to_cluster, save_dir, field_filters=None):
    """Copy the annotation file through, adding the cluster id to each annotation.

    The structure of the input file is preserved exactly, which is the contract
    the downstream tooling relies on. Annotations with no cluster get ``None``.
    """
    data_config = config['data']
    uuid_key, output_key = data_config['id_key'], data_config['output_key']

    with open(data_config['annotation_file'], 'r') as f:
        original = json.load(f)

    result = {k: original[k] for k in ('categories', 'images') if k in original}

    annotations = original.get('annotations')
    if annotations is not None:
        if field_filters:
            annotations = [a for a in annotations if matches(a, field_filters)]
            logger.info(f"Filtered {len(annotations)} annotations with "
                        f"{', '.join(f'{k}={v}' for k, v in field_filters.items())}")
        updated, assigned = [], 0
        for annotation in annotations:
            annotation = dict(annotation)
            cluster_id = uuid_to_cluster.get(annotation.get(uuid_key))
            annotation[output_key] = cluster_id
            assigned += cluster_id is not None
            updated.append(annotation)
        logger.info(f"Assigned cluster IDs to {assigned}/{len(updated)} annotations")
        result['annotations'] = updated

    path = os.path.join(save_dir, annots_filename(config, field_filters))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_json(result, path)
    logger.info(f"Saved clustering results to {path}")
    return path


def matches(annotation, field_filters):
    """True when an annotation satisfies every field filter, by substring.

    A filter matches either the field itself or its ``name_<field>`` variant,
    because the annotation files use both spellings.
    """
    for field, value in field_filters.items():
        field_value = annotation.get(field) or annotation.get(f"name_{field}")
        if field_value is None or str(value) not in str(field_value):
            return False
    return True
