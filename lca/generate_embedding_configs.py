"""
Generate embedding-specific config files and runner scripts for all baselines.

For each (species, embedding, method) combination, this script:
  1. Reads the miewid template config
  2. Overrides embedding_file, output_path, and log_file with embedding-suffixed paths
  3. Writes a new config file with the embedding suffix

It also generates a shell script that runs every baseline for every embedding.

Usage:
    python3 generate_embedding_configs.py

Assumptions:
  - MegaDescriptor and AlFReID embedding files follow naming convention:
    {embedding}_embeddings_{species}.pickle in the same directory as miewid's
  - Output paths get _{embedding} suffix
  - Log paths get _{embedding} suffix before _output.log

Methods covered: hdbscan, manual_review, thresholded_review, np3_aas
NIS and GT Top-K need updates to run_nis_reference.py DATASET_CONFIGS.
"""

import os
import re
from pathlib import Path

import yaml

CONFIGS_DIR = Path('/users/PAS2136/nepove/code/lca/lca/configs')
RUN_SCRIPT = Path('/users/PAS2136/nepove/code/lca/lca/run_baselines_embedding.sh')

SPECIES = ['beluga', 'forestelephants', 'GZCD', 'giraffe', 'lion', 'plainszebra', 'whaleshark']
EMBEDDINGS = ['megadescriptor', 'alfreid']

# (method_key, template_filename_format)
# The method_key maps to the file naming pattern.
METHODS = {
    'hdbscan':        'config_{species}_hdbscan.yaml',
    'np3_aas':        'config_{species}_np3_aas.yaml',
    'manual_review':  'config_manual_review_{species}.yaml',
    'thresholded':    'config_thresholded_review_{species}.yaml',
}


def deep_set(d, key_path, value):
    """Set d[a][b][c] = value given key_path='a.b.c'."""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def embedding_pickle_path(species, embedding):
    """Canonical path for an embedding pickle, regardless of the species-specific
    miewid path (some species, e.g. GZCD, keep miewid in a different directory).
    """
    return (f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species}/'
            f'{embedding}_embeddings_{species}.pickle')


def rewrite_path(path_str, embedding, species=None):
    """Replace miewid-specific path fragments with embedding-suffixed ones."""
    if not path_str or not isinstance(path_str, str):
        return path_str
    new = path_str

    # Embedding pickle: replace with the canonical embedding path for the species
    if (new.endswith('.pickle') and species is not None
            and 'embedding' in new.lower() or 'miewid' in new.lower()):
        # Heuristic: if it's any .pickle path (miewid's varied formats),
        # redirect to the canonical embedding pickle path for this species.
        if new.endswith('.pickle'):
            new = embedding_pickle_path(species, embedding)

    # Log file: ..._output.log -> ..._{embedding}_output.log
    if new.endswith('_output.log'):
        new = new[:-len('_output.log')] + f'_{embedding}_output.log'
    elif new.endswith('.log'):
        new = new[:-len('.log')] + f'_{embedding}.log'

    # Output directory: ..._clustering/ -> ..._{embedding}_clustering/
    if new.endswith('_clustering/') or new.endswith('_clustering'):
        trailing = '/' if new.endswith('/') else ''
        stem = new.rstrip('/')
        new = stem[:-len('_clustering')] + f'_{embedding}_clustering' + trailing

    return new


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def embedding_exists(species, embedding):
    """Check whether the embedding pickle file exists for a given species."""
    path = Path(f'/fs/ess/PAS2136/ggr_data/kate/data_embeddings/{species}/'
                f'{embedding}_embeddings_{species}.pickle')
    return path.exists()


def generate_configs():
    commands = []
    # Track which (species, embedding) combos are valid for the runner
    valid_combos = {}
    for species in SPECIES:
        species_dir = CONFIGS_DIR / species
        if not species_dir.exists():
            print(f'SKIP: {species_dir} does not exist')
            continue

        for embedding in EMBEDDINGS:
            if not embedding_exists(species, embedding):
                print(f'SKIP: no {embedding} embedding for {species}')
                continue
            valid_combos[(species, embedding)] = True

        for method, template_fmt in METHODS.items():
            template = species_dir / template_fmt.format(species=species)
            if not template.exists():
                print(f'SKIP: template missing {template}')
                continue

            for embedding in EMBEDDINGS:
                if not valid_combos.get((species, embedding)):
                    continue

                cfg = load_yaml(template)

                # Rewrite all string paths
                for section_key in ('data', 'logging'):
                    section = cfg.get(section_key) or {}
                    for k, v in list(section.items()):
                        section[k] = rewrite_path(v, embedding, species=species)

                # Build output config path
                out_name = template.stem + f'_{embedding}.yaml'
                out_path = template.parent / out_name
                dump_yaml(cfg, out_path)
                print(f'WROTE: {out_path}')

                # Build run command (relative to lca/ working dir)
                rel_cfg = out_path.relative_to(CONFIGS_DIR.parent)
                commands.append((species, embedding, method, str(rel_cfg)))

    return commands


def generate_runner(commands):
    lines = [
        '#!/bin/bash',
        '#',
        '# Run all baselines for each embedding (megadescriptor, alfreid).',
        '# Generated by generate_embedding_configs.py',
        '#',
        '# Continues on failure so one crash does not halt the whole run.',
        '# Check individual log files afterwards for errors.',
        '',
    ]

    # Group by embedding for readability
    by_embedding = {}
    for species, embedding, method, cfg in commands:
        by_embedding.setdefault(embedding, []).append((species, method, cfg))

    for embedding, entries in by_embedding.items():
        lines.append(f'# === {embedding} ===')
        for species, method, cfg in entries:
            lines.append(f'echo "=== {method} | {species} | {embedding} ==="')
            lines.append(f'python3 run_clustering_with_save.py --config {cfg} || echo "FAILED: {method} {species} {embedding}"')
            lines.append('')

        lines.append(f'# --- GT Top-K and NIS for {embedding} ---')
        for species in SPECIES:
            if not embedding_exists(species, embedding):
                continue
            lines.append(f'echo "=== gt_topk | {species} | {embedding} ==="')
            lines.append(f'python3 -u run_gt_topk.py --dataset {species} --topk 10 --embedding {embedding} || echo "FAILED: gt_topk {species} {embedding}"')
            lines.append(f'echo "=== nis | {species} | {embedding} ==="')
            lines.append(f'python3 -u run_nis_reference.py --dataset {species} --sweep --budget_max 5000 --budget_steps 50 --embedding {embedding} || echo "FAILED: nis {species} {embedding}"')
            lines.append('')

    RUN_SCRIPT.write_text('\n'.join(lines))
    RUN_SCRIPT.chmod(0o755)
    print(f'\nWROTE runner: {RUN_SCRIPT}')


if __name__ == '__main__':
    commands = generate_configs()
    generate_runner(commands)
    print(f'\nGenerated {len(commands)} configs')
    print(f'Run with: ./run_baselines_embedding.sh')
