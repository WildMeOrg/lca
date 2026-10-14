#!/usr/bin/env python3
"""Reproduce the paper's GZCD run from the public dataset.

Downloads the Grevy's Zebra Census Dataset from
https://huggingface.co/datasets/imageomics/GZCD, selects the annotations the
paper used, embeds them with MiewID v3, and runs beta-stability on them.

    uv sync --extra dev --extra embeddings
    uv run python scripts/reproduce_gzcd.py --output-dir build/gzcd

It writes, into the output directory, the two annotation files, the embeddings,
the configuration that ran and the resulting clustering. Each step is skipped if
its output is already there, and --stop-after ends at any of `metadata`,
`images`, `annotations` or `embeddings`.
"""

import argparse
import csv
import json
import pathlib
import pickle
import sys

import cv2
import numpy as np
import torch
import yaml
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from beta_stability import config as config_module
from beta_stability import run

DATASET = 'imageomics/GZCD'
MODEL = 'conservationxlabs/miewid-msv3'
METADATA = 'final_metadata.json'

#: The viewpoints the run keeps, matching the configuration's viewpoint_list.
VIEWPOINTS = ['right', 'backright', 'downright', 'frontright']

#: The annotations to exclude, keyed by the release's `annot_uuid`.
EXCLUSIONS = pathlib.Path(__file__).resolve().parent / 'gzcd_excluded_annotations.csv'

#: Every parameter of the reported run. The script fills in the paths only.
CONFIG = pathlib.Path(__file__).resolve().parent / 'gzcd.yaml'

#: The batch size the published embeddings were computed at. Batch size changes
#: an embedding in its low-order bits, so this is kept rather than tuned.
BATCH_SIZE = 4

#: MiewID v3 was trained at 440px with ImageNet normalisation. Changing either
#: destroys the embedding: 660, 880 and 1200 take edge-F1 from 0.606 to 0.125.
IMAGE_SIZE = 440
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_MAX_VALUE = 255.0


def log(message):
    print(f'[gzcd] {message}', flush=True)


# --------------------------------------------------------------------------
# 1. the dataset
# --------------------------------------------------------------------------

def download_metadata(output_dir):
    path = output_dir / METADATA
    if path.exists():
        log(f'metadata already at {path}')
    else:
        log(f'downloading {DATASET}/{METADATA}')
        hf_hub_download(DATASET, METADATA, repo_type='dataset', local_dir=str(output_dir))
    return json.loads(path.read_text())


def box(value):
    """A box as four integers, whether it arrives as a list or as its text."""
    if isinstance(value, str):
        value = json.loads(value)
    return [round(float(v)) for v in value]


def contains(outer, inner, slack=2):
    """Whether `inner` sits inside `outer`, allowing two pixels of rounding."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ix >= ox - slack and iy >= oy - slack
            and ix + iw <= ox + ow + slack and iy + ih <= oy + oh + slack)


def census_regions(metadata):
    """Each Grevy's zebra annotation paired with the census region it carries.

    A row's `region_bbox` is its census region. A row without one contributes
    its `bbox`, provided the row is not hand-added and that box either sits
    inside the same animal's detection on the image or is that animal's only
    box.

    Returns a list of (annotation, box), restricted to annotations with an
    identity and a right-family viewpoint.
    """
    parsed = [(annotation, box(annotation['bbox']))
              for annotation in metadata['annotations']]
    by_animal = {}
    for annotation, bbox in parsed:
        key = (annotation['image_uuid'], annotation.get('individual_id'))
        by_animal.setdefault(key, []).append((annotation, bbox))

    found = []
    for annotation, own in parsed:
        if (annotation['category_id'] != 0
                or annotation['viewpoint'] not in VIEWPOINTS
                or not annotation.get('individual_id')):
            continue
        if annotation.get('region_bbox'):
            found.append((annotation, box(annotation['region_bbox'])))
            continue
        if annotation['manual']:
            continue
        siblings = by_animal[(annotation['image_uuid'], annotation['individual_id'])]
        inside = any(other is not annotation and contains(other_box, own)
                     for other, other_box in siblings)
        if inside or len(siblings) == 1:
            found.append((annotation, own))
    return found


def rows_for(found):
    """One row per (image, box), in the fields the pipeline and the model read.

    The pipeline keys an annotation by `uuid`; the release calls that field
    `annot_uuid` and uses `uuid` for images.
    """
    seen, rows = set(), []
    for annotation, bbox in found:
        key = (annotation['image_uuid'], tuple(bbox))
        if key in seen:
            continue
        seen.add(key)
        rows.append({'uuid': annotation['annot_uuid'],
                     'image_uuid': annotation['image_uuid'],
                     'bbox': bbox,
                     'viewpoint': annotation['viewpoint'],
                     'individual_id': annotation['individual_id'],
                     'category_id': int(annotation['category_id'])})
    return rows


def read_exclusions():
    """The `annot_uuid` of every annotation listed in EXCLUSIONS."""
    with open(EXCLUSIONS, newline='') as handle:
        return {row['annot_uuid'] for row in csv.DictReader(handle)}


def select(metadata):
    """Return (clustered, embedded), the rows for the two annotation files.

    The clustered set is the census regions published as a `region_bbox`, less
    the listed exclusions. The embedded set is every census region, and the
    classifier threshold is fitted over it, so it is the wider of the two.
    """
    found = census_regions(metadata)
    log(f'{len(found)} census regions in the release')

    excluded = read_exclusions()
    published = [(a, bbox) for a, bbox in found if a.get('region_bbox')]
    embedding = rows_for(found)
    clustering = [row for row in rows_for(published) if row['uuid'] not in excluded]
    log(f'  {len(embedding)} to embed, {len(clustering)} to cluster over '
        f'{len({row["individual_id"] for row in clustering})} individuals '
        f'({len(excluded)} listed exclusions)')

    # Two rows can share an (image, box) key, and each set picks the first it
    # meets, so the two can disagree on which annotation id stands for a box.
    missing = {row['uuid'] for row in clustering} - {row['uuid'] for row in embedding}
    if missing:
        raise SystemExit(f'{len(missing)} clustered annotations have no embedding')
    return clustering, embedding


def limit_to_individuals(clustering, embedding, individuals):
    """Cut both sets to the N most-sighted individuals, for a quick check.

    The threshold is fitted over whatever the embedding file holds, so a subset
    moves it and the run is not the reported one.
    """
    by_individual = {}
    for row in clustering:
        by_individual.setdefault(row['individual_id'], []).append(row)
    # the most-sighted first, so a small subset still has several annotations
    # per animal to cluster
    order = sorted(by_individual, key=lambda i: (-len(by_individual[i]), i))
    keep = set(order[:individuals])
    clustering = [row for row in clustering if row['individual_id'] in keep]
    kept = {row['uuid'] for row in clustering}
    embedding = [row for row in embedding if row['uuid'] in kept]
    log(f'limited to {len(keep)} individuals: {len(clustering)} to cluster, '
        f'{len(embedding)} to embed')
    return clustering, embedding


def download_images(output_dir, metadata, annotations):
    """Fetch only the images the selected annotations point at."""
    by_uuid = {image['uuid']: image for image in metadata['images']}
    wanted = sorted({by_uuid[a['image_uuid']]['image_path'] for a in annotations})
    missing = [name for name in wanted if not (output_dir / name).exists()]
    log(f'{len(wanted)} images needed, {len(missing)} to download')
    for number, name in enumerate(missing, 1):
        hf_hub_download(DATASET, name, repo_type='dataset', local_dir=str(output_dir))
        if number % 100 == 0 or number == len(missing):
            log(f'  {number}/{len(missing)}')
    return by_uuid


# --------------------------------------------------------------------------
# 2. the annotation file
# --------------------------------------------------------------------------

def write_annotations(output_dir, name, metadata, rows, by_uuid):
    """Write one annotation file in the format the pipeline reads."""
    path = output_dir / name
    used = sorted({row['image_uuid'] for row in rows})
    payload = {
        # The category id has to be an int: it is merged against the
        # annotations' category_id, and a string on one side raises.
        'categories': [{**category, 'id': int(category['id'])}
                       for category in metadata['categories']],
        'images': [by_uuid[uuid] for uuid in used],
        'annotations': rows,
    }
    path.write_text(json.dumps(payload))
    log(f'wrote {path} ({len(rows)} annotations, {len(used)} images)')
    return path


# --------------------------------------------------------------------------
# 3. the embeddings
# --------------------------------------------------------------------------

class Chips(Dataset):
    """Each annotation's bounding box, cropped and prepared for the model."""

    def __init__(self, annotations, by_uuid, output_dir, transform):
        self.annotations = annotations
        self.by_uuid = by_uuid
        self.output_dir = output_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        path = self.output_dir / self.by_uuid[annotation['image_uuid']]['image_path']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f'{path} is missing; rerun the download step')
        image = np.ascontiguousarray(image[:, :, ::-1])     # the model expects RGB

        # The centre is integer and the rotation is by zero degrees, matching
        # how the published embeddings were cropped. Removing the rotation
        # changes the pixels the model sees.
        x, y, w, h = annotation['bbox']
        centre = ((x + x + w) // 2, (y + y + h) // 2)
        rotation = cv2.getRotationMatrix2D(centre, 0.0, 1)
        rotated = cv2.warpAffine(image, rotation, (image.shape[1], image.shape[0]))
        chip = cv2.getRectSubPix(rotated, (w, h), centre)
        return self.transform(image=chip)['image'], annotation['uuid']


def embed(output_dir, annotations, by_uuid, batch_size, num_workers, device):
    path = output_dir / 'embeddings_gzcd_miewid_msv3.pickle'
    if path.exists():
        log(f'embeddings already at {path}')
        return path

    transform = Compose([
        Resize(IMAGE_SIZE, IMAGE_SIZE),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD,
                  max_pixel_value=IMAGE_MAX_VALUE, p=1.0),
        ToTensorV2(p=1.0),
    ])
    loader = DataLoader(Chips(annotations, by_uuid, output_dir, transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'loading {MODEL} on {device}')
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()

    vectors, uuids = [], []
    with torch.no_grad():
        for batch, batch_uuids in loader:
            vectors.append(model(batch.to(device).float()).cpu().numpy())
            uuids.extend(batch_uuids)
            # the batch is small, so report on a round number of annotations
            # rather than on every batch
            if len(uuids) % 100 < batch_size or len(uuids) == len(annotations):
                log(f'  embedded {len(uuids)}/{len(annotations)}')
    matrix = np.concatenate(vectors).astype(np.float32)

    # The pipeline expects the matrix and the annotation ids its rows follow.
    with open(path, 'wb') as f:
        pickle.dump((matrix, uuids), f)
    log(f'wrote {path} {matrix.shape}')
    return path


# --------------------------------------------------------------------------
# 4. the run
# --------------------------------------------------------------------------

def run_stability(output_dir, annotation_file, embedding_file):
    """Fill the paths into the shipped configuration, then run it."""
    settings = yaml.safe_load(CONFIG.read_text())
    results = output_dir / 'results'
    settings['data']['annotation_file'] = str(annotation_file)
    settings['data']['embedding_file'] = str(embedding_file)
    settings['data']['output_path'] = str(results / 'clustering')
    settings['logging']['log_file'] = str(results / 'GZCD.log')
    settings['logging']['auto_threshold_plot_path'] = str(results / 'GZCD_threshold.png')

    # Written before the run and loaded back, so the file beside the results is
    # exactly what ran.
    results.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / CONFIG.name
    config_file.write_text(yaml.safe_dump(settings, sort_keys=False))
    log(f'wrote {config_file}')

    log('running beta-stability')
    annotations = run.run_from_config(config_module.load(str(config_file)))
    log(f'wrote {annotations}')
    return annotations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('build/gzcd'),
                        help='where everything is written: the downloaded dataset, the\n'
                             'embeddings, the config and the results')
    parser.add_argument('--max-individuals', type=int, default=None,
                        help='keep only the N most-sighted individuals, for a '
                             'quick end-to-end check that does not reproduce '
                             'the reported run')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'default {BATCH_SIZE}, the value the stage that '
                             f'produced the published embeddings used')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--stop-after', choices=['metadata', 'images', 'annotations',
                                                 'embeddings'],
                        help='stop once that step is done')
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = download_metadata(output_dir)
    if args.stop_after == 'metadata':
        return 0
    clustering, embedding = select(metadata)
    if args.max_individuals is not None:
        clustering, embedding = limit_to_individuals(clustering, embedding,
                                                     args.max_individuals)
    if not clustering:
        log('no annotations selected; nothing to do')
        return 1

    # The embedding set is the wider one, so its images cover both.
    by_uuid = download_images(output_dir, metadata, embedding)
    if args.stop_after == 'images':
        return 0

    annotation_file = write_annotations(output_dir, 'annotations_gzcd.json',
                                       metadata, clustering, by_uuid)
    write_annotations(output_dir, 'embedding_set_gzcd.json',
                      metadata, embedding, by_uuid)
    if args.stop_after == 'annotations':
        return 0

    embedding_file = embed(output_dir, embedding, by_uuid, args.batch_size,
                           args.num_workers, args.device)
    if args.stop_after == 'embeddings':
        return 0

    run_stability(output_dir, annotation_file, embedding_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
