# Beta Stability

Stability-driven clustering with active review for wildlife re-identification.

Given re-identification embeddings and an annotation file, the algorithm builds
an identification graph, deactivates positive edges until the graph is
`0`-stable (Phase 0), and then asks a reviewer about the least stable pairs
until either a target `β`-stability is reached or the review budget is spent.

Edges come from one embedding verifier, with its positive/negative threshold
fitted per dataset by a two-component Gaussian mixture over the pair-score
histogram, and the neighbourhood size `K` derived from the cluster count that
threshold implies.

## Requirements

- Python 3.11 or newer
- Dependencies managed with [uv](https://docs.astral.sh/uv/)

## Installation

```
uv sync
```

That creates the virtual environment from `pyproject.toml` and `uv.lock`.
There are two optional extras: `dev` for `pytest` and `ruff`, and `embeddings`
for the model needed by `scripts/reproduce_gzcd.py`. Naming one extra drops the
other, so ask for both when you want both:

```
uv sync --extra dev --extra embeddings
```

Every command below runs through `uv run`, which uses that environment without
activating it. To check the installation:

```
uv run pytest tests/
```

The suite runs the whole pipeline on a small generated dataset, for every
algorithm type, and takes about half a minute.

## Quick start

A run takes one config, which names the algorithm and sets everything it needs:

```
uv run python -m beta_stability --config path/to/your_config.yaml
```

[examples/beta_stability_config.yaml](examples/beta_stability_config.yaml) is a
complete config to start from; add your `data:` paths to it. Every parameter a
config can set is declared, with its default and what it does, in
[beta_stability/config.py](beta_stability/config.py), and anything a config
leaves out takes the default from there.

| flag | what it does |
| --- | --- |
| `--config` | the config to run (required) |
| `--base_config` | a second config to merge `--config` on top of, for sharing one method config across datasets |
| `--save_dir` | where to write the annotation file, overriding `data.output_path`; the graph files still go to `output_path` |
| `--set KEY=VALUE` | override one config key by dotted path, for example `--set stability.target_alpha=0.3`, repeatable |
| `--seed` | override `seed`, and write the results into a `seed<N>/` subdirectory |
| `--prob_human_correct` | override `edge_weights.prob_human_correct`, and write the results into a `phc<NN>/` subdirectory |
| `--interactive`, `-i` | print the resolved config, annotation and output paths, then wait for Enter before running |

### Inputs

The data config must provide:

- `data.annotation_file` — a JSON with one entry per annotation, each with a
  unique id under whatever `data.id_key` names, `uuid` by default.
- `data.name_keys` — the annotation fields that form the ground-truth
  identity, for example `[name]`. Leave it unset to cluster annotations that
  have no identities, which is the production case: the run then reports no
  evaluation, and the per-individual count filters do not apply. Naming a
  field the annotation file does not have stops the run and says which fields
  it does have.
- `data.<field>_list` — a whitelist for that annotation field, so
  `viewpoint_list: [up]` keeps only top-view annotations. Filtering everything
  out is not an error: the run writes every annotation back with a null
  cluster and logs the count at each filtering step.
- `data.embedding_file` — a pickle of `(embedding_matrix, uuid_list)` from the
  re-identification model. The uuid list must match the annotation file.
- `data.output_path` — the directory results are written to.

### Outputs

Written to `output_path`:

- `beta_stability_annots.json` — the input annotations with a `cluster_id`
  added to each one. This is the file downstream curation reads.
- `clustering.json` — clusters as `{cluster_id: [node_id, ...]}`.
- `node2cid.json` — the inverse mapping, `{node_id: cluster_id}`.
- `node2uuid_file.json` — node index to annotation uuid.
- `graph.json` — the identification graph, for analysis.
- Evaluation against the ground-truth identities, when `data.name_keys` names
  them. The reported metric is CEAF-F1 by default, and `metrics.report` can
  ask for `hungarian` or `pairwise` as well.

## How it works

The algorithm treats clustering as a set of local decisions about which edges
are trustworthy, and strengthens the ones a reviewer confirms. It loops over
four steps:

1. **Compute stability** for the whole graph. A PCC is a connected component
   of the positive edges and stands for one individual.
2. **Enforce stability** by deactivating positive edges until every decision
   is at least `0`-stable. Steps 1 and 2 need no human input, and the first
   pass through them is **Phase 0**.
3. **Select pairs** for verification, the least stable first.
4. **Apply the verdicts**, then return to Step 1. Steps 3 and 4 with 1 and 2
   are **Phase 1**.

A run ends when the target stability is reached, when no candidates are left,
or when the review budget is spent.

The loop is [beta_stability/stability/algorithm.py](beta_stability/stability/algorithm.py),
the graph and the stability definitions are
[beta_stability/stability/graph.py](beta_stability/stability/graph.py), and
Step 3 is [beta_stability/stability/candidates.py](beta_stability/stability/candidates.py).

## Usage example

End-to-end run against a beluga dataset, which is where the log
excerpt in step 4 comes from.

**1. Prepare the data files**, outside the repository:

- `annotations_beluga.json`, a COCO-style annotation file with an identity
  field and optionally a `viewpoint` field, each annotation carrying a `uuid`.
- `embeddings_beluga.pickle`, a tuple `(embedding_matrix, uuid_list)` from
  the re-identification model.

**2. Write the config**, `configs/beluga.yaml`. Copy
`examples/beta_stability_config.yaml` and give it your data:

```yaml
exp_name: beluga
species: Beluga whale
data:
  viewpoint_list: [up]          # keep only top-view annotations
  name_keys: [name]             # ground-truth identity comes from this field
  n_filter_min: 1               # drop individuals with fewer annotations
  n_filter_max: 100             # and subsample those with more
  annotation_file: /data/beluga/annotations_beluga.json
  embedding_file: /data/beluga/embeddings_beluga.pickle
  output_path: /results/beluga/beta_stability/
```

**3. Run it:**

```
uv run python -m beta_stability --config configs/beluga.yaml
```

**4. Read the log.** The threshold and the neighbourhood size are reported
before the run starts, and both are derived from the data:

```
Fitting GMM threshold for miewid (spec=auto(0.2)) — not cached yet
✅ Optimal threshold: 0.6662, predicted F1: 0.8726 (predicted precision=0.9473, recall=0.8088)
K̂ from all-positives graph: 163 PCCs (1670 positive edges over 861 nodes, threshold=0.6662)
  Auto initial_topk = 376  (coupon-collector: -ln(1-0.90)·K̂ = 2.30·163)
  Auto review_confidence = 1.000 (evidence ln(p/(1-p))/ln(N); p=1.0000, N=323736, ln(N)=12.69)
  Target alpha: 0.5
  Max human reviews: 5000
=== Phase 0 complete - converged ===
=== Starting Active Review Phase ===
```

The mixture fit gives the threshold that separates same from different pairs,
`K̂` is the number of positive components at that threshold, and `K` follows
from it. Nothing here is tuned per dataset by hand.

**5. Consume the results:**

```python
import json

with open("/results/beluga/beta_stability/beta_stability_annots.json") as f:
    annotations = json.load(f)["annotations"]

for annotation in annotations[:5]:
    print(annotation["uuid"], "->", annotation["cluster_id"])
```

## Reproducing a published run

[scripts/reproduce_gzcd.py](scripts/reproduce_gzcd.py) reproduces the Grevy's
zebra result, going from nothing to a clustering in four steps, each skipped if
its output is already there:

1. download the metadata and the needed images from
   [imageomics/GZCD](https://huggingface.co/datasets/imageomics/GZCD)
2. write the annotation files
3. embed the annotations with MiewID v3,
   [conservationxlabs/miewid-msv3](https://huggingface.co/conservationxlabs/miewid-msv3),
   at the model's default settings
4. fill the data paths into [scripts/gzcd.yaml](scripts/gzcd.yaml), which holds
   every other parameter of the run, and run it

```
uv sync --extra dev --extra embeddings
uv run python scripts/reproduce_gzcd.py --max-individuals 20   # a subset, to start
uv run python scripts/reproduce_gzcd.py                        # all 3,764
```

Downloading and embedding is the slow part, so `--max-individuals N` takes the N
most-sighted animals for an end-to-end check first, and `--stop-after` ends after
any step. The embedding step needs a GPU. The full run clusters 3,764
annotations of 435 individuals and ends on 435 clusters after 580 reviews, at a
CEAF F1 of 0.9880. Those are the numbers to compare a fresh run against.

The selection keeps the census annotations of Grevy's zebra that carry an
identity and a right-side viewpoint, and drops the 43 listed in
[scripts/gzcd_excluded_annotations.csv](scripts/gzcd_excluded_annotations.csv),
which were rejected as too low-quality to identify. What gets embedded is the
census region, not the full animal detection. 

## Repository layout

```text
beta_stability/
├── run.py                    # runs a prepared method and saves the result
├── config.py                 # every parameter, its default and what it does
├── algorithm_preparation.py  # turns a config into a configured method
├── stability/                # the beta-stability method
│   ├── algorithm.py          #   the four-step loop
│   ├── graph.py              #   the identification graph and stability
│   └── candidates.py         #   Step 3, choosing pairs for review
├── baselines/                # the other algorithm types
├── data/                     # annotations in and out, filters, io
├── edge_weights/             # verifiers: classifiers, thresholds, reviewer
├── embeddings/               # embedding backends
└── metrics/                  # evaluation against ground truth, run tracking

scripts/
├── reproduce_gzcd.py             # the GZCD run, from the public dataset
├── gzcd.yaml                     #   its configuration
└── gzcd_excluded_annotations.csv #   the 43 annotations no public field marks

tests/                        # uv run pytest tests/
examples/
└── beta_stability_config.yaml

pyproject.toml                # project metadata, entry points, lint rules
uv.lock                       # locked dependency resolution
```

## Configuration essentials

- `algorithm_type: stability` selects this method. The alternatives are
  `hdbscan`, `nis`, `np3_aas`, `manual_review` and `thresholded_review`. The
  two review baselines need `algorithm.initial_topk: 0`, because they build
  their own candidate list.
- `stability.target_alpha` is `β`, the stability a run works towards.
- `algorithm.initial_topk: auto` uses the coupon-collector count
  `K = -ln(1 - coverage) · K̂`, with `K̂` the number of connected components of
  the positive-edge graph at the selected threshold.
- `stability.verifier_name` names the embedding the edges come from, for
  example `miewid`. Setting it to `auto` instead enables a silhouette-based
  selector that can switch the run to k-means edges. That path is optional and
  far less tested than naming the embedding directly.
- `edge_weights.classifier_thresholds` takes `auto(fraction)` to fit a
  two-component Gaussian mixture to the pair-score histogram and pick the
  threshold from it.
- `edge_weights.prob_human_correct` is the simulated reviewer's accuracy. Set
  it below `1.0` to see how the result holds up when the reviewer makes
  mistakes.
- `edge_weights.verifier_names` lists the verifiers in order, ending in the
  reviewer: `simulated_human` answers from the identity labels, so it needs
  `data.name_keys`; `no_human` runs fully automatically, which is what a run
  on unlabelled data uses; and `ui_human` serves the pairs through a review
  database, which also needs `data.ui_db_path` and a running review
  interface.

## Citation

```text
TBD
```
