import pickle
import yaml
import json
import logging
import networkx as nx


import numpy as np



class EmptyDataframeException(Exception):
    pass

class SingleAnnotationException(Exception):
    """Raised when only one annotation remains after filtering - no clustering needed."""
    def __init__(self, node2uuid, gt_clustering=None, gt_node2cid=None):
        self.node2uuid = node2uuid
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        super().__init__("Only one annotation - skipping Beta Stability clustering")

def kth_diag_indices(a, k):
    r, c = a.shape
    i_start = max(0, -k)
    i_end = min(r, c - k)
    rows = np.arange(i_start, i_end)
    cols = rows + k
    return rows, cols

def lazy(func):
    func.__val = None
    def wrapper():
        computed = func.__val
        if computed:
            return computed
        computed = func()
        func.__val = computed
        return computed
    return wrapper

def load_yaml(file_path):
    logger = logging.getLogger("beta_stability")
    logger.info(f"Loading config from path: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)




def get_config(file_path):

    return load_yaml(file_path)



class SetEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, nx.Graph):
            return nx.cytoscape_data(obj)
        return json.JSONEncoder.default(self, obj)


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def write_json(data, out_path):
    json_object = json.dumps(data, indent=4, cls=SetEncoder)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)



def load_pickle(file):
    with open(file, 'rb') as f_file:
        return pickle.load(f_file)

def load_dataframe_lightweight(config):
    """Load dataframe without embeddings for field discovery."""
    from beta_stability.data.preprocess import preprocess_data

    data_params = config['data']

    # Load embeddings just to get UUIDs
    uuids = load_pickle(data_params['embedding_file'])[1]

    name_keys = data_params['name_keys']
    format_type = data_params['format']
    id_key = data_params['id_key']

    df = preprocess_data(
        data_params['annotation_file'],
        name_keys=name_keys,
        convert_names_to_ids=True,
        n_filter_min=data_params['n_filter_min'],
        n_filter_max=data_params['n_filter_max'],
        images_dir=data_params.get('images_dir'),
        embedding_uuids=uuids,
        id_key=id_key,
        format=format_type,
        print_func=lambda _: None  # Silent loading
    )

    # Filter to available UUIDs
    return df[df[id_key].isin(uuids)]


def discover_field_values_from_df(df, fields):
    """Discover unique values for fields from dataframe."""
    field_values = {}
    for field in fields:
        if field in df.columns:
            unique_values = df[field].dropna().unique().tolist()
        else:
            # Try name_<field> pattern
            name_field = f"name_{field}"
            if name_field in df.columns:
                unique_values = df[name_field].dropna().unique().tolist()
            else:
                continue
        field_values[field] = sorted(unique_values, key=str)
    return field_values

def generate_gt_clusters(df, name_key, id_key='uuid'):
    gt_node2cid = {}
    cids_unique = df[name_key].unique()
    cids = [f'ct{idx}' for idx in range(len(cids_unique))]
    gt_clustering = {cid: [] for cid in cids}
    node2uuid = {}

    for i, row in df.iterrows():
        cid = cids[list(cids_unique).index(row[name_key])]
        gt_clustering[cid].append(i)
        gt_node2cid[i] = cid
        node2uuid[i] = row[id_key]

    gt_clustering = {
        cid: set(cluster) for cid, cluster in gt_clustering.items()
    }

    return gt_clustering, gt_node2cid, node2uuid


def generate_node2uuid(df, id_key='uuid'):
    """The node index to annotation id map, for data with no identities.

    generate_gt_clusters builds this alongside the ground-truth clustering. A
    run without identities needs the map and nothing else.
    """
    return {i: row[id_key] for i, row in df.iterrows()}


def print_intersect_stats(df,individual_key="name"):
    logger = logging.getLogger("beta_stability")
    logger.info("** Dataset statistcs **")
    logger.info(' - Counts: ')
    names = df[individual_key].unique()


    logger.info(f" ---- number of individuals: {len(names)}" )
    logger.info(f" ---- number of annotations: {len(df)}")
    avg_ratio = len(df) / max(1, len(names))
    logger.info(f" ---- average number of annotations per individual: {avg_ratio:.2f}")

    annotation_counts = df[individual_key].value_counts()
    single_annotation_count = (annotation_counts == 1).sum()

    logger.info(f" ---- number of singletons: {single_annotation_count}")


# ---------------------------------------------------------------- evaluation
# checkpoints
#
# `report_at_reviews` lists human-review counts at which a review-driven method
# logs the clustering it holds after exactly that many reviews, even when the
# count falls inside a review batch. The string 'N' stands for the number of
# annotations (one review per annotation). A checkpoint is evaluation only: it
# never changes what the run does next, so a run's regular evaluations are
# identical with or without checkpoints.

CHECKPOINT_LABEL = 'Checkpoint stats'


def resolve_checkpoints(spec, num_nodes):
    """Sorted review counts for `spec` (a list of ints and/or 'N', or a single one)."""
    if spec is None:
        return []
    if isinstance(spec, (int, str)):
        spec = [spec]
    out = set()
    for s in spec:
        s = s.strip() if isinstance(s, str) else s
        if s == 'N':
            out.add(int(num_nodes))
            continue
        v = int(s)
        if v <= 0:
            raise ValueError(f'report_at_reviews entries must be positive review counts, got {s!r}')
        out.add(v)
    return sorted(out)
