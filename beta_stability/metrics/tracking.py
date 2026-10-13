"""Observing a run without being part of it.

The algorithm reports what it is doing at a handful of points and this module
decides what to record. Three separate things live here, and none of them
changes the clustering:

* **validation** against ground truth, at the start, periodically, at the end,
  and at the exact review counts a config asks to be checkpointed;
* **the stability track**, a per-iteration snapshot of every internal and
  external stability, saved raw so any plot can be made offline without
  re-running;
* **statistics logging**, the end-of-run summary of the graph and the review
  tally.

Keeping this out of :mod:`beta_stability.stability.algorithm` means the
algorithm's own code is only the four steps, and that a change to what we
measure cannot change what we compute.
"""

import copy
import json
import logging
import time
from contextlib import contextmanager

from beta_stability.data.io import CHECKPOINT_LABEL, resolve_checkpoints

logger = logging.getLogger("beta_stability")


class RunTracker:
    """Records how a run progressed against the ground truth.

    Does nothing at all when no validator was configured, which is the case
    for a production run with no labels to compare against.
    """

    def __init__(self, validator, *, validation_step=100, report_at_reviews=None,
                 track_stability=False, track_path=None):
        self.validator = validator
        self.validation_step = validation_step
        self.report_at_reviews = report_at_reviews
        self.track_stability = track_stability
        self.track_path = track_path
        self.started = False
        self.snapshots = []
        self._checkpoints = None

    @property
    def active(self):
        """True once validation has a baseline to compare against."""
        return bool(self.validator) and self.started

    # ---------------------------------------------------------------- validation

    def start(self, clustering, node2cid, graph_view, num_human):
        """Record the baseline the rest of the run is compared against."""
        if not self.validator:
            return
        self.validator.trace_start_human(clustering, node2cid, graph_view, num_human)
        self.started = True

    def periodic(self, clustering, node2cid, graph_view, num_human):
        """Score the clustering if enough reviews have happened since the last point."""
        if not self.active:
            return
        if num_human - self.validator.prev_num_human >= self.validation_step:
            self.validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, graph_view)

    def final(self, clustering, node2cid, graph_view, num_human):
        """Score the clustering the run actually ended on.

        Periodic validation only fires every ``validation_step`` reviews, so
        without this a run that spent its full budget would report the state a
        partial batch earlier. ``trace_iter_compare_to_gt`` is a no-op when no
        reviews have happened since the last point, so this is safe to call
        unconditionally.
        """
        if not self.active:
            return
        self.validator.trace_iter_compare_to_gt(clustering, node2cid, num_human, graph_view)

    def phase0(self, clustering, node2cid, iteration):
        """Score a Phase 0 iteration, before any review has happened."""
        if not self.validator:
            return None
        result = self.validator.incremental_stats(
            0, clustering, node2cid,
            self.validator.gt_clustering, self.validator.gt_node2cid,
            f'Phase 0 iteration {iteration}')
        result['phase0_iteration'] = iteration
        return result

    # ---------------------------------------------------------------- checkpoints

    def checkpoint_due(self, num_human, num_nodes):
        """True when this exact review count is one the config asks to report."""
        if self._checkpoints is None:
            self._checkpoints = set(resolve_checkpoints(self.report_at_reviews, num_nodes))
        return num_human in self._checkpoints

    def checkpoint(self, clustering, node2cid, num_human):
        """Score a clustering at an exact review count, mid-batch if need be."""
        if not self.active:
            return
        self.validator.incremental_stats(
            num_human, clustering, node2cid,
            self.validator.gt_clustering, self.validator.gt_node2cid,
            CHECKPOINT_LABEL)

    def checkpoint_on_copy(self, graph, restabilize, clustering_of):
        """Score the graph as it stands now, without disturbing the live run.

        The end-of-batch path is: apply each review to its edge, re-stabilize
        once, then score. A checkpoint runs that same path on a deep copy, so
        the live graph, the validator's trace state and every counter are
        untouched and the run continues exactly as it would have.

        Args:
            graph: the live graph, which is copied rather than modified.
            restabilize: callable applied to the copy, or None to skip it.
            clustering_of: callable turning the copy into ``(clustering, node2cid)``.
        """
        if not self.active:
            return None
        working = copy.deepcopy(graph)
        if restabilize is not None:
            restabilize(working)
        return clustering_of(working)

    # ---------------------------------------------------------- stability track

    def snapshot(self, per_pcc, external, *, phase, phase0_iteration, reviews):
        """Append the raw per-decision stabilities for this iteration.

        Raw rather than summarised, so that any stability plot -- unified,
        split, minimum, distribution, fraction below beta, per-PCC trajectory --
        can be produced offline without re-running. ``internal`` keeps PCC
        identity for the trajectories, ``external`` is the flat list of
        separation-decision stabilities, and ``step`` is a monotonic index
        across both phases.
        """
        if not self.track_stability:
            return
        self.snapshots.append({
            'step': len(self.snapshots),
            'phase': phase,
            'phase0_iter': phase0_iteration,
            'reviews': reviews,
            'n_pccs': len(per_pcc),
            'internal': [[a, s, (None if st is None else round(float(st), 5))]
                         for (a, s, st) in per_pcc],
            'external': [round(float(x), 5) for x in external],
        })

    def dump(self):
        """Write the stability track, if one was collected.

        A failure here is reported and tolerated on purpose: losing an optional
        diagnostic must not discard the run that produced it.
        """
        if not (self.track_stability and self.snapshots):
            return
        path = self.track_path or 'stability_track.json'
        try:
            with open(path, 'w') as f:
                json.dump(self.snapshots, f)
            logger.info(f"Stability track: wrote {len(self.snapshots)} snapshots to {path}")
        except OSError as e:
            logger.warning(f"Stability track dump failed: {e}")


@contextmanager
def timed(label):
    """Log how long a step took.

    The costly steps of a long run are worth timing, but the timing should not
    be three lines of bookkeeping wrapped around every call site.
    """
    start = time.time()
    yield
    logger.info(f"TIMER {label} {time.time() - start:.2f}s")


# -------------------------------------------------------------------- reporting

def log_graph_stats(graph, *, phase, num_human_reviews, target_alpha, pcc_strength):
    """Log the end-of-run summary of the graph."""
    stats = graph.get_graph_stats()
    mst_stats = graph.get_mst_cache_stats()
    counts = stats['edge_counts']

    logger.info("=" * 60)
    logger.info("Beta Stability Algorithm Statistics")
    logger.info("=" * 60)
    logger.info(f"Phase: {phase}")
    logger.info(f"Human reviews: {num_human_reviews}")
    logger.info(f"Target alpha: {target_alpha}")
    logger.info("-" * 60)
    logger.info(f"Nodes: {stats['num_nodes']}")
    logger.info(f"Edges: {stats['num_edges']}")
    logger.info(f"  Positive: {counts['positive']}")
    logger.info(f"  Positive-inactive: {counts['positive_inactive']}")
    logger.info(f"  Negative: {counts['negative']}")
    logger.info("-" * 60)
    logger.info(f"PCCs: {stats['num_pccs']}")
    logger.info(f"PCC sizes: {stats['min_pcc_size']} - {stats['max_pcc_size']}")
    logger.info(f"Min internal stability: {stats['min_internal_stability']:.4f}")
    logger.info(f"Min external stability: {stats['min_external_stability']:.4f}")
    logger.info("-" * 60)
    logger.info("MST Forest:")
    logger.info(f"  Edges: {mst_stats['mst_forest_edges']}")
    logger.info("-" * 60)

    strengths = [pcc_strength.get(i, 0.0) for i in range(stats['num_pccs'])]
    if strengths:
        median = sorted(strengths)[len(strengths) // 2]
        logger.info("PCC supergraph (separation strength):")
        logger.info(f"  Strength: min={min(strengths):.3f}, "
                    f"median={median:.3f}, max={max(strengths):.3f}")
        logger.info(f"  Zero strength (no negative edges): "
                    f"{sum(1 for s in strengths if s == 0.0)}/{len(strengths)}")
        logger.info(f"  Isolated (<=median): "
                    f"{sum(1 for s in strengths if s <= median)}/{len(strengths)}")
    logger.info("=" * 60)


#: How a mistaken review changes the graph, by verdict and the label it overruled.
_ERROR_EFFECTS = {
    ('same', 'neg'): 'MERGE (reversible: rectifier can cut back)',
    ('diff', 'pos'): 'SPLIT (IRREVERSIBLE: cut-only rectifier)',
    ('same', 'pos'): 'entrench-merge',
    ('diff', 'neg'): 'entrench-split',
}


def log_review_tally(tally):
    """Log the breakdown of reviews against ground truth, and of the errors.

    ``tally`` counts ``(correct?, verdict, label, flipped?)``. The four error
    kinds are a verdict crossed with the label it overruled, which is what
    decides whether a mistake can be undone later.
    """
    if not tally:
        return
    total = sum(tally.values())
    errors = {key: n for key, n in tally.items() if key[0] == 'ERR'}

    logger.info("=== Review tally (vs ground truth) ===")
    logger.info(f"  total reviews with GT: {total}, errors: {sum(errors.values())} "
                f"({100 * sum(errors.values()) / total:.1f}%)")
    logger.info("  ERROR breakdown [verdict on current-label -> effect]:")
    for (_, verdict, label, flip), n in sorted(errors.items()):
        effect = _ERROR_EFFECTS[(verdict, label)]
        logger.info(f"    err: say '{verdict}' on {label} edge, {flip:6s} -> {effect}: {n}")

    def flipped(verdict, label):
        return sum(n for (_, v, lab, flip), n in errors.items()
                   if (v, lab, flip) == (verdict, label, 'flip'))

    logger.info(f"  flipping errors -> reversible MERGE: {flipped('same', 'neg')} | "
                f"irreversible SPLIT: {flipped('diff', 'pos')}")
