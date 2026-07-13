"""
Beta Stability Algorithm

Implements the algorithm as defined in:
"LCA V2 Formulation" by Charles Stewart, January 2026

Algorithm phases:
1. Phase 0: Make graph 0-stable (no human input, only deactivate positive edges)
2. Active Review: Iteratively improve stability with human reviews until target alpha

Per PDF:
- Only positive edges can be deactivated (become positive-inactive)
- Negative edges are NEVER deactivated
- Human review: agree adds ch, disagree subtracts ch (flip if negative)
- After human review, re-run 0-stability
"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Optional, Any

from beta_stability.stability_graph import StabilityGraph, EdgeLabel, StabilityCandidate
from beta_stability.util.tools import order_edge

logger = logging.getLogger("beta_stability")


class BetaStabilityAlgorithm:
    """
    Beta Stability stability-driven clustering algorithm.

    Implements the standard algorithm interface (step, is_finished, get_clustering).
    """

    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        # Core graph structure
        self.graph = StabilityGraph()

        # Algorithm state
        self.phase = "PHASE0"  # PHASE0 -> ACTIVE_REVIEW -> FINISHED
        self.target_alpha = config.get('target_alpha', 0.5)
        self.num_human_reviews = 0
        self.max_human_reviews = config.get('max_human_reviews', 1000)

        # Human review parameters
        self.ch = config.get('review_confidence', config.get('human_confidence', 0.5))  # Confidence change per review
        self.edges_per_review_batch = config.get('edges_per_review_batch', 20)  # Max edges per batch
        self.tries_before_edge_done = config.get('tries_before_edge_done', 4)  # Max reviews per edge
        self.human_attempts = {}  # Track review attempts per edge: {(n0, n1): count}

        # Cross-PCC edge discovery
        self._all_edges_sorted = None
        self._sorted_edge_index = 0
        self.cross_pcc_max_edges = config.get('cross_pcc_max_edges', 0)
        self.max_densify_edges = config.get('max_densify_edges', 0)

        # Phase 0 controls
        self.max_phase0_iterations = config.get('max_phase0_iterations', 20)
        self.phase0_alpha = config.get('phase0_alpha', 0.0)
        self.sparsify_on_init = config.get('sparsify_on_init', False)
        self._phase0_prev_pcc_count = None
        self._phase0_stall_count = 0

        # Densification strategy: if True, add likely negative edges first (more aggressive)
        # Default False adds likely positive edges first (less fragmentation)
        self.densify_prioritize_negatives = config.get('densify_prioritize_negatives', False)

        # Adaptive per-batch allocation. When True, EXTERNAL's slot count is
        # capped by its rolling disagree-rate yield over the last
        # `adaptive_window` batches (Laplace-smoothed). Kept as a knob for
        # future selectors even though EXTERNAL is currently the only
        # non-INTERNAL bucket.
        self.adaptive_allocation = bool(config.get('adaptive_allocation', False))
        self.adaptive_window = int(config.get('adaptive_window', 5))
        self.adaptive_min_floor = int(config.get('adaptive_min_floor', 5))

        # Rolling per-batch history of (agree, disagree) counts per candidate
        # type. Used by _compute_adaptive_quotas.
        self._batch_history: List[Dict[str, Tuple[int, int]]] = []

        # Explicit cap on EXTERNAL allocation per batch (None = uncapped,
        # absorbs whatever's leftover after INTERNAL).
        self.external_per_batch = config.get('external_per_batch', None)

        # Runtime metric trajectory tracking. When enabled, on each active-
        # review iteration the algorithm also computes top-N candidates by a
        # set of alternative metrics and reports, against ground truth, how
        # many of each set are actual false negatives. Lets us see how each
        # metric's discriminative power evolves as human reviews accumulate.
        # Requires cluster_validator with ground truth (simulation mode).
        self.metric_tracker_enabled = bool(config.get('metric_tracker_enabled', False))
        self.metric_tracker_methods = list(config.get(
            'metric_tracker_methods',
            ['heat_kernel', 'heat_kernel_weighted', 'ppr', 'ppr_weighted', 'human_signed'],
        ))
        self.metric_tracker_top_n = int(config.get('metric_tracker_top_n', 50))

        # Validation
        self.validation_step = config.get('validation_step', 20)
        self.validation_initialized = False

        # Diagnostic: map pending review edges to their candidate type
        self._pending_review_types: Dict[Tuple[int, int], str] = {}
        # Cumulative agreement/disagreement counts per candidate type
        from collections import defaultdict
        self._cumulative_agree: Dict[str, int] = defaultdict(int)
        self._cumulative_disagree: Dict[str, int] = defaultdict(int)

        # Phase 0 iteration counter for metric logging
        self.phase0_iteration = 0

        logger.info(f"Beta Stability Algorithm initialized")
        logger.info(f"  Target alpha: {self.target_alpha}")
        logger.info(f"  Phase 0 alpha: {self.phase0_alpha}")
        logger.info(f"  Max Phase 0 iterations: {self.max_phase0_iterations}")
        logger.info(f"  Human confidence (ch): {self.ch}")
        logger.info(f"  Max human reviews: {self.max_human_reviews}")
        logger.info(f"  Tries before edge done: {self.tries_before_edge_done}")

    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        """
        Perform one step of the algorithm.

        Args:
            new_edges: List of (n0, n1, score, verifier_name) tuples

        Returns:
            List of edges needing human review: [(n0, n1, score), ...]
        """
        # Process and add incoming edges
        self._process_and_add_edges(new_edges)

        # Count human reviews and apply feedback
        needs_restabilization = False
        # Track agreement/disagreement per candidate type for informativeness metric
        from collections import defaultdict
        type_agree = defaultdict(int)
        type_disagree = defaultdict(int)

        for edge in new_edges:
            if len(edge) > 3 and 'human' in str(edge[3]):
                n0, n1, score, verifier_name = edge[:4]

                # Track human review attempts per edge
                edge_key = (min(n0, n1), max(n0, n1))
                attempts = self.human_attempts.get(edge_key, 0)

                if attempts < self.tries_before_edge_done:
                    self.human_attempts[edge_key] = attempts + 1
                    self.num_human_reviews += 1

                    # Determine agreement BEFORE applying feedback
                    edge_data = self.graph.get_edge(n0, n1)
                    if edge_data:
                        human_decision = score > 0.5
                        current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
                        human_agrees = (human_decision == current_is_positive)
                        ctype = self._pending_review_types.get(edge_key, "UNKNOWN")
                        if human_agrees:
                            type_agree[ctype] += 1
                        else:
                            type_disagree[ctype] += 1

                    if self._apply_human_feedback(n0, n1, score):
                        needs_restabilization = True
                else:
                    logger.warning(f"Edge {edge_key} exceeded max attempts ({self.tries_before_edge_done}), skipping")

        # Log per-type disagreement rates (higher = more informative candidates)
        all_types = set(type_agree.keys()) | set(type_disagree.keys())
        if all_types:
            # Update cumulative counts
            for ctype in all_types:
                self._cumulative_agree[ctype] += type_agree[ctype]
                self._cumulative_disagree[ctype] += type_disagree[ctype]

            # Record this batch's per-type yield for adaptive allocation.
            # Kept as a rolling window of length up to adaptive_window * 2,
            # trimmed in _compute_adaptive_quotas.
            if self.adaptive_allocation:
                batch_record = {
                    ctype: (type_agree[ctype], type_disagree[ctype])
                    for ctype in all_types
                }
                self._batch_history.append(batch_record)
                if len(self._batch_history) > self.adaptive_window * 2:
                    self._batch_history = self._batch_history[-self.adaptive_window:]

            # Batch stats
            parts = []
            for ctype in sorted(all_types):
                a = type_agree[ctype]
                d = type_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (batch): {', '.join(parts)}")

            # Cumulative stats
            all_cumulative = set(self._cumulative_agree.keys()) | set(self._cumulative_disagree.keys())
            parts = []
            for ctype in sorted(all_cumulative):
                a = self._cumulative_agree[ctype]
                d = self._cumulative_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (cumulative): {', '.join(parts)}")

        # Detect PCC splits from human reviews and queue cross-check edges
        # Per PDF step 5: "Update the labels of edges to make the graph 0-stable"
        if needs_restabilization:
            logger.info("Starting re-stabilization...")
            deactivations = self.graph.make_zero_stable(alpha=0.0)
            if deactivations > 0:
                logger.info(f"Re-stabilized after human reviews: {deactivations} deactivations")

        # Phase dispatch
        if self.phase == "PHASE0":
            return self._phase0_step()
        elif self.phase == "ACTIVE_REVIEW":
            return self._active_review_step()
        else:
            return []

    def _phase0_step(self) -> List[Tuple[int, int, float]]:
        """
        Phase 0: Make graph 0-stable without human input.

        Per PDF: "The initial goal is to make positive-inactive assignments that
        will make the graph 0-stable. This is done entirely without human input."
        """
        # Increment phase 0 iteration counter
        self.phase0_iteration += 1
        logger.info(f"=== Phase 0 iteration {self.phase0_iteration} ===")

        # Sparsify initial graph on first iteration: reduce each PCC to its MST
        # so that make_zero_stable converges (tree-structured PCCs guarantee
        # that cutting one MST edge actually separates the unstable pair)
        if self.phase0_iteration == 1 and self.sparsify_on_init:
            sparsified = self.graph.sparsify_pccs()
            if sparsified > 0:
                logger.info(f"Initial sparsification: {sparsified} edges deactivated")

        # Discover cross-PCC edges
        num_discovered = self._discover_cross_pcc_edges()
        logger.info(f"Discovered {num_discovered} cross-PCC edges")

        # Densify PCCs
        pccs = self.graph.get_pccs()
        logger.info(f"Found {len(pccs)} PCCs")

        for pcc in pccs:
            if len(pcc) > 1:
                added = self.graph.densify_component(
                    pcc, self.classifier_manager, self.max_densify_edges,
                    prioritize_negatives=self.densify_prioritize_negatives,
                )
                if added > 0:
                    logger.info(f"Densified PCC of size {len(pcc)}: added {added} edges")

        # Make alpha-stable (phase0_alpha controls aggressiveness, default 0.0)
        deactivations = self.graph.make_zero_stable(alpha=self.phase0_alpha)
        logger.info(f"Made {deactivations} edges positive-inactive for {self.phase0_alpha}-stability")

        # Log metrics only when significant changes occur (saves ~30s per skipped iteration)
        if num_discovered > 0 or deactivations >= 5 or self.phase0_iteration <= 1:
            self._log_phase0_metrics(self.phase0_iteration)

        # Check for transition to ACTIVE_REVIEW
        all_edges_processed = self._all_edges_sorted and self._sorted_edge_index >= len(self._all_edges_sorted)
        no_edges_to_process = self._all_edges_sorted is None or len(self._all_edges_sorted) == 0
        converged = (deactivations == 0 and num_discovered == 0)

        # Detect oscillation: PCC count unchanged for consecutive iterations
        current_pcc_count = len(pccs)
        if current_pcc_count == self._phase0_prev_pcc_count and num_discovered == 0:
            self._phase0_stall_count += 1
        else:
            self._phase0_stall_count = 0
        self._phase0_prev_pcc_count = current_pcc_count

        # Force convergence if: iteration limit reached OR oscillating (stalled 3+ iters)
        force_converge = (
            self.phase0_iteration >= self.max_phase0_iterations or
            (self._phase0_stall_count >= 3 and all_edges_processed)
        )

        if force_converge and not converged:
            logger.info(f"=== Phase 0 complete - forced (iteration={self.phase0_iteration}, "
                       f"stall_count={self._phase0_stall_count}, deactivations={deactivations}) ===")
            converged = True

        if converged or force_converge:
            logger.info("=== Phase 0 complete - converged ===")
            self._log_stats()

            # Validation after Phase 0 completes
            self._handle_validation()

            if self.target_alpha > 0:
                self.phase = "ACTIVE_REVIEW"
                logger.info("=== Starting Active Review Phase ===")
                return self._active_review_step()
            else:
                self.phase = "FINISHED"
                return []

        if all_edges_processed and not converged:
            logger.info(f"All edges processed but not converged (deactivations={deactivations}, discovered={num_discovered}) - continuing")

        # Continue Phase 0 (no human review yet)
        return []

    def _compute_adaptive_quotas(self, remaining_budget: int) -> Dict[str, int]:
        """Compute per-bucket slot allocation for the next batch's selectors.

        With experimental selectors stripped, EXTERNAL is the sole non-INTERNAL
        bucket, so it takes the full remaining budget after INTERNAL (capped
        by external_per_batch if set).
        """
        if remaining_budget <= 0:
            return {'EXTERNAL': 0}
        quota = remaining_budget
        if self.external_per_batch is not None:
            quota = min(quota, int(self.external_per_batch))
        return {'EXTERNAL': quota}

    def _active_review_step(self) -> List[Tuple[int, int, float]]:
        """
        Active Review: Select candidates for human review to improve stability.

        Per PDF algorithm:
        1. Compute internal and external stability for all pairs
        2. Order by increasing stability
        3. Select first k pairs for human review
        4. (Human provides feedback - handled in step())
        5. Update labels to make graph 0-stable
        """
        # Check termination
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"Reached max human reviews ({self.max_human_reviews})")
            self.phase = "FINISHED"
            return []

        # Discover new cross-PCC edges, biased toward isolated PCCs (NIS-style)
        num_isolated = self._discover_edges_for_isolated_pccs()
        if num_isolated > 0:
            logger.info(f"Isolated PCC discovery: {num_isolated} new cross-PCC edges")
            deactivations = self.graph.make_zero_stable(alpha=0.0)
            if deactivations > 0:
                logger.info(f"Post-discovery stabilization: {deactivations} deactivations")

        # Continue regular cross-PCC discovery (PCCs change during active review)
        num_regular = self._discover_cross_pcc_edges()
        if num_regular > 0:
            logger.info(f"Regular discovery: {num_regular} new cross-PCC edges")
            deactivations = self.graph.make_zero_stable(alpha=0.0)
            if deactivations > 0:
                logger.info(f"Post-discovery stabilization: {deactivations} deactivations")

        # Get current stability
        stats = self.graph.get_graph_stats()
        min_internal = stats['min_internal_stability']
        min_external = stats['min_external_stability']

        # Handle inf values correctly:
        # - inf means "perfectly stable" (no conflicts), not "unknown"
        # - Single-element PCCs have inf internal stability (nothing to destabilize)
        # - PCCs with no cross-PCC negative edges have inf external stability
        if min_internal == float('inf') and min_external == float('inf'):
            # Graph is perfectly stable
            current_alpha = float('inf')
        elif min_internal == float('inf'):
            # Internal is perfect, use external only
            current_alpha = min_external
        elif min_external == float('inf'):
            # External is perfect, use internal only
            current_alpha = min_internal
        else:
            current_alpha = min(min_internal, min_external)

        logger.info(f"Current stability: internal={min_internal:.4f}, external={min_external:.4f}")

        if current_alpha >= self.target_alpha:
            logger.info(f"Reached target alpha ({self.target_alpha})")
            self.phase = "FINISHED"
            return []

        # Generate candidate pools — internal, external — without selecting
        # or capping. Caller (here) owns priority order and budget.
        logger.info("Starting candidate generation...")
        verified_edges = set(self.human_attempts.keys())
        pcc_strength = self._compute_pcc_separation_strength()

        pools = self.graph.generate_candidate_pools(
            alpha=self.target_alpha,
            verified_edges=verified_edges,
            unverified_threshold=0.0,
            pcc_separation_strength=pcc_strength,
        )
        logger.info("Candidate generation complete")

        # Compose the batch in strict priority order under ONE shared budget.
        # Order: INTERNAL → EXTERNAL. Internal owns intra-PCC splits, the
        # cheapest high-value reviews; EXTERNAL fills the remaining slots.
        # Total len(batch) <= edges_per_review_batch.
        budget = self.edges_per_review_batch
        batch: List[Tuple[int, int, float]] = []
        self._pending_review_types = {}
        pccs_used: Set[int] = set()
        used_keys: Set[Tuple[int, int]] = set()
        type_counts: Dict[str, int] = {}

        def _add_stability_candidate(candidate) -> bool:
            """Add a StabilityCandidate respecting per-PCC conflict + dedup.

            INTERNAL/EXTERNAL candidates are 1-per-PCC (the MST cut-point
            is a single decision per cluster).
            """
            if len(batch) >= budget:
                return False
            enforce_pcc_conflict = candidate.pcc_id is not None
            if enforce_pcc_conflict and candidate.pcc_id in pccs_used:
                return False
            u, v = candidate.review_edge
            key = (min(u, v), max(u, v))
            if key in used_keys:
                return False
            edge_data = self.graph.get_edge(u, v)
            score = edge_data.score if edge_data else 0.0
            batch.append((u, v, score))
            self._pending_review_types[key] = candidate.candidate_type
            used_keys.add(key)
            if enforce_pcc_conflict:
                pccs_used.add(candidate.pcc_id)
            type_counts[candidate.candidate_type] = type_counts.get(candidate.candidate_type, 0) + 1
            return True

        # Phase 1: INTERNAL (priority). 1-per-PCC enforced.
        for c in pools['internal']:
            if len(batch) >= budget:
                break
            _add_stability_candidate(c)

        # Phase 2: EXTERNAL fills remaining budget (optionally capped).
        remaining_after_internal = budget - len(batch)
        quotas = self._compute_adaptive_quotas(remaining_after_internal)
        if quotas['EXTERNAL'] > 0:
            ext_quota = min(quotas['EXTERNAL'], budget - len(batch))
            ext_added = 0
            for c in pools['external']:
                if len(batch) >= budget or ext_added >= ext_quota:
                    break
                if _add_stability_candidate(c):
                    ext_added += 1

        logger.info(
            f"Batch composition (budget={budget}): "
            f"INTERNAL={type_counts.get('INTERNAL', 0)}, "
            f"EXTERNAL={type_counts.get('EXTERNAL', 0)}, "
            f"total={len(batch)}"
        )

        # Filter out edges that have exceeded max human review attempts
        filtered_batch = []
        skipped_count = 0
        for edge in batch:
            n0, n1 = edge[0], edge[1]
            edge_key = (min(n0, n1), max(n0, n1))
            attempts = self.human_attempts.get(edge_key, 0)
            if attempts < self.tries_before_edge_done:
                filtered_batch.append(edge)
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} edges that exceeded max attempts ({self.tries_before_edge_done})")

        batch = filtered_batch

        if not batch:
            logger.info("No review candidates - graph at maximum stability or all candidates exceeded max attempts")
            self.phase = "FINISHED"
            return []

        logger.info(f"Selected {len(batch)} non-conflicting edges for parallel review")

        # Validation
        logger.info("Starting validation...")
        self._handle_validation()
        logger.info("Validation complete")

        # Per-iteration runtime metric tracking — see how each candidate
        # metric's FN-discrimination evolves as more human reviews accumulate.
        if self.metric_tracker_enabled:
            self._track_metric_trajectories(verified_edges)

        logger.info(f"Returning {len(batch)} edges for human review")
        return batch

    def _track_metric_trajectories(self, verified_edges: Set[Tuple[int, int]]):
        """For each tracked metric, compute its top-N candidates and report
        how many are actual false negatives against ground truth. Skipped if
        no cluster_validator GT is available."""
        validator = self.cluster_validator
        if validator is None:
            return
        gt_node2cid = getattr(validator, 'gt_node2cid', None)
        if not gt_node2cid:
            return

        summary_parts = []
        for method in self.metric_tracker_methods:
            try:
                candidates = self.graph.select_likely_fn_candidates(
                    verified_edges=verified_edges,
                    max_count=self.metric_tracker_top_n,
                    min_score=0.0,  # don't filter for tracking
                    exclude_edges=None,
                    n_eigs=64,
                    heat_kernel_t=10.0,
                    method=method,
                )
            except Exception as e:
                summary_parts.append(f"{method}: ERROR ({e})")
                continue
            n_returned = len(candidates)
            n_fn = 0
            for u, v, _ in candidates:
                gu = gt_node2cid.get(u)
                gv = gt_node2cid.get(v)
                if gu is None or gv is None:
                    continue
                if gu == gv:
                    n_fn += 1
            summary_parts.append(f"{method}: {n_fn}/{n_returned}")
        logger.info(
            f"Metric tracker (top-{self.metric_tracker_top_n} FN/total at iter "
            f"with {self.num_human_reviews} reviews): {', '.join(summary_parts)}"
        )

    def _apply_human_feedback(self, n0: int, n1: int, score: float) -> bool:
        """
        Apply human feedback to an edge.

        Per PDF step 4: Apply confidence change (agree/disagree)

        Returns: True if this could create new instability requiring re-stabilization.
        """
        if not self.graph.has_edge(n0, n1):
            return False

        edge_data = self.graph.get_edge(n0, n1)
        human_decision = score > 0.5  # Human says positive if score > 0.5

        # Determine if human agrees with current label
        current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
        human_agrees = (human_decision == current_is_positive)

        # Check if this could create instability:
        # 1. Disagree on negative -> flips to positive (new positive edge between PCCs)
        # 2. Agree on positive-inactive -> re-activates (merges PCCs, may need re-stabilization)
        was_positive_inactive = edge_data.label == EdgeLabel.POSITIVE_INACTIVE
        could_create_instability = (
            ((not human_agrees) and (not current_is_positive)) or
            (human_agrees and was_positive_inactive)
        )

        # Apply the review
        self.graph.apply_human_review(n0, n1, human_agrees, self.ch)

        logger.info(f"Human review on ({n0},{n1}): decision={'positive' if human_decision else 'negative'}, "
                   f"agrees={human_agrees}")

        return could_create_instability

    def _process_and_add_edges(self, raw_edges: List[Tuple]):
        """Process raw edges and add to graph."""
        prob_human_correct = self.config.get('prob_human_correct', 0.98)

        for edge in raw_edges:
            if len(edge) < 4:
                continue

            n0, n1, score, verifier_name = edge[:4]

            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                # Human edge - don't add directly, handled by _apply_human_feedback
                continue
            else:
                # Algorithmic classification
                classified = self.classifier_manager.classify_edge(n0, n1, verifier_name)
                _, _, score, confidence, label, ranker = classified
                edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
                self.graph.add_edge(n0, n1, edge_label, confidence, score, ranker)

    def _discover_cross_pcc_edges(self) -> int:
        """Discover edges between different PCCs.

        Uses a union-find to track PCC merges within this iteration, so that
        positive cross-PCC edges don't redundantly connect already-merged
        components (which would re-introduce density after sparsification).
        """
        pccs = self.graph.get_pccs()

        if len(pccs) <= 1:
            return 0

        # Get classifier
        first_classifier = self.classifier_manager.algo_classifiers[0] if self.classifier_manager.algo_classifiers else None
        if first_classifier is None:
            return 0

        embeddings, _ = self.classifier_manager.classifier_units[first_classifier]

        # Initialize sorted edge list
        if self._all_edges_sorted is None:
            self._initialize_sorted_edges(embeddings)

        # Build union-find for PCC tracking (stays current as positive edges merge PCCs)
        pcc_parent = {}
        for pcc_id, pcc in enumerate(pccs):
            pcc_parent[pcc_id] = pcc_id
        node_to_pcc = {}
        for pcc_id, pcc in enumerate(pccs):
            for node in pcc:
                node_to_pcc[node] = pcc_id

        def find(x):
            while pcc_parent[x] != x:
                pcc_parent[x] = pcc_parent[pcc_parent[x]]  # path compression
                x = pcc_parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                pcc_parent[ra] = rb

        # Find cross-PCC edges
        # Reset index each iteration so edges skipped (same-PCC at the time) get re-evaluated
        # after make_zero_stable may have split those PCCs
        self._sorted_edge_index = 0

        new_edges = 0
        while new_edges < self.cross_pcc_max_edges and self._sorted_edge_index < len(self._all_edges_sorted):
            n0, n1, score = self._all_edges_sorted[self._sorted_edge_index]
            self._sorted_edge_index += 1

            if self.graph.has_edge(n0, n1):
                continue

            pcc0 = node_to_pcc.get(n0)
            pcc1 = node_to_pcc.get(n1)

            # Use union-find to check if PCCs are already merged
            if find(pcc0) == find(pcc1):
                continue

            # Classify and add
            edge = self.classifier_manager.classify_edge(n0, n1, first_classifier)
            _, _, score, confidence, label, ranker = edge
            edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
            self.graph.add_edge(n0, n1, edge_label, confidence, score, ranker)
            new_edges += 1

            # If positive, merge the PCCs in our union-find
            if edge_label == EdgeLabel.POSITIVE:
                union(pcc0, pcc1)

        return new_edges

    def _compute_pcc_separation_strength(self) -> Dict[int, float]:
        """Compute separation strength for each PCC (NIS-style n_hat analog).

        For each PCC, sums the max negative confidence to each other PCC it connects to.
        High sum = confidently separated from all neighbors (well-known identity).
        Low sum = weakly separated (uncertain, may need merging) = "isolated" in NIS sense.

        Returns: Dict mapping pcc_id -> total separation confidence.
        """
        self.graph._ensure_pcc_cache()
        node_to_pcc = self.graph._node_to_pcc
        from collections import defaultdict

        # For each PCC pair, track max negative confidence
        pcc_pair_max_neg: Dict[Tuple[int, int], float] = {}

        for pcc_id, pcc in enumerate(self.graph._pccs):
            for node in pcc:
                for nbr in self.graph.G.neighbors(node):
                    nbr_pcc = node_to_pcc.get(nbr)
                    if nbr_pcc is None or nbr_pcc == pcc_id:
                        continue
                    edge_data = self.graph.G[node][nbr].get('data')
                    if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                        pair = (min(pcc_id, nbr_pcc), max(pcc_id, nbr_pcc))
                        current = pcc_pair_max_neg.get(pair, 0.0)
                        pcc_pair_max_neg[pair] = max(current, edge_data.confidence)

        # Sum max negative confidence across all PCC pairs for each PCC
        pcc_strength: Dict[int, float] = defaultdict(float)
        for (pcc_a, pcc_b), conf in pcc_pair_max_neg.items():
            pcc_strength[pcc_a] += conf
            pcc_strength[pcc_b] += conf

        return dict(pcc_strength)

    def _discover_edges_for_isolated_pccs(self) -> int:
        """Discover cross-PCC edges biased toward isolated PCCs (NIS-style).

        Uses confidence-based separation strength (NIS n_hat analog) instead of
        discrete degree. PCCs with low separation strength are "isolated" — weakly
        separated from neighbors and likely to benefit from more cross-PCC edges.
        """
        pccs = self.graph.get_pccs()

        if len(pccs) <= 1:
            return 0

        # Get classifier
        first_classifier = self.classifier_manager.algo_classifiers[0] if self.classifier_manager.algo_classifiers else None
        if first_classifier is None:
            return 0

        embeddings, _ = self.classifier_manager.classifier_units[first_classifier]

        # Initialize sorted edge list if needed
        if self._all_edges_sorted is None:
            self._initialize_sorted_edges(embeddings)

        # Compute confidence-based separation strength per PCC
        pcc_strength = self._compute_pcc_separation_strength()
        strengths = [pcc_strength.get(i, 0.0) for i in range(len(pccs))]
        sorted_strengths = sorted(strengths)
        median_strength = sorted_strengths[len(sorted_strengths) // 2]

        # Isolated = below median separation strength (weakly separated from neighbors)
        # PCCs with 0 strength have no negative edges at all — completely unknown
        isolated_pcc_ids = {i for i in range(len(pccs))
                           if pcc_strength.get(i, 0.0) <= median_strength}

        n_zero = sum(1 for s in strengths if s == 0.0)
        logger.info(f"PCC separation strength: min={min(strengths):.3f}, median={median_strength:.3f}, "
                    f"max={max(strengths):.3f}, "
                    f"zero={n_zero}/{len(pccs)}, "
                    f"isolated (<={median_strength:.3f}): {len(isolated_pcc_ids)}/{len(pccs)}")

        if not isolated_pcc_ids:
            return 0

        # Build union-find for PCC tracking (same as _discover_cross_pcc_edges)
        pcc_parent = {}
        for pcc_id, pcc in enumerate(pccs):
            pcc_parent[pcc_id] = pcc_id
        node_to_pcc = {}
        for pcc_id, pcc in enumerate(pccs):
            for node in pcc:
                node_to_pcc[node] = pcc_id

        def find(x):
            while pcc_parent[x] != x:
                pcc_parent[x] = pcc_parent[pcc_parent[x]]
                x = pcc_parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                pcc_parent[ra] = rb

        self._sorted_edge_index = 0

        new_edges = 0
        while new_edges < self.cross_pcc_max_edges and self._sorted_edge_index < len(self._all_edges_sorted):
            n0, n1, score = self._all_edges_sorted[self._sorted_edge_index]
            self._sorted_edge_index += 1

            if self.graph.has_edge(n0, n1):
                continue

            pcc0 = node_to_pcc.get(n0)
            pcc1 = node_to_pcc.get(n1)

            if find(pcc0) == find(pcc1):
                continue

            # NIS-style bias: only accept if at least one PCC is isolated
            if pcc0 not in isolated_pcc_ids and pcc1 not in isolated_pcc_ids:
                continue

            # Classify and add
            edge = self.classifier_manager.classify_edge(n0, n1, first_classifier)
            _, _, score, confidence, label, ranker = edge
            edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
            self.graph.add_edge(n0, n1, edge_label, confidence, score, ranker)
            new_edges += 1

            if edge_label == EdgeLabel.POSITIVE:
                union(pcc0, pcc1)

        return new_edges

    def _initialize_sorted_edges(self, embeddings):
        """Initialize sorted list of all possible edges using vectorized operations."""
        import numpy as np

        graph_nodes = set(self.graph.G.nodes())
        embedding_ids = embeddings.ids

        # Find which embedding indices correspond to our graph nodes
        # Build mapping: embedding_index -> node_id (only for nodes in graph)
        valid_indices = []
        valid_node_ids = []
        for idx, node_id in enumerate(embedding_ids):
            if node_id in graph_nodes:
                valid_indices.append(idx)
                valid_node_ids.append(node_id)

        n_valid = len(valid_indices)
        logger.info(f"Computing scores for {n_valid} nodes ({n_valid * (n_valid - 1) // 2} pairs)...")

        # Get embeddings for valid nodes only
        valid_embeddings = embeddings.embeddings[valid_indices]

        # Compute all pairwise cosine similarities at once using matrix multiplication
        # Normalize embeddings
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = valid_embeddings / norms

        # Cosine similarity matrix = dot product of normalized vectors
        similarity_matrix = np.dot(normalized, normalized.T)

        # Convert to scores using the same formula as embeddings.get_score_from_cosine_distance
        # cosine_distance = 1 - cosine_similarity
        # score = 1 - sign_power(cosine_distance, distance_power) * 0.5
        cosine_dist = 1 - similarity_matrix
        if embeddings.distance_power != 1:
            scores_matrix = 1 - np.sign(cosine_dist) * np.power(np.abs(cosine_dist), embeddings.distance_power) * 0.5
        else:
            scores_matrix = 1 - cosine_dist * 0.5

        # Extract upper triangular indices (i < j pairs only)
        triu_i, triu_j = np.triu_indices(n_valid, k=1)
        scores = scores_matrix[triu_i, triu_j]

        # Sort by score descending
        sorted_order = np.argsort(-scores)

        # Build sorted edge list
        all_edges = [
            (valid_node_ids[triu_i[idx]], valid_node_ids[triu_j[idx]], scores[idx])
            for idx in sorted_order
        ]

        self._all_edges_sorted = all_edges
        logger.info(f"Initialized sorted edge list with {len(all_edges)} edges")

    def is_finished(self) -> bool:
        """Check if algorithm is finished."""
        if self.phase == "FINISHED":
            return True

        # Phase 0 convergence is handled inside _phase0_step() — don't recompute here
        if self.phase == "PHASE0":
            return False

        # Active review termination
        if self.num_human_reviews >= self.max_human_reviews:
            return True

        return False

    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        """Get current clustering results."""
        cluster_dict, node2cid = self.graph.get_clustering()

        # Build NetworkX graph for compatibility
        G = nx.Graph()
        G.add_nodes_from(self.graph.G.nodes())

        for u, v, data in self.graph.G.edges(data=True):
            edge_data = data.get('data')
            if edge_data:
                G.add_edge(u, v,
                          label=edge_data.label.value,
                          confidence=edge_data.confidence,
                          score=edge_data.score,
                          ranker=edge_data.ranker,
                          is_active=(edge_data.label != EdgeLabel.POSITIVE_INACTIVE))

        return cluster_dict, node2cid, G

    def _handle_validation(self):
        """Handle periodic validation against ground truth."""
        if not self.cluster_validator:
            return

        if not self.validation_initialized:
            clustering, node2cid, G = self.get_clustering()
            self.cluster_validator.trace_start_human(clustering, node2cid, G, self.num_human_reviews)
            self.validation_initialized = True
            return

        # Periodic validation based on validation_step
        if hasattr(self.cluster_validator, 'prev_num_human'):
            if self.num_human_reviews - self.cluster_validator.prev_num_human >= self.validation_step:
                clustering, node2cid, G = self.get_clustering()
                self.cluster_validator.trace_iter_compare_to_gt(
                    clustering, node2cid, self.num_human_reviews, G
                )

    def _log_stats(self):
        """Log algorithm statistics."""
        stats = self.graph.get_graph_stats()
        mst_stats = self.graph.get_mst_cache_stats()

        logger.info("=" * 60)
        logger.info("Beta Stability Algorithm Statistics")
        logger.info("=" * 60)
        logger.info(f"Phase: {self.phase}")
        logger.info(f"Human reviews: {self.num_human_reviews}")
        logger.info(f"Target alpha: {self.target_alpha}")
        logger.info("-" * 60)
        logger.info(f"Nodes: {stats['num_nodes']}")
        logger.info(f"Edges: {stats['num_edges']}")
        logger.info(f"  Positive: {stats['edge_counts']['positive']}")
        logger.info(f"  Positive-inactive: {stats['edge_counts']['positive_inactive']}")
        logger.info(f"  Negative: {stats['edge_counts']['negative']}")
        logger.info("-" * 60)
        logger.info(f"PCCs: {stats['num_pccs']}")
        logger.info(f"PCC sizes: {stats['min_pcc_size']} - {stats['max_pcc_size']}")
        logger.info(f"Min internal stability: {stats['min_internal_stability']:.4f}")
        logger.info(f"Min external stability: {stats['min_external_stability']:.4f}")
        logger.info("-" * 60)
        logger.info(f"MST Forest:")
        logger.info(f"  Edges: {mst_stats['mst_forest_edges']}")
        logger.info("-" * 60)
        pcc_strength = self._compute_pcc_separation_strength()
        strengths = [pcc_strength.get(i, 0.0) for i in range(stats['num_pccs'])]
        if strengths:
            sorted_s = sorted(strengths)
            n_zero = sum(1 for s in strengths if s == 0.0)
            median_s = sorted_s[len(sorted_s) // 2]
            n_isolated = sum(1 for s in strengths if s <= median_s)
            logger.info(f"PCC supergraph (separation strength):")
            logger.info(f"  Strength: min={min(strengths):.3f}, "
                        f"median={median_s:.3f}, max={max(strengths):.3f}")
            logger.info(f"  Zero strength (no negative edges): {n_zero}/{len(strengths)}")
            logger.info(f"  Isolated (<=median): {n_isolated}/{len(strengths)}")
        logger.info("=" * 60)

    def show_stats(self):
        """Public method to display algorithm statistics."""
        self._log_stats()

    def _log_phase0_metrics(self, iteration):
        """Log metrics during phase 0 iterations before human review starts."""
        if not self.cluster_validator:
            return

        clustering, node2cid, G = self.get_clustering()

        # Use cluster_validator's incremental_stats directly to avoid prev_num_human check
        gt_clustering = self.cluster_validator.gt_clustering
        gt_node2cid = self.cluster_validator.gt_node2cid

        info_text = f'Phase 0 iteration {iteration}'
        result = self.cluster_validator.incremental_stats(
            0,  # num_human = 0 during phase 0
            clustering, node2cid, gt_clustering, gt_node2cid, info_text
        )
        result['phase0_iteration'] = iteration
        return result
