
import random
import networkx as nx
import numpy as np
import logging
from tools import order_edge
from threshold_utils import find_gaussian_intersection

# Update to graph_consistency.py

class GraphConsistencyAlgorithm(object):
    def __init__(self, config, classifier_manager, cluster_validator=None, G=nx.Graph()):
        """
        Args:
            classifier_manager: Handles single or multiple classifiers uniformly
            cluster_validator: Optional validation against ground truth
        """
        self.G = G.copy()
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        # Validation tracking
        self.validation_step = config.get('validation_step', 20)
        self.num_human_reviews = 0
        self.max_human_reviews = config.get('max_human_reviews', None)  # None = unlimited
        self.validation_initialized = False

        self.tries_before_edge_done = config.get('tries_before_edge_done', 4)
        self.human_attempts = {}

        # Track deactivations for internal iteration
        self.deactivations_this_iteration = 0

        # Warm-up and human review queue
        self.adjustment_warmup_iterations = config.get('adjustment_warmup_iterations', 5)
        self.warmup_iterations = config.get('warmup_iterations', 10)
        self.warmup_completed = False
        self.human_review_queue = set()  # Set to avoid duplicates
        self.edges_per_review_batch = config.get('edges_per_review_batch', 20)

        # Reactivation phase flag - when True, only send to human review, no deactivation
        self.reactivation_phase = False

        # Threshold adjustment parameters
        self.reviews_before_adjustment = config.get('reviews_before_adjustment', 0)  # 0 = disabled
        self.last_adjustment_reviews = 0

        logger = logging.getLogger('lca')
        logger.info(f"GC Algorithm initialized with theta={self.config['theta']}, max_human_reviews={self.max_human_reviews}")

        # Pre-sorted list of all possible edges (initialized lazily)
        self._all_edges_sorted = None
        self._sorted_edge_index = 0  # Track position in the sorted list

    def initialize_sorted_edge_list(self, embeddings):
        """
        Initialize the sorted list of all possible edges by score (descending).
        Called lazily on first cross-PCC edge discovery.
        """
        logger = logging.getLogger('lca')

        all_nodes = list(self.G.nodes())
        n_nodes = len(all_nodes)

        logger.info(f"Computing all pairwise scores for {n_nodes} nodes ({n_nodes * (n_nodes - 1) // 2} edges)...")

        all_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                n0, n1 = all_nodes[i], all_nodes[j]
                score = embeddings.get_score(n0, n1)
                all_edges.append((n0, n1, score))

        # Sort by score descending (highest score = most likely positive, first)
        all_edges.sort(key=lambda x: x[2], reverse=True)

        self._all_edges_sorted = all_edges
        self._sorted_edge_index = 0
        logger.info(f"Initialized sorted edge list with {len(all_edges)} edges (top score: {all_edges[0][2]:.4f}, bottom: {all_edges[-1][2]:.4f})")

    def get_confidence(self, n0, n1):
        """
        Centralized method to allow noise injection at decision time.
        """
        confidence = self.G[n0][n1]['confidence']
        return confidence

    def filter_for_review(self, edges_for_human):
        logger = logging.getLogger('lca')
        final_for_review = []
        for n0, n1, s in edges_for_human:
            edge_key = order_edge(n0, n1)
            attempts = self.human_attempts.get(edge_key, 0)
            
            if attempts < self.tries_before_edge_done:
                self.human_attempts[edge_key] = attempts + 1
                final_for_review.append((n0, n1, s))
            # else:
            #     logger.warning(f"Edge ({n0}, {n1}) exceeded max human attempts ({self.tries_before_edge_done})")
        return final_for_review

    def step(self, new_edges):
        """
        Perform one step of the graph consistency algorithm.
        Iterates internally until convergence or human review needed.
        """
        # Add initial edges
        processed_edges = self.process_raw_edges(new_edges)
        self.add_new_edges(processed_edges)

        # Count human reviews for validation
        human_edges = [e for e in new_edges if 'human' in e[3]]
        self.num_human_reviews += len(human_edges)

        logger = logging.getLogger('lca')

        # Warm-up: Run multiple iterations only on first call
        if not self.warmup_completed:
            # max_iterations = self.warmup_iterations
            self.warmup_completed = True
            logger.info(f"=== WARM-UP PHASE: {self.warmup_iterations} iterations ===")
        else:
            self.warmup_iterations = 1  # After warm-up, single iteration per step

        # Threshold adjustment based on accumulated human reviews
        if self.reviews_before_adjustment > 0:
            reviews_since_adjustment = self.num_human_reviews - self.last_adjustment_reviews
            if reviews_since_adjustment >= self.reviews_before_adjustment:
                logger.info(f"Triggering threshold adjustment after {reviews_since_adjustment} reviews")
                self.adjust_threshold()
                self.warmup_iterations = self.adjustment_warmup_iterations

        accumulated_human_review_edges = set()
        iteration = 0
        while iteration < self.warmup_iterations:
            iteration += 1
        # for iteration in range(max_iterations):
            logger.info(f"Internal iteration {iteration}")
            self.deactivations_this_iteration = 0  # Reset counter

            # Discover cross-PCC edges EVERY iteration to balance precision/recall
            num_discovered, cross_edges_for_human = self.discover_cross_pcc_edges()
            logger.info(f"Discovered {num_discovered} cross-PCC edges")

            # Find inconsistencies
            iPCCs = self.find_inconsistent_PCCs(logger)

            if not iPCCs:
                logger.info("No inconsistent PCCs found")

                if num_discovered > 0:
                    logger.info("Cross-PCC edges discovered, continuing iteration")
                    continue  # Re-check for iPCCs with new edges
                else:
                    if self.config.get('reactivation_batch_size', 0) > 0:
                        # Try reactivating deactivated edges before declaring convergence
                        num_reactivated = self.reactivate_batch()
                        if num_reactivated > 0:
                            logger.info(f"Reactivated {num_reactivated} edges for reconsideration")
                            continue  # Re-check for iPCCs with reactivated edges
                    logger.info("No inconsistencies, no new edges, no deactivated edges - fully converged!")
                    break

            # Process all iPCCs
            edges_needing_classification = []
            for iPCC in iPCCs:
                logger.info(f"iPCC size: {len(iPCC.nodes())}")
                edges_needing_classification.extend(self.process_iPCC(iPCC))

            logger.info(f"{len(edges_needing_classification)} edges need classification")

            # Let ClassifierManager handle algorithmic classification vs human review decision
            
            new_classifications, edges_for_human = self.classifier_manager.classify_or_request_human(
                edges_needing_classification
            )

            # Add new algorithmic classifications to graph
            if new_classifications:
                logger.info(f"Can update {len(new_classifications)} edges")
                self.add_new_edges(new_classifications)

            # Filter and accumulate human review edges
            edges_for_human = edges_for_human
            
            edges_for_human = self.filter_for_review(edges_for_human)
            logger.info(f"{len(edges_for_human)} edges need human review this iteration")

            for n0, n1, s in edges_for_human:
                # Get current confidence from graph
                confidence = self.G[n0][n1]['confidence']
                # Order edge to avoid duplicates like (a,b) and (b,a)
                ordered_edge = (*order_edge(n0, n1), s, confidence)
                accumulated_human_review_edges.add(ordered_edge)

            # Log metrics during phase 0 (warmup) iterations
            if self.num_human_reviews == 0:
                self._log_phase0_metrics(iteration)

            # Check if we made autonomous progress via deactivations
            if self.deactivations_this_iteration == 0:
                # No deactivations - can't make more progress without human input
                logger.info(f"No deactivations in iteration {iteration} - stopping internal loop")
                break
            else:
                logger.info(f"Made {self.deactivations_this_iteration} deactivations - continuing iteration")

        if iteration >= self.warmup_iterations - 1:
            logger.warning(f"Reached max internal iterations ({self.warmup_iterations})")

        # Store current iPCCs for is_finished() check
        self.current_iPCCs = self.find_inconsistent_PCCs(logger)

        # Validation logic
        self._handle_validation()

        

        # Add accumulated edges to review queue
        self.human_review_queue.update(accumulated_human_review_edges)
        logger.info(f"Queue now has {len(self.human_review_queue)} edges")

        # Return next batch from queue (sorted by confidence)
        batch = self._get_next_review_batch()
        logger.info(f"Returning {len(batch)} edges for review")

        return batch

    def _get_next_review_batch(self):
        """Get next batch of edges from review queue, prioritizing low confidence."""
        # Sort queue by confidence (index 3) - low confidence first
        sorted_queue = sorted(self.human_review_queue, key=lambda x: x[3])

        # Take first N edges and convert back to 3-tuple format for human reviewer
        batch_size = self.edges_per_review_batch
        batch = [(n0, n1, s) for n0, n1, s, conf in sorted_queue[:batch_size]]

        # Remove batch edges from queue (need to match 4-tuple format)
        for edge in sorted_queue[:batch_size]:
            self.human_review_queue.discard(edge)

        return batch

    def _handle_validation(self):
        """Handle periodic validation against ground truth."""
        if not self.cluster_validator:
            return
            
        
        # Initial validation after first step
        if not self.validation_initialized:
            clustering, node2cid, G = self.get_clustering()
            self.cluster_validator.trace_start_human(clustering, node2cid, G, self.num_human_reviews)
            self.validation_initialized = True
            return
        
        # Periodic validation
        if hasattr(self.cluster_validator, 'prev_num_human'):
            if self.num_human_reviews - self.cluster_validator.prev_num_human >= self.validation_step:
                self.show_stats()
        else:
            # Fallback if prev_num_human not available
            if self.num_human_reviews % self.validation_step == 0:
                self.show_stats()
        
    def show_stats(self):
        clustering, node2cid, G = self.get_clustering()
        self.cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, self.num_human_reviews, G)

    def _log_phase0_metrics(self, iteration):
        """Log metrics during phase 0 (warmup) iterations before human review starts."""
        if not self.cluster_validator:
            return

        logger = logging.getLogger('lca')
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

    def densify_component(self, subG):
        max_edges_to_add = self.config["max_densify_edges"]
        logger = logging.getLogger('lca')

        # Get first classifier for scoring
        first_classifier = self.classifier_manager.algo_classifiers[0] if self.classifier_manager.algo_classifiers else None
        if first_classifier is None:
            logger.warning("No algorithmic classifiers available for densification")
            return self.G.subgraph(subG.nodes())

        embeddings, _ = self.classifier_manager.classifier_units[first_classifier]

        # Find all missing edges with their scores
        missing_edges_with_scores = []
        nodes = list(subG.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n0, n1 = nodes[i], nodes[j]
                if not self.G.has_edge(n0, n1):
                    score = embeddings.get_score(n0, n1)
                    missing_edges_with_scores.append((n0, n1, score))

        # Limit based on number of edges TO ADD (not total edges)
        if len(missing_edges_with_scores) > max_edges_to_add:
            # Sort by score (ASCENDING) to prioritize low scores (potential negatives)
            # Low scores are more likely to split the cluster
            missing_edges_with_scores.sort(key=lambda x: x[2])
            logger.info(f"Prioritizing {max_edges_to_add}/{len(missing_edges_with_scores)} lowest-scoring edges for densification")
            missing_edges_with_scores = missing_edges_with_scores[:max_edges_to_add]

        # Classify missing edges
        new_edges = []
        for n0, n1, score in missing_edges_with_scores:
            edge = self.classifier_manager.classify_edge(n0, n1, first_classifier)
            new_edges.append(edge)

        self.add_new_edges(new_edges)
        return self.G.subgraph(subG.nodes())

    def densify_iPCCs(self, iPCCs):
        """Add missing edges using classifier system."""
        updated_iPCCs = []
        for subG in iPCCs:
            updated_iPCCs.append(self.densify_component(subG))

        return updated_iPCCs

    def discover_cross_pcc_edges(self):
        """
        Discover edges between different positive connected components.

        Uses a pre-sorted list of ALL possible edges (sorted by score, highest first).
        Iterates through the list and adds top edges that:
        - Are not already in the graph
        - Connect two different PCCs (not within the same PCC)

        Returns number of edges discovered.
        """
        logger = logging.getLogger('lca')

        components = self.get_positive_components(self.G)

        if len(components) <= 1:
            logger.info("Only one PCC, no cross-PCC edges to discover")
            return 0, []

        logger.info(f"Discovering cross-PCC edges between {len(components)} components")

        # Get first classifier for scoring
        first_classifier = self.classifier_manager.algo_classifiers[0] if self.classifier_manager.algo_classifiers else None
        if first_classifier is None:
            logger.warning("No algorithmic classifiers available for edge discovery")
            return 0

        embeddings, _ = self.classifier_manager.classifier_units[first_classifier]

        # Initialize sorted edge list if not already done
        if self._all_edges_sorted is None:
            self.initialize_sorted_edge_list(embeddings)
            logger.info(f"Initialized queue of edges: {len(self._all_edges_sorted)}")

        # Build node-to-component mapping for current PCCs
        node_to_component = {}
        for comp_idx, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = comp_idx

        # Configuration
        max_edges_to_discover = self.config.get('cross_pcc_max_edges', 100)

        new_edges = []
        edges_checked = 0
        edges_skipped_in_graph = 0
        edges_skipped_same_pcc = 0
        human_edges = []
        # Iterate through sorted edges starting from current index (highest score first)
        while len(new_edges) < max_edges_to_discover and self._sorted_edge_index < len(self._all_edges_sorted):
            n0, n1, score = self._all_edges_sorted[self._sorted_edge_index]
            self._sorted_edge_index += 1
            edges_checked += 1

            # Skip if already in graph
            if self.G.has_edge(n0, n1):
                edges_skipped_in_graph += 1
                continue

            # Get component membership for both nodes
            comp0 = node_to_component.get(n0)
            comp1 = node_to_component.get(n1)            

            # Skip if both nodes are in the same PCC (we want cross-PCC edges only)
            if comp0 == comp1:
                edges_skipped_same_pcc += 1
                continue

            # This edge connects two different PCCs - classify and add
            edge = self.classifier_manager.classify_edge(n0, n1, first_classifier)
            ordered_edge = (*order_edge(n0, n1), edge[2])
            human_edges.append(ordered_edge)
            new_edges.append(edge)

        if new_edges:
            logger.info(f"Discovered {len(new_edges)} cross-PCC edges from sorted list "
                       f"(checked {edges_checked}, skipped {edges_skipped_in_graph} in-graph, "
                       f"{edges_skipped_same_pcc} same-PCC, index now at {self._sorted_edge_index}/{len(self._all_edges_sorted)})")
            self.add_new_edges(new_edges)
        else:
            logger.info(f"No new cross-PCC edges found (checked {edges_checked} edges, "
                       f"index at {self._sorted_edge_index}/{len(self._all_edges_sorted)})")

        return len(new_edges), human_edges

    def discover_cross_cluster_inconsistencies(self):
        """
        Find cross-cluster inconsistencies where separation is weaker than internal structure.

        For each pair of clusters A and B:
        - Find the minimum confidence positive edge within A
        - Find the minimum confidence positive edge within B
        - Find the minimum confidence negative edge between A and B
        - If min_inter < min(min_intra_A, min_intra_B) * ratio:
          The separation is relatively weaker than internal structure, send for human review.

        Uses ratio instead of absolute margin to handle small confidence values properly.
        No deactivation - queue is a set so duplicates prevented.
        After human review, edge gets high confidence and won't trigger anymore.
        """
        logger = logging.getLogger('lca')

        components = self.get_positive_components(self.G)

        if len(components) <= 1:
            return []

        ratio = self.config.get('cross_cluster_ratio', 0.8)
        edges_to_reclassify = []

        for i, cluster_A in enumerate(components):
            for j in range(i + 1, len(components)):
                cluster_B = components[j]

                # Find min confidence positive edge within cluster A
                min_intra_A = None
                for node_a in cluster_A:
                    for node_b in cluster_A:
                        if node_a < node_b and self.G.has_edge(node_a, node_b):
                            edge_data = self.G[node_a][node_b]
                            if edge_data.get('label') == 'positive' and edge_data.get('is_active'):
                                conf = edge_data['confidence']
                                if min_intra_A is None or conf < min_intra_A:
                                    min_intra_A = conf

                # Find min confidence positive edge within cluster B
                min_intra_B = None
                for node_a in cluster_B:
                    for node_b in cluster_B:
                        if node_a < node_b and self.G.has_edge(node_a, node_b):
                            edge_data = self.G[node_a][node_b]
                            if edge_data.get('label') == 'positive' and edge_data.get('is_active'):
                                conf = edge_data['confidence']
                                if min_intra_B is None or conf < min_intra_B:
                                    min_intra_B = conf

                # Find min confidence negative edge between clusters
                min_inter = None
                min_inter_edge = None
                for node_a in cluster_A:
                    for node_b in cluster_B:
                        if self.G.has_edge(node_a, node_b):
                            edge_data = self.G[node_a][node_b]
                            if edge_data.get('label') == 'negative' and edge_data.get('is_active'):
                                conf = edge_data['confidence']
                                if min_inter is None or conf < min_inter:
                                    min_inter = conf
                                    min_inter_edge = (node_a, node_b)

                # Skip if missing required edges
                if min_intra_A is None or min_intra_B is None or min_inter is None:
                    continue

                min_intra = min(min_intra_A, min_intra_B)

                if min_inter < min_intra * ratio:
                    node_a, node_b = min_inter_edge
                    ranker = self.G[node_a][node_b].get('ranker', None)
                    edges_to_reclassify.append((node_a, node_b, ranker))
                    # self.G[node_a][node_b]['label'] = 'positive'

                    logger.info(f"Cross-cluster inconsistency (sizes {len(cluster_A)}, {len(cluster_B)}): "
                              f"min_intra={min_intra:.3f}, min_inter={min_inter:.3f}, ratio={min_inter/min_intra:.2f}, edge={min_inter_edge}")

        return edges_to_reclassify

    def deactivate(self, deactivator, deactivatee):

        n0, n1 = deactivator
        d0, d1 = deactivatee

        deativated_tuple = order_edge(d0, d1)

        self.G.edges[n0, n1]['inactivated_edges'].add(deativated_tuple)

        self.G.edges[d0, d1]['is_active'] = False
        self.G.edges[d0, d1]['deactivator'] = deactivator

        # Track that a deactivation occurred
        self.deactivations_this_iteration += 1

    def reactivate(self, n0, n1):
        """
        Reactivate a single deactivated edge.
        Cleans up the deactivator's inactivated_edges set.
        """
        if not self.G.has_edge(n0, n1):
            return
        edge_data = self.G.edges[n0, n1]
        if edge_data.get('is_active', True):
            return  # Already active

        # Clean up deactivator's tracking
        deactivator = edge_data.get('deactivator')
        if deactivator:
            d0, d1 = deactivator
            if self.G.has_edge(d0, d1):
                edge_key = order_edge(n0, n1)
                self.G.edges[d0, d1]['inactivated_edges'].discard(edge_key)

        # Reactivate
        edge_data['is_active'] = True
        edge_data['deactivator'] = None

    def get_deactivated_edges(self):
        """
        Get all currently deactivated edges that can still be sent for human review.
        Excludes edges that have exhausted their human review attempts.
        Sorted by confidence (lowest first).
        """
        deactivated = []
        for u, v, data in self.G.edges(data=True):
            if not data.get('is_active', True):
                # Skip edges that have exhausted human review attempts
                edge_key = order_edge(u, v)
                attempts = self.human_attempts.get(edge_key, 0)
                if attempts >= self.tries_before_edge_done:
                    continue
                deactivated.append((u, v, data.get('confidence', 0)))
        # Sort by confidence - lowest confidence first (most uncertain)
        deactivated.sort(key=lambda x: x[2])
        return [(u, v) for u, v, _ in deactivated]

    def reactivate_batch(self):
        """
        Reactivate a batch of deactivated edges after convergence.
        Returns number of edges reactivated.

        Called multiple times until all deactivated edges are processed.
        Sets reactivation_phase=True on first call (stays True throughout).
        """
        logger = logging.getLogger('lca')
        batch_size = self.config.get('reactivation_batch_size', 0)
        deactivated = self.get_deactivated_edges()

        if not deactivated:
            return 0

        # Enter reactivation phase on first call - only human review, no deactivation
        if not self.reactivation_phase:
            self.reactivation_phase = True
            self.warmup_iterations = self.config.get('warmup_iterations', 10)
            self.warmup_completed = False
            logger.info("Entering reactivation phase - edges will only go to human review")

        batch = deactivated[:batch_size]
        for n0, n1 in batch:
            self.reactivate(n0, n1)
            logger.info(f"Reactivated edge ({n0}, {n1}) for reconsideration")

        return len(batch)

    def is_finished(self):
        """
        Check if the algorithm is finished.

        Returns:
            bool: True if no inconsistent PCCs exist AND review queue is empty AND no deactivated edges
                  OR if max_human_reviews limit is reached
        """
        logger = logging.getLogger('lca')

        # Check max_human_reviews limit first
        if self.max_human_reviews is not None and self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"Reached max human reviews limit ({self.max_human_reviews})")
            return True

        no_ipcc = len(getattr(self, 'current_iPCCs', [])) == 0
        queue_empty = len(self.human_review_queue) == 0
        no_deactivated = self.config.get('reactivation_batch_size', 0) == 0 or len(self.get_deactivated_edges()) == 0
        no_more_edges = self._sorted_edge_index >= len(self._all_edges_sorted)
        logger.info("Checking finish condition:")
        logger.info(f"  - no_ipcc: {no_ipcc}")
        logger.info(f"  - queue_empty: {queue_empty}")
        logger.info(f"  - no_deactivated: {no_deactivated}")
        logger.info(f"  - no_more_edges: {no_more_edges}")
        if self.max_human_reviews is not None:
            logger.info(f"  - human_reviews: {self.num_human_reviews}/{self.max_human_reviews}")
        return no_ipcc and queue_empty and no_deactivated and no_more_edges

    def get_clustering(self):
        """
        Get current clustering results.
        Compatible with LCA interface.
        
        Returns:
            tuple: (clustering_dict, node2cid_dict)
        """
        return (*self.get_positive_clusters(), self.G)

    def get_positive_clusters(self):
        """
        Get clustering from the graph where clusters are connected components of positive edges.

        Args:
            G (nx.Graph): The graph containing nodes and edges with labels.

        Returns:
            cluster_dict (dict): A dictionary mapping cluster IDs to sets of nodes.
            node2cid (dict): A mapping of each node to its cluster ID.
        """

        # Get connected components (already includes singletons)
        clusters = self.get_positive_components(self.G)

        # Convert list to a dictionary {cluster_id: set_of_nodes}
        cluster_dict = {cid: cluster for cid, cluster in enumerate(clusters)}
        
        # Create a node-to-cluster ID mapping
        node2cid = {node: cid for cid, cluster in cluster_dict.items() for node in cluster}

        return cluster_dict, node2cid

    def process_raw_edges(self, raw_edges):
        """Convert raw edges to GC internal format using ClassifierManager."""
        processed_edges = []
        prob_human_correct = self.config.get('prob_human_correct', 0.98)
        
        for n0, n1, score, verifier_name in raw_edges:
            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                # Handle human decisions with config-based confidence
                decision = (score > 0.5)
                confidence = prob_human_correct
                label = "positive" if decision else "negative"
                processed_edges.append((n0, n1, score, confidence, label, verifier_name))
            else:
                # Use ClassifierManager to classify with appropriate classifier
                if verifier_name in self.classifier_manager.classifier_units:
                    embeddings, classifier = self.classifier_manager.classifier_units[verifier_name]
                    label, confidence = classifier.classify(score)
                    processed_edges.append((n0, n1, score, confidence, label, verifier_name))
        
        return processed_edges

    def add_new_edges(self, new_edges):
        """
        Add new edges from the ranker into the consistency graph. 
        Each edge is run through the verifier to generate the label and a confidence score before adding it to the graph.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
        """
        logger = logging.getLogger('lca')

        for n0, n1, score, confidence, label, ranker_name in new_edges:
        
            confidence = np.clip(confidence, 0, 1)
            # print(f"adding edge ... {n0}, {n1} {score}")
            # if label == "positive":
            #     positive_G = self.get_positive_subgraph(self.G)
            #     components = list(nx.connected_components(positive_G))
            #     c0 = next(filter(lambda c: n0 in c, components), {})
            #     c1 = next(filter(lambda c: n1 in c, components), {})
            #     if c0 != c1:
            #         logger.info(f"Added positive edge {n0, n1, float(score), float(confidence)} connecting clusters of size {max(len(c0), 1)} and {max(len(c1), 1)}")
            # else:
            #     logger.info(f"Negative confidence: {confidence}")
            if self.G.has_edge(n0, n1):
                old_edge = self.G.edges[n0, n1]
                if label != old_edge["label"] or confidence < old_edge["confidence"]:
                    # Reactivate all edges that were deactivated by this edge
                    for (v0, v1) in list(old_edge['inactivated_edges']):
                        self.reactivate(v0, v1)
                    old_edge['inactivated_edges'] = set()
                    # Reactivate this edge if it was deactivated
                    self.reactivate(n0, n1)
                    
                # logger.info(f"Existing edge {old_edge}")
                
                # logger.info(f"Updating existing edge ({n0}, {n1}) with label {label}, confidence {confidence}, score {score}, ranker {ranker_name}")
                self.G.edges[n0, n1]["label"] = label
                self.G.edges[n0, n1]["confidence"] = confidence
                self.G.edges[n0, n1]["ranker"] = ranker_name
                self.G.edges[n0, n1]["score"] = score
            else:
                self.G.add_edge(n0, n1, score=score, label=label, confidence=confidence, is_active=True, inactivated_edges=set(), deactivator=None, ranker=ranker_name, auto_flipped=False)
        

    def get_positive_components(self, G, min_confidence=0):
        """
        Get connected components based on positive active edges.
        Uses subgraph_view for efficiency (no copying).

        Returns:
            list of sets: Connected components (including singletons)
        """
        def filter_edge(u, v):
            d = G[u][v]
            return (d.get("label") == "positive" and
                    d.get("confidence", 0) > min_confidence and
                    d.get('is_active', False))

        view = nx.subgraph_view(G, filter_edge=filter_edge)
        components = list(nx.connected_components(view))

        # Add singletons (nodes with no positive edges)
        used_nodes = {n for c in components for n in c}
        singletons = [{n} for n in G.nodes() if n not in used_nodes]

        return components + singletons

    

    def find_inconsistent_PCCs(self, logger):
        """
        Find inconsistent Positive Connected Components, i.e. a connected components generate by positive edges,
        with at least one negative edge between the nodes of that component.

        Returns a list of subgraphs corresponding to inconsistent Positive Connected Components.
        """
        result = []

        # Get connected components (uses view, no copying)
        components = self.get_positive_components(self.G)
        logger.info(f"Found connected components {len(components)}")   
        # logger.info(f"Total positive edges: {np.sum(d.get('label') == 'negative' for _, _, d in self.G.edges(data=True))}")
        # logger.info(f"Total negative edges: {np.sum(d.get('label') == 'positive' for _, _, d in self.G.edges(data=True))}")
        # Check each component for inconsistencies
        maxsize = 0
        for component in components:
            # logger.info(f"Nodes in component: {len(component)}")
            subG = self.G.subgraph(component)  # Get all edges within the component
            maxsize = max(maxsize, len(subG.nodes()))
            subG = self.densify_component(subG)
            # if len(component) > self.config["densify_threshold"]:
            #     u, v, _ = min([(u,v, d["confidence"]) for u, v, d in subG.edges(data=True) if d["label"]=="positive"], key=lambda edge: edge[2])
            #     self.G[u][v]["label"] = "negative"
            # Check for negative edges
            if any(d.get("label") == "negative" and d.get("is_active") for _, _, d in subG.edges(data=True)):
                result.append(subG)
        logger.info(f"Largest component {maxsize}") 
        logger.info(f"Found inconsistent components {len(result)}")   
        return result


    def process_iPCC(self, iPCC):
        """
        Process an inconsistent PCC and return an edge that needs additional review.

        Args:
            iPCC (nx.Graph): a subgraph containing inconsistent PCC
        """
        result = set()

        # Find all negative edges in iPCC
        negative_edges = [(u, v) for u, v, d in iPCC.edges(data=True)
                          if d.get("label") == "negative" and d.get('is_active')]

        if not negative_edges:
            return result

        # Build graph of non-negative active edges with confidence as weight
        nonneg_graph = nx.Graph()
        for u, v, d in iPCC.edges(data=True):
            if d.get("label") != "negative" and d.get('is_active'):
                nonneg_graph.add_edge(u, v, weight=d.get('confidence', 0))

        if nonneg_graph.number_of_edges() == 0:
            return result

        # Maximum Spanning Tree: path between any two nodes in MST is the widest path
        # (path with highest minimum edge weight) in the original graph
        mst = nx.maximum_spanning_tree(nonneg_graph, weight='weight')

        for n0, n1 in negative_edges:
            neg_confidence = self.get_confidence(n0, n1)

            # Find widest path from n0 to n1 (path in MST)
            if not (mst.has_node(n0) and mst.has_node(n1)):
                continue

            try:
                path = nx.shortest_path(mst, n0, n1)
            except nx.NetworkXNoPath:
                continue

            if len(path) < 2:
                continue

            # Find minimum confidence edge in the path
            path_edges = list(zip(path[:-1], path[1:]))
            u, v, min_confidence = min(
                [(a, b, self.get_confidence(a, b)) for a, b in path_edges],
                key=lambda edge: edge[2]
            )

            # During reactivation phase, use infinite theta to prevent deactivation
            theta = float('inf') if self.reactivation_phase else self.config['theta']

            if min_confidence > neg_confidence:
                if abs(min_confidence - neg_confidence) > theta:
                    self.deactivate((u, v), (n0, n1))
                else:
                    edge_key = order_edge(n0, n1)
                    if self.human_attempts.get(edge_key, 0) < self.tries_before_edge_done:
                        result.add((*edge_key, self.G[n0][n1].get('ranker', None)))
                    if self.reactivation_phase:
                        self.deactivate((u, v), (n0, n1))
            else:
                if abs(min_confidence - neg_confidence) > theta:
                    self.deactivate((n0, n1), (u, v))
                else:
                    edge_key = order_edge(u, v)
                    if self.human_attempts.get(edge_key, 0) < self.tries_before_edge_done:
                        result.add((*edge_key, self.G[u][v].get('ranker', None)))
                    if self.reactivation_phase:
                        self.deactivate((n0, n1), (u, v))

        return result

    def adjust_threshold(self):
        """
        Adjust classifier threshold based on Gaussian intersection of assigned labels.
        Reclassifies all active edges that were classified by the adjusted classifier.
        """
        logger = logging.getLogger('lca')

        # Get the first algorithmic classifier (the one we'll adjust)
        if not self.classifier_manager.algo_classifiers:
            logger.warning("No algorithmic classifiers available for threshold adjustment")
            return

        classifier_name = self.classifier_manager.algo_classifiers[0]
        embeddings, classifier = self.classifier_manager.classifier_units[classifier_name]

        # Check if classifier has a threshold attribute
        if not hasattr(classifier, 'threshold'):
            logger.warning(f"Classifier {classifier_name} does not have a threshold attribute")
            return

        # Collect scores and edges to reclassify in single pass
        positive_scores = []
        negative_scores = []
        edges_to_reclassify = []
        old_labels = {}

        for u, v, data in self.G.edges(data=True):
            if not data.get('is_active', False):
                continue

            ranker = data.get('ranker', '')
            label = data.get('label', '')

            # Get score for threshold calculation
            if 'human' in ranker:
                score = embeddings.get_score(u, v)
            else:
                score = data.get('score', 0.0)

            if label == 'positive':
                positive_scores.append(score)
            elif label == 'negative':
                negative_scores.append(score)

            # Track edges that need reclassification
            if ranker == classifier_name:
                edges_to_reclassify.append((u, v))
                old_labels[(u, v)] = label

        # Calculate Gaussian intersection
        new_threshold = find_gaussian_intersection(positive_scores, negative_scores)

        if new_threshold is None:
            logger.info("Could not compute Gaussian intersection, keeping current threshold")
            return

        old_threshold = classifier.threshold
        logger.info(f"Adjusting threshold: {old_threshold:.4f} -> {new_threshold:.4f}")
        classifier.threshold = old_threshold + (new_threshold - old_threshold) * self.config.get('threshold_adjustment_factor', 0.1)

        logger.info(f"Reclassifying {len(edges_to_reclassify)} active edges with new threshold")

        # Reclassify edges and count label flips
        reclassified_edges = []
        neg_to_pos = 0
        pos_to_neg = 0
        for u, v in edges_to_reclassify:
            old_label = old_labels[(u, v)]
            edge = self.classifier_manager.classify_edge(u, v, classifier_name)
            new_label = edge[4]
            if old_label == 'negative' and new_label == 'positive':
                neg_to_pos += 1
            elif old_label == 'positive' and new_label == 'negative':
                pos_to_neg += 1
            reclassified_edges.append(edge)

        # Update graph with reclassified edges
        self.add_new_edges(reclassified_edges)

        logger.info(f"Label flips: {neg_to_pos} negative→positive, {pos_to_neg} positive→negative")

        self.last_adjustment_reviews = self.num_human_reviews

    def get_active_edge_scores(self):
        """
        Get scores of all active edges grouped by assigned label and ground truth label.
        For human-verified edges, uses the original embedding score instead of 0/1.

        Returns:
            dict: {
                'assigned_positive_scores': scores of edges with assigned positive label,
                'assigned_negative_scores': scores of edges with assigned negative label,
                'gt_positive_scores': scores of edges where nodes are in same GT cluster,
                'gt_negative_scores': scores of edges where nodes are in different GT clusters,
                'classification_threshold': the threshold used by the classifier for labeling
            }
        """
        assigned_positive_scores = []
        assigned_negative_scores = []
        gt_positive_scores = []
        gt_negative_scores = []

        # Get embeddings to retrieve original scores for human-verified edges
        classifier_name = self.classifier_manager.algo_classifiers[0]
        embeddings, classifier = self.classifier_manager.classifier_units[classifier_name]

        # Get ground truth node2cid if available
        gt_node2cid = None
        if self.cluster_validator is not None:
            gt_node2cid = self.cluster_validator.gt_node2cid

        for u, v, data in self.G.edges(data=True):
            if data.get('is_active', False):
                # Get the original score from embeddings (not the 0/1 from human review)
                ranker = data.get('ranker', '')
                if 'human' in ranker:
                    score = embeddings.get_score(u, v)
                else:
                    score = data.get('score', 0.0)

                # Assigned label distribution
                label = data.get('label', '')
                if label == 'positive':
                    assigned_positive_scores.append(score)
                elif label == 'negative':
                    assigned_negative_scores.append(score)

                # Ground truth label distribution
                if gt_node2cid is not None and u in gt_node2cid and v in gt_node2cid:
                    if gt_node2cid[u] == gt_node2cid[v]:
                        gt_positive_scores.append(score)
                    else:
                        gt_negative_scores.append(score)

        classification_threshold = classifier.threshold

        return {
            'assigned_positive_scores': assigned_positive_scores,
            'assigned_negative_scores': assigned_negative_scores,
            'gt_positive_scores': gt_positive_scores,
            'gt_negative_scores': gt_negative_scores,
            'classification_threshold': classification_threshold
        }


