"""
Thresholded Review Algorithm.

This algorithm uses weighted edges with thresholds to minimize human review:
- Edges with score < low_threshold are automatically rejected
- Edges with score > high_threshold are automatically accepted
- Edges with score between thresholds require human review
Only positive (accepted) edges are kept to build the graph.
"""

import logging
import numpy as np
import time
from collections import defaultdict
import networkx as nx
from tools import write_json
import os

logger = logging.getLogger('lca')


class ThresholdedReviewAlgorithm:
    """
    Thresholded weighted edges with targeted human review.

    Uses thresholds to automatically accept/reject edges,
    only requesting human review for uncertain cases.
    """

    def __init__(self, config, common_data):
        """
        Initialize the thresholded review algorithm.

        Args:
            config: Algorithm configuration including:
                - low_threshold: Below this, edges are auto-rejected (default: 0.3)
                - high_threshold: Above this, edges are auto-accepted (default: 0.7)
                - topk: Number of top neighbors to consider per node (default: 10)
                - max_reviews: Maximum human reviews to conduct (optional)
                - clustering_method: 'connected_components' or 'community' (default: 'connected_components')
                - review_batch_size: Number of edges to review in each batch (default: 100)
            common_data: Common data from algorithm preparation
        """
        self.config = config
        self.common_data = common_data

        # Algorithm parameters
        self.low_threshold = config.get('low_threshold', 0.3)
        self.high_threshold = config.get('high_threshold', 0.7)
        self.topk = config.get('topk', 10)
        self.max_reviews = config.get('max_reviews', None)
        self.clustering_method = config.get('clustering_method', 'connected_components')
        self.review_batch_size = config.get('review_batch_size', 100)

        # Validate thresholds
        if self.low_threshold >= self.high_threshold:
            raise ValueError(f"low_threshold ({self.low_threshold}) must be less than high_threshold ({self.high_threshold})")

        # Get embeddings and node mappings
        self.node2uuid = common_data['node2uuid']
        self.verifier_name = common_data['verifier_name']
        self.embeddings = common_data['embeddings_dict'][self.verifier_name]

        # Initialize graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.node2uuid.keys())

        # Tracking
        self.auto_accepted = []  # Edges automatically accepted
        self.auto_rejected = []  # Edges automatically rejected
        self.human_reviewed = []  # Edges sent for human review
        self.positive_edges = []  # All accepted edges (auto + human)
        self.negative_edges = []  # All rejected edges (auto + human)
        self.num_reviews = 0  # Track human reviews only
        self.finished = False

        # Edges to review
        self.uncertain_edges = []  # Edges needing human review
        self.current_batch_idx = 0

        # Statistics tracking
        self.stats = {
            'num_nodes': len(self.node2uuid),
            'num_auto_accepted': 0,
            'num_auto_rejected': 0,
            'num_human_reviews': 0,
            'num_human_positive': 0,
            'num_human_negative': 0,
            'start_time': time.time(),
            'review_history': []
        }

        logger.info(f"Initialized Thresholded Review Algorithm")
        logger.info(f"  Low threshold: {self.low_threshold} (auto-reject below)")
        logger.info(f"  High threshold: {self.high_threshold} (auto-accept above)")
        logger.info(f"  Top-k: {self.topk}, Max reviews: {self.max_reviews}")
        logger.info(f"  Number of nodes: {len(self.node2uuid)}")

    def _get_and_classify_edges(self):
        """
        Get candidate edges and classify them based on thresholds.

        Returns:
            tuple: (auto_accepted, auto_rejected, uncertain) lists of edges
        """
        logger.info("Getting and classifying candidate edges...")

        # Get edges efficiently
        num_nodes = len(self.node2uuid)
        expected_edges = num_nodes * self.topk

        # Get edges using embeddings' efficient method
        all_edges = self.embeddings.get_edges(
            topk=self.topk,
            target_edges=expected_edges * 2,
            target_proportion=None
        )

        auto_accepted = []
        auto_rejected = []
        uncertain = []

        edge_set = set()  # To avoid duplicates
        node_neighbor_count = defaultdict(int)

        for n1, n2, score in all_edges:
            # Check if we already have enough neighbors for these nodes
            if node_neighbor_count[n1] >= self.topk and node_neighbor_count[n2] >= self.topk:
                continue

            # Ensure consistent edge ordering
            edge = tuple(sorted([n1, n2]))
            if edge in edge_set:
                continue
            edge_set.add(edge)
            node_neighbor_count[n1] += 1
            node_neighbor_count[n2] += 1

            # Classify based on thresholds
            edge_with_score = (n1, n2, score, self.verifier_name)

            if score < self.low_threshold:
                auto_rejected.append(edge_with_score)
            elif score > self.high_threshold:
                auto_accepted.append(edge_with_score)
            else:
                uncertain.append(edge_with_score)

        # Sort uncertain edges by score (highest score first)
        # This ensures we review the most likely matches first
        # Starting with high-confidence edges maintains precision while building recall
        uncertain.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Edge classification complete:")
        logger.info(f"  Auto-accepted (>{self.high_threshold}): {len(auto_accepted)}")
        logger.info(f"  Auto-rejected (<{self.low_threshold}): {len(auto_rejected)}")
        logger.info(f"  Uncertain (needs review): {len(uncertain)}")

        return auto_accepted, auto_rejected, uncertain

    def step(self, edge_responses):
        """
        Process edge responses and prepare next batch for review.

        Args:
            edge_responses: List of (n1, n2, score, source) tuples
                           where score is 1.0 for match, 0.0 for non-match

        Returns:
            list: Next batch of edges to review, or empty list if finished
        """
        # Process human responses if any
        if edge_responses and len(edge_responses) > 0 and edge_responses[0][3] != self.verifier_name:
            self._process_human_responses(edge_responses)

        # Initialize on first call
        if not self.uncertain_edges and not edge_responses:
            auto_accepted, auto_rejected, uncertain = self._get_and_classify_edges()

            # Process auto-accepted edges
            for edge in auto_accepted:
                n1, n2, score = edge[0], edge[1], edge[2]
                self.auto_accepted.append((n1, n2))
                self.positive_edges.append((n1, n2))
                self.graph.add_edge(n1, n2, weight=score)
                self.stats['num_auto_accepted'] += 1

            # Process auto-rejected edges
            for edge in auto_rejected:
                n1, n2 = edge[0], edge[1]
                self.auto_rejected.append((n1, n2))
                self.negative_edges.append((n1, n2))
                self.stats['num_auto_rejected'] += 1

            # Store uncertain edges for human review
            self.uncertain_edges = uncertain

            logger.info(f"Initial classification complete:")
            logger.info(f"  {self.stats['num_auto_accepted']} edges auto-accepted")
            logger.info(f"  {self.stats['num_auto_rejected']} edges auto-rejected")
            logger.info(f"  {len(self.uncertain_edges)} edges need human review")

        # Check if we've reached the maximum reviews limit
        if self.max_reviews and self.num_reviews >= self.max_reviews:
            logger.info(f"Reached maximum review limit of {self.max_reviews}")
            self.finished = True
            return []

        # Check if we're done
        if self.current_batch_idx >= len(self.uncertain_edges):
            self.finished = True
            logger.info("All uncertain edges have been reviewed")
            return []

        # Get next batch of uncertain edges
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.review_batch_size, len(self.uncertain_edges))

        # Apply max_reviews limit
        if self.max_reviews:
            remaining_reviews = self.max_reviews - self.num_reviews
            if remaining_reviews <= 0:
                self.finished = True
                return []
            end_idx = min(end_idx, start_idx + remaining_reviews)

        batch = self.uncertain_edges[start_idx:end_idx]
        self.current_batch_idx = end_idx

        logger.info(f"Requesting human review for {len(batch)} uncertain edges")
        return batch

    def _process_human_responses(self, edge_responses):
        """
        Process human review responses for uncertain edges.

        Args:
            edge_responses: List of (n1, n2, score, source) tuples
        """
        batch_positive = 0
        batch_negative = 0

        for n1, n2, score, source in edge_responses:
            self.num_reviews += 1
            self.stats['num_human_reviews'] += 1
            self.human_reviewed.append((n1, n2, score))

            if score >= 0.5:  # Positive match
                self.positive_edges.append((n1, n2))
                self.graph.add_edge(n1, n2, weight=score)
                batch_positive += 1
                self.stats['num_human_positive'] += 1
            else:  # Negative match
                self.negative_edges.append((n1, n2))
                batch_negative += 1
                self.stats['num_human_negative'] += 1

        # Log batch statistics
        if edge_responses:
            logger.info(f"Processed {len(edge_responses)} human reviews: "
                       f"{batch_positive} positive, {batch_negative} negative")
            logger.info(f"Total human reviews so far: {self.stats['num_human_reviews']} "
                       f"({self.stats['num_human_positive']} positive, "
                       f"{self.stats['num_human_negative']} negative)")

            # Update review history
            self.stats['review_history'].append({
                'batch_size': len(edge_responses),
                'positive': batch_positive,
                'negative': batch_negative,
                'total_reviews': self.stats['num_human_reviews'],
                'timestamp': time.time() - self.stats['start_time']
            })

            # Print detailed batch statistics
            self._print_batch_statistics()

    def _print_evaluation_metrics(self):
        """Calculate and print evaluation metrics against ground truth over ENTIRE dataset.
        Uses the same pairwise precision/recall calculation as GC and HDBSCAN algorithms."""
        if 'gt_node2cid' not in self.common_data:
            return

        # Get ground truth clustering
        gt_node2cid = self.common_data['gt_node2cid']

        # Get current clustering from the graph
        components = list(nx.connected_components(self.graph))
        current_node2cid = {}

        for cid, component in enumerate(components):
            for node in component:
                current_node2cid[node] = str(cid)

        # Add singletons for isolated nodes
        next_cid = len(components)
        for node in self.node2uuid.keys():
            if node not in current_node2cid:
                current_node2cid[node] = str(next_cid)
                next_cid += 1

        # Calculate pairwise metrics (same as precision_recall in cluster_tools.py)
        tp = 0  # True positive: both nodes in same cluster in both pred and gt
        fp = 0  # False positive: nodes in same cluster in pred but different in gt
        fn = 0  # False negative: nodes in different clusters in pred but same in gt

        # Get all nodes
        nodes = list(self.node2uuid.keys())

        # Check all pairs - this matches the cluster_tools approach
        # Calculate TP and FP from predicted clusters
        pred_clusters = defaultdict(list)
        for node, cid in current_node2cid.items():
            pred_clusters[cid].append(node)

        for pred_nodes in pred_clusters.values():
            if len(pred_nodes) > 1:
                for i, ni in enumerate(pred_nodes):
                    for j in range(i + 1, len(pred_nodes)):
                        nj = pred_nodes[j]
                        # Check if they're in the same GT cluster
                        if ni in gt_node2cid and nj in gt_node2cid:
                            if gt_node2cid[ni] == gt_node2cid[nj]:
                                tp += 1
                            else:
                                fp += 1

        # Calculate FN from ground truth clusters
        gt_clusters = defaultdict(list)
        for node, cid in gt_node2cid.items():
            if node in nodes:  # Only consider nodes in our dataset
                gt_clusters[cid].append(node)

        for gt_nodes in gt_clusters.values():
            if len(gt_nodes) > 1:
                for i, ni in enumerate(gt_nodes):
                    for j in range(i + 1, len(gt_nodes)):
                        nj = gt_nodes[j]
                        # Check if they're in different predicted clusters
                        if ni in current_node2cid and nj in current_node2cid:
                            if current_node2cid[ni] != current_node2cid[nj]:
                                fn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate fraction correct as number of exactly matching clusters (same as GC algorithm)
        num_exact_matches = 0
        for pred_cid, pred_nodes in pred_clusters.items():
            # For each predicted cluster, check if it exactly matches a ground truth cluster
            if len(pred_nodes) > 0:
                # Get the GT cluster for the first node
                first_node = pred_nodes[0]
                if first_node in gt_node2cid:
                    gt_cid = gt_node2cid[first_node]
                    # Get all nodes in this GT cluster that are in our dataset
                    gt_nodes_in_data = [n for n in gt_clusters[gt_cid] if n in nodes]
                    # Check if predicted cluster exactly matches this GT cluster
                    if set(pred_nodes) == set(gt_nodes_in_data):
                        num_exact_matches += 1

        # Fraction correct = number of exactly matching clusters / total predicted clusters
        num_pred_clusters = len(pred_clusters)
        fraction_correct = num_exact_matches / num_pred_clusters if num_pred_clusters > 0 else 0

        # Log metrics
        logger.info(f"\nEvaluation Metrics - FULL DATASET (after {self.stats['num_human_reviews']} reviews):")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Fraction Correct: {fraction_correct:.4f} ({num_exact_matches}/{num_pred_clusters} exact cluster matches)")
        logger.info(f"  Pairwise TP={tp}, FP={fp}, FN={fn}")

        # Also show statistics about what we've reviewed
        num_edges_reviewed = (self.stats['num_auto_accepted'] +
                             self.stats['num_auto_rejected'] +
                             self.stats['num_human_reviews'])
        total_possible_pairs = len(nodes) * (len(nodes) - 1) // 2
        review_coverage = 100 * num_edges_reviewed / max(1, total_possible_pairs)
        logger.info(f"  Edges reviewed so far: {num_edges_reviewed:,} ({review_coverage:.2f}% of all possible)")

        # Store in history
        if 'metrics_history' not in self.stats:
            self.stats['metrics_history'] = []

        self.stats['metrics_history'].append({
            'num_reviews': self.stats['num_human_reviews'],
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fraction_correct': fraction_correct,
            'num_exact_matches': num_exact_matches,
            'num_pred_clusters': num_pred_clusters,
            'num_gt_clusters': len(gt_clusters),
            'num_edges_reviewed': num_edges_reviewed
        })

    def _print_batch_statistics(self):
        """Print detailed statistics after each batch of reviews."""
        batch_num = len(self.stats['review_history'])
        elapsed_time = time.time() - self.stats['start_time']

        logger.info("="*60)
        logger.info(f"Batch {batch_num} Statistics:")
        logger.info("-"*60)

        # Calculate and print evaluation metrics if ground truth is available
        if hasattr(self, 'common_data') and 'gt_node2cid' in self.common_data:
            self._print_evaluation_metrics()

        # Current batch info
        if self.stats['review_history']:
            last_batch = self.stats['review_history'][-1]
            logger.info(f"  Batch size: {last_batch['batch_size']}")
            logger.info(f"  Batch positive rate: {100*last_batch['positive']/max(1,last_batch['batch_size']):.1f}%")

        # Overall statistics
        total_evaluated = (self.stats['num_auto_accepted'] +
                          self.stats['num_auto_rejected'] +
                          self.stats['num_human_reviews'])

        logger.info(f"\nCumulative Statistics:")
        logger.info(f"  Total edges evaluated: {total_evaluated}")
        logger.info(f"  - Auto-accepted: {self.stats['num_auto_accepted']} "
                   f"({100*self.stats['num_auto_accepted']/max(1,total_evaluated):.1f}%)")
        logger.info(f"  - Auto-rejected: {self.stats['num_auto_rejected']} "
                   f"({100*self.stats['num_auto_rejected']/max(1,total_evaluated):.1f}%)")
        logger.info(f"  - Human reviewed: {self.stats['num_human_reviews']} "
                   f"({100*self.stats['num_human_reviews']/max(1,total_evaluated):.1f}%)")

        # Graph statistics
        logger.info(f"\nGraph Status:")
        logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  Edges: {self.graph.number_of_edges()}")
        logger.info(f"  Average degree: {2*self.graph.number_of_edges()/max(1,self.graph.number_of_nodes()):.2f}")

        # Connected components info
        num_components = nx.number_connected_components(self.graph)
        component_sizes = [len(c) for c in nx.connected_components(self.graph)]
        if component_sizes:
            component_sizes.sort(reverse=True)
            logger.info(f"  Connected components: {num_components}")
            logger.info(f"  Largest component: {component_sizes[0]} nodes")
            logger.info(f"  Singletons: {sum(1 for s in component_sizes if s == 1)}")

        # Efficiency metrics
        if self.stats['num_human_reviews'] > 0:
            logger.info(f"\nEfficiency Metrics:")
            logger.info(f"  Human positive rate: {100*self.stats['num_human_positive']/self.stats['num_human_reviews']:.1f}%")
            logger.info(f"  Reviews per second: {self.stats['num_human_reviews']/max(1,elapsed_time):.1f}")
            logger.info(f"  Time elapsed: {elapsed_time:.1f} seconds")

        # Remaining work estimate
        if self.current_batch_idx < len(self.uncertain_edges):
            remaining = min(len(self.uncertain_edges) - self.current_batch_idx,
                          self.max_reviews - self.stats['num_human_reviews'] if self.max_reviews else float('inf'))
            logger.info(f"\nRemaining:")
            logger.info(f"  Uncertain edges left: {len(self.uncertain_edges) - self.current_batch_idx}")
            logger.info(f"  Reviews remaining: {remaining}")
            if self.stats['num_human_reviews'] > 0:
                rate = self.stats['num_human_reviews'] / elapsed_time
                est_time = remaining / rate
                logger.info(f"  Estimated time: {est_time:.1f} seconds")

        logger.info("="*60)

    def is_finished(self):
        """Check if algorithm has finished reviewing all edges."""
        return self.finished

    def show_stats(self):
        """Display algorithm statistics."""
        elapsed_time = time.time() - self.stats['start_time']

        logger.info("="*60)
        logger.info("Thresholded Review Algorithm Statistics:")
        logger.info("="*60)
        logger.info(f"Thresholds: {self.low_threshold} (low) - {self.high_threshold} (high)")
        logger.info(f"Auto-accepted edges: {self.stats['num_auto_accepted']}")
        logger.info(f"Auto-rejected edges: {self.stats['num_auto_rejected']}")
        logger.info(f"Human reviews conducted: {self.stats['num_human_reviews']}")
        if self.stats['num_human_reviews'] > 0:
            logger.info(f"  Human positive: {self.stats['num_human_positive']} "
                       f"({100*self.stats['num_human_positive']/self.stats['num_human_reviews']:.1f}%)")
            logger.info(f"  Human negative: {self.stats['num_human_negative']} "
                       f"({100*self.stats['num_human_negative']/self.stats['num_human_reviews']:.1f}%)")

        total_evaluated = (self.stats['num_auto_accepted'] +
                          self.stats['num_auto_rejected'] +
                          self.stats['num_human_reviews'])
        human_review_rate = 100 * self.stats['num_human_reviews'] / max(1, total_evaluated)
        logger.info(f"Human review rate: {human_review_rate:.1f}% of all edges")

        logger.info(f"Graph edges created: {self.graph.number_of_edges()}")
        logger.info(f"Graph nodes: {self.graph.number_of_nodes()}")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        if self.stats['num_human_reviews'] > 0:
            logger.info(f"Reviews per second: {self.stats['num_human_reviews']/max(1, elapsed_time):.2f}")
        logger.info("="*60)

    def get_clustering(self):
        """
        Get the final clustering from the constructed graph.

        Returns:
            tuple: (clustering, node2cid, graph_dict) where:
                - clustering: dict mapping cluster_id to list of node_ids
                - node2cid: dict mapping node_id to cluster_id
                - graph_dict: dict representation of the graph
        """
        logger.info(f"Generating clustering using method: {self.clustering_method}")

        if self.clustering_method == 'connected_components':
            # Use connected components for clustering
            components = list(nx.connected_components(self.graph))

            clustering = {}
            node2cid = {}

            for cid, component in enumerate(components):
                clustering[str(cid)] = list(component)
                for node in component:
                    node2cid[node] = str(cid)

            logger.info(f"Found {len(components)} connected components")

        elif self.clustering_method == 'community':
            # Use community detection (Louvain)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph)

                clustering = defaultdict(list)
                node2cid = {}

                for node, cid in partition.items():
                    clustering[str(cid)].append(node)
                    node2cid[node] = str(cid)

                clustering = dict(clustering)
                logger.info(f"Found {len(clustering)} communities")

            except ImportError:
                logger.warning("python-louvain not installed, falling back to connected components")
                self.clustering_method = 'connected_components'
                return self.get_clustering()

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        # Add singleton clusters for isolated nodes
        for node in self.node2uuid.keys():
            if node not in node2cid:
                cid = str(len(clustering))
                clustering[cid] = [node]
                node2cid[node] = cid

        # Create graph dict representation
        graph_dict = {
            'nodes': list(self.graph.nodes()),
            'edges': [(int(u), int(v), float(data.get('weight', 1.0)))
                     for u, v, data in self.graph.edges(data=True)],
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges()
        }

        return clustering, node2cid, graph_dict

    def save_results(self, output_path):
        """
        Save algorithm results to files.

        Args:
            output_path: Directory to save results
        """
        os.makedirs(output_path, exist_ok=True)

        # Get clustering
        clustering, node2cid, graph_dict = self.get_clustering()

        # Save clustering
        write_json(clustering, os.path.join(output_path, 'clustering.json'))
        write_json(node2cid, os.path.join(output_path, 'node2cid.json'))
        write_json(self.node2uuid, os.path.join(output_path, 'node2uuid_file.json'))

        # Save graph
        write_json(graph_dict, os.path.join(output_path, 'graph.json'))

        # Save statistics
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        self.stats['num_clusters'] = len(clustering)
        self.stats['num_edges'] = len(self.positive_edges)

        # Calculate efficiency metrics
        total_evaluated = (self.stats['num_auto_accepted'] +
                          self.stats['num_auto_rejected'] +
                          self.stats['num_human_reviews'])
        self.stats['human_review_rate'] = (self.stats['num_human_reviews'] /
                                           max(1, total_evaluated))
        self.stats['auto_decision_rate'] = 1.0 - self.stats['human_review_rate']

        # Calculate cluster size distribution
        cluster_sizes = [len(nodes) for nodes in clustering.values()]
        self.stats['cluster_size_distribution'] = {
            'min': min(cluster_sizes) if cluster_sizes else 0,
            'max': max(cluster_sizes) if cluster_sizes else 0,
            'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median': np.median(cluster_sizes) if cluster_sizes else 0,
            'num_singletons': sum(1 for s in cluster_sizes if s == 1)
        }

        write_json(self.stats, os.path.join(output_path, 'thresholded_review_stats.json'))

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Final statistics: {self.stats['num_human_reviews']} human reviews, "
                   f"{self.stats['num_auto_accepted']} auto-accepted, "
                   f"{self.stats['num_auto_rejected']} auto-rejected, "
                   f"{len(clustering)} clusters")

        return clustering, node2cid, graph_dict