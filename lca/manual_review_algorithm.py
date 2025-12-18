"""
Manual Review Graph Algorithm.

This algorithm builds a graph from human reviews of top-K neighbors for each annotation.
For each annotation, it reviews top-K or within-τ neighbors, and adds edges when
the human says "match". The output is a positive-edge graph that can be clustered
using connected components or other graph clustering methods.
"""

import logging
import numpy as np
import time
from collections import defaultdict
import networkx as nx
from tools import write_json
import cluster_tools as ct
import os

logger = logging.getLogger('lca')


class ManualReviewAlgorithm:
    """
    Fully-manual graph construction from pairwise reviews.

    For each annotation, review top-K / within-τ neighbors;
    add an edge when the human says "match."
    Output is a positive-edge graph.
    """

    def __init__(self, config, common_data):
        """
        Initialize the manual review algorithm.

        Args:
            config: Algorithm configuration including:
                - topk: Number of top neighbors to review per node (default: 10)
                - threshold: Distance threshold for neighbors (optional)
                - clustering_method: 'connected_components' or 'community' (default: 'connected_components')
                - review_batch_size: Number of edges to review in each batch (default: 100)
            common_data: Common data from algorithm preparation
        """
        self.config = config
        self.common_data = common_data

        # Algorithm parameters
        self.topk = config.get('topk', 10)
        self.threshold = config.get('threshold', None)
        self.clustering_method = config.get('clustering_method', 'connected_components')
        self.review_batch_size = config.get('review_batch_size', 100)
        self.review_all_at_once = config.get('review_all_at_once', False)
        self.max_reviews = config.get('max_reviews', None)  # Maximum total reviews to conduct

        # Get embeddings and node mappings
        self.node2uuid = common_data['node2uuid']
        self.verifier_name = common_data['verifier_name']
        self.embeddings = common_data['embeddings_dict'][self.verifier_name]

        # Initialize graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.node2uuid.keys())

        # Tracking
        self.reviewed_pairs = set()  # Track which pairs have been reviewed
        self.positive_edges = []  # Store positive edges
        self.negative_edges = []  # Store negative edges
        self.num_reviews = 0  # Track total number of human reviews
        self.finished = False

        # Edges to review
        self.edges_to_review = []
        self.current_batch_idx = 0

        # Statistics tracking
        self.stats = {
            'num_nodes': len(self.node2uuid),
            'num_reviews': 0,
            'num_positive': 0,
            'num_negative': 0,
            'start_time': time.time(),
            'review_history': []
        }

        logger.info(f"Initialized Manual Review Algorithm with topk={self.topk}, threshold={self.threshold}")
        logger.info(f"Clustering method: {self.clustering_method}")
        logger.info(f"Number of nodes: {len(self.node2uuid)}")

    def _get_candidate_edges(self):
        """
        Get all candidate edges for review based on top-K neighbors and/or threshold.

        For each node, we want only its top-K neighbors, not all possible edges.

        Returns:
            list: List of (node1, node2, score, verifier_name) tuples
        """
        logger.info("Getting candidate edges for review...")

        # Use a reasonable target number of edges
        # For N nodes with top-k neighbors each, we expect roughly N * k / 2 unique edges
        # (divided by 2 because edges are bidirectional)
        num_nodes = len(self.node2uuid)
        expected_edges = num_nodes * self.topk

        # Get edges using the embeddings' efficient method
        # Use a higher target to ensure we get enough edges after filtering
        all_edges = self.embeddings.get_edges(
            topk=self.topk,
            target_edges=0,#expected_edges * 2,  # Get more than we need
            target_proportion=None
        )

        # Convert to list and add verifier name
        candidate_edges = []
        edge_set = set()  # To avoid duplicates
        node_neighbor_count = defaultdict(int)  # Track neighbors per node

        for n1, n2, score in all_edges:
            # Apply threshold if specified
            if self.threshold is not None and score < self.threshold:
                continue

            # Check if we already have enough neighbors for these nodes
            if node_neighbor_count[n1] >= self.topk and node_neighbor_count[n2] >= self.topk:
                continue

            # Ensure consistent edge ordering
            edge = tuple(sorted([n1, n2]))
            if edge not in edge_set:
                edge_set.add(edge)
                candidate_edges.append((n1, n2, score, self.verifier_name))
                node_neighbor_count[n1] += 1
                node_neighbor_count[n2] += 1

        # Sort by score (highest first) for better review order
        candidate_edges.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Found {len(candidate_edges)} candidate edges for review")
        logger.info(f"Nodes with edges: {len(node_neighbor_count)}")
        avg_neighbors = sum(node_neighbor_count.values()) / max(1, len(node_neighbor_count))
        logger.info(f"Average neighbors per node: {avg_neighbors:.1f}")
        return candidate_edges

    def step(self, edge_responses):
        """
        Process edge responses and prepare next batch for review.

        Args:
            edge_responses: List of (n1, n2, score, source) tuples
                           where score is 1.0 for match, 0.0 for non-match

        Returns:
            list: Next batch of edges to review, or empty list if finished
        """
        # For manual review algorithm, we ignore initial edges from run.py
        # and only process actual human responses
        if edge_responses and edge_responses[0][3] != self.verifier_name:
            # These are human responses, process them
            self._process_responses(edge_responses)
        elif not self.edges_to_review:
            # First call - initialize our own candidate edges
            self.edges_to_review = self._get_candidate_edges()
            logger.info(f"Total candidate edges for review: {len(self.edges_to_review)}")

        # Check if we've reached the maximum reviews limit
        if self.max_reviews and self.num_reviews >= self.max_reviews:
            logger.info(f"Reached maximum review limit of {self.max_reviews}")
            self.finished = True
            return []

        # Check if we're done
        if self.review_all_at_once:
            # Review all edges at once
            if self.current_batch_idx > 0:
                self.finished = True
                return []
            else:
                # Apply max_reviews limit even for all-at-once mode
                edges_to_return = self.edges_to_review
                if self.max_reviews:
                    edges_to_return = edges_to_return[:self.max_reviews]
                self.current_batch_idx = len(edges_to_return)
                return edges_to_return
        else:
            # Review in batches
            if self.current_batch_idx >= len(self.edges_to_review):
                self.finished = True
                return []

        # Get next batch
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.review_batch_size, len(self.edges_to_review))

        # Apply max_reviews limit
        if self.max_reviews:
            remaining_reviews = self.max_reviews - self.num_reviews
            if remaining_reviews <= 0:
                self.finished = True
                return []
            end_idx = min(end_idx, start_idx + remaining_reviews)

        batch = self.edges_to_review[start_idx:end_idx]

        self.current_batch_idx = end_idx

        # Filter out already reviewed pairs
        filtered_batch = []
        for edge in batch:
            n1, n2 = edge[0], edge[1]
            pair = tuple(sorted([n1, n2]))
            if pair not in self.reviewed_pairs:
                filtered_batch.append(edge)

        logger.info(f"Requesting review of {len(filtered_batch)} edges (batch {start_idx//self.review_batch_size + 1})")
        return filtered_batch

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

        # Compute Hungarian matching metrics (cluster-level)
        hungarian = ct.hungarian_cluster_matching(pred_clusters, gt_clusters)

        # Log metrics
        logger.info(f"\nEvaluation Metrics - FULL DATASET (after {self.stats['num_reviews']} reviews):")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Fraction Correct: {fraction_correct:.4f} ({num_exact_matches}/{num_pred_clusters} exact cluster matches)")
        logger.info(f"  Pairwise TP={tp}, FP={fp}, FN={fn}")
        logger.info(f"  Hungarian F1 Score: {hungarian['f1']:.4f}")
        logger.info(f"  Hungarian Precision: {hungarian['precision']:.4f}")
        logger.info(f"  Hungarian Recall: {hungarian['recall']:.4f}")
        logger.info(f"  Hungarian TP={hungarian['tp']}, FP={hungarian['fp']}, FN={hungarian['fn']}")

        # Store in history
        if 'metrics_history' not in self.stats:
            self.stats['metrics_history'] = []

        self.stats['metrics_history'].append({
            'num_reviews': self.stats['num_reviews'],
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fraction_correct': fraction_correct,
            'num_exact_matches': num_exact_matches,
            'num_pred_clusters': num_pred_clusters,
            'num_gt_clusters': len(gt_clusters),
            'hungarian_f1': hungarian['f1'],
            'hungarian_precision': hungarian['precision'],
            'hungarian_recall': hungarian['recall']
        })

    def _process_responses(self, edge_responses):
        """
        Process human review responses.

        Args:
            edge_responses: List of (n1, n2, score, source) tuples
        """
        batch_positive = 0
        batch_negative = 0

        for n1, n2, score, source in edge_responses:
            # Track reviewed pair
            pair = tuple(sorted([n1, n2]))
            self.reviewed_pairs.add(pair)

            # Update statistics
            self.num_reviews += 1
            self.stats['num_reviews'] += 1

            if score >= 0.5:  # Positive match
                self.positive_edges.append((n1, n2))
                self.graph.add_edge(n1, n2, weight=score)
                batch_positive += 1
                self.stats['num_positive'] += 1
            else:  # Negative match
                self.negative_edges.append((n1, n2))
                batch_negative += 1
                self.stats['num_negative'] += 1

        # Log batch statistics
        if edge_responses:
            logger.info(f"Processed {len(edge_responses)} reviews: "
                       f"{batch_positive} positive, {batch_negative} negative")
            logger.info(f"Total reviews so far: {self.stats['num_reviews']} "
                       f"({self.stats['num_positive']} positive, {self.stats['num_negative']} negative)")

            # Update review history
            self.stats['review_history'].append({
                'batch_size': len(edge_responses),
                'positive': batch_positive,
                'negative': batch_negative,
                'total_reviews': self.stats['num_reviews'],
                'timestamp': time.time() - self.stats['start_time']
            })

            # Calculate and print evaluation metrics over full dataset
            self._print_evaluation_metrics()

    def is_finished(self):
        """Check if algorithm has finished reviewing all edges."""
        return self.finished

    def show_stats(self):
        """Display algorithm statistics."""
        elapsed_time = time.time() - self.stats['start_time']

        logger.info("="*60)
        logger.info("Manual Review Algorithm Statistics:")
        logger.info("="*60)
        logger.info(f"Total reviews conducted: {self.stats['num_reviews']}")
        logger.info(f"Positive matches: {self.stats['num_positive']} "
                   f"({100*self.stats['num_positive']/max(1, self.stats['num_reviews']):.1f}%)")
        logger.info(f"Negative matches: {self.stats['num_negative']} "
                   f"({100*self.stats['num_negative']/max(1, self.stats['num_reviews']):.1f}%)")
        logger.info(f"Graph edges created: {self.graph.number_of_edges()}")
        logger.info(f"Graph nodes: {self.graph.number_of_nodes()}")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Reviews per second: {self.stats['num_reviews']/max(1, elapsed_time):.2f}")
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
                # Recursive call with connected_components
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
        self.stats['precision'] = (self.stats['num_positive'] / self.stats['num_reviews']
                                   if self.stats['num_reviews'] > 0 else 0)

        # Calculate cluster size distribution
        cluster_sizes = [len(nodes) for nodes in clustering.values()]
        self.stats['cluster_size_distribution'] = {
            'min': min(cluster_sizes) if cluster_sizes else 0,
            'max': max(cluster_sizes) if cluster_sizes else 0,
            'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median': np.median(cluster_sizes) if cluster_sizes else 0,
            'num_singletons': sum(1 for s in cluster_sizes if s == 1)
        }

        write_json(self.stats, os.path.join(output_path, 'manual_review_stats.json'))

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Final statistics: {self.stats['num_reviews']} reviews, "
                   f"{self.stats['num_positive']} positive edges, "
                   f"{len(clustering)} clusters")

        return clustering, node2cid, graph_dict