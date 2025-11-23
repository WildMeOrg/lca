import numpy as np
import hdbscan
from scipy.spatial.distance import cosine, euclidean
import logging

logger = logging.getLogger('lca')


class hdbscan_verifier(object):
    def __init__(self, embeddings, uuids, node2uuid, min_cluster_size=2, metric='euclidean'):
        self.embeddings = np.array(embeddings)
        self.uuids = uuids
        self.node2uuid = node2uuid
        self.uuid2node = {val: key for (key, val) in node2uuid.items()}
        self.min_cluster_size = min_cluster_size
        self.metric = metric

        # Train HDBSCAN clusterer with prediction_data=True to enable soft clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            prediction_data=True,  # Enable soft clustering and membership probabilities
            gen_min_span_tree=True  # Generate minimum spanning tree for mutual reachability
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)

        # Create UUID to cluster mapping
        self.uuid_to_cluster = {}
        for node_id, label in enumerate(self.labels):
            uuid = self.node2uuid.get(node_id)
            if uuid:
                self.uuid_to_cluster[uuid] = label

        logger.info(f"HDBSCAN trained: {len(set(self.labels)) - (1 if -1 in self.labels else 0)} clusters found")

    def get_embedding(self, uuid):
        """Get embedding for a UUID."""
        if uuid not in self.uuid2node:
            return None
        node_id = self.uuid2node[uuid]
        return self.embeddings[node_id]

    def get_mutual_reachability_distance(self, uuid1, uuid2):
        """Get mutual reachability distance between two points from HDBSCAN's tree."""
        if uuid1 not in self.uuid2node or uuid2 not in self.uuid2node:
            return float('inf')

        node1 = self.uuid2node[uuid1]
        node2 = self.uuid2node[uuid2]

        # Get the mutual reachability distance from HDBSCAN's internal structure
        if hasattr(self.clusterer, 'mutual_reachability_'):
            try:
                dist = self.clusterer.mutual_reachability_[node1, node2]
                return float(dist)
            except:
                pass

        return float('inf')

    def get_membership_probability(self, uuid1, uuid2):
        """
        Calculate the probability that two points belong to the same cluster.
        Uses HDBSCAN's soft clustering probabilities.
        """
        if uuid1 not in self.uuid2node or uuid2 not in self.uuid2node:
            return 0.0

        node1 = self.uuid2node[uuid1]
        node2 = self.uuid2node[uuid2]

        # If both points are in the same cluster (not noise), calculate shared membership probability
        cluster1 = self.labels[node1]
        cluster2 = self.labels[node2]

        if cluster1 == cluster2 and cluster1 != -1:
            # Both in same cluster - high confidence
            if hasattr(self.clusterer, 'probabilities_'):
                # Use the minimum of their cluster membership probabilities
                prob1 = self.clusterer.probabilities_[node1]
                prob2 = self.clusterer.probabilities_[node2]
                return float(min(prob1, prob2))
            else:
                return 1.0  # Hard clustering - they're in the same cluster

        # Different clusters or noise - calculate soft membership overlap
        if hasattr(self.clusterer, 'prediction_data_'):
            try:
                # Get soft cluster memberships for both points
                all_clusters = list(set(self.labels[self.labels != -1]))

                # Calculate membership vector for each point
                membership1 = hdbscan.membership_vector(self.clusterer, self.embeddings[node1:node1+1])[0]
                membership2 = hdbscan.membership_vector(self.clusterer, self.embeddings[node2:node2+1])[0]

                # Calculate overlap (dot product of membership vectors)
                overlap = np.dot(membership1, membership2)
                return float(overlap)
            except:
                pass

        return 0.0  # No shared cluster membership

    def get_cluster_confidence_score(self, uuid1, uuid2):
        """
        Get confidence score that two points should be in same/different clusters.
        Returns value between 0 (definitely different) and 1 (definitely same).
        """
        membership_prob = self.get_membership_probability(uuid1, uuid2)

        # Get their actual cluster assignments
        cluster1 = self.uuid_to_cluster.get(uuid1, -1)
        cluster2 = self.uuid_to_cluster.get(uuid2, -1)

        if cluster1 == cluster2 and cluster1 != -1:
            # Same cluster - return membership probability
            return membership_prob
        else:
            # Different clusters - return inverse of membership probability
            # This represents confidence they should be in different clusters
            return 1.0 - membership_prob

    def convert_query(self, n0, n1):
        """Convert node IDs to UUIDs and get cluster info."""
        uuid1 = self.node2uuid.get(n0)
        uuid2 = self.node2uuid.get(n1)

        if uuid1 is None or uuid2 is None:
            return None

        cluster1 = self.uuid_to_cluster.get(uuid1, -1)
        cluster2 = self.uuid_to_cluster.get(uuid2, -1)

        return (uuid1, cluster1, uuid2, cluster2)

    def __call__(self, query):
        """
        Check if node pairs would be clustered together by HDBSCAN.
        Returns list of (uuid1, uuid2) tuples that would be in same cluster.
        """
        nodes_in_same_cluster = []

        for (n0, n1) in query:
            result = self.convert_query(n0, n1)
            if result is None:
                continue

            uuid1, cluster1, uuid2, cluster2 = result

            # Check if in same cluster (and not noise points)
            if cluster1 == cluster2 and cluster1 != -1:
                nodes_in_same_cluster.append((uuid1, uuid2))

        return nodes_in_same_cluster

    def get_score(self, n0, n1):
        """Get HDBSCAN membership probability score between two nodes (Embeddings interface)."""
        uuid1 = self.node2uuid.get(n0)
        uuid2 = self.node2uuid.get(n1)

        if uuid1 is None or uuid2 is None:
            return 0.0

        return self.get_membership_probability(uuid1, uuid2)

    def verify_pair(self, uuid1, uuid2):
        """
        Verify if two UUIDs would cluster together and return cluster membership info.
        Returns: (same_cluster, cluster1, cluster2, membership_probability, confidence_score)
        """
        # Get cluster assignments
        cluster1 = self.uuid_to_cluster.get(uuid1, -1)
        cluster2 = self.uuid_to_cluster.get(uuid2, -1)
        same_cluster = (cluster1 == cluster2) and (cluster1 != -1)

        # Calculate probability-based scores
        membership_prob = self.get_membership_probability(uuid1, uuid2)
        confidence_score = self.get_cluster_confidence_score(uuid1, uuid2)

        return same_cluster, cluster1, cluster2, membership_prob, confidence_score