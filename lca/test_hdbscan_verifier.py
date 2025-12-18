#!/usr/bin/env python3
"""
Test script for HDBSCAN Verifier
Demonstrates how to check if two nodes would cluster together
"""

import json
import numpy as np
from hdbscan_verifier import HDBSCANVerifier
from tools import load_pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


def test_with_sample_data():
    """Test with synthetic data."""
    print("\n=== Testing with Synthetic Data ===")

    # Create synthetic embeddings
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 128

    # Create 3 clear clusters
    cluster1 = np.random.randn(30, embedding_dim) + np.array([5] * embedding_dim)
    cluster2 = np.random.randn(30, embedding_dim) + np.array([-5] * embedding_dim)
    cluster3 = np.random.randn(30, embedding_dim) + np.array([0] * embedding_dim)
    noise = np.random.randn(10, embedding_dim) * 5  # Spread out noise points

    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])
    uuids = [f"uuid_{i:04d}" for i in range(n_samples)]

    # Create verifier
    verifier = HDBSCANVerifier(
        embeddings=embeddings,
        uuids=uuids,
        min_cluster_size=5,
        metric='euclidean'
    )

    # Test pairs from same cluster
    print("\nTesting pairs from same cluster (should cluster together):")
    result = verifier.verify_pair("uuid_0000", "uuid_0010")  # Both from cluster1
    print(f"uuid_0000 vs uuid_0010: same_cluster={result['same_cluster']}, "
          f"distance={result['distance']:.3f}, similarity={result['similarity']:.3f}")

    # Test pairs from different clusters
    print("\nTesting pairs from different clusters (should NOT cluster together):")
    result = verifier.verify_pair("uuid_0000", "uuid_0040")  # cluster1 vs cluster2
    print(f"uuid_0000 vs uuid_0040: same_cluster={result['same_cluster']}, "
          f"distance={result['distance']:.3f}, similarity={result['similarity']:.3f}")

    # Test noise points
    print("\nTesting noise points:")
    result = verifier.verify_pair("uuid_0090", "uuid_0095")  # Both noise
    print(f"uuid_0090 vs uuid_0095: same_cluster={result['same_cluster']}, "
          f"clusters=({result['cluster1']}, {result['cluster2']})")

    # Get clustering statistics
    stats = verifier.get_cluster_statistics()
    print(f"\nClustering Statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Number of clusters: {stats['n_clusters']}")
    print(f"  Noise points: {stats['n_noise']} ({stats['noise_ratio']:.1%})")
    print(f"  Average cluster size: {stats['avg_cluster_size']:.1f}")


def test_batch_verification():
    """Test batch verification of multiple pairs."""
    print("\n=== Testing Batch Verification ===")

    # Create sample data
    np.random.seed(42)
    embeddings = np.random.randn(50, 64)
    uuids = [f"sample_{i:03d}" for i in range(50)]

    verifier = HDBSCANVerifier(
        embeddings=embeddings,
        uuids=uuids,
        min_cluster_size=3,
        metric='cosine'
    )

    # Create test pairs
    test_pairs = [
        ("sample_000", "sample_001"),
        ("sample_000", "sample_010"),
        ("sample_010", "sample_020"),
        ("sample_030", "sample_040"),
    ]

    results = verifier.batch_verify(test_pairs)

    print("\nBatch Verification Results:")
    for result in results:
        print(f"{result['uuid1']} vs {result['uuid2']}:")
        print(f"  Same cluster: {result['same_cluster']}")
        print(f"  LCA Score: {result['lca_score']:.4f}")
        print(f"  Similarity: {result['similarity']:.4f}")
        print()


def test_with_config_file():
    """Test loading from a config and embeddings file."""
    print("\n=== Testing with Config File ===")

    # Create a sample config
    sample_config = {
        "test_pairs": [
            ["uuid1", "uuid2"],
            ["uuid3", "uuid4"]
        ],
        "hdbscan_params": {
            "min_cluster_size": 2,
            "metric": "euclidean"
        }
    }

    # Save sample config
    with open("/tmp/test_pairs.json", "w") as f:
        json.dump(sample_config, f)

    print("Sample config saved to /tmp/test_pairs.json")
    print("To use with real data:")
    print("python hdbscan_verifier.py --embeddings path/to/embeddings.pickle "
          "--pairs_file /tmp/test_pairs.json --output results.json")


def main():
    """Run all tests."""
    print("=" * 60)
    print("HDBSCAN Verifier Test Suite")
    print("=" * 60)

    # Run tests
    test_with_sample_data()
    test_batch_verification()
    test_with_config_file()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()