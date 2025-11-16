#!/usr/bin/env python3
"""
Run clustering algorithm and save results in the correct format.
This script combines running the clustering and formatting the output.
Supports any clustering algorithm (HDBSCAN, GC, LCA, etc.).
"""

import argparse
import os
import sys
import subprocess
import json
import datetime
from cluster_tools import percent_and_PR, build_node_to_cluster_mapping


def load_config(config_path):
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_ground_truth_clustering(anno_file, uuid_key='uuid', name_keys=None):
    """Load ground truth clustering from annotation file using name_keys as ground truth."""
    with open(anno_file, 'r') as f:
        data = json.load(f)

    # Use name_keys for ground truth if provided, otherwise try cluster_id
    gt_field = 'cluster_id'
    if name_keys and len(name_keys) > 0:
        gt_field = name_keys[0]  # Use first name_key as ground truth
        print(f"Using '{gt_field}' from name_keys as ground truth field")

    # Build ground truth clustering from specified field
    gt_clustering = {}
    gt_node2uuid = {}
    node_id = 0

    for ann in data.get('annotations', []):
        cluster_id = ann.get(gt_field)
        uuid = ann.get(uuid_key)

        if cluster_id is not None and uuid is not None:
            if cluster_id not in gt_clustering:
                gt_clustering[cluster_id] = []
            gt_clustering[cluster_id].append(node_id)
            gt_node2uuid[str(node_id)] = uuid
            node_id += 1

    # Convert lists to the expected format
    gt_clustering = {str(cid): nodes for cid, nodes in gt_clustering.items()}

    return gt_clustering, gt_node2uuid


def calculate_evaluation_metrics(est_clustering, est_node2uuid, gt_clustering, gt_node2uuid):
    """Calculate evaluation metrics between estimated and ground truth clusterings."""

    # Create uuid to node mappings for alignment
    uuid2est_node = {uuid: node for node, uuid in est_node2uuid.items()}
    uuid2gt_node = {uuid: node for node, uuid in gt_node2uuid.items()}

    # Find common UUIDs
    common_uuids = set(uuid2est_node.keys()) & set(uuid2gt_node.keys())

    if not common_uuids:
        return {
            'error': 'No common UUIDs between estimated and ground truth',
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'frac_correct': 0.0
        }

    # Align clusterings based on common UUIDs
    aligned_est_clustering = {}
    aligned_gt_clustering = {}
    node_mapping = {}  # Maps aligned node IDs to UUIDs

    aligned_node_id = 0
    for uuid in common_uuids:
        est_node = uuid2est_node[uuid]
        gt_node = uuid2gt_node[uuid]

        # Find clusters for this UUID
        est_cid = None
        gt_cid = None

        for cid, nodes in est_clustering.items():
            if int(est_node) in nodes or str(est_node) in [str(n) for n in nodes]:
                est_cid = cid
                break

        for cid, nodes in gt_clustering.items():
            if int(gt_node) in nodes or str(gt_node) in [str(n) for n in nodes]:
                gt_cid = cid
                break

        if est_cid and gt_cid:
            if est_cid not in aligned_est_clustering:
                aligned_est_clustering[est_cid] = []
            if gt_cid not in aligned_gt_clustering:
                aligned_gt_clustering[gt_cid] = []

            aligned_est_clustering[est_cid].append(aligned_node_id)
            aligned_gt_clustering[gt_cid].append(aligned_node_id)
            node_mapping[aligned_node_id] = uuid
            aligned_node_id += 1

    # Convert lists to sets for cluster_tools compatibility
    aligned_est_clustering = {cid: set(nodes) for cid, nodes in aligned_est_clustering.items()}
    aligned_gt_clustering = {cid: set(nodes) for cid, nodes in aligned_gt_clustering.items()}

    # Build node2cluster mappings
    est_n2c = build_node_to_cluster_mapping(aligned_est_clustering)
    gt_n2c = build_node_to_cluster_mapping(aligned_gt_clustering)

    # Calculate metrics using cluster_tools functions
    try:
        frac_correct, precision, recall, per_size, non_equal, f1 = percent_and_PR(
            aligned_est_clustering, est_n2c, aligned_gt_clustering, gt_n2c
        )

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'frac_correct': frac_correct,
            'num_estimated_clusters': len(est_clustering),
            'num_ground_truth_clusters': len(gt_clustering),
            'num_common_uuids': len(common_uuids),
            'per_size_metrics': per_size
        }
    except Exception as e:
        return {
            'error': str(e),
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'frac_correct': 0.0
        }


def append_metrics_to_config_log(log_file, metrics_summary):
    """Append evaluation metrics to the config-specified log file."""
    if log_file and os.path.exists(log_file):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"CLUSTERING EVALUATION METRICS - {timestamp}\n")
            f.write("="*60 + "\n")
            f.write(metrics_summary)
            f.write("="*60 + "\n")


def evaluate_field_separated_results(output_base, anno_file, config_log_file,
                                    separate_by_fields, uuid_key, name_keys=None):
    """Evaluate metrics for field-separated clustering results and append to config log."""
    import glob
    import re

    # Build regex pattern to match and extract field values
    regex_pattern = "_".join([f"{field}-([^_]+)" for field in separate_by_fields])
    regex = re.compile(regex_pattern)

    # Find all directories matching the pattern
    glob_pattern = "_".join([f"{field}-*" for field in separate_by_fields])
    pattern_path = os.path.join(output_base, glob_pattern)

    print(f"Evaluating field-separated clustering results...")

    all_metrics = []
    metrics_text = []

    for field_dir in glob.glob(pattern_path):
        if os.path.isdir(field_dir):
            dir_name = os.path.basename(field_dir)
            match = regex.match(dir_name)

            if match:
                # Extract field values from regex groups
                field_combo = dict(zip(separate_by_fields, match.groups()))
                field_str = ', '.join(f"{k}={v}" for k, v in field_combo.items())

                # Load clustering results for this field combination
                clustering_file = os.path.join(field_dir, "clustering.json")
                node2uuid_file = os.path.join(field_dir, "node2uuid_file.json")

                if not os.path.exists(clustering_file) or not os.path.exists(node2uuid_file):
                    continue

                try:
                    with open(clustering_file, 'r') as f:
                        est_clustering = json.load(f)
                    with open(node2uuid_file, 'r') as f:
                        est_node2uuid = json.load(f)

                    # Load ground truth
                    gt_clustering, gt_node2uuid = load_ground_truth_clustering(anno_file, uuid_key, name_keys)

                    # Calculate metrics
                    metrics = calculate_evaluation_metrics(est_clustering, est_node2uuid,
                                                         gt_clustering, gt_node2uuid)

                    # Add field info to metrics
                    metrics['field_combination'] = field_combo
                    all_metrics.append(metrics)

                    # Format metrics for this field
                    field_text = f"\n{field_str}:\n"
                    field_text += f"  F1 Score:         {metrics.get('f1', 0):.4f}\n"
                    field_text += f"  Precision:        {metrics.get('precision', 0):.4f}\n"
                    field_text += f"  Recall:           {metrics.get('recall', 0):.4f}\n"
                    field_text += f"  Fraction Correct: {metrics.get('frac_correct', 0):.4f}\n"
                    metrics_text.append(field_text)

                except Exception as e:
                    print(f"  Error evaluating {field_str}: {e}")

    # Calculate and format average metrics
    if all_metrics:
        avg_metrics = {
            'f1': sum(m.get('f1', 0) for m in all_metrics) / len(all_metrics),
            'precision': sum(m.get('precision', 0) for m in all_metrics) / len(all_metrics),
            'recall': sum(m.get('recall', 0) for m in all_metrics) / len(all_metrics),
            'frac_correct': sum(m.get('frac_correct', 0) for m in all_metrics) / len(all_metrics),
        }

        summary_text = "".join(metrics_text)
        summary_text += "\n" + "-"*40 + "\n"
        summary_text += "AVERAGE METRICS:\n"
        summary_text += f"  F1 Score:         {avg_metrics['f1']:.4f}\n"
        summary_text += f"  Precision:        {avg_metrics['precision']:.4f}\n"
        summary_text += f"  Recall:           {avg_metrics['recall']:.4f}\n"
        summary_text += f"  Fraction Correct: {avg_metrics['frac_correct']:.4f}\n"

        # Append to config log file
        append_metrics_to_config_log(config_log_file, summary_text)

        # Print summary
        print(summary_text)

        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run clustering algorithm and save results in the correct format"
    )

    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--output_prefix", type=str, default="clustering",
                       help="Output filename prefix (default: clustering)")
    parser.add_argument("--output_suffix", type=str, default="results",
                       help="Output filename suffix (default: results)")
    parser.add_argument("--save_dir", type=str,
                       help="Directory to save formatted results (default: same as clustering output)")
    parser.add_argument("--interactive", "-i", action='store_true',
                       help="Enable interactive mode")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get algorithm type
    algorithm_type = config.get('algorithm_type', 'unknown')
    print(f"Algorithm type: {algorithm_type}")

    # Use algorithm-specific prefix if not specified
    if args.output_prefix == "clustering" and algorithm_type != 'unknown':
        args.output_prefix = algorithm_type

    # Get paths from config
    data_config = config.get('data', {})
    anno_file = data_config.get('annotation_file')
    output_base = data_config.get('output_path', 'tmp')
    uuid_key = data_config.get('id_key', 'uuid')
    separate_by_fields = data_config.get('separate_by_fields')
    name_keys = data_config.get('name_keys', [])

    # Get log file from config
    logging_config = config.get('logging', {})
    config_log_file = logging_config.get('log_file')

    if not anno_file:
        print("Error: annotation_file not specified in config")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        print(f"Config file: {os.path.abspath(args.config)}")
        print(f"Annotation file: {anno_file}")
        print(f"Output path: {output_base}")
        if separate_by_fields:
            print(f"Separating by fields: {separate_by_fields}")
        input("Press Enter to continue...")

    # Step 1: Run clustering algorithm
    print("\n" + "="*60)
    print(f"Step 1: Running {algorithm_type.upper()} clustering...")
    print("="*60)

    cmd = ["python3", "run.py", "--config", args.config]
    if args.interactive:
        cmd.append("-i")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {algorithm_type.upper()} clustering failed")
        sys.exit(1)

    print(f"\n{algorithm_type.upper()} clustering completed successfully!")

    # Step 2: Save results in correct format
    print("\n" + "="*60)
    print("Step 2: Formatting and saving results...")
    print("="*60)

    save_dir = args.save_dir or output_base

    if separate_by_fields:
        # Handle field-separated results
        cmd = [
            "python3", "save_clustering_results.py",
            anno_file,
            output_base,
            save_dir,
            "--prefix", args.output_prefix,
            "--suffix", args.output_suffix,
            "--uuid_key", uuid_key,
            "--separate_by_fields"
        ] + separate_by_fields

        print(f"Processing field-separated results...")
    else:
        # Handle single result
        cmd = [
            "python3", "save_clustering_results.py",
            anno_file,
            output_base,
            save_dir,
            "--prefix", args.output_prefix,
            "--suffix", args.output_suffix,
            "--uuid_key", uuid_key
        ]

        print(f"Processing single clustering result...")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("Error: Result formatting failed")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"{algorithm_type.upper()} clustering and formatting completed successfully!")
    print(f"Results saved to: {save_dir}")
    print("="*60)

    # Step 3: Calculate evaluation metrics (always run if ground truth available)
    if config_log_file and name_keys:
        print("\n" + "="*60)
        print("Step 3: Calculating evaluation metrics...")
        print("="*60)

        # Handle field-separated vs single clustering results
        if separate_by_fields:
            # Evaluate all field combinations
            evaluate_field_separated_results(
                output_base, anno_file, config_log_file,
                separate_by_fields, uuid_key, name_keys
            )
        else:
            try:
                # Load the clustering results from output_base
                clustering_file = os.path.join(output_base, "clustering.json")
                node2uuid_file = os.path.join(output_base, "node2uuid_file.json")

                if os.path.exists(clustering_file) and os.path.exists(node2uuid_file):
                    with open(clustering_file, 'r') as f:
                        est_clustering = json.load(f)
                    with open(node2uuid_file, 'r') as f:
                        est_node2uuid = json.load(f)

                    # Load ground truth from annotation file
                    gt_clustering, gt_node2uuid = load_ground_truth_clustering(anno_file, uuid_key, name_keys)

                    # Calculate metrics
                    metrics = calculate_evaluation_metrics(est_clustering, est_node2uuid, gt_clustering, gt_node2uuid)

                    # Format metrics text
                    metrics_text = f"F1 Score:         {metrics.get('f1', 0):.4f}\n"
                    metrics_text += f"Precision:        {metrics.get('precision', 0):.4f}\n"
                    metrics_text += f"Recall:           {metrics.get('recall', 0):.4f}\n"
                    metrics_text += f"Fraction Correct: {metrics.get('frac_correct', 0):.4f}\n"

                    # Append to config log file
                    append_metrics_to_config_log(config_log_file, metrics_text)

                    # Display metrics
                    print(metrics_text)
                else:
                    print("Clustering files not found, skipping evaluation.")

            except Exception as e:
                print(f"Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("All processing completed!")
    print("="*60)


if __name__ == "__main__":
    main()