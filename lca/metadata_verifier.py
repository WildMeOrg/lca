import sqlite3
import math
import pandas as pd
import os

from pathlib import Path

import numpy as np
import time
import logging

logger = logging.getLogger('lca')


class MetadataEmbeddings:
    """
    Embeddings wrapper that filters based on geographic/temporal metadata.
    Returns 0.0 for implausible matches, original score for plausible matches.
    """
    
    def __init__(self, base_embeddings, data_df, node2uuid):
        self.base_embeddings = base_embeddings
        self.metadata_filter = metadata_verifier(data_df, node2uuid)
        
        logger.info(f"Created MetadataEmbeddings wrapping {type(base_embeddings).__name__}")
    
    def _is_plausible(self, n0, n1):
        """Check if edge is plausible based on metadata."""
        plausible_pairs = self.metadata_filter([(n0, n1)])
        return len(plausible_pairs) > 0
    
    def get_score(self, n0, n1):
        """Return 0.0 if implausible, base score if plausible."""
        if self._is_plausible(n0, n1):
            return self.base_embeddings.get_score(n0, n1)
        else:
            return 0.0
    
    def _apply_metadata_filter_to_distmat(self, distmat, ids, start_idx):
        """Apply metadata filtering to distance matrix by setting implausible edges to infinity."""
        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                if i + start_idx != j:  # Don't filter self-edges
                    id1, id2 = ids[i + start_idx], ids[j]
                    if not self._is_plausible(id1, id2):
                        distmat[i, j] = 1
        return distmat
    
    def get_edges(self, topk=5, target_edges=10000, target_proportion=None, uuids_filter=None):
        """
        Get edges filtered by metadata plausibility.
        Applies metadata filtering to distance matrix BEFORE top-k selection.
        """
        start_time = time.time()
        print("Calculating distances with metadata filtering...")
        
        # Get distance matrix chunks from base embeddings  
        chunks, ids = self.base_embeddings.get_distmat_chunks(uuids_filter=uuids_filter)
        result = []
        start = 0
        
        if target_proportion is None:
            embeds_num = len(ids)
            total_edges = (embeds_num * embeds_num - embeds_num) / 2
            target_proportion = np.clip(target_edges / total_edges, 0, 1)
        else:
            target_edges = int(total_edges * target_proportion)
        
        print(f"Target: {target_edges}/{total_edges}")
        
        for distmat in chunks:
            # FIRST: Apply metadata filtering to distance matrix
            distmat = self._apply_metadata_filter_to_distmat(distmat, ids, start)
            
            # THEN: Apply top-k selection on filtered distance matrix
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk
            
            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)
            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]
            
            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            filtered = np.logical_or(sorted_dists, selected_dists)
            inds_y, inds_x = np.nonzero(filtered)
            
            # Create edges from filtered distance matrix
            for (ind1, ind2) in zip(inds_y, inds_x):
                score = 1 - distmat[ind1, ind2]
                if score > 0 and not np.isinf(score):  # Only valid edges
                    id1, id2 = ids[ind1 + start], ids[ind2]
                    result.append((*sorted([id1, id2]), score))
            
            print(f"Chunk result: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += distmat.shape[0]
            
        print(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        print(f"Metadata filtered edges: {len(set(result))}")
        return set(result)
    
    def __getattr__(self, name):
        """Delegate other methods to base embeddings."""
        return getattr(self.base_embeddings, name)
    
    def get_stats(self, df, filter_key):
        """Get statistics with metadata filtering applied."""
        start_time = time.time()
        print("Calculating distances with metadata filtering...")
        print(f"{len(self.base_embeddings.embeddings)}/{len(self.base_embeddings.ids)}")
        
        # Get distance matrix chunks from base embeddings
        chunks, ids = self.base_embeddings.get_distmat_chunks()
        filtered_chunks = []
        start = 0
        
        # Apply metadata filtering to each chunk
        for distmat in chunks:
            distmat = self._apply_metadata_filter_to_distmat(distmat, ids, start)
            filtered_chunks.append(distmat)
            start += distmat.shape[0]
        
        # Concatenate filtered chunks
        distmat = np.concatenate(filtered_chunks, axis=0)
        
        # Get labels and compute top-k statistics on filtered matrix
        labels = [df.loc[df['uuid_x'] == self.base_embeddings.uuids[id], filter_key].values[0] for id in ids]
        top1, top3, top5, top10 = self.base_embeddings.get_top_ks(labels, distmat, ks=[1, 3, 5, 10])
        
        return top1, top3, top5, top10

class metadata_verifier(object):
    def __init__(self, data_df, node2uuid):
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.uuid2node = {val:key for (key, val) in node2uuid.items()}

    def get_image_metadata(self, uuid):
        # Get the file_name from the dataframe
        longitude = self.data_df.loc[self.data_df['uuid_x'] == uuid, "longitude"].squeeze()
        latitude = self.data_df.loc[self.data_df['uuid_x'] == uuid, "latitude"].squeeze()
        datetime = self.data_df.loc[self.data_df['uuid_x'] == uuid, "datetime"].squeeze()
        file_name = self.data_df.loc[self.data_df['uuid_x'] == uuid, "file_name"].squeeze()
        return [str(longitude), str(latitude), str(datetime), str(file_name)]

    def convert_query(self, n0, n1):
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        return (uuid1, self.get_image_metadata(uuid1), uuid2, self.get_image_metadata(uuid2))

    def __call__(self, query):
        nodes_to_review = []
        nodes_query = [self.convert_query(n0, n1) for (n0, n1) in query]

        # Thresholds for plausibility (customize as needed)
        MAX_ZEBRA_SPEED_KMH = 65  # Maximum plausible speed in km/h

        for (uuid1, meta1, uuid2, meta2) in nodes_query:
            file1, file2 = meta1[-1], meta2[-1]
            if file1 == file2:
                continue

            # Extract datetimes and locations
            try:
                dt1 = pd.to_datetime(meta1[2])
                dt2 = pd.to_datetime(meta2[2])
                time_diff_hours = abs((dt1 - dt2).total_seconds() / 3600)
            except Exception:
                continue

            try:
                lon1, lat1 = float(meta1[0]), float(meta1[1])
                lon2, lat2 = float(meta2[0]), float(meta2[1])
            except Exception:
                continue

            if lon1 == -1 or lat1 == -1 or lon2 == -1 or lat2 == -1:
                nodes_to_review.append((uuid1, uuid2))
                continue

            # Haversine distance calculation
            R = 6371
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c

            # Avoid division by zero for same timestamp
            if time_diff_hours <= 0.1:
                plausible = distance_km < 0.1  # 100 meters, basically the same spot
            else:
                required_speed = distance_km / time_diff_hours
                plausible = required_speed <= MAX_ZEBRA_SPEED_KMH

            if plausible:
                nodes_to_review.append((uuid1, uuid2))

        return nodes_to_review