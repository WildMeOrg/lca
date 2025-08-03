import sqlite3
import os

from pathlib import Path

class human_db(object):
    def __init__(self, db_path, data_df, node2uuid, id_key='uuid'):
        self.db_path = db_path
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.uuid2node = {val:key for (key, val) in node2uuid.items()}
        self.id_key = id_key
        self.path_key = "file_path" if "file_path" in self.data_df.columns else "image_path"

    def init_db(self, db_path="./zebra_verification.db"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_verification (
            id TEXT PRIMARY KEY,
            uuid1 TEXT,
            image1_path TEXT,
            bbox1 TEXT,
            cluster1 TEXT,
            uuid2 TEXT,
            image2_path TEXT,
            bbox2 TEXT,
            cluster2 TEXT,
            status TEXT CHECK(status IN ('awaiting', 'in_progress', 'checked', 'sent')) DEFAULT 'awaiting',
            decision TEXT CHECK(decision IN ('none', 'correct', 'incorrect', 'cant_tell')) DEFAULT 'none',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            instance_id TEXT,
            heartbeat TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def get_image_path(self, uuid):
        # Get the file_name from the dataframe
        file_name = self.data_df.loc[self.data_df[self.id_key] == uuid, self.path_key].squeeze()
        return str(file_name)
    
    def get_bbox(self, uuid):
        # Get bbox from dataframe and convert from [x,y,w,h] to [x1,y1,x2,y2] format
        import json
        import pandas as pd
        import numpy as np
        
        if 'bbox' in self.data_df.columns:
            bbox = self.data_df.loc[self.data_df[self.id_key] == uuid, 'bbox'].squeeze()
            
            # Check if bbox is None or NaN (handling both scalar and array cases)
            if bbox is None:
                return None
            if isinstance(bbox, float) and pd.isna(bbox):
                return None
                
            # bbox is in [x, y, w, h] format, convert to [x1, y1, x2, y2]
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                return json.dumps([int(x), int(y), int(x + w), int(y + h)])
            else:
                return None
        else:
            return None
    
    def get_cluster(self, uuid):
        # Get cluster from tracking_id, use empty string if not available
        if 'tracking_id' in self.data_df.columns:
            cluster = self.data_df.loc[self.data_df[self.id_key] == uuid, 'tracking_id'].squeeze()
            return str(cluster) if cluster is not None and str(cluster) != 'nan' else ""
        else:
            return ""

    def convert_query(self, n0, n1):
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        unique_id = uuid1 + uuid2 if uuid1<uuid2 else uuid2+uuid1
        return (unique_id, uuid1, self.get_image_path(uuid1), self.get_bbox(uuid1), self.get_cluster(uuid1),
                uuid2, self.get_image_path(uuid2), self.get_bbox(uuid2), self.get_cluster(uuid2))

    def __call__(self, query):
        if not os.path.exists(self.db_path):
            print("DB doesn't exist")
            self.init_db(self.db_path)

        reviews = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Send new queries to db
            db_query = [self.convert_query(n0, n1) for (n0, n1) in query]
            cursor.executemany("""
                INSERT OR IGNORE INTO image_verification (id, uuid1, image1_path, bbox1, cluster1, uuid2, image2_path, bbox2, cluster2, status, decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'awaiting', 'none')
            """, db_query)
            inserted_count = cursor.rowcount
            print(f"Sent {inserted_count} new pairs for checking.")

            # Get checked images from db
            cursor.execute("""
                SELECT id, uuid1, uuid2, decision FROM image_verification
                WHERE status = 'checked'
            """)
            result = cursor.fetchall()
            if result:
                cursor.executemany("UPDATE image_verification SET status='sent' WHERE id=?", [(res[0],) for res in result])
            
            reviews = [(self.uuid2node[uuid1], self.uuid2node[uuid2], decision=='correct') for (_, uuid1, uuid2, decision) in result]

        return reviews, False