import sqlite3
import os

from pathlib import Path

class human_db(object):
    def __init__(self, db_path, data_df, node2uuid):
        self.db_path = db_path
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.uuid2node = {val:key for (key, val) in node2uuid.items()}

    def init_db(self, db_path="./zebra_verification.db"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_verification (
            id TEXT PRIMARY KEY,
            uuid1 TEXT,
            image1_path TEXT,
            uuid2 TEXT,
            image2_path TEXT,
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
        file_name = self.data_df.loc[self.data_df['uuid_x'] == uuid, "file_name"].squeeze()
        return str(file_name)

    def convert_query(self, n0, n1):
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        unique_id = uuid1 + uuid2 if uuid1<uuid2 else uuid2+uuid1
        return (unique_id, uuid1, self.get_image_path(uuid1), uuid2, self.get_image_path(uuid2))

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
                INSERT OR IGNORE INTO image_verification (id, uuid1, image1_path, uuid2, image2_path, status, decision)
                VALUES (?, ?, ?, ?, ?, 'awaiting', 'none')
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