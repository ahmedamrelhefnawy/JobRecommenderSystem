import sqlite3
import pymssql
import numpy as np
import os
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class EmbeddingDB:
    def __init__(self, db_path="Data/embeddings.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.initialize_db()

    def initialize_db(self):
        """Create the database and tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create Jobs table - embeddings for title, content, and work_type
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id INTEGER PRIMARY KEY,
                    title BLOB,           -- embedding for title
                    content BLOB,         -- embedding for content
                    work_type BLOB,       -- embedding for work_type
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create Users table - embeddings for all relevant fields from user model
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    title BLOB,                  -- embedding for title
                    about BLOB,                  -- embedding for about
                    preferred_work_types BLOB,   -- embedding for preferred_work_types
                    experience_level BLOB,       -- embedding for experience_level
                    skills BLOB,                 -- embedding for skills
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    def _numpy_to_blob(self, arr: Optional[np.ndarray]) -> Optional[bytes]:
        """Convert numpy array to binary blob"""
        if arr is None:
            return None
        buf = BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    def _blob_to_numpy(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        """Convert binary blob back to numpy array"""
        if blob is None:
            return None
        buf = BytesIO(blob)
        return np.load(buf, allow_pickle=False)

    def store_job_embeddings(self, job_id: int, embeddings: dict):
        """Store job embeddings in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO jobs 
                (job_id, title, content, work_type)
                VALUES (?, ?, ?, ?)
            ''', (
                job_id,
                self._numpy_to_blob(embeddings.title),
                self._numpy_to_blob(embeddings.content),
                self._numpy_to_blob(embeddings.work_type)
            ))

            conn.commit()

    def store_user_embeddings(self, user_id: int, embeddings: dict):
        """Store user embeddings in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, title, about, preferred_work_types, experience_level, skills)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                self._numpy_to_blob(embeddings.title),
                self._numpy_to_blob(embeddings.about),
                self._numpy_to_blob(embeddings.preferred_work_types),
                self._numpy_to_blob(embeddings.experience_level),
                self._numpy_to_blob(embeddings.skills)
            ))

            conn.commit()

    def get_job_embeddings(self, job_id: int) -> dict:
        """Retrieve job embeddings from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT title, content, work_type
                FROM jobs WHERE job_id = ?
            ''', (job_id,))

            result = cursor.fetchone()
        if result:
            return {
                'title': self._blob_to_numpy(result[0]),
                'content': self._blob_to_numpy(result[1]),
                'work_type': self._blob_to_numpy(result[2])
            }
        return None
    
    def get_user_embeddings(self, user_id: int) -> dict:
        """Retrieve user embeddings from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT title, about, preferred_work_types, experience_level, skills
                FROM users WHERE user_id = ?
            ''', (user_id,))

            result = cursor.fetchone()
        if result:
            return {
                'title': self._blob_to_numpy(result[0]),
                'about': self._blob_to_numpy(result[1]),
                'preferred_work_types': self._blob_to_numpy(result[2]),
                'experience_level': self._blob_to_numpy(result[3]),
                'skills': self._blob_to_numpy(result[4])
            }
        return None

    def get_jobs_column_embeddings(self, job_ids: list[int], column_name: str) -> list[Optional[np.ndarray]]:
        """
        Retrieve embeddings for a specific column for multiple job IDs.
        Returns embeddings in the same order as the input job_ids list.
        
        Args:
            job_ids: List of job IDs to fetch embeddings for
            column_name: Name of the column to fetch ('title', 'content', or 'work_type')
            
        Returns:
            List of numpy arrays in the same order as job_ids. 
            If a job_id is not found, None will be in its place.
        """
        if not job_ids:
            return []
            
        # Get all column names from jobs table except job_id
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(jobs)")
            valid_columns = {row[1] for row in cursor.fetchall() if row[1] != 'job_id'}
        
        if column_name not in valid_columns:
            raise ValueError(f"Invalid column name: {column_name}. Must be one of: title, content, work_type")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create placeholders for SQL IN clause
            placeholders = ','.join('?' * len(job_ids))
            
            # Query the specific column for all job_ids
            cursor.execute(f'''
                SELECT job_id, {column_name}
                FROM jobs 
                WHERE job_id IN ({placeholders})
            ''', job_ids)
            
            # Create a mapping of job_id to embedding
            results_dict = {
                job_id: self._blob_to_numpy(embedding_blob)
                for job_id, embedding_blob in cursor.fetchall()
            }
            
        # Stack all embeddings into a single array
        embeddings = np.stack([results_dict[job_id][0,:] for job_id in job_ids], axis=0)
        return embeddings
    
    def get_missing_job_ids(self, job_ids: list[int]) -> list[int]:
        """
        Get list of job IDs that don't exist in the database.
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            List of job IDs that are not found in the database
        """
        if not job_ids:
            return []
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create placeholders for SQL IN clause
            placeholders = ','.join('?' * len(job_ids))
            
            # Get existing job IDs
            cursor.execute(f'''
                SELECT job_id 
                FROM jobs 
                WHERE job_id IN ({placeholders})
            ''', job_ids)

            existing_ids = {row[0] for row in cursor.fetchall()}
            
        # Return IDs that don't exist in database
        return [job_id for job_id in job_ids if job_id not in existing_ids]
    
    def get_missing_user_ids(self, user_ids: list[int]) -> list[int]:
        """
        Get list of user IDs that don't exist in the database.
        
        Args:
            user_ids: List of user IDs to check
            
        Returns:
            List of user IDs that are not found in the database
        """
        if not user_ids:
            return []
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create placeholders for SQL IN clause
            placeholders = ','.join('?' * len(user_ids))
            
            # Get existing user IDs
            cursor.execute(f'''
                SELECT user_id 
                FROM users 
                WHERE user_id IN ({placeholders})
            ''', user_ids)
            
            existing_ids = {row[0] for row in cursor.fetchall()}
            
        # Return IDs that don't exist in database
        return [user_id for user_id in user_ids if user_id not in existing_ids]



# Connect to MS SQL Server
DB_SERVER= os.environ.get('DB_SERVER')
DB_DATABASE = os.environ.get('DB_DATABASE')
DB_USER_ID = os.environ.get('DB_USER_ID')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT')


def get_mssql_connection():
    return pymssql.connect(server=DB_SERVER, port=DB_PORT, user=DB_USER_ID, password=DB_PASSWORD, database=DB_DATABASE)


class ServerDB:
    def __init__(self):
        pass
    
    def get_unflagged_users(self):
        """Fetch unflaged users from the database"""
        
        with get_mssql_connection() as conn, conn.cursor(as_dict= True) as cursor:
            cursor.execute("SELECT Id, Title, About, JobTypePreference FROM AspNetUsers WHERE Flag = 0;")
            users = cursor.fetchall()

        return users
    
    def get_unflagged_jobs(self):
        """Fetch unflaged jobs from the database"""
        with get_mssql_connection() as conn, conn.cursor(as_dict= True) as cursor:
            cursor.execute("SELECT JobId, JobTitle, Description, JobType FROM Jobs WHERE Flag = 0;")
            jobs = cursor.fetchall()
        
        return jobs
    
    def flag_users(self, user_ids: list[int]):
        """Flag users in the database"""
        with get_mssql_connection() as conn, conn.cursor(as_dict= True) as cursor:
            cursor.execute(f"UPDATE AspNetUsers SET Flag = 1 WHERE Id IN ({','.join(map(str, user_ids))});")
            conn.commit()

    
    def flag_jobs(self, job_ids: list[int]):
        """Flag jobs in the database"""
        with get_mssql_connection() as conn, conn.cursor(as_dict= True) as cursor:
            cursor.execute(f"UPDATE Jobs SET Flag = 1 WHERE JobId IN ({','.join(map(str, job_ids))});")
            conn.commit()