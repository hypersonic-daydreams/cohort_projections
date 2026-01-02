import hashlib
import logging
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_NAME = "demography_db"
DB_USER = "nhaarstad"  # Assuming current user
# DB_HOST = "localhost" # Default


def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            # host=DB_HOST
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


@contextmanager
def get_db_cursor(commit=False):
    """Context manager for database cursor."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def calculate_file_hash(file_path: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def register_source_file(cursor, file_path: Path, description: str = None) -> int:
    """
    Registers a source file in the meta.source_files table.
    Returns the source_id.
    If file already exists (by hash), returns existing source_id.
    """
    try:
        # 1. Calculate Hash
        file_hash = calculate_file_hash(file_path)
        file_name = file_path.name
        abs_path = str(file_path.absolute())

        # 2. Check if exists
        cursor.execute("SELECT source_id FROM meta.source_files WHERE file_hash = %s", (file_hash,))
        result = cursor.fetchone()

        if result:
            logger.info(f"File {file_name} already registered (ID: {result['source_id']}).")
            return result["source_id"]

        # 3. Insert if not exists
        cursor.execute(
            """
            INSERT INTO meta.source_files (file_path, file_name, file_hash, description)
            VALUES (%s, %s, %s, %s)
            RETURNING source_id
            """,
            (abs_path, file_name, file_hash, description),
        )
        source_id = cursor.fetchone()["source_id"]
        logger.info(f"Registered new file {file_name} (ID: {source_id}).")
        return source_id

    except Exception as e:
        logger.error(f"Error registering file {file_path}: {e}")
        raise
