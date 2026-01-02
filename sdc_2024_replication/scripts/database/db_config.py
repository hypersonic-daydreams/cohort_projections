import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Database Credentials - typically these should be env vars
# Using defaults for local dev
DB_NAME = "demography_db"
DB_USER = "nhaarstad"
DB_HOST = "localhost"  # or socket default
DB_PORT = "5432"

DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_db_connection():
    """Returns a raw psycopg2 connection."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER)


def get_db_engine():
    """Returns a SQLAlchemy engine."""
    return create_engine(DATABASE_URL)


def load_table_as_df(table_name: str, schema: str = "public") -> pd.DataFrame:
    """Loads an entire table into a DataFrame."""
    engine = get_db_engine()
    query = f"SELECT * FROM {schema}.{table_name}"
    return pd.read_sql(query, engine)


def run_query(query: str, params: dict = None) -> pd.DataFrame:
    """Runs a SQL query and returns a DataFrame."""
    engine = get_db_engine()
    return pd.read_sql(query, engine, params=params)
