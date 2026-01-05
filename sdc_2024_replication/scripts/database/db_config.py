"""PostgreSQL connection helpers for SDC replication scripts.

Configuration is provided via environment variables (preferred), with safe
defaults for local development that avoid hard-coding user-specific values.

Supported environment variables:
  - DEMOGRAPHY_DATABASE_URL: Full SQLAlchemy URL (takes precedence)
  - DEMOGRAPHY_DB_NAME
  - DEMOGRAPHY_DB_USER
  - DEMOGRAPHY_DB_PASSWORD (optional)
  - DEMOGRAPHY_DB_HOST
  - DEMOGRAPHY_DB_PORT
"""

from __future__ import annotations

import getpass
import os
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

DB_NAME = os.getenv("DEMOGRAPHY_DB_NAME", "demography_db")
DB_USER = os.getenv("DEMOGRAPHY_DB_USER", getpass.getuser())
DB_PASSWORD = os.getenv("DEMOGRAPHY_DB_PASSWORD")
DB_HOST = os.getenv("DEMOGRAPHY_DB_HOST", "localhost")
DB_PORT = os.getenv("DEMOGRAPHY_DB_PORT", "5432")


def _build_database_url() -> str:
    """Build a SQLAlchemy PostgreSQL URL from environment variables."""
    explicit_url = os.getenv("DEMOGRAPHY_DATABASE_URL")
    if explicit_url:
        return explicit_url

    if DB_PASSWORD:
        password = quote_plus(DB_PASSWORD)
        return f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


DATABASE_URL = _build_database_url()


def get_db_connection() -> psycopg2.extensions.connection:
    """Return a psycopg2 connection using environment configuration."""
    kwargs: dict[str, Any] = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "host": DB_HOST,
        "port": DB_PORT,
    }
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    return psycopg2.connect(**kwargs)


def get_db_engine():
    """Return a SQLAlchemy engine for the configured database."""
    return create_engine(DATABASE_URL)


def load_table_as_df(table_name: str, schema: str = "public") -> pd.DataFrame:
    """Load an entire table into a DataFrame."""
    engine = get_db_engine()
    query = f"SELECT * FROM {schema}.{table_name}"
    return pd.read_sql(query, engine)


def run_query(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    engine = get_db_engine()
    return pd.read_sql(query, engine, params=params)
