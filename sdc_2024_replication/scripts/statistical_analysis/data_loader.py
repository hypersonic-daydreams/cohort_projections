"""Central data access for SDC statistical analysis modules.

This module prefers PostgreSQL (when available) but supports file-based fallbacks
from `data/processed/immigration/analysis/` for reproducibility and portability.

To force a data source:
  - Set `SDC_ANALYSIS_DATA_SOURCE=db` to require PostgreSQL
  - Set `SDC_ANALYSIS_DATA_SOURCE=files` to skip PostgreSQL and use local files
  - Default (`auto`) tries PostgreSQL then falls back to files
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import ConfigLoader, setup_logger

# Add scripts directory to path to find `database/`
sys.path.append(str(Path(__file__).parent.parent))
from database import db_config  # noqa: E402

logger = setup_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _immigration_analysis_dir() -> Path:
    """Return the immigration analysis directory from config (with fallback)."""
    root = _project_root()
    cfg = ConfigLoader().get_projection_config()
    processed_dir = (
        cfg.get("data_sources", {})
        .get("acs_moved_from_abroad", {})
        .get("processed_dir", "data/processed/immigration/analysis")
    )
    return root / processed_dir


def _data_source_mode() -> str:
    return os.getenv("SDC_ANALYSIS_DATA_SOURCE", "auto").strip().lower()


def _should_try_db() -> bool:
    return _data_source_mode() in {"auto", "db"}


def _should_use_files_only() -> bool:
    return _data_source_mode() == "files"


def _read_sql(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    conn = db_config.get_db_connection()
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


def load_migration_summary() -> pd.DataFrame:
    """Load ND migration summary series (DB preferred; CSV fallback)."""
    if _should_try_db() and not _should_use_files_only():
        query = """
        WITH state_data AS (
            SELECT
                year,
                state_name,
                intl_migration,
                population
            FROM census.state_components
            WHERE state_name NOT IN ('Puerto Rico', 'United States', 'US Region', 'US Division')
              AND state_name IS NOT NULL
        ),
        us_stats AS (
            SELECT
                year,
                SUM(intl_migration) as us_intl_migration,
                SUM(population) as us_population
            FROM state_data
            GROUP BY year
        ),
        nd_stats AS (
            SELECT
                year,
                intl_migration as nd_intl_migration,
                population as nd_population
            FROM state_data
            WHERE state_name = 'North Dakota'
        )
        SELECT
            u.year,
            n.nd_intl_migration,
            u.us_intl_migration,
            (CAST(n.nd_intl_migration AS FLOAT) / NULLIF(u.us_intl_migration, 0)) * 100 as nd_share_of_us_intl_pct,
            (CAST(n.nd_population AS FLOAT) / NULLIF(u.us_population, 0)) * 100 as nd_share_of_us_pop_pct
        FROM us_stats u
        JOIN nd_stats n ON u.year = n.year
        ORDER BY u.year
        """
        try:
            return _read_sql(query)
        except Exception as exc:
            if _data_source_mode() == "db":
                raise
            logger.warning("PostgreSQL migration summary load failed; falling back to files (%s).", exc)

    csv_path = _immigration_analysis_dir() / "nd_migration_summary.csv"
    return pd.read_csv(csv_path)


def load_panel_data() -> pd.DataFrame:
    """Load state-year panel data (DB preferred; CSV fallback)."""
    if _should_try_db() and not _should_use_files_only():
        query = """
        SELECT
            year,
            state_name as state,
            intl_migration,
            population,
            pop_change,
            births,
            deaths,
            domestic_migration,
            natural_change,
            net_migration
        FROM census.state_components
        WHERE state_name IS NOT NULL
          AND state_name NOT IN ('Puerto Rico', 'United States', 'US Region', 'US Division')
        ORDER BY state_name, year
        """
        try:
            return _read_sql(query)
        except Exception as exc:
            if _data_source_mode() == "db":
                raise
            logger.warning("PostgreSQL panel load failed; falling back to files (%s).", exc)

    csv_path = _immigration_analysis_dir() / "combined_components_of_change.csv"
    return pd.read_csv(csv_path)


def load_refugee_arrivals() -> pd.DataFrame:
    """Load refugee arrivals (DB preferred; parquet fallback)."""
    if _should_try_db() and not _should_use_files_only():
        query = """
        SELECT
            fiscal_year,
            destination_state as state,
            nationality,
            arrivals
        FROM rpc.refugee_arrivals
        """
        try:
            return _read_sql(query)
        except Exception as exc:
            if _data_source_mode() == "db":
                raise
            logger.warning("PostgreSQL refugee load failed; falling back to files (%s).", exc)

    parquet_path = _immigration_analysis_dir() / "refugee_arrivals_by_state_nationality.parquet"
    return pd.read_parquet(parquet_path)


def load_amerasian_siv_arrivals() -> pd.DataFrame:
    """Load Amerasian/SIV arrivals (DB optional; parquet fallback)."""
    if _should_try_db() and not _should_use_files_only():
        query = """
        SELECT
            fiscal_year,
            destination_state as state,
            nationality,
            arrivals
        FROM rpc.amerasian_siv_arrivals
        """
        try:
            return _read_sql(query)
        except Exception as exc:
            if _data_source_mode() == "db":
                raise
            logger.warning(
                "PostgreSQL Amerasian/SIV load failed; falling back to files (%s).", exc
            )

    parquet_path = (
        _immigration_analysis_dir() / "amerasian_siv_arrivals_by_state_nationality.parquet"
    )
    return pd.read_parquet(parquet_path)


def load_state_components() -> pd.DataFrame:
    """Load basic state components (DB preferred; CSV fallback)."""
    if _should_try_db() and not _should_use_files_only():
        query = """
        SELECT
            year,
            state_name as state,
            intl_migration,
            domestic_migration,
            population as pop_estimate
        FROM census.state_components
        WHERE state_name IS NOT NULL
        """
        try:
            return _read_sql(query)
        except Exception as exc:
            if _data_source_mode() == "db":
                raise
            logger.warning("PostgreSQL state components load failed; falling back to files (%s).", exc)

    csv_path = _immigration_analysis_dir() / "combined_components_of_change.csv"
    df = pd.read_csv(csv_path, usecols=["year", "state", "intl_migration", "domestic_migration", "population"])
    return df.rename(columns={"population": "pop_estimate"})
