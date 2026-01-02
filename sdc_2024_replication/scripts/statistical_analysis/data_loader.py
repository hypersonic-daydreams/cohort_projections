import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path to find db_config
# Assuming this file is in scripts/statistical_analysis/
sys.path.append(str(Path(__file__).parent.parent))
from database import db_config


def load_migration_summary() -> pd.DataFrame:
    """
    Load migration summary data from PostgreSQL.
    Replicates the structure of nd_migration_summary.csv by aggregating
    census.state_components data.

    Returns DataFrame with columns:
        - year
        - nd_intl_migration
        - us_intl_migration
        - nd_share_of_us_intl_pct
        - nd_share_of_us_pop_pct
    """
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
    conn = db_config.get_db_connection()
    try:
        print("Executing SQL query to aggregated migration stats...")
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def load_panel_data() -> pd.DataFrame:
    """
    Load state-level panel data from PostgreSQL.
    Replaces combined_components_of_change.csv loading.
    """
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
    conn = db_config.get_db_connection()
    try:
        print("Executing SQL query for panel data...")
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()
