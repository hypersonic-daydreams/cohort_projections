#!/usr/bin/env python3
"""
Build a PostgreSQL analytics layer from Census PEP (POPEST) parquet files.

Created: 2026-02-03
ADR: 034 (Census PEP Data Archive — Phase 5)
Author: nhaarstad

Purpose
-------
Create harmonized cross-vintage population totals tables in PostgreSQL for
interactive SQL analysis of Census PEP data. The raw Census files use
different column layouts, YEAR code conventions, and geographic identifiers
across vintages (1970s-2024), making ad-hoc cross-vintage queries impractical
on raw files. This script unpivots wide-format POPESTIMATE columns into a
uniform long-format schema (vintage, year, geoid, population) and loads the
result into a PostgreSQL analytics schema with appropriate indexes and
materialized views for North Dakota filtering.

Method
------
1. Load the shared catalog.yaml and identify datasets with parquet files at
   the place and county geographic levels.
2. For each place-level parquet file:
   a. Read the parquet file into a DataFrame.
   b. Unpivot POPESTIMATEYYYY columns into long format (year, population).
   c. Construct standardized FIPS codes (state_fips, place_fips, geoid).
   d. Yield a chunk DataFrame with the harmonized schema.
3. For each county-level parquet file:
   a. Handle two distinct source layouts: wide-format (POPESTIMATEYYYY
      columns) and the intercensal ASRH format (YEAR code + TOT_POP columns,
      filtered to AGEGRP=0 for totals, YEAR codes 2-12 mapped to calendar
      years 2010-2020).
   b. Construct standardized FIPS codes (state_fips, county_fips, geoid).
   c. Yield a chunk DataFrame with the harmonized schema.
4. For each target table (place_totals, county_totals):
   a. Create a staging table ({table}__new) with the target DDL.
   b. Write each chunk to a temporary CSV, then COPY into the staging table.
   c. Create indexes (geoid+year, vintage+geoid+year, state+year).
   d. Drop existing materialized views, atomically swap the staging table to
      the final name, then recreate materialized views (e.g.,
      mv_place_totals_nd for state_fips='38').
5. Record a build timestamp in the build_info table and commit.

Key design decisions
--------------------
- **Atomic table swap**: The staging table is built as {table}__new, then the
  old table is dropped and the new one is renamed. This avoids downtime: the
  table is either fully old or fully new, never partially loaded.
- **CSV COPY for bulk loading**: Using PostgreSQL COPY FROM STDIN with CSV
  format is 10-50x faster than row-by-row INSERT for the hundreds of thousands
  of rows being loaded. Temporary CSV files are cleaned up after each chunk.
- **Materialized views for ND filtering**: Since most analytical queries filter
  to state_fips='38' (North Dakota), pre-filtered materialized views
  (mv_place_totals_nd, mv_county_totals_nd) avoid full table scans and provide
  sub-millisecond query times for interactive analysis.
- **YEAR code translation for intercensal data**: The 2010-2020 county
  intercensal file uses YEAR codes (2=April 2010, 3=July 2010, ..., 12=July
  2020) rather than calendar years. The script maps codes 2-12 to years
  2010-2020 and filters to AGEGRP=0 (total population) to match the totals
  schema.

Validation results (2026-02-03)
-------------------------------
- place_totals: loaded from 4 place-level datasets (2000-2010 through 2020-2024)
- county_totals: loaded from 5 county-level datasets (2000-2009 through 2020-2024)
- ND materialized views created with indexes
- Build timestamp recorded in popest_analytics.build_info

Inputs
------
- $CENSUS_POPEST_DIR/catalog.yaml
    Shared dataset catalog identifying parquet files per dataset.
- $CENSUS_POPEST_DIR/parquet/{vintage}/{level}/*.parquet
    Parquet files produced by Phase 2 (convert_popest_to_parquet.py).
- CENSUS_POPEST_PG_DSN environment variable
    PostgreSQL connection string (e.g., "host=localhost dbname=census").

Output
------
- PostgreSQL schema: popest_analytics
    Tables: place_totals, county_totals, build_info
    Materialized views: mv_place_totals_nd, mv_county_totals_nd
    Indexes on (geoid, year), (vintage, geoid, year), (state_fips, year)

Usage
-----
    export CENSUS_POPEST_PG_DSN="host=localhost dbname=census"
    python scripts/data/build_popest_postgres.py
    python scripts/data/build_popest_postgres.py --skip-place
    python scripts/data/build_popest_postgres.py --skip-county --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2

from cohort_projections.data.popest_shared import (
    configure_logging,
    load_catalog,
    resolve_popest_paths,
)
from cohort_projections.utils.reproducibility import log_execution

LOGGER = logging.getLogger(__name__)

ANALYTICS_SCHEMA = "popest_analytics"


def _require_pg_dsn() -> str:
    dsn = os.getenv("CENSUS_POPEST_PG_DSN")
    if not dsn:
        raise ValueError("CENSUS_POPEST_PG_DSN is not set.")
    return dsn


def _ensure_schema(cur) -> None:
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ANALYTICS_SCHEMA}")


def _write_df_to_temp_csv(df: pd.DataFrame) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    df.to_csv(tmp_path, index=False, na_rep="\\N")
    return tmp_path


def _copy_csv_to_table(cur, csv_path: Path, fq_table: str, columns: list[str]) -> None:
    cols_sql = ", ".join(columns)
    copy_sql = f"COPY {fq_table} ({cols_sql}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N')"
    with csv_path.open("r", encoding="utf-8") as f:
        cur.copy_expert(copy_sql, f)


def _zfill_series(series: pd.Series, width: int) -> pd.Series:
    return series.astype("string").str.zfill(width)


def _unpivot_wide_popestimate(
    df: pd.DataFrame,
    *,
    vintage: str,
    source_dataset_id: str,
    id_cols: list[str],
) -> pd.DataFrame:
    # Some vintages encode month+year (e.g., POPESTIMATE072000, POPESTIMATE042010).
    estimate_cols = [c for c in df.columns if re.fullmatch(r"POPESTIMATE\d{4,6}", str(c))]
    if not estimate_cols:
        raise ValueError(f"No POPESTIMATEYYYY columns found for dataset {source_dataset_id}")

    long_df = df.melt(
        id_vars=id_cols, value_vars=estimate_cols, var_name="year", value_name="population"
    )
    long_df["vintage"] = vintage
    long_df["source_dataset_id"] = source_dataset_id
    long_df["year"] = long_df["year"].astype("string").str.extract(r"(\d{4})$")[0].astype("Int64")
    long_df["population"] = pd.to_numeric(long_df["population"], errors="coerce").astype("Int64")
    return long_df


def _iter_level_datasets(catalog: dict[str, Any], level: str) -> list[dict[str, Any]]:
    return [
        d for d in catalog.get("datasets", []) if d.get("level") == level and d.get("parquet_file")
    ]


def _place_chunk(ds: dict[str, Any], parquet_dir: Path) -> pd.DataFrame:
    parquet_path = parquet_dir / ds["parquet_file"]
    df = pd.read_parquet(parquet_path)

    vintage = str(ds.get("vintage"))
    source_dataset_id = str(ds.get("id"))

    id_cols = [
        c for c in ["SUMLEV", "STATE", "COUNTY", "PLACE", "NAME", "STNAME"] if c in df.columns
    ]
    long_df = _unpivot_wide_popestimate(
        df, vintage=vintage, source_dataset_id=source_dataset_id, id_cols=id_cols
    )

    long_df["state_fips"] = _zfill_series(
        long_df.get("STATE", pd.Series([pd.NA] * len(long_df))), 2
    )
    long_df["place_fips"] = _zfill_series(
        long_df.get("PLACE", pd.Series([pd.NA] * len(long_df))), 5
    )
    long_df["county_fips"] = _zfill_series(
        long_df.get("COUNTY", pd.Series([pd.NA] * len(long_df))), 3
    )

    long_df["geoid"] = (long_df["state_fips"] + long_df["place_fips"]).where(  # type: ignore[call-overload]
        long_df["state_fips"].notna() & long_df["place_fips"].notna(),
        None,
    )

    out = pd.DataFrame(
        {
            "vintage": long_df["vintage"],
            "year": long_df["year"],
            "geoid": long_df["geoid"],
            "state_fips": long_df["state_fips"],
            "county_fips": long_df["county_fips"],
            "place_fips": long_df["place_fips"],
            "name": long_df.get("NAME", pd.Series([pd.NA] * len(long_df))),
            "stname": long_df.get("STNAME", pd.Series([pd.NA] * len(long_df))),
            "sumlev": long_df.get("SUMLEV", pd.Series([pd.NA] * len(long_df))),
            "population": long_df["population"],
            "source_dataset_id": long_df["source_dataset_id"],
        }
    )
    return out


def _county_chunk(ds: dict[str, Any], parquet_dir: Path) -> pd.DataFrame:
    parquet_path = parquet_dir / ds["parquet_file"]
    df = pd.read_parquet(parquet_path)

    vintage = str(ds.get("vintage"))
    source_dataset_id = str(ds.get("id"))

    if "YEAR" in df.columns and "TOT_POP" in df.columns:
        # Intercensal ASRH county file (cc-est2020int-alldata): use AGEGRP=0 and YEAR codes.
        subset_cols = [
            c
            for c in ["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "YEAR", "AGEGRP", "TOT_POP"]
            if c in df.columns
        ]
        subset = df[subset_cols].copy()
        subset = subset[subset["AGEGRP"].astype("string") == "0"]
        subset["source_year_code"] = pd.to_numeric(subset["YEAR"], errors="coerce").astype("Int64")
        subset = subset[(subset["source_year_code"] >= 2) & (subset["source_year_code"] <= 12)]
        subset["year"] = (subset["source_year_code"] + 2008).astype("Int64")
        subset["population"] = pd.to_numeric(subset["TOT_POP"], errors="coerce").astype("Int64")
        subset["vintage"] = vintage
        subset["source_dataset_id"] = source_dataset_id
        working = subset
    else:
        id_cols = [c for c in ["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME"] if c in df.columns]
        working = _unpivot_wide_popestimate(
            df, vintage=vintage, source_dataset_id=source_dataset_id, id_cols=id_cols
        )
        working["source_year_code"] = pd.NA

    working["state_fips"] = _zfill_series(
        working.get("STATE", pd.Series([pd.NA] * len(working))), 2
    )
    working["county_fips"] = _zfill_series(
        working.get("COUNTY", pd.Series([pd.NA] * len(working))), 3
    )
    working["geoid"] = (working["state_fips"] + working["county_fips"]).where(  # type: ignore[call-overload]
        working["state_fips"].notna() & working["county_fips"].notna(),
        None,
    )

    out = pd.DataFrame(
        {
            "vintage": working["vintage"],
            "year": working["year"].astype("Int64"),
            "geoid": working["geoid"],
            "state_fips": working["state_fips"],
            "county_fips": working["county_fips"],
            "stname": working.get("STNAME", pd.Series([pd.NA] * len(working))),
            "ctyname": working.get("CTYNAME", pd.Series([pd.NA] * len(working))),
            "sumlev": working.get("SUMLEV", pd.Series([pd.NA] * len(working))),
            "source_year_code": working.get(
                "source_year_code", pd.Series([pd.NA] * len(working))
            ).astype("Int64"),
            "population": working["population"].astype("Int64"),
            "source_dataset_id": working["source_dataset_id"],
        }
    )
    return out


def _swap_in_table(cur, *, table_name: str) -> None:
    fq_new = f"{ANALYTICS_SCHEMA}.{table_name}__new"
    fq_final = f"{ANALYTICS_SCHEMA}.{table_name}"
    cur.execute(f"DROP TABLE IF EXISTS {fq_final}")
    cur.execute(f"ALTER TABLE {fq_new} RENAME TO {table_name}")


def _build_table_from_chunks(
    *,
    conn,
    table_name: str,
    ddl_sql: str,
    index_sql: list[str],
    matviews_sql: list[str],
    chunk_iter,
) -> None:
    fq_new = f"{ANALYTICS_SCHEMA}.{table_name}__new"

    with conn.cursor() as cur:
        _ensure_schema(cur)
        cur.execute(f"DROP TABLE IF EXISTS {fq_new}")
        cur.execute(ddl_sql.format(fq_table=fq_new))

        for chunk_df in chunk_iter:
            csv_path = _write_df_to_temp_csv(chunk_df)
            try:
                _copy_csv_to_table(cur, csv_path, fq_new, list(chunk_df.columns))
            finally:
                with suppress(OSError):
                    csv_path.unlink()

        for sql in index_sql:
            cur.execute(sql.format(fq_table=fq_new, schema=ANALYTICS_SCHEMA, table=table_name))

        pre_drop_matviews = [
            sql for sql in matviews_sql if sql.strip().upper().startswith("DROP MATERIALIZED VIEW")
        ]
        post_swap_sql = [sql for sql in matviews_sql if sql not in pre_drop_matviews]

        for sql in pre_drop_matviews:
            cur.execute(sql.format(schema=ANALYTICS_SCHEMA))

        _swap_in_table(cur, table_name=table_name)

        for sql in post_swap_sql:
            cur.execute(sql.format(schema=ANALYTICS_SCHEMA))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build POPEST Postgres analytics schema (ADR-034 Phase 5)"
    )
    parser.add_argument(
        "--popest-dir", help="Override CENSUS_POPEST_DIR for the shared data directory."
    )
    parser.add_argument("--skip-place", action="store_true", help="Skip building place totals.")
    parser.add_argument("--skip-county", action="store_true", help="Skip building county totals.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    configure_logging(args.verbose)
    dsn = _require_pg_dsn()
    paths = resolve_popest_paths(args.popest_dir)
    catalog = load_catalog(paths.catalog_path)

    place_datasets = [] if args.skip_place else _iter_level_datasets(catalog, "place")
    county_datasets = [] if args.skip_county else _iter_level_datasets(catalog, "county")

    with (
        log_execution(
            __file__,
            parameters={
                "skip_place": args.skip_place,
                "skip_county": args.skip_county,
                "schema": ANALYTICS_SCHEMA,
            },
            inputs=[paths.catalog_path],
            outputs=[],
        ),
        psycopg2.connect(dsn) as conn,
    ):
        conn.autocommit = False

        if place_datasets:
            LOGGER.info(
                "Building %s.place_totals (%d parquet sources)",
                ANALYTICS_SCHEMA,
                len(place_datasets),
            )
            _build_table_from_chunks(
                conn=conn,
                table_name="place_totals",
                ddl_sql="""
                    CREATE TABLE {fq_table} (
                        vintage TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        geoid TEXT,
                        state_fips TEXT,
                        county_fips TEXT,
                        place_fips TEXT,
                        name TEXT,
                        stname TEXT,
                        sumlev TEXT,
                        population BIGINT,
                        source_dataset_id TEXT NOT NULL
                    )
                """,
                index_sql=[
                    "CREATE INDEX place_totals__new_geoid_year_idx ON {fq_table} (geoid, year)",
                    "CREATE INDEX place_totals__new_vintage_geoid_year_idx ON {fq_table} (vintage, geoid, year)",
                    "CREATE INDEX place_totals__new_state_year_idx ON {fq_table} (state_fips, year)",
                ],
                matviews_sql=[
                    "DROP MATERIALIZED VIEW IF EXISTS {schema}.mv_place_totals_nd",
                    """
                    CREATE MATERIALIZED VIEW {schema}.mv_place_totals_nd AS
                    SELECT *
                    FROM {schema}.place_totals
                    WHERE state_fips = '38'
                    """,
                    "CREATE INDEX mv_place_totals_nd_geoid_year_idx ON {schema}.mv_place_totals_nd (geoid, year)",
                ],
                chunk_iter=(_place_chunk(ds, paths.parquet_dir) for ds in place_datasets),
            )

        if county_datasets:
            LOGGER.info(
                "Building %s.county_totals (%d parquet sources)",
                ANALYTICS_SCHEMA,
                len(county_datasets),
            )
            _build_table_from_chunks(
                conn=conn,
                table_name="county_totals",
                ddl_sql="""
                    CREATE TABLE {fq_table} (
                        vintage TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        geoid TEXT,
                        state_fips TEXT,
                        county_fips TEXT,
                        stname TEXT,
                        ctyname TEXT,
                        sumlev TEXT,
                        source_year_code INTEGER,
                        population BIGINT,
                        source_dataset_id TEXT NOT NULL
                    )
                """,
                index_sql=[
                    "CREATE INDEX county_totals__new_geoid_year_idx ON {fq_table} (geoid, year)",
                    "CREATE INDEX county_totals__new_vintage_geoid_year_idx ON {fq_table} (vintage, geoid, year)",
                    "CREATE INDEX county_totals__new_state_year_idx ON {fq_table} (state_fips, year)",
                ],
                matviews_sql=[
                    "DROP MATERIALIZED VIEW IF EXISTS {schema}.mv_county_totals_nd",
                    """
                    CREATE MATERIALIZED VIEW {schema}.mv_county_totals_nd AS
                    SELECT *
                    FROM {schema}.county_totals
                    WHERE state_fips = '38'
                    """,
                    "CREATE INDEX mv_county_totals_nd_geoid_year_idx ON {schema}.mv_county_totals_nd (geoid, year)",
                ],
                chunk_iter=(_county_chunk(ds, paths.parquet_dir) for ds in county_datasets),
            )

        with conn.cursor() as cur:
            _ensure_schema(cur)
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {ANALYTICS_SCHEMA}.build_info (built_at TIMESTAMP, note TEXT)"
            )
            cur.execute(
                f"INSERT INTO {ANALYTICS_SCHEMA}.build_info (built_at, note) VALUES (%s, %s)",
                (datetime.now(UTC), "Built via scripts/data/build_popest_postgres.py"),
            )

        conn.commit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
