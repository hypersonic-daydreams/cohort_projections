#!/usr/bin/env python3
"""
Extract County-Level Net Migration from Census PEP with Full Metadata Tracking

Enhanced version of ADR-035 Phase 1 implementation that populates comprehensive
metadata tables in PostgreSQL for rigorous time series construction.

Key enhancements:
- Populates all 5 metadata tables (datasets, migration, validation, rules, log)
- Extracts RESIDUAL column for data quality assessment
- Handles 2020 overlap with explicit preference rules
- Documents uncertainty levels and revision status
- Enables future agents to understand data provenance and quality

Database: census_popest (via CENSUS_POPEST_PG_DSN environment variable)

Author: Generated for ADR-035 Phase 1 + Metadata Enhancement
Date: 2026-02-03
"""

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Constants
ND_FIPS = "38"  # North Dakota state FIPS code
PEP_BASE = Path.home() / "workspace" / "shared-data" / "census" / "popest" / "parquet"
DOCS_BASE = Path.home() / "workspace" / "shared-data" / "census" / "popest"
OUTPUT_DIR = Path("data/processed")
STATE_FILE = Path("data/processed/immigration/state_migration_components_2000_2024.csv")
SCRIPT_VERSION = "2.0.0-metadata"

# Vintage file mappings with enhanced metadata
VINTAGES = {
    "2000-2009": {
        "dataset_id": "co-est2009-alldata",
        "file": PEP_BASE / "2000-2009/county/co-est2009-alldata.parquet",
        "years": list(range(2000, 2010)),
        "estimate_type": "postcensal",
        "revision_status": "final",
        "uncertainty_level": "moderate",
        "description": "Postcensal estimates (2000-2009) - completed decade",
        "methodology_doc_path": "derived/docs/2024-subcounty-methodology/fulltext.txt",
        "file_layout_doc_path": "derived/docs/co-est2024-alldata-layout/fulltext.txt",
        "notes": "Completed decade, postcensal estimates not revised by subsequent census",
    },
    "2010-2019": {
        "dataset_id": "co-est2019-alldata",
        "file": PEP_BASE / "2010-2019/county/co-est2019-alldata.parquet",
        "years": list(range(2010, 2020)),
        "estimate_type": "postcensal",
        "revision_status": "superseded",
        "uncertainty_level": "moderate",
        "description": "Postcensal estimates (2010-2019) - NOTE: lower hierarchical validation pass rate",
        "methodology_doc_path": "derived/docs/2024-subcounty-methodology/fulltext.txt",
        "file_layout_doc_path": "derived/docs/co-est2024-alldata-layout/fulltext.txt",
        "notes": "Superseded by 2010-2020 intercensal for population, but intercensal components not published. Hierarchical validation pass rate only 30%. Includes Bakken boom/bust volatility.",
    },
    "2020-2024": {
        "dataset_id": "co-est2024-alldata",
        "file": PEP_BASE / "2020-2024/county/co-est2024-alldata.parquet",
        "years": list(range(2020, 2025)),
        "estimate_type": "postcensal",
        "revision_status": "current",
        "uncertainty_level": "high",
        "description": "Postcensal estimates (2020-2024) - current, not yet census-aligned",
        "methodology_doc_path": "derived/docs/2024-subcounty-methodology/fulltext.txt",
        "file_layout_doc_path": "derived/docs/co-est2024-alldata-layout/fulltext.txt",
        "notes": "Most recent data, not yet revised by 2030 census. 2020 base integrates 2020 Census, Vintage 2020 estimates, and 2020 Demographic Analysis.",
    },
}


def get_db_connection():
    """Get PostgreSQL database connection from environment variable."""
    dsn = os.environ.get("CENSUS_POPEST_PG_DSN")
    if not dsn:
        raise ValueError(
            "CENSUS_POPEST_PG_DSN environment variable not set.\n"
            "Please set it to your PostgreSQL connection string, e.g.:\n"
            "  export CENSUS_POPEST_PG_DSN='postgresql:///census_popest'"
        )
    return psycopg2.connect(dsn)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_doc_sha256(doc_path: str) -> str | None:
    """Compute SHA256 of documentation file if it exists."""
    full_path = DOCS_BASE / doc_path
    if full_path.exists():
        return compute_file_sha256(full_path)
    return None


def populate_datasets_table(conn, vintages: dict):
    """
    Populate census_pep_datasets table with metadata for all vintages.

    Args:
        conn: psycopg2 connection
        vintages: VINTAGES dictionary with metadata
    """
    print(f"\n{'=' * 70}")
    print("POPULATING METADATA: census_pep_datasets")

    cursor = conn.cursor()

    for vintage_name, info in vintages.items():
        file_path = info["file"]

        # Compute checksums
        source_sha256 = compute_file_sha256(file_path)
        methodology_sha256 = compute_doc_sha256(info["methodology_doc_path"])
        layout_sha256 = compute_doc_sha256(info["file_layout_doc_path"])

        # Relative path from shared-data base
        relative_path = str(file_path.relative_to(PEP_BASE.parent))

        dataset_data = {
            "dataset_id": info["dataset_id"],
            "vintage_label": vintage_name,
            "estimate_type": info["estimate_type"],
            "revision_status": info["revision_status"],
            "uncertainty_level": info["uncertainty_level"],
            "year_range_start": min(info["years"]),
            "year_range_end": max(info["years"]),
            "source_file_path": relative_path,
            "source_file_sha256": source_sha256,
            "extracted_by": f"extract_pep_county_migration_with_metadata.py v{SCRIPT_VERSION}",
            "methodology_doc_path": info["methodology_doc_path"],
            "methodology_doc_sha256": methodology_sha256,
            "file_layout_doc_path": info["file_layout_doc_path"],
            "file_layout_doc_sha256": layout_sha256,
            "notes": info["notes"],
        }

        # Insert or update
        cursor.execute(
            """
            INSERT INTO census_pep_datasets (
                dataset_id, vintage_label, estimate_type, revision_status,
                uncertainty_level, year_range_start, year_range_end,
                source_file_path, source_file_sha256, extracted_by,
                methodology_doc_path, methodology_doc_sha256,
                file_layout_doc_path, file_layout_doc_sha256, notes
            ) VALUES (
                %(dataset_id)s, %(vintage_label)s, %(estimate_type)s, %(revision_status)s,
                %(uncertainty_level)s, %(year_range_start)s, %(year_range_end)s,
                %(source_file_path)s, %(source_file_sha256)s, %(extracted_by)s,
                %(methodology_doc_path)s, %(methodology_doc_sha256)s,
                %(file_layout_doc_path)s, %(file_layout_doc_sha256)s, %(notes)s
            )
            ON CONFLICT (dataset_id) DO UPDATE SET
                vintage_label = EXCLUDED.vintage_label,
                estimate_type = EXCLUDED.estimate_type,
                revision_status = EXCLUDED.revision_status,
                uncertainty_level = EXCLUDED.uncertainty_level,
                year_range_start = EXCLUDED.year_range_start,
                year_range_end = EXCLUDED.year_range_end,
                source_file_path = EXCLUDED.source_file_path,
                source_file_sha256 = EXCLUDED.source_file_sha256,
                extraction_timestamp = NOW(),
                extracted_by = EXCLUDED.extracted_by,
                methodology_doc_path = EXCLUDED.methodology_doc_path,
                methodology_doc_sha256 = EXCLUDED.methodology_doc_sha256,
                file_layout_doc_path = EXCLUDED.file_layout_doc_path,
                file_layout_doc_sha256 = EXCLUDED.file_layout_doc_sha256,
                notes = EXCLUDED.notes
        """,
            dataset_data,
        )

        print(f"  ✓ Inserted/updated dataset: {info['dataset_id']} ({vintage_name})")

    conn.commit()
    print(f"  Total datasets: {len(vintages)}")


def load_state_totals() -> pd.DataFrame:
    """
    Load state-level migration totals for validation.

    Returns:
        DataFrame with columns: year, NETMIG, INTERNATIONALMIG, DOMESTICMIG
    """
    print(f"\n{'=' * 70}")
    print("LOADING STATE-LEVEL TOTALS FOR VALIDATION")
    print(f"  Source: {STATE_FILE}")

    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State-level PEP file not found: {STATE_FILE}\n"
            "This file is required for validation. Please ensure it exists."
        )

    df = pd.read_csv(STATE_FILE)

    # Filter for North Dakota only
    df = df[df["NAME"] == "North Dakota"].copy()

    # Select relevant columns
    df = df[["year", "NETMIG", "INTERNATIONALMIG", "DOMESTICMIG"]].copy()

    # Convert year to integer
    df["year"] = df["year"].astype(int)

    print(f"  ✓ Loaded {len(df)} years ({df['year'].min()}-{df['year'].max()})")

    return df


def extract_vintage(vintage_name: str, vintage_info: dict) -> pd.DataFrame:
    """
    Extract and reshape migration data from a single vintage file.

    Args:
        vintage_name: Name identifier for the vintage
        vintage_info: Dictionary containing file path, years, metadata

    Returns:
        Long-format DataFrame with columns:
        - year, state_fips, county_fips, state_name, county_name
        - netmig, intl_mig, domestic_mig, residual
        - dataset_id, estimate_type, revision_status, uncertainty_level
    """
    file_path = vintage_info["file"]
    years = vintage_info["years"]
    dataset_id = vintage_info["dataset_id"]

    print(f"\n{'=' * 70}")
    print(f"EXTRACTING VINTAGE: {vintage_name}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Description: {vintage_info['description']}")
    print(f"  File: {file_path}")
    print(f"  Years: {min(years)}-{max(years)}")

    if not file_path.exists():
        raise FileNotFoundError(f"Vintage file not found: {file_path}")

    # Load full vintage file
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter for North Dakota counties only (COUNTY != '000' excludes state summary)
    df_nd = df[(df["STATE"] == ND_FIPS) & (df["COUNTY"] != "000")].copy()
    print(f"  Filtered to {len(df_nd)} ND counties")

    if len(df_nd) == 0:
        raise ValueError(f"No North Dakota counties found in {file_path}")

    # Store geographic identifiers
    geo_cols = ["STATE", "COUNTY", "STNAME", "CTYNAME"]
    geo_df = df_nd[geo_cols].copy()

    # Extract migration columns for each year
    records = []

    for year in years:
        year_str = str(year)

        # Column names
        netmig_col = f"NETMIG{year_str}"
        intl_col = f"INTERNATIONALMIG{year_str}"
        dom_col = f"DOMESTICMIG{year_str}"
        residual_col = f"RESIDUAL{year_str}"

        # Check if columns exist
        if netmig_col not in df_nd.columns:
            print(f"  WARNING: {netmig_col} not found, skipping year {year}")
            continue

        # Extract data for this year
        year_data = geo_df.copy()
        year_data["year"] = year
        year_data["netmig"] = pd.to_numeric(df_nd[netmig_col], errors="coerce")
        year_data["intl_mig"] = (
            pd.to_numeric(df_nd[intl_col], errors="coerce") if intl_col in df_nd.columns else np.nan
        )
        year_data["domestic_mig"] = (
            pd.to_numeric(df_nd[dom_col], errors="coerce") if dom_col in df_nd.columns else np.nan
        )
        year_data["residual"] = (
            pd.to_numeric(df_nd[residual_col], errors="coerce")
            if residual_col in df_nd.columns
            else np.nan
        )

        # Add metadata
        year_data["dataset_id"] = dataset_id
        year_data["estimate_type"] = vintage_info["estimate_type"]
        year_data["revision_status"] = vintage_info["revision_status"]
        year_data["uncertainty_level"] = vintage_info["uncertainty_level"]

        records.append(year_data)

    # Combine all years
    result = pd.concat(records, ignore_index=True)

    print(f"  ✓ Extracted {len(result):,} county-year observations")
    print(f"    Missing NETMIG: {result['netmig'].isna().sum()}")
    print(f"    Missing INTL_MIG: {result['intl_mig'].isna().sum()}")
    print(f"    Missing DOMESTIC_MIG: {result['domestic_mig'].isna().sum()}")
    print(f"    Missing RESIDUAL: {result['residual'].isna().sum()}")

    return result


def validate_hierarchical_consistency(
    county_df: pd.DataFrame, state_df: pd.DataFrame, tolerance: float = 0.01
) -> tuple[bool, pd.DataFrame]:
    """
    Validate that county-level net migration sums to state totals within tolerance.

    Args:
        county_df: County-level data with 'year', 'netmig', 'dataset_id'
        state_df: State-level data with 'year' and 'NETMIG'
        tolerance: Acceptable relative error (default 1%)

    Returns:
        Tuple of (all_valid: bool, comparison_df: DataFrame)
    """
    print(f"\n{'=' * 70}")
    print("VALIDATION: Hierarchical Consistency Check")
    print(f"  Tolerance: {tolerance * 100:.1f}%")

    # Sum county-level data by year and dataset
    county_sums = county_df.groupby(["year", "dataset_id"])["netmig"].sum().reset_index()
    county_sums = county_sums.rename(columns={"netmig": "county_sum"})

    # Merge with state totals
    comparison = state_df[["year", "NETMIG"]].merge(county_sums, on="year", how="outer")
    comparison = comparison.rename(columns={"NETMIG": "state_total"})

    # Calculate differences
    comparison["difference"] = comparison["county_sum"] - comparison["state_total"]
    comparison["abs_diff"] = comparison["difference"].abs()
    comparison["pct_diff"] = (comparison["difference"] / comparison["state_total"].abs()) * 100
    comparison["abs_pct_diff"] = comparison["pct_diff"].abs()

    # Flag validation failures
    comparison["passed"] = comparison["abs_pct_diff"] <= (tolerance * 100)

    # Summary statistics
    print(f"\n  Years covered: {comparison['year'].min()}-{comparison['year'].max()}")
    print(f"  Total years: {len(comparison)}")
    print(f"  Years passing validation: {comparison['passed'].sum()}")
    print(f"  Years failing validation: {(~comparison['passed']).sum()}")

    # Detailed failure report
    failures = comparison[~comparison["passed"]]
    if len(failures) > 0:
        print(f"\n  VALIDATION FAILURES ({len(failures)} years):")
        for _, row in failures.iterrows():
            print(
                f"    Year {int(row['year'])} [{row['dataset_id']}]: "
                f"County sum = {row['county_sum']:,.0f}, "
                f"State total = {row['state_total']:,.0f}, "
                f"Diff = {row['difference']:,.0f} ({row['pct_diff']:.2f}%)"
            )

    all_valid = bool(comparison["passed"].all())

    return all_valid, comparison


def populate_validation_table(conn, validation_df: pd.DataFrame):
    """
    Populate census_pep_validation table with hierarchical validation results.

    Args:
        conn: psycopg2 connection
        validation_df: DataFrame with validation results by year/dataset
    """
    print(f"\n{'=' * 70}")
    print("POPULATING METADATA: census_pep_validation")

    cursor = conn.cursor()

    # Prepare data for insertion
    validation_data = []
    for _, row in validation_df.iterrows():
        if pd.notna(row.get("dataset_id")):
            validation_data.append(
                (
                    int(row["year"]),
                    row["dataset_id"],
                    "hierarchical_consistency",
                    bool(row["passed"]),
                    float(row["county_sum"]) if pd.notna(row["county_sum"]) else None,
                    float(row["state_total"]) if pd.notna(row["state_total"]) else None,
                    float(row["difference"]) if pd.notna(row["difference"]) else None,
                    float(row["pct_diff"]) if pd.notna(row["pct_diff"]) else None,
                    f"County sum vs. state total: {row['pct_diff']:.2f}% difference",
                )
            )

    # Delete existing validations for these years/datasets
    cursor.execute("""
        DELETE FROM census_pep_validation
        WHERE validation_type = 'hierarchical_consistency'
    """)

    # Insert new validation results
    execute_values(
        cursor,
        """
        INSERT INTO census_pep_validation (
            year, dataset_id, validation_type, passed,
            county_sum, state_total, absolute_difference, percent_difference, notes
        ) VALUES %s
    """,
        validation_data,
    )

    conn.commit()
    print(f"  ✓ Inserted {len(validation_data)} validation records")


def populate_timeseries_rules(conn):
    """
    Populate census_pep_timeseries_rules table with explicit preference rules.

    Handles the 2020 overlap: prefer 2020-2024 vintage for year 2020.
    """
    print(f"\n{'=' * 70}")
    print("POPULATING METADATA: census_pep_timeseries_rules")

    cursor = conn.cursor()

    # Define rules for each year
    # 2000-2009: only one source
    rules: list[tuple[int, str, str | None, str]] = [
        (year, "co-est2009-alldata", None, f"Only source for year {year}")
        for year in range(2000, 2010)
    ]

    # 2010-2019: only one source (2010-2019 postcensal)
    rules.extend(
        (
            year,
            "co-est2019-alldata",
            None,
            f"Only components source for year {year} (intercensal population exists but not components)",
        )
        for year in range(2010, 2020)
    )

    # 2020: OVERLAP - prefer 2020-2024
    rules.append(
        (
            2020,
            "co-est2024-alldata",
            "co-est2019-alldata",
            "Year 2020 appears in both co-est2019-alldata and co-est2024-alldata. Prefer co-est2024-alldata for continuous series alignment and 2020 Census base integration.",
        )
    )

    # 2021-2024: only one source
    rules.extend(
        (year, "co-est2024-alldata", None, f"Only source for year {year}")
        for year in range(2021, 2025)
    )

    # Delete existing rules
    cursor.execute("DELETE FROM census_pep_timeseries_rules")

    # Insert new rules
    execute_values(
        cursor,
        """
        INSERT INTO census_pep_timeseries_rules (
            year, preferred_dataset_id, alternative_dataset_id, rationale
        ) VALUES %s
    """,
        rules,
    )

    conn.commit()
    print(f"  ✓ Inserted {len(rules)} time series rules (2000-2024)")
    print("  ✓ Year 2020 overlap handled: prefer co-est2024-alldata")


def assess_data_quality(row, validation_df: pd.DataFrame) -> str:
    """
    Assess data quality score for an observation.

    Args:
        row: DataFrame row with year and dataset_id
        validation_df: Validation results

    Returns:
        'pass', 'warning', or 'fail'
    """
    # Check if this year/dataset passed validation
    mask = (validation_df["year"] == row["year"]) & (
        validation_df["dataset_id"] == row["dataset_id"]
    )
    if mask.any():
        val_row = validation_df[mask].iloc[0]
        if val_row["passed"]:
            return "pass"
        elif abs(val_row["pct_diff"]) < 5.0:  # Within 5%
            return "warning"
        else:
            return "fail"

    # If no validation record, assume pass
    return "pass"


def populate_migration_table(
    conn, county_df: pd.DataFrame, validation_df: pd.DataFrame, timeseries_rules: dict[int, str]
):
    """
    Populate census_pep_county_migration table with observation-level data and metadata.

    Args:
        conn: psycopg2 connection
        county_df: Combined county migration data
        validation_df: Validation results
        timeseries_rules: Dict mapping year to preferred dataset_id
    """
    print(f"\n{'=' * 70}")
    print("POPULATING METADATA: census_pep_county_migration")

    # Add data quality scores
    county_df["data_quality_score"] = county_df.apply(
        lambda row: assess_data_quality(row, validation_df), axis=1
    )

    # Add is_preferred_estimate flag based on timeseries rules
    county_df["is_preferred_estimate"] = county_df.apply(
        lambda row: timeseries_rules.get(row["year"]) == row["dataset_id"], axis=1
    )

    # Create geoid
    county_df["geoid"] = county_df["STATE"] + county_df["COUNTY"]

    cursor = conn.cursor()

    # Prepare data for insertion
    migration_data = []
    for _, row in county_df.iterrows():
        validation_note = None
        if row["data_quality_score"] != "pass":
            mask = (validation_df["year"] == row["year"]) & (
                validation_df["dataset_id"] == row["dataset_id"]
            )
            if mask.any():
                val_row = validation_df[mask].iloc[0]
                validation_note = f"Hierarchical validation: {val_row['pct_diff']:.2f}% difference"

        migration_data.append(
            (
                row["geoid"],
                int(row["year"]),
                row["STATE"],
                row["COUNTY"],
                row["CTYNAME"],
                float(row["netmig"]) if pd.notna(row["netmig"]) else None,
                float(row["intl_mig"]) if pd.notna(row["intl_mig"]) else None,
                float(row["domestic_mig"]) if pd.notna(row["domestic_mig"]) else None,
                float(row["residual"]) if pd.notna(row["residual"]) else None,
                row["dataset_id"],
                row["estimate_type"],
                row["revision_status"],
                row["uncertainty_level"],
                row["data_quality_score"],
                validation_note,
                bool(row["is_preferred_estimate"]),
            )
        )

    # Delete existing data
    cursor.execute("DELETE FROM census_pep_county_migration")

    # Insert new data
    execute_values(
        cursor,
        """
        INSERT INTO census_pep_county_migration (
            geoid, year, state_fips, county_fips, county_name,
            netmig, intl_mig, domestic_mig, residual,
            dataset_id, estimate_type, revision_status, uncertainty_level,
            data_quality_score, validation_notes, is_preferred_estimate
        ) VALUES %s
    """,
        migration_data,
    )

    conn.commit()
    print(f"  ✓ Inserted {len(migration_data):,} county-year observations")
    print(f"  ✓ Preferred estimates: {county_df['is_preferred_estimate'].sum():,}")
    print(
        f"  ✓ Quality scores: pass={sum(1 for x in migration_data if x[13] == 'pass')}, "
        f"warning={sum(1 for x in migration_data if x[13] == 'warning')}, "
        f"fail={sum(1 for x in migration_data if x[13] == 'fail')}"
    )


def populate_extraction_log(
    conn, extraction_id: str, combined_df: pd.DataFrame, validation_passed: bool, output_file: Path
):
    """
    Populate census_pep_extraction_log table with extraction metadata.

    Args:
        conn: psycopg2 connection
        extraction_id: UUID for this extraction
        combined_df: Combined data DataFrame
        validation_passed: Whether hierarchical validation passed
        output_file: Path to output parquet file
    """
    print(f"\n{'=' * 70}")
    print("POPULATING METADATA: census_pep_extraction_log")

    cursor = conn.cursor()

    # Count rows by dataset
    rows_by_dataset = combined_df.groupby("dataset_id").size().to_dict()

    # Compute output file checksum
    output_sha256 = compute_file_sha256(output_file) if output_file.exists() else None

    for dataset_id, row_count in rows_by_dataset.items():
        cursor.execute(
            """
            INSERT INTO census_pep_extraction_log (
                extraction_id, dataset_id, script_name, script_version,
                configuration_yaml, rows_extracted, validation_passed,
                output_file_path, output_file_sha256, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """,
            (
                extraction_id,
                dataset_id,
                "extract_pep_county_migration_with_metadata.py",
                SCRIPT_VERSION,
                json.dumps(
                    VINTAGES[next(k for k, v in VINTAGES.items() if v["dataset_id"] == dataset_id)],
                    default=str,
                    indent=2,
                ),
                int(row_count),
                bool(validation_passed),
                str(output_file),
                output_sha256,
                f"Extraction run at {datetime.now(tz=datetime.now().astimezone().tzinfo).isoformat()}",
            ),
        )

    conn.commit()
    print(f"  ✓ Logged extraction for {len(rows_by_dataset)} datasets")
    print(f"  ✓ Extraction ID: {extraction_id}")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the extracted data.

    Args:
        df: Harmonized county-year DataFrame

    Returns:
        DataFrame with summary statistics by period/vintage
    """
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")

    stats_records = []

    # Determine county column name (case-insensitive)
    county_col = (
        "county_fips"
        if "county_fips" in df.columns
        else "COUNTY"
        if "COUNTY" in df.columns
        else None
    )
    if not county_col:
        raise ValueError("Could not find county column in DataFrame")

    # Overall statistics (preferred estimates only)
    df_pref = df[df["is_preferred_estimate"]].copy()
    stats_records.append(
        {
            "period": "Full Period (Preferred)",
            "years": f"{df_pref['year'].min()}-{df_pref['year'].max()}",
            "n_counties": df_pref[county_col].nunique(),
            "n_years": df_pref["year"].nunique(),
            "n_observations": len(df_pref),
            "mean_netmig": df_pref["netmig"].mean(),
            "median_netmig": df_pref["netmig"].median(),
            "total_netmig": df_pref["netmig"].sum(),
            "min_netmig": df_pref["netmig"].min(),
            "max_netmig": df_pref["netmig"].max(),
            "pct_negative": (df_pref["netmig"] < 0).mean() * 100,
        }
    )

    # By dataset
    for dataset_id in df["dataset_id"].unique():
        ddf = df[df["dataset_id"] == dataset_id]
        stats_records.append(
            {
                "period": dataset_id,
                "years": f"{ddf['year'].min()}-{ddf['year'].max()}",
                "n_counties": ddf[county_col].nunique(),
                "n_years": ddf["year"].nunique(),
                "n_observations": len(ddf),
                "mean_netmig": ddf["netmig"].mean(),
                "median_netmig": ddf["netmig"].median(),
                "total_netmig": ddf["netmig"].sum(),
                "min_netmig": ddf["netmig"].min(),
                "max_netmig": ddf["netmig"].max(),
                "pct_negative": (ddf["netmig"] < 0).mean() * 100,
            }
        )

    # By regime (using preferred estimates)
    regimes = [
        ("Pre-Bakken", 2000, 2010),
        ("Bakken Boom", 2011, 2015),
        ("Bust + COVID", 2016, 2021),
        ("Recovery", 2022, 2024),
    ]

    for regime_name, start_year, end_year in regimes:
        rdf = df_pref[(df_pref["year"] >= start_year) & (df_pref["year"] <= end_year)]
        if len(rdf) > 0:
            stats_records.append(
                {
                    "period": regime_name,
                    "years": f"{start_year}-{end_year}",
                    "n_counties": rdf[county_col].nunique(),
                    "n_years": rdf["year"].nunique(),
                    "n_observations": len(rdf),
                    "mean_netmig": rdf["netmig"].mean(),
                    "median_netmig": rdf["netmig"].median(),
                    "total_netmig": rdf["netmig"].sum(),
                    "min_netmig": rdf["netmig"].min(),
                    "max_netmig": rdf["netmig"].max(),
                    "pct_negative": (rdf["netmig"] < 0).mean() * 100,
                }
            )

    stats_df = pd.DataFrame(stats_records)

    # Print summary
    print(f"\n{stats_df.to_string(index=False)}")

    return stats_df


def main():
    """Main extraction and metadata population pipeline."""

    print("=" * 70)
    print("CENSUS PEP COUNTY MIGRATION EXTRACTION + METADATA")
    print("ADR-035 Phase 1 with Enhanced Metadata Tracking")
    print("=" * 70)

    extraction_id = str(uuid.uuid4())

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get database connection
    try:
        conn = get_db_connection()
        print("✓ Connected to database")
    except Exception as e:
        print(f"\nERROR: Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Step 1: Populate datasets metadata table
        populate_datasets_table(conn, VINTAGES)

        # Step 2: Load state-level totals for validation
        state_df = load_state_totals()

        # Step 3: Extract data from each vintage
        vintage_dfs = []
        for vintage_name, vintage_info in VINTAGES.items():
            try:
                vdf = extract_vintage(vintage_name, vintage_info)
                vintage_dfs.append(vdf)
            except Exception as e:
                print(f"\n  ERROR processing {vintage_name}: {e}")
                raise

        # Step 4: Combine all vintages
        print(f"\n{'=' * 70}")
        print("HARMONIZATION: Combining vintages")

        combined_df = pd.concat(vintage_dfs, ignore_index=True)
        print(f"  Total observations: {len(combined_df):,}")
        print(f"  Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
        print(f"  Counties: {combined_df['COUNTY'].nunique()}")
        print(f"  Datasets: {combined_df['dataset_id'].nunique()}")

        # Step 5: Validate hierarchical consistency
        all_valid, validation_df = validate_hierarchical_consistency(combined_df, state_df)

        if not all_valid:
            print("\n  WARNING: Some years failed hierarchical validation!")
            print("  This is documented in metadata tables.")
        else:
            print("\n  ✓ All years passed hierarchical validation!")

        # Step 6: Populate validation table
        populate_validation_table(conn, validation_df)

        # Step 7: Populate timeseries rules
        populate_timeseries_rules(conn)

        # Step 8: Get timeseries rules for marking preferred estimates
        cursor = conn.cursor()
        cursor.execute("SELECT year, preferred_dataset_id FROM census_pep_timeseries_rules")
        timeseries_rules = {row[0]: row[1] for row in cursor.fetchall()}

        # Step 9: Save flat file outputs (for backward compatibility)
        output_file = OUTPUT_DIR / "pep_county_components_2000_2024.parquet"

        # Rename columns for consistency
        combined_df_renamed = combined_df.rename(
            columns={
                "STATE": "state_fips",
                "COUNTY": "county_fips",
                "STNAME": "state_name",
                "CTYNAME": "county_name",
            }
        )

        # Create geoid
        combined_df_renamed["geoid"] = (
            combined_df_renamed["state_fips"] + combined_df_renamed["county_fips"]
        )

        # Add is_preferred_estimate flag
        combined_df_renamed["is_preferred_estimate"] = combined_df_renamed.apply(
            lambda row: timeseries_rules.get(row["year"]) == row["dataset_id"], axis=1
        )

        # Sort by year and county
        combined_df_renamed = combined_df_renamed.sort_values(["year", "geoid"]).reset_index(
            drop=True
        )

        # Save parquet
        combined_df_renamed.to_parquet(output_file, index=False, compression="gzip")
        print(f"\n{'=' * 70}")
        print("SAVING OUTPUTS")
        print(f"  ✓ Saved harmonized data: {output_file}")
        print(f"    Rows: {len(combined_df_renamed):,}")
        print(f"    Size: {output_file.stat().st_size / 1024:.1f} KB")

        # Save CSV
        csv_file = OUTPUT_DIR / "pep_county_components_2000_2024.csv"
        combined_df_renamed.to_csv(csv_file, index=False)
        print(f"  ✓ Saved CSV version: {csv_file}")

        # Step 10: Populate migration table
        populate_migration_table(conn, combined_df, validation_df, timeseries_rules)

        # Step 11: Populate extraction log
        populate_extraction_log(conn, extraction_id, combined_df, all_valid, output_file)

        # Step 12: Update preferred estimates using helper function
        print(f"\n{'=' * 70}")
        print("FINALIZING: Applying time series construction rules")
        cursor = conn.cursor()
        cursor.execute("SELECT update_preferred_estimates()")
        rows_updated = cursor.fetchone()[0]
        conn.commit()
        print(f"  ✓ Updated {rows_updated} rows with is_preferred_estimate flags")

        # Step 13: Generate summary statistics
        stats_df = generate_summary_statistics(combined_df_renamed)

        stats_file = OUTPUT_DIR / "pep_county_summary_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"  ✓ Saved summary statistics: {stats_file}")

        # Step 14: Final summary
        print(f"\n{'=' * 70}")
        print("EXTRACTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"\nHierarchical Validation: {'PASS ✓' if all_valid else 'PARTIAL ⚠'}")
        print(f"Total County-Year Observations: {len(combined_df_renamed):,}")
        print(f"Preferred Estimates: {combined_df_renamed['is_preferred_estimate'].sum():,}")
        print(
            f"Time Coverage: {combined_df_renamed['year'].min()}-{combined_df_renamed['year'].max()} ({combined_df_renamed['year'].nunique()} years)"
        )
        print(f"Counties: {combined_df_renamed['county_fips'].nunique()}")
        print("\nDatabase: census_popest")
        print("  - 5 metadata tables populated")
        print(f"  - {len(combined_df_renamed):,} observations with full metadata")
        print(f"  - {validation_df['dataset_id'].notna().sum()} validation records")
        print(f"  - {len(timeseries_rules)} time series rules")
        print(f"  - Extraction ID: {extraction_id}")
        print("\nFlat file outputs (backward compatible):")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
        print(f"  - {stats_file}")
        print("\nNext steps: Phase 2 - Regime Analysis")
        print("  See: docs/governance/adrs/035-migration-data-source-census-pep.md")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {e}")
        print(f"{'=' * 70}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
