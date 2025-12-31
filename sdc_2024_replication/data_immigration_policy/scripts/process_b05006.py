#!/usr/bin/env python3
# mypy: ignore-errors
"""
Process Census Bureau ACS Table B05006: Place of Birth for Foreign-Born Population
Transform wide format to long format with clean column names.
Calculate North Dakota's share of foreign-born by origin compared to national.

RESOLVED (P3.05 - 2025-12-31):
==============================
Fixed variable-depth hierarchy parsing. Census B05006 has different depths:
- Asia: Total!!Asia!!South Central Asia!!India (4 levels)
- Latin America: Total!!Americas!!Latin America!!Central America!!Mexico (5 levels)

The fix detects intermediate categories (Central America, South America, Caribbean)
and correctly assigns the 5th-level element as the country.

RESOLVED (P3.06 - 2025-12-31):
==============================
Fixed year-varying variable codes. Census Bureau changed B05006 variable codes between
ACS years. For example:
- In 2015-2019: B05006_059E = India
- In 2023: B05006_059E = Bhutan, B05006_060E = India (they added Bhutan)

The fix loads year-specific label files (b05006_variable_labels_{year}.json) when
processing each row, with caching to avoid repeated file reads.
"""

import json
import re
from pathlib import Path

import pandas as pd

# Configuration - Use project-level data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/

# Input: raw census data
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "immigration" / "census_foreign_born"

# Output: analysis goes to project-level processed directory
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"

# Cache for year-specific labels to avoid repeated file reads
_labels_cache: dict[int, dict] = {}


def load_labels_for_year(year: int) -> dict:
    """Load variable labels for a specific year with caching.

    Census Bureau variable codes can change between ACS years, so we need
    year-specific label files. Falls back to the most recent labels if
    year-specific file not found.

    Args:
        year: The ACS year to load labels for

    Returns:
        Dictionary mapping variable codes to their labels
    """
    if year in _labels_cache:
        return _labels_cache[year]

    # Try year-specific label file first
    year_labels_file = SOURCE_DIR / f"b05006_variable_labels_{year}.json"
    if year_labels_file.exists():
        with open(year_labels_file) as f:
            labels = json.load(f)
        _labels_cache[year] = labels
        return labels

    # Fall back to default labels file (most recent year)
    default_labels_file = SOURCE_DIR / "b05006_variable_labels.json"
    if default_labels_file.exists():
        with open(default_labels_file) as f:
            labels = json.load(f)
        _labels_cache[year] = labels
        print(f"  Warning: Using default labels for year {year} (year-specific file not found)")
        return labels

    # No labels found
    print(f"  Error: No labels found for year {year}")
    return {}


def parse_label(label: str) -> dict:
    """Parse a B05006 variable label into components.

    Handles variable-depth hierarchies in Census B05006:
    - Standard 4-level: Total!!Region!!SubRegion!!Country
    - Latin America 5-level: Total!!Region!!Latin America!!IntermediateCategory!!Country

    Intermediate categories in Latin America (Central America, South America, Caribbean)
    are detected and the hierarchy is adjusted so the 5th level is correctly treated
    as the country.

    NOTE: Census Bureau changed label format around 2019, adding colons after hierarchy
    elements (e.g., "Total:!!Europe:!!" instead of "Total!!Europe!!"). This function
    strips trailing colons to handle both formats.
    """
    # Remove "Estimate!!" or "Margin of Error!!" prefix
    if label.startswith("Estimate!!"):
        is_estimate = True
        label = label.replace("Estimate!!", "")
    elif label.startswith("Margin of Error!!"):
        is_estimate = False
        label = label.replace("Margin of Error!!", "")
    else:
        return None

    # Parse hierarchy: Total!!Region!!Subregion!!Country...
    parts = label.split("!!")

    # Strip trailing colons from each part (Census added colons in 2019+)
    parts = [p.rstrip(":") for p in parts]

    if not parts or parts[0] != "Total":
        return None

    result = {
        "is_estimate": is_estimate,
        "region": None,
        "sub_region": None,
        "country": None,
        "detail": None,
        "level": "total",
    }

    # Intermediate categories that appear between sub_region and country in Latin America
    # These create a 5-level hierarchy instead of the standard 4-level
    latin_america_intermediate = {"Central America", "South America", "Caribbean"}

    if len(parts) == 1:
        result["level"] = "total"
    elif len(parts) == 2:
        result["region"] = parts[1]
        result["level"] = "region"
    elif len(parts) == 3:
        result["region"] = parts[1]
        result["sub_region"] = parts[2]
        result["level"] = "sub_region"
    elif len(parts) == 4:
        result["region"] = parts[1]
        result["sub_region"] = parts[2]
        # Check if this is an intermediate category (sub-sub-region)
        if parts[3] in latin_america_intermediate:
            # e.g., Total!!Americas!!Latin America!!Central America
            # This is a sub-sub-region total, not a country
            result["sub_region"] = f"{parts[2]} - {parts[3]}"
            result["level"] = "sub_region"
        else:
            result["country"] = parts[3]
            result["level"] = "country"
    elif len(parts) == 5:
        result["region"] = parts[1]
        # Check if parts[3] is an intermediate category
        if parts[3] in latin_america_intermediate:
            # e.g., Total!!Americas!!Latin America!!Central America!!Mexico
            # Combine sub_region with intermediate category for full context
            result["sub_region"] = f"{parts[2]} - {parts[3]}"
            result["country"] = parts[4]
            result["level"] = "country"
        else:
            # Standard 5-level with detail
            result["sub_region"] = parts[2]
            result["country"] = parts[3]
            result["detail"] = parts[4]
            result["level"] = "detail"
    elif len(parts) >= 6:
        result["region"] = parts[1]
        # Check if parts[3] is an intermediate category
        if parts[3] in latin_america_intermediate:
            # e.g., Total!!Americas!!Latin America!!Central America!!Mexico!!SomeDetail
            result["sub_region"] = f"{parts[2]} - {parts[3]}"
            result["country"] = parts[4]
            result["detail"] = "!!".join(parts[5:])
            result["level"] = "detail"
        else:
            # Standard 6+ level
            result["sub_region"] = parts[2]
            result["country"] = parts[3]
            result["detail"] = "!!".join(parts[4:])
            result["level"] = "detail"

    return result


def load_and_process_data():
    """Load raw data and transform to long format.

    Uses year-specific variable labels to correctly map variable codes to
    countries/regions, since Census Bureau codes can change between ACS years.
    """

    # Load combined data
    data_file = SOURCE_DIR / "b05006_states_all_years.csv"
    df = pd.read_csv(data_file, dtype=str)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify estimate columns (end with E, not EA) and MOE columns (end with M, not MA)
    estimate_cols = [c for c in df.columns if re.match(r"B05006_\d{3}E$", c)]
    moe_cols = [c for c in df.columns if re.match(r"B05006_\d{3}M$", c)]

    print(f"Found {len(estimate_cols)} estimate columns, {len(moe_cols)} MOE columns")

    # Create mapping of estimate col to MOE col
    est_to_moe = {e: e[:-1] + "M" for e in estimate_cols}

    # Identify unique years to pre-load and parse labels
    years_in_data = df["year"].dropna().unique()
    print(f"Years in data: {sorted(int(y) for y in years_in_data)}")

    # Pre-load and parse labels for all years (with caching)
    var_info_by_year = {}
    for year_str in years_in_data:
        year = int(year_str)
        labels = load_labels_for_year(year)
        var_info = {}
        for var_id, label in labels.items():
            parsed = parse_label(label)
            if parsed:
                var_info[var_id] = parsed
        var_info_by_year[year] = var_info
        print(f"  Loaded {len(var_info)} variable labels for year {year}")

    # Melt to long format
    id_vars = ["NAME", "state", "year", "GEO_ID"]
    id_vars = [c for c in id_vars if c in df.columns]

    records = []
    for _, row in df.iterrows():
        state_name = row.get("NAME", "Unknown")
        state_fips = row.get("state", "")
        year_str = row.get("year", "")

        # Get year-specific variable info
        try:
            year = int(year_str) if year_str else None
        except (ValueError, TypeError):
            year = None

        var_info = var_info_by_year.get(year, {}) if year else {}

        for est_col in estimate_cols:
            moe_col = est_to_moe.get(est_col)
            est_val = row.get(est_col)
            moe_val = row.get(moe_col) if moe_col else None

            # Get variable info for this year's labels
            info = var_info.get(est_col, {})

            # Skip if no info
            if not info:
                continue

            # Convert values to numeric
            try:
                est_val = int(est_val) if est_val and est_val not in ["null", "N"] else None
            except (ValueError, TypeError):
                est_val = None

            try:
                moe_val = int(moe_val) if moe_val and moe_val not in ["null", "N"] else None
            except (ValueError, TypeError):
                moe_val = None

            records.append(
                {
                    "year": year,
                    "state_fips": state_fips,
                    "state_name": state_name,
                    "variable": est_col,
                    "region": info.get("region"),
                    "sub_region": info.get("sub_region"),
                    "country": info.get("country"),
                    "detail": info.get("detail"),
                    "level": info.get("level"),
                    "foreign_born_pop": est_val,
                    "margin_of_error": moe_val,
                }
            )

    long_df = pd.DataFrame(records)

    # Clean column names
    long_df.columns = [c.lower().replace(" ", "_") for c in long_df.columns]

    print(f"Created long format with {len(long_df)} rows")

    return long_df


def calculate_nd_share(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate North Dakota's share of foreign-born by origin compared to national."""

    # Filter to valid data
    df_valid = df[df["foreign_born_pop"].notna()].copy()

    # Get national totals (state_fips could be various for US total - we'll sum all states)
    national = (
        df_valid.groupby(["year", "region", "sub_region", "country", "level", "variable"])
        .agg(
            {
                "foreign_born_pop": "sum",
                "margin_of_error": lambda x: (x.dropna() ** 2).sum() ** 0.5
                if len(x.dropna()) > 0
                else None,  # Propagate MOE
            }
        )
        .reset_index()
    )
    national.columns = [
        "year",
        "region",
        "sub_region",
        "country",
        "level",
        "variable",
        "national_foreign_born",
        "national_moe",
    ]

    # Get North Dakota data (FIPS = 38)
    nd = df_valid[df_valid["state_fips"] == "38"].copy()
    nd = nd.rename(columns={"foreign_born_pop": "nd_foreign_born", "margin_of_error": "nd_moe"})

    # Merge
    merged = nd.merge(
        national, on=["year", "region", "sub_region", "country", "level", "variable"], how="left"
    )

    # Calculate share
    merged["nd_share_of_national"] = (
        merged["nd_foreign_born"] / merged["national_foreign_born"]
    ).where(merged["national_foreign_born"] > 0)

    return merged


def main():
    """Main processing function."""

    print("Loading and processing B05006 data...")
    long_df = load_and_process_data()

    print("\nCalculating North Dakota share of national...")
    nd_share_df = calculate_nd_share(long_df)

    # Create output directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full long format data
    full_output = ANALYSIS_DIR / "acs_foreign_born_by_state_origin.parquet"
    long_df.to_parquet(full_output, index=False)
    print(f"\nSaved full dataset: {full_output}")
    print(f"  Rows: {len(long_df)}")

    # Also save ND share analysis
    nd_output = ANALYSIS_DIR / "acs_foreign_born_nd_share.parquet"
    nd_share_df.to_parquet(nd_output, index=False)
    print(f"\nSaved ND share analysis: {nd_output}")
    print(f"  Rows: {len(nd_share_df)}")

    # Print summary stats
    print("\n=== Summary Statistics ===")

    # Years covered
    years = sorted(long_df["year"].dropna().unique())
    print(f"Years: {years[0]} - {years[-1]}")

    # States covered
    states = long_df["state_fips"].nunique()
    print(f"States/territories: {states}")

    # Regions
    regions = long_df[long_df["level"] == "region"]["region"].unique()
    print(f"Regions: {', '.join(sorted(r for r in regions if r))}")

    # ND top origins
    print("\n=== North Dakota Top 10 Foreign-Born Origins (2023) ===")
    nd_2023 = (
        nd_share_df[
            (nd_share_df["year"] == 2023) & (nd_share_df["level"].isin(["country", "sub_region"]))
        ]
        .sort_values("nd_foreign_born", ascending=False)
        .head(10)
    )

    for _, row in nd_2023.iterrows():
        origin = row["country"] or row["sub_region"]
        pop = row["nd_foreign_born"]
        share = row["nd_share_of_national"]
        print(f"  {origin}: {pop:,.0f} ({share:.4%} of national)")

    return long_df, nd_share_df


if __name__ == "__main__":
    main()
