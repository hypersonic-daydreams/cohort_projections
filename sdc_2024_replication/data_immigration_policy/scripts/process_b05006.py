#!/usr/bin/env python3
# mypy: ignore-errors
"""
Process Census Bureau ACS Table B05006: Place of Birth for Foreign-Born Population
Transform wide format to long format with clean column names.
Calculate North Dakota's share of foreign-born by origin compared to national.
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


def parse_label(label: str) -> dict:
    """Parse a B05006 variable label into components."""
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
        result["country"] = parts[3]
        result["level"] = "country"
    elif len(parts) >= 5:
        result["region"] = parts[1]
        result["sub_region"] = parts[2]
        result["country"] = parts[3]
        result["detail"] = "!!".join(parts[4:])
        result["level"] = "detail"

    return result


def load_and_process_data():
    """Load raw data and transform to long format."""

    # Load variable labels
    labels_file = SOURCE_DIR / "b05006_variable_labels.json"
    with open(labels_file) as f:
        labels = json.load(f)

    # Parse labels to identify estimate vs MOE columns and hierarchy
    var_info = {}
    for var_id, label in labels.items():
        parsed = parse_label(label)
        if parsed:
            var_info[var_id] = parsed

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

    # Melt to long format
    id_vars = ["NAME", "state", "year", "GEO_ID"]
    id_vars = [c for c in id_vars if c in df.columns]

    records = []
    for _, row in df.iterrows():
        state_name = row.get("NAME", "Unknown")
        state_fips = row.get("state", "")
        year = row.get("year", "")

        for est_col in estimate_cols:
            moe_col = est_to_moe.get(est_col)
            est_val = row.get(est_col)
            moe_val = row.get(moe_col) if moe_col else None

            # Get variable info
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
                    "year": int(year) if year else None,
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
