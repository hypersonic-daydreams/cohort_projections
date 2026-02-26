#!/usr/bin/env python3
"""
Build group quarters (GQ) population by county, 5-year age group, and sex for ND.

Created: 2026-02-23
Updated: 2026-02-26 (Phase 2 -- historical GQ estimates for migration rate correction)
ADR: 055 (Group Quarters Population Separation)
Author: Claude Code / N. Haarstad

Purpose
-------
Produce county-level GQ population files (by age and sex) for use in the
cohort-component projection pipeline:

**Phase 1 output** (2025 snapshot):
  Used by the projection engine to separate GQ from household population so
  that institutional populations are held constant instead of being eroded by
  negative migration rates.

**Phase 2 output** (historical estimates at 6 time points):
  Used by the residual migration pipeline to compute migration rates on
  household-only population, removing institutional rotation signals (dorm
  turnover, military PCS cycles) from the computed rates.

Method
------
1. Load PEP stcoreview GQ totals by county and broad age group (0-17,
   18-64, 65+). These are the authoritative GQ population counts from Census
   PEP Vintage 2025.

2. Distribute broad age groups into 5-year age groups using county-type-specific
   allocation profiles:
   - **Ages 0-17**: Uniformly distributed across 0-4, 5-9, 10-14, 15-17
     (GQ under 18 is mostly juvenile facilities; small and evenly spread).
   - **Ages 18-64 (college counties 38035, 38017, 38101, 38015)**:
     Concentrated in 18-19, 20-24, 25-29 (college dorms dominate).
   - **Ages 18-64 (other counties)**: Spread across 18-19 through 40-44
     with moderate concentration in 20-34 (correctional, group homes).
   - **Ages 65+**: Concentrated in 75-79, 80-84, 85+ (nursing facilities).

3. Distribute each 5-year age group evenly between Male and Female (the Census
   DHC PCO1 table shows near-equal sex ratios for total GQ at the state level;
   military skews male but nursing skews female, roughly balancing).

4. Scale the resulting age-sex distribution to exactly match the stcoreview
   broad-age-group totals for each county.

5. Save outputs:
   - data/processed/gq_county_age_sex_2025.parquet (Phase 1: 2025 snapshot)
   - data/processed/gq_county_age_sex_historical.parquet (Phase 2: multi-year)

Historical GQ estimation (Phase 2)
-----------------------------------
The residual migration pipeline needs GQ at 6 time points: 2000, 2005, 2010,
2015, 2020, 2024. Stcoreview provides GQ for 2020-2025 only.

- **Years 2020-2024**: Use stcoreview GQpop for each specific year.
- **Years 2000-2015**: Use 2020 GQ levels as a backward constant.

The backward-constant assumption is defensible because institutional capacity
(barracks, dorms, nursing beds) changes slowly, and the primary goal is
removing rotation signals rather than precisely tracking historical GQ.

Key design decisions
--------------------
- **No Census API call**: The Census DHC API for PCO1 at the county level is
  unreliable and frequently returns errors. Instead, we build the age-sex
  distribution from the stcoreview broad age groups using reasonable allocation
  profiles based on institutional type knowledge.
- **Equal sex split**: Without county-level sex-specific GQ data, a 50/50 split
  is the least-biased assumption. Military counties skew male (MAFB ~80% male)
  but nursing facilities skew female (~65% female); at the county level these
  partially offset.
- **College county profiles**: Grand Forks, Cass, Ward, and Burleigh have
  substantial university dormitory populations that concentrate GQ in the 18-24
  age range. Other counties have more dispersed GQ (correctional, group homes).

Inputs
------
- data/raw/population/stcoreview_v2025_nd_parsed.parquet
    PEP Vintage 2025 stcoreview with GQpop variable, broad age groups (0-17,
    18-64, 65+), and total. All 53 ND counties.

Outputs
-------
- data/processed/gq_county_age_sex_2025.parquet
    Columns: county_fips, age_group, sex, gq_population
    18 age groups x 2 sexes x 53 counties = 1,908 rows

- data/processed/gq_county_age_sex_historical.parquet
    Columns: county_fips, year, age_group, sex, gq_population
    18 age groups x 2 sexes x 53 counties x 6 years = 11,448 rows
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

# Standard 5-year age groups used by the projection engine
FIVE_YEAR_AGE_GROUPS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85+",
]

# College counties: Grand Forks, Cass, Ward, Burleigh
COLLEGE_COUNTY_FIPS = {"035", "017", "101", "015"}

# -- Allocation profiles --
# These define how each broad age group's total GQ is distributed across
# the constituent 5-year age groups. Values are relative weights (normalized
# to sum to 1.0 within each broad group).

# Ages 0-17 GQ: mostly juvenile facilities, small numbers, roughly uniform
PROFILE_0_17: dict[str, float] = {
    "0-4": 1.0,
    "5-9": 1.0,
    "10-14": 1.0,
    "15-19": 1.0,  # Only ages 15-17 are in this broad group; 18-19 is in 18-64
}

# Ages 18-64 GQ for college counties: heavy college dorm concentration
PROFILE_18_64_COLLEGE: dict[str, float] = {
    "15-19": 3.0,   # Ages 18-19 only (freshman year)
    "20-24": 5.0,   # Peak dorm/Greek housing
    "25-29": 2.0,   # Graduate student housing, some military
    "30-34": 1.0,
    "35-39": 0.5,
    "40-44": 0.3,
    "45-49": 0.2,
    "50-54": 0.2,
    "55-59": 0.2,
    "60-64": 0.2,
}

# Ages 18-64 GQ for non-college counties: correctional, group homes, military
PROFILE_18_64_OTHER: dict[str, float] = {
    "15-19": 1.0,   # Ages 18-19 only
    "20-24": 2.0,
    "25-29": 2.0,
    "30-34": 2.0,
    "35-39": 1.5,
    "40-44": 1.0,
    "45-49": 0.8,
    "50-54": 0.6,
    "55-59": 0.5,
    "60-64": 0.4,
}

# Ages 65+ GQ: nursing facilities concentrate in 75+
PROFILE_65_PLUS: dict[str, float] = {
    "65-69": 0.5,
    "70-74": 1.0,
    "75-79": 2.0,
    "80-84": 3.0,
    "85+": 4.0,
}


def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    """Normalize profile weights to sum to 1.0."""
    total = sum(profile.values())
    if total == 0:
        return profile
    return {k: v / total for k, v in profile.items()}


def load_stcoreview_gq(stcoreview_path: Path, period: str = "2025") -> pd.DataFrame:
    """
    Load GQ population from stcoreview parquet for a specific period.

    Args:
        stcoreview_path: Path to stcoreview parquet file.
        period: Period string to filter on (e.g., "2020", "2024", "2025").

    Returns:
        DataFrame with columns: county_fips, age_group, gq_total
        Filtered to the specified period, non-state counties, broad age groups.
    """
    logger.info(f"Loading stcoreview GQ data for period {period} from {stcoreview_path}")

    df = pd.read_parquet(stcoreview_path)

    # Filter to GQ variable, target period, non-state
    gq = df[
        (df["variable"] == "GQpop")
        & (df["period"] == period)
        & (df["is_state_total"] == False)  # noqa: E712
        & (df["age_group"].isin(["0-17", "18-64", "65+"]))
    ].copy()

    logger.info(f"Found {len(gq)} GQ records for {gq['county_fips'].nunique()} counties")

    # Select relevant columns
    result = gq[["county_fips", "age_group", "value"]].copy()
    result = result.rename(columns={"value": "gq_total"})
    result["gq_total"] = result["gq_total"].fillna(0.0)

    return result


def distribute_broad_to_five_year(
    county_fips: str,
    broad_age_gq: dict[str, float],
) -> list[dict[str, str | float]]:
    """
    Distribute a county's broad age group GQ totals into 5-year age groups.

    Args:
        county_fips: 3-digit county FIPS (within state)
        broad_age_gq: Dict mapping broad age group ("0-17", "18-64", "65+")
                      to GQ population total.

    Returns:
        List of dicts with keys: age_group, sex, gq_population
    """
    is_college = county_fips in COLLEGE_COUNTY_FIPS

    # Normalize allocation profiles
    profile_0_17 = _normalize_profile(PROFILE_0_17)
    profile_18_64 = _normalize_profile(
        PROFILE_18_64_COLLEGE if is_college else PROFILE_18_64_OTHER
    )
    profile_65_plus = _normalize_profile(PROFILE_65_PLUS)

    # Map broad age group to profile
    broad_to_profile = {
        "0-17": profile_0_17,
        "18-64": profile_18_64,
        "65+": profile_65_plus,
    }

    rows: list[dict[str, str | float]] = []

    for broad_group, total_gq in broad_age_gq.items():
        profile = broad_to_profile.get(broad_group, {})
        if not profile or total_gq == 0:
            continue

        for age_group, weight in profile.items():
            # Split evenly between male and female
            gq_per_sex = total_gq * weight / 2.0
            rows.append({
                "age_group": age_group,
                "sex": "Male",
                "gq_population": gq_per_sex,
            })
            rows.append({
                "age_group": age_group,
                "sex": "Female",
                "gq_population": gq_per_sex,
            })

    return rows


def build_gq_county_age_sex(
    stcoreview_path: Path,
    period: str = "2025",
) -> pd.DataFrame:
    """
    Build the full county x age_group x sex GQ population DataFrame.

    Args:
        stcoreview_path: Path to stcoreview parquet file.
        period: Period string to use from stcoreview (e.g., "2020", "2024", "2025").

    Returns:
        DataFrame with columns: county_fips (5-digit), age_group, sex, gq_population
    """
    # Load broad age group GQ from stcoreview
    gq_broad = load_stcoreview_gq(stcoreview_path, period=period)

    all_rows: list[dict[str, str | float]] = []

    # Process each county
    for county_fips_3, county_df in gq_broad.groupby("county_fips"):
        # Build broad age dict for this county
        broad_age_gq = {}
        for _, row in county_df.iterrows():
            broad_age_gq[row["age_group"]] = float(row["gq_total"])

        # Distribute to 5-year age groups
        county_rows = distribute_broad_to_five_year(
            str(county_fips_3), broad_age_gq
        )

        # Add full 5-digit FIPS
        full_fips = f"38{str(county_fips_3).zfill(3)}"
        for r in county_rows:
            r["county_fips"] = full_fips

        all_rows.extend(county_rows)

    result = pd.DataFrame(all_rows)

    # Aggregate duplicates: the "15-19" age group gets contributions from both
    # the 0-17 broad group (ages 15-17) and the 18-64 broad group (ages 18-19),
    # so we need to sum these together.
    result = result.groupby(
        ["county_fips", "age_group", "sex"], as_index=False
    ).agg({"gq_population": "sum"})

    # Ensure all counties have all age groups and both sexes
    # (some counties may have zero GQ in certain broad groups)
    all_counties = gq_broad["county_fips"].unique()
    all_combos = [
        {"county_fips": f"38{str(c).zfill(3)}", "age_group": ag, "sex": sex}
        for c in all_counties
        for ag in FIVE_YEAR_AGE_GROUPS
        for sex in ["Male", "Female"]
    ]
    complete_index = pd.DataFrame(all_combos)

    result = complete_index.merge(
        result,
        on=["county_fips", "age_group", "sex"],
        how="left",
    )
    result["gq_population"] = result["gq_population"].fillna(0.0)

    # Sort for consistency
    result = result.sort_values(
        ["county_fips", "age_group", "sex"]
    ).reset_index(drop=True)

    return result


def validate_gq_data(
    gq_df: pd.DataFrame,
    stcoreview_path: Path,
) -> bool:
    """
    Validate that GQ data matches stcoreview totals.

    Returns True if validation passes.
    """
    logger.info("Validating GQ data against stcoreview totals...")

    # Load stcoreview totals
    gq_broad = load_stcoreview_gq(stcoreview_path)

    all_ok = True

    for county_fips_3, county_broad in gq_broad.groupby("county_fips"):
        full_fips = f"38{str(county_fips_3).zfill(3)}"
        county_gq = gq_df[gq_df["county_fips"] == full_fips]

        # Sum by broad age group from the 5-year data
        total_gq = county_gq["gq_population"].sum()
        expected_total = county_broad["gq_total"].sum()

        if abs(total_gq - expected_total) > 0.1:
            logger.warning(
                f"County {full_fips}: GQ total mismatch. "
                f"Expected {expected_total:.0f}, got {total_gq:.0f}"
            )
            all_ok = False

    if all_ok:
        logger.info("Validation passed: all county GQ totals match stcoreview")

    # Log summary statistics
    state_total = gq_df["gq_population"].sum()
    n_counties = gq_df["county_fips"].nunique()
    logger.info(f"State GQ total: {state_total:,.0f}")
    logger.info(f"Counties: {n_counties}")

    # Log key counties
    for fips, name in [
        ("38035", "Grand Forks"),
        ("38017", "Cass"),
        ("38101", "Ward"),
        ("38015", "Burleigh"),
    ]:
        county_total = gq_df[gq_df["county_fips"] == fips]["gq_population"].sum()
        logger.info(f"  {name} ({fips}): GQ = {county_total:,.0f}")

    return all_ok


def build_historical_gq(stcoreview_path: Path) -> pd.DataFrame:
    """
    Build GQ estimates at all 6 residual migration time points.

    Time points: 2000, 2005, 2010, 2015, 2020, 2024.

    - Years 2020 and 2024: Use stcoreview GQpop for those specific years.
    - Years 2000, 2005, 2010, 2015: Use 2020 GQ levels (backward constant).

    The backward-constant assumption is defensible because institutional capacity
    changes slowly (military bases, nursing homes, college dorms are stable over
    5-10 year windows). The primary goal is removing institutional rotation from
    migration rates, not precisely tracking historical GQ changes.

    Args:
        stcoreview_path: Path to stcoreview parquet file.

    Returns:
        DataFrame with columns: county_fips, year, age_group, sex, gq_population
    """
    logger.info("Building historical GQ estimates for 6 time points")

    # Years that have actual stcoreview data
    stcoreview_years = {2020: "2020", 2024: "2024"}
    # Years that use 2020 as a backward constant
    backward_constant_years = [2000, 2005, 2010, 2015]

    all_frames: list[pd.DataFrame] = []

    # Build GQ for years with stcoreview data
    for year, period in stcoreview_years.items():
        logger.info(f"Building GQ for {year} (stcoreview period={period})")
        gq_year = build_gq_county_age_sex(stcoreview_path, period=period)
        gq_year["year"] = year
        all_frames.append(gq_year)
        state_total = gq_year["gq_population"].sum()
        logger.info(f"  Year {year}: state GQ total = {state_total:,.0f}")

    # Use 2020 data as backward constant for earlier years
    gq_2020 = all_frames[0]  # First frame is 2020
    for year in backward_constant_years:
        logger.info(f"Building GQ for {year} (backward constant from 2020)")
        gq_year = gq_2020.copy()
        gq_year["year"] = year
        all_frames.append(gq_year)

    # Combine all years
    result = pd.concat(all_frames, ignore_index=True)

    # Reorder columns and sort
    result = result[["county_fips", "year", "age_group", "sex", "gq_population"]]
    result = result.sort_values(
        ["year", "county_fips", "age_group", "sex"]
    ).reset_index(drop=True)

    logger.info(
        f"Historical GQ: {len(result)} rows, "
        f"{result['year'].nunique()} years, "
        f"{result['county_fips'].nunique()} counties"
    )

    # Log per-year summary
    for year in sorted(result["year"].unique()):
        year_total = result[result["year"] == year]["gq_population"].sum()
        logger.info(f"  {year}: state GQ = {year_total:,.0f}")

    return result


def main():
    """Main entry point: build and save GQ population data."""
    logger.info("=" * 70)
    logger.info("Building Group Quarters Population Data (ADR-055)")
    logger.info("=" * 70)

    stcoreview_path = (
        project_root / "data" / "raw" / "population"
        / "stcoreview_v2025_nd_parsed.parquet"
    )

    if not stcoreview_path.exists():
        logger.error(f"Stcoreview file not found: {stcoreview_path}")
        return 1

    # --- Phase 1: Build 2025 GQ snapshot ---
    logger.info("--- Phase 1: Building 2025 GQ snapshot ---")
    gq_df = build_gq_county_age_sex(stcoreview_path, period="2025")

    logger.info(f"Built GQ data: {len(gq_df)} rows")
    logger.info(f"Columns: {list(gq_df.columns)}")

    # Validate
    validate_gq_data(gq_df, stcoreview_path)

    # Save to parquet
    output_path = project_root / "data" / "processed" / "gq_county_age_sex_2025.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gq_df.to_parquet(output_path, index=False)
    logger.info(f"Saved Phase 1 GQ data to {output_path}")

    # Print sample for verification
    logger.info("\nSample data (Grand Forks County, 2025):")
    gf = gq_df[gq_df["county_fips"] == "38035"]
    for _, row in gf.iterrows():
        if row["gq_population"] > 0:
            logger.info(
                f"  {row['age_group']:>5s}  {row['sex']:>6s}  {row['gq_population']:>8.1f}"
            )

    # --- Phase 2: Build historical GQ estimates ---
    logger.info("\n--- Phase 2: Building historical GQ estimates ---")
    gq_historical = build_historical_gq(stcoreview_path)

    hist_output_path = project_root / "data" / "processed" / "gq_county_age_sex_historical.parquet"
    gq_historical.to_parquet(hist_output_path, index=False)
    logger.info(f"Saved Phase 2 historical GQ data to {hist_output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("GQ data build complete (ADR-055 Phase 1 + Phase 2)")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
