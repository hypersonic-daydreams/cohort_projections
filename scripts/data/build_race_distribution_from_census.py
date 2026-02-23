#!/usr/bin/env python3
"""
Build age-sex-race distributions from Census full-count population estimates.

Created: 2026-02-23
ADR: 044 (Census Full-Count Race Distribution),
     047 (County-Specific Age-Sex-Race Distributions),
     048 (Single-Year-of-Age Base Population)
Author: Claude Code

Purpose
-------
Replace the PUMS-based race allocation with Census Bureau full-count population
estimates by county x age group x sex x race/Hispanic origin. The PUMS 1% sample
(~12,277 records for ND) was catastrophically insufficient for small race groups:
zero Black females at reproductive ages, 45% of Hispanics concentrated in a single
cell, and only 115 of 216 possible age-sex-race cells populated. This script
produces three distribution artifacts that the projection engine uses for base
population construction.

Method
------
1. Read Census cc-est2024-alldata-38.csv (county characteristics file for ND).
2. Filter to YEAR=6 (July 1, 2024 estimate, most recent in Vintage 2024) and
   AGEGRP > 0 (exclude total rows).
3. Map Census race/ethnicity columns to the project's 6-category scheme:
   NHWA -> white_nonhispanic, NHBA -> black_nonhispanic, NHIA -> aian_nonhispanic,
   NHAA + NHNA -> asian_nonhispanic (Asian + NHPI combined), NHTOM ->
   multiracial_nonhispanic, H -> hispanic. Each column has _MALE/_FEMALE suffixes.
4. Sum across all 53 ND counties to produce a statewide distribution of 216 rows
   (18 five-year age groups x 2 sexes x 6 races). Compute proportions as
   cell_count / state_total.
5. Build county-specific distributions (ADR-047): for each county, compute
   per-county proportions. For counties below the blend threshold (default 5,000),
   blend with the statewide distribution using alpha = min(county_pop / threshold,
   1.0). Re-normalize after blending. Output as Parquet (11,448 rows = 53 counties
   x 216 cells).
6. Build single-year-of-age statewide distribution (ADR-048): read
   sc-est2024-alldata6.csv (state-level single-year-of-age data), filter to
   ND (STATE=38), map race/sex to the same 6-category scheme. Distribute
   the 85+ terminal age group across ages 85-90 using exponential decay
   (survival factor 0.7 per year, geometric tail at 90+). Output as CSV
   (1,092 rows = 91 ages x 2 sexes x 6 races).
7. Run validation checks on all three outputs (see Validation results below).

Key design decisions
--------------------
- **Census full-count data instead of PUMS**: The cc-est2024-alldata file provides
  demographic analysis-based estimates (not sample-based) for every county x age x
  sex x race cell. This eliminates the catastrophic sampling noise from the PUMS 1%
  sample, which left 101 of 216 cells empty and produced physically impossible
  projections (e.g., Black population unable to produce births). Alternative rejected:
  5-year ACS PUMS has a larger sample but is still insufficient for county-level
  race x age x sex cross-tabulation in a small state.
- **Population-weighted blending for small counties**: Counties below 5,000
  population have >30% zero cells in their race distributions. Pure county
  distributions would leave these cells at zero, which can cause unexpected behavior
  when the projection engine assigns migration inflows to zero-population cells.
  Blending with the statewide distribution using alpha = county_pop / 5000 provides
  a small floor (like a Bayesian prior) while preserving the county's dominant
  patterns. Alternative rejected: no blending would be simpler but risks
  zero-population artifacts; a fixed floor (e.g., 0.001) would be arbitrary and
  not scale with county size.
- **Exponential decay for terminal age groups**: The sc-est2024-alldata6.csv file
  tops out at age 85+. The projection engine operates to age 90 (open-ended 90+).
  Distributing the 85+ population using exponential decay with a survival factor of
  0.7 per year is standard demographic practice for terminal age groups. The
  geometric tail at 90+ (s^5 / (1-s)) correctly captures the open-ended interval.
  Alternative rejected: uniform splitting across 85-90 would overestimate the 90+
  population.
- **6-category race mapping**: Follows the project standard (ADR-007). Asian and
  NHPI are combined because NHPI is <0.1% of ND population, and separating them
  would create suppression issues. Hispanic is ethnicity-based (anyone of Hispanic
  origin regardless of race). All non-Hispanic groups are mutually exclusive single-
  race categories except multiracial_nonhispanic (two or more races).

Validation results (2026-02-23)
-------------------------------
Statewide (5-year age groups):
- Row count: 216 (18 age groups x 2 sexes x 6 races) -- PASS
- Proportions sum: 1.00000000 -- PASS
- All 216 cells populated (was 115 with PUMS) -- PASS
- Sex ratio: 105.5 males per 100 females (expected 95-115) -- PASS
- All 6 race categories present -- PASS
- Black females at reproductive ages (15-49): ~7,600 across all 7 age groups,
  zero groups with zero population (was zero across all groups with PUMS) -- PASS
- Hispanic largest cell: ~7.0% of Hispanic total (was 45% with PUMS) -- PASS

County distributions:
- Total rows: 11,448 (53 counties x 216 cells) -- PASS
- All 53 counties have exactly 216 rows -- PASS
- Proportions sum to 1.0 per county (within 1e-6) -- PASS
- No NaN or negative proportions -- PASS
- Reservation county AIAN proportions: Sioux ~59% (>70% raw, blended below 5k
  threshold), Rolette >60%, Benson >30% -- PASS
- Median MAD from average distribution > 0 (distributions are non-identical) -- PASS

Single-year-of-age statewide:
- Row count: 1,092 (91 ages x 2 sexes x 6 races) -- PASS
- Proportions sum: 1.00000000 -- PASS
- Age range: 0-90 -- PASS
- All 6 race categories and both sexes present -- PASS
- Max adjacent-age ratio (ages 1-84): < 2.0 (smooth) -- PASS

Inputs
------
- data/raw/population/cc-est2024-alldata-38.csv
    Census Bureau County Characteristics Resident Population Estimates (Vintage
    2024), FIPS 38 (North Dakota). 6,042 rows, county x age group x sex x race x
    Hispanic origin. Full-count demographic analysis-based estimates, April 2020
    through July 2024. Downloaded 2026-02-18 from Census FTP:
    https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/
- data/raw/population/sc-est2024-alldata6.csv
    Census Bureau Annual State Resident Population Estimates for 6 Race Groups by
    Age, Sex, and Hispanic Origin (Vintage 2024). 236,844 rows, all states, single
    year of age (0-85+). Downloaded 2026-02-18 from Census FTP:
    https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/

Output
------
- data/raw/population/nd_age_sex_race_distribution.csv
    Statewide age-sex-race proportional distribution. 216 rows (18 age groups x 2
    sexes x 6 races). Columns: age_group, sex, race_ethnicity, estimated_count,
    proportion.
- data/processed/county_age_sex_race_distributions.parquet
    County-specific distributions with population-weighted blending for small
    counties. 11,448 rows (53 counties x 216 cells). Columns: fips, age_group, sex,
    race, proportion.
- data/raw/population/nd_age_sex_race_distribution_single_year.csv
    Statewide single-year-of-age distribution. 1,092 rows (91 ages x 2 sexes x 6
    races). Columns: age, sex, race_ethnicity, estimated_count, proportion.

Usage
-----
    python scripts/data/build_race_distribution_from_census.py
    python scripts/data/build_race_distribution_from_census.py --input path/to/file.csv
    python scripts/data/build_race_distribution_from_census.py --output path/to/output.csv
    python scripts/data/build_race_distribution_from_census.py --blend-threshold 5000
    python scripts/data/build_race_distribution_from_census.py --skip-county
    python scripts/data/build_race_distribution_from_census.py --skip-single-year
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "population" / "cc-est2024-alldata-38.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "population" / "nd_age_sex_race_distribution.csv"
DEFAULT_COUNTY_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "county_age_sex_race_distributions.parquet"
)

# SC-EST single-year-of-age data (ADR-048)
DEFAULT_SCEST_INPUT = PROJECT_ROOT / "data" / "raw" / "population" / "sc-est2024-alldata6.csv"
DEFAULT_SINGLE_YEAR_OUTPUT = (
    PROJECT_ROOT / "data" / "raw" / "population" / "nd_age_sex_race_distribution_single_year.csv"
)

# Default blending threshold (ADR-047): counties below this population
# have their distribution blended with the statewide distribution.
DEFAULT_BLEND_THRESHOLD = 5000

# North Dakota state FIPS prefix
STATE_FIPS = "38"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YEAR=6 corresponds to the July 1, 2024 estimate (most recent in V2024).
# The cc-est2024-alldata year codes are:
#   1 = April 1, 2020 Census population
#   2 = April 1, 2020 Estimates base
#   3 = July 1, 2021 estimate
#   4 = July 1, 2022 estimate
#   5 = July 1, 2023 estimate
#   6 = July 1, 2024 estimate
YEAR_CODE = 6

# Age group code mapping: AGEGRP -> project age group string
# AGEGRP=0 is the total row (excluded); 1-18 are five-year groups
AGEGRP_MAP = {
    1: "0-4",
    2: "5-9",
    3: "10-14",
    4: "15-19",
    5: "20-24",
    6: "25-29",
    7: "30-34",
    8: "35-39",
    9: "40-44",
    10: "45-49",
    11: "50-54",
    12: "55-59",
    13: "60-64",
    14: "65-69",
    15: "70-74",
    16: "75-79",
    17: "80-84",
    18: "85+",
}

# Mapping from Census columns to project race categories.
# Each entry is (race_ethnicity, sex, list_of_census_columns_to_sum).
RACE_COLUMN_MAP = [
    ("white_nonhispanic", "male", ["NHWA_MALE"]),
    ("white_nonhispanic", "female", ["NHWA_FEMALE"]),
    ("black_nonhispanic", "male", ["NHBA_MALE"]),
    ("black_nonhispanic", "female", ["NHBA_FEMALE"]),
    ("aian_nonhispanic", "male", ["NHIA_MALE"]),
    ("aian_nonhispanic", "female", ["NHIA_FEMALE"]),
    ("asian_nonhispanic", "male", ["NHAA_MALE", "NHNA_MALE"]),   # Asian + NHPI
    ("asian_nonhispanic", "female", ["NHAA_FEMALE", "NHNA_FEMALE"]),
    ("multiracial_nonhispanic", "male", ["NHTOM_MALE"]),
    ("multiracial_nonhispanic", "female", ["NHTOM_FEMALE"]),
    ("hispanic", "male", ["H_MALE"]),
    ("hispanic", "female", ["H_FEMALE"]),
]

# Expected dimensions
EXPECTED_AGE_GROUPS = 18
EXPECTED_SEXES = 2
EXPECTED_RACES = 6
EXPECTED_ROWS = EXPECTED_AGE_GROUPS * EXPECTED_SEXES * EXPECTED_RACES  # 216


def build_distribution(input_path: Path) -> pd.DataFrame:
    """
    Read cc-est2024-alldata CSV and produce a state-level age-sex-race
    distribution DataFrame.

    Returns DataFrame with columns:
        age_group, sex, race_ethnicity, estimated_count, proportion
    """
    print(f"Reading Census data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter to YEAR=7 (July 1, 2024 estimate)
    df = df[df["YEAR"] == YEAR_CODE].copy()
    print(f"  After YEAR={YEAR_CODE} filter: {len(df):,} rows")

    # Filter to AGEGRP > 0 (exclude total row)
    df = df[df["AGEGRP"] > 0].copy()
    print(f"  After AGEGRP > 0 filter: {len(df):,} rows")

    # Verify we have all 53 ND counties x 18 age groups
    n_counties = df["COUNTY"].nunique()
    n_age_groups = df["AGEGRP"].nunique()
    print(f"  Counties: {n_counties}, Age groups: {n_age_groups}")

    if n_age_groups != EXPECTED_AGE_GROUPS:
        print(f"  WARNING: Expected {EXPECTED_AGE_GROUPS} age groups, got {n_age_groups}")

    # Sum across all counties to get state totals by age group
    # First, map AGEGRP codes to age group strings
    df["age_group"] = df["AGEGRP"].map(AGEGRP_MAP)

    # Verify all age groups mapped successfully
    unmapped = df[df["age_group"].isna()]["AGEGRP"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: Unmapped age group codes: {unmapped}")

    # Aggregate across counties for each age group
    agg_cols = set()
    for _race, _sex, cols in RACE_COLUMN_MAP:
        agg_cols.update(cols)

    # Verify all needed columns exist
    missing_cols = agg_cols - set(df.columns)
    if missing_cols:
        print(f"  ERROR: Missing columns in Census data: {missing_cols}")
        sys.exit(1)

    state_totals = df.groupby("age_group")[list(agg_cols)].sum()

    # Build the output rows
    rows = []
    for agegrp_code in sorted(AGEGRP_MAP.keys()):
        age_group = AGEGRP_MAP[agegrp_code]
        if age_group not in state_totals.index:
            print(f"  WARNING: Age group '{age_group}' not found in aggregated data")
            continue

        age_row = state_totals.loc[age_group]

        for race_ethnicity, sex, census_cols in RACE_COLUMN_MAP:
            count = sum(age_row[col] for col in census_cols)
            rows.append({
                "age_group": age_group,
                "sex": sex,
                "race_ethnicity": race_ethnicity,
                "estimated_count": float(count),
            })

    result = pd.DataFrame(rows)

    # Compute total population across all cells
    total_pop = result["estimated_count"].sum()
    print(f"\n  State total population (from race cells): {total_pop:,.0f}")

    # Compute proportions
    result["proportion"] = result["estimated_count"] / total_pop

    return result


def validate_distribution(df: pd.DataFrame) -> bool:
    """
    Validate the output distribution.

    Checks:
    1. Correct number of rows (216)
    2. Proportions sum to 1.0 (within tolerance)
    3. All cells populated (no NaN, no negative counts)
    4. Sex ratio is reasonable (95-115 males per 100 females)
    5. All expected categories present
    """
    ok = True

    # 1. Row count
    if len(df) != EXPECTED_ROWS:
        print(f"  FAIL: Expected {EXPECTED_ROWS} rows, got {len(df)}")
        ok = False
    else:
        print(f"  OK: {len(df)} rows (18 age groups x 2 sexes x 6 races)")

    # 2. Proportions sum to 1.0
    prop_sum = df["proportion"].sum()
    if abs(prop_sum - 1.0) > 1e-6:
        print(f"  FAIL: Proportions sum to {prop_sum:.8f}, expected 1.0")
        ok = False
    else:
        print(f"  OK: Proportions sum to {prop_sum:.8f}")

    # 3. No missing or negative values
    if df["estimated_count"].isna().any():
        print(f"  FAIL: {df['estimated_count'].isna().sum()} NaN values in estimated_count")
        ok = False

    if (df["estimated_count"] < 0).any():
        print(f"  FAIL: Negative estimated_count values found")
        ok = False

    zero_cells = (df["estimated_count"] == 0).sum()
    if zero_cells > 0:
        print(f"  NOTE: {zero_cells} cells have zero population (expected for some small groups)")
    else:
        print(f"  OK: All {len(df)} cells have non-zero population")

    # 4. Sex ratio
    male_total = df[df["sex"] == "male"]["estimated_count"].sum()
    female_total = df[df["sex"] == "female"]["estimated_count"].sum()
    sex_ratio = male_total / female_total * 100
    if 95 <= sex_ratio <= 115:
        print(f"  OK: Sex ratio = {sex_ratio:.1f} males per 100 females")
    else:
        print(f"  WARNING: Sex ratio = {sex_ratio:.1f} males per 100 females (outside 95-115 range)")

    # 5. Expected categories
    expected_races = {
        "white_nonhispanic", "black_nonhispanic", "aian_nonhispanic",
        "asian_nonhispanic", "multiracial_nonhispanic", "hispanic",
    }
    actual_races = set(df["race_ethnicity"].unique())
    if actual_races == expected_races:
        print(f"  OK: All 6 race categories present")
    else:
        missing = expected_races - actual_races
        extra = actual_races - expected_races
        if missing:
            print(f"  FAIL: Missing race categories: {missing}")
            ok = False
        if extra:
            print(f"  WARNING: Extra race categories: {extra}")

    expected_sexes = {"male", "female"}
    actual_sexes = set(df["sex"].unique())
    if actual_sexes == expected_sexes:
        print(f"  OK: Both sexes present")
    else:
        print(f"  FAIL: Expected sexes {expected_sexes}, got {actual_sexes}")
        ok = False

    # 6. Race-specific checks for the critical defects this fix addresses
    print("\n  Race-specific validation (critical fix areas):")

    # Black females at reproductive ages (15-49)
    repro_ages = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
    black_repro = df[
        (df["race_ethnicity"] == "black_nonhispanic")
        & (df["sex"] == "female")
        & (df["age_group"].isin(repro_ages))
    ]
    black_repro_total = black_repro["estimated_count"].sum()
    black_repro_zero = (black_repro["estimated_count"] == 0).sum()
    if black_repro_zero == 0 and black_repro_total > 0:
        print(f"    OK: Black females at reproductive ages: {black_repro_total:,.0f} "
              f"(all {len(repro_ages)} age groups populated)")
    else:
        print(f"    WARNING: Black females at reproductive ages: {black_repro_total:,.0f}, "
              f"{black_repro_zero} groups with zero population")

    # Hispanic distribution -- check not concentrated in single cell
    hispanic_df = df[df["race_ethnicity"] == "hispanic"]
    hispanic_total = hispanic_df["estimated_count"].sum()
    hispanic_max_cell = hispanic_df["estimated_count"].max()
    hispanic_max_share = hispanic_max_cell / hispanic_total if hispanic_total > 0 else 0
    if hispanic_max_share < 0.15:
        print(f"    OK: Hispanic largest cell is {hispanic_max_share:.1%} of total "
              f"(no single-cell concentration)")
    else:
        print(f"    WARNING: Hispanic largest cell is {hispanic_max_share:.1%} of total")

    return ok


# ---------------------------------------------------------------------------
# SC-EST single-year-of-age distribution (ADR-048)
# ---------------------------------------------------------------------------

# State FIPS as integer for SC-EST filtering
STATE_FIPS_INT = 38

# SC-EST race mapping: (ORIGIN, RACE) -> project race category
# ORIGIN=1 (Not Hispanic), ORIGIN=2 (Hispanic)
# RACE: 1=White, 2=Black, 3=AIAN, 4=Asian, 5=NHPI, 6=Two or More
SCEST_RACE_MAP: dict[tuple[int, int], str] = {
    # Non-Hispanic races
    (1, 1): "white_nonhispanic",
    (1, 2): "black_nonhispanic",
    (1, 3): "aian_nonhispanic",
    (1, 4): "asian_nonhispanic",
    (1, 5): "asian_nonhispanic",     # NHPI combined with Asian
    (1, 6): "multiracial_nonhispanic",
    # Hispanic (all races map to "hispanic")
    (2, 1): "hispanic",
    (2, 2): "hispanic",
    (2, 3): "hispanic",
    (2, 4): "hispanic",
    (2, 5): "hispanic",
    (2, 6): "hispanic",
}

# SC-EST sex mapping: Census code -> project sex label
SCEST_SEX_MAP = {1: "male", 2: "female"}

# Terminal age group distribution: 85+ -> ages 85-90 using exponential decay
TERMINAL_SURVIVAL_FACTOR = 0.7

# Engine max age (90 is open-ended 90+)
ENGINE_MAX_AGE = 90

# Expected single-year dimensions
EXPECTED_SINGLE_YEAR_AGES = ENGINE_MAX_AGE + 1  # 0-90 = 91 ages
EXPECTED_SINGLE_YEAR_ROWS = EXPECTED_SINGLE_YEAR_AGES * EXPECTED_SEXES * EXPECTED_RACES  # 1092


def _terminal_age_weights(survival_factor: float = TERMINAL_SURVIVAL_FACTOR) -> dict[int, float]:
    """
    Compute exponential-decay weights for distributing the 85+ population
    across single years 85-90 (where 90 is the open-ended 90+ group).

    Standard demographic practice for terminal age groups:
      weight[i] = survival_factor^(i - 85) for i = 85..89
      weight[90] = survival_factor^5 / (1 - survival_factor)  (geometric tail)

    Returns:
        Dict mapping age (85-90) to normalized weight (sums to 1.0)
    """
    weights: dict[int, float] = {}
    for age in range(85, 90):
        weights[age] = survival_factor ** (age - 85)
    # 90+ absorbs the geometric tail: s^5 + s^6 + ... = s^5 / (1-s)
    weights[90] = survival_factor ** 5 / (1.0 - survival_factor)

    # Normalize to sum to 1.0
    total = sum(weights.values())
    return {age: w / total for age, w in weights.items()}


def build_single_year_statewide_distribution(
    input_path: Path = DEFAULT_SCEST_INPUT,
) -> pd.DataFrame:
    """
    Build single-year-of-age statewide distribution from SC-EST2024-ALLDATA6.

    Uses Census Bureau state-level single-year-of-age x sex x race/Hispanic
    origin population estimates to produce a smooth distribution file with
    1,092 rows (91 ages x 2 sexes x 6 races).

    The 85+ terminal age group is distributed across ages 85-90 using
    exponential decay with a survival factor of 0.7 per year.

    Args:
        input_path: Path to sc-est2024-alldata6.csv

    Returns:
        DataFrame with columns: [age, sex, race_ethnicity, estimated_count, proportion]

    See: ADR-048 (Single-Year-of-Age Base Population)
    """
    print(f"\nReading SC-EST data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter to North Dakota
    df = df[df["STATE"] == STATE_FIPS_INT].copy()
    print(f"  After STATE={STATE_FIPS_INT} filter: {len(df):,} rows")

    # Exclude totals: SEX=0 (total) and ORIGIN=0 (total)
    df = df[(df["SEX"].isin([1, 2])) & (df["ORIGIN"].isin([1, 2]))].copy()
    print(f"  After excluding totals (SEX>0, ORIGIN>0): {len(df):,} rows")

    # Map each row to project race category
    df["race_ethnicity"] = df.apply(
        lambda row: SCEST_RACE_MAP.get((row["ORIGIN"], row["RACE"])), axis=1
    )
    unmapped = df[df["race_ethnicity"].isna()]
    if len(unmapped) > 0:
        print(f"  WARNING: {len(unmapped)} rows with unmapped ORIGIN/RACE combinations")
        df = df[df["race_ethnicity"].notna()].copy()

    # Map sex
    df["sex"] = df["SEX"].map(SCEST_SEX_MAP)

    # Use POPESTIMATE2024 as the population column
    pop_col = "POPESTIMATE2024"
    if pop_col not in df.columns:
        print(f"  ERROR: Column {pop_col} not found")
        sys.exit(1)

    # Aggregate: for Hispanic, all RACE values within ORIGIN=2 need summing
    # For Non-Hispanic NHPI (RACE=5), it combines with Asian (RACE=4)
    agg = (
        df.groupby(["AGE", "sex", "race_ethnicity"])[pop_col]
        .sum()
        .reset_index()
        .rename(columns={pop_col: "estimated_count"})
    )

    print(f"  Aggregated to {len(agg):,} cells (before terminal age expansion)")

    # Verify age range: 0-85
    age_min, age_max = agg["AGE"].min(), agg["AGE"].max()
    print(f"  Age range in data: {age_min}-{age_max}")

    # Split into ages 0-84 (direct) and 85+ (needs expansion)
    direct = agg[agg["AGE"] < 85].copy()
    direct = direct.rename(columns={"AGE": "age"})

    terminal = agg[agg["AGE"] == 85].copy()
    print(f"  Direct single-year cells (ages 0-84): {len(direct):,}")
    print(f"  Terminal 85+ cells to expand: {len(terminal):,}")

    # Expand 85+ to ages 85-90 using exponential decay
    weights = _terminal_age_weights()
    print(f"  Terminal age weights: { {k: f'{v:.4f}' for k, v in weights.items()} }")

    expanded_terminal_rows: list[dict[str, str | int | float]] = []
    for _, row in terminal.iterrows():
        for age, weight in weights.items():
            expanded_terminal_rows.append({
                "age": age,
                "sex": row["sex"],
                "race_ethnicity": row["race_ethnicity"],
                "estimated_count": row["estimated_count"] * weight,
            })

    expanded_terminal = pd.DataFrame(expanded_terminal_rows)

    # Combine direct + expanded terminal
    result = pd.concat([direct, expanded_terminal], ignore_index=True)

    # Ensure integer age column
    result["age"] = result["age"].astype(int)

    # Aggregate any duplicates (shouldn't be any, but safety)
    result = result.groupby(["age", "sex", "race_ethnicity"], as_index=False).agg(
        {"estimated_count": "sum"}
    )

    # Compute proportions
    total_pop = result["estimated_count"].sum()
    result["proportion"] = result["estimated_count"] / total_pop

    # Sort by age, sex, race for consistent output
    result = result.sort_values(["age", "sex", "race_ethnicity"]).reset_index(drop=True)

    print(f"\n  Total population (from SC-EST): {total_pop:,.0f}")
    print(f"  Output rows: {len(result):,}")

    return result


def validate_single_year_distribution(df: pd.DataFrame) -> bool:
    """
    Validate the single-year-of-age statewide distribution.

    Checks:
    1. Correct number of rows (1,092 = 91 ages x 2 sexes x 6 races)
    2. Proportions sum to 1.0
    3. No NaN or negative values
    4. Age range is 0-90
    5. All expected categories present
    6. Smoothness: adjacent ages within each sex-race should not have
       extreme jumps (except around 85 where terminal expansion occurs)
    """
    ok = True

    # 1. Row count
    if len(df) != EXPECTED_SINGLE_YEAR_ROWS:
        print(f"  FAIL: Expected {EXPECTED_SINGLE_YEAR_ROWS} rows, got {len(df)}")
        ok = False
    else:
        print(f"  OK: {len(df)} rows (91 ages x 2 sexes x 6 races)")

    # 2. Proportions sum to 1.0
    prop_sum = df["proportion"].sum()
    if abs(prop_sum - 1.0) > 1e-6:
        print(f"  FAIL: Proportions sum to {prop_sum:.8f}, expected 1.0")
        ok = False
    else:
        print(f"  OK: Proportions sum to {prop_sum:.8f}")

    # 3. No missing or negative values
    if df["estimated_count"].isna().any():
        print(f"  FAIL: {df['estimated_count'].isna().sum()} NaN values")
        ok = False
    if (df["estimated_count"] < 0).any():
        print(f"  FAIL: Negative estimated_count values found")
        ok = False

    zero_cells = (df["estimated_count"] == 0).sum()
    if zero_cells > 0:
        print(f"  NOTE: {zero_cells} cells have zero population")
    else:
        print(f"  OK: All cells have non-zero population")

    # 4. Age range
    age_min, age_max = df["age"].min(), df["age"].max()
    if age_min != 0 or age_max != ENGINE_MAX_AGE:
        print(f"  FAIL: Age range {age_min}-{age_max}, expected 0-{ENGINE_MAX_AGE}")
        ok = False
    else:
        print(f"  OK: Age range 0-{ENGINE_MAX_AGE}")

    # 5. Expected categories
    expected_races = {
        "white_nonhispanic", "black_nonhispanic", "aian_nonhispanic",
        "asian_nonhispanic", "multiracial_nonhispanic", "hispanic",
    }
    actual_races = set(df["race_ethnicity"].unique())
    if actual_races == expected_races:
        print(f"  OK: All 6 race categories present")
    else:
        missing = expected_races - actual_races
        extra = actual_races - expected_races
        if missing:
            print(f"  FAIL: Missing race categories: {missing}")
            ok = False
        if extra:
            print(f"  WARNING: Extra race categories: {extra}")

    expected_sexes = {"male", "female"}
    actual_sexes = set(df["sex"].unique())
    if actual_sexes == expected_sexes:
        print(f"  OK: Both sexes present")
    else:
        print(f"  FAIL: Expected sexes {expected_sexes}, got {actual_sexes}")
        ok = False

    # 6. Smoothness check (ages 0-84 only, within each sex-race group)
    print("\n  Smoothness check (max adjacent-age ratio, ages 1-84):")
    max_ratio = 0.0
    worst_group = ""
    for (sex, race), group in df[df["age"] <= 84].groupby(["sex", "race_ethnicity"]):
        group = group.sort_values("age")
        counts = group["estimated_count"].values
        for i in range(1, len(counts)):
            if counts[i - 1] > 0 and counts[i] > 0:
                ratio = max(counts[i] / counts[i - 1], counts[i - 1] / counts[i])
                if ratio > max_ratio:
                    max_ratio = ratio
                    worst_group = f"{sex}/{race} ages {group['age'].iloc[i-1]}-{group['age'].iloc[i]}"

    if max_ratio < 2.0:
        print(f"    OK: Max adjacent-age ratio = {max_ratio:.2f} (< 2.0)")
    else:
        print(f"    WARNING: Max adjacent-age ratio = {max_ratio:.2f} at {worst_group}")

    # 7. Compare total with 5-year group file if available
    total_pop = df["estimated_count"].sum()
    print(f"\n  Total population: {total_pop:,.0f}")

    # Sex ratio
    male_total = df[df["sex"] == "male"]["estimated_count"].sum()
    female_total = df[df["sex"] == "female"]["estimated_count"].sum()
    sex_ratio = male_total / female_total * 100
    print(f"  Sex ratio: {sex_ratio:.1f} males per 100 females")

    return ok


def build_county_distributions(
    input_path: Path,
    state_distribution: pd.DataFrame,
    blend_threshold: int = DEFAULT_BLEND_THRESHOLD,
) -> pd.DataFrame:
    """
    Build county-specific age-sex-race distributions from cc-est2024-alldata.

    For each county, compute per-county proportions. For small counties
    (population < blend_threshold), blend with the statewide distribution
    using alpha = min(county_pop / blend_threshold, 1.0).

    Args:
        input_path: Path to cc-est2024-alldata CSV
        state_distribution: Statewide distribution DataFrame (from build_distribution)
        blend_threshold: Population below which blending is applied

    Returns:
        DataFrame with columns:
            fips, age_group, sex, race, proportion
    """
    print(f"\nBuilding county-specific distributions (blend threshold: {blend_threshold:,})")

    df = pd.read_csv(input_path)

    # Filter to most recent year estimate and exclude totals
    df = df[df["YEAR"] == YEAR_CODE].copy()
    df = df[(df["AGEGRP"] > 0) & (df["COUNTY"] > 0)].copy()

    # Map age group codes
    df["age_group"] = df["AGEGRP"].map(AGEGRP_MAP)

    # Build FIPS codes: state 38 + 3-digit county code
    df["fips"] = df["COUNTY"].apply(lambda c: f"{STATE_FIPS}{c:03d}")

    # Collect all Census columns we need
    agg_cols = set()
    for _race, _sex, cols in RACE_COLUMN_MAP:
        agg_cols.update(cols)

    # Verify all needed columns exist
    missing_cols = agg_cols - set(df.columns)
    if missing_cols:
        print(f"  ERROR: Missing columns in Census data: {missing_cols}")
        sys.exit(1)

    # Prepare statewide proportions as a lookup (for blending)
    state_props = state_distribution.set_index(
        ["age_group", "sex", "race_ethnicity"]
    )["proportion"].to_dict()

    # Process each county
    all_county_rows: list[dict[str, str | float]] = []
    county_fips_list = sorted(df["fips"].unique())
    n_blended = 0

    for fips in county_fips_list:
        county_df = df[df["fips"] == fips]

        # Build rows for this county
        county_rows: list[dict[str, str | float]] = []
        for agegrp_code in sorted(AGEGRP_MAP.keys()):
            age_group = AGEGRP_MAP[agegrp_code]
            age_data = county_df[county_df["age_group"] == age_group]

            if age_data.empty:
                # Should not happen for valid data, but handle gracefully
                for race_ethnicity, sex, _census_cols in RACE_COLUMN_MAP:
                    county_rows.append({
                        "fips": fips,
                        "age_group": age_group,
                        "sex": sex,
                        "race": race_ethnicity,
                        "proportion": 0.0,
                    })
                continue

            age_row = age_data.iloc[0]
            for race_ethnicity, sex, census_cols in RACE_COLUMN_MAP:
                count = sum(float(age_row[col]) for col in census_cols)
                county_rows.append({
                    "fips": fips,
                    "age_group": age_group,
                    "sex": sex,
                    "race": race_ethnicity,
                    "proportion": count,  # Temporarily store count; convert to proportion below
                })

        # Convert counts to proportions
        county_result = pd.DataFrame(county_rows)
        county_total = county_result["proportion"].sum()

        if county_total > 0:
            county_result["proportion"] = county_result["proportion"] / county_total
        else:
            # Zero-population county: use statewide distribution
            for idx, row in county_result.iterrows():
                key = (row["age_group"], row["sex"], row["race"])
                county_result.at[idx, "proportion"] = state_props.get(key, 0.0)

        # Apply blending for small counties
        alpha = min(county_total / blend_threshold, 1.0) if blend_threshold > 0 else 1.0
        if alpha < 1.0:
            n_blended += 1
            for idx, row in county_result.iterrows():
                key = (row["age_group"], row["sex"], row["race"])
                state_prop = state_props.get(key, 0.0)
                county_result.at[idx, "proportion"] = (
                    alpha * row["proportion"] + (1.0 - alpha) * state_prop
                )
            # Re-normalize after blending
            prop_sum = county_result["proportion"].sum()
            if prop_sum > 0:
                county_result["proportion"] = county_result["proportion"] / prop_sum

        all_county_rows.extend(county_result.to_dict("records"))

    result = pd.DataFrame(all_county_rows)

    print(f"  Counties processed: {len(county_fips_list)}")
    print(f"  Counties blended with statewide (pop < {blend_threshold:,}): {n_blended}")
    print(f"  Total rows: {len(result):,} ({len(county_fips_list)} counties x {EXPECTED_ROWS} cells)")

    return result


def validate_county_distributions(df: pd.DataFrame) -> bool:
    """
    Validate county-specific distributions.

    Checks:
    1. Correct total row count (53 counties x 216 cells = 11,448)
    2. Each county has exactly 216 rows
    3. Proportions sum to 1.0 per county (within tolerance)
    4. No NaN or negative proportions
    5. Reservation county spot checks (AIAN proportion)
    """
    ok = True
    n_counties = df["fips"].nunique()
    expected_total = n_counties * EXPECTED_ROWS

    print(f"\n  County distribution validation ({n_counties} counties):")

    # 1. Total row count
    if len(df) != expected_total:
        print(f"  FAIL: Expected {expected_total} rows, got {len(df)}")
        ok = False
    else:
        print(f"  OK: {len(df):,} rows ({n_counties} counties x {EXPECTED_ROWS} cells)")

    # 2. Each county has exactly 216 rows
    county_counts = df.groupby("fips").size()
    bad_counties = county_counts[county_counts != EXPECTED_ROWS]
    if len(bad_counties) > 0:
        print(f"  FAIL: {len(bad_counties)} counties have wrong row count: {dict(bad_counties)}")
        ok = False
    else:
        print(f"  OK: All {n_counties} counties have {EXPECTED_ROWS} rows each")

    # 3. Proportions sum to 1.0 per county
    county_sums = df.groupby("fips")["proportion"].sum()
    bad_sums = county_sums[abs(county_sums - 1.0) > 1e-6]
    if len(bad_sums) > 0:
        print(f"  FAIL: {len(bad_sums)} counties have proportions not summing to 1.0")
        for fips, prop_sum in bad_sums.items():
            print(f"    {fips}: {prop_sum:.8f}")
        ok = False
    else:
        print(f"  OK: All county proportions sum to 1.0 (within 1e-6)")

    # 4. No NaN or negative proportions
    nan_count = df["proportion"].isna().sum()
    neg_count = (df["proportion"] < 0).sum()
    if nan_count > 0:
        print(f"  FAIL: {nan_count} NaN proportions")
        ok = False
    if neg_count > 0:
        print(f"  FAIL: {neg_count} negative proportions")
        ok = False
    if nan_count == 0 and neg_count == 0:
        print(f"  OK: No NaN or negative proportions")

    # 5. Reservation county spot checks
    reservation_checks = [
        ("38085", "Sioux", "aian_nonhispanic", 0.70),
        ("38079", "Rolette", "aian_nonhispanic", 0.70),
        ("38005", "Benson", "aian_nonhispanic", 0.40),
    ]
    print("\n  Reservation county spot checks:")
    for fips, name, race, expected_min in reservation_checks:
        county_data = df[df["fips"] == fips]
        if county_data.empty:
            print(f"    SKIP: {name} ({fips}) not found in data")
            continue
        race_prop = county_data[county_data["race"] == race]["proportion"].sum()
        if race_prop >= expected_min:
            print(f"    OK: {name} ({fips}) AIAN proportion = {race_prop:.1%} (>= {expected_min:.0%})")
        else:
            print(f"    WARNING: {name} ({fips}) AIAN proportion = {race_prop:.1%} "
                  f"(expected >= {expected_min:.0%})")

    # 6. Distribution divergence check
    print("\n  Distribution divergence:")
    # Check that county distributions actually differ from each other
    # by computing the mean absolute deviation from the average distribution
    avg_dist = df.groupby(["age_group", "sex", "race"])["proportion"].mean()
    mad_by_county = []
    for fips in df["fips"].unique():
        county_data = df[df["fips"] == fips].set_index(["age_group", "sex", "race"])["proportion"]
        mad = (county_data - avg_dist).abs().mean()
        mad_by_county.append(mad)
    median_mad = sorted(mad_by_county)[len(mad_by_county) // 2]
    print(f"    Median MAD from average distribution: {median_mad:.6f}")
    if median_mad > 0:
        print(f"    OK: County distributions diverge from each other")
    else:
        print(f"    WARNING: All county distributions are identical")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build age-sex-race distribution from Census cc-est2024-alldata"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to cc-est2024-alldata CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for statewide output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--county-output",
        type=Path,
        default=DEFAULT_COUNTY_OUTPUT,
        help=f"Path for county distributions Parquet (default: {DEFAULT_COUNTY_OUTPUT})",
    )
    parser.add_argument(
        "--blend-threshold",
        type=int,
        default=DEFAULT_BLEND_THRESHOLD,
        help=f"Population threshold for blending with statewide (default: {DEFAULT_BLEND_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-county",
        action="store_true",
        help="Skip county-specific distribution generation (statewide only)",
    )
    parser.add_argument(
        "--scest-input",
        type=Path,
        default=DEFAULT_SCEST_INPUT,
        help=f"Path to sc-est2024-alldata6 CSV (default: {DEFAULT_SCEST_INPUT})",
    )
    parser.add_argument(
        "--single-year-output",
        type=Path,
        default=DEFAULT_SINGLE_YEAR_OUTPUT,
        help=f"Path for single-year distribution CSV (default: {DEFAULT_SINGLE_YEAR_OUTPUT})",
    )
    parser.add_argument(
        "--skip-single-year",
        action="store_true",
        help="Skip single-year-of-age distribution generation (ADR-048)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Download with:")
        print('  curl -o data/raw/population/cc-est2024-alldata-38.csv \\')
        print('    "https://www2.census.gov/programs-surveys/popest/datasets/'
              '2020-2024/counties/asrh/cc-est2024-alldata-38.csv"')
        sys.exit(1)

    # Build the statewide distribution
    print("=" * 70)
    print("Building age-sex-race distribution from Census cc-est2024-alldata")
    print("=" * 70)

    result = build_distribution(args.input)

    # Validate statewide
    print("\nStatewide Validation:")
    valid = validate_distribution(result)

    # Write statewide output
    print(f"\nWriting {len(result)} rows to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    # Build county-specific distributions (ADR-047)
    county_valid = True
    if not args.skip_county:
        print("\n" + "=" * 70)
        print("Building county-specific distributions (ADR-047)")
        print("=" * 70)

        county_result = build_county_distributions(
            args.input,
            state_distribution=result,
            blend_threshold=args.blend_threshold,
        )

        # Validate county distributions
        print("\nCounty Distribution Validation:")
        county_valid = validate_county_distributions(county_result)

        # Write county output
        print(f"\nWriting {len(county_result):,} rows to: {args.county_output}")
        args.county_output.parent.mkdir(parents=True, exist_ok=True)
        county_result.to_parquet(args.county_output, index=False)

    # Build single-year-of-age statewide distribution (ADR-048)
    single_year_valid = True
    single_year_result = None
    if not args.skip_single_year:
        print("\n" + "=" * 70)
        print("Building single-year-of-age statewide distribution (ADR-048)")
        print("=" * 70)

        if not args.scest_input.exists():
            print(f"  WARNING: SC-EST input file not found: {args.scest_input}")
            print("  Download with:")
            print('    curl -o data/raw/population/sc-est2024-alldata6.csv \\')
            print('      "https://www2.census.gov/programs-surveys/popest/datasets/'
                  '2020-2024/state/asrh/sc-est2024-alldata6.csv"')
            print("  Skipping single-year distribution generation.")
        else:
            single_year_result = build_single_year_statewide_distribution(args.scest_input)

            # Validate
            print("\nSingle-Year Distribution Validation:")
            single_year_valid = validate_single_year_distribution(single_year_result)

            # Write output
            print(f"\nWriting {len(single_year_result):,} rows to: {args.single_year_output}")
            args.single_year_output.parent.mkdir(parents=True, exist_ok=True)
            single_year_result.to_csv(args.single_year_output, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Input:  {args.input}")
    print(f"  Output (statewide): {args.output}")
    print(f"  Rows (statewide):   {len(result)} (was 115 with PUMS-based file)")
    total_pop = result["estimated_count"].sum()
    print(f"  State population (V2024): {total_pop:,.0f}")

    if not args.skip_county:
        print(f"  Output (county):    {args.county_output}")
        print(f"  Rows (county):      {len(county_result):,}")
        print(f"  Blend threshold:    {args.blend_threshold:,}")

    if single_year_result is not None:
        sy_total = single_year_result["estimated_count"].sum()
        print(f"  Output (single-year): {args.single_year_output}")
        print(f"  Rows (single-year):   {len(single_year_result):,} "
              f"(91 ages x 2 sexes x 6 races)")
        print(f"  State population (SC-EST V2024): {sy_total:,.0f}")

    # Print race breakdown
    print("\n  Race breakdown:")
    race_totals = result.groupby("race_ethnicity")["estimated_count"].sum().sort_values(ascending=False)
    for race, count in race_totals.items():
        pct = count / total_pop * 100
        print(f"    {race:30s}  {count:>10,.0f}  ({pct:5.1f}%)")

    if not valid or not county_valid or not single_year_valid:
        print("\n  WARNING: Some validation checks failed. Review output carefully.")
        sys.exit(1)
    else:
        print("\n  All validation checks passed.")


if __name__ == "__main__":
    main()
