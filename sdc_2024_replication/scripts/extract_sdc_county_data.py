#!/usr/bin/env python3
"""
Extract county-level demographic data from SDC 2024 Excel workbooks.

This script reads the SDC Excel files and extracts:
1. Base population by county, age group, and sex (Census 2020)
2. Fertility rates by county and age group
3. Survival rates by sex (state-level, applied to all counties)
4. Migration rates by county, age group, and sex

Output files are saved to the sdc_2024_replication/data/ directory.

Note: This is a standalone data extraction script. Type hints are relaxed
for pandas Excel reading operations which have complex union types.
"""
# mypy: disable-error-code="arg-type,union-attr"

from pathlib import Path

import pandas as pd

# Define paths relative to script location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # sdc_2024_replication -> cohort_projections

SDC_DATA_DIR = _PROJECT_ROOT / "data/raw/nd_sdc_2024_projections"
OUTPUT_DIR = _SCRIPT_DIR.parent / "data"  # sdc_2024_replication/data

# Source files
PROJECTIONS_BASE_FILE = SDC_DATA_DIR / "source_files/results/Projections_Base_2023.xlsx"
COUNTY_PROJECTIONS_FILE = SDC_DATA_DIR / "County_Population_Projections_2023.xlsx"
MIGRATION_FILE = SDC_DATA_DIR / "source_files/migration/Mig Rate 2000-2020_final.xlsx"

# Age group labels (18 five-year cohorts)
AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]

# Fertility age groups (subset used for fertility calculations)
FERTILITY_AGE_GROUPS = ["10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]


def get_county_list() -> pd.DataFrame:
    """Extract county list with FIPS codes from County List sheet."""
    df = pd.read_excel(COUNTY_PROJECTIONS_FILE, sheet_name="County List", header=None)

    # Extract region, county_id, and county name (rows 1-53)
    counties = []
    for i in range(1, 54):
        region = df.iloc[i, 0]
        county_id = int(df.iloc[i, 1])
        county_name = df.iloc[i, 2].replace(" County", "")
        counties.append({"county_id": county_id, "county_name": county_name, "region": region})

    return pd.DataFrame(counties)


def extract_base_population() -> pd.DataFrame:
    """
    Extract Census 2020 base population by county, age group, and sex.

    Returns DataFrame with columns:
    - county_name: County name (without "County" suffix)
    - age_group: Age group label (e.g., "0-4", "5-9", ...)
    - sex: "male" or "female"
    - population: Population count
    """
    print("Extracting base population (Census 2020)...")

    # Read the Census 2020 sheet from Projections_Base_2023.xlsx
    df = pd.read_excel(PROJECTIONS_BASE_FILE, sheet_name="Census 2020", header=None)

    # Get county names from row 1 (columns 4 onwards for males, and same for females at row 24)
    # Column structure: 0=label, 1=Age Group, 2=Range, 3=North Dakota (state), 4+=counties

    # Male section starts at row 1 (header) with data rows 2-19
    # Female section starts at row 24 (header) with data rows 25-42

    # Get county names from male header row
    county_names = []
    for col in range(4, df.shape[1]):
        name = df.iloc[1, col]
        if pd.notna(name):
            county_names.append(str(name).replace(" County", "").strip())

    records = []

    # Extract male population (rows 2-19, columns 4 onwards)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 2  # Rows 2-19 contain male data
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            pop = df.iloc[data_row, col]
            if pd.notna(pop):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "sex": "male",
                        "population": int(pop),
                    }
                )

    # Extract female population (rows 25-42, columns 4 onwards)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 25  # Rows 25-42 contain female data
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            pop = df.iloc[data_row, col]
            if pd.notna(pop):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "sex": "female",
                        "population": int(pop),
                    }
                )

    result = pd.DataFrame(records)
    print(f"  Extracted {len(result)} population records for {len(county_names)} counties")
    return result


def extract_fertility_rates() -> pd.DataFrame:
    """
    Extract fertility rates by county and age group.

    SDC uses 5-year fertility rates (births per female over 5 years).

    Returns DataFrame with columns:
    - county_name: County name
    - age_group: Fertility age group (10-14 through 45-49)
    - fertility_rate: 5-year fertility rate
    """
    print("Extracting fertility rates...")

    df = pd.read_excel(PROJECTIONS_BASE_FILE, sheet_name="Fer 2020 - 2025", header=None)

    # Fertility rates are in rows 38-45 (for age groups 10-14 through 45-49)
    # Columns: 3=North Dakota (state), 4+=counties
    # Row 37 has headers

    # Get county names from row 37
    county_names = []
    for col in range(4, 57):  # Up to 53 counties + state
        name = df.iloc[37, col]
        if pd.notna(name):
            county_names.append(str(name).replace(" County", "").strip())

    records = []

    # Extract fertility rates (rows 38-45)
    for row_idx, age_group in enumerate(FERTILITY_AGE_GROUPS):
        data_row = row_idx + 38
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            rate = df.iloc[data_row, col]
            if pd.notna(rate):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "fertility_rate": float(rate),
                    }
                )

    result = pd.DataFrame(records)
    print(f"  Extracted {len(result)} fertility rate records for {len(county_names)} counties")
    return result


def extract_survival_rates() -> pd.DataFrame:
    """
    Extract survival rates by sex (state-level).

    SDC uses the same survival rates for all counties (from CDC life tables).

    Returns DataFrame with columns:
    - age_group: Age group
    - sex: "male" or "female"
    - survival_rate: 5-year survival rate (probability of surviving 5 years)
    """
    print("Extracting survival rates...")

    df = pd.read_excel(PROJECTIONS_BASE_FILE, sheet_name="5-Year Survival Rate By Sex", header=None)

    records = []

    # Male survival rates (rows 2-19, column 3 for 2020 rates)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 2
        rate = df.iloc[data_row, 4]  # Column 4 has the 2020 rates
        if pd.notna(rate):
            records.append({"age_group": age_group, "sex": "male", "survival_rate": float(rate)})

    # Female survival rates (rows 25-42, column 4 for 2020 rates)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 25
        rate = df.iloc[data_row, 4]  # Column 4 has the 2020 rates
        if pd.notna(rate):
            records.append({"age_group": age_group, "sex": "female", "survival_rate": float(rate)})

    result = pd.DataFrame(records)
    print(f"  Extracted {len(result)} survival rate records (state-level by sex)")
    return result


def extract_migration_rates() -> pd.DataFrame:
    """
    Extract migration rates by county, age group, and sex.

    These are the RAW migration rates from 2000-2020 (before any dampening).
    Migration rate represents net migration as a proportion of population.
    Positive = net in-migration, Negative = net out-migration.

    Returns DataFrame with columns:
    - county_name: County name
    - age_group: Age group
    - sex: "male" or "female"
    - migration_rate: 5-year net migration rate (as proportion)
    """
    print("Extracting migration rates...")

    df = pd.read_excel(PROJECTIONS_BASE_FILE, sheet_name="Mig_Rate", header=None)

    # Get county names from row 1 (columns 4 onwards)
    # Note: The column numbering uses odd numbers (1, 3, 5, ...) as FIPS-like codes
    county_names = []
    for col in range(4, df.shape[1]):
        name = df.iloc[1, col]
        if pd.notna(name):
            county_names.append(str(name).replace(" County", "").strip())

    records = []

    # Male migration rates (rows 2-19)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 2
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            rate = df.iloc[data_row, col]
            if pd.notna(rate):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "sex": "male",
                        "migration_rate": float(rate),
                    }
                )

    # Female migration rates (rows 25-42)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 25
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            rate = df.iloc[data_row, col]
            if pd.notna(rate):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "sex": "female",
                        "migration_rate": float(rate),
                    }
                )

    result = pd.DataFrame(records)
    print(f"  Extracted {len(result)} migration rate records for {len(county_names)} counties")
    return result


def extract_adjustment_factors() -> pd.DataFrame:
    """
    Extract adjustment factors by county and age group.

    These adjustments correct for discrepancies between projected and actual populations.
    They are applied to male populations only in the SDC methodology.

    Returns DataFrame with columns:
    - county_name: County name
    - age_group: Age group
    - sex: "male" (adjustments only apply to males in SDC methodology)
    - adjustment: Adjustment factor (typically negative)
    """
    print("Extracting adjustment factors...")

    df = pd.read_excel(PROJECTIONS_BASE_FILE, sheet_name="Adjustments 2020-2025", header=None)

    # Get county names from row 1
    county_names = []
    for col in range(4, df.shape[1]):
        name = df.iloc[1, col]
        if pd.notna(name):
            county_names.append(str(name).replace(" County", "").strip())

    records = []

    # Male adjustments (rows 2-19)
    for row_idx, age_group in enumerate(AGE_GROUPS):
        data_row = row_idx + 2
        for col_idx, county_name in enumerate(county_names):
            col = col_idx + 4
            adj = df.iloc[data_row, col]
            if pd.notna(adj):
                records.append(
                    {
                        "county_name": county_name,
                        "age_group": age_group,
                        "sex": "male",
                        "adjustment": float(adj),
                    }
                )

    result = pd.DataFrame(records)
    print(f"  Extracted {len(result)} adjustment factor records")
    return result


def main():
    """Main extraction routine."""
    print("=" * 60)
    print("SDC 2024 County Data Extraction")
    print("=" * 60)
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract county list
    counties = get_county_list()
    print(f"Found {len(counties)} counties in county list")
    print()

    # Extract base population
    base_pop = extract_base_population()
    base_pop_file = OUTPUT_DIR / "base_population_by_county.csv"
    base_pop.to_csv(base_pop_file, index=False)
    print(f"  Saved to: {base_pop_file}")
    print()

    # Extract fertility rates
    fertility = extract_fertility_rates()
    fertility_file = OUTPUT_DIR / "fertility_rates_by_county.csv"
    fertility.to_csv(fertility_file, index=False)
    print(f"  Saved to: {fertility_file}")
    print()

    # Extract survival rates
    survival = extract_survival_rates()
    survival_file = OUTPUT_DIR / "survival_rates_by_county.csv"
    survival.to_csv(survival_file, index=False)
    print(f"  Saved to: {survival_file}")
    print()

    # Extract migration rates
    migration = extract_migration_rates()
    migration_file = OUTPUT_DIR / "migration_rates_by_county.csv"
    migration.to_csv(migration_file, index=False)
    print(f"  Saved to: {migration_file}")
    print()

    # Extract adjustment factors (bonus - useful for replication)
    adjustments = extract_adjustment_factors()
    adjustments_file = OUTPUT_DIR / "adjustment_factors_by_county.csv"
    adjustments.to_csv(adjustments_file, index=False)
    print(f"  Saved to: {adjustments_file}")
    print()

    # Print summary statistics
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print()

    print("Base Population (Census 2020):")
    print(f"  Total counties: {base_pop['county_name'].nunique()}")
    print(f"  Total state population: {base_pop['population'].sum():,}")
    print(f"  Male population: {base_pop[base_pop['sex'] == 'male']['population'].sum():,}")
    print(f"  Female population: {base_pop[base_pop['sex'] == 'female']['population'].sum():,}")
    print()

    print("Fertility Rates:")
    print(f"  Counties with rates: {fertility['county_name'].nunique()}")
    print(f"  Age groups: {fertility['age_group'].nunique()}")
    print(
        f"  Mean rate (15-19): {fertility[fertility['age_group'] == '15-19']['fertility_rate'].mean():.4f}"
    )
    print(
        f"  Mean rate (25-29): {fertility[fertility['age_group'] == '25-29']['fertility_rate'].mean():.4f}"
    )
    print()

    print("Survival Rates (State-level):")
    male_surv = survival[survival["sex"] == "male"]
    female_surv = survival[survival["sex"] == "female"]
    print(
        f"  Male 0-4 survival: {male_surv[male_surv['age_group'] == '0-4']['survival_rate'].iloc[0]:.6f}"
    )
    print(
        f"  Female 0-4 survival: {female_surv[female_surv['age_group'] == '0-4']['survival_rate'].iloc[0]:.6f}"
    )
    print(
        f"  Male 85+ survival: {male_surv[male_surv['age_group'] == '85+']['survival_rate'].iloc[0]:.6f}"
    )
    print(
        f"  Female 85+ survival: {female_surv[female_surv['age_group'] == '85+']['survival_rate'].iloc[0]:.6f}"
    )
    print()

    print("Migration Rates (Raw, before dampening):")
    print(f"  Counties: {migration['county_name'].nunique()}")
    print(f"  Mean rate (all): {migration['migration_rate'].mean():.4f}")
    print(f"  Min rate: {migration['migration_rate'].min():.4f}")
    print(f"  Max rate: {migration['migration_rate'].max():.4f}")

    # Show counties with highest/lowest overall migration
    county_avg_mig = migration.groupby("county_name")["migration_rate"].mean().sort_values()
    print(
        f"  Highest out-migration county: {county_avg_mig.index[0]} ({county_avg_mig.iloc[0]:.4f})"
    )
    print(
        f"  Highest in-migration county: {county_avg_mig.index[-1]} ({county_avg_mig.iloc[-1]:.4f})"
    )
    print()

    print("=" * 60)
    print("Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
