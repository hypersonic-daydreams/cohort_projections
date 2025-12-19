"""
Example: Processing Migration Rates for Cohort Projections

Demonstrates how to:
1. Generate synthetic IRS county-to-county migration data
2. Generate synthetic international migration data
3. Process migration data through distribution pipeline
4. Create migration rates for projection engine
5. Integrate with complete projection workflow

This example uses synthetic data but shows the exact workflow for real IRS/Census data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import migration processing functions
from cohort_projections.data.process.migration_rates import (
    load_irs_migration_data,
    load_international_migration_data,
    get_standard_age_migration_pattern,
    distribute_migration_by_age,
    distribute_migration_by_sex,
    distribute_migration_by_race,
    calculate_net_migration,
    create_migration_rate_table,
    validate_migration_data,
    process_migration_rates
)

print("=" * 80)
print("MIGRATION RATES PROCESSING EXAMPLE")
print("=" * 80)
print()

# Setup paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
raw_migration_dir = data_dir / "raw" / "migration"
processed_dir = data_dir / "processed"

# Create directories
raw_migration_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1: Generate Synthetic IRS County-to-County Migration Data
# =============================================================================

print("PART 1: Generating Synthetic IRS Migration Data")
print("-" * 80)

# North Dakota counties (sample)
nd_counties = [
    '38001',  # Adams
    '38003',  # Barnes
    '38005',  # Benson
    '38017',  # Cass (Fargo area)
    '38035',  # Grand Forks
    '38059',  # Morton (Bismarck area)
    '38101',  # Ward (Minot area)
]

# Nearby states' counties (for out-migration)
nearby_counties = [
    '27053',  # Hennepin, MN (Minneapolis)
    '27123',  # Ramsey, MN (St. Paul)
    '46099',  # Minnehaha, SD (Sioux Falls)
    '31055',  # Douglas, NE (Omaha)
]

all_counties = nd_counties + nearby_counties

# Generate synthetic IRS migration flows
np.random.seed(42)

irs_flows = []
years = range(2018, 2023)

for year in years:
    for from_county in all_counties:
        for to_county in all_counties:
            if from_county != to_county:
                # Determine migration volume based on county types
                from_nd = from_county.startswith('38')
                to_nd = to_county.startswith('38')

                if from_nd and to_nd:
                    # Within ND: moderate flows
                    base_migrants = np.random.randint(20, 100)
                elif from_nd and not to_nd:
                    # ND to outside: out-migration
                    base_migrants = np.random.randint(50, 200)
                elif not from_nd and to_nd:
                    # Outside to ND: in-migration
                    base_migrants = np.random.randint(40, 180)
                else:
                    # Outside to outside: skip
                    continue

                # Add some variation by year
                year_factor = 1.0 + (year - 2020) * 0.05  # Slight trend
                migrants = int(base_migrants * year_factor)

                irs_flows.append({
                    'from_county_fips': from_county,
                    'to_county_fips': to_county,
                    'migrants': migrants,
                    'year': year
                })

irs_df = pd.DataFrame(irs_flows)

# Save synthetic IRS data
irs_file = raw_migration_dir / "irs_flows_2018_2022.csv"
irs_df.to_csv(irs_file, index=False)

print(f"Created synthetic IRS data: {len(irs_df)} migration flows")
print(f"Saved to: {irs_file}")
print(f"\nSample IRS flows:")
print(irs_df.head(10))
print()

# =============================================================================
# PART 2: Generate Synthetic International Migration Data
# =============================================================================

print("\nPART 2: Generating Synthetic International Migration Data")
print("-" * 80)

intl_migration = []

for year in years:
    for county in nd_counties:
        # ND has modest international migration
        # Larger counties (Cass, Grand Forks) have more
        if county == '38017':  # Cass (Fargo)
            base = np.random.randint(400, 600)
        elif county == '38035':  # Grand Forks
            base = np.random.randint(200, 350)
        elif county in ['38059', '38101']:  # Bismarck, Minot
            base = np.random.randint(100, 200)
        else:
            base = np.random.randint(10, 50)

        intl_migration.append({
            'county_fips': county,
            'international_migrants': base,
            'year': year
        })

intl_df = pd.DataFrame(intl_migration)

# Save synthetic international migration data
intl_file = raw_migration_dir / "international_2018_2022.csv"
intl_df.to_csv(intl_file, index=False)

print(f"Created synthetic international migration data: {len(intl_df)} records")
print(f"Saved to: {intl_file}")
print(f"\nSample international migration:")
print(intl_df.groupby('county_fips')['international_migrants'].mean())
print()

# =============================================================================
# PART 3: Generate Synthetic Base Population (for distribution)
# =============================================================================

print("\nPART 3: Generating Synthetic Base Population")
print("-" * 80)

# Standard categories
races = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)"
]

sexes = ["Male", "Female"]
ages = list(range(91))  # 0-90

# Generate synthetic population (simplified - not county-specific)
population_records = []

for age in ages:
    for sex in sexes:
        for race in races:
            # Assign population based on typical ND demographics
            if race == "White alone, Non-Hispanic":
                base_pop = np.random.randint(5000, 8000)
            elif race == "AIAN alone, Non-Hispanic":
                base_pop = np.random.randint(200, 500)
            elif race == "Hispanic (any race)":
                base_pop = np.random.randint(100, 300)
            else:
                base_pop = np.random.randint(50, 200)

            # Age pattern
            if 20 <= age <= 40:
                pop = base_pop * 1.2  # More people in working ages
            elif age < 18:
                pop = base_pop * 0.8
            elif age >= 65:
                pop = base_pop * 0.7
            else:
                pop = base_pop

            population_records.append({
                'age': age,
                'sex': sex,
                'race_ethnicity': race,
                'population': int(pop)
            })

population_df = pd.DataFrame(population_records)

# Save synthetic base population
pop_file = processed_dir / "base_population.parquet"
population_df.to_parquet(pop_file, compression='gzip', index=False)

print(f"Created synthetic base population: {len(population_df)} cohorts")
print(f"Saved to: {pop_file}")
print(f"\nTotal population: {population_df['population'].sum():,}")
print(f"\nPopulation by race:")
print(population_df.groupby('race_ethnicity')['population'].sum().sort_values(ascending=False))
print()

# =============================================================================
# PART 4: Demonstrate Age Pattern Generation
# =============================================================================

print("\nPART 4: Demonstrating Migration Age Patterns")
print("-" * 80)

# Simplified pattern
print("Simplified Age Pattern (default):")
simplified_pattern = get_standard_age_migration_pattern(peak_age=25, method='simplified')
print(simplified_pattern.head(15))
print(f"\nPropensities sum to: {simplified_pattern['migration_propensity'].sum():.6f}")
print(f"Peak age: {simplified_pattern.loc[simplified_pattern['migration_propensity'].idxmax(), 'age']}")

# Rogers-Castro pattern
print("\n\nRogers-Castro Age Pattern (optional):")
rogers_castro_pattern = get_standard_age_migration_pattern(peak_age=25, method='rogers_castro')
print(rogers_castro_pattern.head(15))
print(f"\nPropensities sum to: {rogers_castro_pattern['migration_propensity'].sum():.6f}")
print(f"Peak age: {rogers_castro_pattern.loc[rogers_castro_pattern['migration_propensity'].idxmax(), 'age']}")
print()

# =============================================================================
# PART 5: Step-by-Step Distribution (Manual Process)
# =============================================================================

print("\nPART 5: Manual Step-by-Step Distribution")
print("-" * 80)

# Calculate net migration from IRS data
print("Step 1: Calculate Net Domestic Migration")
nd_irs = irs_df[
    (irs_df['from_county_fips'].str.startswith('38')) |
    (irs_df['to_county_fips'].str.startswith('38'))
]

in_migration_total = nd_irs[nd_irs['to_county_fips'].str.startswith('38')]['migrants'].sum()
out_migration_total = nd_irs[nd_irs['from_county_fips'].str.startswith('38')]['migrants'].sum()
net_domestic = in_migration_total - out_migration_total

print(f"  In-migration:  {in_migration_total:,}")
print(f"  Out-migration: {out_migration_total:,}")
print(f"  Net domestic:  {net_domestic:+,}")

# Add international migration
print("\nStep 2: Add International Migration")
net_international = intl_df['international_migrants'].sum()
print(f"  Net international: {net_international:+,}")

total_net = net_domestic + net_international
print(f"  Total net migration: {total_net:+,}")

# Distribute to ages
print("\nStep 3: Distribute to Ages")
age_migration = distribute_migration_by_age(total_net, simplified_pattern)
print(f"  Created age distribution: {len(age_migration)} ages")
print(f"  Total preserved: {age_migration['migrants'].sum():,.1f}")
print(f"\n  Top 5 ages by migration:")
print(age_migration.nlargest(5, 'migrants'))

# Distribute to sex
print("\nStep 4: Distribute to Sex")
age_sex_migration = distribute_migration_by_sex(age_migration, sex_ratio=0.5)
print(f"  Created age-sex distribution: {len(age_sex_migration)} combinations")
print(f"  Total preserved: {age_sex_migration['migrants'].sum():,.1f}")
print(f"\n  Sample:")
print(age_sex_migration.head(10))

# Distribute to race
print("\nStep 5: Distribute to Race/Ethnicity")
age_sex_race_migration = distribute_migration_by_race(age_sex_migration, population_df)
print(f"  Created age-sex-race distribution: {len(age_sex_race_migration)} cohorts")
print(f"  Total preserved: {age_sex_race_migration['migrants'].sum():,.1f}")

# Rename for consistency
age_sex_race_migration.rename(columns={'migrants': 'net_migration'}, inplace=True)

print(f"\n  Migration by race:")
print(age_sex_race_migration.groupby('race_ethnicity')['net_migration'].sum().sort_values(ascending=False))
print()

# =============================================================================
# PART 6: Create Complete Migration Table
# =============================================================================

print("\nPART 6: Creating Complete Migration Table")
print("-" * 80)

migration_table = create_migration_rate_table(
    age_sex_race_migration,
    population_df=population_df,
    as_rates=False,  # Use absolute numbers
    validate=True
)

print(f"Created complete migration table: {len(migration_table)} cohorts")
print(f"All 1,092 cohorts present: {len(migration_table) == 1092}")
print(f"\nSample:")
print(migration_table.head(20))
print(f"\nMigration summary:")
print(f"  Positive net migration cohorts: {(migration_table['net_migration'] > 0).sum()}")
print(f"  Negative net migration cohorts: {(migration_table['net_migration'] < 0).sum()}")
print(f"  Zero migration cohorts: {(migration_table['net_migration'] == 0).sum()}")
print(f"  Total net migration: {migration_table['net_migration'].sum():+,.0f}")
print()

# =============================================================================
# PART 7: Validate Migration Data
# =============================================================================

print("\nPART 7: Validating Migration Data")
print("-" * 80)

validation_result = validate_migration_data(migration_table, population_df)

print(f"Validation result: {'PASSED' if validation_result['valid'] else 'FAILED'}")
print(f"\nTotal net migration: {validation_result['total_net_migration']:+,.0f}")
print(f"\nMigration by direction:")
for key, value in validation_result['net_by_direction'].items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:,.0f}")
    else:
        print(f"  {key}: {value}")

if validation_result['warnings']:
    print(f"\nValidation warnings ({len(validation_result['warnings'])}):")
    for warning in validation_result['warnings']:
        print(f"  - {warning}")

if validation_result['errors']:
    print(f"\nValidation errors ({len(validation_result['errors'])}):")
    for error in validation_result['errors']:
        print(f"  - {error}")
print()

# =============================================================================
# PART 8: Complete Pipeline (Single Function Call)
# =============================================================================

print("\nPART 8: Complete Pipeline with Single Function Call")
print("-" * 80)

# This is the recommended way for production use
print("Running complete migration processing pipeline...")
print()

migration_rates_final = process_migration_rates(
    irs_path=irs_file,
    intl_path=intl_file,
    population_path=pop_file,
    year_range=(2018, 2022),
    target_county_fips='38',  # North Dakota
    as_rates=False  # Use absolute numbers
)

print(f"\nFinal migration rates table: {len(migration_rates_final)} cohorts")
print(f"\nOutput files created in: {processed_dir / 'migration'}")
print(f"  - migration_rates.parquet")
print(f"  - migration_rates.csv")
print(f"  - migration_rates_metadata.json")
print()

# =============================================================================
# PART 9: Integration with Projection Engine
# =============================================================================

print("\nPART 9: Integration with Projection Engine")
print("-" * 80)

print("Migration rates are now ready for the projection engine:")
print()
print("Example usage:")
print("""
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=population_df,
    fertility_rates=fertility_df,
    survival_rates=survival_df,
    migration_rates=migration_rates_final  # <-- Our processed migration
)

results = projection.run_projection(
    start_year=2025,
    end_year=2045
)
""")
print()

# =============================================================================
# PART 10: Summary and Key Takeaways
# =============================================================================

print("\nPART 10: Summary and Key Takeaways")
print("=" * 80)

print("""
KEY POINTS:

1. DATA SOURCES:
   - IRS county-to-county flows (aggregate, no demographics)
   - Census/ACS international migration (aggregate, no demographics)
   - Base population (needed for race distribution)

2. DISTRIBUTION ALGORITHM:
   - Age: Standard demographic pattern (simplified or Rogers-Castro)
   - Sex: 50/50 split (configurable)
   - Race: Proportional to population composition

3. OUTPUT:
   - 1,092 cohorts (91 ages × 2 sexes × 6 races)
   - Net migration (can be positive or negative)
   - Absolute numbers OR rates (configurable)

4. VALIDATION:
   - All cohorts present
   - Age pattern plausible (peaks at 20-35)
   - Migration not extreme (< 20% of population)
   - Won't cause negative populations

5. PRODUCTION USAGE:
   Use process_migration_rates() for complete pipeline:
   - Handles all distribution steps
   - Validates results
   - Saves multiple output formats
   - Generates metadata

6. REAL DATA:
   Replace synthetic data with:
   - IRS SOI Migration Data (download from IRS.gov)
   - Census Population Estimates (from Census API or downloads)
   - Same workflow applies!
""")

print("=" * 80)
print("MIGRATION RATES PROCESSING EXAMPLE COMPLETE")
print("=" * 80)
print()
print(f"Check output directory: {processed_dir / 'migration'}")
print()
