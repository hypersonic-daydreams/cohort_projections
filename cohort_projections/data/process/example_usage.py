"""
Example usage script for base_population.py

Demonstrates how to use the base population processor with sample data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from base_population module
from cohort_projections.data.process.base_population import (
    harmonize_race_categories,
    create_cohort_matrix,
    validate_cohort_matrix,
    process_state_population,
    process_county_population,
    process_place_population,
    get_cohort_summary,
    RACE_ETHNICITY_MAP
)


def create_sample_state_data(n_records: int = 1000) -> pd.DataFrame:
    """
    Create sample state-level population data for testing.

    Args:
        n_records: Number of records to generate

    Returns:
        Sample DataFrame with required columns
    """
    np.random.seed(42)

    # Age distribution (more realistic)
    ages = np.random.choice(range(0, 91), n_records,
                           p=np.exp(-np.arange(91) * 0.03) / np.exp(-np.arange(91) * 0.03).sum())

    # Sex distribution (roughly equal)
    sexes = np.random.choice(['Male', 'Female'], n_records)

    # Race distribution (reflecting ND demographics)
    races = np.random.choice(
        ['WA_NH', 'BA_NH', 'IA_NH', 'AA_NH', 'TOM_NH', 'H'],
        n_records,
        p=[0.82, 0.03, 0.05, 0.02, 0.03, 0.05]  # Approximate ND distribution
    )

    # Population counts (100-500 per cell)
    populations = np.random.randint(100, 500, n_records)

    return pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'race_ethnicity': races,
        'population': populations
    })


def create_sample_county_data(n_counties: int = 53) -> pd.DataFrame:
    """
    Create sample county-level population data for testing.

    Args:
        n_counties: Number of counties (default 53 for ND)

    Returns:
        Sample DataFrame with county identifier
    """
    # Generate base data
    state_data = create_sample_state_data(n_records=5000)

    # Assign random counties (ND FIPS codes: 38001-38105, odd numbers only)
    county_fips = [f"38{str(i).zfill(3)}" for i in range(1, n_counties * 2, 2)]
    state_data['county'] = np.random.choice(county_fips, len(state_data))

    return state_data


def create_sample_place_data(n_places: int = 20) -> pd.DataFrame:
    """
    Create sample place-level population data for testing.

    Args:
        n_places: Number of places

    Returns:
        Sample DataFrame with place identifier
    """
    # Generate base data
    state_data = create_sample_state_data(n_records=2000)

    # Assign random place codes
    place_codes = [f"38{str(i * 1000).zfill(5)}" for i in range(1, n_places + 1)]
    state_data['place'] = np.random.choice(place_codes, len(state_data))

    return state_data


def example_1_basic_harmonization():
    """
    Example 1: Basic race category harmonization.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Race Category Harmonization")
    print("="*80)

    # Sample data with various race codes
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'race_ethnicity': ['WA_NH', 'NH_WHITE', 'NHWA', 'BA_NH', 'H', 'HISP'],
        'population': [1000, 950, 800, 150, 200, 180]
    })

    print("\nOriginal data:")
    print(sample_data)

    # Harmonize
    harmonized = harmonize_race_categories(sample_data)

    print("\nHarmonized data:")
    print(harmonized[['age', 'sex', 'race_ethnicity', 'population']])

    print("\nAvailable race mappings:")
    for code, category in sorted(RACE_ETHNICITY_MAP.items())[:10]:
        print(f"  {code:15} -> {category}")
    print(f"  ... and {len(RACE_ETHNICITY_MAP) - 10} more")


def example_2_cohort_matrix():
    """
    Example 2: Creating a cohort matrix.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Creating Cohort Matrix")
    print("="*80)

    # Create sample data
    sample_data = create_sample_state_data(n_records=500)

    print(f"\nSample data: {len(sample_data)} records")
    print(f"Age range: {sample_data['age'].min()}-{sample_data['age'].max()}")
    print(f"Total population: {sample_data['population'].sum():,}")

    # Harmonize and create matrix
    harmonized = harmonize_race_categories(sample_data)
    cohort_matrix = create_cohort_matrix(
        harmonized,
        geography_level='state',
        geography_id='38'
    )

    print(f"\nCohort matrix created: {len(cohort_matrix)} cells")
    print(f"Expected cells: 91 ages × 2 sexes × 6 races = {91*2*6} cells")
    print(f"Total population: {cohort_matrix['population'].sum():,.0f}")

    print("\nSample cohort matrix rows:")
    print(cohort_matrix.head(10))

    # Summary statistics
    summary = get_cohort_summary(cohort_matrix)
    print("\nPopulation summary:")
    print(summary)


def example_3_validation():
    """
    Example 3: Validating cohort matrices.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Validation")
    print("="*80)

    # Create sample data
    sample_data = create_sample_state_data(n_records=500)
    harmonized = harmonize_race_categories(sample_data)
    cohort_matrix = create_cohort_matrix(
        harmonized,
        geography_level='state',
        geography_id='38'
    )

    # Validate
    validation = validate_cohort_matrix(cohort_matrix, geography_level='state')

    print(f"\nValidation results:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Errors: {len(validation['errors'])}")
    print(f"  Warnings: {len(validation['warnings'])}")

    if validation['errors']:
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  - {error}")

    if validation['warnings']:
        print("\nWarnings (first 5):")
        for warning in validation['warnings'][:5]:
            print(f"  - {warning}")


def example_4_full_processing():
    """
    Example 4: Full processing pipeline (state, county, place).
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Full Processing Pipeline")
    print("="*80)

    # Create sample data for each level
    print("\n--- Processing State Level ---")
    state_data = create_sample_state_data(n_records=1000)
    print(f"Input records: {len(state_data)}")

    try:
        # Note: In production, specify a test output directory
        # For this example, we'll skip actual file writing
        harmonized = harmonize_race_categories(state_data)
        state_matrix = create_cohort_matrix(harmonized, 'state', '38')
        print(f"State matrix created: {len(state_matrix)} cells")
        print(f"Total population: {state_matrix['population'].sum():,.0f}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Processing County Level ---")
    county_data = create_sample_county_data(n_counties=53)
    print(f"Input records: {len(county_data)}")
    print(f"Counties: {county_data['county'].nunique()}")

    try:
        harmonized = harmonize_race_categories(county_data)
        # Process each county
        all_counties = []
        for county_id in sorted(county_data['county'].unique()):
            county_subset = harmonized[harmonized['county'] == county_id].copy()
            county_matrix = create_cohort_matrix(county_subset, 'county', county_id)
            all_counties.append(county_matrix)

        combined = pd.concat(all_counties, ignore_index=True)
        print(f"County matrices created: {len(combined)} total cells")
        print(f"Counties processed: {combined['geography_id'].nunique()}")
        print(f"Total population: {combined['population'].sum():,.0f}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Processing Place Level ---")
    place_data = create_sample_place_data(n_places=20)
    print(f"Input records: {len(place_data)}")
    print(f"Places: {place_data['place'].nunique()}")

    try:
        harmonized = harmonize_race_categories(place_data)
        # Process each place
        all_places = []
        for place_id in sorted(place_data['place'].unique()):
            place_subset = harmonized[harmonized['place'] == place_id].copy()
            place_matrix = create_cohort_matrix(place_subset, 'place', place_id)
            all_places.append(place_matrix)

        combined = pd.concat(all_places, ignore_index=True)
        print(f"Place matrices created: {len(combined)} total cells")
        print(f"Places processed: {combined['geography_id'].nunique()}")
        print(f"Total population: {combined['population'].sum():,.0f}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("BASE POPULATION PROCESSOR - EXAMPLE USAGE")
    print("="*80)
    print("\nThis script demonstrates the core functionality of base_population.py")
    print("using synthetic data that mimics Census PEP/ACS structure.")

    try:
        example_1_basic_harmonization()
        example_2_cohort_matrix()
        example_3_validation()
        example_4_full_processing()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
