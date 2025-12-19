"""
Example script demonstrating basic cohort component projection.

This script shows how to:
1. Load or create input data (population, fertility, survival, migration)
2. Initialize the projection engine
3. Run a projection
4. Export and analyze results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.core import CohortComponentProjection
from cohort_projections.utils.config_loader import ConfigLoader


def create_sample_data():
    """
    Create sample input data for demonstration.

    In a real application, this data would come from:
    - Population: Census/ACS base population estimates
    - Fertility: SEER age-specific fertility rates
    - Survival: SEER life tables or CDC mortality data
    - Migration: IRS county flows + Census international migration
    """
    print("Creating sample input data...")

    # Configuration
    ages = list(range(0, 91))  # 0-90 (90+ is open-ended)
    sexes = ['Male', 'Female']
    races = [
        'White alone, Non-Hispanic',
        'Black alone, Non-Hispanic',
        'Hispanic (any race)'
    ]

    # 1. Base Population (2025)
    print("  - Base population...")
    base_pop_data = []

    for age in ages:
        for sex in sexes:
            for race in races:
                # Create realistic age distribution
                # Young ages: higher population
                # Working ages: peak
                # Older ages: declining
                if age < 18:
                    base_pop = 1000
                elif age < 25:
                    base_pop = 1200
                elif age < 65:
                    base_pop = 1500
                else:
                    base_pop = max(100, 1500 - (age - 65) * 50)

                # Vary by race (simple example)
                if race == 'White alone, Non-Hispanic':
                    base_pop *= 3.0
                elif race == 'Hispanic (any race)':
                    base_pop *= 1.5

                base_pop_data.append({
                    'year': 2025,
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'population': base_pop
                })

    base_population = pd.DataFrame(base_pop_data)
    total_pop = base_population['population'].sum()
    print(f"    Total base population: {total_pop:,.0f}")

    # 2. Fertility Rates
    print("  - Fertility rates...")
    fertility_data = []

    for age in range(15, 50):  # Reproductive ages
        for race in races:
            # Age-specific fertility pattern (typical U.S. pattern)
            if age < 20:
                rate = 0.02  # Low for teens
            elif age < 25:
                rate = 0.08
            elif age < 30:
                rate = 0.12  # Peak fertility
            elif age < 35:
                rate = 0.10
            elif age < 40:
                rate = 0.05
            else:
                rate = 0.01  # Low for late 40s

            # Vary by race (general patterns)
            if race == 'Hispanic (any race)':
                rate *= 1.2  # Slightly higher Hispanic fertility
            elif race == 'Black alone, Non-Hispanic':
                rate *= 1.1

            fertility_data.append({
                'age': age,
                'race': race,
                'fertility_rate': rate
            })

    fertility_rates = pd.DataFrame(fertility_data)
    print(f"    Fertility rate records: {len(fertility_rates)}")

    # 3. Survival Rates
    print("  - Survival rates...")
    survival_data = []

    for age in ages:
        for sex in sexes:
            for race in races:
                # Age-specific survival pattern
                if age == 0:
                    # Infant mortality
                    survival_rate = 0.9935  # ~6.5 deaths per 1000 births
                elif age < 1:
                    survival_rate = 0.9990
                elif age < 15:
                    survival_rate = 0.9995  # Very low child mortality
                elif age < 45:
                    survival_rate = 0.9990
                elif age < 65:
                    survival_rate = 0.995
                elif age < 75:
                    survival_rate = 0.98
                elif age < 85:
                    survival_rate = 0.93
                elif age < 90:
                    survival_rate = 0.85
                else:
                    survival_rate = 0.70  # 90+ group

                # Sex differential (females live longer)
                if sex == 'Male':
                    survival_rate *= 0.995  # Slightly lower male survival

                # Race differential (simplified)
                if race == 'Black alone, Non-Hispanic':
                    survival_rate *= 0.997  # Small mortality disadvantage

                # Ensure within bounds
                survival_rate = max(0.5, min(1.0, survival_rate))

                survival_data.append({
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'survival_rate': survival_rate
                })

    survival_rates = pd.DataFrame(survival_data)
    print(f"    Survival rate records: {len(survival_rates)}")

    # 4. Migration Rates
    print("  - Migration rates...")
    migration_data = []

    for age in ages:
        for sex in sexes:
            for race in races:
                # Age-specific migration pattern
                # Young adults most mobile
                if age < 18:
                    net_mig = 5  # Children migrate with parents
                elif age < 25:
                    net_mig = 50  # College age
                elif age < 35:
                    net_mig = 100  # Young professionals (peak)
                elif age < 45:
                    net_mig = 30
                elif age < 65:
                    net_mig = 10
                else:
                    net_mig = -5  # Slight out-migration of retirees

                # Vary by race
                if race == 'Hispanic (any race)':
                    net_mig *= 1.5  # Higher Hispanic migration

                migration_data.append({
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'net_migration': net_mig
                })

    migration_rates = pd.DataFrame(migration_data)
    print(f"    Migration rate records: {len(migration_rates)}")

    return base_population, fertility_rates, survival_rates, migration_rates


def main():
    """Run example projection."""
    print("=" * 70)
    print("Cohort Component Projection - Example Run")
    print("=" * 70)
    print()

    # Create sample data
    base_population, fertility_rates, survival_rates, migration_rates = create_sample_data()
    print()

    # Initialize projection
    print("Initializing projection engine...")
    projection = CohortComponentProjection(
        base_population=base_population,
        fertility_rates=fertility_rates,
        survival_rates=survival_rates,
        migration_rates=migration_rates
    )
    print()

    # Run projection (5 years for quick demo)
    print("Running projection (2025-2030)...")
    results = projection.run_projection(
        start_year=2025,
        end_year=2030,
        scenario='baseline'
    )
    print()

    # Display results summary
    print("=" * 70)
    print("PROJECTION RESULTS SUMMARY")
    print("=" * 70)
    print()

    summary = projection.get_projection_summary()
    print(summary[['year', 'total_population', 'male_population', 'female_population']])
    print()

    # Population growth
    print("Population Growth:")
    for i in range(len(summary)):
        year = summary.iloc[i]['year']
        total = summary.iloc[i]['total_population']
        if i > 0:
            prev_total = summary.iloc[i-1]['total_population']
            change = total - prev_total
            pct_change = (change / prev_total) * 100
            print(f"  {year}: {total:,.0f} ({change:+,.0f}, {pct_change:+.2f}%)")
        else:
            print(f"  {year}: {total:,.0f} (base year)")
    print()

    # Age structure
    print("Age Structure (2030):")
    final_pop = projection.get_population_by_year(2030)
    under_18 = final_pop[final_pop['age'] < 18]['population'].sum()
    working_age = final_pop[(final_pop['age'] >= 18) & (final_pop['age'] < 65)]['population'].sum()
    seniors = final_pop[final_pop['age'] >= 65]['population'].sum()
    total = final_pop['population'].sum()

    print(f"  Under 18:    {under_18:>12,.0f}  ({under_18/total*100:>5.1f}%)")
    print(f"  Working Age: {working_age:>12,.0f}  ({working_age/total*100:>5.1f}%)")
    print(f"  65+:         {seniors:>12,.0f}  ({seniors/total*100:>5.1f}%)")
    print()

    # Export results
    output_dir = project_root / "output" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting results...")
    projection.export_results(
        output_dir / "example_projection.parquet",
        format='parquet'
    )
    projection.export_summary(
        output_dir / "example_summary.csv",
        format='csv'
    )
    print(f"  Results saved to: {output_dir}")
    print()

    print("=" * 70)
    print("Example projection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
