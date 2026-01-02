"""
Example script for processing survival rates from SEER/CDC life tables.

Demonstrates the complete pipeline for converting life tables into survival
rates for use in cohort component population projections.

This script shows:
1. Creating synthetic life table data for testing
2. Processing life tables into survival rates
3. Calculating life expectancy for validation
4. Integrating with the projection engine
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.survival_rates import (  # noqa: E402
    apply_mortality_improvement,
    calculate_life_expectancy,
    calculate_survival_rates_from_life_table,
    create_survival_rate_table,
    harmonize_mortality_race_categories,
    load_life_table_data,
    process_survival_rates,
    validate_survival_rates,
)
from cohort_projections.utils import get_logger_from_config  # noqa: E402

logger = get_logger_from_config(__name__)


def create_synthetic_life_table():
    """
    Create synthetic life table data for testing.

    Generates realistic life table with:
    - Ages 0-90
    - Males and Females
    - 6 race/ethnicity categories
    - Typical U.S. mortality patterns

    Returns:
        DataFrame with synthetic life table data
    """
    logger.info("Creating synthetic life table data...")

    data = []
    races = ["White NH", "Black NH", "Hispanic", "AIAN NH", "Asian/PI NH", "Two+ Races NH"]
    sexes = ["Male", "Female"]

    for sex in sexes:
        for race in races:
            # Start with radix of 100,000
            lx = 100000

            # Temporary storage for this race-sex group
            group_data = []

            for age in range(91):
                # Calculate age-specific death probability (qx)
                if age == 0:
                    qx = 0.0060  # Infant mortality ~6 per 1000
                elif age < 15:
                    qx = 0.0002 + age * 0.00001  # Very low child mortality
                elif age < 45:
                    qx = 0.0008 + (age - 15) * 0.00003  # Increasing slowly
                elif age < 65:
                    qx = 0.0025 + (age - 45) * 0.0002  # Faster increase
                elif age < 85:
                    qx = 0.012 + (age - 65) * 0.006  # Rapid increase
                else:
                    qx = 0.15 + (age - 85) * 0.03  # Very high elderly

                # Adjust by sex (females live longer)
                if sex == "Female":
                    qx *= 0.80  # 20% lower mortality

                # Adjust by race (based on CDC patterns)
                if "Hispanic" in race or "Asian" in race:
                    qx *= 0.85  # Hispanic and Asian/PI have lower mortality
                elif "Black" in race:
                    qx *= 1.20  # Black NH has higher mortality
                elif "AIAN" in race:
                    qx *= 1.25  # AIAN has higher mortality

                # Cap at 1.0
                qx = min(qx, 1.0)

                # Calculate dx (deaths in interval)
                dx = lx * qx

                # Calculate person_years (person-years lived in interval)
                # For infants, use special formula
                if age == 0:
                    person_years = 0.1 * lx + 0.9 * (lx - dx)  # Infants die early in year
                elif age == 90:
                    # Open-ended interval - use formula from life table theory
                    # Assume stationary population
                    person_years = lx / qx if qx < 1.0 else lx * 0.5
                else:
                    person_years = (lx + (lx - dx)) / 2  # Average population

                group_data.append({"age": age, "lx": lx, "qx": qx, "dx": dx, "Lx": person_years})

                # Update lx for next age
                lx = lx - dx
                if lx < 0:
                    lx = 0

            # Calculate Tx (total person-years) backwards
            for i in range(90, -1, -1):
                if i == 90:
                    group_data[i]["Tx"] = group_data[i]["Lx"]
                else:
                    group_data[i]["Tx"] = group_data[i]["Lx"] + group_data[i + 1]["Tx"]

            # Calculate ex (life expectancy)
            for i in range(91):
                if group_data[i]["lx"] > 0:
                    group_data[i]["ex"] = group_data[i]["Tx"] / group_data[i]["lx"]
                else:
                    group_data[i]["ex"] = 0

            # Add sex and race to each record
            for record in group_data:
                record["sex"] = sex
                record["race"] = race
                data.append(record)

    df = pd.DataFrame(data)

    logger.info(f"Created life table with {len(df)} records")
    logger.info(f"  Ages: {df['age'].min()}-{df['age'].max()}")
    logger.info(f"  Sexes: {df['sex'].nunique()}")
    logger.info(f"  Races: {df['race'].nunique()}")
    logger.info(
        f"  Life expectancy at birth (e0) range: {df[df['age'] == 0]['ex'].min():.1f}-{df[df['age'] == 0]['ex'].max():.1f}"
    )

    return df


def demo_step_by_step_processing():
    """Demonstrate step-by-step processing of survival rates."""
    logger.info("=" * 70)
    logger.info("DEMO 1: Step-by-Step Survival Rate Processing")
    logger.info("=" * 70)

    # Create synthetic life table
    life_table = create_synthetic_life_table()

    # Save to temporary file
    temp_dir = project_root / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "synthetic_life_table.csv"

    life_table.to_csv(temp_file, index=False)
    logger.info(f"Saved synthetic life table to {temp_file}")

    # Step 1: Load data
    logger.info("\n--- Step 1: Load Life Table ---")
    df = load_life_table_data(temp_file)
    logger.info(f"Loaded {len(df)} records")

    # Step 2: Harmonize race categories
    logger.info("\n--- Step 2: Harmonize Race Categories ---")
    df = harmonize_mortality_race_categories(df)
    logger.info(f"Harmonized to {df['race_ethnicity'].nunique()} race categories")

    # Step 3: Calculate survival rates
    logger.info("\n--- Step 3: Calculate Survival Rates ---")
    survival_df = calculate_survival_rates_from_life_table(df, method="lx")
    logger.info(f"Calculated survival rates for {len(survival_df)} cohorts")
    logger.info(f"  Mean survival rate: {survival_df['survival_rate'].mean():.6f}")
    logger.info(
        f"  Min survival rate: {survival_df['survival_rate'].min():.6f} (age {survival_df.loc[survival_df['survival_rate'].idxmin(), 'age']:.0f})"
    )
    logger.info(
        f"  Max survival rate: {survival_df['survival_rate'].max():.6f} (age {survival_df.loc[survival_df['survival_rate'].idxmax(), 'age']:.0f})"
    )

    # Step 4: Apply mortality improvement (optional)
    logger.info("\n--- Step 4: Apply Mortality Improvement ---")
    improved_df = apply_mortality_improvement(
        survival_df, base_year=2020, projection_year=2030, improvement_factor=0.005
    )
    logger.info("Applied 10 years of 0.5% annual improvement")
    logger.info(
        f"  Mean survival rate improved from {survival_df['survival_rate'].mean():.6f} to {improved_df['survival_rate'].mean():.6f}"
    )

    # Step 5: Create complete survival table
    logger.info("\n--- Step 5: Create Complete Survival Rate Table ---")
    survival_table = create_survival_rate_table(improved_df, validate=True)
    logger.info(f"Created table with {len(survival_table)} cells")

    # Step 6: Calculate life expectancy
    logger.info("\n--- Step 6: Calculate Life Expectancy (e0) ---")
    life_exp = calculate_life_expectancy(survival_table)
    logger.info("Life expectancy at birth by sex-race:")
    for key, value in sorted(life_exp.items()):
        logger.info(f"  {key}: {value:.1f} years")

    # Step 7: Validate
    logger.info("\n--- Step 7: Validate Survival Rates ---")
    validation = validate_survival_rates(survival_table)
    logger.info(f"Validation result: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation["errors"]:
        logger.error(f"Errors: {validation['errors']}")
    if validation["warnings"]:
        logger.warning(f"Warnings: {validation['warnings']}")

    logger.info("\n" + "=" * 70)
    logger.info("Step-by-step processing complete!")
    logger.info("=" * 70)

    return survival_table


def demo_complete_pipeline():
    """Demonstrate complete pipeline with single function call."""
    logger.info("\n\n" + "=" * 70)
    logger.info("DEMO 2: Complete Pipeline (Single Function)")
    logger.info("=" * 70)

    # Create synthetic life table
    life_table = create_synthetic_life_table()

    # Save to file
    temp_dir = project_root / "data" / "raw" / "mortality"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "synthetic_life_table_2020.csv"

    life_table.to_csv(temp_file, index=False)
    logger.info(f"Saved synthetic life table to {temp_file}")

    # Process in one function call
    logger.info("\n--- Processing with single function call ---")
    survival_rates = process_survival_rates(
        input_path=temp_file, base_year=2020, improvement_factor=0.005
    )

    logger.info(f"\nProcessed {len(survival_rates)} survival rate records")
    logger.info("Output files created in: data/processed/mortality/")

    logger.info("\n" + "=" * 70)
    logger.info("Complete pipeline demo finished!")
    logger.info("=" * 70)

    return survival_rates


def demo_integration_with_projection():
    """Demonstrate integration with cohort component projection engine."""
    logger.info("\n\n" + "=" * 70)
    logger.info("DEMO 3: Integration with Projection Engine")
    logger.info("=" * 70)

    # This would integrate with the actual projection engine
    # For now, just show how the data would be used

    logger.info("\nSurvival rates are ready for use in projection engine:")
    logger.info("""
    from cohort_projections.core import CohortComponentProjection
    from cohort_projections.data.process import process_survival_rates

    # Process survival rates
    survival_rates = process_survival_rates(
        input_path='data/raw/mortality/seer_lifetables_2020.csv',
        base_year=2020,
        improvement_factor=0.005
    )

    # Use in projection
    projection = CohortComponentProjection(
        base_population=base_pop_df,
        fertility_rates=fertility_df,
        survival_rates=survival_rates,  # <-- Use processed rates
        migration_rates=migration_df
    )

    # Run projection
    results = projection.run_projection(
        start_year=2025,
        end_year=2045
    )
    """)

    logger.info("=" * 70)
    logger.info("Integration demo complete!")
    logger.info("=" * 70)


def main():
    """Run all demonstrations."""
    logger.info("=" * 70)
    logger.info("SURVIVAL RATES PROCESSING - EXAMPLE DEMONSTRATIONS")
    logger.info("=" * 70)

    # Demo 1: Step-by-step processing
    demo_step_by_step_processing()

    # Demo 2: Complete pipeline
    demo_complete_pipeline()

    # Demo 3: Integration example
    demo_integration_with_projection()

    # Summary
    logger.info("\n\n" + "=" * 70)
    logger.info("ALL DEMONSTRATIONS COMPLETE")
    logger.info("=" * 70)
    logger.info("\nKey takeaways:")
    logger.info("1. Life tables can be processed from multiple formats (CSV, Excel, etc.)")
    logger.info("2. Multiple conversion methods supported (lx, qx, Lx)")
    logger.info("3. Special handling for age 90+ open-ended group")
    logger.info("4. Mortality improvement can be applied for future projections")
    logger.info("5. Comprehensive validation ensures data quality")
    logger.info("6. Life expectancy calculated for quality assurance")
    logger.info("7. Output ready for direct use in projection engine")

    logger.info("\nOutput files:")
    logger.info("  - data/processed/mortality/survival_rates.parquet")
    logger.info("  - data/processed/mortality/survival_rates.csv")
    logger.info("  - data/processed/mortality/survival_rates_metadata.json")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
