"""
Example: Process fertility rates for cohort projections.

Demonstrates the complete fertility rate processing pipeline from raw SEER data
to projection-ready format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.fertility_rates import (
    load_seer_fertility_data,
    harmonize_fertility_race_categories,
    calculate_average_fertility_rates,
    create_fertility_rate_table,
    validate_fertility_rates,
    process_fertility_rates
)
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


def create_sample_seer_data(output_path: Path) -> None:
    """
    Create sample SEER-format fertility data for testing.

    Creates realistic age-specific fertility rates following typical
    demographic patterns:
    - Peak fertility around age 28
    - Lower rates for younger and older women
    - Variation by race/ethnicity reflecting U.S. patterns
    """
    logger.info("Creating sample SEER fertility data")

    sample_data = []

    # SEER-style race codes (as they appear in real data)
    races = {
        'White NH': 1.8,      # TFR target
        'Black NH': 1.7,
        'Hispanic': 2.0,
        'AIAN NH': 1.9,
        'Asian/PI NH': 1.5,
        'Two+ Races NH': 1.8
    }

    # Generate 5 years of data (2018-2022)
    for year in range(2018, 2023):
        for age in range(15, 50):  # Reproductive ages
            for race, target_tfr in races.items():

                # Create realistic age pattern (Hadwiger function approximation)
                # Peak around age 28, standard deviation ~6 years
                age_effect = np.exp(-((age - 28) ** 2) / 72)

                # Scale to achieve target TFR (sum across ages ≈ TFR)
                base_rate = (target_tfr / 35) * age_effect  # 35 ages, adjust scaling

                # Add small random variation by year (±5%)
                yearly_variation = 1 + np.random.uniform(-0.05, 0.05)
                fertility_rate = base_rate * yearly_variation

                # Simulate female population (for weighted averaging)
                # Larger populations for White, Hispanic; smaller for others
                base_pop = 5000 if race in ['White NH', 'Hispanic'] else 500
                population = base_pop + np.random.randint(-100, 100)

                # Calculate births (for metadata)
                births = int(population * fertility_rate)

                sample_data.append({
                    'year': year,
                    'age': age,
                    'race': race,  # SEER codes that need harmonization
                    'fertility_rate': round(fertility_rate, 6),
                    'births': births,
                    'population': population
                })

    df = pd.DataFrame(sample_data)

    # Save to CSV (SEER format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample data: {len(df)} records")
    logger.info(f"Years: {df['year'].min()}-{df['year'].max()}")
    logger.info(f"Ages: {df['age'].min()}-{df['age'].max()}")
    logger.info(f"Races: {df['race'].nunique()}")
    logger.info(f"Saved to: {output_path}")


def demo_step_by_step_processing():
    """Demonstrate step-by-step processing with individual functions."""

    logger.info("=" * 70)
    logger.info("DEMO: Step-by-Step Fertility Rate Processing")
    logger.info("=" * 70)

    # Create sample data
    data_dir = project_root / "data" / "raw" / "fertility"
    sample_file = data_dir / "sample_seer_fertility.csv"

    if not sample_file.exists():
        create_sample_seer_data(sample_file)

    # Step 1: Load data
    logger.info("\n--- Step 1: Load SEER Data ---")
    raw_df = load_seer_fertility_data(
        sample_file,
        year_range=(2018, 2022)
    )
    logger.info(f"Loaded {len(raw_df)} records")
    logger.info(f"Columns: {list(raw_df.columns)}")

    # Step 2: Harmonize race categories
    logger.info("\n--- Step 2: Harmonize Race Categories ---")
    harmonized_df = harmonize_fertility_race_categories(raw_df)
    logger.info(f"Race categories after harmonization:")
    for race in sorted(harmonized_df['race_ethnicity'].unique()):
        count = (harmonized_df['race_ethnicity'] == race).sum()
        logger.info(f"  {race}: {count} records")

    # Step 3: Calculate averages
    logger.info("\n--- Step 3: Calculate 5-Year Averages ---")
    averaged_df = calculate_average_fertility_rates(
        harmonized_df,
        averaging_period=5
    )
    logger.info(f"Averaged rates: {len(averaged_df)} age-race combinations")

    # Step 4: Create complete table
    logger.info("\n--- Step 4: Create Complete Fertility Table ---")
    fertility_table = create_fertility_rate_table(
        averaged_df,
        validate=False  # We'll validate separately
    )
    logger.info(f"Complete table: {len(fertility_table)} cells")
    logger.info(f"Ages: {fertility_table['age'].min()}-{fertility_table['age'].max()}")
    logger.info(f"Races: {fertility_table['race_ethnicity'].nunique()}")

    # Step 5: Validate
    logger.info("\n--- Step 5: Validate Fertility Rates ---")
    validation = validate_fertility_rates(fertility_table)

    logger.info(f"Valid: {validation['valid']}")
    logger.info(f"Errors: {len(validation['errors'])}")
    logger.info(f"Warnings: {len(validation['warnings'])}")

    if validation['errors']:
        logger.error("Validation errors:")
        for error in validation['errors']:
            logger.error(f"  - {error}")

    if validation['warnings']:
        logger.warning("Validation warnings:")
        for warning in validation['warnings'][:5]:  # Show first 5
            logger.warning(f"  - {warning}")

    logger.info("\nTotal Fertility Rates (TFR) by Race:")
    for race, tfr in sorted(validation['tfr_by_race'].items()):
        logger.info(f"  {race}: {tfr:.2f}")
    logger.info(f"\nOverall TFR: {validation['overall_tfr']:.2f}")

    # Step 6: Show sample rates
    logger.info("\n--- Sample Fertility Rates ---")
    sample_ages = [20, 25, 30, 35, 40]
    sample_race = 'White alone, Non-Hispanic'

    logger.info(f"Fertility rates for {sample_race}:")
    for age in sample_ages:
        rate = fertility_table[
            (fertility_table['age'] == age) &
            (fertility_table['race_ethnicity'] == sample_race)
        ]['fertility_rate'].values[0]
        logger.info(f"  Age {age}: {rate:.4f} (births per woman)")

    logger.info("\n" + "=" * 70)


def demo_complete_pipeline():
    """Demonstrate complete pipeline with single function call."""

    logger.info("=" * 70)
    logger.info("DEMO: Complete Processing Pipeline")
    logger.info("=" * 70)

    # Create sample data if needed
    data_dir = project_root / "data" / "raw" / "fertility"
    sample_file = data_dir / "sample_seer_fertility.csv"

    if not sample_file.exists():
        create_sample_seer_data(sample_file)

    # Process with single function call
    logger.info("\nRunning complete processing pipeline...")

    fertility_rates = process_fertility_rates(
        input_path=sample_file,
        year_range=(2018, 2022),
        averaging_period=5
    )

    logger.info("\n" + "=" * 70)
    logger.info("Processing Complete!")
    logger.info("=" * 70)

    # Show output location
    output_dir = project_root / "data" / "processed" / "fertility"
    logger.info(f"\nOutput files created in: {output_dir}")
    logger.info("  - fertility_rates.parquet (primary data)")
    logger.info("  - fertility_rates.csv (human-readable)")
    logger.info("  - fertility_rates_metadata.json (provenance)")

    # Verify output can be loaded
    logger.info("\nVerifying output files...")
    parquet_file = output_dir / "fertility_rates.parquet"
    if parquet_file.exists():
        loaded_df = pd.read_parquet(parquet_file)
        logger.info(f"✓ Successfully loaded {len(loaded_df)} records from parquet")
    else:
        logger.error("✗ Parquet file not found")

    csv_file = output_dir / "fertility_rates.csv"
    if csv_file.exists():
        loaded_csv = pd.read_csv(csv_file)
        logger.info(f"✓ Successfully loaded {len(loaded_csv)} records from CSV")
    else:
        logger.error("✗ CSV file not found")

    return fertility_rates


def demo_integration_with_projection_engine():
    """Show how processed fertility rates integrate with projection engine."""

    logger.info("=" * 70)
    logger.info("DEMO: Integration with Projection Engine")
    logger.info("=" * 70)

    # Load processed fertility rates
    output_dir = project_root / "data" / "processed" / "fertility"
    parquet_file = output_dir / "fertility_rates.parquet"

    if not parquet_file.exists():
        logger.warning("Processed fertility rates not found. Running pipeline first...")
        demo_complete_pipeline()

    logger.info(f"\nLoading processed fertility rates from: {parquet_file}")
    fertility_rates = pd.read_parquet(parquet_file)

    logger.info(f"Loaded {len(fertility_rates)} records")
    logger.info("\nData structure:")
    logger.info(f"Columns: {list(fertility_rates.columns)}")
    logger.info(f"\nFirst few rows:")
    print(fertility_rates.head(10))

    logger.info("\n--- Ready for Projection Engine ---")
    logger.info("This DataFrame can be passed directly to:")
    logger.info("  from cohort_projections.core import CohortComponentProjection")
    logger.info("  projection = CohortComponentProjection(")
    logger.info("      base_population=pop_df,")
    logger.info("      fertility_rates=fertility_rates,  # <-- Use this DataFrame")
    logger.info("      survival_rates=surv_df,")
    logger.info("      migration_rates=mig_df")
    logger.info("  )")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    """Run all demos."""

    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 12 + "FERTILITY RATE PROCESSING EXAMPLES" + " " * 22 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("\n")

    # Run demos
    try:
        # Demo 1: Step-by-step processing
        demo_step_by_step_processing()

        print("\n" + "─" * 70 + "\n")

        # Demo 2: Complete pipeline
        demo_complete_pipeline()

        print("\n" + "─" * 70 + "\n")

        # Demo 3: Integration example
        demo_integration_with_projection_engine()

        logger.info("\n✓ All examples completed successfully!")

    except Exception as e:
        logger.error(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
