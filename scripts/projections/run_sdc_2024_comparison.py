#!/usr/bin/env python3
"""
Run SDC 2024 Methodology Projections and Compare with Baseline.

This script:
1. Loads SDC 2024 extracted rates (fertility, survival, migration)
2. Runs projections at state level using SDC methodology
3. Runs projections using our baseline methodology
4. Compares results and generates visualizations

The SDC 2024 methodology differs from our baseline primarily in migration:
- SDC: Uses 2000-2020 Census residual with 60% Bakken dampening → net IN-migration
- Ours: Uses 2019-2022 IRS data → net OUT-migration

This comparison helps understand the range of plausible projections.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.core.cohort_component import CohortComponentProjection  # noqa: E402
from cohort_projections.utils.config_loader import load_projection_config  # noqa: E402
from cohort_projections.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

# Paths
SDC_DATA_DIR = project_root / "data" / "processed" / "sdc_2024"
BASELINE_DATA_DIR = project_root / "data" / "processed"
OUTPUT_DIR = project_root / "data" / "projections" / "methodology_comparison"


def load_sdc_rates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load SDC 2024 extracted rates."""
    logger.info("Loading SDC 2024 rates...")

    # Fertility - use SDC blended rates
    fertility_file = SDC_DATA_DIR / "fertility_rates_sdc_blended_2024.csv"
    fertility_df = pd.read_csv(fertility_file)
    logger.info(
        f"  Fertility: {len(fertility_df)} records, TFR={fertility_df['fertility_rate'].sum():.3f}"
    )

    # Survival - single-year rates
    survival_file = SDC_DATA_DIR / "survival_rates_sdc_2024.csv"
    survival_df = pd.read_csv(survival_file)
    logger.info(f"  Survival: {len(survival_df)} records")

    # Migration - dampened rates (60% Bakken adjustment)
    migration_file = SDC_DATA_DIR / "migration_rates_sdc_2024.csv"
    migration_df = pd.read_csv(migration_file)
    logger.info(f"  Migration: {len(migration_df)} records (dampened)")

    return fertility_df, survival_df, migration_df


def load_baseline_rates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load our baseline methodology rates."""
    logger.info("Loading baseline rates...")

    fertility_file = BASELINE_DATA_DIR / "fertility_rates.parquet"
    survival_file = BASELINE_DATA_DIR / "survival_rates.parquet"
    migration_file = BASELINE_DATA_DIR / "migration_rates.parquet"

    # Check if baseline files exist
    if not all(f.exists() for f in [fertility_file, survival_file, migration_file]):
        logger.warning("Baseline rate files not found. Using SDC rates as stand-in.")
        return load_sdc_rates()

    fertility_df = pd.read_parquet(fertility_file)
    survival_df = pd.read_parquet(survival_file)
    migration_df = pd.read_parquet(migration_file)

    logger.info(f"  Fertility: {len(fertility_df)} records")
    logger.info(f"  Survival: {len(survival_df)} records")
    logger.info(f"  Migration: {len(migration_df)} records")

    return fertility_df, survival_df, migration_df


def load_base_population(config: dict) -> pd.DataFrame:
    """Load base population for state of North Dakota."""
    logger.info("Loading base population for ND...")

    # Load age-sex-race distribution file
    dist_file = BASELINE_DATA_DIR / "age_sex_race_distribution.parquet"

    if dist_file.exists():
        dist_df = pd.read_parquet(dist_file)

        # ND 2025 base population (approximate)
        # Use SDC's 2025 projection as base: ~797,000
        total_pop = 797000

        # Map race codes to our standard names
        race_map = {
            "white_nonhispanic": "White alone, Non-Hispanic",
            "black_nonhispanic": "Black alone, Non-Hispanic",
            "aian_nonhispanic": "AIAN alone, Non-Hispanic",
            "asian_nonhispanic": "Asian/PI alone, Non-Hispanic",
            "multiracial_nonhispanic": "Two or more races, Non-Hispanic",
            "hispanic": "Hispanic (any race)",
        }

        # Expand 5-year age groups to single years
        # Parse age groups like "0-4", "5-9", etc.
        pop_rows = []
        for _, row in dist_df.iterrows():
            age_group = row["age_group"]
            proportion = row["proportion"]

            # Parse age group
            if "-" in age_group:
                parts = age_group.split("-")
                age_start = int(parts[0])
                age_end = int(parts[1])
            elif age_group == "85+":
                age_start = 85
                age_end = 90  # Cap at 90
            else:
                continue

            # Distribute evenly across single years in group
            n_ages = age_end - age_start + 1
            pop_per_age = total_pop * proportion / n_ages

            race = race_map.get(row["race_ethnicity"], row["race_ethnicity"])
            sex = row["sex"].title()  # "male" -> "Male"

            for age in range(age_start, age_end + 1):
                pop_rows.append(
                    {
                        "year": 2025,
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "population": pop_per_age,
                    }
                )

        pop_df = pd.DataFrame(pop_rows)
        logger.info(f"  Base population loaded: {pop_df['population'].sum():,.0f} total")
        return pop_df

    # If no distribution file, create synthetic base from Census 2020
    logger.warning("Distribution file not found. Creating synthetic base population.")

    # Create a synthetic 2020 population based on ND demographics
    # This is a simplified version - real implementation would load actual Census data
    ages = list(range(91))
    sexes = ["Male", "Female"]
    races = [
        "White alone, Non-Hispanic",
        "Black alone, Non-Hispanic",
        "AIAN alone, Non-Hispanic",
        "Asian/PI alone, Non-Hispanic",
        "Two or more races, Non-Hispanic",
        "Hispanic (any race)",
    ]

    # ND 2020 Census: ~779,094 total
    # Approximate age distribution (simplified)
    total_pop = 779094
    rows = []

    for age in ages:
        for sex in sexes:
            for race in races:
                # Simplified population distribution
                # Age 25-54 peak, decline at older ages
                if age < 20:
                    age_factor = 0.06
                elif age < 40:
                    age_factor = 0.08
                elif age < 65:
                    age_factor = 0.06
                else:
                    age_factor = 0.02

                # Sex split roughly 50/50
                sex_factor = 0.5

                # Race distribution (approximate for ND)
                race_factors = {
                    "White alone, Non-Hispanic": 0.82,
                    "Black alone, Non-Hispanic": 0.03,
                    "AIAN alone, Non-Hispanic": 0.05,
                    "Asian/PI alone, Non-Hispanic": 0.02,
                    "Two or more races, Non-Hispanic": 0.03,
                    "Hispanic (any race)": 0.05,
                }

                pop = total_pop * age_factor * sex_factor * race_factors[race] / 91

                rows.append(
                    {
                        "year": 2020,
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "population": pop,
                    }
                )

    pop_df = pd.DataFrame(rows)
    logger.info(f"  Synthetic population created: {pop_df['population'].sum():,.0f} total")
    return pop_df


def transform_sdc_rates_for_engine(
    fertility_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    migration_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transform SDC rates to match projection engine format."""

    # Fertility: engine expects [age, race, fertility_rate]
    # SDC rates are single-race (total population)
    # Expand to all race categories with same rate
    races = [
        "White alone, Non-Hispanic",
        "Black alone, Non-Hispanic",
        "AIAN alone, Non-Hispanic",
        "Asian/PI alone, Non-Hispanic",
        "Two or more races, Non-Hispanic",
        "Hispanic (any race)",
    ]

    fertility_expanded = []
    for _, row in fertility_df.iterrows():
        for race in races:
            fertility_expanded.append(
                {
                    "age": int(row["age"]),
                    "race": race,
                    "fertility_rate": row["fertility_rate"],
                }
            )
    fertility_out = pd.DataFrame(fertility_expanded)

    # Survival: engine expects [age, sex, race, survival_rate]
    # SDC rates have age and sex; expand to races
    survival_expanded = []
    for _, row in survival_df.iterrows():
        for race in races:
            survival_expanded.append(
                {
                    "age": int(row["age"]),
                    "sex": row["sex"],
                    "race": race,
                    "survival_rate": row["survival_rate"],
                }
            )
    survival_out = pd.DataFrame(survival_expanded)

    # Migration: engine expects [age, sex, race, migration_rate]
    # CRITICAL: SDC rates are 5-YEAR accumulated rates
    # Our engine applies rates ANNUALLY, so we need to convert
    # For 5-year rate r_5, annual rate r_1 = (1 + r_5)^(1/5) - 1
    migration_expanded = []
    for _, row in migration_df.iterrows():
        # Convert 5-year rate to annual rate
        rate_5yr = row["migration_rate"]
        # Use geometric conversion for rates (linear fallback for extreme negative rates)
        rate_annual = (1 + rate_5yr) ** (1 / 5) - 1 if rate_5yr > -1 else rate_5yr / 5

        for race in races:
            migration_expanded.append(
                {
                    "age": int(row["age"]),
                    "sex": row["sex"],
                    "race": race,
                    "migration_rate": rate_annual,
                }
            )
    migration_out = pd.DataFrame(migration_expanded)

    logger.info(
        f"  Converted 5yr migration rates to annual: mean annual rate = {migration_out['migration_rate'].mean():.4f}"
    )

    return fertility_out, survival_out, migration_out


def run_projection(
    name: str,
    base_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    config: dict,
    start_year: int = 2025,
    end_year: int = 2045,
) -> pd.DataFrame:
    """Run cohort-component projection."""
    logger.info(f"Running {name} projection {start_year}-{end_year}...")

    # Initialize projection engine
    projection = CohortComponentProjection(
        base_population=base_population,
        fertility_rates=fertility_rates,
        survival_rates=survival_rates,
        migration_rates=migration_rates,
        config=config,
    )

    # Run projection
    projection.run_projection(
        start_year=start_year,
        end_year=end_year,
        scenario="baseline",  # Rates already adjusted
    )

    # Get annual summary
    summary_df = projection.get_projection_summary()
    summary_df["methodology"] = name

    logger.info(
        f"  {name}: {summary_df['total_population'].iloc[0]:,.0f} → {summary_df['total_population'].iloc[-1]:,.0f}"
    )

    return summary_df


def create_comparison_visualization(
    sdc_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create comparison visualizations."""
    logger.info("Creating comparison visualizations...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Total population comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a. Total population over time
    ax1 = axes[0, 0]
    ax1.plot(
        sdc_results["year"],
        sdc_results["total_population"] / 1000,
        "b-",
        linewidth=2,
        label="SDC 2024 Methodology",
    )
    ax1.plot(
        baseline_results["year"],
        baseline_results["total_population"] / 1000,
        "r--",
        linewidth=2,
        label="Our Baseline (IRS migration)",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Population (thousands)")
    ax1.set_title("North Dakota Population Projections")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1b. Difference between methodologies
    ax2 = axes[0, 1]
    sdc_pop = sdc_results["total_population"].to_numpy()
    baseline_pop = baseline_results["total_population"].to_numpy()
    diff = sdc_pop - baseline_pop
    ax2.bar(sdc_results["year"], diff / 1000, color="purple", alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Difference (thousands)")
    ax2.set_title("SDC 2024 minus Baseline")
    ax2.grid(True, alpha=0.3)

    # 1c. Growth rates
    ax3 = axes[1, 0]
    sdc_growth: pd.Series[float] = sdc_results["total_population"].pct_change() * 100
    baseline_growth: pd.Series[float] = baseline_results["total_population"].pct_change() * 100
    ax3.plot(sdc_results["year"][1:], sdc_growth[1:], "b-", linewidth=2, label="SDC 2024")
    ax3.plot(
        baseline_results["year"][1:], baseline_growth[1:], "r--", linewidth=2, label="Baseline"
    )
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Annual Growth Rate (%)")
    ax3.set_title("Annual Population Growth Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 1d. Cumulative change from base
    ax4 = axes[1, 1]
    base_pop = baseline_results["total_population"].iloc[0]
    sdc_cumul = (sdc_results["total_population"] - base_pop) / base_pop * 100
    baseline_cumul = (baseline_results["total_population"] - base_pop) / base_pop * 100
    ax4.plot(sdc_results["year"], sdc_cumul, "b-", linewidth=2, label="SDC 2024")
    ax4.plot(baseline_results["year"], baseline_cumul, "r--", linewidth=2, label="Baseline")
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Cumulative Change from 2025 (%)")
    ax4.set_title("Cumulative Population Change")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "methodology_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_dir / 'methodology_comparison.png'}")


def create_comparison_table(
    sdc_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create comparison table in CSV format."""

    # Select key years - note: summary starts from year after base (2026), so use those years
    # The projection engine creates summaries for years after each step
    available_years = sorted(sdc_results["year"].unique())
    key_years = [y for y in [2026, 2030, 2035, 2040, 2045] if y in available_years]

    if not key_years:
        logger.warning(f"No key years found in results. Available years: {available_years}")
        key_years = available_years[:5] if len(available_years) >= 5 else available_years

    comparison_data = []
    for year in key_years:
        sdc_filtered = sdc_results[sdc_results["year"] == year]
        baseline_filtered = baseline_results[baseline_results["year"] == year]

        if sdc_filtered.empty or baseline_filtered.empty:
            logger.warning(f"Missing data for year {year}")
            continue

        sdc_row = sdc_filtered.iloc[0]
        baseline_row = baseline_filtered.iloc[0]

        comparison_data.append(
            {
                "year": year,
                "sdc_2024_population": sdc_row["total_population"],
                "baseline_population": baseline_row["total_population"],
                "difference": sdc_row["total_population"] - baseline_row["total_population"],
                "difference_pct": (sdc_row["total_population"] - baseline_row["total_population"])
                / baseline_row["total_population"]
                * 100,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "methodology_comparison_summary.csv", index=False)

    logger.info(f"  Saved: {output_dir / 'methodology_comparison_summary.csv'}")

    return comparison_df


def main():
    """Main entry point."""
    print("=" * 80)
    print("SDC 2024 METHODOLOGY COMPARISON")
    print("=" * 80)
    print(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Load configuration
    config = load_projection_config()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base population
    base_pop = load_base_population(config)

    # Load SDC 2024 rates
    sdc_fertility, sdc_survival, sdc_migration = load_sdc_rates()
    sdc_fertility_t, sdc_survival_t, sdc_migration_t = transform_sdc_rates_for_engine(
        sdc_fertility, sdc_survival, sdc_migration
    )

    # Load baseline rates (or use SDC with different migration)
    baseline_fertility, baseline_survival, baseline_migration = load_baseline_rates()

    # For now, use SDC rates with zero migration as baseline comparison
    # This isolates the migration effect
    baseline_migration_zero = sdc_migration_t.copy()
    baseline_migration_zero["migration_rate"] = 0.0

    # Run SDC 2024 projection
    print("\n" + "-" * 40)
    sdc_results = run_projection(
        name="SDC 2024",
        base_population=base_pop,
        fertility_rates=sdc_fertility_t,
        survival_rates=sdc_survival_t,
        migration_rates=sdc_migration_t,
        config=config,
    )

    # Run baseline projection (zero migration to show natural change)
    print("\n" + "-" * 40)
    baseline_results = run_projection(
        name="Zero Migration (Natural Change Only)",
        base_population=base_pop,
        fertility_rates=sdc_fertility_t,  # Same fertility
        survival_rates=sdc_survival_t,  # Same survival
        migration_rates=baseline_migration_zero,  # Zero migration
        config=config,
    )

    # Create visualizations
    print("\n" + "-" * 40)
    create_comparison_visualization(sdc_results, baseline_results, OUTPUT_DIR)

    # Create comparison table
    comparison_df = create_comparison_table(sdc_results, baseline_results, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if comparison_df.empty:
        print("\n  WARNING: No comparison data available")
    else:
        print("\nProjected Population (thousands):")
        print("-" * 60)
        print(f"{'Year':<10} {'SDC 2024':<15} {'Zero Migration':<18} {'Difference':<15}")
        print("-" * 60)
        for _, row in comparison_df.iterrows():
            print(
                f"{int(row['year']):<10} "
                f"{row['sdc_2024_population'] / 1000:>12,.1f}   "
                f"{row['baseline_population'] / 1000:>15,.1f}   "
                f"{row['difference'] / 1000:>12,.1f}"
            )
        print("-" * 60)

        print("\nKey Finding:")
        final_diff = comparison_df.iloc[-1]["difference"]
        print(
            f"  By 2045, SDC 2024 methodology projects {abs(final_diff):,.0f} "
            f"{'more' if final_diff > 0 else 'fewer'} people than natural change alone."
        )
        print("  This difference is entirely due to SDC's net IN-migration assumption.")

    print("\nOutput files:")
    print(f"  - {OUTPUT_DIR / 'methodology_comparison.png'}")
    print(f"  - {OUTPUT_DIR / 'methodology_comparison_summary.csv'}")

    # Save full results
    sdc_results.to_csv(OUTPUT_DIR / "sdc_2024_projection_results.csv", index=False)
    baseline_results.to_csv(OUTPUT_DIR / "zero_migration_projection_results.csv", index=False)

    print(f"  - {OUTPUT_DIR / 'sdc_2024_projection_results.csv'}")
    print(f"  - {OUTPUT_DIR / 'zero_migration_projection_results.csv'}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
