"""
Cohort Component Projection Engine.

The main projection engine implementing the standard demographic cohort-component
method for population projections.

The method projects population forward by:
1. Aging: Advance each cohort by 1 year (age t -> age t+1)
2. Survival: Apply survival rates to account for mortality
3. Fertility: Apply age-specific fertility rates to females -> births
4. Migration: Add net migration by cohort
5. Sum: Total population at t+1
"""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger_from_config
from .fertility import apply_fertility_scenario, calculate_births, validate_fertility_rates
from .migration import apply_migration, apply_migration_scenario, validate_migration_data
from .mortality import apply_survival_rates, validate_survival_rates

logger = get_logger_from_config(__name__)


class CohortComponentProjection:
    """
    Cohort Component Population Projection Engine.

    Implements the standard demographic cohort-component method for
    population projections by age, sex, and race/ethnicity.
    """

    def __init__(
        self,
        base_population: pd.DataFrame,
        fertility_rates: pd.DataFrame,
        survival_rates: pd.DataFrame,
        migration_rates: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize cohort component projection.

        Args:
            base_population: Starting population DataFrame
                            Columns: [year, age, sex, race, population]
            fertility_rates: Age-specific fertility rates
                           Columns: [age, race, fertility_rate]
            survival_rates: Age/sex/race-specific survival rates
                          Columns: [age, sex, race, survival_rate]
            migration_rates: Net migration by cohort
                           Columns: [age, sex, race, net_migration] or migration_rate
            config: Configuration dictionary (optional)

        Raises:
            ValueError: If input data validation fails
        """
        # Load configuration
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.get_projection_config()

        self.config = config
        self.base_year = config.get("project", {}).get("base_year", 2025)
        self.projection_horizon = config.get("project", {}).get("projection_horizon", 20)
        self.max_age = config.get("demographics", {}).get("age_groups", {}).get("max_age", 90)

        logger.info(
            f"Initializing Cohort Component Projection: "
            f"{self.base_year} to {self.base_year + self.projection_horizon}"
        )

        # Store input data
        self.base_population = base_population.copy()
        self.fertility_rates = fertility_rates.copy()
        self.survival_rates = survival_rates.copy()
        self.migration_rates = migration_rates.copy()

        # Validate inputs
        self._validate_inputs()

        # Initialize results storage
        self.projection_results: pd.DataFrame = pd.DataFrame()
        self.annual_summaries: list[dict[str, Any]] = []

        logger.info("Cohort Component Projection initialized successfully")

    def _validate_inputs(self):
        """Validate all input data."""
        logger.info("Validating input data...")

        # Validate base population
        required_pop_cols = ["year", "age", "sex", "race", "population"]
        missing_cols = [col for col in required_pop_cols if col not in self.base_population.columns]
        if missing_cols:
            raise ValueError(f"base_population missing columns: {missing_cols}")

        if self.base_population.empty:
            raise ValueError("base_population is empty")

        if (self.base_population["population"] < 0).any():
            raise ValueError("base_population contains negative values")

        total_pop = self.base_population["population"].sum()
        logger.info(f"Base population ({self.base_year}): {total_pop:,.0f}")

        # Validate fertility rates
        is_valid, issues = validate_fertility_rates(self.fertility_rates, self.config)
        if not is_valid:
            logger.warning(f"Fertility validation issues: {issues}")

        # Validate survival rates
        is_valid, issues = validate_survival_rates(self.survival_rates, self.config)
        if not is_valid:
            logger.warning(f"Survival validation issues: {issues}")

        # Validate migration data
        is_valid, issues = validate_migration_data(
            self.migration_rates, self.base_population, self.config
        )
        if not is_valid:
            logger.warning(f"Migration validation issues: {issues}")

        logger.info("Input validation complete")

    def project_single_year(
        self, population: pd.DataFrame, year: int, scenario: str | None = None
    ) -> pd.DataFrame:
        """
        Project population one year forward.

        Implements the cohort-component method:
        1. Apply survival rates (aging + mortality)
        2. Calculate births from fertility
        3. Apply net migration
        4. Combine all components

        Args:
            population: Population at start of year
                       Columns: [year, age, sex, race, population]
            year: Current year (will project to year+1)
            scenario: Optional scenario name for rate adjustments

        Returns:
            Population at year+1
        """
        logger.info(f"Projecting year {year} -> {year + 1}")

        # Prepare scenario-adjusted rates if needed
        survival_rates = self.survival_rates.copy()
        fertility_rates = self.fertility_rates.copy()
        migration_rates = self.migration_rates.copy()

        if scenario:
            logger.debug(f"Applying scenario: {scenario}")
            scenario_config = self.config.get("scenarios", {}).get(scenario, {})

            if scenario_config:
                # Apply fertility scenario
                fertility_scenario = scenario_config.get("fertility", "constant")
                fertility_rates = apply_fertility_scenario(
                    fertility_rates, fertility_scenario, year, self.base_year
                )

                # Apply migration scenario
                migration_scenario = scenario_config.get("migration", "recent_average")
                migration_rates = apply_migration_scenario(
                    migration_rates, migration_scenario, year, self.base_year
                )

        # Step 1: Apply survival (aging + mortality)
        logger.debug(f"Year {year}: Applying survival rates")
        survived_population = apply_survival_rates(population, survival_rates, year, self.config)

        # Step 2: Calculate births
        logger.debug(f"Year {year}: Calculating births")

        # Extract female population for fertility calculation
        female_pop = population[population["sex"] == "Female"].copy()

        births = calculate_births(female_pop, fertility_rates, year, self.config)

        # Step 3: Apply migration to survived population
        logger.debug(f"Year {year}: Applying migration")
        population_with_migration = apply_migration(
            survived_population, migration_rates, year, self.config
        )

        # Step 4: Apply migration to births (if any born during year)
        # Typically births don't experience migration in birth year, so skip

        # Step 5: Combine survived+migrated population with births
        if not births.empty:
            # Ensure births have correct year
            births["year"] = year + 1

            # Combine
            combined_population = pd.concat([population_with_migration, births], ignore_index=True)
        else:
            combined_population = population_with_migration

        # Step 6: Aggregate any duplicate cohorts (shouldn't happen, but defensive)
        combined_population = combined_population.groupby(
            ["year", "age", "sex", "race"], as_index=False
        ).agg({"population": "sum"})

        # Step 7: Validation checks
        total_pop = combined_population["population"].sum()
        if total_pop <= 0:
            logger.error(f"Year {year + 1}: Total population is {total_pop}")
            raise ValueError(f"Invalid population state at year {year + 1}")

        negative_cohorts = combined_population["population"] < 0
        if negative_cohorts.any():
            num_negative = negative_cohorts.sum()
            logger.warning(
                f"Year {year + 1}: {num_negative} cohorts with negative population, "
                f"setting to 0"
            )
            combined_population.loc[negative_cohorts, "population"] = 0.0

        logger.info(f"Year {year + 1}: Total population = {total_pop:,.0f}")

        return combined_population

    def run_projection(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        scenario: str | None = None,
    ) -> pd.DataFrame:
        """
        Run full multi-year projection.

        Args:
            start_year: Starting year (default: base_year)
            end_year: Ending year (default: base_year + projection_horizon)
            scenario: Optional scenario name ('baseline', 'high_growth', etc.)

        Returns:
            Time series DataFrame with all projection years
            Columns: [year, age, sex, race, population]
        """
        if start_year is None:
            start_year = self.base_year

        if end_year is None:
            end_year = self.base_year + self.projection_horizon

        if scenario is None:
            scenario = "baseline"

        logger.info(
            f"Starting projection run: {start_year} to {end_year} "
            f"({end_year - start_year} years, scenario: {scenario})"
        )

        # Initialize with base population
        current_population = self.base_population.copy()

        # Store base year
        all_years = [current_population]

        # Project each year
        for year in range(start_year, end_year):
            try:
                # Project one year forward
                current_population = self.project_single_year(current_population, year, scenario)

                # Store result
                all_years.append(current_population.copy())

                # Create annual summary
                summary = self._create_annual_summary(current_population, year + 1)
                self.annual_summaries.append(summary)

            except Exception as e:
                logger.error(f"Projection failed at year {year}: {str(e)}")
                raise

        # Combine all years
        projection_results = pd.concat(all_years, ignore_index=True)

        # Sort by year, age, sex, race
        projection_results = projection_results.sort_values(
            ["year", "age", "sex", "race"]
        ).reset_index(drop=True)

        self.projection_results = projection_results

        logger.info(
            f"Projection complete: {len(all_years)} years, "
            f"{len(projection_results):,} total records"
        )

        return projection_results

    def _create_annual_summary(self, population: pd.DataFrame, year: int) -> dict[str, Any]:
        """
        Create summary statistics for a year.

        Args:
            population: Population for the year
            year: Year

        Returns:
            Dictionary with summary statistics
        """
        total_pop = population["population"].sum()

        summary = {
            "year": year,
            "total_population": total_pop,
            "male_population": population[population["sex"] == "Male"]["population"].sum(),
            "female_population": population[population["sex"] == "Female"]["population"].sum(),
        }

        # Population by race
        for race in population["race"].unique():
            race_pop = population[population["race"] == race]["population"].sum()
            summary[f"population_{race}"] = race_pop

        # Age structure
        summary["median_age"] = self._calculate_median_age(population)
        summary["dependency_ratio"] = self._calculate_dependency_ratio(population)

        # Population under 18
        summary["population_under_18"] = population[population["age"] < 18]["population"].sum()

        # Working age (18-64)
        summary["population_working_age"] = population[
            (population["age"] >= 18) & (population["age"] < 65)
        ]["population"].sum()

        # Seniors (65+)
        summary["population_65_plus"] = population[population["age"] >= 65]["population"].sum()

        return summary

    def _calculate_median_age(self, population: pd.DataFrame) -> float:
        """Calculate median age of population."""
        # Expand population to individual ages
        ages = []
        for _, row in population.iterrows():
            ages.extend([row["age"]] * int(row["population"]))

        if len(ages) == 0:
            return 0.0

        return float(np.median(ages))

    def _calculate_dependency_ratio(self, population: pd.DataFrame) -> float:
        """
        Calculate dependency ratio.

        Dependency ratio = (Pop < 18 + Pop 65+) / Pop 18-64
        """
        dependent = population[(population["age"] < 18) | (population["age"] >= 65)][
            "population"
        ].sum()

        working_age = population[(population["age"] >= 18) & (population["age"] < 65)][
            "population"
        ].sum()

        if working_age == 0:
            return 0.0

        return dependent / working_age

    def get_projection_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of projection results.

        Returns:
            DataFrame with annual summary statistics
        """
        if not self.annual_summaries:
            logger.warning("No projection summaries available")
            return pd.DataFrame()

        return pd.DataFrame(self.annual_summaries)

    def get_population_by_year(self, year: int) -> pd.DataFrame:
        """
        Get population for a specific year.

        Args:
            year: Year to extract

        Returns:
            DataFrame with population for that year
        """
        if self.projection_results.empty:
            logger.warning("No projection results available")
            return pd.DataFrame()

        year_data = self.projection_results[self.projection_results["year"] == year].copy()

        if year_data.empty:
            logger.warning(f"No data for year {year}")

        return year_data

    def get_cohort_trajectory(self, birth_year: int, sex: str, race: str) -> pd.DataFrame:
        """
        Track a specific birth cohort over time.

        Args:
            birth_year: Year of birth for the cohort
            sex: Sex of cohort
            race: Race/ethnicity of cohort

        Returns:
            DataFrame showing cohort size over time
        """
        if self.projection_results.empty:
            logger.warning("No projection results available")
            return pd.DataFrame()

        # Filter to cohort
        cohort_data = self.projection_results[
            (self.projection_results["sex"] == sex) & (self.projection_results["race"] == race)
        ].copy()

        # Calculate birth year from age and year
        cohort_data["birth_year"] = cohort_data["year"] - cohort_data["age"]

        # Filter to specific birth year
        cohort_data = cohort_data[cohort_data["birth_year"] == birth_year]

        # Sort by year
        cohort_data = cohort_data.sort_values("year")

        return cohort_data[["year", "age", "population"]]

    def export_results(
        self, output_path: Path, format: str = "parquet", compression: str | None = None
    ):
        """
        Export projection results to file.

        Args:
            output_path: Path to output file
            format: Output format ('parquet', 'csv', 'excel')
            compression: Optional compression ('gzip', 'snappy', etc.)
        """
        if self.projection_results.empty:
            logger.error("No projection results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting results to {output_path}")

        if format == "parquet":
            self.projection_results.to_parquet(
                output_path,
                compression=cast(
                    Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None,
                    compression or "gzip",
                ),
                index=False,
            )

        elif format == "csv":
            self.projection_results.to_csv(
                output_path,
                index=False,
                compression=cast(
                    Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None, compression
                ),
            )

        elif format == "excel":
            self.projection_results.to_excel(output_path, index=False, engine="openpyxl")

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results exported successfully to {output_path}")

    def export_summary(self, output_path: Path, format: str = "csv"):
        """
        Export summary statistics to file.

        Args:
            output_path: Path to output file
            format: Output format ('csv', 'excel')
        """
        summary_df = self.get_projection_summary()

        if summary_df.empty:
            logger.error("No summary data to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting summary to {output_path}")

        if format == "csv":
            summary_df.to_csv(output_path, index=False)
        elif format == "excel":
            summary_df.to_excel(output_path, index=False, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Summary exported successfully to {output_path}")
