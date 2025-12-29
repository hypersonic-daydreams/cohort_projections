#!/usr/bin/env python3
"""
SDC 2024 Replication: Standalone 5-Year Cohort-Component Projection Engine

This is a STANDALONE script implementing the North Dakota State Data Center's
2024 population projection methodology. It does NOT import from the main
cohort_projections package.

Methodology Reference: SDC 2024 Population Projections Report (February 6, 2024)

Key Features:
- 5-year age groups: 0-4, 5-9, 10-14, ..., 80-84, 85+
- 5-year time steps: 2020, 2025, 2030, 2035, 2040, 2045, 2050
- County-level projections summed to state
- Male and Female projected separately
- Period-varying migration multipliers (Bakken dampening)

Author: SDC 2024 Replication Project
Date: 2025-12-28
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Age group definitions (18 five-year groups)
AGE_GROUPS: list[str] = [
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

# Number of age groups
N_AGE_GROUPS: int = len(AGE_GROUPS)

# Sex categories
SEXES: list[str] = ["Male", "Female"]

# Projection years
BASE_YEAR: int = 2020
PROJECTION_YEARS: list[int] = [2025, 2030, 2035, 2040, 2045, 2050]
ALL_YEARS: list[int] = [BASE_YEAR] + PROJECTION_YEARS

# Period multipliers for migration (SDC Bakken dampening)
# Key is the END year of the 5-year period
PERIOD_MULTIPLIERS: dict[int, float] = {
    2025: 0.2,  # 2020-2025: COVID + post-Bakken transition
    2030: 0.6,  # 2025-2030: Bakken dampening
    2035: 0.6,  # 2030-2035: Bakken dampening
    2040: 0.5,  # 2035-2040: Further reduced
    2045: 0.7,  # 2040-2045: Increasing toward normal
    2050: 0.7,  # 2045-2050: Increasing toward normal
}

# Childbearing age groups (indices into AGE_GROUPS)
# 15-19 through 45-49 (indices 3 through 9)
CHILDBEARING_AGE_INDICES: list[int] = [3, 4, 5, 6, 7, 8, 9]

# Sex ratio at birth (proportion male)
MALE_BIRTH_RATIO: float = 0.512  # Approximately 51.2% male births

# Number of years in each projection period
PERIOD_LENGTH: int = 5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ProjectionInputs:
    """Container for all projection input data.

    Attributes:
        base_population: DataFrame with columns [county, age_group, sex, population]
        survival_rates: DataFrame with columns [age_group, sex, survival_rate]
        fertility_rates: DataFrame with columns [county, age_group, fertility_rate]
            or [age_group, fertility_rate] for uniform rates
        migration_rates: DataFrame with columns [county, age_group, sex, migration_rate]
        adjustments: Optional DataFrame with columns
            [county, age_group, sex, year, adjustment]
        counties: List of county FIPS codes or names
    """

    base_population: pd.DataFrame
    survival_rates: pd.DataFrame
    fertility_rates: pd.DataFrame
    migration_rates: pd.DataFrame
    adjustments: pd.DataFrame | None = None
    counties: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Extract county list from base population if not provided."""
        if not self.counties:
            self.counties = sorted(self.base_population["county"].unique().tolist())


@dataclass
class ProjectionOutputs:
    """Container for projection results.

    Attributes:
        population: DataFrame with projections by county, age, sex, year
        births: DataFrame with births by county, year
        deaths: DataFrame with deaths by county, year
        migration: DataFrame with net migration by county, year
        state_totals: DataFrame with state-level totals by year
    """

    population: pd.DataFrame
    births: pd.DataFrame
    deaths: pd.DataFrame
    migration: pd.DataFrame
    state_totals: pd.DataFrame


# =============================================================================
# Core Projection Functions
# =============================================================================


def calculate_births(
    female_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    county: str,
) -> tuple[float, float]:
    """Calculate births for a single county and period.

    The SDC methodology calculates births as:
        Births = SUM[Female_Pop[age] * Fertility_Rate[age]] * 5 years

    Then splits by sex ratio for male/female infant cohorts.

    Args:
        female_population: DataFrame with female population by age group
            Must have columns [age_group, population]
        fertility_rates: DataFrame with fertility rates by age group
            Must have columns [age_group, fertility_rate] or
            [county, age_group, fertility_rate]
        county: County identifier for county-specific rates

    Returns:
        Tuple of (male_births, female_births) over the 5-year period
    """
    # Get fertility rates for this county (or use uniform rates)
    if "county" in fertility_rates.columns:
        county_rates = fertility_rates[fertility_rates["county"] == county]
    else:
        county_rates = fertility_rates

    total_births = 0.0

    for age_idx in CHILDBEARING_AGE_INDICES:
        age_group = AGE_GROUPS[age_idx]

        # Get female population in this age group
        pop_row = female_population[female_population["age_group"] == age_group]
        if pop_row.empty:
            continue
        female_pop = pop_row["population"].values[0]

        # Get fertility rate for this age group
        rate_row = county_rates[county_rates["age_group"] == age_group]
        if rate_row.empty:
            continue
        fertility_rate = rate_row["fertility_rate"].values[0]

        # Births = population * rate * period_length
        # Note: fertility rates are typically per 1,000 women per year
        # Adjust if rates are already per 5 years
        age_births = female_pop * fertility_rate * PERIOD_LENGTH
        total_births += age_births

    # Split by sex ratio
    male_births = total_births * MALE_BIRTH_RATIO
    female_births = total_births * (1 - MALE_BIRTH_RATIO)

    return male_births, female_births


def apply_infant_survival(
    births: float,
    survival_rates: pd.DataFrame,
    sex: str,
) -> float:
    """Apply survival rate to births to get 0-4 population.

    Args:
        births: Number of births over the 5-year period
        survival_rates: DataFrame with survival rates by age and sex
        sex: "Male" or "Female"

    Returns:
        Surviving population in the 0-4 age group at end of period
    """
    # Get survival rate for 0-4 age group
    rate_row = survival_rates[
        (survival_rates["age_group"] == "0-4") & (survival_rates["sex"] == sex)
    ]

    if rate_row.empty:
        logger.warning("No survival rate found for 0-4 %s, using 0.99", sex)
        survival_rate = 0.99
    else:
        survival_rate = rate_row["survival_rate"].values[0]

    return births * survival_rate


def age_cohort_forward(
    population: pd.DataFrame,
    survival_rates: pd.DataFrame,
    sex: str,
) -> pd.DataFrame:
    """Age a population cohort forward by 5 years using survival rates.

    SDC Methodology:
        Natural_Growth[age+5, t+5] = Population[age, t] * Survival_Rate[age, sex]

    Special handling for 85+ open-ended group:
        Natural_Growth[85+] = Pop[80-84] * Survival[80-84] + Pop[85+] * Survival[85+]

    Args:
        population: DataFrame with current population by age group
            Must have columns [age_group, population]
        survival_rates: DataFrame with survival rates
            Must have columns [age_group, sex, survival_rate]
        sex: "Male" or "Female"

    Returns:
        DataFrame with natural growth (survived population) by age group
        Note: Does NOT include 0-4 age group (calculated separately from births)
    """
    sex_survival = survival_rates[survival_rates["sex"] == sex]

    results = []

    for i in range(1, N_AGE_GROUPS):  # Start from 5-9, skip 0-4
        target_age = AGE_GROUPS[i]

        if i < N_AGE_GROUPS - 1:
            # Normal aging: people in age group i-1 survive to age group i
            source_age = AGE_GROUPS[i - 1]

            pop_row = population[population["age_group"] == source_age]
            if pop_row.empty:
                nat_grow = 0.0
            else:
                source_pop = pop_row["population"].values[0]

                rate_row = sex_survival[sex_survival["age_group"] == source_age]
                if rate_row.empty:
                    logger.warning("No survival rate for %s %s, using 0.99", source_age, sex)
                    survival = 0.99
                else:
                    survival = rate_row["survival_rate"].values[0]

                nat_grow = source_pop * survival

        else:
            # Special case: 85+ open-ended group
            # Survivors from 80-84 plus survivors from 85+
            nat_grow = 0.0

            # Survivors from 80-84 -> 85+
            pop_80_84 = population[population["age_group"] == "80-84"]
            if not pop_80_84.empty:
                rate_80_84 = sex_survival[sex_survival["age_group"] == "80-84"]
                if not rate_80_84.empty:
                    nat_grow += (
                        pop_80_84["population"].values[0] * rate_80_84["survival_rate"].values[0]
                    )

            # Survivors within 85+ (those who don't die)
            pop_85plus = population[population["age_group"] == "85+"]
            if not pop_85plus.empty:
                rate_85plus = sex_survival[sex_survival["age_group"] == "85+"]
                if not rate_85plus.empty:
                    nat_grow += (
                        pop_85plus["population"].values[0] * rate_85plus["survival_rate"].values[0]
                    )

        results.append({"age_group": target_age, "natural_growth": nat_grow})

    return pd.DataFrame(results)


def calculate_migration(
    natural_growth: pd.DataFrame,
    migration_rates: pd.DataFrame,
    county: str,
    sex: str,
    period_end_year: int,
) -> pd.DataFrame:
    """Calculate migration for a county/sex in a projection period.

    SDC Methodology:
        Migration[age] = Natural_Growth[age] * Migration_Rate[age,sex,county]
                        * Period_Multiplier[period]

    Args:
        natural_growth: DataFrame with natural growth by age group
            Must have columns [age_group, natural_growth]
        migration_rates: DataFrame with migration rates
            Must have columns [county, age_group, sex, migration_rate]
        county: County identifier
        sex: "Male" or "Female"
        period_end_year: End year of projection period (e.g., 2025, 2030, etc.)

    Returns:
        DataFrame with migration by age group
    """
    period_multiplier = PERIOD_MULTIPLIERS.get(period_end_year, 0.6)

    county_sex_rates = migration_rates[
        (migration_rates["county"] == county) & (migration_rates["sex"] == sex)
    ]

    results = []

    for _, row in natural_growth.iterrows():
        age_group = row["age_group"]
        nat_grow = row["natural_growth"]

        rate_row = county_sex_rates[county_sex_rates["age_group"] == age_group]
        mig_rate = 0.0 if rate_row.empty else rate_row["migration_rate"].values[0]

        migration = nat_grow * mig_rate * period_multiplier

        results.append({"age_group": age_group, "migration": migration})

    return pd.DataFrame(results)


def get_adjustments(
    adjustments: pd.DataFrame | None,
    county: str,
    sex: str,
    year: int,
) -> pd.DataFrame:
    """Get manual adjustments for a county/sex/year combination.

    SDC applies manual adjustments for:
    - College-age population corrections
    - Bakken region sex-ratio balancing
    - Regional economic factors

    Args:
        adjustments: DataFrame with adjustments or None
            Must have columns [county, age_group, sex, year, adjustment]
        county: County identifier
        sex: "Male" or "Female"
        year: Projection year

    Returns:
        DataFrame with adjustments by age group (zeros if no adjustments provided)
    """
    if adjustments is None or adjustments.empty:
        return pd.DataFrame({"age_group": AGE_GROUPS, "adjustment": [0.0] * N_AGE_GROUPS})

    filtered = adjustments[
        (adjustments["county"] == county)
        & (adjustments["sex"] == sex)
        & (adjustments["year"] == year)
    ]

    if filtered.empty:
        return pd.DataFrame({"age_group": AGE_GROUPS, "adjustment": [0.0] * N_AGE_GROUPS})

    return filtered[["age_group", "adjustment"]].copy()


def project_county_sex(
    population: pd.DataFrame,
    survival_rates: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    adjustments: pd.DataFrame | None,
    county: str,
    sex: str,
    from_year: int,
    to_year: int,
) -> tuple[pd.DataFrame, dict]:
    """Project population for a single county/sex from one period to the next.

    This is the core projection function implementing the SDC methodology:
        Population[age, t+5] = Natural_Growth[age] + Migration[age] + Adjustments[age]

    Args:
        population: Current population by age group
        survival_rates: Survival rates by age group and sex
        fertility_rates: Fertility rates by age group (and optionally county)
        migration_rates: Migration rates by county, age group, and sex
        adjustments: Optional manual adjustments
        county: County identifier
        sex: "Male" or "Female"
        from_year: Starting year
        to_year: Ending year

    Returns:
        Tuple of (projected_population DataFrame, components dict)
        Components dict contains: births, deaths, migration, adjustments totals
    """
    components = {"births": 0.0, "deaths": 0.0, "migration": 0.0, "adjustments": 0.0}

    # Calculate natural growth (aging with survival)
    natural_growth = age_cohort_forward(population, survival_rates, sex)

    # Calculate deaths (difference between original and survived)
    original_pop = population[population["age_group"] != "0-4"]["population"].sum()
    survived_pop = natural_growth["natural_growth"].sum()
    components["deaths"] = original_pop - survived_pop

    # Calculate migration
    migration = calculate_migration(natural_growth, migration_rates, county, sex, to_year)

    components["migration"] = migration["migration"].sum()

    # Get adjustments
    adj = get_adjustments(adjustments, county, sex, to_year)

    components["adjustments"] = adj["adjustment"].sum()

    # Build final population
    results = []

    # Handle 0-4 age group (from births)
    if sex == "Female":
        # Calculate births (only for females, but we need male births too)
        female_pop = population[["age_group", "population"]]
        male_births, female_births = calculate_births(female_pop, fertility_rates, county)

        # Births split in components (only counted once for females)
        components["births"] = male_births + female_births

        # Survive female births to 0-4
        pop_0_4 = apply_infant_survival(female_births, survival_rates, sex)
    else:
        # For males, we need to get births from female calculation
        # This is a simplification - in practice, births are calculated once
        # and split by sex. We'll handle this in the county-level function.
        pop_0_4 = 0.0  # Will be set by caller

    results.append({"age_group": "0-4", "population": pop_0_4})

    # Combine natural growth, migration, and adjustments for ages 5+
    for _, ng_row in natural_growth.iterrows():
        age_group = ng_row["age_group"]
        nat_grow = ng_row["natural_growth"]

        mig_row = migration[migration["age_group"] == age_group]
        mig = mig_row["migration"].values[0] if not mig_row.empty else 0.0

        adj_row = adj[adj["age_group"] == age_group]
        adjustment = adj_row["adjustment"].values[0] if not adj_row.empty else 0.0

        final_pop = max(0.0, nat_grow + mig + adjustment)  # Ensure non-negative

        results.append({"age_group": age_group, "population": final_pop})

    projected = pd.DataFrame(results)
    projected["county"] = county
    projected["sex"] = sex
    projected["year"] = to_year

    return projected, components


def project_county(
    population: pd.DataFrame,
    inputs: ProjectionInputs,
    county: str,
    from_year: int,
    to_year: int,
) -> tuple[pd.DataFrame, dict]:
    """Project population for a single county (both sexes).

    Args:
        population: Current population for this county (all ages, both sexes)
        inputs: ProjectionInputs with all rate data
        county: County identifier
        from_year: Starting year
        to_year: Ending year

    Returns:
        Tuple of (projected_population DataFrame, components dict)
    """
    # Get female population for birth calculation
    female_pop = population[population["sex"] == "Female"][["age_group", "population"]]

    # Calculate births once (then split by sex)
    male_births, female_births = calculate_births(female_pop, inputs.fertility_rates, county)

    all_results = []
    total_components = {"births": 0.0, "deaths": 0.0, "migration": 0.0, "adjustments": 0.0}

    for sex in SEXES:
        sex_pop = population[population["sex"] == sex][["age_group", "population"]]

        projected, components = project_county_sex(
            population=sex_pop,
            survival_rates=inputs.survival_rates,
            fertility_rates=inputs.fertility_rates,
            migration_rates=inputs.migration_rates,
            adjustments=inputs.adjustments,
            county=county,
            sex=sex,
            from_year=from_year,
            to_year=to_year,
        )

        # Set 0-4 population from pre-calculated births
        births = male_births if sex == "Male" else female_births
        survived_births = apply_infant_survival(births, inputs.survival_rates, sex)

        projected.loc[projected["age_group"] == "0-4", "population"] = survived_births

        all_results.append(projected)

        # Aggregate components (births only counted once)
        if sex == "Female":
            total_components["births"] = male_births + female_births
        total_components["deaths"] += components["deaths"]
        total_components["migration"] += components["migration"]
        total_components["adjustments"] += components["adjustments"]

    combined = pd.concat(all_results, ignore_index=True)

    return combined, total_components


# =============================================================================
# Main Projection Runner
# =============================================================================


def run_projections(inputs: ProjectionInputs) -> ProjectionOutputs:
    """Run the full SDC 2024 projection methodology.

    This function orchestrates the complete projection process:
    1. For each projection period (5 years)
    2. For each county
    3. Project population forward using cohort-component method
    4. Sum counties to get state totals

    Args:
        inputs: ProjectionInputs with all required data

    Returns:
        ProjectionOutputs with full results
    """
    logger.info("=" * 60)
    logger.info("Starting SDC 2024 Cohort-Component Projection")
    logger.info("=" * 60)
    logger.info("Base year: %d", BASE_YEAR)
    logger.info("Projection years: %s", PROJECTION_YEARS)
    logger.info("Counties: %d", len(inputs.counties))
    logger.info("Age groups: %d", N_AGE_GROUPS)

    all_populations = []
    all_births = []
    all_deaths = []
    all_migrations = []

    # Initialize with base population
    base_pop = inputs.base_population.copy()
    base_pop["year"] = BASE_YEAR
    all_populations.append(base_pop)

    current_population = base_pop.copy()

    # Project each period
    for i, to_year in enumerate(PROJECTION_YEARS):
        from_year = ALL_YEARS[i]
        logger.info("")
        logger.info("-" * 40)
        logger.info("Projecting period: %d -> %d", from_year, to_year)
        logger.info("Period multiplier: %.1f", PERIOD_MULTIPLIERS.get(to_year, 0.6))

        period_populations = []
        period_births = []
        period_deaths = []
        period_migrations = []

        for county in inputs.counties:
            county_pop = current_population[current_population["county"] == county]

            projected, components = project_county(
                population=county_pop,
                inputs=inputs,
                county=county,
                from_year=from_year,
                to_year=to_year,
            )

            period_populations.append(projected)

            period_births.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "births": components["births"],
                }
            )
            period_deaths.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "deaths": components["deaths"],
                }
            )
            period_migrations.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "migration": components["migration"],
                }
            )

        # Combine period results
        period_pop = pd.concat(period_populations, ignore_index=True)
        all_populations.append(period_pop)
        all_births.extend(period_births)
        all_deaths.extend(period_deaths)
        all_migrations.extend(period_migrations)

        # Update current population for next period
        current_population = period_pop.copy()

        # Log period summary
        period_total = period_pop["population"].sum()
        logger.info("Period %d total population: %,.0f", to_year, period_total)

    # Combine all results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Projection Complete")
    logger.info("=" * 60)

    population_df = pd.concat(all_populations, ignore_index=True)
    births_df = pd.DataFrame(all_births)
    deaths_df = pd.DataFrame(all_deaths)
    migration_df = pd.DataFrame(all_migrations)

    # Calculate state totals
    state_totals = calculate_state_totals(population_df, births_df, deaths_df, migration_df)

    # Log final summary
    for year in ALL_YEARS:
        year_total = population_df[population_df["year"] == year]["population"].sum()
        logger.info("Year %d: %,.0f", year, year_total)

    return ProjectionOutputs(
        population=population_df,
        births=births_df,
        deaths=deaths_df,
        migration=migration_df,
        state_totals=state_totals,
    )


def calculate_state_totals(
    population: pd.DataFrame,
    births: pd.DataFrame,
    deaths: pd.DataFrame,
    migration: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate state-level totals from county projections.

    SDC methodology: State totals are the sum of 53 county projections.

    Args:
        population: County-level population projections
        births: County-level births by period
        deaths: County-level deaths by period
        migration: County-level migration by period

    Returns:
        DataFrame with state totals by year
    """
    results = []

    for year in ALL_YEARS:
        year_pop = population[population["year"] == year]
        total_pop = year_pop["population"].sum()

        row = {
            "year": year,
            "population": total_pop,
            "male_population": year_pop[year_pop["sex"] == "Male"]["population"].sum(),
            "female_population": year_pop[year_pop["sex"] == "Female"]["population"].sum(),
        }

        # Add age distribution
        for age_group in AGE_GROUPS:
            row[f"pop_{age_group}"] = year_pop[year_pop["age_group"] == age_group][
                "population"
            ].sum()

        results.append(row)

    state_df = pd.DataFrame(results)

    # Add period components (births, deaths, migration)
    for year in PROJECTION_YEARS:
        period_births = births[births["period_end"] == year]["births"].sum()
        period_deaths = deaths[deaths["period_end"] == year]["deaths"].sum()
        period_migration = migration[migration["period_end"] == year]["migration"].sum()

        idx = state_df[state_df["year"] == year].index[0]
        state_df.loc[idx, "period_births"] = period_births
        state_df.loc[idx, "period_deaths"] = period_deaths
        state_df.loc[idx, "period_migration"] = period_migration
        state_df.loc[idx, "natural_change"] = period_births - period_deaths

    return state_df


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_base_population(filepath: str | Path) -> pd.DataFrame:
    """Load base population from CSV file.

    Expected columns: county, age_group, sex, population

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with base population
    """
    logger.info("Loading base population from: %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = ["county", "age_group", "sex", "population"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d rows, total population: %,.0f", len(df), df["population"].sum())
    return df


def load_survival_rates(filepath: str | Path) -> pd.DataFrame:
    """Load survival rates from CSV file.

    Expected columns: age_group, sex, survival_rate

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with survival rates
    """
    logger.info("Loading survival rates from: %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = ["age_group", "sex", "survival_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d survival rate entries", len(df))
    return df


def load_fertility_rates(filepath: str | Path) -> pd.DataFrame:
    """Load fertility rates from CSV file.

    Expected columns: age_group, fertility_rate
    Optional: county (for county-specific rates)

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with fertility rates
    """
    logger.info("Loading fertility rates from: %s", filepath)
    df = pd.read_csv(filepath)

    if "fertility_rate" not in df.columns:
        raise ValueError("Missing required column: fertility_rate")

    logger.info("Loaded %d fertility rate entries", len(df))
    return df


def load_migration_rates(filepath: str | Path) -> pd.DataFrame:
    """Load migration rates from CSV file.

    Expected columns: county, age_group, sex, migration_rate

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with migration rates
    """
    logger.info("Loading migration rates from: %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = ["county", "age_group", "sex", "migration_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d migration rate entries", len(df))
    return df


def load_adjustments(filepath: str | Path) -> pd.DataFrame | None:
    """Load manual adjustments from CSV file.

    Expected columns: county, age_group, sex, year, adjustment

    Args:
        filepath: Path to CSV file (returns None if file doesn't exist)

    Returns:
        DataFrame with adjustments or None
    """
    path = Path(filepath)
    if not path.exists():
        logger.info("No adjustments file found at: %s", filepath)
        return None

    logger.info("Loading adjustments from: %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = ["county", "age_group", "sex", "year", "adjustment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d adjustment entries", len(df))
    return df


def load_all_inputs(data_dir: str | Path) -> ProjectionInputs:
    """Load all projection inputs from a data directory.

    Expected files:
    - base_population.csv
    - survival_rates.csv
    - fertility_rates.csv
    - migration_rates.csv
    - adjustments.csv (optional)

    Args:
        data_dir: Path to directory containing input files

    Returns:
        ProjectionInputs with all loaded data
    """
    data_path = Path(data_dir)

    base_pop = load_base_population(data_path / "base_population.csv")
    survival = load_survival_rates(data_path / "survival_rates.csv")
    fertility = load_fertility_rates(data_path / "fertility_rates.csv")
    migration = load_migration_rates(data_path / "migration_rates.csv")
    adjustments = load_adjustments(data_path / "adjustments.csv")

    return ProjectionInputs(
        base_population=base_pop,
        survival_rates=survival,
        fertility_rates=fertility,
        migration_rates=migration,
        adjustments=adjustments,
    )


# =============================================================================
# Output Functions
# =============================================================================


def save_results(
    outputs: ProjectionOutputs, output_dir: str | Path, prefix: str = "sdc_replication"
) -> None:
    """Save projection results to CSV files.

    Args:
        outputs: ProjectionOutputs to save
        output_dir: Directory to save files
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save population projections
    pop_file = output_path / f"{prefix}_population.csv"
    outputs.population.to_csv(pop_file, index=False)
    logger.info("Saved population to: %s", pop_file)

    # Save births
    births_file = output_path / f"{prefix}_births.csv"
    outputs.births.to_csv(births_file, index=False)
    logger.info("Saved births to: %s", births_file)

    # Save deaths
    deaths_file = output_path / f"{prefix}_deaths.csv"
    outputs.deaths.to_csv(deaths_file, index=False)
    logger.info("Saved deaths to: %s", deaths_file)

    # Save migration
    mig_file = output_path / f"{prefix}_migration.csv"
    outputs.migration.to_csv(mig_file, index=False)
    logger.info("Saved migration to: %s", mig_file)

    # Save state totals
    state_file = output_path / f"{prefix}_state_totals.csv"
    outputs.state_totals.to_csv(state_file, index=False)
    logger.info("Saved state totals to: %s", state_file)


def print_summary(outputs: ProjectionOutputs) -> None:
    """Print a summary of projection results.

    Args:
        outputs: ProjectionOutputs to summarize
    """
    print("\n" + "=" * 60)
    print("SDC 2024 REPLICATION - PROJECTION SUMMARY")
    print("=" * 60)

    print("\nState Population by Year:")
    print("-" * 40)
    for _, row in outputs.state_totals.iterrows():
        year = int(row["year"])
        pop = row["population"]
        male = row["male_population"]
        female = row["female_population"]
        print(f"  {year}: {pop:>10,.0f}  (M: {male:>9,.0f}  F: {female:>9,.0f})")

    print("\nComponents of Change:")
    print("-" * 40)
    for _, row in outputs.state_totals.iterrows():
        year = int(row["year"])
        if year == BASE_YEAR:
            continue
        births = row.get("period_births", 0)
        deaths = row.get("period_deaths", 0)
        migration = row.get("period_migration", 0)
        natural = row.get("natural_change", 0)
        print(f"  {year - 5}-{year}:")
        print(f"    Births:     {births:>10,.0f}")
        print(f"    Deaths:     {deaths:>10,.0f}")
        print(f"    Natural:    {natural:>10,.0f}")
        print(f"    Migration:  {migration:>10,.0f}")

    print("\n" + "=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================


def main(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_outputs: bool = True,
) -> ProjectionOutputs:
    """Main entry point for running SDC 2024 replication projections.

    Args:
        data_dir: Directory containing input data files
            If None, uses ../data relative to this script
        output_dir: Directory to save outputs
            If None, uses ../output relative to this script
        save_outputs: Whether to save results to files

    Returns:
        ProjectionOutputs with all results
    """
    # Determine paths
    script_dir = Path(__file__).parent
    if data_dir is None:
        data_dir = script_dir.parent / "data"
    if output_dir is None:
        output_dir = script_dir.parent / "output"

    logger.info("Data directory: %s", data_dir)
    logger.info("Output directory: %s", output_dir)

    # Load inputs
    inputs = load_all_inputs(data_dir)

    # Run projections
    outputs = run_projections(inputs)

    # Print summary
    print_summary(outputs)

    # Save results
    if save_outputs:
        save_results(outputs, output_dir)

    return outputs


if __name__ == "__main__":
    # When run directly, don't execute - just print instructions
    print(
        """
SDC 2024 Replication: Cohort-Component Projection Engine
=========================================================

This script implements the SDC 2024 projection methodology.

Before running, ensure the following input files exist in the data directory:
- base_population.csv (county, age_group, sex, population)
- survival_rates.csv (age_group, sex, survival_rate)
- fertility_rates.csv (age_group, fertility_rate)
- migration_rates.csv (county, age_group, sex, migration_rate)
- adjustments.csv (optional: county, age_group, sex, year, adjustment)

To run projections:

    from projection_engine import main
    outputs = main(data_dir="/path/to/data", output_dir="/path/to/output")

Or import and use individual functions:

    from projection_engine import (
        load_all_inputs,
        run_projections,
        print_summary,
        save_results
    )

See METHODOLOGY_SPEC.md for detailed documentation of the SDC methodology.
"""
    )
