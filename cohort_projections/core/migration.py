"""
Migration module for cohort component projection.

Applies net migration (domestic + international) to population cohorts.
"""

from typing import Any

import pandas as pd

from ..utils import get_logger_from_config

logger = get_logger_from_config(__name__)


def apply_migration(
    population: pd.DataFrame,
    migration_rates: pd.DataFrame,
    year: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Apply net migration to population cohorts.

    Net migration is the balance of in-migration and out-migration.
    Can be positive (net in-migration) or negative (net out-migration).
    Applied after aging and survival to reflect mid-period migration.

    Args:
        population: DataFrame with columns [year, age, sex, race, population]
                   Should be post-survival population
        migration_rates: DataFrame with columns [age, sex, race, net_migration]
                        or [age, sex, race, migration_rate]
        year: Projection year
        config: Configuration dictionary

    Returns:
        DataFrame with population after migration
        Columns: [year, age, sex, race, population]

    Notes:
        - Net migration can be specified as:
          1. Absolute numbers (net_migration column)
          2. Rates (migration_rate column, applied to population)
        - Domestic migration is typically from IRS county-to-county flows
        - International migration from ACS/Census estimates
        - Migration patterns vary significantly by age (young adults most mobile)
    """
    if population.empty:
        logger.warning(f"Year {year}: Empty population, no migration applied")
        return population.copy()

    # Default configuration
    if config is None:
        config = {}

    logger.debug(f"Year {year}: Applying migration")

    # Validate inputs
    required_pop_cols = ["age", "sex", "race", "population"]
    missing_cols = [col for col in required_pop_cols if col not in population.columns]
    if missing_cols:
        raise ValueError(f"population must have columns: {missing_cols}")

    # Check migration data format
    has_net_migration = "net_migration" in migration_rates.columns
    has_migration_rate = "migration_rate" in migration_rates.columns

    if not has_net_migration and not has_migration_rate:
        raise ValueError(
            "migration_rates must have either 'net_migration' or 'migration_rate' column"
        )

    # Merge population with migration data
    merged = population.merge(
        migration_rates, on=["age", "sex", "race"], how="left", suffixes=("_pop", "_mig")
    )

    # Calculate net migration amount
    if has_net_migration:
        # Absolute migration numbers provided
        merged["net_migration"] = merged["net_migration"].fillna(0.0)
        migration_amount = merged["net_migration"]

    elif has_migration_rate:
        # Migration rate provided (proportion of population)
        merged["migration_rate"] = merged["migration_rate"].fillna(0.0)
        migration_amount = merged["population"] * merged["migration_rate"]

    # Apply migration
    merged["population_after_migration"] = merged["population"] + migration_amount

    # Ensure non-negative population
    negative_pops = merged["population_after_migration"] < 0
    if negative_pops.any():
        num_negative = negative_pops.sum()
        logger.warning(
            f"Year {year}: {num_negative} cohorts have negative population after migration, "
            f"setting to 0"
        )
        merged.loc[negative_pops, "population_after_migration"] = 0.0

    # Build result DataFrame
    result = merged[["age", "sex", "race", "population_after_migration"]].copy()
    result.columns = pd.Index(["age", "sex", "race", "population"])
    result["year"] = year + 1  # Migration applied for next year

    # Log migration statistics
    total_before = population["population"].sum()
    total_after = result["population"].sum()
    net_migration_total = total_after - total_before

    logger.info(
        f"Year {year}: Population before migration: {total_before:,.0f}, "
        f"after: {total_after:,.0f}, net migration: {net_migration_total:+,.0f}"
    )

    # Log by direction if significant
    if abs(net_migration_total) > 0:
        if net_migration_total > 0:
            logger.debug(f"Year {year}: Net in-migration")
        else:
            logger.debug(f"Year {year}: Net out-migration")

    return result


def apply_migration_scenario(
    migration_rates: pd.DataFrame,
    scenario: str | dict,
    year: int,
    base_year: int,
    county_population: float | None = None,
) -> pd.DataFrame:
    """
    Apply migration scenario adjustments to base migration data.

    For time-varying scenarios (type='time_varying'), the CBO-derived factor
    is applied to the **international** component of migration only (ADR-040).
    Domestic migration is unaffected.  The decomposition uses ``intl_share``
    (proportion of net migration that is international, from PEP components):

        effective_factor = 1 - intl_share * (1 - factor)
        adjusted_rate = base_rate * effective_factor

    When ``intl_share`` is not provided, it defaults to 1.0 (entire rate
    treated as international), preserving backward compatibility.

    For additive reduction scenarios (type='additive_reduction', ADR-050),
    a per-capita rate decrement is subtracted from all cells. This avoids
    the sign-interaction bug where a multiplicative factor on net-negative
    migration rates produces ordering violations (restricted > baseline).
    The formula:

        reduction_rate = ref_intl * (1 - factor) / ref_pop
        adjusted_rate = base_rate - reduction_rate

    Since migration rates are already per-capita, the same reduction_rate
    applies uniformly to every cell. The total person-reduction for a county
    is proportional to its population (reduction_rate * county_pop).
    This guarantees ``restricted <= baseline`` for all counties regardless
    of the sign of the base migration rate.

    Args:
        migration_rates: Base migration rates or amounts
        scenario: Scenario name ('recent_average', '+25_percent', '-25_percent', 'zero')
                  or dict with type='time_varying'/'additive_reduction', schedule,
                  default_factor, and type-specific parameters (ADR-037, ADR-040, ADR-050)
        year: Current projection year
        base_year: Base year for projection
        county_population: Total population for the county being projected.
                          Retained for backward compatibility but not used by
                          the additive_reduction type (rates are per-capita).

    Returns:
        Adjusted migration rates
    """
    adjusted_rates = migration_rates.copy()

    # Determine which column to adjust
    if "net_migration" in adjusted_rates.columns:
        migration_col = "net_migration"
    elif "migration_rate" in adjusted_rates.columns:
        migration_col = "migration_rate"
    else:
        logger.warning("No migration column found, returning unchanged")
        return adjusted_rates

    # Handle time-varying scenario (dict config from ADR-037, updated ADR-040)
    if isinstance(scenario, dict) and scenario.get("type") == "time_varying":
        schedule = scenario.get("schedule", {})
        default_factor = scenario.get("default_factor", 1.0)
        factor = schedule.get(year, default_factor)
        if factor != 1.0:
            # Apply factor to international migration only (ADR-040).
            # intl_share is the proportion of total net migration that is
            # international, derived from PEP components data.  When the
            # share is available the effective multiplier is:
            #   adjusted = rate * (1 - intl_share * (1 - factor))
            # This leaves the domestic component unchanged and scales only
            # the international component by ``factor``.
            intl_share = scenario.get("intl_share", 1.0)
            effective_factor = 1.0 - intl_share * (1.0 - factor)
            adjusted_rates[migration_col] = adjusted_rates[migration_col] * effective_factor
            logger.info(
                f"Year {year}: Time-varying migration factor = {factor:.2f}, "
                f"intl_share = {intl_share:.3f}, "
                f"effective_factor = {effective_factor:.4f}"
            )
        return adjusted_rates

    # Handle additive reduction scenario (ADR-050)
    # Subtracts a per-capita rate decrement from all cells, guaranteeing
    # restricted <= baseline regardless of the sign of the base rate.
    if isinstance(scenario, dict) and scenario.get("type") == "additive_reduction":
        schedule = scenario.get("schedule", {})
        default_factor = scenario.get("default_factor", 1.0)
        factor = schedule.get(year, default_factor)
        if factor < 1.0:
            ref_intl = scenario.get("reference_intl_migration", 0)
            ref_pop = scenario.get("reference_population", 1)
            annual_reduction = ref_intl * (1.0 - factor)  # persons/year not arriving
            reduction_rate = annual_reduction / ref_pop  # per-capita rate decrement

            # Since rates are already per-capita, apply the same decrement to
            # every cell. Total person-reduction scales with county population.
            adjusted_rates[migration_col] = (
                adjusted_rates[migration_col] - reduction_rate
            )
            logger.info(
                f"Year {year}: Additive migration reduction — factor={factor:.2f}, "
                f"annual_reduction={annual_reduction:,.0f} persons, "
                f"reduction_rate={reduction_rate:.6f}"
            )
        return adjusted_rates

    if scenario == "recent_average" or scenario == "constant":
        # No change - use base migration
        pass

    elif scenario == "+25_percent":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 1.25

    elif scenario == "-25_percent":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 0.75

    elif scenario == "-15_percent":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 0.85

    elif scenario == "+5_percent":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 1.05

    elif scenario == "-5_percent":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 0.95

    elif scenario == "zero":
        adjusted_rates[migration_col] = 0.0

    elif scenario == "double":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 2.0

    elif scenario == "half":
        adjusted_rates[migration_col] = adjusted_rates[migration_col] * 0.5

    else:
        logger.warning(f"Unknown migration scenario '{scenario}', using base migration")

    return adjusted_rates


def validate_migration_data(
    migration_rates: pd.DataFrame,
    population: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[bool, list]:
    """
    Validate migration data for plausibility.

    Args:
        migration_rates: DataFrame with migration data
        population: Optional population data for rate validation
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for required columns
    has_net_migration = "net_migration" in migration_rates.columns
    has_migration_rate = "migration_rate" in migration_rates.columns

    if not has_net_migration and not has_migration_rate:
        issues.append("Must have either 'net_migration' or 'migration_rate' column")
        return False, issues

    required_id_cols = ["age", "sex", "race"]
    missing_cols = [col for col in required_id_cols if col not in migration_rates.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return False, issues

    # Check for extreme values
    if has_net_migration:
        # Check for implausibly large absolute migration
        max_abs_migration = migration_rates["net_migration"].abs().max()
        if max_abs_migration > 10000:
            issues.append(
                f"Very large net migration value found: {max_abs_migration:,.0f} "
                f"(possible data error)"
            )

    if has_migration_rate:
        # Check for rates outside reasonable bounds
        if (migration_rates["migration_rate"] < -1).any():
            issues.append("Migration rates < -1.0 found (more than 100% out-migration)")

        if (migration_rates["migration_rate"] > 1).any():
            issues.append("Migration rates > 1.0 found (more than 100% in-migration)")

        # Check for extreme rates
        extreme_rates = migration_rates[migration_rates["migration_rate"].abs() > 0.5]
        if not extreme_rates.empty:
            issues.append(f"Extreme migration rates (>50%) found for {len(extreme_rates)} cohorts")

    # If population provided, check if migration would cause negative population
    if population is not None and has_net_migration:
        merged = population.merge(migration_rates, on=["age", "sex", "race"], how="left")
        merged["net_migration"] = merged["net_migration"].fillna(0.0)
        merged["result_pop"] = merged["population"] + merged["net_migration"]

        negative_results = merged["result_pop"] < 0
        if negative_results.any():
            issues.append(
                f"Migration would cause negative population for {negative_results.sum()} cohorts"
            )

    # Check for missing age-sex-race combinations
    expected_sexes = migration_rates["sex"].unique()
    expected_races = migration_rates["race"].unique()
    expected_ages = migration_rates["age"].unique()
    expected_combinations = len(expected_sexes) * len(expected_races) * len(expected_ages)
    actual_combinations = len(migration_rates)

    if actual_combinations < expected_combinations:
        issues.append(
            f"Missing age-sex-race combinations: expected {expected_combinations}, "
            f"got {actual_combinations}"
        )

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Migration data validation found {len(issues)} issues")
    else:
        logger.info("Migration data validated successfully")

    return is_valid, issues


def distribute_international_migration(
    total_international: float,
    population: pd.DataFrame,
    age_distribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Distribute total international migration to cohorts.

    International migration often comes as a state-level total that needs
    to be distributed across age/sex/race groups.

    Args:
        total_international: Total international net migration for the year
        population: Current population by age/sex/race
        age_distribution: Optional age distribution of immigrants
                         If None, proportional to existing population

    Returns:
        DataFrame with net_migration by age/sex/race
    """
    if age_distribution is None:
        # Distribute proportional to population
        total_pop = population["population"].sum()

        if total_pop == 0:
            logger.warning("Cannot distribute migration to zero population")
            result = population[["age", "sex", "race"]].copy()
            result["net_migration"] = 0.0
            return result

        population["proportion"] = population["population"] / total_pop
        population["net_migration"] = total_international * population["proportion"]

        result = population[["age", "sex", "race", "net_migration"]].copy()

    else:
        # Use provided age distribution
        # Merge population with age distribution
        merged = population.merge(age_distribution, on=["age", "sex", "race"], how="left")

        # Normalize distribution
        if "weight" in age_distribution.columns:
            total_weight = merged["weight"].sum()
            merged["weight"] = merged["weight"].fillna(0.0)
            merged["proportion"] = merged["weight"] / total_weight if total_weight > 0 else 0
        else:
            logger.warning("No 'weight' column in age_distribution, using equal distribution")
            merged["proportion"] = 1.0 / len(merged)

        merged["net_migration"] = total_international * merged["proportion"]

        result = merged[["age", "sex", "race", "net_migration"]].copy()

    logger.info(
        f"Distributed {total_international:,.0f} international migrants "
        f"across {len(result)} cohorts"
    )

    return result


def combine_domestic_international(
    domestic_migration: pd.DataFrame, international_migration: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine domestic and international migration into total net migration.

    Args:
        domestic_migration: Domestic net migration by cohort
        international_migration: International net migration by cohort

    Returns:
        Combined net migration DataFrame
    """
    # Ensure both have net_migration column
    if "net_migration" not in domestic_migration.columns:
        raise ValueError("domestic_migration must have 'net_migration' column")

    if "net_migration" not in international_migration.columns:
        raise ValueError("international_migration must have 'net_migration' column")

    # Merge on cohort identifiers
    combined = domestic_migration.merge(
        international_migration,
        on=["age", "sex", "race"],
        how="outer",
        suffixes=("_domestic", "_international"),
    )

    # Fill NaN with 0
    combined["net_migration_domestic"] = combined["net_migration_domestic"].fillna(0.0)
    combined["net_migration_international"] = combined["net_migration_international"].fillna(0.0)

    # Sum migration components
    combined["net_migration"] = (
        combined["net_migration_domestic"] + combined["net_migration_international"]
    )

    result = combined[["age", "sex", "race", "net_migration"]].copy()

    total_domestic = combined["net_migration_domestic"].sum()
    total_international = combined["net_migration_international"].sum()
    total_combined = result["net_migration"].sum()

    logger.info(
        f"Combined migration - Domestic: {total_domestic:+,.0f}, "
        f"International: {total_international:+,.0f}, "
        f"Total: {total_combined:+,.0f}"
    )

    return result
