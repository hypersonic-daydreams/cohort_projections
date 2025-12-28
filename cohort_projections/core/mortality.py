"""
Mortality module for cohort component projection.

Applies survival rates to age the population and account for deaths.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


def apply_survival_rates(
    population: pd.DataFrame,
    survival_rates: pd.DataFrame,
    year: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Apply survival rates to population cohorts.

    The cohort component method applies age/sex/race-specific survival rates
    to move population from age t to age t+1. Survival rate is the probability
    of surviving from one age to the next (typically from July 1 to July 1).

    Args:
        population: DataFrame with columns [year, age, sex, race, population]
        survival_rates: DataFrame with columns [age, sex, race, survival_rate]
                       Rates are probabilities (0-1) of surviving to next age
        year: Projection year
        config: Configuration dictionary

    Returns:
        DataFrame with survived population (aged by 1 year)
        Columns: [year, age, sex, race, population]

    Notes:
        - Survival rates are typically derived from life tables
        - Age 90+ is an open-ended group requiring special handling
        - Survival rate for age 90+ applies within-group survival
        - Infants (age 0) use infant survival rate
    """
    if population.empty:
        logger.warning(f"Year {year}: Empty population, no survival applied")
        return pd.DataFrame(columns=["year", "age", "sex", "race", "population"])

    # Default configuration
    if config is None:
        config = {}

    demographics_config = config.get("demographics", {})
    max_age = demographics_config.get("age_groups", {}).get("max_age", 90)
    mortality_config = config.get("rates", {}).get("mortality", {})
    improvement_factor = mortality_config.get("improvement_factor", 0.0)
    base_year = config.get("project", {}).get("base_year", 2025)

    logger.debug(f"Year {year}: Applying survival rates (max_age={max_age})")

    # Validate inputs
    required_cols = ["age", "sex", "race", "population"]
    missing_cols = [col for col in required_cols if col not in population.columns]
    if missing_cols:
        raise ValueError(f"population must have columns: {missing_cols}")

    required_rate_cols = ["age", "sex", "race", "survival_rate"]
    missing_rate_cols = [col for col in required_rate_cols if col not in survival_rates.columns]
    if missing_rate_cols:
        raise ValueError(f"survival_rates must have columns: {missing_rate_cols}")

    # Apply mortality improvement if configured
    if improvement_factor > 0:
        survival_rates = apply_mortality_improvement(
            survival_rates, year, base_year, improvement_factor
        )

    # Separate regular ages from open-ended age group
    regular_pop = population[population["age"] < max_age].copy()
    open_age_pop = population[population["age"] == max_age].copy()

    survived_records = []

    # Process regular ages (0 to max_age-1)
    if not regular_pop.empty:
        # Merge population with survival rates
        merged = regular_pop.merge(
            survival_rates, on=["age", "sex", "race"], how="left", suffixes=("_pop", "_rate")
        )

        # Handle missing survival rates (log warning and set to 0)
        missing_rates = merged["survival_rate"].isna()
        if missing_rates.any():
            logger.warning(
                f"Year {year}: Missing survival rates for {missing_rates.sum()} cohorts, "
                f"setting to 0"
            )
            merged["survival_rate"] = merged["survival_rate"].fillna(0.0)

        # Calculate survived population
        merged["survived_population"] = merged["population"] * merged["survival_rate"]

        # Age population by 1 year
        merged["age"] = merged["age"] + 1

        # Build records
        for _, row in merged.iterrows():
            survived_records.append(
                {
                    "year": year + 1,
                    "age": row["age"],
                    "sex": row["sex"],
                    "race": row["race"],
                    "population": row["survived_population"],
                }
            )

    # Process open-ended age group (90+)
    # These individuals stay at age 90+ but experience within-group mortality
    if not open_age_pop.empty:
        merged_open = open_age_pop.merge(
            survival_rates[survival_rates["age"] == max_age],
            on=["age", "sex", "race"],
            how="left",
            suffixes=("_pop", "_rate"),
        )

        # Handle missing survival rates for 90+
        missing_rates = merged_open["survival_rate"].isna()
        if missing_rates.any():
            logger.warning(
                f"Year {year}: Missing survival rates for age {max_age}+ group, "
                f"setting to 0.5 (default)"
            )
            merged_open["survival_rate"] = merged_open["survival_rate"].fillna(0.5)

        merged_open["survived_population"] = (
            merged_open["population"] * merged_open["survival_rate"]
        )

        # Age remains at max_age (open-ended group)
        for _, row in merged_open.iterrows():
            survived_records.append(
                {
                    "year": year + 1,
                    "age": max_age,
                    "sex": row["sex"],
                    "race": row["race"],
                    "population": row["survived_population"],
                }
            )

    # Handle new entrants to 90+ group (those who were 89 and survived to 90)
    # These are already included in the regular_pop processing above

    survived_df = pd.DataFrame(survived_records)

    if not survived_df.empty:
        total_before = population["population"].sum()
        total_after = survived_df["population"].sum()
        deaths = total_before - total_after
        crude_death_rate = deaths / total_before if total_before > 0 else 0

        logger.info(
            f"Year {year}: Population before: {total_before:,.0f}, "
            f"after survival: {total_after:,.0f}, "
            f"deaths: {deaths:,.0f} (CDR: {crude_death_rate:.4f})"
        )
    else:
        logger.warning(f"Year {year}: No survived population generated")

    return survived_df


def apply_mortality_improvement(
    survival_rates: pd.DataFrame, current_year: int, base_year: int, improvement_factor: float
) -> pd.DataFrame:
    """
    Apply mortality improvement to survival rates over time.

    Mortality improvement means death rates decline over time, so survival
    rates increase. Common assumption: 0.5% annual improvement.

    Args:
        survival_rates: Base survival rates
        current_year: Current projection year
        base_year: Base year for improvement calculation
        improvement_factor: Annual improvement rate (e.g., 0.005 = 0.5%)

    Returns:
        Adjusted survival rates

    Notes:
        - Improvement is compounded annually
        - Survival rates are capped at 1.0 (cannot exceed 100%)
        - Death rate improvement: DR_t = DR_base * (1 - improvement)^t
        - Survival rate improvement: SR_t = 1 - DR_t
    """
    years_elapsed = current_year - base_year

    if years_elapsed <= 0 or improvement_factor == 0:
        return survival_rates.copy()

    adjusted_rates = survival_rates.copy()

    # Convert survival rate to death rate
    adjusted_rates["death_rate"] = 1 - adjusted_rates["survival_rate"]

    # Apply improvement to death rates
    improvement_multiplier = (1 - improvement_factor) ** years_elapsed
    adjusted_rates["death_rate"] = adjusted_rates["death_rate"] * improvement_multiplier

    # Convert back to survival rate
    adjusted_rates["survival_rate"] = 1 - adjusted_rates["death_rate"]

    # Cap at 1.0
    adjusted_rates["survival_rate"] = adjusted_rates["survival_rate"].clip(upper=1.0)

    # Drop intermediate column
    adjusted_rates = adjusted_rates.drop(columns=["death_rate"])

    logger.debug(f"Applied {years_elapsed} years of {improvement_factor:.4f} mortality improvement")

    return adjusted_rates


def validate_survival_rates(
    survival_rates: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[bool, list]:
    """
    Validate survival rates for plausibility.

    Args:
        survival_rates: DataFrame with survival rates
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for required columns
    required_cols = ["age", "sex", "race", "survival_rate"]
    missing_cols = [col for col in required_cols if col not in survival_rates.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return False, issues

    # Check for rates outside [0, 1]
    if (survival_rates["survival_rate"] < 0).any():
        issues.append("Negative survival rates found")

    if (survival_rates["survival_rate"] > 1).any():
        issues.append("Survival rates > 1.0 found")

    # Check for implausibly low infant survival (< 0.99 in developed countries)
    infant_rates = survival_rates[survival_rates["age"] == 0]
    if not infant_rates.empty:
        min_infant_survival = infant_rates["survival_rate"].min()
        if min_infant_survival < 0.98:
            issues.append(f"Implausibly low infant survival rate: {min_infant_survival:.4f}")

    # Check for increasing mortality with age
    # Group by sex and race, check if survival generally decreases with age
    for (sex, race), group in survival_rates.groupby(["sex", "race"]):
        sorted_group = group.sort_values("age")
        # Allow some variation, but check for major inversions
        survival_values = np.asarray(sorted_group["survival_rate"].values)
        # Check if early ages have lower survival than much older ages
        if len(survival_values) > 10 and float(np.mean(survival_values[:5])) < float(
            np.mean(survival_values[-5:])
        ):
            issues.append(
                f"Implausible pattern: young ages have lower survival than old ages "
                f"for {sex}/{race}"
            )

    # Check for missing age-sex-race combinations
    expected_sexes = survival_rates["sex"].unique()
    expected_races = survival_rates["race"].unique()
    expected_ages = survival_rates["age"].unique()
    expected_combinations = len(expected_sexes) * len(expected_races) * len(expected_ages)
    actual_combinations = len(survival_rates)

    if actual_combinations < expected_combinations:
        issues.append(
            f"Missing age-sex-race combinations: expected {expected_combinations}, "
            f"got {actual_combinations}"
        )

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Survival rate validation found {len(issues)} issues")
    else:
        logger.info("Survival rates validated successfully")

    return is_valid, issues


def calculate_life_expectancy(
    survival_rates: pd.DataFrame, age_start: int = 0, max_age: int = 90
) -> pd.DataFrame:
    """
    Calculate life expectancy from survival rates.

    This is a utility function to compute period life expectancy at various ages.

    Args:
        survival_rates: DataFrame with survival rates
        age_start: Starting age for calculation
        max_age: Maximum age (open-ended group)

    Returns:
        DataFrame with life expectancy by sex and race
    """
    # This is a simplified calculation for demonstration
    # Full life table calculation would include person-years lived in interval

    life_exp_results = []

    for (sex, race), group in survival_rates.groupby(["sex", "race"]):
        sorted_group = group.sort_values("age")

        # Simple calculation: sum of survival probabilities from age_start
        sorted_group = sorted_group[sorted_group["age"] >= age_start]

        if len(sorted_group) == 0:
            continue

        # Cumulative survival from age_start
        cumulative_survival = sorted_group["survival_rate"].cumprod()

        # Life expectancy ≈ sum of survival probabilities
        # (simplified; proper method uses Lx and Tx from life table)
        life_exp = cumulative_survival.sum()

        life_exp_results.append(
            {"sex": sex, "race": race, "age": age_start, "life_expectancy": life_exp}
        )

    return pd.DataFrame(life_exp_results)
