"""
Fertility module for cohort component projection.

Calculates births by applying age-specific fertility rates to female population.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


def calculate_births(
    female_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    year: int,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calculate births by applying fertility rates to female population.

    The cohort component method applies age-specific fertility rates (ASFR)
    to the female population in reproductive ages to generate births.
    Births are then split by sex and race/ethnicity.

    Args:
        female_population: DataFrame with columns [age, race, population]
                          Should only contain females
        fertility_rates: DataFrame with columns [age, race, fertility_rate]
                        Rates are births per woman
        year: Projection year
        config: Configuration dictionary with birth assumptions

    Returns:
        DataFrame with newborn population by sex and race
        Columns: [year, age, sex, race, population]

    Notes:
        - Fertility rates typically apply to ages 15-49
        - Standard sex ratio at birth: 51% male, 49% female (or 105:100)
        - Births are assigned age=0
        - Mother's race/ethnicity is assigned to newborn
    """
    if female_population.empty:
        logger.warning(f"Year {year}: Empty female population, no births calculated")
        return pd.DataFrame(columns=['year', 'age', 'sex', 'race', 'population'])

    # Default configuration
    if config is None:
        config = {}

    birth_config = config.get('rates', {}).get('fertility', {})
    reproductive_ages = birth_config.get('apply_to_ages', [15, 49])
    sex_ratio_at_birth = birth_config.get('sex_ratio_male', 0.51)  # proportion male

    logger.debug(f"Year {year}: Calculating births for ages {reproductive_ages[0]}-{reproductive_ages[1]}")

    # Validate inputs
    if 'age' not in female_population.columns or 'race' not in female_population.columns:
        raise ValueError("female_population must have 'age' and 'race' columns")

    if 'age' not in fertility_rates.columns or 'race' not in fertility_rates.columns:
        raise ValueError("fertility_rates must have 'age' and 'race' columns")

    # Filter to reproductive ages
    reproductive_pop = female_population[
        (female_population['age'] >= reproductive_ages[0]) &
        (female_population['age'] <= reproductive_ages[1])
    ].copy()

    if reproductive_pop.empty:
        logger.warning(f"Year {year}: No reproductive age females found")
        return pd.DataFrame(columns=['year', 'age', 'sex', 'race', 'population'])

    # Merge population with fertility rates
    births_by_mother = reproductive_pop.merge(
        fertility_rates,
        on=['age', 'race'],
        how='left',
        suffixes=('_pop', '_rate')
    )

    # Handle missing fertility rates (set to 0)
    if 'fertility_rate' not in births_by_mother.columns:
        logger.warning(f"Year {year}: No fertility_rate column after merge, using 0")
        births_by_mother['fertility_rate'] = 0.0
    else:
        births_by_mother['fertility_rate'] = births_by_mother['fertility_rate'].fillna(0.0)

    # Calculate total births by mother's age and race
    births_by_mother['births'] = (
        births_by_mother['population'] * births_by_mother['fertility_rate']
    )

    # Aggregate births by race (sum across mother's age)
    total_births_by_race = births_by_mother.groupby('race', as_index=False)['births'].sum()

    total_births = total_births_by_race['births'].sum()
    logger.info(f"Year {year}: Total births calculated: {total_births:,.0f}")

    # Split births by sex
    births_records = []

    for _, row in total_births_by_race.iterrows():
        race = row['race']
        total_births_race = row['births']

        # Male births
        births_records.append({
            'year': year,
            'age': 0,
            'sex': 'Male',
            'race': race,
            'population': total_births_race * sex_ratio_at_birth
        })

        # Female births
        births_records.append({
            'year': year,
            'age': 0,
            'sex': 'Female',
            'race': race,
            'population': total_births_race * (1 - sex_ratio_at_birth)
        })

    births_df = pd.DataFrame(births_records)

    if not births_df.empty:
        logger.debug(
            f"Year {year}: Births by sex - "
            f"Male: {births_df[births_df['sex']=='Male']['population'].sum():,.0f}, "
            f"Female: {births_df[births_df['sex']=='Female']['population'].sum():,.0f}"
        )

    return births_df


def validate_fertility_rates(
    fertility_rates: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, list]:
    """
    Validate fertility rates for plausibility.

    Args:
        fertility_rates: DataFrame with fertility rates
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for required columns
    required_cols = ['age', 'race', 'fertility_rate']
    missing_cols = [col for col in required_cols if col not in fertility_rates.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return False, issues

    # Check for negative rates
    if (fertility_rates['fertility_rate'] < 0).any():
        issues.append("Negative fertility rates found")

    # Check for implausibly high rates (> 0.35 births per woman per year)
    max_plausible_rate = 0.35
    if (fertility_rates['fertility_rate'] > max_plausible_rate).any():
        issues.append(f"Fertility rates > {max_plausible_rate} found (implausibly high)")

    # Check for fertility outside typical reproductive ages
    if config:
        reproductive_ages = config.get('rates', {}).get('fertility', {}).get('apply_to_ages', [15, 49])
        outside_range = fertility_rates[
            (fertility_rates['age'] < reproductive_ages[0]) |
            (fertility_rates['age'] > reproductive_ages[1])
        ]
        if not outside_range.empty and (outside_range['fertility_rate'] > 0.001).any():
            issues.append(
                f"Non-zero fertility rates outside age range {reproductive_ages[0]}-{reproductive_ages[1]}"
            )

    # Check for missing age-race combinations
    expected_races = fertility_rates['race'].unique()
    expected_ages = fertility_rates['age'].unique()
    expected_combinations = len(expected_races) * len(expected_ages)
    actual_combinations = len(fertility_rates)

    if actual_combinations < expected_combinations:
        issues.append(
            f"Missing age-race combinations: expected {expected_combinations}, got {actual_combinations}"
        )

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Fertility rate validation found {len(issues)} issues")
    else:
        logger.info("Fertility rates validated successfully")

    return is_valid, issues


def apply_fertility_scenario(
    fertility_rates: pd.DataFrame,
    scenario: str,
    year: int,
    base_year: int
) -> pd.DataFrame:
    """
    Apply fertility scenario adjustments to base rates.

    Args:
        fertility_rates: Base fertility rates
        scenario: Scenario name ('constant', '+10_percent', '-10_percent', 'trending')
        year: Current projection year
        base_year: Base year for projection

    Returns:
        Adjusted fertility rates
    """
    adjusted_rates = fertility_rates.copy()

    if scenario == 'constant':
        # No change
        pass

    elif scenario == '+10_percent':
        adjusted_rates['fertility_rate'] = adjusted_rates['fertility_rate'] * 1.10

    elif scenario == '-10_percent':
        adjusted_rates['fertility_rate'] = adjusted_rates['fertility_rate'] * 0.90

    elif scenario == 'trending':
        # Linear trend: assume 0.5% annual decline (common in developed countries)
        years_elapsed = year - base_year
        trend_factor = (1 - 0.005) ** years_elapsed
        adjusted_rates['fertility_rate'] = adjusted_rates['fertility_rate'] * trend_factor

    else:
        logger.warning(f"Unknown fertility scenario '{scenario}', using constant rates")

    # Ensure non-negative
    adjusted_rates['fertility_rate'] = adjusted_rates['fertility_rate'].clip(lower=0)

    return adjusted_rates
