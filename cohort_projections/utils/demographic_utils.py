"""
Demographic utility functions for cohort projections.

Common demographic calculations and helper functions.
"""

import numpy as np
import pandas as pd


def create_age_groups(min_age: int = 0, max_age: int = 90, group_size: int = 1) -> list[str]:
    """
    Create age group labels.

    Args:
        min_age: Minimum age
        max_age: Maximum age (open-ended, e.g., 90+)
        group_size: Size of age groups (1 for single year, 5 for quinquennial)

    Returns:
        List of age group labels
    """
    age_groups = []

    for age in range(min_age, max_age, group_size):
        if age + group_size <= max_age:
            if group_size == 1:
                age_groups.append(str(age))
            else:
                age_groups.append(f"{age}-{age + group_size - 1}")
        else:
            age_groups.append(f"{age}+")
            break

    # Ensure we have the open-ended final group
    if not age_groups[-1].endswith("+"):
        age_groups.append(f"{max_age}+")

    return age_groups


def calculate_sex_ratio(population: pd.DataFrame) -> float:
    """
    Calculate sex ratio (males per 100 females).

    Args:
        population: DataFrame with 'sex' and 'population' columns

    Returns:
        Sex ratio
    """
    male_pop = population[population["sex"] == "Male"]["population"].sum()
    female_pop = population[population["sex"] == "Female"]["population"].sum()

    if female_pop == 0:
        return np.nan

    return (male_pop / female_pop) * 100


def calculate_dependency_ratio(population: pd.DataFrame) -> dict:
    """
    Calculate age dependency ratios.

    Args:
        population: DataFrame with 'age' and 'population' columns

    Returns:
        Dictionary with youth, old-age, and total dependency ratios
    """
    # Assuming single-year ages
    youth_pop = population[population["age"] < 18]["population"].sum()
    working_pop = population[(population["age"] >= 18) & (population["age"] < 65)][
        "population"
    ].sum()
    elderly_pop = population[population["age"] >= 65]["population"].sum()

    if working_pop == 0:
        return {
            "youth_dependency": np.nan,
            "old_age_dependency": np.nan,
            "total_dependency": np.nan,
        }

    return {
        "youth_dependency": (youth_pop / working_pop) * 100,
        "old_age_dependency": (elderly_pop / working_pop) * 100,
        "total_dependency": ((youth_pop + elderly_pop) / working_pop) * 100,
    }


def calculate_median_age(population: pd.DataFrame) -> float:
    """
    Calculate median age of population.

    Args:
        population: DataFrame with 'age' and 'population' columns

    Returns:
        Median age
    """
    # Expand to individual records
    ages = []
    for _, row in population.iterrows():
        age = row["age"]
        count = int(row["population"])
        # Handle age groups like "90+"
        if isinstance(age, str) and "+" in age:
            age = int(age.replace("+", ""))
        ages.extend([age] * count)

    if not ages:
        return np.nan

    return float(np.median(ages))


def interpolate_missing_ages(
    population: pd.DataFrame, age_column: str = "age", value_column: str = "population"
) -> pd.DataFrame:
    """
    Interpolate missing age groups.

    Args:
        population: DataFrame with age and population
        age_column: Name of age column
        value_column: Name of value column to interpolate

    Returns:
        DataFrame with interpolated values
    """
    df = population.copy()

    # Convert age to numeric (handle "90+" etc.)
    df["age_numeric"] = df[age_column].apply(
        lambda x: int(str(x).replace("+", "")) if pd.notna(x) else np.nan
    )

    # Sort by age
    df = df.sort_values("age_numeric")

    # Interpolate missing values
    df[value_column] = df[value_column].interpolate(method="linear")

    return df


def aggregate_race_categories(
    population: pd.DataFrame,
    race_column: str = "race",
    aggregation_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate detailed race categories into broader groups.

    Args:
        population: DataFrame with race column
        race_column: Name of race column
        aggregation_map: Dictionary mapping detailed races to broader categories

    Returns:
        DataFrame with aggregated race categories
    """
    if aggregation_map is None:
        # Default aggregation to 6 major categories
        aggregation_map = {
            "White alone, Non-Hispanic": "White NH",
            "Black alone, Non-Hispanic": "Black NH",
            "AIAN alone, Non-Hispanic": "AIAN NH",
            "Asian/PI alone, Non-Hispanic": "Asian/PI NH",
            "Two or more races, Non-Hispanic": "Multiracial NH",
            "Hispanic (any race)": "Hispanic",
        }

    df = population.copy()
    df["race_aggregated"] = df[race_column].map(aggregation_map)

    return (
        df.groupby(
            [col for col in df.columns if col not in [race_column, "population", "race_aggregated"]]
            + ["race_aggregated"]
        )["population"]
        .sum()
        .reset_index()
    )


def calculate_growth_rate(pop_start: float, pop_end: float, years: int) -> float:
    """
    Calculate compound annual growth rate.

    Args:
        pop_start: Starting population
        pop_end: Ending population
        years: Number of years

    Returns:
        Annual growth rate (as decimal)
    """
    if pop_start <= 0 or years == 0:
        return np.nan

    return (pop_end / pop_start) ** (1 / years) - 1


def validate_cohort_sums(
    cohorts: pd.DataFrame, total: float, tolerance: float = 0.01
) -> tuple[bool, float]:
    """
    Validate that cohort populations sum to expected total.

    Args:
        cohorts: DataFrame with cohort populations
        total: Expected total population
        tolerance: Acceptable relative error

    Returns:
        Tuple of (is_valid, relative_error)
    """
    cohort_sum = cohorts["population"].sum()
    relative_error = abs(cohort_sum - total) / total if total > 0 else np.nan

    is_valid = relative_error <= tolerance

    return is_valid, relative_error
