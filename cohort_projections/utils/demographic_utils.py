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


# ---------------------------------------------------------------------------
# Sprague osculatory interpolation (ADR-048)
# ---------------------------------------------------------------------------

# Standard Sprague multiplier matrix for graduating 5-year age groups to
# single years of age. Each row produces one single-year value from the
# surrounding 5-year group totals. The multipliers are applied to a window
# of 5 consecutive group totals [Y_{i-2}, Y_{i-1}, Y_i, Y_{i+1}, Y_{i+2}]
# to produce 5 single-year values within group i (the center group).
#
# Key property: column sums = [0, 0, 1, 0, 0], ensuring the 5 interpolated
# values sum exactly to the center group's total.
#
# Reference: Sprague (1880); Siegel & Swanson, "The Methods and Materials
# of Demography", 2nd ed.; UN Population Division; DemoTools R package.
SPRAGUE_MULTIPLIERS: np.ndarray = np.array([
    [-0.0128,  0.0848,  0.1504, -0.0240,  0.0016],
    [-0.0016,  0.0144,  0.2224, -0.0416,  0.0064],
    [ 0.0064, -0.0336,  0.2544, -0.0336,  0.0064],
    [ 0.0064, -0.0416,  0.2224,  0.0144, -0.0016],
    [ 0.0016, -0.0240,  0.1504,  0.0848, -0.0128],
])


def _pad_groups(group_totals: np.ndarray) -> np.ndarray:
    """
    Pad a group-total array with 2 virtual groups on each end using
    linear extrapolation, following the DemoTools / UN approach.

    This ensures the standard center-group Sprague multipliers can be
    applied to every original group (including the first two and last two)
    by always having 2 neighbors on each side.

    Args:
        group_totals: 1-D array of n group totals.

    Returns:
        1-D array of length n + 4 (2 padded on each end).
    """
    n = len(group_totals)
    padded = np.empty(n + 4)

    # Copy original values to positions 2..n+1
    padded[2 : n + 2] = group_totals

    # Linear extrapolation for the left boundary:
    #   padded[1] = 2 * group_totals[0] - group_totals[1]
    #   padded[0] = 2 * padded[1] - group_totals[0]
    # Simplified:
    padded[1] = 2.0 * group_totals[0] - group_totals[1]
    padded[0] = 2.0 * padded[1] - group_totals[0]

    # Linear extrapolation for the right boundary:
    padded[n + 2] = 2.0 * group_totals[n - 1] - group_totals[n - 2]
    padded[n + 3] = 2.0 * padded[n + 2] - group_totals[n - 1]

    return padded


def sprague_graduate(
    group_totals: np.ndarray,
    *,
    clamp_negatives: bool = True,
) -> np.ndarray:
    """
    Graduate 5-year age group totals to single-year-of-age values using
    Sprague osculatory interpolation (ADR-048).

    This is the standard method used by the UN Population Division and
    Census Bureau for converting quinquennial population data to single-year
    estimates. The method produces smooth transitions between age groups
    while preserving each group's total (up to floating-point precision).

    The implementation pads the input with 2 linearly-extrapolated virtual
    groups on each end, then applies the standard 5-coefficient Sprague
    multipliers with every original group at the center of its window.
    This ensures exact group-total preservation for all groups including
    the first and last.

    Args:
        group_totals: 1-D array of 5-year age group totals. Must have at
            least 5 elements. For the standard 18-group demographic age
            schedule (0-4 through 85+), pass an 18-element array.
        clamp_negatives: If True, clamp any negative interpolated values
            to zero and renormalize within each group so the 5 single-year
            values still sum to the original group total. This is needed
            for very small populations where Sprague can produce small
            negative overshoots at extreme ages. Default True.

    Returns:
        1-D array of single-year values. Length = 5 * len(group_totals).

    Raises:
        ValueError: If fewer than 5 group totals are provided.

    Example:
        >>> totals = np.array([5000, 4800, 4500, 4200, 3900,
        ...                    3700, 3500, 3300, 3100, 2900,
        ...                    2700, 2500, 2300, 2100, 1900,
        ...                    1700, 1500, 1300])
        >>> single_years = sprague_graduate(totals)
        >>> len(single_years)
        90
        >>> abs(single_years[:5].sum() - totals[0]) < 0.01
        True

    See Also:
        ADR-048: Single-Year-of-Age Base Population from SC-EST Data
        DemoTools R package: ``graduate_sprague()``
    """
    group_totals = np.asarray(group_totals, dtype=float)
    n_groups = len(group_totals)

    if n_groups < 5:
        raise ValueError(
            f"Sprague interpolation requires at least 5 groups, got {n_groups}"
        )

    # Pad with 2 virtual groups on each side so every original group can
    # sit at the center of a 5-group window.
    padded = _pad_groups(group_totals)

    single_years = np.empty(n_groups * 5)

    for i in range(n_groups):
        # In the padded array, original group i is at index i + 2.
        # The 5-group window centered on it spans indices i..i+4.
        window = padded[i : i + 5]

        # Apply standard Sprague multipliers (center-group formula)
        values = SPRAGUE_MULTIPLIERS @ window

        # Place the 5 single-year values into the output
        start = i * 5
        single_years[start : start + 5] = values

    # Clamp negatives and renormalize within each group
    if clamp_negatives:
        for i in range(n_groups):
            start = i * 5
            group_vals = single_years[start : start + 5]
            if np.any(group_vals < 0):
                group_vals = np.maximum(group_vals, 0.0)
                group_sum = group_vals.sum()
                original_total = group_totals[i]
                if group_sum > 0:
                    group_vals = group_vals * (original_total / group_sum)
                else:
                    # All values were negative/zero; distribute uniformly
                    group_vals = np.full(5, original_total / 5.0)
                single_years[start : start + 5] = group_vals

    return single_years
