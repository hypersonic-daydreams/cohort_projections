"""
Base population loader for cohort projections.

Loads and transforms base population data into the format required by
the cohort-component projection engine. Creates age x sex x race cohort
matrices from raw Census data and pre-computed distributions.

The projection engine expects DataFrames with columns:
    [year, age, sex, race, population]

Where:
    - year: Base year (e.g., 2025)
    - age: Single year of age (0-90)
    - sex: "Male" or "Female"
    - race: One of the 6 standard race/ethnicity categories
    - population: Population count for that cohort
"""

from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils.config_loader import ConfigLoader
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


# Mapping from raw data race codes to standard projection categories
RACE_CODE_MAP = {
    "white_nonhispanic": "White alone, Non-Hispanic",
    "black_nonhispanic": "Black alone, Non-Hispanic",
    "aian_nonhispanic": "AIAN alone, Non-Hispanic",
    "asian_nonhispanic": "Asian/PI alone, Non-Hispanic",
    "nhpi_nonhispanic": "Asian/PI alone, Non-Hispanic",  # Combine NHPI with Asian
    "multiracial_nonhispanic": "Two or more races, Non-Hispanic",
    "other_nonhispanic": "Two or more races, Non-Hispanic",  # Map other to multiracial
    "hispanic": "Hispanic (any race)",
}

# Age group to single-year age mapping
AGE_GROUP_RANGES = {
    "0-4": list(range(5)),
    "5-9": list(range(5, 10)),
    "10-14": list(range(10, 15)),
    "15-19": list(range(15, 20)),
    "20-24": list(range(20, 25)),
    "25-29": list(range(25, 30)),
    "30-34": list(range(30, 35)),
    "35-39": list(range(35, 40)),
    "40-44": list(range(40, 45)),
    "45-49": list(range(45, 50)),
    "50-54": list(range(50, 55)),
    "55-59": list(range(55, 60)),
    "60-64": list(range(60, 65)),
    "65-69": list(range(65, 70)),
    "70-74": list(range(70, 75)),
    "75-79": list(range(75, 80)),
    "80-84": list(range(80, 85)),
    "85+": list(range(85, 91)),  # 85-90 (90 is the max age group)
}


def load_state_age_sex_race_distribution(
    distribution_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load state-level age-sex-race distribution.

    This distribution is used to allocate county total populations into
    detailed cohorts when county-specific detailed data is not available.

    Args:
        distribution_path: Path to distribution CSV file
                          (default: data/raw/population/nd_age_sex_race_distribution.csv)
        config: Optional configuration dictionary

    Returns:
        DataFrame with columns:
        - age: Single year of age (0-90)
        - sex: "Male" or "Female"
        - race: Standard race category
        - proportion: Proportion of total population in this cohort

    Example:
        >>> dist = load_state_age_sex_race_distribution()
        >>> dist[dist['age'] == 25].head()
              age     sex                        race  proportion
        ...    25    Male  White alone, Non-Hispanic    0.00523
    """
    logger.info("Loading state-level age-sex-race distribution")

    # Set default path
    if distribution_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        distribution_path = (
            project_root / "data" / "raw" / "population" / "nd_age_sex_race_distribution.csv"
        )

    distribution_path = Path(distribution_path)

    if not distribution_path.exists():
        raise FileNotFoundError(f"Distribution file not found: {distribution_path}")

    # Load raw distribution data
    raw_dist = pd.read_csv(distribution_path)

    logger.info(f"Loaded {len(raw_dist)} records from {distribution_path}")

    # Expand age groups to single-year ages
    expanded_rows = []
    for _, row in raw_dist.iterrows():
        age_group = row["age_group"]
        sex = row["sex"].title()  # Capitalize: "male" -> "Male"

        # Map race code to standard category
        race_code = row["race_ethnicity"]
        race = RACE_CODE_MAP.get(race_code, race_code)

        # Get single-year ages for this group
        if age_group in AGE_GROUP_RANGES:
            ages = AGE_GROUP_RANGES[age_group]
            # Distribute proportion evenly across single years in group
            proportion_per_year = row["proportion"] / len(ages)

            for age in ages:
                expanded_rows.append(
                    {
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "proportion": proportion_per_year,
                    }
                )
        else:
            logger.warning(f"Unknown age group: {age_group}")

    expanded_df = pd.DataFrame(expanded_rows)

    # Aggregate duplicate cohorts (e.g., NHPI + Asian -> Asian/PI)
    distribution = expanded_df.groupby(["age", "sex", "race"], as_index=False).agg(
        {"proportion": "sum"}
    )

    # Normalize to ensure proportions sum to 1.0
    total_proportion = distribution["proportion"].sum()
    if abs(total_proportion - 1.0) > 0.01:
        logger.warning(
            f"Distribution proportions sum to {total_proportion:.4f}, normalizing to 1.0"
        )
        distribution["proportion"] = distribution["proportion"] / total_proportion

    # Load config to get expected categories
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    demographics = config.get("demographics", {})
    expected_sexes = demographics.get("sex", ["Male", "Female"])
    expected_races = demographics.get("race_ethnicity", {}).get(
        "categories",
        [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ],
    )
    min_age = demographics.get("age_groups", {}).get("min_age", 0)
    max_age = demographics.get("age_groups", {}).get("max_age", 90)

    # Ensure all expected cohorts exist (fill missing with 0)
    all_cohorts = []
    for age in range(min_age, max_age + 1):
        for sex in expected_sexes:
            for race in expected_races:
                all_cohorts.append({"age": age, "sex": sex, "race": race})

    complete_index = pd.DataFrame(all_cohorts)

    distribution = complete_index.merge(
        distribution,
        on=["age", "sex", "race"],
        how="left",
    )
    distribution["proportion"] = distribution["proportion"].fillna(0.0)

    # Re-normalize after adding zero-proportion cohorts
    total_proportion = distribution["proportion"].sum()
    if total_proportion > 0:
        distribution["proportion"] = distribution["proportion"] / total_proportion

    logger.info(
        f"Created distribution with {len(distribution)} cohorts "
        f"(ages {min_age}-{max_age}, {len(expected_sexes)} sexes, "
        f"{len(expected_races)} races)"
    )

    return distribution


def load_county_populations(
    population_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load county-level total population data.

    Args:
        population_path: Path to county population CSV file
                        (default: data/raw/population/nd_county_population.csv)

    Returns:
        DataFrame with columns:
        - county_fips: 5-digit county FIPS code
        - county_name: County name
        - population: Total population

    Example:
        >>> pops = load_county_populations()
        >>> pops[pops['county_name'] == 'Cass'].head()
           county_fips county_name  population
        ...      38017        Cass      200945
    """
    logger.info("Loading county population data")

    # Set default path
    if population_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        population_path = project_root / "data" / "raw" / "population" / "nd_county_population.csv"

    population_path = Path(population_path)

    if not population_path.exists():
        raise FileNotFoundError(f"County population file not found: {population_path}")

    # Load data
    raw_data = pd.read_csv(population_path, dtype={"county_fips": str})

    # Ensure FIPS is 5 digits
    raw_data["county_fips"] = raw_data["county_fips"].str.zfill(5)

    # Get the most recent population column
    # The file has population_2024 as the most recent
    pop_col = "population_2024"
    if pop_col not in raw_data.columns:
        # Fall back to any column containing 'population' or just 'pop'
        pop_cols = [c for c in raw_data.columns if "pop" in c.lower()]
        if pop_cols:
            pop_col = pop_cols[0]
        else:
            raise ValueError("No population column found in county population file")

    # Create standardized output
    result = pd.DataFrame(
        {
            "county_fips": raw_data["county_fips"],
            "county_name": raw_data["county_name"],
            "population": raw_data[pop_col],
        }
    )

    logger.info(f"Loaded population data for {len(result)} counties")
    logger.info(f"Total state population: {result['population'].sum():,.0f}")

    return result


def load_base_population_for_county(
    fips: str,
    config: dict[str, Any] | None = None,
    distribution: pd.DataFrame | None = None,
    county_populations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load base population for a single county.

    Creates the age x sex x race cohort matrix required by the projection
    engine. Uses the state-level distribution to allocate county total
    population to detailed cohorts.

    Args:
        fips: 5-digit county FIPS code (e.g., "38017" for Cass County)
        config: Optional configuration dictionary
        distribution: Optional pre-loaded state distribution (for efficiency
                     when loading multiple counties)
        county_populations: Optional pre-loaded county populations DataFrame

    Returns:
        DataFrame with columns [year, age, sex, race, population] suitable
        for the CohortComponentProjection engine

    Raises:
        ValueError: If county FIPS not found in population data

    Example:
        >>> cass_pop = load_base_population_for_county("38017")
        >>> cass_pop.head()
           year  age     sex                        race  population
        0  2025    0    Male  White alone, Non-Hispanic       800.5
        1  2025    0    Male  Black alone, Non-Hispanic        25.3
        ...
        >>> cass_pop['population'].sum()
        200945.0
    """
    logger.info(f"Loading base population for county FIPS: {fips}")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    base_year = config.get("project", {}).get("base_year", 2025)

    # Load distribution if not provided
    if distribution is None:
        distribution = load_state_age_sex_race_distribution(config=config)

    # Load county populations if not provided
    if county_populations is None:
        county_populations = load_county_populations()

    # Ensure FIPS is 5 digits
    fips = str(fips).zfill(5)

    # Find this county's population
    county_row = county_populations[county_populations["county_fips"] == fips]

    if county_row.empty:
        raise ValueError(f"County FIPS {fips} not found in population data")

    total_population = county_row["population"].values[0]
    county_name = county_row["county_name"].values[0]

    logger.info(f"County: {county_name} ({fips}), Total population: {total_population:,.0f}")

    # Apply distribution to total population
    base_pop = distribution.copy()
    base_pop["population"] = base_pop["proportion"] * total_population
    base_pop["year"] = base_year

    # Select and reorder columns for projection engine
    base_pop = base_pop[["year", "age", "sex", "race", "population"]]

    # Sort for consistency
    base_pop = base_pop.sort_values(["age", "sex", "race"]).reset_index(drop=True)

    # Validate
    actual_total = base_pop["population"].sum()
    if abs(actual_total - total_population) > 1.0:
        logger.warning(
            f"Population mismatch after distribution: "
            f"expected {total_population:,.0f}, got {actual_total:,.0f}"
        )

    logger.info(f"Created base population with {len(base_pop)} cohorts, total: {actual_total:,.0f}")

    return base_pop


def load_base_population_for_all_counties(
    config: dict[str, Any] | None = None,
    fips_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load base population for all North Dakota counties.

    Returns a dictionary mapping county FIPS codes to their base population
    DataFrames, in the format expected by `run_multi_geography_projections`.

    Args:
        config: Optional configuration dictionary
        fips_list: Optional list of FIPS codes to load (if None, loads all)

    Returns:
        Dictionary mapping county FIPS (str) to base population DataFrame.
        Each DataFrame has columns [year, age, sex, race, population].

    Example:
        >>> all_pops = load_base_population_for_all_counties()
        >>> len(all_pops)
        53
        >>> cass = all_pops["38017"]
        >>> cass['population'].sum()
        200945.0
    """
    logger.info("Loading base population for all counties")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    # Load shared data once for efficiency
    distribution = load_state_age_sex_race_distribution(config=config)
    county_populations = load_county_populations()

    # Get list of counties to process
    if fips_list is None:
        fips_list = county_populations["county_fips"].tolist()

    logger.info(f"Loading base population for {len(fips_list)} counties")

    # Load each county
    base_population_by_geography: dict[str, pd.DataFrame] = {}
    failed_counties: list[str] = []

    for fips in fips_list:
        try:
            base_pop = load_base_population_for_county(
                fips=fips,
                config=config,
                distribution=distribution,
                county_populations=county_populations,
            )
            base_population_by_geography[fips] = base_pop

        except Exception as e:
            logger.warning(f"Failed to load county {fips}: {e}")
            failed_counties.append(fips)

    logger.info(
        f"Successfully loaded {len(base_population_by_geography)} counties, "
        f"{len(failed_counties)} failed"
    )

    if failed_counties:
        logger.warning(f"Failed counties: {failed_counties}")

    # Calculate and log total population
    total_pop = sum(df["population"].sum() for df in base_population_by_geography.values())
    logger.info(f"Total state population across all counties: {total_pop:,.0f}")

    return base_population_by_geography


def load_base_population_for_state(
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load base population for North Dakota state total.

    Aggregates all county populations or uses direct state-level data.

    Args:
        config: Optional configuration dictionary

    Returns:
        DataFrame with columns [year, age, sex, race, population]
        representing the state total.

    Example:
        >>> state_pop = load_base_population_for_state()
        >>> state_pop['population'].sum()
        779094.0  # Total ND population
    """
    logger.info("Loading base population for state")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    # Load all county populations and aggregate
    all_counties = load_base_population_for_all_counties(config=config)

    if not all_counties:
        raise ValueError("No county data available to aggregate to state level")

    # Combine all counties
    all_dfs = list(all_counties.values())
    combined = pd.concat(all_dfs, ignore_index=True)

    # Aggregate by cohort
    state_pop = combined.groupby(["year", "age", "sex", "race"], as_index=False).agg(
        {"population": "sum"}
    )

    # Sort for consistency
    state_pop = state_pop.sort_values(["age", "sex", "race"]).reset_index(drop=True)

    logger.info(
        f"Created state base population: {len(state_pop)} cohorts, "
        f"total: {state_pop['population'].sum():,.0f}"
    )

    return state_pop


if __name__ == "__main__":
    """Example usage and testing."""

    logger.info("Base population loader module test")
    logger.info("=" * 70)

    # Test loading state distribution
    logger.info("\n1. Testing state-level distribution loading...")
    try:
        dist = load_state_age_sex_race_distribution()
        logger.info(f"   Loaded distribution with {len(dist)} cohorts")
        logger.info(f"   Proportion sum: {dist['proportion'].sum():.6f}")
        logger.info(f"   Sample:\n{dist.head()}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading county populations
    logger.info("\n2. Testing county population loading...")
    try:
        pops = load_county_populations()
        logger.info(f"   Loaded {len(pops)} counties")
        logger.info(f"   Total population: {pops['population'].sum():,.0f}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading single county
    logger.info("\n3. Testing single county base population...")
    try:
        # Cass County (Fargo)
        cass = load_base_population_for_county("38017")
        logger.info(f"   Cass County cohorts: {len(cass)}")
        logger.info(f"   Cass County population: {cass['population'].sum():,.0f}")
        logger.info(f"   Columns: {list(cass.columns)}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading all counties
    logger.info("\n4. Testing all counties loading...")
    try:
        all_counties = load_base_population_for_all_counties()
        logger.info(f"   Loaded {len(all_counties)} counties")
        total = sum(df["population"].sum() for df in all_counties.values())
        logger.info(f"   Total population: {total:,.0f}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("Base population loader tests complete")
