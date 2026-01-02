"""
Base population processor for cohort projections.

Processes raw Census data (PEP/ACS) into cohort matrices for projection.
Creates age x sex x race/ethnicity cohort matrices at state, county, and place levels.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils.config_loader import ConfigLoader
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


# Census race/ethnicity code mappings to 6-category system
RACE_ETHNICITY_MAP = {
    # White alone, Non-Hispanic
    "WA_NH": "White alone, Non-Hispanic",
    "NH_WHITE": "White alone, Non-Hispanic",
    "NHWA": "White alone, Non-Hispanic",
    # Black alone, Non-Hispanic
    "BA_NH": "Black alone, Non-Hispanic",
    "NH_BLACK": "Black alone, Non-Hispanic",
    "NHBA": "Black alone, Non-Hispanic",
    # AIAN alone, Non-Hispanic
    "IA_NH": "AIAN alone, Non-Hispanic",
    "NH_AIAN": "AIAN alone, Non-Hispanic",
    "NHIA": "AIAN alone, Non-Hispanic",
    # Asian/PI alone, Non-Hispanic
    "AA_NH": "Asian/PI alone, Non-Hispanic",
    "NH_ASIAN": "Asian/PI alone, Non-Hispanic",
    "NHAA": "Asian/PI alone, Non-Hispanic",
    "NH_API": "Asian/PI alone, Non-Hispanic",
    # Two or more races, Non-Hispanic
    "TOM_NH": "Two or more races, Non-Hispanic",
    "NH_TOM": "Two or more races, Non-Hispanic",
    "NHTOM": "Two or more races, Non-Hispanic",
    "NH_TWO_OR_MORE": "Two or more races, Non-Hispanic",
    # Hispanic (any race)
    "H": "Hispanic (any race)",
    "HISP": "Hispanic (any race)",
    "HISPANIC": "Hispanic (any race)",
}


def harmonize_race_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Census race codes to 6-category system.

    Standardizes race/ethnicity categories from various Census data sources
    to the 6-category system used in projections.

    Args:
        df: DataFrame with race/ethnicity column

    Returns:
        DataFrame with harmonized 'race_ethnicity' column

    Raises:
        ValueError: If race column not found or contains unmapped categories
    """
    logger.info("Harmonizing race/ethnicity categories")

    # Try to find the race column
    race_col = None
    for col in ["race_ethnicity", "RACE_ETHNICITY", "race", "RACE", "ORIGIN"]:
        if col in df.columns:
            race_col = col
            break

    if race_col is None:
        raise ValueError(
            "No race/ethnicity column found. Expected one of: "
            "race_ethnicity, RACE_ETHNICITY, race, RACE, ORIGIN"
        )

    # Create a copy to avoid modifying original
    df = df.copy()

    # Map race categories
    df["race_ethnicity"] = df[race_col].map(RACE_ETHNICITY_MAP)

    # Check for unmapped categories
    unmapped = df[df["race_ethnicity"].isna()][race_col].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped race categories found: {unmapped}")
        logger.warning("These rows will be dropped")
        df = df.dropna(subset=["race_ethnicity"])

    # Drop original race column if different
    if race_col != "race_ethnicity" and race_col in df.columns:
        df = df.drop(columns=[race_col])

    logger.info(f"Harmonized {len(df)} records across {df['race_ethnicity'].nunique()} categories")

    return df


def create_cohort_matrix(
    df: pd.DataFrame, geography_level: str, geography_id: str | None = None
) -> pd.DataFrame:
    """
    Create age x sex x race cohort matrix.

    Transforms population data into a structured cohort matrix with all
    demographic dimensions.

    Args:
        df: DataFrame with columns: age, sex, race_ethnicity, population
        geography_level: One of 'state', 'county', 'place'
        geography_id: Geographic identifier (FIPS code or place code)

    Returns:
        DataFrame with cohort matrix indexed by age, sex, race_ethnicity

    Raises:
        ValueError: If required columns missing
    """
    logger.info(
        f"Creating cohort matrix for {geography_level} {geography_id if geography_id else ''}"
    )

    # Validate required columns
    required_cols = ["age", "sex", "race_ethnicity", "population"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Load configuration for demographic categories
    config = ConfigLoader()
    demographics = config.get_parameter("demographics")

    # Get expected categories
    expected_sexes = demographics["sex"]
    expected_races = list(demographics["race_ethnicity"]["categories"])
    min_age = demographics["age_groups"]["min_age"]
    max_age = demographics["age_groups"]["max_age"]

    # Standardize sex values
    df = df.copy()
    df["sex"] = df["sex"].str.title()  # Male, Female

    # Ensure age is within bounds (cap at max_age for 90+)
    df["age"] = df["age"].clip(upper=max_age)

    # Group by cohort dimensions
    cohort_matrix: pd.DataFrame = df.groupby(["age", "sex", "race_ethnicity"], as_index=False).agg(
        {"population": "sum"}
    )

    # Create complete index (all combinations)
    ages = list(range(min_age, max_age + 1))

    complete_index = pd.MultiIndex.from_product(
        [ages, expected_sexes, expected_races], names=["age", "sex", "race_ethnicity"]
    )

    # Set index and reindex to include all combinations
    cohort_matrix = cohort_matrix.set_index(["age", "sex", "race_ethnicity"])
    cohort_matrix = cohort_matrix.reindex(complete_index, fill_value=0)
    cohort_matrix = cohort_matrix.reset_index()

    # Add geography identifiers
    cohort_matrix["geography_level"] = geography_level
    if geography_id:
        cohort_matrix["geography_id"] = geography_id

    # Add metadata columns
    cohort_matrix["base_year"] = config.get_parameter("project", "base_year")
    cohort_matrix["processing_date"] = datetime.now(UTC).strftime("%Y-%m-%d")

    logger.info(
        f"Created cohort matrix with {len(cohort_matrix)} cells, "
        f"total population: {cohort_matrix['population'].sum():,.0f}"
    )

    return cohort_matrix


def validate_cohort_matrix(
    df: pd.DataFrame, geography_level: str, expected_counties: int | None = None
) -> dict[str, Any]:
    """
    Validate cohort matrix for completeness and plausibility.

    Args:
        df: Cohort matrix DataFrame
        geography_level: Geographic level being validated
        expected_counties: Number of counties expected (for county data)

    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating cohort matrix for {geography_level}")

    errors: list[str] = []
    warnings: list[str] = []
    validation_results: dict[str, Any] = {"valid": True, "warnings": warnings, "errors": errors}

    # Load config for validation thresholds
    config = ConfigLoader()
    demographics = config.get_parameter("demographics")

    # Check for missing cohorts (all should exist, even if 0)
    expected_ages = list(
        range(demographics["age_groups"]["min_age"], demographics["age_groups"]["max_age"] + 1)
    )
    expected_sexes = demographics["sex"]
    expected_races = demographics["race_ethnicity"]["categories"]

    total_cohorts = len(expected_ages) * len(expected_sexes) * len(expected_races)

    if geography_level == "state":
        if len(df) != total_cohorts:
            errors.append(f"Expected {total_cohorts} cohorts, found {len(df)}")
            validation_results["valid"] = False

    elif geography_level == "county":
        # Check if all counties are present
        if "geography_id" in df.columns:
            unique_counties = df["geography_id"].nunique()
            if expected_counties and unique_counties != expected_counties:
                warnings.append(f"Expected {expected_counties} counties, found {unique_counties}")

        # Each county should have full cohort matrix
        if "geography_id" in df.columns:
            county_sizes = df.groupby("geography_id").size()
            incomplete = county_sizes[county_sizes != total_cohorts]
            if len(incomplete) > 0:
                errors.append(f"Incomplete cohort matrices for {len(incomplete)} counties")
                validation_results["valid"] = False

    # Check for negative populations
    if (df["population"] < 0).any():
        errors.append("Negative population values found")
        validation_results["valid"] = False

    # Check for extremely high sex ratios (warning only)
    for race in expected_races:
        for age in expected_ages:
            age_race = df[(df["age"] == age) & (df["race_ethnicity"] == race)]
            if len(age_race) > 0:
                male_pop = age_race[age_race["sex"] == "Male"]["population"].sum()
                female_pop = age_race[age_race["sex"] == "Female"]["population"].sum()

                if female_pop > 0:
                    sex_ratio = male_pop / female_pop
                    if sex_ratio > 2.0 or sex_ratio < 0.5:
                        warnings.append(f"Unusual sex ratio at age {age}, {race}: {sex_ratio:.2f}")

    # Check total population is reasonable
    total_pop = df["population"].sum()
    if total_pop == 0:
        errors.append("Total population is zero")
        validation_results["valid"] = False

    logger.info(f"Validation complete: {len(errors)} errors, {len(warnings)} warnings")

    return validation_results


def process_state_population(
    raw_data: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Process state-level base population.

    Converts raw Census data into cohort matrix for North Dakota state total.

    Args:
        raw_data: Raw Census DataFrame with age, sex, race, population columns
        output_dir: Directory to save processed data (default: data/processed/base_population)

    Returns:
        Processed cohort matrix DataFrame

    Raises:
        ValueError: If validation fails
    """
    logger.info("Processing state-level base population")

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "base_population"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Harmonize race categories
    df = harmonize_race_categories(raw_data)

    # Create cohort matrix
    cohort_matrix = create_cohort_matrix(df, geography_level="state", geography_id="38")

    # Validate
    validation = validate_cohort_matrix(cohort_matrix, geography_level="state")

    if not validation["valid"]:
        error_msg = "State population validation failed: " + "; ".join(validation["errors"])
        logger.error(error_msg)
        raise ValueError(error_msg)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(f"State population: {warning}")

    # Save to parquet
    config = ConfigLoader()
    compression = config.get_parameter("output", "compression", default="gzip")

    output_file = output_dir / "state_base_population.parquet"
    cohort_matrix.to_parquet(output_file, compression=compression, index=False)

    logger.info(f"State base population saved to {output_file}")
    logger.info(f"Total state population: {cohort_matrix['population'].sum():,.0f}")

    return cohort_matrix


def process_county_population(
    raw_data: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Process county-level base population for all 53 North Dakota counties.

    Converts raw Census data into cohort matrices for each county.

    Args:
        raw_data: Raw Census DataFrame with county, age, sex, race, population
        output_dir: Directory to save processed data

    Returns:
        Processed cohort matrix DataFrame for all counties

    Raises:
        ValueError: If validation fails or counties missing
    """
    logger.info("Processing county-level base population")

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "base_population"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate county column exists
    county_col = None
    for col in ["county", "COUNTY", "county_fips", "COUNTYFP", "geo_id"]:
        if col in raw_data.columns:
            county_col = col
            break

    if county_col is None:
        raise ValueError(
            "No county identifier column found. Expected one of: "
            "county, COUNTY, county_fips, COUNTYFP, geo_id"
        )

    # Harmonize race categories
    df = harmonize_race_categories(raw_data)

    # Process each county
    all_counties = []
    unique_counties = df[county_col].unique()

    logger.info(f"Processing {len(unique_counties)} counties")

    for county_id in sorted(unique_counties):
        county_data = df[df[county_col] == county_id].copy()

        # Create cohort matrix for this county
        county_matrix = create_cohort_matrix(
            county_data, geography_level="county", geography_id=str(county_id)
        )

        all_counties.append(county_matrix)

    # Combine all counties
    cohort_matrix = pd.concat(all_counties, ignore_index=True)

    # Validate
    validation = validate_cohort_matrix(
        cohort_matrix, geography_level="county", expected_counties=53
    )

    if not validation["valid"]:
        error_msg = "County population validation failed: " + "; ".join(validation["errors"])
        logger.error(error_msg)
        raise ValueError(error_msg)

    if validation["warnings"]:
        for warning in validation["warnings"][:10]:  # Limit warning output
            logger.warning(f"County population: {warning}")
        if len(validation["warnings"]) > 10:
            logger.warning(f"... and {len(validation['warnings']) - 10} more warnings")

    # Save to parquet
    config = ConfigLoader()
    compression = config.get_parameter("output", "compression", default="gzip")

    output_file = output_dir / "county_base_population.parquet"
    cohort_matrix.to_parquet(output_file, compression=compression, index=False)

    logger.info(f"County base population saved to {output_file}")
    logger.info(f"Total county population: {cohort_matrix['population'].sum():,.0f}")
    logger.info(f"Counties processed: {cohort_matrix['geography_id'].nunique()}")

    return cohort_matrix


def process_place_population(
    raw_data: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Process place-level base population (cities/towns).

    Converts raw Census data into cohort matrices for incorporated places.

    Args:
        raw_data: Raw Census DataFrame with place, age, sex, race, population
        output_dir: Directory to save processed data

    Returns:
        Processed cohort matrix DataFrame for all places

    Raises:
        ValueError: If validation fails
    """
    logger.info("Processing place-level base population")

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "base_population"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate place column exists
    place_col = None
    for col in ["place", "PLACE", "place_fips", "PLACEFP", "geo_id"]:
        if col in raw_data.columns:
            place_col = col
            break

    if place_col is None:
        raise ValueError(
            "No place identifier column found. Expected one of: "
            "place, PLACE, place_fips, PLACEFP, geo_id"
        )

    # Harmonize race categories
    df = harmonize_race_categories(raw_data)

    # Process each place
    all_places = []
    unique_places = df[place_col].unique()

    logger.info(f"Processing {len(unique_places)} places")

    for place_id in sorted(unique_places):
        place_data = df[df[place_col] == place_id].copy()

        # Create cohort matrix for this place
        place_matrix = create_cohort_matrix(
            place_data, geography_level="place", geography_id=str(place_id)
        )

        all_places.append(place_matrix)

    # Combine all places
    cohort_matrix = pd.concat(all_places, ignore_index=True)

    # Validate (less strict for places)
    validation = validate_cohort_matrix(cohort_matrix, geography_level="place")

    if not validation["valid"]:
        error_msg = "Place population validation failed: " + "; ".join(validation["errors"])
        logger.error(error_msg)
        raise ValueError(error_msg)

    if validation["warnings"]:
        for warning in validation["warnings"][:10]:  # Limit warning output
            logger.warning(f"Place population: {warning}")
        if len(validation["warnings"]) > 10:
            logger.warning(f"... and {len(validation['warnings']) - 10} more warnings")

    # Save to parquet
    config = ConfigLoader()
    compression = config.get_parameter("output", "compression", default="gzip")

    output_file = output_dir / "place_base_population.parquet"
    cohort_matrix.to_parquet(output_file, compression=compression, index=False)

    logger.info(f"Place base population saved to {output_file}")
    logger.info(f"Total place population: {cohort_matrix['population'].sum():,.0f}")
    logger.info(f"Places processed: {cohort_matrix['geography_id'].nunique()}")

    return cohort_matrix


def get_cohort_summary(cohort_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for cohort matrix.

    Args:
        cohort_matrix: Processed cohort matrix

    Returns:
        DataFrame with summary statistics by demographic group
    """
    summary_stats = []

    # Overall total
    summary_stats.append(
        {"category": "Total", "group": "All", "population": cohort_matrix["population"].sum()}
    )

    # By sex
    for sex in cohort_matrix["sex"].unique():
        pop = cohort_matrix[cohort_matrix["sex"] == sex]["population"].sum()
        summary_stats.append({"category": "Sex", "group": sex, "population": pop})

    # By race/ethnicity
    for race in cohort_matrix["race_ethnicity"].unique():
        pop = cohort_matrix[cohort_matrix["race_ethnicity"] == race]["population"].sum()
        summary_stats.append({"category": "Race/Ethnicity", "group": race, "population": pop})

    # By age groups
    age_groups = [("0-17", 0, 17), ("18-64", 18, 64), ("65+", 65, 90)]

    for group_name, min_age, max_age in age_groups:
        pop = cohort_matrix[(cohort_matrix["age"] >= min_age) & (cohort_matrix["age"] <= max_age)][
            "population"
        ].sum()
        summary_stats.append({"category": "Age Group", "group": group_name, "population": pop})

    summary_df = pd.DataFrame(summary_stats)
    summary_df["percentage"] = summary_df["population"] / cohort_matrix["population"].sum() * 100

    return summary_df


if __name__ == "__main__":
    """
    Example usage and testing.

    To test with real data, prepare DataFrames with columns:
    - For state: age, sex, race_ethnicity, population
    - For county: county, age, sex, race_ethnicity, population
    - For place: place, age, sex, race_ethnicity, population
    """

    # This is a placeholder for demonstration
    # In practice, data would come from Census API or files

    logger.info("Base population processor loaded successfully")
    logger.info("Ready to process Census data into cohort matrices")
