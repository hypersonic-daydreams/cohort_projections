"""
Migration rates processor for cohort projections.

Processes raw IRS county-to-county migration flows and Census/ACS international
migration data into age-specific, sex-specific, race-specific net migration
rates or counts needed by the projection engine.

Migration is the most complex component because raw data is aggregate (no age/sex/race
breakdown) and requires distribution algorithms to allocate to demographic cohorts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.utils.config_loader import load_projection_config
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


# SEER/Census race/ethnicity code mappings (consistent with fertility/survival processors)
MIGRATION_RACE_MAP = {
    # White alone, Non-Hispanic
    "White Non-Hispanic": "White alone, Non-Hispanic",
    "White NH": "White alone, Non-Hispanic",
    "NH White": "White alone, Non-Hispanic",
    "Non-Hispanic White": "White alone, Non-Hispanic",
    "WNH": "White alone, Non-Hispanic",
    "1": "White alone, Non-Hispanic",
    # Black alone, Non-Hispanic
    "Black Non-Hispanic": "Black alone, Non-Hispanic",
    "Black NH": "Black alone, Non-Hispanic",
    "NH Black": "Black alone, Non-Hispanic",
    "Non-Hispanic Black": "Black alone, Non-Hispanic",
    "BNH": "Black alone, Non-Hispanic",
    "2": "Black alone, Non-Hispanic",
    # AIAN alone, Non-Hispanic
    "AIAN Non-Hispanic": "AIAN alone, Non-Hispanic",
    "AIAN NH": "AIAN alone, Non-Hispanic",
    "NH AIAN": "AIAN alone, Non-Hispanic",
    "American Indian/Alaska Native Non-Hispanic": "AIAN alone, Non-Hispanic",
    "AI/AN Non-Hispanic": "AIAN alone, Non-Hispanic",
    "3": "AIAN alone, Non-Hispanic",
    # Asian/PI alone, Non-Hispanic
    "Asian/PI Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "Asian/Pacific Islander Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "Asian NH": "Asian/PI alone, Non-Hispanic",
    "NH Asian": "Asian/PI alone, Non-Hispanic",
    "API Non-Hispanic": "Asian/PI alone, Non-Hispanic",
    "4": "Asian/PI alone, Non-Hispanic",
    # Two or more races, Non-Hispanic
    "Two or More Races Non-Hispanic": "Two or more races, Non-Hispanic",
    "Two+ Races NH": "Two or more races, Non-Hispanic",
    "NH Two or More Races": "Two or more races, Non-Hispanic",
    "Multiracial Non-Hispanic": "Two or more races, Non-Hispanic",
    "5": "Two or more races, Non-Hispanic",
    # Hispanic (any race)
    "Hispanic": "Hispanic (any race)",
    "Hispanic (any race)": "Hispanic (any race)",
    "All Hispanic": "Hispanic (any race)",
    "Hisp": "Hispanic (any race)",
    "6": "Hispanic (any race)",
}


def load_irs_migration_data(
    file_path: str | Path,
    year_range: tuple[int, int] | None = None,
    target_county_fips: str | None = None,
) -> pd.DataFrame:
    """
    Load IRS county-to-county migration flows.

    IRS provides aggregate migration counts with no age/sex/race breakdown.
    These flows must be distributed to demographic cohorts using standard
    age patterns.

    Args:
        file_path: Path to IRS migration data file (CSV, Excel, Parquet, TXT)
        year_range: Optional tuple of (min_year, max_year) to filter data
        target_county_fips: Optional county FIPS to filter as destination
                           (e.g., '38' for North Dakota state-level)

    Returns:
        DataFrame with columns:
        - from_county_fips: Origin county FIPS code
        - to_county_fips: Destination county FIPS code
        - migrants: Number of migrants (aggregate, no demographics)
        - year: Year of migration data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported or required columns missing

    Example:
        >>> df = load_irs_migration_data(
        ...     'data/raw/migration/irs_county_flows_2018_2022.csv',
        ...     year_range=(2018, 2022),
        ...     target_county_fips='38'
        ... )
        >>> df.columns
        Index(['from_county_fips', 'to_county_fips', 'migrants', 'year'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"IRS migration data file not found: {file_path}")

    logger.info(f"Loading IRS migration data from {file_path}")

    # Determine file format and load
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".txt":
            # IRS files are often tab-delimited
            df = pd.read_csv(file_path, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .csv, .txt, .xlsx, .xls, .parquet"
            )
    except Exception as e:
        raise ValueError(f"Error reading IRS migration file: {e}") from e

    logger.info(f"Loaded {len(df)} records from {file_path.name}")

    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()

    # Find required columns with flexible naming
    from_col = None
    for col in ["from_county_fips", "from_fips", "origin_fips", "from_county", "origin"]:
        if col in df.columns:
            from_col = col
            break

    to_col = None
    for col in ["to_county_fips", "to_fips", "dest_fips", "to_county", "destination"]:
        if col in df.columns:
            to_col = col
            break

    migrants_col = None
    for col in ["migrants", "migration", "count", "returns", "n"]:
        if col in df.columns:
            migrants_col = col
            break

    if from_col is None:
        raise ValueError("No 'from' county column found in IRS data")
    if to_col is None:
        raise ValueError("No 'to' county column found in IRS data")
    if migrants_col is None:
        raise ValueError("No migrants count column found in IRS data")

    # Standardize column names
    df["from_county_fips"] = df[from_col].astype(str).str.strip()
    df["to_county_fips"] = df[to_col].astype(str).str.strip()
    df["migrants"] = pd.to_numeric(df[migrants_col], errors="coerce")

    # Filter to year range if provided
    if year_range is not None:
        if "year" not in df.columns:
            logger.warning("No 'year' column found, cannot filter by year range")
        else:
            min_year, max_year = year_range
            original_len = len(df)
            df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].copy()
            logger.info(
                f"Filtered to years {min_year}-{max_year}: "
                f"{len(df)}/{original_len} records retained"
            )

    # Filter to target county/state if provided
    if target_county_fips is not None:
        original_len = len(df)
        # Include flows TO or FROM target area
        df = df[
            (df["to_county_fips"].str.startswith(target_county_fips))
            | (df["from_county_fips"].str.startswith(target_county_fips))
        ].copy()
        logger.info(
            f"Filtered to county/state {target_county_fips}: "
            f"{len(df)}/{original_len} records retained"
        )

    # Remove NaN migrants
    df = df.dropna(subset=["migrants"])

    logger.info(f"Successfully loaded IRS migration data with {len(df)} flows")

    return df


def load_international_migration_data(
    file_path: str | Path,
    year_range: tuple[int, int] | None = None,
    target_county_fips: str | None = None,
) -> pd.DataFrame:
    """
    Load Census/ACS international migration estimates.

    International migration is typically reported as net flows by county or state,
    with no age/sex/race breakdown. Must be distributed to cohorts.

    Args:
        file_path: Path to international migration data file
        year_range: Optional tuple of (min_year, max_year) to filter data
        target_county_fips: Optional county FIPS to filter

    Returns:
        DataFrame with columns:
        - county_fips: County or state FIPS code
        - international_migrants: Net international migration (can be negative)
        - year: Year of estimate

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported

    Example:
        >>> df = load_international_migration_data(
        ...     'data/raw/migration/international_2018_2022.csv',
        ...     year_range=(2018, 2022),
        ...     target_county_fips='38'
        ... )
        >>> df.columns
        Index(['county_fips', 'international_migrants', 'year'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"International migration file not found: {file_path}")

    logger.info(f"Loading international migration data from {file_path}")

    # Load file
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .csv, .txt, .xlsx, .xls, .parquet"
            )
    except Exception as e:
        raise ValueError(f"Error reading international migration file: {e}") from e

    logger.info(f"Loaded {len(df)} records from {file_path.name}")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Find county column
    county_col = None
    for col in ["county_fips", "fips", "geoid", "county", "geography"]:
        if col in df.columns:
            county_col = col
            break

    if county_col is None:
        raise ValueError("No county/geography column found")

    # Find migrants column
    migrants_col = None
    for col in ["international_migrants", "net_international", "intl_migration", "migrants"]:
        if col in df.columns:
            migrants_col = col
            break

    if migrants_col is None:
        raise ValueError("No international migrants column found")

    # Standardize
    df["county_fips"] = df[county_col].astype(str).str.strip()
    df["international_migrants"] = pd.to_numeric(df[migrants_col], errors="coerce")

    # Filter by year range if provided
    if year_range is not None and "year" in df.columns:
        min_year, max_year = year_range
        original_len = len(df)
        df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].copy()
        logger.info(
            f"Filtered to years {min_year}-{max_year}: "
            f"{len(df)}/{original_len} records retained"
        )

    # Filter to target area if provided
    if target_county_fips is not None:
        original_len = len(df)
        df = df[df["county_fips"].str.startswith(target_county_fips)].copy()
        logger.info(
            f"Filtered to area {target_county_fips}: " f"{len(df)}/{original_len} records retained"
        )

    # Remove NaN migrants
    df = df.dropna(subset=["international_migrants"])

    logger.info(f"Successfully loaded international migration data with {len(df)} records")

    return df


def get_standard_age_migration_pattern(
    peak_age: int = 25, method: str = "simplified"
) -> pd.DataFrame:
    """
    Get standard migration age pattern.

    Migration propensity varies systematically by age, with young adults
    (ages 20-35) being most mobile. This function returns a standard
    age pattern for distributing aggregate migration to ages.

    Args:
        peak_age: Age at which migration peaks (default: 25)
        method: Pattern method - 'simplified' or 'rogers_castro'

    Returns:
        DataFrame with columns:
        - age: int (0-90)
        - migration_propensity: float (relative migration rate, sums to 1.0)

    Notes:
        - Simplified method uses age-group multipliers (easier to understand)
        - Rogers-Castro method uses demographic model (more accurate)
        - Propensities sum to 1.0 for easy distribution of aggregate totals

    Example:
        >>> pattern = get_standard_age_migration_pattern(peak_age=25, method='simplified')
        >>> pattern.head()
           age  migration_propensity
        0    0              0.002
        1    1              0.002
        2    2              0.002
        3    3              0.002
        4    4              0.002
    """
    logger.info(f"Generating standard age migration pattern (method: {method}, peak: {peak_age})")

    ages = list(range(91))  # 0-90

    if method == "simplified":
        # Simplified age pattern based on demographic knowledge
        propensities = []
        for age in ages:
            if age < 10:
                # Young children migrate with parents
                prop = 0.3
            elif age < 18:
                # Teenagers
                prop = 0.5
            elif age < 20:
                # Late teens (leaving home)
                prop = 0.8
            elif age < 30:
                # Peak migration ages (college, career, family formation)
                prop = 1.0
            elif age < 40:
                # Still quite mobile
                prop = 0.75
            elif age < 50:
                # Less mobile, established careers/families
                prop = 0.45
            elif age < 65:
                # Settled, lower mobility
                prop = 0.25
            elif age < 75:
                # Retirement, some mobility
                prop = 0.20
            else:
                # Oldest ages, least mobile
                prop = 0.10

            propensities.append(prop)

    elif method == "rogers_castro":
        # Rogers-Castro model - peaked migration profile
        # Based on standard demographic migration model
        propensities = []

        # Parameters for standard Rogers-Castro model
        a1 = 0.02  # Childhood migration (with parents)
        alpha1 = 0.08  # Rate of decrease for childhood

        a2 = 0.06  # Young adult peak
        mu2 = peak_age  # Age at peak migration
        alpha2 = 0.5  # Rate of decrease from peak
        lambda2 = 0.4  # Shape of peak

        c = 0.001  # Constant (baseline mobility)

        for age in ages:
            # Childhood component (decreasing)
            comp1 = a1 * np.exp(-alpha1 * age)

            # Labor force mobility peak
            comp2 = a2 * np.exp(-alpha2 * (age - mu2) - np.exp(-lambda2 * (age - mu2)))

            # Baseline constant
            comp3 = c

            propensity = comp1 + comp2 + comp3
            propensities.append(max(0, propensity))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'simplified' or 'rogers_castro'")

    # Create DataFrame
    pattern_df = pd.DataFrame({"age": ages, "migration_propensity": propensities})

    # Normalize to sum to 1.0
    total = pattern_df["migration_propensity"].sum()
    pattern_df["migration_propensity"] = pattern_df["migration_propensity"] / total

    logger.info(
        f"Generated age pattern with peak at age {peak_age}, "
        f"sum of propensities: {pattern_df['migration_propensity'].sum():.4f}"
    )

    return pattern_df


def distribute_migration_by_age(total_migration: float, age_pattern: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute aggregate migration to ages using pattern.

    Takes total migration count (aggregate, no age detail) and distributes
    to single-year ages using a standard age pattern.

    Args:
        total_migration: Total net migration (can be positive or negative)
        age_pattern: DataFrame with 'age' and 'migration_propensity' columns

    Returns:
        DataFrame with columns:
        - age: int (0-90)
        - migrants: float (migration count by age)

    Notes:
        - Negative total migration distributes negative values (out-migration)
        - Propensities should sum to 1.0 for accurate distribution

    Example:
        >>> age_pattern = get_standard_age_migration_pattern()
        >>> age_migration = distribute_migration_by_age(1000, age_pattern)
        >>> age_migration.head()
           age  migrants
        0    0      3.2
        1    1      3.2
        2    2      3.2
        3    3      3.2
        4    4      3.2
    """
    logger.debug(f"Distributing {total_migration:,.0f} migrants across ages")

    # Distribute total migration proportional to age pattern
    age_migration = age_pattern.copy()
    age_migration["migrants"] = total_migration * age_migration["migration_propensity"]

    # Drop propensity column (no longer needed)
    age_migration = age_migration[["age", "migrants"]]

    logger.debug(
        f"Distributed migration to {len(age_migration)} ages, "
        f"sum: {age_migration['migrants'].sum():,.1f}"
    )

    return age_migration


def distribute_migration_by_sex(
    age_migration: pd.DataFrame, sex_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Distribute age-specific migration to sex.

    Takes age-specific migration and splits to Male/Female using a sex ratio.
    Default is 50/50 split, but can be adjusted based on data.

    Args:
        age_migration: DataFrame with 'age' and 'migrants' columns
        sex_ratio: Proportion going to males (default: 0.5 for 50/50 split)

    Returns:
        DataFrame with columns:
        - age: int
        - sex: str ('Male' or 'Female')
        - migrants: float

    Notes:
        - sex_ratio = 0.5 means equal split (50% male, 50% female)
        - sex_ratio = 0.52 would mean 52% male, 48% female
        - Works for both positive and negative migration

    Example:
        >>> age_migration = pd.DataFrame({'age': [25, 30], 'migrants': [100, 80]})
        >>> age_sex = distribute_migration_by_sex(age_migration, sex_ratio=0.5)
        >>> age_sex
           age    sex  migrants
        0   25   Male      50.0
        1   25 Female      50.0
        2   30   Male      40.0
        3   30 Female      40.0
    """
    logger.debug(f"Distributing migration by sex (male proportion: {sex_ratio:.2%})")

    records = []

    for _, row in age_migration.iterrows():
        age = row["age"]
        migrants = row["migrants"]

        # Split by sex
        male_migrants = migrants * sex_ratio
        female_migrants = migrants * (1 - sex_ratio)

        records.append({"age": age, "sex": "Male", "migrants": male_migrants})

        records.append({"age": age, "sex": "Female", "migrants": female_migrants})

    result_df = pd.DataFrame(records)

    logger.debug(
        f"Distributed to {len(result_df)} age-sex combinations, "
        f"sum: {result_df['migrants'].sum():,.1f}"
    )

    return result_df


def distribute_migration_by_race(
    age_sex_migration: pd.DataFrame, population_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Distribute to race/ethnicity proportional to population.

    Takes age-sex-specific migration and distributes to race/ethnicity groups
    based on the demographic composition of the population. This assumes
    migration propensity is proportional to population share within each
    age-sex group.

    Args:
        age_sex_migration: DataFrame with 'age', 'sex', 'migrants' columns
        population_df: Base population DataFrame with 'age', 'sex', 'race_ethnicity', 'population'

    Returns:
        DataFrame with columns:
        - age: int
        - sex: str
        - race_ethnicity: str
        - migrants: float

    Notes:
        - Uses population proportions as allocation weights
        - If age-sex group has zero population, distributes equally across races
        - Preserves total migration (sum of distributed = sum of input)

    Example:
        >>> age_sex_mig = pd.DataFrame({
        ...     'age': [25, 25],
        ...     'sex': ['Male', 'Female'],
        ...     'migrants': [50, 50]
        ... })
        >>> pop = pd.DataFrame({
        ...     'age': [25, 25, 25],
        ...     'sex': ['Male', 'Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic', 'Black alone, Non-Hispanic', 'Hispanic (any race)'],
        ...     'population': [700, 200, 100]
        ... })
        >>> distributed = distribute_migration_by_race(age_sex_mig, pop)
        >>> # White gets 70%, Black 20%, Hispanic 10% of 50 male migrants at age 25
    """
    logger.info("Distributing migration by race/ethnicity proportional to population")

    # Ensure population has required columns
    required_cols = ["age", "sex", "race_ethnicity", "population"]
    missing_cols = [col for col in required_cols if col not in population_df.columns]
    if missing_cols:
        raise ValueError(f"population_df missing required columns: {missing_cols}")

    # Calculate population proportions by age-sex-race
    pop_totals = population_df.groupby(["age", "sex"], as_index=False).agg({"population": "sum"})
    pop_totals = pop_totals.rename(columns={"population": "total_population"})

    pop_with_totals = population_df.merge(pop_totals, on=["age", "sex"], how="left")
    pop_with_totals["proportion"] = (
        pop_with_totals["population"] / pop_with_totals["total_population"]
    )

    # Handle zero-population age-sex groups (distribute equally)
    zero_pop_mask = pop_with_totals["total_population"] == 0
    if zero_pop_mask.any():
        logger.warning(
            f"Found {zero_pop_mask.sum()} age-sex-race groups with zero total population, "
            f"distributing equally across races"
        )
        # Equal distribution for zero-population groups
        races_per_group = pop_with_totals.groupby(["age", "sex"]).size()
        pop_with_totals["equal_share"] = pop_with_totals.apply(
            lambda row: 1.0 / races_per_group.get((row["age"], row["sex"]), 1), axis=1
        )
        pop_with_totals.loc[zero_pop_mask, "proportion"] = pop_with_totals.loc[
            zero_pop_mask, "equal_share"
        ]

    # Merge migration with population proportions
    merged = age_sex_migration.merge(
        pop_with_totals[["age", "sex", "race_ethnicity", "proportion"]],
        on=["age", "sex"],
        how="left",
    )

    # Distribute migrants to races
    merged["migrants"] = merged["migrants"] * merged["proportion"]

    # Select final columns
    result_df = merged[["age", "sex", "race_ethnicity", "migrants"]].copy()

    # Remove any NaN (shouldn't happen, but safety check)
    result_df = result_df.dropna(subset=["migrants"])

    total_input = age_sex_migration["migrants"].sum()
    total_output = result_df["migrants"].sum()

    logger.info(
        f"Distributed to {len(result_df)} age-sex-race combinations, "
        f"input sum: {total_input:,.1f}, output sum: {total_output:,.1f}"
    )

    if abs(total_input - total_output) > 1.0:
        logger.warning(
            f"Migration total changed after race distribution: "
            f"input {total_input:,.1f} vs output {total_output:,.1f}"
        )

    return result_df


def calculate_net_migration(
    in_migration: pd.DataFrame, out_migration: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate net migration (in - out) by cohort.

    Takes separate in-migration and out-migration DataFrames and computes
    net migration. Handles cases where cohorts appear in one but not the other.

    Args:
        in_migration: DataFrame with in-migration by age, sex, race
        out_migration: DataFrame with out-migration by age, sex, race

    Returns:
        DataFrame with columns:
        - age: int
        - sex: str
        - race_ethnicity: str
        - net_migration: float (positive = net in-migration, negative = net out-migration)

    Notes:
        - Net migration can be negative (more out-migration than in-migration)
        - Cohorts present in only one DataFrame get net migration equal to that value
        - Preserves demographic detail from input DataFrames

    Example:
        >>> in_mig = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'sex': ['Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 2,
        ...     'migrants': [150, 120]
        ... })
        >>> out_mig = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'sex': ['Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 2,
        ...     'migrants': [100, 130]
        ... })
        >>> net = calculate_net_migration(in_mig, out_mig)
        >>> net['net_migration'].tolist()
        [50, -10]  # net in-migration at 25, net out-migration at 30
    """
    logger.info("Calculating net migration (in - out)")

    # Ensure both have 'migrants' column
    if "migrants" not in in_migration.columns:
        raise ValueError("in_migration must have 'migrants' column")
    if "migrants" not in out_migration.columns:
        raise ValueError("out_migration must have 'migrants' column")

    # Merge on cohort identifiers
    merged = in_migration.merge(
        out_migration, on=["age", "sex", "race_ethnicity"], how="outer", suffixes=("_in", "_out")
    )

    # Fill NaN with 0 (cohort present in only one direction)
    merged["migrants_in"] = merged["migrants_in"].fillna(0.0)
    merged["migrants_out"] = merged["migrants_out"].fillna(0.0)

    # Calculate net migration
    merged["net_migration"] = merged["migrants_in"] - merged["migrants_out"]

    # Select final columns
    result_df = merged[["age", "sex", "race_ethnicity", "net_migration"]].copy()

    total_in = merged["migrants_in"].sum()
    total_out = merged["migrants_out"].sum()
    total_net = result_df["net_migration"].sum()

    logger.info(
        f"Net migration calculated: "
        f"in={total_in:,.0f}, out={total_out:,.0f}, net={total_net:+,.0f}"
    )

    return result_df


def combine_domestic_international_migration(
    domestic_df: pd.DataFrame, international_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine domestic and international migration into total net migration.

    Args:
        domestic_df: Domestic net migration DataFrame (from IRS flows)
        international_df: International net migration DataFrame (from ACS/Census)

    Returns:
        DataFrame with combined net migration by cohort

    Notes:
        - Both components summed for each cohort
        - Handles cases where cohorts present in only one component
        - Result is total net migration (domestic + international)

    Example:
        >>> domestic = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'sex': ['Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 2,
        ...     'net_migration': [-50, 30]  # net out domestically at 25
        ... })
        >>> intl = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'sex': ['Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 2,
        ...     'net_migration': [20, 10]  # net in internationally
        ... })
        >>> combined = combine_domestic_international_migration(domestic, intl)
        >>> combined['net_migration'].tolist()
        [-30, 40]  # combined net
    """
    logger.info("Combining domestic and international migration")

    # Validate inputs
    if "net_migration" not in domestic_df.columns:
        raise ValueError("domestic_df must have 'net_migration' column")
    if "net_migration" not in international_df.columns:
        raise ValueError("international_df must have 'net_migration' column")

    # Merge on cohort identifiers
    combined = domestic_df.merge(
        international_df,
        on=["age", "sex", "race_ethnicity"],
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

    result = combined[["age", "sex", "race_ethnicity", "net_migration"]].copy()

    total_domestic = combined["net_migration_domestic"].sum()
    total_international = combined["net_migration_international"].sum()
    total_combined = result["net_migration"].sum()

    logger.info(
        f"Combined migration - Domestic: {total_domestic:+,.0f}, "
        f"International: {total_international:+,.0f}, "
        f"Total: {total_combined:+,.0f}"
    )

    return result


def create_migration_rate_table(
    df: pd.DataFrame,
    population_df: pd.DataFrame | None = None,
    as_rates: bool = False,
    validate: bool = True,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Create final migration table for projection.

    Ensures all age (0-90) × sex (2) × race (6) combinations are present.
    Can produce absolute migration numbers or rates (relative to population).

    Args:
        df: DataFrame with net_migration by age, sex, race
        population_df: Optional base population (required if as_rates=True)
        as_rates: Whether to express as migration rates (vs absolute numbers)
        validate: Whether to validate the table (default: True)
        config: Optional configuration dictionary

    Returns:
        DataFrame ready for projection engine with columns:
        - age: int (0-90)
        - sex: str ('Male', 'Female')
        - race_ethnicity: str (6 categories)
        - net_migration (if as_rates=False): float
        - migration_rate (if as_rates=True): float

    Raises:
        ValueError: If validation fails and validate=True

    Notes:
        - Total combinations: 91 ages × 2 sexes × 6 races = 1,092 rows
        - Missing cohorts filled with 0
        - Migration rates = net_migration / population
        - Both absolute and rate forms work in projection engine

    Example:
        >>> mig_df = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'sex': ['Male', 'Male'],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 2,
        ...     'net_migration': [50, -20]
        ... })
        >>> table = create_migration_rate_table(mig_df)
        >>> len(table)
        1092  # All cohorts present
    """
    logger.info("Creating migration rate table for projection")

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Get expected categories from config
    demographics = config.get("demographics", {})
    expected_races = demographics.get("race_ethnicity", {}).get("categories", [])
    expected_sexes = demographics.get("sex", ["Male", "Female"])

    # Get age range from config
    age_config = demographics.get("age_groups", {})
    min_age = age_config.get("min_age", 0)
    max_age = age_config.get("max_age", 90)

    # Validate input has required columns
    required_cols = ["age", "sex", "race_ethnicity", "net_migration"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to valid age range
    df = df[(df["age"] >= min_age) & (df["age"] <= max_age)].copy()

    logger.info(f"Filtered to ages {min_age}-{max_age}: {len(df)} records")

    # Create complete index (all age-sex-race combinations)
    ages = list(range(min_age, max_age + 1))

    complete_index = pd.MultiIndex.from_product(
        [ages, expected_sexes, expected_races], names=["age", "sex", "race_ethnicity"]
    )

    # Set index and reindex to include all combinations
    df_indexed = df.set_index(["age", "sex", "race_ethnicity"])

    # Reindex with 0 for missing combinations
    df_complete = df_indexed.reindex(complete_index, fill_value=0)
    df_complete = df_complete.reset_index()

    # Ensure net_migration column exists
    if "net_migration" not in df_complete.columns:
        df_complete["net_migration"] = 0.0

    # Convert to rates if requested
    if as_rates:
        if population_df is None:
            raise ValueError("population_df required to calculate migration rates")

        logger.info("Converting net migration to rates")

        # Merge with population
        merged = df_complete.merge(
            population_df[["age", "sex", "race_ethnicity", "population"]],
            on=["age", "sex", "race_ethnicity"],
            how="left",
        )

        # Calculate rate
        merged["migration_rate"] = np.where(
            merged["population"] > 0, merged["net_migration"] / merged["population"], 0.0
        )

        # Keep only rate column
        df_complete = merged[["age", "sex", "race_ethnicity", "migration_rate"]].copy()

        logger.info(
            f"Converted to migration rates, mean absolute rate: "
            f"{df_complete['migration_rate'].abs().mean():.6f}"
        )

    # Add metadata
    df_complete["processing_date"] = datetime.now(UTC).strftime("%Y-%m-%d")

    logger.info(
        f"Created migration table with {len(df_complete)} cells "
        f"({len(ages)} ages × {len(expected_sexes)} sexes × {len(expected_races)} races)"
    )

    # Validate if requested
    if validate:
        validation_result = validate_migration_data(df_complete, population_df, config)

        if not validation_result["valid"]:
            error_msg = "Migration data validation failed:\n" + "\n".join(
                validation_result["errors"]
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Migration validation: {warning}")

    return df_complete


def validate_migration_data(
    df: pd.DataFrame, population_df: pd.DataFrame | None = None, config: dict | None = None
) -> dict[str, Any]:
    """
    Validate migration data for plausibility.

    Performs comprehensive validation including:
    - All age-sex-race combinations present
    - Age pattern plausible (peak at young adult ages)
    - Migration rates/amounts not extreme
    - Won't cause negative populations

    Args:
        df: Migration DataFrame to validate
        population_df: Optional base population for rate validation
        config: Optional configuration dictionary

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'total_net_migration': float,
            'net_by_direction': Dict[str, float]
        }

    Notes:
        - Checks for implausibly large migration (>20% of population)
        - Validates age pattern (should peak at ages 20-35)
        - Ensures no cohorts would have negative population after migration

    Example:
        >>> mig_df = pd.DataFrame({
        ...     'age': list(range(91)) * 12,  # All ages
        ...     'sex': ['Male'] * 546 + ['Female'] * 546,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 91 * 2,
        ...     'net_migration': [10] * 1092
        ... })
        >>> result = validate_migration_data(mig_df)
        >>> result['valid']
        True
    """
    logger.info("Validating migration data")

    errors: list[str] = []
    warnings: list[str] = []
    net_by_direction: dict[str, Any] = {}
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": errors,
        "warnings": warnings,
        "total_net_migration": 0.0,
        "net_by_direction": net_by_direction,
    }

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Get expected values from config
    demographics = config.get("demographics", {})
    expected_races = demographics.get("race_ethnicity", {}).get("categories", [])
    expected_sexes = demographics.get("sex", ["Male", "Female"])
    age_config = demographics.get("age_groups", {})
    min_age = age_config.get("min_age", 0)
    max_age = age_config.get("max_age", 90)

    # Check for required columns
    has_net_migration = "net_migration" in df.columns
    has_migration_rate = "migration_rate" in df.columns

    if not has_net_migration and not has_migration_rate:
        errors.append("Must have either 'net_migration' or 'migration_rate' column")
        validation_result["valid"] = False
        return validation_result

    migration_col = "net_migration" if has_net_migration else "migration_rate"

    # Check all ages present
    expected_ages = set(range(min_age, max_age + 1))
    actual_ages = set(df["age"].unique())
    missing_ages = expected_ages - actual_ages
    if missing_ages:
        errors.append(f"Missing ages: {sorted(missing_ages)}")
        validation_result["valid"] = False

    # Check all sex categories present
    actual_sexes = set(df["sex"].unique())
    missing_sexes = set(expected_sexes) - actual_sexes
    if missing_sexes:
        errors.append(f"Missing sex categories: {list(missing_sexes)}")
        validation_result["valid"] = False

    # Check all race categories present
    actual_races = set(df["race_ethnicity"].unique())
    missing_races = set(expected_races) - actual_races
    if missing_races:
        errors.append(f"Missing race categories: {list(missing_races)}")
        validation_result["valid"] = False

    # Calculate total net migration
    total_net = df[migration_col].sum()
    validation_result["total_net_migration"] = float(total_net)

    # Count positive vs negative
    positive_count = (df[migration_col] > 0).sum()
    negative_count = (df[migration_col] < 0).sum()
    zero_count = (df[migration_col] == 0).sum()

    validation_result["net_by_direction"] = {
        "net_in_migration": float(df[df[migration_col] > 0][migration_col].sum()),
        "net_out_migration": float(df[df[migration_col] < 0][migration_col].sum()),
        "cohorts_positive": int(positive_count),
        "cohorts_negative": int(negative_count),
        "cohorts_zero": int(zero_count),
    }

    # Check for extreme values
    if has_net_migration:
        max_abs_migration = df["net_migration"].abs().max()
        if max_abs_migration > 10000:
            warnings.append(
                f"Very large net migration value: {max_abs_migration:,.0f} "
                f"(possible data error)"
            )

    if has_migration_rate:
        # Check for extreme rates
        extreme_positive = df[df["migration_rate"] > 0.20]
        if not extreme_positive.empty:
            warnings.append(
                f"Very high migration rates (>20% of population) found in "
                f"{len(extreme_positive)} cohorts"
            )

        extreme_negative = df[df["migration_rate"] < -0.20]
        if not extreme_negative.empty:
            warnings.append(
                f"Very high out-migration rates (<-20% of population) found in "
                f"{len(extreme_negative)} cohorts"
            )

    # Validate age pattern (should peak at young adult ages)
    if has_net_migration:
        # Calculate mean migration by age
        age_pattern = df.groupby("age")[migration_col].mean().abs()

        # Find peak age
        peak_age = int(age_pattern.idxmax())

        if peak_age < 15 or peak_age > 45:
            warnings.append(f"Migration peak at age {peak_age} is unusual (expected: 20-35)")

    # If population provided, check if migration would cause negative population
    if population_df is not None and has_net_migration:
        merged = population_df.merge(df, on=["age", "sex", "race_ethnicity"], how="left")
        merged["net_migration"] = merged["net_migration"].fillna(0.0)
        merged["result_pop"] = merged["population"] + merged["net_migration"]

        negative_results = merged["result_pop"] < 0
        if negative_results.any():
            warnings.append(
                f"Migration would cause negative population for {negative_results.sum()} cohorts"
            )

    # Check for missing combinations
    expected_combinations = len(expected_ages) * len(expected_sexes) * len(expected_races)
    actual_combinations = len(df)

    if actual_combinations < expected_combinations:
        errors.append(
            f"Missing age-sex-race combinations: expected {expected_combinations}, "
            f"got {actual_combinations}"
        )
        validation_result["valid"] = False

    # Summary logging
    if validation_result["valid"]:
        logger.info(
            f"Migration data validated successfully. Total net migration: {total_net:+,.0f}"
        )
    else:
        logger.error(f"Migration data validation failed with {len(errors)} errors")

    if warnings:
        logger.warning(f"Migration data validation produced {len(warnings)} warnings")

    return validation_result


def process_migration_rates(
    irs_path: str | Path,
    intl_path: str | Path | None = None,
    population_path: str | Path | None = None,
    output_dir: Path | None = None,
    config: dict | None = None,
    year_range: tuple[int, int] | None = None,
    target_county_fips: str | None = None,
    as_rates: bool = False,
) -> pd.DataFrame:
    """
    Main processing function for migration rates.

    Complete pipeline: Load → Calculate net → Distribute → Combine → Validate → Save

    Args:
        irs_path: Path to IRS county-to-county migration flows
        intl_path: Optional path to international migration data
        population_path: Path to base population (required for distribution)
        output_dir: Directory to save processed data (default: data/processed/migration)
        config: Optional configuration dictionary
        year_range: Optional (min_year, max_year) tuple to filter data
        target_county_fips: Optional county/state FIPS code (e.g., '38' for ND)
        as_rates: Whether to output migration rates vs absolute numbers

    Returns:
        Processed migration DataFrame

    Raises:
        ValueError: If processing or validation fails

    Notes:
        - IRS data is required (domestic migration)
        - International migration is optional (defaults to 0 if not provided)
        - Population data required for distribution to cohorts
        - Saves three output files: parquet, CSV, and metadata JSON

    Example:
        >>> migration_rates = process_migration_rates(
        ...     irs_path='data/raw/migration/irs_flows_2018_2022.csv',
        ...     intl_path='data/raw/migration/international_2018_2022.csv',
        ...     population_path='data/processed/base_population.parquet',
        ...     year_range=(2018, 2022),
        ...     target_county_fips='38'  # North Dakota
        ... )
        >>> # Output: data/processed/migration/migration_rates.parquet
    """
    logger.info("=" * 70)
    logger.info("Starting migration rates processing pipeline")
    logger.info("=" * 70)

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "migration"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base population (required for distribution)
    logger.info("Step 1: Loading base population for distribution")

    if population_path is None:
        raise ValueError("population_path required to distribute aggregate migration to cohorts")

    population_path = Path(population_path)
    if not population_path.exists():
        raise FileNotFoundError(f"Population file not found: {population_path}")

    if population_path.suffix == ".parquet":
        population_df = pd.read_parquet(population_path)
    elif population_path.suffix == ".csv":
        population_df = pd.read_csv(population_path)
    else:
        raise ValueError(f"Unsupported population file format: {population_path.suffix}")

    logger.info(f"Loaded population with {len(population_df)} cohorts")

    # Step 2: Load IRS migration data
    logger.info("Step 2: Loading IRS county-to-county migration flows")

    irs_df = load_irs_migration_data(
        irs_path, year_range=year_range, target_county_fips=target_county_fips
    )

    # Step 3: Calculate net domestic migration by county
    logger.info("Step 3: Calculating net domestic migration")

    # Sum in-migration (to target)
    in_migration = irs_df[
        irs_df["to_county_fips"].str.startswith(target_county_fips if target_county_fips else "")
    ].copy()
    total_in = in_migration["migrants"].sum()

    # Sum out-migration (from target)
    out_migration = irs_df[
        irs_df["from_county_fips"].str.startswith(target_county_fips if target_county_fips else "")
    ].copy()
    total_out = out_migration["migrants"].sum()

    # Net domestic migration
    net_domestic = total_in - total_out

    logger.info(
        f"Domestic migration: in={total_in:,.0f}, out={total_out:,.0f}, "
        f"net={net_domestic:+,.0f}"
    )

    # Step 4: Load international migration (optional)
    logger.info("Step 4: Loading international migration data")

    if intl_path is not None:
        intl_df = load_international_migration_data(
            intl_path, year_range=year_range, target_county_fips=target_county_fips
        )
        net_international = intl_df["international_migrants"].sum()
        logger.info(f"International migration: {net_international:+,.0f}")
    else:
        net_international = 0.0
        logger.info("No international migration data provided, using 0")

    # Total net migration
    total_net_migration = net_domestic + net_international

    logger.info(f"Total net migration: {total_net_migration:+,.0f}")

    # Step 5: Distribute to ages using standard pattern
    logger.info("Step 5: Distributing migration to age groups")

    age_pattern = get_standard_age_migration_pattern(peak_age=25, method="simplified")
    age_migration = distribute_migration_by_age(total_net_migration, age_pattern)

    # Step 6: Distribute to sex
    logger.info("Step 6: Distributing migration by sex")

    age_sex_migration = distribute_migration_by_sex(age_migration, sex_ratio=0.5)

    # Step 7: Distribute to race/ethnicity
    logger.info("Step 7: Distributing migration by race/ethnicity")

    age_sex_race_migration = distribute_migration_by_race(age_sex_migration, population_df)

    # Rename 'migrants' to 'net_migration' for consistency
    age_sex_race_migration.rename(columns={"migrants": "net_migration"}, inplace=True)

    # Step 8: Create complete migration table
    logger.info("Step 8: Creating migration rate table")

    migration_table = create_migration_rate_table(
        age_sex_race_migration,
        population_df=population_df if as_rates else None,
        as_rates=as_rates,
        validate=True,
        config=config,
    )

    # Step 9: Save processed data
    logger.info("Step 9: Saving processed migration rates")

    # Save as parquet (primary format)
    output_file = output_dir / "migration_rates.parquet"
    compression = config.get("output", {}).get("compression", "gzip")

    migration_table.to_parquet(output_file, compression=compression, index=False)
    logger.info(f"Saved migration rates to {output_file}")

    # Also save as CSV for human readability
    csv_file = output_dir / "migration_rates.csv"
    migration_table.to_csv(csv_file, index=False)
    logger.info(f"Saved migration rates to {csv_file}")

    # Step 10: Generate and save metadata
    logger.info("Step 10: Generating metadata")

    validation_result = validate_migration_data(migration_table, population_df, config)

    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "source_files": {
            "irs_data": str(irs_path),
            "international_data": str(intl_path) if intl_path else None,
            "population_data": str(population_path),
        },
        "year_range": year_range,
        "target_area": target_county_fips,
        "output_format": "migration_rates" if as_rates else "net_migration",
        "total_records": len(migration_table),
        "age_range": [int(migration_table["age"].min()), int(migration_table["age"].max())],
        "sex_categories": list(migration_table["sex"].unique()),
        "race_categories": list(migration_table["race_ethnicity"].unique()),
        "migration_summary": {
            "total_net_migration": float(total_net_migration),
            "net_domestic": float(net_domestic),
            "net_international": float(net_international),
            "in_migration": float(total_in),
            "out_migration": float(total_out),
        },
        "distribution_method": {
            "age_pattern": "simplified",
            "sex_distribution": "50/50",
            "race_distribution": "proportional_to_population",
        },
        "validation_summary": validation_result["net_by_direction"],
        "validation_warnings": validation_result["warnings"],
        "config_used": {
            "ages": config.get("demographics", {}).get("age_groups", {}),
            "race_categories": config.get("demographics", {})
            .get("race_ethnicity", {})
            .get("categories", []),
        },
    }

    metadata_file = output_dir / "migration_rates_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    # Summary statistics
    logger.info("=" * 70)
    logger.info("Processing Complete - Summary Statistics")
    logger.info("=" * 70)
    logger.info(f"Total records: {len(migration_table)}")
    logger.info(f"Age range: {migration_table['age'].min()}-{migration_table['age'].max()}")
    logger.info(f"Sex categories: {migration_table['sex'].nunique()}")
    logger.info(f"Race categories: {migration_table['race_ethnicity'].nunique()}")
    logger.info("\nMigration Summary:")
    logger.info(f"  Total net migration: {total_net_migration:+,.0f}")
    logger.info(f"  Net domestic: {net_domestic:+,.0f}")
    logger.info(f"  Net international: {net_international:+,.0f}")
    logger.info(f"  In-migration: {total_in:,.0f}")
    logger.info(f"  Out-migration: {total_out:,.0f}")
    logger.info("=" * 70)

    return migration_table


if __name__ == "__main__":
    """
    Example usage and testing.

    To test with real data, prepare:
    1. IRS county-to-county flows CSV with: from_county_fips, to_county_fips, migrants, year
    2. International migration CSV with: county_fips, international_migrants, year
    3. Base population parquet/CSV with: age, sex, race_ethnicity, population
    """

    logger.info("Migration rates processor loaded successfully")
    logger.info("Ready to process IRS and Census migration data")

    # Example with synthetic data
    logger.info("\nExample: Creating sample migration data")

    # Create synthetic IRS flows
    np.random.seed(42)

    sample_irs = []
    counties = ["38001", "38003", "38005", "38017", "38035"]  # Sample ND counties

    for year in range(2018, 2023):
        for from_county in counties:
            for to_county in counties:
                if from_county != to_county:
                    migrants = np.random.randint(10, 100)
                    sample_irs.append(
                        {
                            "from_county_fips": from_county,
                            "to_county_fips": to_county,
                            "migrants": migrants,
                            "year": year,
                        }
                    )

    irs_df = pd.DataFrame(sample_irs)
    logger.info(f"Created sample IRS data: {len(irs_df)} flows")

    # Create synthetic international migration
    sample_intl = []
    for year in range(2018, 2023):
        for county in counties:
            intl_migrants = np.random.randint(5, 25)
            sample_intl.append(
                {"county_fips": county, "international_migrants": intl_migrants, "year": year}
            )

    intl_df = pd.DataFrame(sample_intl)
    logger.info(f"Created sample international data: {len(intl_df)} records")

    # Demonstrate age pattern
    logger.info("\nGenerating migration age pattern:")
    age_pattern = get_standard_age_migration_pattern(peak_age=25, method="simplified")
    logger.info(f"Age pattern has {len(age_pattern)} ages")
    logger.info(
        f"Peak migration propensity at age {age_pattern.loc[age_pattern['migration_propensity'].idxmax(), 'age']}"
    )
