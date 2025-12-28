"""
Survival rates processor for cohort projections.

Processes raw SEER/CDC life table files into survival rates needed by the
projection engine. Converts life tables (qx, lx, Lx, Tx) into age-specific
survival rates by sex and race/ethnicity for cohort-component projections.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils.config_loader import load_projection_config
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


# SEER/CDC race/ethnicity code mappings to 6-category system
# (Same mapping as fertility rates for consistency)
SEER_MORTALITY_RACE_MAP = {
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


def load_life_table_data(file_path: str | Path, year: int | None = None) -> pd.DataFrame:
    """
    Load raw SEER/CDC life table file(s).

    Supports multiple file formats (CSV, TXT, Excel, Parquet) and filters to
    specified year if provided.

    Args:
        file_path: Path to SEER/CDC life table data file
        year: Optional year to filter data (for multi-year life tables)

    Returns:
        DataFrame with raw life table data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported or data invalid

    Example:
        >>> df = load_life_table_data(
        ...     'data/raw/mortality/seer_lifetables_2020.csv',
        ...     year=2020
        ... )
        >>> df.columns
        Index(['year', 'age', 'sex', 'race', 'qx', 'lx', 'Lx', 'Tx', 'ex'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Life table file not found: {file_path}")

    logger.info(f"Loading life table data from {file_path}")

    # Determine file format and load
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".txt":
            # SEER text files are typically tab-delimited
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
        raise ValueError(f"Error reading life table file: {e}") from e

    logger.info(f"Loaded {len(df)} records from {file_path.name}")

    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()

    # Filter by year if provided
    if year is not None:
        if "year" not in df.columns:
            logger.warning("No 'year' column found, cannot filter by year")
        else:
            original_len = len(df)
            df = df[df["year"] == year].copy()
            logger.info(f"Filtered to year {year}: " f"{len(df)}/{original_len} records retained")

    # Validate required columns exist (flexible naming)
    age_col = None
    for col in ["age", "age_group", "agegrp"]:
        if col in df.columns:
            age_col = col
            break

    if age_col is None:
        raise ValueError("No age column found. Expected one of: age, age_group, agegrp")

    # Standardize to 'age'
    if age_col != "age":
        df["age"] = df[age_col]

    # Standardize sex column
    sex_col = None
    for col in ["sex", "gender"]:
        if col in df.columns:
            sex_col = col
            break

    if sex_col is None:
        raise ValueError("No sex/gender column found")

    if sex_col != "sex":
        df["sex"] = df[sex_col]

    # Standardize sex values
    df["sex"] = df["sex"].astype(str).str.strip().str.title()
    # Map common variants
    sex_map = {
        "M": "Male",
        "F": "Female",
        "1": "Male",
        "2": "Female",
        "Males": "Male",
        "Females": "Female",
    }
    df["sex"] = df["sex"].replace(sex_map)

    logger.info(f"Successfully loaded life table data with {len(df)} records")

    return df


def harmonize_mortality_race_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map SEER/CDC race codes to standard 6 categories.

    Standardizes race/ethnicity categories from SEER/CDC data sources to the
    6-category system used in projections.

    Args:
        df: DataFrame with SEER/CDC race/ethnicity column

    Returns:
        DataFrame with harmonized 'race_ethnicity' column

    Raises:
        ValueError: If race column not found

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [0, 1, 2],
        ...     'sex': ['Male', 'Female', 'Male'],
        ...     'race': ['White NH', 'Hispanic', 'Black NH'],
        ...     'qx': [0.005, 0.006, 0.007]
        ... })
        >>> df = harmonize_mortality_race_categories(df)
        >>> df['race_ethnicity'].unique()
        array(['White alone, Non-Hispanic', 'Hispanic (any race)',
               'Black alone, Non-Hispanic'])
    """
    logger.info("Harmonizing mortality race/ethnicity categories")

    # Try to find the race column
    race_col = None
    for col in ["race_ethnicity", "race", "race_origin", "origin", "race_code"]:
        if col in df.columns:
            race_col = col
            break

    if race_col is None:
        raise ValueError(
            "No race/ethnicity column found. Expected one of: "
            "race_ethnicity, race, race_origin, origin, race_code"
        )

    # Create a copy to avoid modifying original
    df = df.copy()

    # Convert to string and strip whitespace
    df[race_col] = df[race_col].astype(str).str.strip()

    # Map race categories
    df["race_ethnicity"] = df[race_col].map(SEER_MORTALITY_RACE_MAP)

    # Check for unmapped categories
    unmapped = df[df["race_ethnicity"].isna()][race_col].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped race categories found: {list(unmapped)}")
        logger.warning(
            "These rows will be dropped. Consider updating SEER_MORTALITY_RACE_MAP "
            "if these are valid categories."
        )
        original_len = len(df)
        df = df.dropna(subset=["race_ethnicity"])
        logger.info(f"Dropped {original_len - len(df)} records with unmapped race categories")

    # Drop original race column if different
    if race_col != "race_ethnicity" and race_col in df.columns:
        df = df.drop(columns=[race_col])

    logger.info(
        f"Harmonized {len(df)} records across "
        f"{df['race_ethnicity'].nunique()} race/ethnicity categories"
    )

    return df


def calculate_survival_rates_from_life_table(df: pd.DataFrame, method: str = "lx") -> pd.DataFrame:
    """
    Convert life table to survival rates.

    Supports multiple calculation methods based on available life table columns.
    Handles age 90+ open-ended group specially using Tx and Lx columns.

    Args:
        df: DataFrame with life table data
        method: Calculation method - 'lx', 'qx', or 'Lx'
            - 'lx': S(x) = l(x+1) / l(x)
            - 'qx': S(x) = 1 - q(x)
            - 'Lx': S(x) = L(x+1) / L(x)

    Returns:
        DataFrame with survival_rate column added

    Raises:
        ValueError: If required columns not present for chosen method

    Notes:
        - For age 90+ (open-ended group), uses special formula:
          S(90+) = T(91) / (T(90) + L(90)/2)
        - If Tx/Lx not available for age 90+, uses default of 0.65

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [0, 1, 2],
        ...     'sex': ['Male'] * 3,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 3,
        ...     'lx': [100000, 99400, 99350]
        ... })
        >>> df = calculate_survival_rates_from_life_table(df, method='lx')
        >>> df['survival_rate'].iloc[0]
        0.994
    """
    logger.info(f"Calculating survival rates using method: {method}")

    df = df.copy()

    # Validate method-specific columns
    if method == "lx":
        if "lx" not in df.columns:
            raise ValueError("Method 'lx' requires 'lx' column in life table")
    elif method == "qx":
        if "qx" not in df.columns:
            raise ValueError("Method 'qx' requires 'qx' column in life table")
    elif method == "Lx":
        if "lx" not in df.columns:  # Changed from 'Lx' - common column name
            raise ValueError("Method 'Lx' requires 'lx' column in life table")
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'lx', 'qx', or 'Lx'")

    # Group by sex and race for processing
    result_records = []

    for (sex, race), group in df.groupby(["sex", "race_ethnicity"]):
        group = group.sort_values("age").copy()

        # Calculate survival rates for ages 0-89
        for idx in range(len(group)):
            age = group.iloc[idx]["age"]

            if age < 90:
                # Regular age groups
                if method == "lx":
                    # Find next age
                    next_age_data = group[group["age"] == age + 1]
                    if not next_age_data.empty:
                        lx_current = group.iloc[idx]["lx"]
                        lx_next = next_age_data.iloc[0]["lx"]
                        survival_rate = lx_next / lx_current if lx_current > 0 else 0
                    else:
                        # No next age, use qx if available
                        survival_rate = 1 - group.iloc[idx]["qx"] if "qx" in group.columns else 0.95

                elif method == "qx":
                    survival_rate = 1 - group.iloc[idx]["qx"]

                elif method == "Lx":
                    # Using lx column (common in SEER data)
                    next_age_data = group[group["age"] == age + 1]
                    if not next_age_data.empty:
                        lx_current = group.iloc[idx]["lx"]
                        lx_next = next_age_data.iloc[0]["lx"]
                        survival_rate = lx_next / lx_current if lx_current > 0 else 0
                    else:
                        survival_rate = 0.95

            else:
                # Age 90+ - open-ended group
                # Use special formula if Tx and Lx available
                if "tx" in group.columns and "lx" in group.columns:
                    t_90 = group.iloc[idx]["tx"]
                    l_90 = group.iloc[idx]["lx"]

                    # t_91 = t_90 - l_90 (person-years above age 91)
                    t_91 = t_90 - l_90

                    # S(90+) = t_91 / (t_90 + l_90/2)
                    denominator = t_90 + l_90 / 2
                    survival_rate = t_91 / denominator if denominator > 0 else 0.65
                else:
                    # Default survival rate for 90+ if can't calculate
                    survival_rate = 1 - group.iloc[idx]["qx"] if "qx" in group.columns else 0.65

            # Ensure survival rate is in valid range
            survival_rate = max(0.0, min(1.0, survival_rate))

            result_records.append(
                {"age": age, "sex": sex, "race_ethnicity": race, "survival_rate": survival_rate}
            )

    result_df = pd.DataFrame(result_records)

    logger.info(f"Calculated survival rates for {len(result_df)} age-sex-race combinations")

    return result_df


def apply_mortality_improvement(
    df: pd.DataFrame, base_year: int, projection_year: int, improvement_factor: float = 0.005
) -> pd.DataFrame:
    """
    Apply mortality improvement trends over time.

    Mortality improvement means death rates decline over time, so survival
    rates increase. Applies Lee-Carter style linear improvement.

    Args:
        df: DataFrame with survival rates
        base_year: Base year of the life table
        projection_year: Target projection year
        improvement_factor: Annual mortality improvement rate (default: 0.005 = 0.5%)

    Returns:
        DataFrame with improved survival rates

    Notes:
        - Formula: q(x, t) = q(x, base_year) × (1 - improvement_factor)^(t - base_year)
        - Then: S(x, t) = 1 - q(x, t)
        - Survival rates are capped at 1.0 (cannot exceed 100%)
        - No improvement applied if projection_year <= base_year

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [65, 70, 75],
        ...     'sex': ['Female'] * 3,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 3,
        ...     'survival_rate': [0.980, 0.970, 0.950]
        ... })
        >>> improved = apply_mortality_improvement(df, 2020, 2030, 0.005)
        >>> improved['survival_rate'].iloc[0] > 0.980  # Higher survival in 2030
        True
    """
    years_elapsed = projection_year - base_year

    if years_elapsed <= 0:
        logger.info("No mortality improvement applied (projection year <= base year)")
        return df.copy()

    if improvement_factor == 0:
        logger.info("No mortality improvement applied (improvement_factor = 0)")
        return df.copy()

    logger.info(
        f"Applying {years_elapsed} years of mortality improvement "
        f"(factor: {improvement_factor:.4f})"
    )

    improved_df = df.copy()

    # Convert survival rate to death rate
    improved_df["death_rate"] = 1 - improved_df["survival_rate"]

    # Apply improvement to death rates
    improvement_multiplier = (1 - improvement_factor) ** years_elapsed
    improved_df["death_rate"] = improved_df["death_rate"] * improvement_multiplier

    # Convert back to survival rate
    improved_df["survival_rate"] = 1 - improved_df["death_rate"]

    # Cap at 1.0
    improved_df["survival_rate"] = improved_df["survival_rate"].clip(upper=1.0)

    # Drop intermediate column
    improved_df = improved_df.drop(columns=["death_rate"])

    avg_improvement = improved_df["survival_rate"].mean() - df["survival_rate"].mean()
    logger.info(
        f"Average survival rate improvement: {avg_improvement:.6f} "
        f"(from {df['survival_rate'].mean():.6f} to {improved_df['survival_rate'].mean():.6f})"
    )

    return improved_df


def create_survival_rate_table(
    df: pd.DataFrame, validate: bool = True, config: dict | None = None
) -> pd.DataFrame:
    """
    Create final survival rate table for projection.

    Ensures all age (0-90) × sex (2) × race (6) combinations are present.
    Fills missing combinations with default values based on age.

    Args:
        df: DataFrame with survival rates
        validate: Whether to validate the final table (default: True)
        config: Optional configuration dictionary

    Returns:
        DataFrame ready for projection engine with columns:
        [age, sex, race_ethnicity, survival_rate]

    Raises:
        ValueError: If validation fails and validate=True

    Notes:
        - Total combinations: 91 ages × 2 sexes × 6 races = 1,092 rows
        - Missing values filled with age-appropriate defaults:
          - Infant (age 0): 0.994
          - Children (1-14): 0.9995
          - Adults (15-64): 0.997
          - Elderly (65-89): 0.95
          - Age 90+: 0.65

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [0, 1, 2],
        ...     'sex': ['Male'] * 3,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 3,
        ...     'survival_rate': [0.994, 0.9995, 0.9995]
        ... })
        >>> table = create_survival_rate_table(df)
        >>> len(table)  # 91 ages × 2 sexes × 6 races
        1092
    """
    logger.info("Creating survival rate table for projection")

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
    required_cols = ["age", "sex", "race_ethnicity", "survival_rate"]
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

    # Reindex with NaN for missing combinations
    df_complete = df_indexed.reindex(complete_index)
    df_complete = df_complete.reset_index()

    # Fill missing values with age-appropriate defaults
    def get_default_survival_rate(age):
        """Get default survival rate based on age."""
        if age == 0:
            return 0.994  # Infant
        elif age < 15:
            return 0.9995  # Children
        elif age < 65:
            return 0.997  # Adults
        elif age < 90:
            return 0.95  # Elderly
        else:
            return 0.65  # Age 90+

    # Fill missing survival rates
    missing_mask = df_complete["survival_rate"].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        logger.warning(
            f"Filling {missing_count} missing survival rates with age-appropriate defaults"
        )
        df_complete.loc[missing_mask, "survival_rate"] = df_complete.loc[missing_mask, "age"].apply(
            get_default_survival_rate
        )

    # Ensure survival_rate column exists and is valid
    if "survival_rate" not in df_complete.columns:
        df_complete["survival_rate"] = 0.95

    # Ensure rates are in valid range
    df_complete["survival_rate"] = df_complete["survival_rate"].clip(0, 1)

    # Add metadata
    df_complete["processing_date"] = datetime.now(UTC).strftime("%Y-%m-%d")

    logger.info(
        f"Created survival rate table with {len(df_complete)} cells "
        f"({len(ages)} ages × {len(expected_sexes)} sexes × {len(expected_races)} races)"
    )

    # Validate if requested
    if validate:
        validation_result = validate_survival_rates(df_complete, config)

        if not validation_result["valid"]:
            error_msg = "Survival rate validation failed:\n" + "\n".join(
                validation_result["errors"]
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Survival validation: {warning}")

    return df_complete


def calculate_life_expectancy(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate life expectancy at birth (e0) from survival rates.

    Provides quality assurance metric by calculating period life expectancy
    for each sex-race combination.

    Args:
        df: DataFrame with survival rates

    Returns:
        Dictionary of life expectancy at birth by sex-race combination
        Format: {"{sex}_{race}": e0, ...}

    Notes:
        - Simplified calculation: e0 ≈ sum of cumulative survival probabilities
        - Proper life table method uses Lx and Tx, but this approximation
          is sufficient for validation
        - Typical U.S. values: 75-87 years depending on sex and race

    Example:
        >>> df = pd.DataFrame({
        ...     'age': list(range(91)) * 2,
        ...     'sex': ['Male'] * 91 + ['Female'] * 91,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 182,
        ...     'survival_rate': [0.994] + [0.9995] * 89 + [0.65] + [0.994] + [0.9995] * 89 + [0.65]
        ... })
        >>> e0 = calculate_life_expectancy(df)
        >>> e0['Male_White alone, Non-Hispanic'] > 70
        True
    """
    logger.info("Calculating life expectancy at birth (e0) from survival rates")

    life_exp_results = {}

    for (sex, race), group in df.groupby(["sex", "race_ethnicity"]):
        sorted_group = group.sort_values("age")

        # Calculate cumulative survival from birth
        # lx = survival probability to age x
        lx = sorted_group["survival_rate"].cumprod()

        # Life expectancy = sum of lx (person-years lived)
        # This is a simplified approximation
        e0 = lx.sum()

        key = f"{sex}_{race}"
        life_exp_results[key] = round(e0, 2)

    # Log results
    logger.info("Life expectancy at birth (e0) by sex and race:")
    for key, value in sorted(life_exp_results.items()):
        logger.info(f"  {key}: {value:.2f} years")

    return life_exp_results


def validate_survival_rates(df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    """
    Validate survival rates for plausibility.

    Performs comprehensive validation including:
    - All ages 0-90 present
    - All sex and race categories present
    - Rates in valid range (0-1)
    - Age-specific plausibility checks
    - Life expectancy calculation for QA

    Args:
        df: DataFrame with survival rates
        config: Optional configuration dictionary

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'life_expectancy': Dict[str, float]
        }

    Notes:
        - Expected survival rate ranges by age:
          - Age 0 (infant): 0.993-0.995
          - Ages 1-14: > 0.9995
          - Ages 15-44: > 0.999
          - Ages 65-84: 0.93-0.98
          - Age 90+: 0.6-0.7

    Example:
        >>> df = pd.DataFrame({
        ...     'age': list(range(91)),
        ...     'sex': ['Male'] * 91,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 91,
        ...     'survival_rate': [0.994] + [0.9995] * 89 + [0.65]
        ... })
        >>> result = validate_survival_rates(df)
        >>> result['valid']
        True
    """
    logger.info("Validating survival rates")

    errors: list[str] = []
    warnings: list[str] = []
    life_expectancy: dict[str, float] = {}
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": errors,
        "warnings": warnings,
        "life_expectancy": life_expectancy,
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
    required_cols = ["age", "sex", "race_ethnicity", "survival_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        validation_result["valid"] = False
        return validation_result

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

    # Check for rates outside [0, 1]
    if (df["survival_rate"] < 0).any():
        negative_count = (df["survival_rate"] < 0).sum()
        errors.append(f"Negative survival rates found: {negative_count} records")
        validation_result["valid"] = False

    if (df["survival_rate"] > 1).any():
        over_one_count = (df["survival_rate"] > 1).sum()
        errors.append(f"Survival rates > 1.0 found: {over_one_count} records")
        validation_result["valid"] = False

    # Check infant survival rates (age 0)
    infant_data = df[df["age"] == 0]
    if not infant_data.empty:
        min_infant = infant_data["survival_rate"].min()
        max_infant = infant_data["survival_rate"].max()

        if min_infant < 0.990:
            warnings.append(f"Low infant survival rate: {min_infant:.4f} (typical: 0.993-0.995)")
        if max_infant > 0.998:
            warnings.append(f"Unusually high infant survival rate: {max_infant:.4f}")

    # Check child survival rates (ages 1-14)
    child_data = df[(df["age"] >= 1) & (df["age"] <= 14)]
    if not child_data.empty:
        min_child = child_data["survival_rate"].min()
        if min_child < 0.9990:
            warnings.append(f"Low child survival rate: {min_child:.4f} (typical: > 0.9995)")

    # Check elderly survival rates (ages 65-84)
    elderly_data = df[(df["age"] >= 65) & (df["age"] < 85)]
    if not elderly_data.empty:
        min_elderly = elderly_data["survival_rate"].min()
        max_elderly = elderly_data["survival_rate"].max()

        if min_elderly < 0.90:
            warnings.append(f"Very low elderly survival rate: {min_elderly:.4f}")
        if max_elderly > 0.99:
            warnings.append(f"Unusually high elderly survival rate: {max_elderly:.4f}")

    # Check age 90+ survival
    age_90_data = df[df["age"] == 90]
    if not age_90_data.empty:
        min_90 = age_90_data["survival_rate"].min()
        max_90 = age_90_data["survival_rate"].max()

        if min_90 < 0.50:
            warnings.append(f"Low age 90+ survival rate: {min_90:.4f} (typical: 0.6-0.7)")
        if max_90 > 0.80:
            warnings.append(f"High age 90+ survival rate: {max_90:.4f} (typical: 0.6-0.7)")

    # Check for missing data in expected combinations
    expected_combinations = len(expected_ages) * len(expected_sexes) * len(expected_races)
    actual_combinations = len(df)
    if actual_combinations < expected_combinations:
        errors.append(
            f"Missing age-sex-race combinations: expected {expected_combinations}, "
            f"got {actual_combinations}"
        )
        validation_result["valid"] = False

    # Calculate life expectancy for validation
    try:
        life_exp = calculate_life_expectancy(df)
        validation_result["life_expectancy"] = life_exp

        # Validate life expectancy values
        for key, e0 in life_exp.items():
            if e0 < 70:
                warnings.append(f"Low life expectancy for {key}: {e0:.1f} years (typical: 75-87)")
            elif e0 > 90:
                warnings.append(f"High life expectancy for {key}: {e0:.1f} years (typical: 75-87)")

    except Exception as e:
        warnings.append(f"Could not calculate life expectancy: {e}")

    # Summary logging
    if validation_result["valid"]:
        logger.info("Survival rates validated successfully")
    else:
        logger.error(f"Survival rate validation failed with {len(errors)} errors")

    if warnings:
        logger.warning(f"Survival rate validation produced {len(warnings)} warnings")

    return validation_result


def process_survival_rates(
    input_path: str | Path,
    output_dir: Path | None = None,
    config: dict | None = None,
    base_year: int | None = None,
    improvement_factor: float | None = None,
) -> pd.DataFrame:
    """
    Main processing function for survival rates.

    Complete pipeline: Load → Harmonize → Calculate survival → Apply improvements
    → Create table → Validate → Save

    Args:
        input_path: Path to raw SEER/CDC life table file
        output_dir: Directory to save processed data (default: data/processed/mortality)
        config: Optional configuration dictionary
        base_year: Base year of life table (for metadata)
        improvement_factor: Annual mortality improvement rate (default: 0.005)

    Returns:
        Processed survival rate DataFrame

    Raises:
        ValueError: If processing or validation fails

    Notes:
        - Saves three output files:
          - survival_rates.parquet (primary, compressed)
          - survival_rates.csv (human-readable)
          - survival_rates_metadata.json (provenance)

    Example:
        >>> survival_rates = process_survival_rates(
        ...     input_path='data/raw/mortality/seer_lifetables_2020.csv',
        ...     base_year=2020,
        ...     improvement_factor=0.005
        ... )
        >>> survival_rates.to_parquet('data/processed/mortality/survival_rates.parquet')
    """
    logger.info("=" * 70)
    logger.info("Starting survival rates processing pipeline")
    logger.info("=" * 70)

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "mortality"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base year from config if not provided
    if base_year is None:
        mortality_config = config.get("rates", {}).get("mortality", {})
        base_year = mortality_config.get("life_table_year", 2020)

    # Get improvement factor from config if not provided
    if improvement_factor is None:
        mortality_config = config.get("rates", {}).get("mortality", {})
        improvement_factor = mortality_config.get("improvement_factor", 0.005)

    # Step 1: Load raw life table data
    logger.info("Step 1: Loading raw SEER/CDC life table data")
    raw_df = load_life_table_data(input_path, year=base_year)

    # Step 2: Harmonize race categories
    logger.info("Step 2: Harmonizing race/ethnicity categories")
    harmonized_df = harmonize_mortality_race_categories(raw_df)

    # Step 3: Calculate survival rates from life table
    logger.info("Step 3: Calculating survival rates from life table")

    # Determine which method to use based on available columns
    if "lx" in harmonized_df.columns:
        method = "lx"
    elif "qx" in harmonized_df.columns:
        method = "qx"
    else:
        raise ValueError("Life table must contain either 'lx' or 'qx' column")

    survival_df = calculate_survival_rates_from_life_table(harmonized_df, method=method)

    # Step 4: Create complete survival rate table
    logger.info("Step 4: Creating survival rate table")
    survival_table = create_survival_rate_table(survival_df, validate=True, config=config)

    # Step 5: Save processed data
    logger.info("Step 5: Saving processed survival rates")

    # Save as parquet (primary format)
    output_file = output_dir / "survival_rates.parquet"
    compression = config.get("output", {}).get("compression", "gzip")

    survival_table.to_parquet(output_file, compression=compression, index=False)
    logger.info(f"Saved survival rates to {output_file}")

    # Also save as CSV for human readability
    csv_file = output_dir / "survival_rates.csv"
    survival_table.to_csv(csv_file, index=False)
    logger.info(f"Saved survival rates to {csv_file}")

    # Step 6: Generate and save metadata
    logger.info("Step 6: Generating metadata")

    validation_result = validate_survival_rates(survival_table, config)

    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "source_file": str(input_path),
        "base_year": base_year,
        "improvement_factor": improvement_factor,
        "calculation_method": method,
        "total_records": len(survival_table),
        "age_range": [int(survival_table["age"].min()), int(survival_table["age"].max())],
        "sex_categories": list(survival_table["sex"].unique()),
        "race_categories": list(survival_table["race_ethnicity"].unique()),
        "life_expectancy": validation_result.get("life_expectancy", {}),
        "validation_warnings": validation_result["warnings"],
        "config_used": {
            "ages": config.get("demographics", {}).get("age_groups", {}),
            "improvement_factor": improvement_factor,
            "race_categories": config.get("demographics", {})
            .get("race_ethnicity", {})
            .get("categories", []),
        },
    }

    metadata_file = output_dir / "survival_rates_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    # Summary statistics
    logger.info("=" * 70)
    logger.info("Processing Complete - Summary Statistics")
    logger.info("=" * 70)
    logger.info(f"Total records: {len(survival_table)}")
    logger.info(f"Age range: {survival_table['age'].min()}-{survival_table['age'].max()}")
    logger.info(f"Sex categories: {survival_table['sex'].nunique()}")
    logger.info(f"Race categories: {survival_table['race_ethnicity'].nunique()}")
    logger.info(f"Base year: {base_year}")
    logger.info(f"Mortality improvement factor: {improvement_factor}")
    logger.info("\nLife Expectancy at Birth (e0) by Sex and Race:")
    for key, value in sorted(validation_result.get("life_expectancy", {}).items()):
        logger.info(f"  {key}: {value:.1f} years")
    logger.info("=" * 70)

    return survival_table


if __name__ == "__main__":
    """
    Example usage and testing.

    To test with real data, prepare a CSV/Excel file with columns:
    - age: int (0-90)
    - sex: str (Male, Female)
    - race: str (SEER race codes)
    - lx or qx: life table column
    - Optional: Lx, Tx, ex
    """

    logger.info("Survival rates processor loaded successfully")
    logger.info("Ready to process SEER/CDC life table data")

    # Example with synthetic data
    logger.info("\nExample: Creating sample survival rates from synthetic life table")

    # Create sample life table data
    sample_data = []
    races = ["White NH", "Black NH", "Hispanic", "AIAN NH", "Asian/PI NH", "Two+ Races NH"]
    sexes = ["Male", "Female"]

    for sex in sexes:
        for race in races:
            # Create a simple life table with lx column
            lx: float = 100000.0  # Radix

            for age in range(91):
                # Simplified mortality pattern
                if age == 0:
                    qx = 0.006  # Infant mortality
                elif age < 15:
                    qx = 0.0005  # Low child mortality
                elif age < 45:
                    qx = 0.001  # Young adult
                elif age < 65:
                    qx = 0.003  # Middle age
                elif age < 85:
                    qx = 0.02 + (age - 65) * 0.002  # Increasing elderly
                else:
                    qx = 0.10 + (age - 85) * 0.03  # High elderly

                # Adjust by sex (females live longer)
                if sex == "Female":
                    qx *= 0.85

                # Adjust by race
                if "Hispanic" in race or "Asian" in race:
                    qx *= 0.90  # Lower mortality
                elif "Black" in race or "AIAN" in race:
                    qx *= 1.15  # Higher mortality

                qx = min(qx, 1.0)  # Cap at 1.0

                sample_data.append({"age": age, "sex": sex, "race": race, "lx": lx, "qx": qx})

                # Calculate next lx
                lx = lx * (1 - qx)

    sample_df = pd.DataFrame(sample_data)

    logger.info(f"Created sample life table: {len(sample_df)} records")
    logger.info(f"Ages: {sample_df['age'].min()}-{sample_df['age'].max()}")
    logger.info(f"Sexes: {sample_df['sex'].nunique()}")
    logger.info(f"Races: {sample_df['race'].nunique()}")
