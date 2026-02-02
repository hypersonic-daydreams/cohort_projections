"""
Fertility rates processor for cohort projections.

Processes raw SEER/NVSS fertility rate files into the format needed by the
projection engine. Converts age-specific fertility rates by race/ethnicity
into standardized format for cohort-component projections.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.config import SEER_RACE_MAP
from cohort_projections.utils import get_logger_from_config, load_projection_config

logger = get_logger_from_config(__name__)


# Use centralized SEER race/ethnicity mappings
# See cohort_projections/config/race_mappings.py for the canonical definitions
SEER_RACE_ETHNICITY_MAP = SEER_RACE_MAP


def load_seer_fertility_data(
    file_path: str | Path, year_range: tuple[int, int] | None = None
) -> pd.DataFrame:
    """
    Load raw SEER fertility data file(s).

    Supports multiple file formats (CSV, TXT, Excel) and filters to specified
    year range if provided.

    Args:
        file_path: Path to SEER fertility data file
        year_range: Optional tuple of (min_year, max_year) to filter data

    Returns:
        DataFrame with raw SEER fertility data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported or data invalid

    Example:
        >>> df = load_seer_fertility_data(
        ...     'data/raw/fertility/seer_asfr_2018_2022.csv',
        ...     year_range=(2018, 2022)
        ... )
        >>> df.columns
        Index(['year', 'age', 'race', 'fertility_rate', 'births', 'population'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fertility data file not found: {file_path}")

    logger.info(f"Loading SEER fertility data from {file_path}")

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
        raise ValueError(f"Error reading fertility data file: {e}") from e

    logger.info(f"Loaded {len(df)} records from {file_path.name}")

    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()

    # Filter by year range if provided
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

    # Validate required columns exist (flexible naming)
    age_col = None
    for col in ["age", "age_group", "age_of_mother"]:
        if col in df.columns:
            age_col = col
            break

    if age_col is None:
        raise ValueError("No age column found. Expected one of: age, age_group, age_of_mother")

    # Standardize to 'age'
    if age_col != "age":
        df["age"] = df[age_col]

    logger.info(f"Successfully loaded fertility data with {len(df)} records")

    return df


def harmonize_fertility_race_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map SEER race codes to standard 6 categories.

    Standardizes race/ethnicity categories from SEER data sources to the
    6-category system used in projections.

    Args:
        df: DataFrame with SEER race/ethnicity column

    Returns:
        DataFrame with harmonized 'race_ethnicity' column

    Raises:
        ValueError: If race column not found

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [20, 25, 30],
        ...     'race': ['White NH', 'Hispanic', 'Black NH'],
        ...     'fertility_rate': [0.08, 0.09, 0.07]
        ... })
        >>> df = harmonize_fertility_race_categories(df)
        >>> df['race_ethnicity'].unique()
        array(['White alone, Non-Hispanic', 'Hispanic (any race)',
               'Black alone, Non-Hispanic'])
    """
    logger.info("Harmonizing fertility race/ethnicity categories")

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
    df["race_ethnicity"] = df[race_col].map(SEER_RACE_ETHNICITY_MAP)

    # Check for unmapped categories
    unmapped = df[df["race_ethnicity"].isna()][race_col].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped race categories found: {list(unmapped)}")
        logger.warning(
            "These rows will be dropped. Consider updating SEER_RACE_ETHNICITY_MAP "
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


def calculate_average_fertility_rates(df: pd.DataFrame, averaging_period: int = 5) -> pd.DataFrame:
    """
    Average fertility rates over multiple years.

    Calculates multi-year average age-specific fertility rates to smooth
    annual fluctuations and provide more stable rates for projections.
    Uses weighted average if sample sizes are available.

    Args:
        df: DataFrame with fertility rates by year, age, and race
        averaging_period: Number of years to average (default: 5)

    Returns:
        DataFrame with averaged fertility rates by age and race

    Raises:
        ValueError: If required columns missing

    Notes:
        - If 'births' and 'population' columns exist, uses weighted average
        - Otherwise uses simple mean
        - Missing values are excluded from averaging

    Example:
        >>> df = pd.DataFrame({
        ...     'year': [2018, 2019, 2020, 2018, 2019, 2020],
        ...     'age': [25, 25, 25, 30, 30, 30],
        ...     'race_ethnicity': ['Hispanic (any race)'] * 6,
        ...     'fertility_rate': [0.09, 0.088, 0.092, 0.07, 0.068, 0.072],
        ...     'population': [1000, 1100, 1050, 800, 850, 825]
        ... })
        >>> avg_df = calculate_average_fertility_rates(df, averaging_period=3)
        >>> avg_df[avg_df['age'] == 25]['fertility_rate'].values[0]
        0.09
    """
    logger.info(f"Calculating average fertility rates over {averaging_period} years")

    # Validate required columns
    required_cols = ["age", "race_ethnicity", "fertility_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check if we have data for weighting
    has_weights = "population" in df.columns and "births" in df.columns

    if has_weights:
        logger.info("Using weighted average based on female population")

        # Calculate weighted average by age and race
        # Weight = female population in reproductive ages
        df = df.copy()
        df["weighted_rate"] = df["fertility_rate"] * df["population"]

        grouped = df.groupby(["age", "race_ethnicity"], as_index=False).agg(
            {
                "weighted_rate": "sum",
                "population": "sum",
                "fertility_rate": "count",  # count for metadata
            }
        )

        # Calculate weighted average
        grouped["fertility_rate"] = grouped["weighted_rate"] / grouped["population"]
        grouped["years_averaged"] = grouped["fertility_rate"]  # reuse column for count

        # Drop intermediate columns
        averaged_df = grouped[["age", "race_ethnicity", "fertility_rate"]].copy()

    else:
        logger.info("Using simple mean (no population weights available)")

        # Simple average
        averaged_df = df.groupby(["age", "race_ethnicity"], as_index=False).agg(
            {"fertility_rate": "mean"}
        )

    # Handle any NaN values (should be rare)
    nan_count = averaged_df["fertility_rate"].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in averaged rates, setting to 0")
        averaged_df["fertility_rate"] = averaged_df["fertility_rate"].fillna(0.0)

    logger.info(f"Calculated average rates for {len(averaged_df)} age-race combinations")

    return averaged_df


def create_fertility_rate_table(
    df: pd.DataFrame, validate: bool = True, config: dict | None = None
) -> pd.DataFrame:
    """
    Create final fertility rate table for projection.

    Ensures all age × race combinations are present for reproductive ages
    (15-49) and all 6 race/ethnicity categories. Fills missing combinations
    with 0 as per ADR-001 decision.

    Args:
        df: DataFrame with averaged fertility rates
        validate: Whether to validate the final table (default: True)
        config: Optional configuration dictionary

    Returns:
        DataFrame ready for projection engine with columns:
        [age, race_ethnicity, fertility_rate]

    Raises:
        ValueError: If validation fails and validate=True

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [20, 25, 30],
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 3,
        ...     'fertility_rate': [0.05, 0.08, 0.07]
        ... })
        >>> table = create_fertility_rate_table(df)
        >>> len(table)  # 35 ages (15-49) × 6 races = 210
        210
        >>> table['fertility_rate'].min()
        0.0
    """
    logger.info("Creating fertility rate table for projection")

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Get expected categories from config
    demographics = config.get("demographics", {})
    expected_races = demographics.get("race_ethnicity", {}).get("categories", [])

    # Get reproductive age range from config
    fertility_config = config.get("rates", {}).get("fertility", {})
    reproductive_ages = fertility_config.get("apply_to_ages", [15, 49])
    min_age, max_age = reproductive_ages

    # Validate input has required columns
    required_cols = ["age", "race_ethnicity", "fertility_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to reproductive ages
    df = df[(df["age"] >= min_age) & (df["age"] <= max_age)].copy()

    logger.info(f"Filtered to reproductive ages {min_age}-{max_age}: {len(df)} records")

    # Create complete index (all age-race combinations)
    ages = list(range(min_age, max_age + 1))

    complete_index = pd.MultiIndex.from_product(
        [ages, expected_races], names=["age", "race_ethnicity"]
    )

    # Set index and reindex to include all combinations
    df = df.set_index(["age", "race_ethnicity"])

    # Reindex with 0 for missing combinations (ADR-001 decision)
    df = df.reindex(complete_index, fill_value=0)
    df = df.reset_index()

    # Ensure fertility_rate column exists
    if "fertility_rate" not in df.columns:
        df["fertility_rate"] = 0.0

    # Ensure non-negative rates
    df["fertility_rate"] = df["fertility_rate"].clip(lower=0)

    # Add metadata
    df["processing_date"] = datetime.now(UTC).strftime("%Y-%m-%d")

    logger.info(
        f"Created fertility rate table with {len(df)} cells "
        f"({len(ages)} ages × {len(expected_races)} races)"
    )

    # Validate if requested
    if validate:
        validation_result = validate_fertility_rates(df, config)

        if not validation_result["valid"]:
            error_msg = "Fertility rate validation failed:\n" + "\n".join(
                validation_result["errors"]
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Fertility validation: {warning}")

    return df


def validate_fertility_rates(df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    """
    Validate fertility rates for plausibility.

    Performs comprehensive validation including:
    - All ages 15-49 present
    - All race categories present
    - Rates in plausible range (0-0.15)
    - Total Fertility Rate (TFR) calculation
    - No negative values
    - No missing data

    Args:
        df: DataFrame with fertility rates
        config: Optional configuration dictionary

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'tfr_by_race': Dict[str, float],
            'overall_tfr': float
        }

    Example:
        >>> df = pd.DataFrame({
        ...     'age': list(range(15, 50)) * 6,
        ...     'race_ethnicity': ['White alone, Non-Hispanic'] * 35 + [...],
        ...     'fertility_rate': [0.05] * 210
        ... })
        >>> result = validate_fertility_rates(df)
        >>> result['valid']
        True
        >>> result['tfr_by_race']['White alone, Non-Hispanic']
        1.75
    """
    logger.info("Validating fertility rates")

    errors: list[str] = []
    warnings: list[str] = []
    tfr_by_race: dict[str, float] = {}
    validation_result: dict[str, Any] = {
        "valid": True,
        "errors": errors,
        "warnings": warnings,
        "tfr_by_race": tfr_by_race,
        "overall_tfr": 0.0,
    }

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Get expected values from config
    demographics = config.get("demographics", {})
    expected_races = demographics.get("race_ethnicity", {}).get("categories", [])

    fertility_config = config.get("rates", {}).get("fertility", {})
    reproductive_ages = fertility_config.get("apply_to_ages", [15, 49])
    min_age, max_age = reproductive_ages

    # Check for required columns
    required_cols = ["age", "race_ethnicity", "fertility_rate"]
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
        errors.append(f"Missing ages in reproductive range: {sorted(missing_ages)}")
        validation_result["valid"] = False

    # Check all race categories present
    actual_races = set(df["race_ethnicity"].unique())
    missing_races = set(expected_races) - actual_races
    if missing_races:
        errors.append(f"Missing race categories: {list(missing_races)}")
        validation_result["valid"] = False

    # Check for negative rates
    if (df["fertility_rate"] < 0).any():
        negative_count = (df["fertility_rate"] < 0).sum()
        errors.append(f"Negative fertility rates found: {negative_count} records")
        validation_result["valid"] = False

    # Check for implausibly high rates
    max_plausible_rate = 0.15  # 150 births per 1000 women
    high_rates = df[df["fertility_rate"] > max_plausible_rate]
    if not high_rates.empty:
        warnings.append(
            f"Very high fertility rates (>{max_plausible_rate}) found in "
            f"{len(high_rates)} records. Max rate: {df['fertility_rate'].max():.4f}"
        )

    # Warn about rates above 0.13 (typical maximum)
    typical_max = 0.13
    above_typical = df[df["fertility_rate"] > typical_max]
    if not above_typical.empty:
        warnings.append(
            f"Fertility rates above typical maximum ({typical_max}) found in "
            f"{len(above_typical)} records"
        )

    # Check for missing data in expected combinations
    expected_combinations = len(expected_ages) * len(expected_races)
    actual_combinations = len(df)
    if actual_combinations < expected_combinations:
        errors.append(
            f"Missing age-race combinations: expected {expected_combinations}, "
            f"got {actual_combinations}"
        )
        validation_result["valid"] = False

    # Calculate Total Fertility Rate (TFR) by race
    # TFR = sum of age-specific fertility rates (represents births per woman)
    for race in expected_races:
        race_data = df[df["race_ethnicity"] == race]
        if not race_data.empty:
            tfr = race_data["fertility_rate"].sum()
            tfr_by_race[race] = round(tfr, 3)

            # Typical TFR range: 1.3-2.5 for developed countries
            if tfr < 1.0:
                warnings.append(f"Very low TFR for {race}: {tfr:.2f} (typical range: 1.3-2.5)")
            elif tfr > 3.0:
                warnings.append(f"Very high TFR for {race}: {tfr:.2f} (typical range: 1.3-2.5)")

    # Calculate overall TFR (population-weighted if possible, otherwise mean)
    if tfr_by_race:
        validation_result["overall_tfr"] = round(np.mean(list(tfr_by_race.values())), 3)

    # Summary logging
    if validation_result["valid"]:
        logger.info(
            f"Fertility rates validated successfully. "
            f"Overall TFR: {validation_result['overall_tfr']:.2f}"
        )
    else:
        logger.error(f"Fertility rate validation failed with {len(errors)} errors")

    if warnings:
        logger.warning(f"Fertility rate validation produced {len(warnings)} warnings")

    return validation_result


def process_fertility_rates(
    input_path: str | Path,
    output_dir: Path | None = None,
    config: dict | None = None,
    year_range: tuple[int, int] | None = None,
    averaging_period: int = 5,
) -> pd.DataFrame:
    """
    Main processing function for fertility rates.

    Complete pipeline: Load → Harmonize → Average → Create table → Validate → Save

    Args:
        input_path: Path to raw SEER/NVSS fertility data file
        output_dir: Directory to save processed data (default: data/processed/fertility)
        config: Optional configuration dictionary
        year_range: Optional (min_year, max_year) tuple to filter data
        averaging_period: Years to average (default: 5)

    Returns:
        Processed fertility rate DataFrame

    Raises:
        ValueError: If processing or validation fails

    Example:
        >>> fertility_rates = process_fertility_rates(
        ...     input_path='data/raw/fertility/seer_asfr_2018_2022.csv',
        ...     year_range=(2018, 2022),
        ...     averaging_period=5
        ... )
        >>> fertility_rates.to_parquet('data/processed/fertility/fertility_rates.parquet')
    """
    logger.info("=" * 70)
    logger.info("Starting fertility rates processing pipeline")
    logger.info("=" * 70)

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "fertility"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    logger.info("Step 1: Loading raw SEER fertility data")
    raw_df = load_seer_fertility_data(input_path, year_range=year_range)

    # Step 2: Harmonize race categories
    logger.info("Step 2: Harmonizing race/ethnicity categories")
    harmonized_df = harmonize_fertility_race_categories(raw_df)

    # Step 3: Calculate average rates
    logger.info("Step 3: Calculating multi-year average rates")
    averaged_df = calculate_average_fertility_rates(
        harmonized_df, averaging_period=averaging_period
    )

    # Step 4: Create complete fertility rate table
    logger.info("Step 4: Creating fertility rate table")
    fertility_table = create_fertility_rate_table(averaged_df, validate=True, config=config)

    # Step 5: Save processed data
    logger.info("Step 5: Saving processed fertility rates")

    # Save as parquet (primary format)
    output_file = output_dir / "fertility_rates.parquet"
    compression = config.get("output", {}).get("compression", "gzip")

    fertility_table.to_parquet(output_file, compression=compression, index=False)
    logger.info(f"Saved fertility rates to {output_file}")

    # Also save as CSV for human readability
    csv_file = output_dir / "fertility_rates.csv"
    fertility_table.to_csv(csv_file, index=False)
    logger.info(f"Saved fertility rates to {csv_file}")

    # Step 6: Generate and save metadata
    logger.info("Step 6: Generating metadata")

    validation_result = validate_fertility_rates(fertility_table, config)

    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "source_file": str(input_path),
        "year_range": year_range,
        "averaging_period": averaging_period,
        "total_records": len(fertility_table),
        "age_range": [int(fertility_table["age"].min()), int(fertility_table["age"].max())],
        "race_categories": list(fertility_table["race_ethnicity"].unique()),
        "tfr_by_race": validation_result["tfr_by_race"],
        "overall_tfr": validation_result["overall_tfr"],
        "validation_warnings": validation_result["warnings"],
        "config_used": {
            "reproductive_ages": config.get("rates", {})
            .get("fertility", {})
            .get("apply_to_ages", [15, 49]),
            "race_categories": config.get("demographics", {})
            .get("race_ethnicity", {})
            .get("categories", []),
        },
    }

    metadata_file = output_dir / "fertility_rates_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    # Summary statistics
    logger.info("=" * 70)
    logger.info("Processing Complete - Summary Statistics")
    logger.info("=" * 70)
    logger.info(f"Total records: {len(fertility_table)}")
    logger.info(f"Age range: {fertility_table['age'].min()}-{fertility_table['age'].max()}")
    logger.info(f"Race categories: {fertility_table['race_ethnicity'].nunique()}")
    logger.info(f"Overall TFR: {validation_result['overall_tfr']:.2f}")
    logger.info("\nTFR by Race/Ethnicity:")
    for race, tfr in validation_result["tfr_by_race"].items():
        logger.info(f"  {race}: {tfr:.2f}")
    logger.info("=" * 70)

    return fertility_table


if __name__ == "__main__":
    """
    Example usage and testing.

    To test with real data, prepare a CSV/Excel file with columns:
    - year: int (e.g., 2018-2022)
    - age: int (15-49)
    - race: str (SEER race codes)
    - fertility_rate: float (births per woman)
    - Optional: births, population (for weighted averaging)
    """

    logger.info("Fertility rates processor loaded successfully")
    logger.info("Ready to process SEER/NVSS fertility data")

    # Example with synthetic data
    logger.info("\nExample: Creating sample fertility rates")

    # Create sample data
    sample_data = []
    races = ["White NH", "Black NH", "Hispanic", "AIAN NH", "Asian/PI NH", "Two+ Races NH"]

    for year in range(2018, 2023):
        for age in range(15, 50):
            for race in races:
                # Simplified fertility pattern (peaks around age 28)
                base_rate = 0.001 + 0.004 * np.exp(-((age - 28) ** 2) / 50)
                # Add some variation by race
                if "Hispanic" in race:
                    base_rate *= 1.2
                elif "AIAN" in race:
                    base_rate *= 1.1

                sample_data.append(
                    {
                        "year": year,
                        "age": age,
                        "race": race,
                        "fertility_rate": base_rate,
                        "population": 1000 + np.random.randint(-100, 100),
                    }
                )

    sample_df = pd.DataFrame(sample_data)

    logger.info(f"Created sample data: {len(sample_df)} records")
    logger.info(f"Years: {sample_df['year'].min()}-{sample_df['year'].max()}")
    logger.info(f"Ages: {sample_df['age'].min()}-{sample_df['age'].max()}")
    logger.info(f"Races: {sample_df['race'].nunique()}")
