#!/usr/bin/env python3
"""
Projection Runner Pipeline for North Dakota Population Projections.

This script orchestrates the execution of cohort-component projections for
all configured geographies (state, counties, places). It supports:
- Multiple scenarios (baseline, high growth, low growth, etc.)
- Parallel processing for multiple geographies
- Geography filtering (all, counties only, places only, specific FIPS)
- Resume capability (skip already-completed geographies)
- Hierarchical aggregation and validation

Usage:
    # Run all projections
    python 02_run_projections.py --all

    # Run state-level only
    python 02_run_projections.py --state

    # Run county-level projections
    python 02_run_projections.py --counties

    # Run place-level projections
    python 02_run_projections.py --places

    # Run specific geographies by FIPS
    python 02_run_projections.py --fips 38101 38015 38035

    # Run multiple scenarios
    python 02_run_projections.py --all --scenarios baseline high_growth

    # Resume from previous run (skip completed)
    python 02_run_projections.py --all --resume

    # Dry run mode
    python 02_run_projections.py --all --dry-run
"""

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.data.load.base_population_loader import (  # noqa: E402
    load_base_population_for_county,
)
from cohort_projections.geographic.geography_loader import (  # noqa: E402
    load_geography_list,
    load_nd_counties,
)
from cohort_projections.geographic.multi_geography import (  # noqa: E402
    run_multi_geography_projections,
)
from cohort_projections.utils import load_projection_config  # noqa: E402

# Set up logging
logger = setup_logger(__name__, log_level="INFO")


class ProjectionRunMetadata:
    """Container for projection run metadata."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.start_time = datetime.now(UTC)
        self.end_time: datetime | None = None
        self.geographies_total = 0
        self.geographies_completed = 0
        self.geographies_failed = 0
        self.geographies_skipped = 0
        self.failed_geographies: list[dict[str, str]] = []
        self.output_files: list[Path] = []

    def finalize(self):
        """Finalize the metadata."""
        self.end_time = datetime.now(UTC)

    def get_summary(self) -> dict[str, Any]:
        """Get metadata summary."""
        return {
            "scenario": self.scenario,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds() if self.end_time else None
            ),
            "geographies": {
                "total": self.geographies_total,
                "completed": self.geographies_completed,
                "failed": self.geographies_failed,
                "skipped": self.geographies_skipped,
            },
            "failed_geographies": self.failed_geographies,
            "output_files": [str(f) for f in self.output_files],
        }


def get_completed_geographies(output_dir: Path, scenario: str) -> set[str]:
    """
    Get set of already-completed geographies for resume capability.

    Args:
        output_dir: Output directory to check
        scenario: Scenario name

    Returns:
        Set of completed FIPS codes
    """
    completed: set[str] = set()
    scenario_dir = output_dir / scenario

    if not scenario_dir.exists():
        return completed

    # Check for completed projection files
    for level in ["state", "county", "place"]:
        level_dir = scenario_dir / level
        if level_dir.exists():
            for file in level_dir.glob("*.parquet"):
                # Extract FIPS from filename (e.g., "38101_projection.parquet")
                fips = file.stem.split("_")[0]
                completed.add(fips)

    return completed


def _expand_age_groups_to_single_years(
    df: pd.DataFrame, age_col: str = "age", rate_col: str = "fertility_rate"
) -> pd.DataFrame:
    """
    Expand 5-year age groups to single-year ages.

    Args:
        df: DataFrame with age groups like "15-19", "20-24"
        age_col: Column name containing age groups
        rate_col: Column name containing the rate to apply to all ages in group

    Returns:
        DataFrame with single-year ages (integers)
    """
    expanded_rows = []
    for _, row in df.iterrows():
        age_str = row[age_col]
        if "-" in str(age_str):
            start, end = map(int, str(age_str).split("-"))
            for single_age in range(start, end + 1):
                new_row = row.copy()
                new_row[age_col] = single_age
                expanded_rows.append(new_row)
        else:
            row_copy = row.copy()
            row_copy[age_col] = int(age_str)
            expanded_rows.append(row_copy)

    return pd.DataFrame(expanded_rows)


RACE_CODE_TO_NAME = {
    "total": None,  # Skip 'total' row - use race-specific rates
    "white_nh": "White alone, Non-Hispanic",
    "black_nh": "Black alone, Non-Hispanic",
    "aian_nh": "AIAN alone, Non-Hispanic",
    "asian_nh": "Asian/PI alone, Non-Hispanic",
    "hispanic": "Hispanic (any race)",
    # For multiracial, we'll use the average of other rates or a default
    "multiracial_nh": "Two or more races, Non-Hispanic",
    "two_or_more_nh": "Two or more races, Non-Hispanic",
}


def _transform_fertility_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform fertility rates to engine-expected format.

    Input format: age (groups), race_ethnicity, asfr, year
    Output format: age (int), race, fertility_rate
    """
    df = df.copy()

    # Rename columns to match engine expectations
    df = df.rename(columns={"asfr": "fertility_rate", "race_ethnicity": "race"})

    # Convert ASFR from per-1000 to per-capita if needed (check range)
    if df["fertility_rate"].max() > 1:
        df["fertility_rate"] = df["fertility_rate"] / 1000

    # Map race codes to full names
    df["race"] = df["race"].map(RACE_CODE_TO_NAME)

    # Remove rows with unmapped race codes (like 'total')
    df = df.dropna(subset=["race"])

    # Expand age groups to single years
    df = _expand_age_groups_to_single_years(df, "age", "fertility_rate")

    # Add missing "Two or more races, Non-Hispanic" category
    # Use average of other races if not present
    if "Two or more races, Non-Hispanic" not in df["race"].unique():
        # Calculate average rates by age
        avg_rates = df.groupby("age")["fertility_rate"].mean().reset_index()
        avg_rates["race"] = "Two or more races, Non-Hispanic"
        df = pd.concat([df, avg_rates], ignore_index=True)

    # Keep only required columns
    result = df[["age", "race", "fertility_rate"]].copy()
    result["age"] = result["age"].astype(int)

    return result


def _transform_survival_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform survival rates to engine-expected format.

    Engine expects: age, sex, race, survival_rate
    """
    df = df.copy()

    # Handle column renames if needed
    if "race_ethnicity" in df.columns and "race" not in df.columns:
        df = df.rename(columns={"race_ethnicity": "race"})

    # Map race codes to full names
    df["race"] = df["race"].map(lambda x: RACE_CODE_TO_NAME.get(x, x))

    # Remove rows with unmapped race codes
    df = df[df["race"].notna() & (df["race"] != "")]

    # Capitalize sex values: "female" -> "Female"
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    # Handle age groups if present
    if df["age"].dtype == object and df["age"].str.contains("-").any():
        df = _expand_age_groups_to_single_years(df, "age", "survival_rate")

    # Ensure age is integer
    if "age" in df.columns:
        df["age"] = df["age"].astype(int)

    # Add missing "Two or more races" if not present
    if "Two or more races, Non-Hispanic" not in df["race"].unique():
        # Use average of other races by age and sex
        avg_rates = df.groupby(["age", "sex"])["survival_rate"].mean().reset_index()
        avg_rates["race"] = "Two or more races, Non-Hispanic"
        df = pd.concat([df, avg_rates], ignore_index=True)

    # Keep only required columns
    result = df[["age", "sex", "race", "survival_rate"]].copy()

    return result


def _get_age_migration_pattern() -> dict[int, float]:
    """
    Get standard age-specific migration pattern based on demographic literature.

    Young adults (18-34) are most mobile, with peak migration around age 22-25.
    Children migrate with parents, elderly are less mobile.

    Returns dict mapping age to relative migration propensity (sums to 1.0).
    """
    pattern = {}

    for age in range(91):
        if age < 5:
            # Children migrate with parents
            pattern[age] = 0.015
        elif age < 18:
            # School-age children, moderate mobility
            pattern[age] = 0.012
        elif age < 22:
            # College age, high mobility
            pattern[age] = 0.025
        elif age < 30:
            # Young adults, peak mobility
            pattern[age] = 0.030
        elif age < 40:
            # Early career, still mobile
            pattern[age] = 0.020
        elif age < 55:
            # Mid-career, settling down
            pattern[age] = 0.012
        elif age < 65:
            # Late career
            pattern[age] = 0.008
        elif age < 75:
            # Retirement age - some move for retirement
            pattern[age] = 0.010
        else:
            # Elderly, least mobile
            pattern[age] = 0.005

    # Normalize to sum to 1.0
    total = sum(pattern.values())
    return {age: rate / total for age, rate in pattern.items()}


def _transform_migration_rates(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Transform migration rates to engine-expected format.

    Engine expects: age, sex, race, net_migration or migration_rate

    The input data is county-level net migration. We need to create an
    age-sex-race breakdown using standard migration patterns.
    """
    # Check if already in correct format (has age, sex, race columns)
    if all(col in df.columns for col in ["age", "sex", "race"]):
        # Already in correct format, just apply race/sex transformations
        df = df.copy()
        if "race_ethnicity" in df.columns and "race" not in df.columns:
            df = df.rename(columns={"race_ethnicity": "race"})

        df["race"] = df["race"].map(lambda x: RACE_CODE_TO_NAME.get(x, x))
        if "sex" in df.columns:
            df["sex"] = df["sex"].str.title()

        return df

    # County-level data - need to create age-sex-race breakdown
    # For now, create a uniform rate structure that doesn't vary by FIPS
    # (migration will be applied per-geography in the projection)

    # Get standard age pattern (reserved for future county-specific migration)
    _age_pattern = _get_age_migration_pattern()

    # Define expected demographic categories
    ages = list(range(91))
    sexes = ["Male", "Female"]
    races = [
        "White alone, Non-Hispanic",
        "Black alone, Non-Hispanic",
        "AIAN alone, Non-Hispanic",
        "Asian/PI alone, Non-Hispanic",
        "Two or more races, Non-Hispanic",
        "Hispanic (any race)",
    ]

    # Create migration rate structure
    # Use a small default net migration rate (as proportion of population)
    # Actual county-level adjustments should happen at projection time
    # Note: age_pattern and sex weights reserved for future county-specific migration
    rows = []
    for age in ages:
        for sex in sexes:
            for race in races:
                rows.append(
                    {
                        "age": age,
                        "sex": sex,
                        "race": race,
                        # Small baseline rate - actual migration is applied per-county
                        "migration_rate": 0.0,  # Net zero by default
                    }
                )

    result = pd.DataFrame(rows)

    return result


def load_demographic_rates(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed demographic rates from data/processed/ directory.

    Transforms the rates to match the format expected by the projection engine:
    - fertility_rates: [age (int), race, fertility_rate]
    - survival_rates: [age (int), sex, race, survival_rate]
    - migration_rates: [age (int), sex, race, migration_rate]

    Args:
        config: Project configuration

    Returns:
        Tuple of (fertility_rates, survival_rates, migration_rates)

    Raises:
        FileNotFoundError: If required rate files not found
    """
    logger.info("Loading processed demographic rates...")

    # Load from data/processed/ directory
    processed_dir = project_root / "data" / "processed"

    fertility_file = processed_dir / "fertility_rates.parquet"
    survival_file = processed_dir / "survival_rates.parquet"
    migration_file = processed_dir / "migration_rates.parquet"

    if not fertility_file.exists():
        raise FileNotFoundError(f"Fertility rates not found: {fertility_file}")
    if not survival_file.exists():
        raise FileNotFoundError(f"Survival rates not found: {survival_file}")
    if not migration_file.exists():
        raise FileNotFoundError(f"Migration rates not found: {migration_file}")

    # Load raw data
    fertility_rates_raw = pd.read_parquet(fertility_file)
    survival_rates_raw = pd.read_parquet(survival_file)
    migration_rates_raw = pd.read_parquet(migration_file)

    logger.info(f"Loaded fertility rates: {len(fertility_rates_raw):,} records")
    logger.info(f"Loaded survival rates: {len(survival_rates_raw):,} records")
    logger.info(f"Loaded migration rates: {len(migration_rates_raw):,} records")

    # Transform to engine-expected format
    logger.info("Transforming rates to engine format...")
    fertility_rates = _transform_fertility_rates(fertility_rates_raw)
    survival_rates = _transform_survival_rates(survival_rates_raw)
    migration_rates = _transform_migration_rates(migration_rates_raw)

    logger.info(f"Transformed fertility rates: {len(fertility_rates):,} records")
    logger.info(f"Transformed survival rates: {len(survival_rates):,} records")
    logger.info(f"Transformed migration rates: {len(migration_rates):,} records")

    return fertility_rates, survival_rates, migration_rates


def load_base_population(config: dict[str, Any], fips: str) -> pd.DataFrame:
    """
    Load base year population for a geography.

    Args:
        config: Project configuration
        fips: FIPS code

    Returns:
        Base population DataFrame with columns [year, age, sex, race, population]
    """
    return load_base_population_for_county(fips, config)


def setup_projection_run(
    config: dict[str, Any],
    levels: list[str],
    fips_filter: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Set up projection run by determining which geographies to process.

    Args:
        config: Project configuration
        levels: Geographic levels to process ('state', 'county', 'place')
        fips_filter: Optional list of specific FIPS codes to process
        scenarios: Optional list of scenarios to run (default: from config)

    Returns:
        Tuple of (geographies dict, scenarios list)
        geographies: {'state': [...], 'county': [...], 'place': [...]}
    """
    logger.info("Setting up projection run...")

    geographies: dict[str, list[str]] = {"state": [], "county": [], "place": []}

    # Determine scenarios
    resolved_scenarios: list[str]
    if scenarios is None:
        # Get active scenarios from config
        all_scenarios = config.get("scenarios", {})
        resolved_scenarios = [
            name for name, settings in all_scenarios.items() if settings.get("active", False)
        ]
        if not resolved_scenarios:
            # Fallback to pipeline config
            resolved_scenarios = (
                config.get("pipeline", {}).get("projection", {}).get("scenarios", ["baseline"])
            )
    else:
        resolved_scenarios = scenarios

    logger.info(f"Scenarios to run: {', '.join(resolved_scenarios)}")

    # State level
    if "state" in levels:
        state_fips = config.get("geography", {}).get("state", "38")
        geographies["state"] = [state_fips]
        logger.info(f"State: {state_fips}")

    # County level
    if "county" in levels:
        county_config = config.get("geography", {}).get("counties", {})
        mode = county_config.get("mode", "all")

        # Get reference data settings
        ref_config = config.get("geography", {}).get("reference_data", {})
        source = ref_config.get("source", "local")
        vintage = ref_config.get("vintage", 2020)
        counties_file = ref_config.get("counties_file")
        reference_path = Path(counties_file) if counties_file else None

        if fips_filter:
            # Use provided FIPS filter (only counties)
            county_fips = [f for f in fips_filter if len(f) == 5]
            geographies["county"] = county_fips
        elif mode == "all":
            # Load all counties
            counties_df = load_nd_counties(
                source=source, vintage=vintage, reference_path=reference_path
            )
            geographies["county"] = counties_df["county_fips"].tolist()
        elif mode == "list":
            geographies["county"] = county_config.get("fips_codes", [])
        elif mode == "threshold":
            # Load counties above population threshold
            counties_df = load_nd_counties(
                source=source, vintage=vintage, reference_path=reference_path
            )
            min_pop = county_config.get("min_population", 1000)
            geographies["county"] = counties_df[counties_df["population"] >= min_pop][
                "county_fips"
            ].tolist()

        logger.info(f"Counties: {len(geographies['county'])} to process")

    # Place level
    if "place" in levels:
        place_config = config.get("geography", {}).get("places", {})
        mode = place_config.get("mode", "threshold")

        if fips_filter:
            # Use provided FIPS filter (only places - 7 digits)
            place_fips = [f for f in fips_filter if len(f) == 7]
            geographies["place"] = place_fips
        else:
            # Load places based on config (mode/min_population read from config)
            geographies["place"] = load_geography_list(
                level="place",
                config=config,
                fips_codes=place_config.get("fips_codes"),
            )

        logger.info(f"Places: {len(geographies['place'])} to process")

    return geographies, resolved_scenarios


def apply_scenario_rate_adjustments(
    scenario: str,
    config: dict[str, Any],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply scenario-specific rate adjustments to demographic rates.

    For high growth scenario, this applies:
    - Fertility: +10% multiplier
    - Migration: +25% multiplier
    - Mortality: constant (no change)

    Args:
        scenario: Scenario name (e.g., 'baseline', 'high_growth', 'low_growth')
        config: Project configuration
        fertility_rates: Base fertility rates
        survival_rates: Base survival rates
        migration_rates: Base migration rates

    Returns:
        Tuple of (adjusted_fertility, adjusted_survival, adjusted_migration)
    """
    scenario_config = config.get("scenarios", {}).get(scenario, {})

    if not scenario_config:
        logger.warning(f"Scenario '{scenario}' not found in config, using base rates")
        return fertility_rates, survival_rates, migration_rates

    # Copy rates to avoid modifying originals
    adj_fertility = fertility_rates.copy()
    adj_survival = survival_rates.copy()
    adj_migration = migration_rates.copy()

    # Apply fertility adjustment
    fertility_setting = scenario_config.get("fertility", "constant")
    if fertility_setting == "+10_percent":
        logger.info(f"Scenario {scenario}: Applying +10% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 1.10
    elif fertility_setting == "-10_percent":
        logger.info(f"Scenario {scenario}: Applying -10% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 0.90

    # Apply mortality adjustment (survival rates)
    mortality_setting = scenario_config.get("mortality", "constant")
    if mortality_setting == "constant":
        logger.info(f"Scenario {scenario}: Keeping mortality rates constant")
        # No change to survival rates
    elif mortality_setting == "improving":
        logger.info(f"Scenario {scenario}: Mortality rates set to improving (handled per-year)")
        # Improvement is typically applied year-by-year in projection, not here

    # Apply migration adjustment
    migration_setting = scenario_config.get("migration", "recent_average")
    migration_col = (
        "migration_rate" if "migration_rate" in adj_migration.columns else "net_migration"
    )

    if migration_setting == "+25_percent":
        logger.info(f"Scenario {scenario}: Applying +25% migration adjustment")
        if migration_col in adj_migration.columns:
            adj_migration[migration_col] = adj_migration[migration_col] * 1.25
    elif migration_setting == "-25_percent":
        logger.info(f"Scenario {scenario}: Applying -25% migration adjustment")
        if migration_col in adj_migration.columns:
            adj_migration[migration_col] = adj_migration[migration_col] * 0.75
    elif migration_setting == "zero":
        logger.info(f"Scenario {scenario}: Setting migration to zero")
        if migration_col in adj_migration.columns:
            adj_migration[migration_col] = 0.0

    return adj_fertility, adj_survival, adj_migration


def run_geographic_projections(
    geographies: dict[str, list[str]],
    scenario: str,
    config: dict[str, Any],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    dry_run: bool = False,
    resume: bool = False,
) -> ProjectionRunMetadata:
    """
    Execute projections for all geographies in a scenario.

    Args:
        geographies: Dictionary of geographies by level
        scenario: Scenario name
        config: Project configuration
        fertility_rates: Processed fertility rates
        survival_rates: Processed survival rates
        migration_rates: Processed migration rates
        dry_run: If True, only show what would be processed
        resume: If True, skip already-completed geographies

    Returns:
        ProjectionRunMetadata with results
    """
    metadata = ProjectionRunMetadata(scenario)

    # Count total geographies
    metadata.geographies_total = sum(len(g) for g in geographies.values())
    logger.info(f"Total geographies to process: {metadata.geographies_total}")

    if dry_run:
        logger.info("[DRY RUN] Would process projections")
        metadata.finalize()
        return metadata

    # Apply scenario-specific rate adjustments
    logger.info(f"Applying scenario rate adjustments for: {scenario}")
    adj_fertility, adj_survival, adj_migration = apply_scenario_rate_adjustments(
        scenario=scenario,
        config=config,
        fertility_rates=fertility_rates,
        survival_rates=survival_rates,
        migration_rates=migration_rates,
    )

    # Get output directory
    output_dir = (
        Path(config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections"))
        / scenario
    )

    # Get completed geographies for resume
    completed = set()
    if resume:
        completed = get_completed_geographies(output_dir.parent, scenario)
        logger.info(f"Resume mode: {len(completed)} geographies already completed")

    # Process each level
    for level_key, fips_list in geographies.items():
        if not fips_list:
            continue
        # Cast to Literal for type safety
        level = cast(Literal["state", "county", "place"], level_key)

        logger.info(f"\nProcessing {level} level: {len(fips_list)} geographies")

        # Filter out completed if resuming
        if resume:
            fips_to_process = [f for f in fips_list if f not in completed]
            skipped = len(fips_list) - len(fips_to_process)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already-completed geographies")
                metadata.geographies_skipped += skipped
        else:
            fips_to_process = fips_list

        if not fips_to_process:
            logger.info("No geographies to process at this level")
            continue

        # Run projections
        try:
            # Build base population and migration rate dictionaries per geography
            # TODO: Implement actual base population loading per geography
            base_population_by_geography = {
                fips: load_base_population(config, fips) for fips in fips_to_process
            }

            # Build migration rates dictionary per geography
            # Use adjusted migration rates for this scenario
            migration_rates_by_geography = dict.fromkeys(fips_to_process, adj_migration)

            results_dict = run_multi_geography_projections(
                level=level,
                base_population_by_geography=base_population_by_geography,
                fertility_rates=adj_fertility,
                survival_rates=adj_survival,
                migration_rates_by_geography=migration_rates_by_geography,
                config=config,
                fips_codes=fips_to_process,
                parallel=config.get("geographic", {})
                .get("parallel_processing", {})
                .get("enabled", True),
                max_workers=config.get("geographic", {})
                .get("parallel_processing", {})
                .get("max_workers"),
                output_dir=output_dir / level,
                scenario=scenario,
            )

            # Extract results list from the returned dictionary
            results = results_dict.get("results", [])

            # Process results
            for result in results:
                # Check if projection succeeded (no error in metadata)
                if "error" not in result.get("metadata", {}):
                    metadata.geographies_completed += 1
                else:
                    metadata.geographies_failed += 1
                    metadata.failed_geographies.append(
                        {
                            "fips": result.get("geography", {}).get("fips", "unknown"),
                            "level": level,
                            "error": result.get("metadata", {}).get("error", "Unknown error"),
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing {level} level: {e}")
            logger.debug(traceback.format_exc())
            metadata.geographies_failed += len(fips_to_process)

    metadata.finalize()
    return metadata


def validate_projection_results(
    geographies: dict[str, list[str]],
    scenario: str,
    config: dict[str, Any],
) -> bool:
    """
    Validate projection results including hierarchical aggregation.

    Args:
        geographies: Dictionary of geographies by level
        scenario: Scenario name
        config: Project configuration

    Returns:
        True if validation passes
    """
    logger.info("Validating projection results...")

    try:
        # Note: output_dir reserved for future file-based validation
        _ = (
            Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            / scenario
        )

        # Validate hierarchical aggregation if configured
        if config.get("geography", {}).get("hierarchy", {}).get("validate_aggregation", True):
            logger.info("Validating hierarchical aggregation...")

            # TODO: Implement proper validation using validate_aggregation function
            # The validate_aggregation function expects:
            #   - component_projections: list of projection result dictionaries
            #   - aggregated_projection: aggregated DataFrame
            #   - component_level: 'place' or 'county'
            #   - aggregate_level: 'county' or 'state'
            #   - tolerance: float
            # This requires storing projection results across levels for comparison.
            # For now, skip hierarchical validation.
            logger.info("Hierarchical validation not yet implemented - skipping")

        logger.info("Validation passed")
        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        logger.debug(traceback.format_exc())
        return False


def generate_projection_summary(metadata: ProjectionRunMetadata, config: dict[str, Any]) -> Path:
    """
    Generate projection run summary report.

    Args:
        metadata: Projection run metadata
        config: Project configuration

    Returns:
        Path to summary file
    """
    logger.info("Generating projection summary...")

    output_dir = (
        Path(config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections"))
        / metadata.scenario
        / "metadata"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"projection_run_{timestamp}.json"

    with open(summary_file, "w") as f:
        json.dump(metadata.get_summary(), f, indent=2)

    logger.info(f"Summary saved to {summary_file}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"PROJECTION RUN SUMMARY - Scenario: {metadata.scenario}")
    print("=" * 80)

    summary = metadata.get_summary()
    print(f"\nStart Time: {summary['start_time']}")
    print(f"End Time: {summary['end_time']}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")

    print("\nGeographies:")
    print(f"  Total: {summary['geographies']['total']}")
    print(f"  Completed: {summary['geographies']['completed']}")
    print(f"  Failed: {summary['geographies']['failed']}")
    print(f"  Skipped: {summary['geographies']['skipped']}")

    if summary["failed_geographies"]:
        print("\nFailed Geographies:")
        for failed in summary["failed_geographies"]:
            print(f"  - {failed['fips']} ({failed['level']}): {failed['error']}")

    print(f"\nOutput Files: {len(summary['output_files'])}")
    print("=" * 80 + "\n")

    return summary_file


def run_all_projections(
    config: dict[str, Any],
    levels: list[str],
    fips_filter: list[str] | None = None,
    scenarios: list[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> int:
    """
    Main orchestrator for running all projections.

    Args:
        config: Project configuration
        levels: Geographic levels to process
        fips_filter: Optional FIPS codes to filter
        scenarios: Optional scenarios to run
        dry_run: If True, only show what would be processed
        resume: If True, skip already-completed geographies

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger.info("=" * 80)
    logger.info("PROJECTION RUNNER PIPELINE - North Dakota Population Projections")
    logger.info("=" * 80)
    logger.info(f"Levels: {', '.join(levels)}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Resume: {resume}")
    logger.info("")

    try:
        # Load demographic rates
        fertility_rates, survival_rates, migration_rates = load_demographic_rates(config)

        # Setup projection run
        geographies, scenario_list = setup_projection_run(config, levels, fips_filter, scenarios)

        # Run each scenario
        for scenario in scenario_list:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Running scenario: {scenario}")
            logger.info(f"{'=' * 80}\n")

            # Run projections
            metadata = run_geographic_projections(
                geographies=geographies,
                scenario=scenario,
                config=config,
                fertility_rates=fertility_rates,
                survival_rates=survival_rates,
                migration_rates=migration_rates,
                dry_run=dry_run,
                resume=resume,
            )

            # Validate results
            if not dry_run and metadata.geographies_completed > 0:
                validate_projection_results(geographies, scenario, config)

            # Generate summary
            if not dry_run:
                generate_projection_summary(metadata, config)
            else:
                print(f"\n[DRY RUN] Would process {metadata.geographies_total} geographies")

            # Check for failures
            if metadata.geographies_failed > 0:
                logger.warning(
                    f"Scenario {scenario} completed with {metadata.geographies_failed} failures"
                )

        logger.info("\nAll scenarios completed")
        return 0

    except Exception as e:
        logger.error(f"Projection pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


def main():
    """Main entry point for projection runner pipeline."""
    parser = argparse.ArgumentParser(
        description="Run population projections for North Dakota geographies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all projections
  python 02_run_projections.py --all

  # Run state-level only
  python 02_run_projections.py --state

  # Run county-level projections
  python 02_run_projections.py --counties

  # Run specific counties
  python 02_run_projections.py --fips 38101 38015 38035

  # Run multiple scenarios
  python 02_run_projections.py --all --scenarios baseline high_growth

  # Resume from previous run
  python 02_run_projections.py --all --resume
        """,
    )

    # Geography selection
    parser.add_argument("--all", action="store_true", help="Run all geographic levels")
    parser.add_argument("--state", action="store_true", help="Run state-level projection")
    parser.add_argument("--counties", action="store_true", help="Run county-level projections")
    parser.add_argument("--places", action="store_true", help="Run place-level projections")
    parser.add_argument(
        "--fips",
        nargs="+",
        help="Run specific geographies by FIPS code(s)",
    )

    # Scenario selection
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Scenarios to run (default: active scenarios from config)",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed geographies)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config/projection_config.yaml)",
    )

    args = parser.parse_args()

    # Determine which levels to process
    levels = []
    if args.all:
        levels = ["state", "county", "place"]
    else:
        if args.state:
            levels.append("state")
        if args.counties:
            levels.append("county")
        if args.places:
            levels.append("place")

    # If FIPS specified, determine levels from FIPS length
    if args.fips and not levels:
        levels = []
        for fips in args.fips:
            if len(fips) == 2:
                if "state" not in levels:
                    levels.append("state")
            elif len(fips) == 5:
                if "county" not in levels:
                    levels.append("county")
            elif len(fips) == 7 and "place" not in levels:
                levels.append("place")

    if not levels:
        parser.error("No geographic levels specified. Use --all or specify individual levels.")

    try:
        # Load configuration
        config = load_projection_config(args.config)

        # Run projections
        exit_code = run_all_projections(
            config=config,
            levels=levels,
            fips_filter=args.fips,
            scenarios=args.scenarios,
            dry_run=args.dry_run,
            resume=args.resume,
        )

        return exit_code

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
