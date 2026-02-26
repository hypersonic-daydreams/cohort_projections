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

Key ADRs and config:
    ADR-004: Core projection engine architecture
    ADR-037: CBO-grounded scenario methodology (amended by ADR-039, ADR-040)
    ADR-041: Census+PUMS hybrid base population distribution
    ADR-054: Bottom-up state derivation (aggregate counties instead of
             independent state projection to avoid Jensen's inequality)
    Config: scenarios.{baseline,restricted_growth,high_growth}
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
    load_county_populations,
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


def _transform_migration_rates(
    df: pd.DataFrame | dict[str, pd.DataFrame], config: dict | None = None
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Transform migration rates to engine-expected format.

    Engine expects: age, sex, race, net_migration or migration_rate

    The input data is county-level net migration. We need to create an
    age-sex-race breakdown using standard migration patterns.

    Accepts either a single DataFrame or a dict of DataFrames keyed by
    county FIPS. When given a dict, transforms each county's rates
    individually and returns a dict.
    """
    # Handle dict case (PEP per-county rates)
    if isinstance(df, dict):
        return {fips: _transform_migration_rates(r) for fips, r in df.items()}  # type: ignore[misc]

    # Check if already in correct format (has age, sex, race/race_ethnicity columns)
    has_race = "race" in df.columns or "race_ethnicity" in df.columns
    if all(col in df.columns for col in ["age", "sex"]) and has_race:
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
    rows = [
        {"age": age, "sex": sex, "race": race, "migration_rate": 0.0}
        for age in ages
        for sex in sexes
        for race in races
    ]

    result = pd.DataFrame(rows)

    return result


def load_demographic_rates(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | dict[str, pd.DataFrame],
    dict[str, dict[int, pd.DataFrame]] | None,
    dict[int, pd.DataFrame] | None,
]:
    """
    Load processed demographic rates from data/processed/ directory.

    Transforms the rates to match the format expected by the projection engine:
    - fertility_rates: [age (int), race, fertility_rate]
    - survival_rates: [age (int), sex, race, survival_rate]
    - migration_rates: [age (int), sex, race, migration_rate] (single DataFrame)
      OR dict[county_fips, DataFrame] when method is PEP_components
    - migration_rates_by_year_by_county: per-county time-varying migration
      (Phase 2 convergence output), or None if not available
    - survival_rates_by_year: time-varying survival rates
      (Phase 3 mortality improvement output), or None if not available

    Args:
        config: Project configuration

    Returns:
        Tuple of (fertility_rates, survival_rates, migration_rates,
                  migration_rates_by_year_by_county, survival_rates_by_year)

    Raises:
        FileNotFoundError: If required rate files not found
    """
    logger.info("Loading processed demographic rates...")

    # Load from data/processed/ directory
    processed_dir = project_root / "data" / "processed"

    fertility_file = processed_dir / "fertility_rates.parquet"
    survival_file = processed_dir / "survival_rates.parquet"

    if not fertility_file.exists():
        raise FileNotFoundError(f"Fertility rates not found: {fertility_file}")
    if not survival_file.exists():
        raise FileNotFoundError(f"Survival rates not found: {survival_file}")

    # Load raw data for fertility and survival
    fertility_rates_raw = pd.read_parquet(fertility_file)
    survival_rates_raw = pd.read_parquet(survival_file)

    logger.info(f"Loaded fertility rates: {len(fertility_rates_raw):,} records")
    logger.info(f"Loaded survival rates: {len(survival_rates_raw):,} records")

    # Determine migration method from config
    migration_method = (
        config.get("rates", {})
        .get("migration", {})
        .get("domestic", {})
        .get("method", "IRS_county_flows")
    )

    migration_rates: pd.DataFrame | dict[str, pd.DataFrame]

    if migration_method == "PEP_components":
        # Load multi-county PEP rates
        pep_path = project_root / Path(
            config.get("pipeline", {})
            .get("data_processing", {})
            .get("migration", {})
            .get("pep_output", "data/processed/migration/migration_rates_pep_baseline.parquet")
        )
        if not pep_path.exists():
            raise FileNotFoundError(f"PEP migration rates not found: {pep_path}")

        migration_rates_raw = pd.read_parquet(pep_path)
        logger.info(f"Loaded PEP migration rates: {len(migration_rates_raw):,} records")

        # Split into per-county dict and transform each county's rates
        migration_rates = {
            str(county_fips): _transform_migration_rates(  # type: ignore[misc]
                group.drop(columns=["county_fips"]).reset_index(drop=True)
            )
            for county_fips, group in migration_rates_raw.groupby("county_fips")
        }
        logger.info(f"Loaded PEP migration rates for {len(migration_rates)} counties")
    else:
        # Existing IRS behavior - load single rate set
        migration_file = processed_dir / "migration_rates.parquet"
        if not migration_file.exists():
            raise FileNotFoundError(f"Migration rates not found: {migration_file}")

        migration_rates_raw = pd.read_parquet(migration_file)
        logger.info(f"Loaded migration rates: {len(migration_rates_raw):,} records")
        migration_rates = _transform_migration_rates(migration_rates_raw)
        logger.info(f"Transformed migration rates: {len(migration_rates):,} records")

    # Transform to engine-expected format
    logger.info("Transforming rates to engine format...")
    fertility_rates = _transform_fertility_rates(fertility_rates_raw)
    survival_rates = _transform_survival_rates(survival_rates_raw)

    logger.info(f"Transformed fertility rates: {len(fertility_rates):,} records")
    logger.info(f"Transformed survival rates: {len(survival_rates):,} records")

    # -----------------------------------------------------------------------
    # Phase 4: Load time-varying rates (convergence + mortality improvement)
    # -----------------------------------------------------------------------

    # Load convergence rates (Phase 2 output)
    migration_rates_by_year_by_county: dict[str, dict[int, pd.DataFrame]] | None = None
    convergence_path = processed_dir / "migration" / "convergence_rates_by_year.parquet"
    if convergence_path.exists():
        convergence_df = pd.read_parquet(convergence_path)
        convergence_meta_path = processed_dir / "migration" / "convergence_metadata.json"
        if _convergence_rates_need_annualization(convergence_meta_path):
            period_years = _estimate_legacy_residual_period_years(config)
            logger.warning(
                "Convergence rates are missing annual-rate metadata; "
                f"annualizing legacy rates using period_years={period_years:.2f}"
            )
            convergence_df = _annualize_legacy_convergence_rates(convergence_df, period_years)
        migration_rates_by_year_by_county = _build_convergence_rate_dicts(convergence_df)
        logger.info(
            f"Loaded convergence rates for {len(migration_rates_by_year_by_county)} counties"
        )
    else:
        logger.info("No convergence rates found — using constant migration rates")

    # Load ND-adjusted survival projections (Phase 3 output)
    survival_rates_by_year: dict[int, pd.DataFrame] | None = None
    survival_proj_path = processed_dir / "mortality" / "nd_adjusted_survival_projections.parquet"
    if survival_proj_path.exists():
        survival_df = pd.read_parquet(survival_proj_path)
        survival_rates_by_year = _build_survival_rates_by_year(survival_df)
        logger.info(f"Loaded mortality improvement for {len(survival_rates_by_year)} years")
    else:
        logger.info("No mortality improvement data found — using constant survival rates")

    return (
        fertility_rates,
        survival_rates,
        migration_rates,
        migration_rates_by_year_by_county,
        survival_rates_by_year,
    )


def _load_scenario_convergence_rates(
    config: dict[str, Any],
    scenario: str,
) -> dict[str, dict[int, pd.DataFrame]] | None:
    """Load scenario-specific convergence rates if a convergence_variant is configured.

    When a scenario has ``convergence_variant`` set (e.g., ``"high"``), this
    function loads the corresponding convergence file (e.g.,
    ``convergence_rates_by_year_high.parquet``) and returns the per-county,
    per-year-offset rate dicts.

    If no ``convergence_variant`` is set, returns ``None`` to signal that the
    caller should use the default (baseline) convergence rates.

    Args:
        config: Project configuration.
        scenario: Scenario name (e.g., ``"high_growth"``).

    Returns:
        Per-county convergence rate dicts, or ``None`` if no variant override.
    """
    scenario_config = config.get("scenarios", {}).get(scenario, {})
    convergence_variant = scenario_config.get("convergence_variant")

    if not convergence_variant:
        return None

    processed_dir = project_root / "data" / "processed"
    variant_path = (
        processed_dir / "migration" / f"convergence_rates_by_year_{convergence_variant}.parquet"
    )

    if not variant_path.exists():
        logger.warning(
            f"Scenario '{scenario}' requests convergence_variant='{convergence_variant}' "
            f"but file not found: {variant_path}. Falling back to baseline convergence rates."
        )
        return None

    logger.info(
        f"Loading scenario-specific convergence rates for '{scenario}' "
        f"(variant={convergence_variant}) from {variant_path.name}"
    )
    convergence_df = pd.read_parquet(variant_path)
    variant_meta_path = (
        processed_dir / "migration" / f"convergence_metadata_{convergence_variant}.json"
    )
    if _convergence_rates_need_annualization(variant_meta_path):
        period_years = _estimate_legacy_residual_period_years(config)
        logger.warning(
            f"Variant convergence rates missing annual-rate metadata; "
            f"annualizing legacy rates using period_years={period_years:.2f}"
        )
        convergence_df = _annualize_legacy_convergence_rates(convergence_df, period_years)
    result = _build_convergence_rate_dicts(convergence_df)
    logger.info(
        f"Loaded {convergence_variant} convergence rates for {len(result)} counties"
    )
    return result


def load_base_population(config: dict[str, Any], fips: str) -> pd.DataFrame:
    """
    Load base year population for a geography.

    For state-level FIPS (2 digits), sums all county populations and applies
    the statewide age-sex-race distribution. For county-level FIPS (5 digits),
    delegates to load_base_population_for_county.

    Note (ADR-054): As of ADR-054, state-level projections are derived
    bottom-up by aggregating county results rather than running an independent
    state projection through the cohort-component engine. The state-level
    base population loading path (2-digit FIPS) is retained for diagnostic
    purposes and for the fallback case where ``--state`` is used without
    ``--counties``.

    Args:
        config: Project configuration
        fips: FIPS code (2-digit state or 5-digit county)

    Returns:
        Base population DataFrame with columns [year, age, sex, race, population]
    """
    if len(str(fips).strip()) <= 2:
        # State-level: sum all county populations
        from cohort_projections.data.load.base_population_loader import (
            load_state_age_sex_race_distribution,
        )

        base_year = config.get("project", {}).get("base_year", 2025)
        county_pops = load_county_populations()
        total_population = county_pops["population"].sum()
        distribution = load_state_age_sex_race_distribution(config=config)

        base_pop = distribution.copy()
        base_pop["population"] = base_pop["proportion"] * total_population
        base_pop["year"] = base_year
        base_pop = base_pop[["year", "age", "sex", "race", "population"]]
        base_pop = base_pop.sort_values(["age", "sex", "race"]).reset_index(drop=True)

        logger.info(
            f"State FIPS {fips}: total population {total_population:,.0f}, "
            f"{len(base_pop)} cohorts"
        )
        return base_pop

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


def _apply_migration_scenario_to_df(df: pd.DataFrame, migration_setting: str) -> pd.DataFrame:
    """
    Apply a scenario migration adjustment to a single DataFrame.

    Args:
        df: Migration rates DataFrame (must already be a copy)
        migration_setting: Scenario migration setting string

    Returns:
        Adjusted DataFrame
    """
    migration_col = "migration_rate" if "migration_rate" in df.columns else "net_migration"

    if migration_setting == "+25_percent" and migration_col in df.columns:
        df[migration_col] = df[migration_col] * 1.25
    elif migration_setting == "-25_percent" and migration_col in df.columns:
        df[migration_col] = df[migration_col] * 0.75
    elif migration_setting == "-15_percent" and migration_col in df.columns:
        df[migration_col] = df[migration_col] * 0.85
    elif migration_setting == "+5_percent" and migration_col in df.columns:
        df[migration_col] = df[migration_col] * 1.05
    elif migration_setting == "-5_percent" and migration_col in df.columns:
        df[migration_col] = df[migration_col] * 0.95
    elif migration_setting == "zero" and migration_col in df.columns:
        df[migration_col] = 0.0

    return df


def apply_scenario_rate_adjustments(
    scenario: str,
    config: dict[str, Any],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame | dict[str, pd.DataFrame],
    migration_rates_by_year_by_county: dict[str, dict[int, pd.DataFrame]] | None = None,
    survival_rates_by_year: dict[int, pd.DataFrame] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | dict[str, pd.DataFrame],
    dict[str, dict[int, pd.DataFrame]] | None,
    dict[int, pd.DataFrame] | None,
]:
    """
    Apply scenario-specific rate adjustments to demographic rates.

    For high growth scenario, this applies:
    - Fertility: +10% multiplier
    - Migration: +25% multiplier
    - Mortality: constant (no change)

    Time-varying rate dicts (migration_rates_by_year_by_county and
    survival_rates_by_year) are passed through unchanged for now.
    Scenario adjustments to time-varying rates are a future enhancement.

    Args:
        scenario: Scenario name (e.g., 'baseline', 'restricted_growth', 'high_growth')
        config: Project configuration
        fertility_rates: Base fertility rates
        survival_rates: Base survival rates
        migration_rates: Base migration rates (DataFrame or dict of DataFrames)
        migration_rates_by_year_by_county: Optional time-varying migration (Phase 4)
        survival_rates_by_year: Optional time-varying survival (Phase 4)

    Returns:
        Tuple of (adjusted_fertility, adjusted_survival, adjusted_migration,
                  migration_rates_by_year_by_county, survival_rates_by_year)
    """
    scenario_config = config.get("scenarios", {}).get(scenario, {})

    if not scenario_config:
        logger.warning(f"Scenario '{scenario}' not found in config, using base rates")
        return (
            fertility_rates,
            survival_rates,
            migration_rates,
            migration_rates_by_year_by_county,
            survival_rates_by_year,
        )

    # Copy rates to avoid modifying originals
    adj_fertility = fertility_rates.copy()
    adj_survival = survival_rates.copy()

    # Copy migration rates (handle both DataFrame and dict)
    adj_migration: pd.DataFrame | dict[str, pd.DataFrame]
    if isinstance(migration_rates, dict):
        adj_migration = {fips: df.copy() for fips, df in migration_rates.items()}
    else:
        adj_migration = migration_rates.copy()

    # Apply fertility adjustment
    fertility_setting = scenario_config.get("fertility", "constant")
    if fertility_setting == "+10_percent":
        logger.info(f"Scenario {scenario}: Applying +10% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 1.10
    elif fertility_setting == "-10_percent":
        logger.info(f"Scenario {scenario}: Applying -10% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 0.90
    elif fertility_setting == "+5_percent":
        logger.info(f"Scenario {scenario}: Applying +5% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 1.05
    elif fertility_setting == "-5_percent":
        logger.info(f"Scenario {scenario}: Applying -5% fertility adjustment")
        adj_fertility["fertility_rate"] = adj_fertility["fertility_rate"] * 0.95

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

    # Dict-based migration scenarios are handled per-year by the engine
    # (ADR-037 time_varying, ADR-050 additive_reduction)
    if isinstance(migration_setting, dict):
        mig_type = migration_setting.get("type", "unknown")
        logger.info(
            f"Scenario {scenario}: {mig_type} migration (handled per-year by engine)"
        )
        # No upfront adjustment needed
    elif migration_setting not in ("recent_average", "constant", "sdc_2024_dampened"):
        if isinstance(adj_migration, dict):
            logger.info(
                f"Scenario {scenario}: Applying '{migration_setting}' migration adjustment "
                f"to {len(adj_migration)} counties"
            )
            for fips in adj_migration:
                adj_migration[fips] = _apply_migration_scenario_to_df(
                    adj_migration[fips], migration_setting
                )
        else:
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
            elif migration_setting == "-15_percent":
                logger.info(f"Scenario {scenario}: Applying -15% migration adjustment")
                if migration_col in adj_migration.columns:
                    adj_migration[migration_col] = adj_migration[migration_col] * 0.85
            elif migration_setting == "+5_percent":
                logger.info(f"Scenario {scenario}: Applying +5% migration adjustment")
                if migration_col in adj_migration.columns:
                    adj_migration[migration_col] = adj_migration[migration_col] * 1.05
            elif migration_setting == "-5_percent":
                logger.info(f"Scenario {scenario}: Applying -5% migration adjustment")
                if migration_col in adj_migration.columns:
                    adj_migration[migration_col] = adj_migration[migration_col] * 0.95
            elif migration_setting == "zero":
                logger.info(f"Scenario {scenario}: Setting migration to zero")
                if migration_col in adj_migration.columns:
                    adj_migration[migration_col] = 0.0

    return (
        adj_fertility,
        adj_survival,
        adj_migration,
        migration_rates_by_year_by_county,
        survival_rates_by_year,
    )


def _create_zero_migration_rates() -> pd.DataFrame:
    """
    Create a zero-migration DataFrame for counties without PEP data.

    Returns a 1,092-row DataFrame (91 ages x 2 sexes x 6 races) with
    zero migration_rate for all age/sex/race combinations, matching the
    structure produced by _transform_migration_rates().
    """
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

    rows = [
        {"age": age, "sex": sex, "race": race, "migration_rate": 0.0}
        for age in ages
        for sex in sexes
        for race in races
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 4: Bridge functions for time-varying rates
# ---------------------------------------------------------------------------


def _convergence_rates_need_annualization(metadata_path: Path) -> bool:
    """Return True when convergence rates should be annualized at load time.

    Convergence files produced before the migration-rate unit fix do not carry
    rate-unit metadata and store multi-year period rates. New files set
    ``rate_unit = "annual_rate"`` in convergence metadata.
    """
    if not metadata_path.exists():
        return True

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("rate_unit") != "annual_rate"
    except Exception as e:
        logger.warning(f"Could not read convergence metadata at {metadata_path}: {e}")
        return True


def _estimate_legacy_residual_period_years(config: dict[str, Any]) -> float:
    """Estimate period length for legacy residual migration rates.

    Legacy convergence files are based on residual periods configured in
    ``rates.migration.domestic.residual.periods``. We use the mean period
    length from config as the annualization exponent denominator.
    """
    periods = (
        config.get("rates", {})
        .get("migration", {})
        .get("domestic", {})
        .get("residual", {})
        .get("periods", [])
    )

    lengths: list[float] = []
    for p in periods:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            start, end = p
            lengths.append(float(end) - float(start))

    if lengths:
        return sum(lengths) / len(lengths)

    return 5.0


def _annualize_legacy_convergence_rates(
    convergence_df: pd.DataFrame, period_years: float
) -> pd.DataFrame:
    """Convert legacy multi-year migration rates to annual rates."""
    if period_years <= 0:
        raise ValueError(f"period_years must be positive, got {period_years}")

    result = convergence_df.copy()
    if "migration_rate" not in result.columns:
        raise ValueError("convergence_df must contain 'migration_rate'")

    clipped = result["migration_rate"].clip(lower=-1.0)
    annualized = (1.0 + clipped) ** (1.0 / period_years) - 1.0
    result["migration_rate"] = annualized.where(clipped > -1.0, -1.0)
    return result


# Standard 6 race/ethnicity categories used by the projection engine
_ENGINE_RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]

# Mapping from 5-year age group labels to single-year age ranges
_AGE_GROUP_RANGES: dict[str, tuple[int, int]] = {
    "0-4": (0, 4),
    "5-9": (5, 9),
    "10-14": (10, 14),
    "15-19": (15, 19),
    "20-24": (20, 24),
    "25-29": (25, 29),
    "30-34": (30, 34),
    "35-39": (35, 39),
    "40-44": (40, 44),
    "45-49": (45, 49),
    "50-54": (50, 54),
    "55-59": (55, 59),
    "60-64": (60, 64),
    "65-69": (65, 69),
    "70-74": (70, 74),
    "75-79": (75, 79),
    "80-84": (80, 84),
    "85+": (85, 90),
}


def expand_5yr_migration_to_engine_format(df: pd.DataFrame) -> pd.DataFrame:
    """Expand 5-year age-group migration rates to engine format.

    Phase 1/2 produce 36 rows per county (18 age groups x 2 sexes).
    The engine expects 1,092 rows per county (91 ages x 2 sexes x 6 races).

    Steps:
    1. Expand each 5-year rate to constituent single-year ages (same rate).
       For "85+", expand to ages 85-90.
    2. Distribute across all 6 race/ethnicity categories (uniform rate).
    3. Ensure column names match engine expectations.

    Args:
        df: DataFrame with columns [age_group, sex, migration_rate].
            Expected shape: 36 rows (18 age groups x 2 sexes).

    Returns:
        DataFrame with columns [age, sex, race, migration_rate].
        Shape: 1,092 rows (91 ages x 2 sexes x 6 races).
    """
    expanded_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        age_group = str(row["age_group"])
        sex = row["sex"]
        rate = row["migration_rate"]

        # Determine single-year age range
        if age_group in _AGE_GROUP_RANGES:
            start_age, end_age = _AGE_GROUP_RANGES[age_group]
        else:
            logger.warning(f"Unknown age group '{age_group}', skipping")
            continue

        # Expand to single-year ages and all races
        expanded_rows.extend(
            {"age": age, "sex": sex, "race": race, "migration_rate": rate}
            for age in range(start_age, end_age + 1)
            for race in _ENGINE_RACES
        )

    result = pd.DataFrame(expanded_rows)
    return result


def _build_convergence_rate_dicts(
    convergence_df: pd.DataFrame,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Restructure convergence output into per-county, per-year-offset dicts.

    Args:
        convergence_df: DataFrame with columns
            [year_offset, county_fips, age_group, sex, migration_rate].

    Returns:
        Dict mapping county_fips -> dict[year_offset -> engine-format DataFrame].
        Each inner DataFrame has 1,092 rows (91 ages x 2 sexes x 6 races).
    """
    result: dict[str, dict[int, pd.DataFrame]] = {}

    for (county_fips, year_offset), group in convergence_df.groupby(["county_fips", "year_offset"]):
        county_key = str(county_fips)
        offset_key = int(year_offset)  # type: ignore[call-overload]

        # Expand 5-year groups to engine format
        engine_df = expand_5yr_migration_to_engine_format(
            group[["age_group", "sex", "migration_rate"]].reset_index(drop=True)
        )

        if county_key not in result:
            result[county_key] = {}
        result[county_key][offset_key] = engine_df

    return result


def _build_survival_rates_by_year(
    survival_df: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Build per-year survival rate DataFrames from Phase 3 output.

    The Phase 3 output has no race column — survival rates are sex/age only.
    The engine expects [age, sex, race, survival_rate], so we expand each
    year's rates across the 6 standard race categories.

    Args:
        survival_df: DataFrame with columns [year, age, sex, survival_rate, source].

    Returns:
        Dict mapping calendar year -> engine-format survival DataFrame.
    """
    result: dict[int, pd.DataFrame] = {}

    for year, group in survival_df.groupby("year"):
        year_key = int(year)  # type: ignore[arg-type]
        base = group[["age", "sex", "survival_rate"]].reset_index(drop=True)

        # Expand across all races (same rate for each race)
        race_expanded_rows: list[pd.DataFrame] = []
        for race in _ENGINE_RACES:
            race_df = base.copy()
            race_df["race"] = race
            race_expanded_rows.append(race_df)

        expanded = pd.concat(race_expanded_rows, ignore_index=True)
        result[year_key] = expanded[["age", "sex", "race", "survival_rate"]]

    return result


def aggregate_county_results_to_state(
    output_dir: Path,
    scenario: str,
    config: dict[str, Any],
) -> bool:
    """
    Derive state-level projection by summing all county projections (ADR-054).

    Instead of running an independent state projection through the cohort-component
    engine (which suffers from Jensen's inequality due to population-weighted averaging
    of nonlinear compound growth rates), this function aggregates county results
    bottom-up. The county projections are the canonical source; the state is simply
    their sum.

    The function reads all county parquet files from ``{output_dir}/{scenario}/county/``,
    concatenates them, groups by ``(year, age, sex, race)`` summing ``population``,
    and writes the result to ``{output_dir}/{scenario}/state/`` using the same file
    naming and metadata conventions as the engine-based outputs.

    Args:
        output_dir: Root projections output directory (e.g. ``data/projections``).
        scenario: Scenario name (e.g. ``"baseline"``).
        config: Project configuration dict.

    Returns:
        True if aggregation succeeded, False otherwise.
    """
    county_dir = output_dir / scenario / "county"
    state_dir = output_dir / scenario / "state"

    if not county_dir.exists():
        logger.error(f"County output directory not found: {county_dir}")
        return False

    # Read all county parquet files
    county_parquets = sorted(county_dir.glob("nd_county_*_projection_*.parquet"))
    if not county_parquets:
        logger.error(f"No county parquet files found in {county_dir}")
        return False

    logger.info(
        f"ADR-054: Aggregating {len(county_parquets)} county projections "
        f"to derive state total for scenario '{scenario}'"
    )

    county_dfs = []
    for pq_file in county_parquets:
        try:
            df = pd.read_parquet(pq_file)
            county_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pq_file.name}: {e}")

    if not county_dfs:
        logger.error("No county parquet files could be read")
        return False

    # Concatenate and aggregate: sum population across counties
    # for each (year, age, sex, race) combination
    all_counties = pd.concat(county_dfs, ignore_index=True)
    state_df = (
        all_counties.groupby(["year", "age", "sex", "race"], as_index=False)["population"]
        .sum()
        .sort_values(["year", "age", "sex", "race"])
        .reset_index(drop=True)
    )

    # Determine base and end years from the data
    base_year = int(state_df["year"].min())
    end_year = int(state_df["year"].max())
    state_fips = config.get("geography", {}).get("state", "38")

    # Create output directory
    state_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet
    parquet_name = f"nd_state_{state_fips}_projection_{base_year}_{end_year}_{scenario}.parquet"
    parquet_path = state_dir / parquet_name
    state_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved state parquet: {parquet_path}")

    # Generate summary CSV (same format as engine outputs)
    _write_state_summary_csv(state_df, state_dir, state_fips, base_year, end_year, scenario)

    # Generate metadata JSON
    base_pop = float(state_df.loc[state_df["year"] == base_year, "population"].sum())
    final_pop = float(state_df.loc[state_df["year"] == end_year, "population"].sum())
    metadata = {
        "geography": {
            "level": "state",
            "fips": state_fips,
            "name": "North Dakota",
            "base_population": base_pop,
        },
        "projection": {
            "base_year": base_year,
            "end_year": end_year,
            "scenario": scenario,
            "processing_date": datetime.now(UTC).isoformat(),
            "method": "bottom_up_county_aggregation",
            "adr": "ADR-054",
        },
        "summary_statistics": {
            "base_population": base_pop,
            "final_population": final_pop,
            "absolute_growth": final_pop - base_pop,
            "growth_rate": (final_pop - base_pop) / base_pop if base_pop > 0 else 0.0,
            "years_projected": end_year - base_year,
        },
        "validation": {
            "negative_populations": int((state_df["population"] < 0).sum()),
            "all_checks_passed": str((state_df["population"] >= 0).all()),
        },
        "aggregation": {
            "county_files_read": len(county_dfs),
            "total_county_files": len(county_parquets),
        },
    }

    meta_name = f"nd_state_{state_fips}_projection_{base_year}_{end_year}_{scenario}_metadata.json"
    meta_path = state_dir / meta_name
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved state metadata: {meta_path}")

    # Write level-aggregate metadata (matches engine output format)
    level_meta = {
        "level": "state",
        "num_geographies": 1,
        "successful": 1,
        "failed": 0,
        "parallel": False,
        "max_workers": 1,
        "total_processing_time_seconds": 0.0,
        "processing_date": datetime.now(UTC).isoformat(),
        "method": "bottom_up_county_aggregation (ADR-054)",
    }
    level_meta_path = state_dir / "states_metadata.json"
    with open(level_meta_path, "w") as f:
        json.dump(level_meta, f, indent=2)

    logger.info(
        f"ADR-054 state aggregation complete: "
        f"{base_pop:,.0f} ({base_year}) -> {final_pop:,.0f} ({end_year}), "
        f"growth {(final_pop - base_pop) / base_pop * 100:.1f}%"
    )

    return True


def _write_state_summary_csv(
    state_df: pd.DataFrame,
    state_dir: Path,
    state_fips: str,
    base_year: int,
    end_year: int,
    scenario: str,
) -> None:
    """
    Write a summary CSV for the aggregated state projection.

    Produces one row per year with total population, sex breakdown, race breakdown,
    median age, dependency ratio, and broad age group populations. This matches the
    format produced by the engine-based projection outputs.

    Args:
        state_df: Aggregated state projection DataFrame with columns
            [year, age, sex, race, population].
        state_dir: Output directory for state files.
        state_fips: State FIPS code (e.g. "38").
        base_year: First projection year.
        end_year: Last projection year.
        scenario: Scenario name.
    """
    summary_rows = []
    for year, year_df in state_df.groupby("year"):
        total_pop = year_df["population"].sum()
        male_pop = year_df.loc[year_df["sex"] == "Male", "population"].sum()
        female_pop = year_df.loc[year_df["sex"] == "Female", "population"].sum()

        # Race breakdown
        race_pops = year_df.groupby("race")["population"].sum()

        # Age groups for dependency ratio
        under_18 = year_df.loc[year_df["age"] < 18, "population"].sum()
        working_age = year_df.loc[
            (year_df["age"] >= 18) & (year_df["age"] < 65), "population"
        ].sum()
        over_65 = year_df.loc[year_df["age"] >= 65, "population"].sum()
        dependency_ratio = (under_18 + over_65) / working_age if working_age > 0 else 0.0

        # Approximate median age: find age where cumulative pop crosses 50%
        age_pop = year_df.groupby("age")["population"].sum().sort_index()
        cumulative = age_pop.cumsum()
        half = total_pop / 2
        median_age = int(cumulative[cumulative >= half].index[0]) if total_pop > 0 else 0

        row = {
            "year": int(year),  # type: ignore[arg-type]
            "total_population": total_pop,
            "male_population": male_pop,
            "female_population": female_pop,
        }
        # Add race columns in a consistent order
        for race_name in sorted(race_pops.index):
            row[f"population_{race_name}"] = race_pops[race_name]

        row["median_age"] = median_age
        row["dependency_ratio"] = dependency_ratio
        row["population_under_18"] = under_18
        row["population_working_age"] = working_age
        row["population_65_plus"] = over_65

        summary_rows.append(row)

    summary_csv = pd.DataFrame(summary_rows)
    csv_name = (
        f"nd_state_{state_fips}_projection_{base_year}_{end_year}_{scenario}_summary.csv"
    )
    csv_path = state_dir / csv_name
    summary_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved state summary CSV: {csv_path}")

    # Also write the level-aggregate summary (matches engine convention)
    level_csv_path = state_dir / "states_summary.csv"
    summary_csv.to_csv(level_csv_path, index=False)


def run_geographic_projections(
    geographies: dict[str, list[str]],
    scenario: str,
    config: dict[str, Any],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame | dict[str, pd.DataFrame],
    dry_run: bool = False,
    resume: bool = False,
    migration_rates_by_year_by_county: dict[str, dict[int, pd.DataFrame]] | None = None,
    survival_rates_by_year: dict[int, pd.DataFrame] | None = None,
) -> ProjectionRunMetadata:
    """
    Execute projections for all geographies in a scenario.

    Args:
        geographies: Dictionary of geographies by level
        scenario: Scenario name
        config: Project configuration
        fertility_rates: Processed fertility rates
        survival_rates: Processed survival rates
        migration_rates: Processed migration rates (DataFrame or dict of per-county DataFrames)
        dry_run: If True, only show what would be processed
        resume: If True, skip already-completed geographies
        migration_rates_by_year_by_county: Optional per-county time-varying migration (Phase 4)
        survival_rates_by_year: Optional time-varying survival rates (Phase 4)

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
    (
        adj_fertility,
        adj_survival,
        adj_migration,
        adj_migration_by_year_by_county,
        adj_survival_by_year,
    ) = apply_scenario_rate_adjustments(
        scenario=scenario,
        config=config,
        fertility_rates=fertility_rates,
        survival_rates=survival_rates,
        migration_rates=migration_rates,
        migration_rates_by_year_by_county=migration_rates_by_year_by_county,
        survival_rates_by_year=survival_rates_by_year,
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

    # -------------------------------------------------------------------
    # ADR-054: Bottom-up state derivation
    #
    # When both "state" and "county" are requested (the normal --all flow),
    # skip the independent state projection and instead derive the state
    # total by summing county results after all counties complete.
    #
    # This avoids Jensen's inequality: population-weighted averaging of
    # county migration rates and running a single state projection
    # produces a ~10.6% discrepancy vs. summing the 53 independent
    # county projections (each of which compounds growth nonlinearly).
    #
    # If "state" is requested WITHOUT "county" (e.g. --state alone), the
    # old independent state projection behavior is preserved.
    # -------------------------------------------------------------------
    state_requested = bool(geographies.get("state"))
    counties_requested = bool(geographies.get("county"))
    derive_state_from_counties = state_requested and counties_requested

    if derive_state_from_counties:
        logger.info(
            "ADR-054: State projection will be derived bottom-up from county "
            "aggregation (skipping independent state engine run)"
        )

    # Process each level
    for level_key, fips_list in geographies.items():
        if not fips_list:
            continue

        # ADR-054: Skip state in the main loop when we will derive it
        # from county results after all counties complete.
        if level_key == "state" and derive_state_from_counties:
            logger.info(
                "Skipping independent state projection (will derive from counties per ADR-054)"
            )
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
            if isinstance(adj_migration, dict):
                # PEP method: use per-county rates, zero migration for missing counties
                default_migration = _create_zero_migration_rates()
                migration_rates_by_geography = {}
                for fips in fips_to_process:
                    if fips in adj_migration:
                        migration_rates_by_geography[fips] = adj_migration[fips]
                    elif len(str(fips).strip()) <= 2:
                        # State-level: compute population-weighted average of county rates
                        county_pops = load_county_populations()
                        pop_map = dict(
                            zip(county_pops["county_fips"], county_pops["population"])
                        )
                        total_pop = county_pops["population"].sum()
                        # Detect the rate column name (net_migration or migration_rate)
                        sample_df = next(iter(adj_migration.values()))
                        rate_col = (
                            "net_migration"
                            if "net_migration" in sample_df.columns
                            else "migration_rate"
                        )
                        weighted_parts = []
                        for cfips, rates_df in adj_migration.items():
                            w = pop_map.get(cfips, 0) / total_pop
                            part = rates_df.copy()
                            part[rate_col] = part[rate_col] * w
                            weighted_parts.append(part)
                        if weighted_parts:
                            state_rates = pd.concat(weighted_parts)
                            group_cols = [
                                c for c in state_rates.columns if c != rate_col
                            ]
                            # Drop non-groupable columns (e.g. processing_date)
                            group_cols = [
                                c
                                for c in group_cols
                                if c in ("age", "sex", "race", "age_group")
                            ]
                            state_rates = (
                                state_rates.groupby(group_cols, as_index=False)[
                                    rate_col
                                ].sum()
                            )
                            migration_rates_by_geography[fips] = state_rates
                        else:
                            migration_rates_by_geography[fips] = default_migration
                        logger.info("Computed population-weighted state migration rates")
                    else:
                        migration_rates_by_geography[fips] = default_migration
                logger.info(
                    f"Using PEP per-county migration rates "
                    f"({len(adj_migration)} counties with data)"
                )
            else:
                # IRS method: same rates for all counties (existing behavior)
                migration_rates_by_geography = dict.fromkeys(fips_to_process, adj_migration)

            # For state-level, compute population-weighted year-by-year convergence rates
            effective_migration_by_year = adj_migration_by_year_by_county
            if (
                level == "state"
                and adj_migration_by_year_by_county
                and any(len(str(f).strip()) <= 2 for f in fips_to_process)
            ):
                from cohort_projections.data.load.base_population_loader import (
                    load_county_populations as _load_county_pops,
                )

                _county_pops = _load_county_pops()
                _pop_map = dict(
                    zip(_county_pops["county_fips"], _county_pops["population"])
                )
                _total_pop = _county_pops["population"].sum()

                for state_fips in [f for f in fips_to_process if len(str(f).strip()) <= 2]:
                    # Collect all year offsets
                    all_offsets: set[int] = set()
                    for yr_dict in adj_migration_by_year_by_county.values():
                        all_offsets.update(yr_dict.keys())

                    state_yr_dict: dict[int, pd.DataFrame] = {}
                    for offset in sorted(all_offsets):
                        weighted_parts = []
                        for cfips, yr_dict in adj_migration_by_year_by_county.items():
                            if offset not in yr_dict:
                                continue
                            w = _pop_map.get(cfips, 0) / _total_pop
                            part = yr_dict[offset].copy()
                            part["migration_rate"] = part["migration_rate"] * w
                            weighted_parts.append(part)
                        if weighted_parts:
                            merged = pd.concat(weighted_parts)
                            merged = (
                                merged.groupby(
                                    [c for c in merged.columns if c != "migration_rate"],
                                    as_index=False,
                                )["migration_rate"]
                                .sum()
                            )
                            state_yr_dict[offset] = merged

                    if effective_migration_by_year is None:
                        effective_migration_by_year = {}
                    effective_migration_by_year[state_fips] = state_yr_dict
                    logger.info(
                        f"Computed state-level year-by-year convergence rates "
                        f"({len(state_yr_dict)} year offsets)"
                    )

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
                migration_rates_by_year_by_geography=effective_migration_by_year,
                survival_rates_by_year=adj_survival_by_year,
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

    # -------------------------------------------------------------------
    # ADR-054: Derive state projection from county aggregation
    # -------------------------------------------------------------------
    if derive_state_from_counties and metadata.geographies_completed > 0:
        logger.info("\nADR-054: Deriving state projection from county aggregation...")
        try:
            success = aggregate_county_results_to_state(
                output_dir=output_dir.parent,  # Pass root projections dir (without scenario)
                scenario=scenario,
                config=config,
            )
            if success:
                metadata.geographies_completed += 1
                logger.info("ADR-054: State projection derived successfully")
            else:
                metadata.geographies_failed += 1
                metadata.failed_geographies.append(
                    {
                        "fips": config.get("geography", {}).get("state", "38"),
                        "level": "state",
                        "error": "Bottom-up county aggregation failed",
                    }
                )
        except Exception as e:
            logger.error(f"ADR-054: State aggregation failed: {e}")
            logger.debug(traceback.format_exc())
            metadata.geographies_failed += 1
            metadata.failed_geographies.append(
                {
                    "fips": config.get("geography", {}).get("state", "38"),
                    "level": "state",
                    "error": str(e),
                }
            )
    elif derive_state_from_counties and metadata.geographies_completed == 0:
        logger.warning(
            "ADR-054: Skipping state aggregation because no county projections succeeded"
        )
        metadata.geographies_failed += 1

    metadata.finalize()
    return metadata


def validate_projection_results(
    geographies: dict[str, list[str]],
    scenario: str,
    config: dict[str, Any],
) -> bool:
    """
    Validate projection results including bottom-up aggregation diagnostics.

    After ADR-054, the state projection is derived by summing county results.
    This function logs diagnostic totals at key years (base year, and each
    decade through the projection horizon) to support review and auditing.

    Also checks for negative populations and logs growth rates.

    Args:
        geographies: Dictionary of geographies by level
        scenario: Scenario name
        config: Project configuration

    Returns:
        True if validation passes
    """
    logger.info("Validating projection results...")

    try:
        output_base = Path(
            config.get("pipeline", {})
            .get("projection", {})
            .get("output_dir", "data/projections")
        )
        scenario_dir = output_base / scenario

        # -----------------------------------------------------------
        # ADR-054 diagnostics: Log county-sum totals at key years
        # -----------------------------------------------------------
        state_dir = scenario_dir / "state"
        state_fips = config.get("geography", {}).get("state", "38")
        state_parquets = sorted(state_dir.glob(f"nd_state_{state_fips}_projection_*.parquet"))

        if state_parquets:
            state_df = pd.read_parquet(state_parquets[0])
            base_year = int(state_df["year"].min())
            end_year = int(state_df["year"].max())

            # Key diagnostic years: base year + each decade + end year
            diagnostic_years = sorted(
                {base_year}
                | set(range(base_year + 10, end_year + 1, 10))
                | {end_year}
            )

            logger.info(f"  ADR-054 diagnostics for scenario '{scenario}':")
            base_pop = None
            for year in diagnostic_years:
                year_pop = float(
                    state_df.loc[state_df["year"] == year, "population"].sum()
                )
                if base_pop is None:
                    base_pop = year_pop
                    growth_str = "(base year)"
                else:
                    growth_pct = (year_pop - base_pop) / base_pop * 100
                    growth_str = f"({growth_pct:+.1f}% from {base_year})"
                logger.info(f"    {year}: {year_pop:>12,.0f}  {growth_str}")

            # Check for negative populations
            neg_count = int((state_df["population"] < 0).sum())
            if neg_count > 0:
                logger.warning(
                    f"  WARNING: {neg_count} negative population values in state projection"
                )
        else:
            logger.info("  No state projection file found for diagnostics")

        # County-level diagnostics: log total counties and total population
        county_dir = scenario_dir / "county"
        if county_dir.exists() and geographies.get("county"):
            county_parquets = sorted(county_dir.glob("nd_county_*_projection_*.parquet"))
            if county_parquets:
                logger.info(
                    f"  County projections: {len(county_parquets)} files in {county_dir}"
                )

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
        # Load demographic rates (including Phase 4 time-varying rates if available)
        (
            fertility_rates,
            survival_rates,
            migration_rates,
            migration_rates_by_year_by_county,
            survival_rates_by_year,
        ) = load_demographic_rates(config)

        # Setup projection run.
        # In dry-run mode, allow a graceful fallback if place reference data
        # is structurally incomplete so entrypoint wiring can still be validated.
        try:
            geographies, scenario_list = setup_projection_run(config, levels, fips_filter, scenarios)
        except ValueError as exc:
            if (
                dry_run
                and "place" in levels
                and "Place data missing required columns" in str(exc)
            ):
                logger.warning(
                    "Dry run fallback: skipping place level due place reference schema issue: "
                    f"{exc}"
                )
                fallback_levels = [level for level in levels if level != "place"]
                geographies, scenario_list = setup_projection_run(
                    config, fallback_levels, fips_filter, scenarios
                )
            else:
                raise

        # Run each scenario
        for scenario in scenario_list:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Running scenario: {scenario}")
            logger.info(f"{'=' * 80}\n")

            # Check for scenario-specific convergence rates (ADR-046)
            scenario_convergence = _load_scenario_convergence_rates(config, scenario)
            effective_convergence = (
                scenario_convergence
                if scenario_convergence is not None
                else migration_rates_by_year_by_county
            )

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
                migration_rates_by_year_by_county=effective_convergence,
                survival_rates_by_year=survival_rates_by_year,
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
