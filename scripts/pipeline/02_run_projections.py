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

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import json
import traceback

import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.geographic.multi_geography import (
    run_single_geography_projection,
    run_multiple_geography_projections,
    aggregate_county_to_state,
    aggregate_place_to_county,
    validate_aggregation,
)
from cohort_projections.geographic.geography_loader import (
    load_geography_list,
    load_nd_counties,
    get_geography_name,
)
from cohort_projections.utils.config_loader import load_projection_config
from cohort_projections.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__, log_level="INFO")


class ProjectionRunMetadata:
    """Container for projection run metadata."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.geographies_total = 0
        self.geographies_completed = 0
        self.geographies_failed = 0
        self.geographies_skipped = 0
        self.failed_geographies: List[Dict[str, str]] = []
        self.output_files: List[Path] = []

    def finalize(self):
        """Finalize the metadata."""
        self.end_time = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get metadata summary."""
        return {
            "scenario": self.scenario,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
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


def get_completed_geographies(output_dir: Path, scenario: str) -> Set[str]:
    """
    Get set of already-completed geographies for resume capability.

    Args:
        output_dir: Output directory to check
        scenario: Scenario name

    Returns:
        Set of completed FIPS codes
    """
    completed = set()
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


def load_demographic_rates(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed demographic rates.

    Args:
        config: Project configuration

    Returns:
        Tuple of (fertility_rates, survival_rates, migration_rates)

    Raises:
        FileNotFoundError: If required rate files not found
    """
    logger.info("Loading processed demographic rates...")

    pipeline_config = config.get("pipeline", {}).get("data_processing", {})

    fertility_file = Path(pipeline_config.get("fertility", {}).get("output_file", ""))
    survival_file = Path(pipeline_config.get("survival", {}).get("output_file", ""))
    migration_file = Path(pipeline_config.get("migration", {}).get("output_file", ""))

    if not fertility_file.exists():
        raise FileNotFoundError(f"Fertility rates not found: {fertility_file}")
    if not survival_file.exists():
        raise FileNotFoundError(f"Survival rates not found: {survival_file}")
    if not migration_file.exists():
        raise FileNotFoundError(f"Migration rates not found: {migration_file}")

    fertility_rates = pd.read_parquet(fertility_file)
    survival_rates = pd.read_parquet(survival_file)
    migration_rates = pd.read_parquet(migration_file)

    logger.info(f"Loaded fertility rates: {len(fertility_rates):,} records")
    logger.info(f"Loaded survival rates: {len(survival_rates):,} records")
    logger.info(f"Loaded migration rates: {len(migration_rates):,} records")

    return fertility_rates, survival_rates, migration_rates


def load_base_population(config: Dict[str, Any], fips: str) -> pd.DataFrame:
    """
    Load base year population for a geography.

    Args:
        config: Project configuration
        fips: FIPS code

    Returns:
        Base population DataFrame

    Note:
        This is a placeholder - actual implementation would load from
        processed base population files
    """
    # TODO: Implement actual base population loading
    # For now, return empty DataFrame as placeholder
    logger.warning(f"Base population loading not yet implemented for {fips}")
    return pd.DataFrame()


def setup_projection_run(
    config: Dict[str, Any],
    levels: List[str],
    fips_filter: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
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

    geographies = {"state": [], "county": [], "place": []}

    # Determine scenarios
    if scenarios is None:
        # Get active scenarios from config
        all_scenarios = config.get("scenarios", {})
        scenarios = [
            name for name, settings in all_scenarios.items() if settings.get("active", False)
        ]
        if not scenarios:
            # Fallback to pipeline config
            scenarios = config.get("pipeline", {}).get("projection", {}).get("scenarios", ["baseline"])

    logger.info(f"Scenarios to run: {', '.join(scenarios)}")

    # State level
    if "state" in levels:
        state_fips = config.get("geography", {}).get("state", "38")
        geographies["state"] = [state_fips]
        logger.info(f"State: {state_fips}")

    # County level
    if "county" in levels:
        county_config = config.get("geography", {}).get("counties", {})
        mode = county_config.get("mode", "all")

        if fips_filter:
            # Use provided FIPS filter (only counties)
            county_fips = [f for f in fips_filter if len(f) == 5]
            geographies["county"] = county_fips
        elif mode == "all":
            # Load all counties
            counties_df = load_nd_counties(config)
            geographies["county"] = counties_df["fips"].tolist()
        elif mode == "list":
            geographies["county"] = county_config.get("fips_codes", [])
        elif mode == "threshold":
            # Load counties above population threshold
            counties_df = load_nd_counties(config)
            min_pop = county_config.get("min_population", 1000)
            geographies["county"] = counties_df[
                counties_df["population"] >= min_pop
            ]["fips"].tolist()

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
            # Load places based on mode
            places_df = load_geography_list(
                level="place",
                config=config,
                mode=mode,
                min_population=place_config.get("min_population", 500),
                fips_list=place_config.get("fips_codes", []),
            )
            geographies["place"] = places_df["fips"].tolist()

        logger.info(f"Places: {len(geographies['place'])} to process")

    return geographies, scenarios


def run_geographic_projections(
    geographies: Dict[str, List[str]],
    scenario: str,
    config: Dict[str, Any],
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

    # Get output directory
    output_dir = Path(
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    ) / scenario

    # Get completed geographies for resume
    completed = set()
    if resume:
        completed = get_completed_geographies(output_dir.parent, scenario)
        logger.info(f"Resume mode: {len(completed)} geographies already completed")

    # Process each level
    for level, fips_list in geographies.items():
        if not fips_list:
            continue

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
            results = run_multiple_geography_projections(
                fips_codes=fips_to_process,
                level=level,
                fertility_rates=fertility_rates,
                survival_rates=survival_rates,
                migration_rates=migration_rates,
                config=config,
                output_dir=output_dir / level,
                parallel=config.get("geographic", {}).get("parallel_processing", {}).get("enabled", True),
                max_workers=config.get("geographic", {}).get("parallel_processing", {}).get("max_workers"),
            )

            # Process results
            for result in results:
                if result.get("success", False):
                    metadata.geographies_completed += 1
                    if result.get("output_file"):
                        metadata.output_files.append(Path(result["output_file"]))
                else:
                    metadata.geographies_failed += 1
                    metadata.failed_geographies.append({
                        "fips": result.get("fips", "unknown"),
                        "level": level,
                        "error": result.get("error", "Unknown error"),
                    })

        except Exception as e:
            logger.error(f"Error processing {level} level: {e}")
            logger.debug(traceback.format_exc())
            metadata.geographies_failed += len(fips_to_process)

    metadata.finalize()
    return metadata


def validate_projection_results(
    geographies: Dict[str, List[str]],
    scenario: str,
    config: Dict[str, Any],
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
        output_dir = Path(
            config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
        ) / scenario

        # Validate hierarchical aggregation if configured
        if config.get("geography", {}).get("hierarchy", {}).get("validate_aggregation", True):
            logger.info("Validating hierarchical aggregation...")

            # County to state aggregation
            if geographies.get("county") and geographies.get("state"):
                is_valid = validate_aggregation(
                    lower_level_dir=output_dir / "county",
                    upper_level_dir=output_dir / "state",
                    lower_fips_codes=geographies["county"],
                    upper_fips_code=geographies["state"][0],
                    tolerance=config.get("geography", {}).get("hierarchy", {}).get("aggregation_tolerance", 0.01),
                )
                if not is_valid:
                    logger.warning("County to state aggregation validation failed")
                    return False

        logger.info("Validation passed")
        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        logger.debug(traceback.format_exc())
        return False


def generate_projection_summary(
    metadata: ProjectionRunMetadata, config: Dict[str, Any]
) -> Path:
    """
    Generate projection run summary report.

    Args:
        metadata: Projection run metadata
        config: Project configuration

    Returns:
        Path to summary file
    """
    logger.info("Generating projection summary...")

    output_dir = Path(
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    ) / metadata.scenario / "metadata"

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    print(f"\nGeographies:")
    print(f"  Total: {summary['geographies']['total']}")
    print(f"  Completed: {summary['geographies']['completed']}")
    print(f"  Failed: {summary['geographies']['failed']}")
    print(f"  Skipped: {summary['geographies']['skipped']}")

    if summary["failed_geographies"]:
        print(f"\nFailed Geographies:")
        for failed in summary["failed_geographies"]:
            print(f"  - {failed['fips']} ({failed['level']}): {failed['error']}")

    print(f"\nOutput Files: {len(summary['output_files'])}")
    print("=" * 80 + "\n")

    return summary_file


def run_all_projections(
    config: Dict[str, Any],
    levels: List[str],
    fips_filter: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
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
        geographies, scenario_list = setup_projection_run(
            config, levels, fips_filter, scenarios
        )

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
    parser.add_argument(
        "--all", action="store_true", help="Run all geographic levels"
    )
    parser.add_argument(
        "--state", action="store_true", help="Run state-level projection"
    )
    parser.add_argument(
        "--counties", action="store_true", help="Run county-level projections"
    )
    parser.add_argument(
        "--places", action="store_true", help="Run place-level projections"
    )
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
            elif len(fips) == 7:
                if "place" not in levels:
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
