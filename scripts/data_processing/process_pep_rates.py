#!/usr/bin/env python3
"""
Process PEP county migration data into age/sex/race-specific migration rates.

Phase 3 of ADR-035 implementation: Takes harmonized Census PEP county components
of change data and produces scenario-specific migration rate tables for each
North Dakota county.

Usage:
    python scripts/data_processing/process_pep_rates.py
    python scripts/data_processing/process_pep_rates.py --scenarios baseline low high
    python scripts/data_processing/process_pep_rates.py --pep-path data/processed/pep_county_components_2000_2024.parquet

Author: Generated for ADR-035 Phase 3
Date: 2026-02-12
"""

import argparse
import sys
from pathlib import Path

from cohort_projections.data.process.migration_rates import process_pep_migration_rates
from cohort_projections.utils import load_projection_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process PEP county migration data into migration rate tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --scenarios baseline low high\n"
            "  %(prog)s --pep-path data/processed/pep_county_components_2000_2024.parquet\n"
            "  %(prog)s --as-rates\n"
        ),
    )

    parser.add_argument(
        "--pep-path",
        type=Path,
        default=None,
        help=(
            "Path to PEP county components parquet file. "
            "Default: data/processed/pep_county_components_2000_2024.parquet"
        ),
    )
    parser.add_argument(
        "--population-path",
        type=Path,
        default=None,
        help=(
            "Path to base population parquet file. "
            "Default: resolved from config/projection_config.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: data/processed/migration/",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "low", "high"],
        help="Scenarios to generate. Default: baseline",
    )
    parser.add_argument(
        "--as-rates",
        action="store_true",
        default=False,
        help="Express migration as rates instead of absolute numbers.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve default paths for PEP data, population, and output directory.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (pep_path, population_path, output_dir).
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # PEP data path
    if args.pep_path is not None:
        pep_path = args.pep_path
    else:
        pep_path = project_root / "data" / "processed" / "pep_county_components_2000_2024.parquet"

    # Population path: try to resolve from config, then common defaults
    if args.population_path is not None:
        population_path = args.population_path
    else:
        # Try to resolve from config
        try:
            config = load_projection_config()
            pop_file = (
                config.get("pipeline", {})
                .get("data_processing", {})
                .get("population", {})
                .get("output_file", None)
            )
            if pop_file:
                population_path = project_root / pop_file
            else:
                # Fall back to common locations
                population_path = project_root / "data" / "processed" / "base_population.parquet"
        except Exception:
            population_path = project_root / "data" / "processed" / "base_population.parquet"

    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = project_root / "data" / "processed" / "migration"

    return pep_path, population_path, output_dir


def main() -> int:
    """Main entry point for PEP migration rate processing."""
    args = parse_args()

    print("=" * 70)
    print("PEP MIGRATION RATE PROCESSING")
    print("ADR-035 Phase 3: Rate Calculation and Age/Sex Allocation")
    print("=" * 70)

    # Resolve paths
    pep_path, population_path, output_dir = resolve_paths(args)

    print(f"\nPEP data:       {pep_path}")
    print(f"Population:     {population_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Scenarios:      {args.scenarios}")
    print(f"As rates:       {args.as_rates}")

    # Validate inputs exist
    if not pep_path.exists():
        print(f"\nERROR: PEP data file not found: {pep_path}")
        print("Run Phase 1 extraction first:")
        print("  python scripts/data_processing/extract_pep_county_migration.py")
        return 1

    if not population_path.exists():
        print(f"\nERROR: Population file not found: {population_path}")
        print("Ensure base population data has been processed.")
        return 1

    # Run processing
    print(f"\n{'=' * 70}")
    print("Processing...")
    print(f"{'=' * 70}\n")

    results = process_pep_migration_rates(
        pep_path=pep_path,
        population_path=population_path,
        output_dir=output_dir,
        scenarios=args.scenarios,
        as_rates=args.as_rates,
    )

    # Print summary
    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE - SUMMARY")
    print(f"{'=' * 70}")

    migration_col = "migration_rate" if args.as_rates else "net_migration"

    for scenario_name, scenario_df in results.items():
        total_net = scenario_df[migration_col].sum()
        n_counties = scenario_df["county_fips"].nunique()
        n_rows = len(scenario_df)

        county_totals = scenario_df.groupby("county_fips")[migration_col].sum()
        positive_counties = int((county_totals > 0).sum())
        negative_counties = int((county_totals < 0).sum())

        print(f"\n  Scenario: {scenario_name}")
        print(f"    Rows: {n_rows:,}")
        print(f"    Counties: {n_counties}")
        print(f"    Total net migration: {total_net:+,.0f}")
        print(f"    Counties gaining population: {positive_counties}")
        print(f"    Counties losing population: {negative_counties}")

    print(f"\nOutput files saved to: {output_dir}")
    for scenario_name in results:
        print(f"  - migration_rates_pep_{scenario_name}.parquet")
        print(f"  - migration_rates_pep_{scenario_name}.csv")

    print(f"\n{'=' * 70}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
