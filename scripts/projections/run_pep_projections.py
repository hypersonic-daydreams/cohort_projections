#!/usr/bin/env python3
"""
Run PEP-Based Population Projections for All 53 North Dakota Counties.

Production deployment script for ADR-035 Phase 5. Runs cohort-component
projections using Census PEP migration data instead of IRS county flows.

The PEP method provides per-county, regime-weighted migration rates that
incorporate both domestic and international migration over 24 years of
history (2000-2024), addressing the ~74K-80K person divergence identified
in the IRS-based projections.

Usage:
    # Run baseline scenario (default)
    python scripts/projections/run_pep_projections.py

    # Run multiple scenarios
    python scripts/projections/run_pep_projections.py --scenarios baseline high_growth low_growth

    # Dry run (show what would be processed)
    python scripts/projections/run_pep_projections.py --dry-run

    # Custom output directory
    python scripts/projections/run_pep_projections.py --output-dir data/projections/pep_v2
"""

import argparse
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.utils import load_projection_config  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

# Scenario labels for PEP rate files
PEP_SCENARIO_FILE_MAP = {
    "baseline": "baseline",
    "high_growth": "high",
    "low_growth": "low",
}

# Key years for comparison reporting
REPORT_YEARS = [2025, 2035, 2045]


def _resolve_pipeline_module() -> Any:
    """Import the pipeline module for load/run functions.

    Returns the 02_run_projections module so callers can access
    ``load_demographic_rates``, ``run_geographic_projections``, etc.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_projections",
        project_root / "scripts" / "pipeline" / "02_run_projections.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load 02_run_projections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run PEP-based population projections for all 53 ND counties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline scenario only
  python scripts/projections/run_pep_projections.py

  # Multiple scenarios
  python scripts/projections/run_pep_projections.py --scenarios baseline high_growth low_growth

  # Dry run
  python scripts/projections/run_pep_projections.py --dry-run
        """,
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline"],
        help="Scenarios to run (default: baseline). Options: baseline, high_growth, low_growth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for projection results (default: data/projections/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running projections",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to projection config YAML (default: config/projection_config.yaml)",
    )

    return parser.parse_args()


def verify_pep_rate_files(scenarios: list[str]) -> dict[str, Path]:
    """Verify that PEP migration rate files exist for requested scenarios.

    Args:
        scenarios: List of scenario names to check.

    Returns:
        Dictionary mapping scenario name to verified file path.

    Raises:
        FileNotFoundError: If any required rate file is missing.
    """
    rate_dir = project_root / "data" / "processed" / "migration"
    verified: dict[str, Path] = {}

    for scenario in scenarios:
        pep_label = PEP_SCENARIO_FILE_MAP.get(scenario, "baseline")
        rate_file = rate_dir / f"migration_rates_pep_{pep_label}.parquet"

        if not rate_file.exists():
            raise FileNotFoundError(
                f"PEP migration rate file not found for scenario '{scenario}': {rate_file}\n"
                f"Run the PEP rate processing pipeline first:\n"
                f"  python scripts/pipeline/01_process_demographic_data.py"
            )

        df = pd.read_parquet(rate_file)
        n_counties = df["county_fips"].nunique() if "county_fips" in df.columns else 0
        n_rows = len(df)
        logger.info(
            f"Verified PEP rates for '{scenario}': {rate_file.name} "
            f"({n_rows:,} rows, {n_counties} counties)"
        )
        verified[scenario] = rate_file

    return verified


def verify_config_uses_pep(config: dict[str, Any]) -> None:
    """Verify that the config is set to use PEP migration method.

    Args:
        config: Loaded projection configuration.

    Raises:
        ValueError: If migration method is not PEP_components.
    """
    migration_method = (
        config.get("rates", {})
        .get("migration", {})
        .get("domestic", {})
        .get("method", "IRS_county_flows")
    )

    if migration_method != "PEP_components":
        raise ValueError(
            f"Config migration method is '{migration_method}', expected 'PEP_components'.\n"
            f"Update config/projection_config.yaml:\n"
            f"  rates.migration.domestic.method: PEP_components"
        )

    logger.info(f"Config confirmed: migration method = {migration_method}")


def load_irs_baseline_totals() -> pd.DataFrame | None:
    """Load existing IRS-based projection totals for comparison.

    Looks for state-level or aggregated county results from previous
    IRS-based runs. Returns None if not found.

    Returns:
        DataFrame with year and total_population columns, or None.
    """
    irs_dir = project_root / "data" / "projections" / "baseline" / "county"

    if not irs_dir.exists():
        logger.info("No IRS baseline results found for comparison")
        return None

    parquet_files = list(irs_dir.glob("*.parquet"))
    if not parquet_files:
        logger.info("No IRS baseline county parquet files found")
        return None

    logger.info(f"Loading IRS baseline results from {len(parquet_files)} county files...")

    county_dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            county_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {pf.name}: {e}")

    if not county_dfs:
        return None

    all_counties = pd.concat(county_dfs, ignore_index=True)

    # Aggregate to state totals by year
    if "year" in all_counties.columns and "population" in all_counties.columns:
        state_totals = (
            all_counties.groupby("year")["population"]
            .sum()
            .reset_index()
            .rename(columns={"population": "total_population"})
        )
        logger.info(f"Loaded IRS baseline: {len(state_totals)} years of state totals")
        return state_totals

    return None


def aggregate_county_results(output_dir: Path, scenario: str) -> pd.DataFrame | None:
    """Aggregate county-level projection results to state totals.

    Args:
        output_dir: Base output directory for projections.
        scenario: Scenario name.

    Returns:
        DataFrame with year and total_population columns, or None.
    """
    county_dir = output_dir / scenario / "county"
    if not county_dir.exists():
        return None

    parquet_files = list(county_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    county_dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            county_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {pf.name}: {e}")

    if not county_dfs:
        return None

    all_counties = pd.concat(county_dfs, ignore_index=True)

    if "year" in all_counties.columns and "population" in all_counties.columns:
        state_totals = (
            all_counties.groupby("year")["population"]
            .sum()
            .reset_index()
            .rename(columns={"population": "total_population"})
        )
        return state_totals

    return None


def print_comparison_table(
    pep_totals: pd.DataFrame | None,
    irs_totals: pd.DataFrame | None,
) -> None:
    """Print a comparison table: PEP baseline vs IRS baseline at key years.

    Args:
        pep_totals: PEP-based state-level totals by year.
        irs_totals: IRS-based state-level totals by year (may be None).
    """
    print("\n" + "=" * 78)
    print("PEP vs IRS BASELINE COMPARISON (State-Level Totals)")
    print("=" * 78)

    if pep_totals is None:
        print("  No PEP results available for comparison.")
        return

    if irs_totals is not None:
        print(f"\n{'Year':<10} {'PEP Baseline':>15} {'IRS Baseline':>15} {'Difference':>15}")
        print("-" * 58)

        for year in REPORT_YEARS:
            pep_row = pep_totals[pep_totals["year"] == year]
            irs_row = irs_totals[irs_totals["year"] == year]

            pep_val = float(pep_row["total_population"].iloc[0]) if not pep_row.empty else None
            irs_val = float(irs_row["total_population"].iloc[0]) if not irs_row.empty else None

            if pep_val is not None and irs_val is not None:
                diff = pep_val - irs_val
                print(f"{year:<10} {pep_val:>15,.0f} {irs_val:>15,.0f} {diff:>+15,.0f}")
            elif pep_val is not None:
                print(f"{year:<10} {pep_val:>15,.0f} {'N/A':>15} {'N/A':>15}")
            else:
                print(f"{year:<10} {'N/A':>15} {'N/A':>15} {'N/A':>15}")

        print("-" * 58)
    else:
        print("\n  IRS baseline results not available for comparison.")
        print("  PEP baseline state totals:")
        print(f"\n{'Year':<10} {'PEP Baseline':>15}")
        print("-" * 28)
        for year in REPORT_YEARS:
            pep_row = pep_totals[pep_totals["year"] == year]
            if not pep_row.empty:
                pep_val = float(pep_row["total_population"].iloc[0])
                print(f"{year:<10} {pep_val:>15,.0f}")
        print("-" * 28)


def print_county_growth_summary(output_dir: Path, scenario: str) -> None:
    """Print top 5 growing and top 5 declining counties.

    Args:
        output_dir: Base output directory for projections.
        scenario: Scenario name.
    """
    county_dir = output_dir / scenario / "county"
    if not county_dir.exists():
        logger.info("No county results to summarize")
        return

    parquet_files = list(county_dir.glob("*.parquet"))
    if not parquet_files:
        return

    county_changes: list[dict[str, Any]] = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if "year" not in df.columns or "population" not in df.columns:
                continue

            fips = pf.stem.split("_")[0]
            yearly = df.groupby("year")["population"].sum()

            if yearly.empty:
                continue

            base_year = yearly.index.min()
            final_year = yearly.index.max()
            base_pop = float(yearly.iloc[0])
            final_pop = float(yearly.iloc[-1])

            pct_change = (final_pop - base_pop) / base_pop * 100 if base_pop > 0 else 0.0

            county_changes.append(
                {
                    "fips": fips,
                    "base_pop": base_pop,
                    "final_pop": final_pop,
                    "change": final_pop - base_pop,
                    "pct_change": pct_change,
                    "base_year": int(base_year),
                    "final_year": int(final_year),
                }
            )
        except Exception as e:
            logger.warning(f"Could not process {pf.name}: {e}")

    if not county_changes:
        return

    changes_df = pd.DataFrame(county_changes).sort_values("pct_change", ascending=False)

    print("\n" + "=" * 78)
    print(f"COUNTY GROWTH SUMMARY - Scenario: {scenario}")
    print("=" * 78)

    # Top 5 growing
    top_growing = changes_df.head(5)
    print(
        f"\nTop 5 Growing Counties ({top_growing.iloc[0]['base_year']}-{top_growing.iloc[0]['final_year']}):"
    )
    print(f"{'FIPS':<10} {'Base Pop':>12} {'Final Pop':>12} {'Change':>12} {'% Change':>10}")
    print("-" * 58)
    for _, row in top_growing.iterrows():
        print(
            f"{row['fips']:<10} {row['base_pop']:>12,.0f} {row['final_pop']:>12,.0f} "
            f"{row['change']:>+12,.0f} {row['pct_change']:>+9.1f}%"
        )

    # Top 5 declining
    top_declining = changes_df.tail(5).sort_values("pct_change")
    print(
        f"\nTop 5 Declining Counties ({top_declining.iloc[0]['base_year']}-{top_declining.iloc[0]['final_year']}):"
    )
    print(f"{'FIPS':<10} {'Base Pop':>12} {'Final Pop':>12} {'Change':>12} {'% Change':>10}")
    print("-" * 58)
    for _, row in top_declining.iterrows():
        print(
            f"{row['fips']:<10} {row['base_pop']:>12,.0f} {row['final_pop']:>12,.0f} "
            f"{row['change']:>+12,.0f} {row['pct_change']:>+9.1f}%"
        )

    # State total
    total_base = changes_df["base_pop"].sum()
    total_final = changes_df["final_pop"].sum()
    total_change = total_final - total_base
    total_pct = (total_change / total_base * 100) if total_base > 0 else 0.0

    print(
        f"\n{'State Total':<10} {total_base:>12,.0f} {total_final:>12,.0f} "
        f"{total_change:>+12,.0f} {total_pct:>+9.1f}%"
    )
    print("=" * 78)


def run_pep_projections(
    config: dict[str, Any],
    scenarios: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Run PEP-based projections using the existing pipeline.

    Orchestrates the full projection run by:
    1. Verifying PEP rate files exist
    2. Loading demographic rates via the pipeline
    3. Setting up geographies (all 53 counties)
    4. Running projections for each scenario
    5. Printing comparison and summary reports

    Args:
        config: Loaded projection configuration.
        scenarios: List of scenario names to run.
        output_dir: Output directory for results.
        dry_run: If True, show what would run without executing.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    pipeline = _resolve_pipeline_module()

    # Verify config is set to PEP method
    verify_config_uses_pep(config)

    # Verify PEP rate files exist
    verify_pep_rate_files(scenarios)

    if dry_run:
        logger.info("[DRY RUN] Would run PEP projections for scenarios: %s", scenarios)
        logger.info("[DRY RUN] Output directory: %s", output_dir)
        logger.info("[DRY RUN] Config migration method: PEP_components")

        # Show geography counts
        geographies, scenario_list = pipeline.setup_projection_run(
            config,
            levels=["county"],
            scenarios=scenarios,
        )
        n_counties = len(geographies.get("county", []))
        logger.info(
            "[DRY RUN] Would process %d counties for %d scenarios", n_counties, len(scenario_list)
        )
        print(f"\n[DRY RUN] Would process {n_counties} counties x {len(scenario_list)} scenarios")
        return 0

    # Override the output dir in config if custom
    if output_dir != Path(
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    ):
        config.setdefault("pipeline", {}).setdefault("projection", {})["output_dir"] = str(
            output_dir
        )

    # Run projections using existing pipeline orchestration
    exit_code = pipeline.run_all_projections(
        config=config,
        levels=["county"],
        scenarios=scenarios,
        dry_run=False,
        resume=False,
    )

    if exit_code != 0:
        logger.error("Projection pipeline returned non-zero exit code: %d", exit_code)
        return exit_code

    # Post-processing: comparison and summary reports
    for scenario in scenarios:
        # Aggregate PEP results
        pep_totals = aggregate_county_results(output_dir, scenario)

        # Load IRS baseline for comparison (only compare with baseline)
        irs_totals = None
        if scenario == "baseline":
            irs_totals = load_irs_baseline_totals()

        # Print comparison table
        if scenario == "baseline":
            print_comparison_table(pep_totals, irs_totals)

        # Print county growth summary
        print_county_growth_summary(output_dir, scenario)

    return 0


def main() -> int:
    """Main entry point for PEP projection runner."""
    print("=" * 78)
    print("PEP-BASED POPULATION PROJECTIONS - North Dakota (ADR-035)")
    print("=" * 78)
    print(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("Method: Census PEP Components of Change (2000-2024)")
    print()

    args = parse_args()

    try:
        # Load configuration
        config = load_projection_config(args.config)

        # Determine output directory
        output_dir = args.output_dir or Path(
            config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
        )

        logger.info("Scenarios: %s", args.scenarios)
        logger.info("Output directory: %s", output_dir)
        logger.info("Dry run: %s", args.dry_run)

        # Run projections
        exit_code = run_pep_projections(
            config=config,
            scenarios=args.scenarios,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )

        if exit_code == 0:
            print("\n" + "=" * 78)
            print("PEP PROJECTIONS COMPLETE")
            print("=" * 78)
        else:
            print("\n" + "=" * 78)
            print("PEP PROJECTIONS FINISHED WITH ERRORS")
            print("=" * 78)

        return exit_code

    except FileNotFoundError as e:
        logger.error("Missing required file: %s", e)
        return 1
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
