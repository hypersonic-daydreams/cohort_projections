#!/usr/bin/env python3
"""
Analyze Census PEP migration data using regime-weighted averages.

Phase 2 of ADR-035: classify ND counties, compute per-regime migration
statistics, and produce regime-weighted average net migration suitable
for the projection engine.

Outputs:
    - data/processed/migration_regimes_by_county.csv
    - docs/analysis/migration_regime_analysis.md

Usage:
    python scripts/data_processing/analyze_pep_regimes.py
    python scripts/data_processing/analyze_pep_regimes.py --pep-path data/processed/pep_county_components_2000_2024.parquet
    python scripts/data_processing/analyze_pep_regimes.py --output-dir data/processed

Author: Generated for ADR-035 Phase 2
Date: 2026-02-12
"""

import argparse
import logging
import sys
from pathlib import Path

# Resolve project root for imports when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cohort_projections.data.process.pep_regime_analysis import (  # noqa: E402
    DEFAULT_DAMPENING,
    DEFAULT_REGIME_WEIGHTS,
    MIGRATION_REGIMES,
    calculate_regime_averages,
    calculate_regime_weighted_average,
    classify_counties,
    generate_regime_analysis_report,
    load_pep_preferred_estimates,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Census PEP migration data using regime-weighted averages."
    )
    parser.add_argument(
        "--pep-path",
        type=str,
        default=str(
            PROJECT_ROOT / "data" / "processed" / "pep_county_components_2000_2024.parquet"
        ),
        help="Path to PEP county components parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory for CSV output.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "analysis" / "migration_regime_analysis.md"),
        help="Path for the Markdown analysis report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full regime analysis pipeline."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pep_path = Path(args.pep_path)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load preferred estimates
    logger.info("Step 1: Loading PEP preferred estimates")
    pep_data = load_pep_preferred_estimates(pep_path)
    logger.info(
        f"  Loaded {len(pep_data)} observations, "
        f"{pep_data['county_fips'].nunique()} counties, "
        f"years {pep_data['year'].min()}-{pep_data['year'].max()}"
    )

    # Step 2: Classify counties
    logger.info("Step 2: Classifying counties")
    county_fips_list = sorted(pep_data["county_fips"].unique().tolist())
    classifications = classify_counties(county_fips_list)
    logger.info(f"  Classified {len(classifications)} counties")

    # Step 3: Calculate regime averages
    logger.info("Step 3: Calculating regime averages")
    regime_averages = calculate_regime_averages(pep_data)
    logger.info(f"  Computed {len(regime_averages)} county-regime records")

    # Step 4: Calculate regime-weighted averages
    logger.info("Step 4: Computing regime-weighted averages")
    weighted_averages = calculate_regime_weighted_average(regime_averages)
    logger.info(f"  Computed weighted averages for {len(weighted_averages)} counties")

    # Step 5: Save CSV
    csv_path = output_dir / "migration_regimes_by_county.csv"
    merged = weighted_averages.merge(classifications, on="county_fips", how="left")
    merged.to_csv(csv_path, index=False)
    logger.info(f"Step 5: Saved regime data to {csv_path}")

    # Step 6: Generate report
    logger.info("Step 6: Generating analysis report")
    report_file = generate_regime_analysis_report(
        pep_data=pep_data,
        regime_averages=regime_averages,
        classifications=classifications,
        weighted_averages=weighted_averages,
        output_path=report_path,
    )
    logger.info(f"  Report written to {report_file}")

    # Step 7: Print summary to stdout
    print("=" * 70)
    print("REGIME ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nCounties analysed: {len(weighted_averages)}")
    print(f"Regimes: {list(MIGRATION_REGIMES.keys())}")
    print(f"Weights: {DEFAULT_REGIME_WEIGHTS}")
    print(f"Dampening: {DEFAULT_DAMPENING}")
    print()

    type_summary = merged.groupby("county_type")["weighted_avg_netmig"].agg(
        ["count", "mean", "median", "min", "max"]
    )
    print("Weighted average net migration by county type:")
    print(type_summary.to_string())
    print()

    print(f"State total weighted avg net mig: {merged['weighted_avg_netmig'].sum():,.1f}")
    print()

    print("Output files:")
    print(f"  - {csv_path}")
    print(f"  - {report_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        sys.exit(1)
