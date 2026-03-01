#!/usr/bin/env python3
"""
Run housing-unit method projections pipeline stage (PP-005 / ADR-060).

This stage produces complementary short-term place projections using
housing units x persons-per-household.  It is a cross-check for the
share-trending place projections from stage 02a, NOT a replacement.

Created: 2026-03-01
ADR: 060 (Housing-Unit Method for Place Projections)
Author: Claude Code / N. Haarstad

Purpose
-------
Consume ACS housing-unit data, fit per-place trend models, and produce
projected populations for the configured projection years.  Outputs are
written as Parquet alongside a JSON metadata sidecar.

Usage:
    python scripts/pipeline/02c_run_housing_unit_projections.py
    python scripts/pipeline/02c_run_housing_unit_projections.py --dry-run
    python scripts/pipeline/02c_run_housing_unit_projections.py --scenarios baseline high_growth

Inputs
------
- data/raw/housing/nd_place_housing_units.csv
    ACS 5-year housing-unit and PPH data (from fetch_census_housing_data.py).
- config/projection_config.yaml
    housing_unit_method config block.

Outputs
-------
- data/projections/{scenario}/place/housing_unit_projections.parquet
    Per-place HU projected population.
- data/projections/{scenario}/place/housing_unit_metadata.json
    Run metadata (timestamp, config, row counts).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.data.process.place_housing_unit_projection import (  # noqa: E402
    run_housing_unit_projections,
)
from cohort_projections.utils import load_projection_config  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")


def _resolve_path(path_value: str | Path, root: Path) -> Path:
    """Resolve a configured path value to an absolute filesystem path."""
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _resolve_scenarios(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    """Resolve scenario list from CLI override or active config entries."""
    if requested:
        return requested

    config_scenarios = config.get("scenarios", {})
    active = [
        name
        for name, settings in config_scenarios.items()
        if isinstance(settings, dict) and settings.get("active", False)
    ]
    if active:
        return active

    fallback = config.get("pipeline", {}).get("projection", {}).get("scenarios", ["baseline"])
    if isinstance(fallback, list) and fallback:
        return [str(s) for s in fallback]
    return ["baseline"]


def _projection_output_root(config: dict[str, Any]) -> Path:
    """Resolve projections root directory from config."""
    hu_cfg = config.get("housing_unit_method", {})
    output_dir = hu_cfg.get(
        "output_dir",
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections"),
    )
    return _resolve_path(str(output_dir), project_root)


def _dry_run_validate(config: dict[str, Any], scenarios: list[str]) -> bool:
    """Validate stage dependencies without writing outputs."""
    hu_cfg = config.get("housing_unit_method", {})
    housing_path = _resolve_path(
        hu_cfg.get("housing_data_path", "data/raw/housing/nd_place_housing_units.csv"),
        project_root,
    )

    ok = True
    if housing_path.exists():
        logger.info(f"[DRY RUN] OK: housing data found: {housing_path}")
    else:
        logger.error(f"[DRY RUN] MISSING: housing data: {housing_path}")
        ok = False

    output_root = _projection_output_root(config)
    for scenario in scenarios:
        place_dir = output_root / scenario / "place"
        logger.info(f"[DRY RUN] Would write HU outputs for {scenario}: {place_dir}")

    return ok


def _write_outputs(
    result: Any,
    config: dict[str, Any],
    scenario: str,
) -> Path:
    """Write HU projection outputs and metadata sidecar."""
    import pandas as pd

    output_root = _projection_output_root(config)
    place_dir = output_root / scenario / "place"
    place_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = place_dir / "housing_unit_projections.parquet"
    if isinstance(result, pd.DataFrame) and not result.empty:
        result.to_parquet(parquet_path, index=False)
        logger.info(f"Wrote {len(result)} rows to {parquet_path}")
    else:
        logger.warning(f"Empty HU results for {scenario}; writing empty parquet")
        pd.DataFrame(
            columns=["place_fips", "year", "hu_projected", "pph_projected", "population_hu", "method"]
        ).to_parquet(parquet_path, index=False)

    # Metadata sidecar
    hu_cfg = config.get("housing_unit_method", {})
    metadata = {
        "stage": "02c_run_housing_unit_projections",
        "scenario": scenario,
        "timestamp": datetime.now(UTC).isoformat(),
        "trend_method": hu_cfg.get("trend_method", "log_linear"),
        "pph_method": hu_cfg.get("pph_method", "hold_last"),
        "projection_years": hu_cfg.get("projection_years", []),
        "row_count": len(result) if isinstance(result, pd.DataFrame) else 0,
        "place_count": (
            result["place_fips"].nunique()
            if isinstance(result, pd.DataFrame) and not result.empty
            else 0
        ),
    }
    meta_path = place_dir / "housing_unit_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Wrote metadata to {meta_path}")

    return parquet_path


def run_stage(
    config: dict[str, Any],
    scenarios: list[str],
    dry_run: bool = False,
) -> int:
    """Run housing-unit projections for each requested scenario."""
    hu_cfg = config.get("housing_unit_method", {})
    if isinstance(hu_cfg, dict) and not hu_cfg.get("enabled", True):
        logger.info("Housing-unit method disabled (`housing_unit_method.enabled: false`); skipping.")
        return 0

    logger.info(f"Scenarios: {', '.join(scenarios)}")

    if dry_run:
        logger.info("Dry run complete; no outputs were written.")
        return 0

    for scenario in scenarios:
        logger.info(f"Running HU projections for scenario: {scenario}")
        result = run_housing_unit_projections(config=config)
        output_path = _write_outputs(result, config, scenario)
        logger.info(
            "Scenario %s complete: places=%d, output=%s",
            scenario,
            result["place_fips"].nunique() if not result.empty else 0,
            output_path,
        )

    logger.info("Housing-unit projection stage completed successfully.")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PP-005 housing-unit method place projection stage",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to projection config YAML (default: config/projection_config.yaml)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario keys to run (default: active scenarios from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dependencies and show planned actions without writing outputs",
    )
    args = parser.parse_args()

    try:
        config = load_projection_config(args.config)
        if not isinstance(config, dict):
            raise ValueError("Projection configuration did not load as a dictionary.")

        scenarios = _resolve_scenarios(config, args.scenarios)
        if not scenarios:
            raise ValueError("No scenarios resolved for HU projection stage.")

        if args.dry_run:
            if _dry_run_validate(config, scenarios):
                logger.info("Dry run checks passed.")
                return 0
            logger.error("Dry run failed due to missing dependencies.")
            return 1

        return run_stage(config=config, scenarios=scenarios, dry_run=False)

    except Exception as exc:
        logger.error(f"Housing-unit projection stage failed: {exc}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
