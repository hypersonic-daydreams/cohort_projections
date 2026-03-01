#!/usr/bin/env python3
"""
Run place projection pipeline stage (PP-003 IMP-11).

This stage consumes county projection outputs plus the accepted PP-003 backtest
winner and generates county-constrained place projections for each scenario.

Usage:
    python scripts/pipeline/02a_run_place_projections.py
    python scripts/pipeline/02a_run_place_projections.py --scenarios baseline high_growth
    python scripts/pipeline/02a_run_place_projections.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.data.process.place_projection_orchestrator import (  # noqa: E402
    run_place_projections,
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
        return [str(scenario) for scenario in fallback]
    return ["baseline"]


def _load_winner_payload(winner_path: Path) -> dict[str, Any]:
    """Load and validate backtest winner payload from JSON."""
    if not winner_path.exists():
        raise FileNotFoundError(f"Backtest winner file not found: {winner_path}")

    with open(winner_path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError("Backtest winner payload must be a JSON object.")

    if "winner_variant_id" not in payload:
        raise ValueError("Backtest winner payload missing `winner_variant_id`.")

    fitting = payload.get("fitting_method")
    constraint = payload.get("constraint_method")
    if fitting is None or constraint is None:
        logger.warning(
            "Winner payload does not include fitting/constraint methods; "
            "orchestrator will resolve methods from variant ID/default config."
        )

    acceptance = payload.get("acceptance", {})
    if isinstance(acceptance, dict) and acceptance.get("all_scored_tiers_pass_primary") is False:
        logger.warning(
            "Winner payload indicates primary acceptance failure. "
            "Proceeding requires explicit human approval."
        )

    return payload


def _projection_output_root(config: dict[str, Any]) -> Path:
    """Resolve projections root directory from config."""
    output_dir = config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    return _resolve_path(str(output_dir), project_root)


def _dry_run_validate(
    config: dict[str, Any],
    scenarios: list[str],
    winner_path: Path,
) -> bool:
    """Validate stage dependencies without writing outputs."""
    missing: list[Path] = []

    if winner_path.exists():
        logger.info(f"[DRY RUN] OK: winner payload found: {winner_path}")
    else:
        logger.error(f"[DRY RUN] MISSING: winner payload: {winner_path}")
        missing.append(winner_path)

    projection_root = _projection_output_root(config)
    for scenario in scenarios:
        county_dir = projection_root / scenario / "county"
        place_dir = projection_root / scenario / "place"
        if county_dir.exists():
            logger.info(f"[DRY RUN] OK: county inputs found for {scenario}: {county_dir}")
        else:
            logger.error(f"[DRY RUN] MISSING: county inputs for {scenario}: {county_dir}")
            missing.append(county_dir)
        logger.info(f"[DRY RUN] Would write place outputs for {scenario}: {place_dir}")

    return len(missing) == 0


def run_stage(
    config: dict[str, Any],
    scenarios: list[str],
    winner_payload: dict[str, Any],
    dry_run: bool = False,
) -> int:
    """Run place projections for each requested scenario."""
    place_cfg = config.get("place_projections", {})
    if isinstance(place_cfg, dict) and not place_cfg.get("enabled", True):
        logger.info("Place projections disabled (`place_projections.enabled: false`); skipping stage.")
        return 0

    winner_variant = winner_payload.get("winner_variant_id", "unknown")
    logger.info(f"Place projection winner variant: {winner_variant}")
    logger.info(f"Scenarios: {', '.join(scenarios)}")

    if dry_run:
        logger.info("Dry run complete; no place outputs were written.")
        return 0

    for scenario in scenarios:
        logger.info(f"Running place projections for scenario: {scenario}")
        result = run_place_projections(
            scenario=scenario,
            config=config,
            variant_winner=winner_payload,
        )
        logger.info(
            "Scenario %s complete: places=%s, balances=%s, summary=%s",
            scenario,
            result.get("places_processed"),
            result.get("balance_rows"),
            result.get("summary_path"),
        )

    logger.info("Place projection stage completed successfully")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PP-003 place projection stage using accepted backtest winner",
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
        "--winner-file",
        type=Path,
        default=None,
        help=(
            "Path to backtest winner JSON "
            "(default: data/backtesting/place_backtest_results/backtest_winner.json)"
        ),
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
            raise ValueError("No scenarios resolved for place projection stage.")

        default_winner = project_root / "data" / "backtesting" / "place_backtest_results" / "backtest_winner.json"
        winner_path = _resolve_path(args.winner_file if args.winner_file else default_winner, project_root)

        if args.dry_run:
            if _dry_run_validate(config, scenarios, winner_path):
                logger.info("Dry run checks passed")
                return 0
            logger.error("Dry run failed due to missing dependencies")
            return 1

        winner_payload = _load_winner_payload(winner_path)
        return run_stage(
            config=config,
            scenarios=scenarios,
            winner_payload=winner_payload,
            dry_run=False,
        )

    except Exception as exc:
        logger.error(f"Place projection stage failed: {exc}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
