#!/usr/bin/env python3
"""
Run PP-005 rolling-origin cross-validation for place share-trending variants.

Created: 2026-03-01
ADR: 057 (Rolling-Origin Backtests)
Author: Claude Code / nhaarstad

Purpose
-------
Replace the static two-window backtest (IMP-08) with expanding-window
rolling-origin cross-validation.  This provides more rigorous variant
selection evidence by evaluating each candidate variant across multiple
non-overlapping train/test splits whose training window grows over time.

Method
------
1. Load historical place shares, county population, and tier assignments
   from config paths (same inputs as ``run_place_backtest.py``).
2. Generate expanding windows via ``generate_rolling_windows()`` using
   config-driven ``min_train_years`` and ``test_horizon``.
3. For each window x variant, call ``run_single_variant()`` from
   ``place_backtest`` to produce projected / actual DataFrames.
4. Compute per-place metrics, tier aggregates, and variant score per window.
5. Aggregate across windows (mean, median, std, min, max).
6. Select winner using the configured ``acceptance_criteria``
   (default: ``mean_score``) with the same tie-breaking rules as the
   static backtest (A > B, I > II).

Key design decisions
--------------------
- **Expanding (not sliding) windows**: Training always starts at
  ``history_start`` so models benefit from all available history, matching
  how the production model is fit.
- **mean_score default**: Arithmetic mean of per-window scores is the
  primary selection criterion; median is available as an alternative.
- **Reuses place_backtest primitives**: No metric logic is duplicated;
  rolling-origin adds only orchestration and aggregation.

Inputs
------
- ``data/processed/place_shares_2000_2024.parquet`` (from config)
- ``data/processed/geographic/place_county_crosswalk_2020.csv`` (from config)
- ``data/processed/place_population_history_2000_2024.parquet`` (from config)

Output
------
- ``rolling_origin_per_window_scores.csv`` — per-window variant scores
- ``rolling_origin_aggregated_scores.csv`` — cross-window aggregates
- ``rolling_origin_winner.json`` — winner payload

Usage
-----
    python scripts/backtesting/run_rolling_origin_backtest.py
    python scripts/backtesting/run_rolling_origin_backtest.py --output-dir data/backtesting/rolling_origin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.data.process.rolling_origin_backtest import (  # noqa: E402
    aggregate_rolling_metrics,
    build_per_window_summary,
    generate_rolling_windows,
    run_rolling_origin_backtest,
    select_rolling_winner,
)
from cohort_projections.utils import load_projection_config  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

VARIANT_SPECS: dict[str, dict[str, str]] = {
    "A-I": {"fitting_method": "ols", "constraint_method": "proportional"},
    "A-II": {"fitting_method": "ols", "constraint_method": "cap_and_redistribute"},
    "B-I": {"fitting_method": "wls", "constraint_method": "proportional"},
    "B-II": {"fitting_method": "wls", "constraint_method": "cap_and_redistribute"},
}


def _resolve_path(path_text: str) -> Path:
    """Resolve config path relative to repository root."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_fips(value: object, width: int) -> str:
    """Normalize FIPS-like values to zero-padded strings."""
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(width)[-width:]


def _load_share_history(config: dict[str, Any]) -> pd.DataFrame:
    """Load historical place shares from config path."""
    shares_path = _resolve_path(
        config.get("place_projections", {}).get(
            "historical_shares_path",
            "data/processed/place_shares_2000_2024.parquet",
        )
    )
    history = pd.read_parquet(shares_path)
    history["county_fips"] = history["county_fips"].map(lambda v: _normalize_fips(v, 5))
    history["place_fips"] = history["place_fips"].map(
        lambda v: _normalize_fips(v, 7) if pd.notna(v) else pd.NA
    )
    return history


def _load_tier_assignments(config: dict[str, Any]) -> pd.DataFrame:
    """Build place tier assignment table with 2024 population weights."""
    crosswalk_path = _resolve_path(
        config.get("place_projections", {}).get(
            "crosswalk_path",
            "data/processed/geographic/place_county_crosswalk_2020.csv",
        )
    )
    population_path = _resolve_path(
        config.get("place_projections", {}).get(
            "place_population_history_path",
            "data/processed/place_population_history_2000_2024.parquet",
        )
    )

    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk["place_fips"] = crosswalk["place_fips"].map(lambda v: _normalize_fips(v, 7))
    crosswalk["confidence_tier"] = crosswalk["confidence_tier"].astype(str).str.upper()

    place_population = pd.read_parquet(population_path)
    place_population["place_fips"] = place_population["place_fips"].map(
        lambda v: _normalize_fips(v, 7)
    )
    place_population["year"] = pd.to_numeric(place_population["year"], errors="coerce").astype(
        "Int64"
    )
    place_population["population"] = pd.to_numeric(
        place_population["population"], errors="coerce"
    )
    pop_2024 = (
        place_population[place_population["year"] == 2024][["place_fips", "population"]]
        .rename(columns={"population": "population_2024"})
        .dropna(subset=["place_fips"])
        .drop_duplicates(subset=["place_fips"])
    )

    tiers = crosswalk[["place_fips", "confidence_tier"]].drop_duplicates(subset=["place_fips"])
    tiers = tiers.merge(pop_2024, on="place_fips", how="left", validate="one_to_one")
    return tiers


def _build_county_population(history: pd.DataFrame) -> pd.DataFrame:
    """Extract county population totals from share history."""
    if "county_population" not in history.columns:
        raise ValueError("share_history must include `county_population` for backtest runner.")

    county_pop = history[["county_fips", "year", "county_population"]].copy()
    county_pop["year"] = pd.to_numeric(county_pop["year"], errors="coerce").astype("Int64")
    county_pop["county_population"] = pd.to_numeric(
        county_pop["county_population"], errors="coerce"
    )
    county_pop = county_pop.dropna(subset=["county_fips", "year", "county_population"]).copy()
    county_pop["year"] = county_pop["year"].astype(int)
    county_pop = (
        county_pop.groupby(["county_fips", "year"], as_index=False)["county_population"]
        .first()
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    return county_pop


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run PP-005 rolling-origin cross-validation for place backtest variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "projection_config.yaml",
        help="Path to projection config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "backtesting" / "rolling_origin_results",
        help="Directory for rolling-origin output artifacts.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_SPECS.keys()),
        choices=list(VARIANT_SPECS.keys()),
        help="Variants to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without writing outputs.",
    )
    args = parser.parse_args()

    config = load_projection_config(args.config)
    if not isinstance(config, dict):
        raise ValueError("Projection configuration did not load as a dictionary.")

    ro_cfg = config.get("place_projections", {}).get("rolling_origin_backtest", {})
    if not ro_cfg.get("enabled", True):
        logger.info("Rolling-origin backtest is disabled in config; exiting.")
        return 0

    # Load data
    share_history = _load_share_history(config)
    county_population = _build_county_population(share_history)
    tier_assignments = _load_tier_assignments(config)

    # Generate windows
    model_cfg = config.get("place_projections", {}).get("model", {})
    history_start = int(model_cfg.get("history_start", 2000))
    history_end = int(model_cfg.get("history_end", 2024))
    min_train = int(ro_cfg.get("min_train_years", 5))
    test_horizon = int(ro_cfg.get("test_horizon", 5))
    acceptance_criteria = str(ro_cfg.get("acceptance_criteria", "mean_score"))

    windows = generate_rolling_windows(
        history_start=history_start,
        history_end=history_end,
        min_train_years=min_train,
        test_horizon=test_horizon,
    )

    if not windows:
        logger.error(
            "No valid windows for history %d-%d with min_train=%d, horizon=%d.",
            history_start,
            history_end,
            min_train,
            test_horizon,
        )
        return 1

    logger.info("Generated %d rolling-origin windows:", len(windows))
    for i, (ts, te, vs, ve) in enumerate(windows, 1):
        logger.info("  Window %d: train %d-%d, test %d-%d", i, ts, te, vs, ve)

    # Filter to requested variants
    selected_variants = {k: v for k, v in VARIANT_SPECS.items() if k in args.variants}
    if not selected_variants:
        raise ValueError(f"No valid variants selected from: {args.variants}")

    # Run rolling-origin backtest
    per_window_results = run_rolling_origin_backtest(
        share_history=share_history,
        county_pop=county_population,
        tier_assignments=tier_assignments,
        variants=selected_variants,
        config=config,
        windows=windows,
    )

    # Aggregate
    per_window_summary = build_per_window_summary(per_window_results)
    aggregated = aggregate_rolling_metrics(per_window_results)
    winner_info = select_rolling_winner(aggregated, acceptance_criteria=acceptance_criteria)

    logger.info(
        "Rolling-origin winner: %s (%s=%.4f, n_windows=%d)",
        winner_info["winner_variant_id"],
        acceptance_criteria,
        winner_info["winner_score"],
        winner_info["winner_n_windows"],
    )

    if args.dry_run:
        logger.info("Dry run complete; winner would be: %s", winner_info)
        return 0

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_window_summary.to_csv(
        args.output_dir / "rolling_origin_per_window_scores.csv", index=False
    )
    aggregated.to_csv(
        args.output_dir / "rolling_origin_aggregated_scores.csv", index=False
    )
    with open(
        args.output_dir / "rolling_origin_winner.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(winner_info, fh, indent=2)

    logger.info(
        "Rolling-origin backtest complete. %d window-variant results written to %s",
        len(per_window_results),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
