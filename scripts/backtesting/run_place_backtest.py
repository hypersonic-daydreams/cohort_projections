#!/usr/bin/env python3
"""
Run PP-003 place backtest variant matrix (IMP-08).

Executes variants A-I, A-II, B-I, B-II across configured backtest windows and
writes scoring artifacts for winner selection.
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

from cohort_projections.data.process.place_backtest import (  # noqa: E402
    compute_per_place_metrics,
    compute_tier_aggregates,
    compute_variant_score,
    run_single_variant,
    select_winner,
)
from cohort_projections.utils import load_projection_config  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

VARIANT_SPECS = {
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
    place_population["place_fips"] = place_population["place_fips"].map(lambda v: _normalize_fips(v, 7))
    place_population["year"] = pd.to_numeric(place_population["year"], errors="coerce").astype("Int64")
    place_population["population"] = pd.to_numeric(place_population["population"], errors="coerce")
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
    county_pop["county_population"] = pd.to_numeric(county_pop["county_population"], errors="coerce")
    county_pop = county_pop.dropna(subset=["county_fips", "year", "county_population"]).copy()
    county_pop["year"] = county_pop["year"].astype(int)
    county_pop = (
        county_pop.groupby(["county_fips", "year"], as_index=False)["county_population"]
        .first()
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    return county_pop


def run_window(
    window_name: str,
    train_window: list[int],
    test_window: list[int],
    share_history: pd.DataFrame,
    county_population: pd.DataFrame,
    tier_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all variants for one backtest window."""
    score_rows: list[dict[str, Any]] = []
    tier_rows: list[pd.DataFrame] = []
    place_metric_rows: list[pd.DataFrame] = []

    for variant_id, spec in VARIANT_SPECS.items():
        logger.info(
            "Window %s: running variant %s (%s, %s)",
            window_name,
            variant_id,
            spec["fitting_method"],
            spec["constraint_method"],
        )
        variant_result = run_single_variant(
            variant_id=variant_id,
            fitting_method=spec["fitting_method"],
            constraint_method=spec["constraint_method"],
            train_years=train_window,
            test_years=test_window,
            share_history=share_history,
            county_pop=county_population,
            config=config,
        )
        place_metrics = compute_per_place_metrics(
            projected=variant_result["projected"],
            actual=variant_result["actual"],
        )
        tier_aggregates = compute_tier_aggregates(
            place_metrics=place_metrics,
            tier_assignments=tier_assignments,
        )
        score = compute_variant_score(tier_aggregates)

        score_rows.append(
            {
                "window": window_name,
                "variant_id": variant_id,
                "fitting_method": spec["fitting_method"],
                "constraint_method": spec["constraint_method"],
                "score": score,
            }
        )

        place_metrics = place_metrics.copy()
        place_metrics["window"] = window_name
        place_metrics["variant_id"] = variant_id
        place_metric_rows.append(place_metrics)

        tier_aggregates = tier_aggregates.copy()
        tier_aggregates["window"] = window_name
        tier_aggregates["variant_id"] = variant_id
        tier_rows.append(tier_aggregates)

    return (
        pd.DataFrame(score_rows),
        pd.concat(tier_rows, ignore_index=True),
        pd.concat(place_metric_rows, ignore_index=True),
    )


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run PP-003 place backtest variant matrix.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "projection_config.yaml",
        help="Path to projection config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "backtesting" / "place_backtest_results",
        help="Directory for backtest output artifacts.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["primary", "secondary"],
        choices=["primary", "secondary"],
        help="Backtest windows to run.",
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

    share_history = _load_share_history(config)
    county_population = _build_county_population(share_history)
    tier_assignments = _load_tier_assignments(config)

    backtest_cfg = config.get("place_projections", {}).get("backtest", {})
    window_defs = {
        "primary": {
            "train": backtest_cfg.get("primary_train", [2000, 2014]),
            "test": backtest_cfg.get("primary_test", [2015, 2024]),
        },
        "secondary": {
            "train": backtest_cfg.get("secondary_train", [2000, 2019]),
            "test": backtest_cfg.get("secondary_test", [2020, 2024]),
        },
    }

    score_frames: list[pd.DataFrame] = []
    tier_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []

    for window_name in args.windows:
        train = window_defs[window_name]["train"]
        test = window_defs[window_name]["test"]
        logger.info("Running backtest window %s (train=%s, test=%s)", window_name, train, test)
        scores, tiers, metrics = run_window(
            window_name=window_name,
            train_window=train,
            test_window=test,
            share_history=share_history,
            county_population=county_population,
            tier_assignments=tier_assignments,
            config=config,
        )
        score_frames.append(scores)
        tier_frames.append(tiers)
        metric_frames.append(metrics)

    all_scores = pd.concat(score_frames, ignore_index=True)
    all_tiers = pd.concat(tier_frames, ignore_index=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)

    primary_scores = all_scores[all_scores["window"] == "primary"][["variant_id", "score"]].copy()
    winner = select_winner(primary_scores)
    winner_row = all_scores[(all_scores["window"] == "primary") & (all_scores["variant_id"] == winner)].iloc[0]

    winner_payload = {
        "winner_variant_id": winner,
        "window": "primary",
        "score": float(winner_row["score"]),
        "fitting_method": winner_row["fitting_method"],
        "constraint_method": winner_row["constraint_method"],
    }

    if args.dry_run:
        logger.info("Dry run complete; winner would be: %s", winner_payload)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.output_dir / "backtest_variant_scores.csv", index=False)
    all_tiers.to_csv(args.output_dir / "backtest_tier_aggregates.csv", index=False)
    all_metrics.to_csv(args.output_dir / "backtest_per_place_metrics.csv", index=False)
    with open(args.output_dir / "backtest_winner.json", "w", encoding="utf-8") as file_handle:
        json.dump(winner_payload, file_handle, indent=2)

    logger.info(
        "Backtest complete: %d score rows, winner %s (score=%.4f).",
        len(all_scores),
        winner,
        float(winner_row["score"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

