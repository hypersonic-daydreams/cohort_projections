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

import numpy as np
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

S05_THRESHOLDS = {
    "HIGH": {"tier_medape": 10.0, "tier_p90_mape": 20.0, "abs_tier_mean_me": 5.0},
    "MODERATE": {"tier_medape": 15.0, "tier_p90_mape": 30.0, "abs_tier_mean_me": 8.0},
    "LOWER": {"tier_medape": 25.0, "tier_p90_mape": 45.0, "abs_tier_mean_me": 12.0},
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


def _load_crosswalk_flags(config: dict[str, Any]) -> pd.DataFrame:
    """Load per-place crosswalk flags used in S05 detail-table reporting."""
    crosswalk_path = _resolve_path(
        config.get("place_projections", {}).get(
            "crosswalk_path",
            "data/processed/geographic/place_county_crosswalk_2020.csv",
        )
    )
    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk["place_fips"] = crosswalk["place_fips"].map(lambda v: _normalize_fips(v, 7))
    crosswalk["county_fips"] = crosswalk["county_fips"].map(lambda v: _normalize_fips(v, 5))
    cols = ["place_fips", "county_fips", "place_name", "assignment_type", "tier_boundary", "historical_only"]
    available_cols = [col for col in cols if col in crosswalk.columns]
    flags = crosswalk[available_cols].drop_duplicates(subset=["place_fips"]).copy()
    if "place_name" in flags.columns:
        flags = flags.rename(columns={"place_name": "crosswalk_place_name"})
    return flags


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


def _compute_place_year_errors(projected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Compute place-year APE/PE rows for PI calibration."""
    merged = projected.merge(
        actual,
        on=["place_fips", "year"],
        how="inner",
        validate="one_to_one",
    ).copy()
    if merged.empty:
        raise ValueError("No overlapping projected/actual rows for place-year error computation.")

    merged["projected_population"] = pd.to_numeric(merged["projected_population"], errors="coerce")
    merged["actual_population"] = pd.to_numeric(merged["actual_population"], errors="coerce")
    merged = merged.dropna(subset=["projected_population", "actual_population"]).copy()

    error = merged["projected_population"] - merged["actual_population"]
    denominator = merged["actual_population"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["APE"] = np.where(
            denominator > 0,
            np.abs(error.to_numpy(dtype=float)) / denominator * 100.0,
            np.nan,
        )
        merged["PE"] = np.where(
            denominator > 0,
            error.to_numpy(dtype=float) / denominator * 100.0,
            np.nan,
        )
    return merged


def _apply_s05_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach S05 pass/fail evaluation columns to tier summary rows."""
    evaluated = summary.copy()
    evaluated["confidence_tier"] = evaluated["confidence_tier"].astype(str).str.upper()

    def threshold_value(tier: str, metric: str) -> float | None:
        row = S05_THRESHOLDS.get(tier, {})
        value = row.get(metric)
        return float(value) if value is not None else None

    evaluated["tier_medape_threshold"] = evaluated["confidence_tier"].map(
        lambda tier: threshold_value(tier, "tier_medape")
    )
    evaluated["tier_p90_mape_threshold"] = evaluated["confidence_tier"].map(
        lambda tier: threshold_value(tier, "tier_p90_mape")
    )
    evaluated["abs_tier_mean_me_threshold"] = evaluated["confidence_tier"].map(
        lambda tier: threshold_value(tier, "abs_tier_mean_me")
    )
    evaluated["abs_tier_mean_me"] = evaluated["tier_mean_me"].abs()

    evaluated["tier_medape_pass"] = (
        evaluated["tier_medape"] <= evaluated["tier_medape_threshold"]
    ) | evaluated["tier_medape_threshold"].isna()
    evaluated["tier_p90_mape_pass"] = (
        evaluated["tier_p90_mape"] <= evaluated["tier_p90_mape_threshold"]
    ) | evaluated["tier_p90_mape_threshold"].isna()
    evaluated["abs_tier_mean_me_pass"] = (
        evaluated["abs_tier_mean_me"] <= evaluated["abs_tier_mean_me_threshold"]
    ) | evaluated["abs_tier_mean_me_threshold"].isna()

    evaluated["scored_tier"] = evaluated["confidence_tier"].isin(S05_THRESHOLDS)
    evaluated["tier_pass"] = (
        evaluated["tier_medape_pass"] & evaluated["tier_p90_mape_pass"] & evaluated["abs_tier_mean_me_pass"]
    )
    evaluated["evaluation_status"] = np.where(
        evaluated["scored_tier"],
        np.where(evaluated["tier_pass"], "PASS", "FAIL"),
        "INFORMATIONAL",
    )
    return evaluated


def _rank_tiers_for_output(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort tiers in reporting order HIGH, MODERATE, LOWER, EXCLUDED, then others."""
    rank = {"HIGH": 0, "MODERATE": 1, "LOWER": 2, "EXCLUDED": 3}
    ordered = summary.copy()
    ordered["tier_rank"] = ordered["confidence_tier"].map(lambda tier: rank.get(str(tier).upper(), 99))
    sort_cols = ["tier_rank", "confidence_tier"]
    if "window" in ordered.columns:
        sort_cols = ["window", *sort_cols]
    ordered = ordered.sort_values(sort_cols).drop(columns=["tier_rank"])
    return ordered.reset_index(drop=True)


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
    if "primary" not in args.windows:
        raise ValueError("Primary window is required because winner selection is primary-window based.")

    share_history = _load_share_history(config)
    county_population = _build_county_population(share_history)
    tier_assignments = _load_tier_assignments(config)
    crosswalk_flags = _load_crosswalk_flags(config)

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

    expected_variants = set(VARIANT_SPECS)
    observed_variants = set(all_scores["variant_id"].unique())
    if observed_variants != expected_variants:
        raise ValueError(
            "Backtest run did not produce all expected variants. "
            f"Expected={sorted(expected_variants)}, observed={sorted(observed_variants)}."
        )

    for window_name in args.windows:
        window_rows = all_scores[all_scores["window"] == window_name]
        if len(window_rows) != len(VARIANT_SPECS):
            raise ValueError(
                f"Window {window_name} produced {len(window_rows)} scores; "
                f"expected {len(VARIANT_SPECS)}."
            )

    primary_scores = all_scores[all_scores["window"] == "primary"][["variant_id", "score"]].copy()
    winner = select_winner(primary_scores)
    winner_row = all_scores[(all_scores["window"] == "primary") & (all_scores["variant_id"] == winner)].iloc[0]

    selected_windows = [window for window in ["primary", "secondary"] if window in args.windows]

    winner_summary_by_window: dict[str, pd.DataFrame] = {}
    winner_detail_frames: list[pd.DataFrame] = []
    winner_pi_rows: list[dict[str, Any]] = []

    winner_fit = str(winner_row["fitting_method"])
    winner_constraint = str(winner_row["constraint_method"])

    for window_name in selected_windows:
        window_train = window_defs[window_name]["train"]
        window_test = window_defs[window_name]["test"]
        winner_result = run_single_variant(
            variant_id=winner,
            fitting_method=winner_fit,
            constraint_method=winner_constraint,
            train_years=window_train,
            test_years=window_test,
            share_history=share_history,
            county_pop=county_population,
            config=config,
        )
        place_metrics = compute_per_place_metrics(
            projected=winner_result["projected"],
            actual=winner_result["actual"],
        )
        place_metrics["window"] = window_name
        place_metrics["variant_id"] = winner

        tier_aggregates = compute_tier_aggregates(
            place_metrics=place_metrics,
            tier_assignments=tier_assignments,
        )
        tier_aggregates["window"] = window_name
        tier_aggregates["variant_id"] = winner
        tier_aggregates = _apply_s05_thresholds(tier_aggregates)
        tier_aggregates = _rank_tiers_for_output(tier_aggregates)
        winner_summary_by_window[window_name] = tier_aggregates

        detail = place_metrics.merge(
            tier_assignments[["place_fips", "confidence_tier"]],
            on="place_fips",
            how="left",
            validate="one_to_one",
        )
        detail = detail.merge(
            crosswalk_flags,
            on="place_fips",
            how="left",
            validate="one_to_one",
        )
        if "crosswalk_place_name" in detail.columns:
            if "place_name" in detail.columns:
                detail["place_name"] = detail["place_name"].fillna(detail["crosswalk_place_name"])
            else:
                detail["place_name"] = detail["crosswalk_place_name"]
        detail["confidence_tier"] = detail["confidence_tier"].fillna("UNASSIGNED").astype(str).str.upper()
        detail["tier_p90_mape_threshold"] = detail["confidence_tier"].map(
            lambda tier: S05_THRESHOLDS.get(str(tier).upper(), {}).get("tier_p90_mape")
        )
        detail["threshold_exceedance"] = np.where(
            detail["tier_p90_mape_threshold"].notna() & (detail["MAPE"] > detail["tier_p90_mape_threshold"]),
            "EXCEEDS_TIER_90TH_CEILING",
            "",
        )
        detail["tier_boundary"] = detail.get("tier_boundary", pd.Series(False, index=detail.index)).fillna(False)
        detail["multi_county_primary_flag"] = (
            detail.get("assignment_type", pd.Series("", index=detail.index))
            .fillna("")
            .astype(str)
            .eq("multi_county_primary")
        )
        detail = detail.rename(
            columns={
                "confidence_tier": "tier",
                "ME": "mean_error_pct",
                "multi_county_primary_flag": "multi_county_primary",
            }
        )
        detail["flag"] = detail["threshold_exceedance"]
        winner_detail_frames.append(detail)

        place_year_errors = _compute_place_year_errors(
            projected=winner_result["projected"],
            actual=winner_result["actual"],
        ).merge(
            tier_assignments[["place_fips", "confidence_tier"]],
            on="place_fips",
            how="left",
            validate="many_to_one",
        )
        place_year_errors["confidence_tier"] = (
            place_year_errors["confidence_tier"].fillna("UNASSIGNED").astype(str).str.upper()
        )
        for tier_name, tier_rows in place_year_errors.groupby("confidence_tier"):
            abs_pe = tier_rows["APE"].dropna().to_numpy(dtype=float)
            if len(abs_pe) == 0:
                continue
            winner_pi_rows.append(
                {
                    "window": window_name,
                    "variant_id": winner,
                    "confidence_tier": tier_name,
                    "pi80_half_width_pct": float(np.nanpercentile(abs_pe, 80)),
                    "pi90_half_width_pct": float(np.nanpercentile(abs_pe, 90)),
                    "n_place_years": int(len(abs_pe)),
                }
            )

    primary_summary = winner_summary_by_window.get("primary")
    if primary_summary is None:
        raise ValueError("Primary window summary was not produced.")
    scored_primary = primary_summary[primary_summary["scored_tier"]].copy()
    if scored_primary.empty:
        raise ValueError("Primary summary has no scored tiers (HIGH/MODERATE/LOWER).")
    primary_all_pass = bool(scored_primary["tier_pass"].all())
    failed_tiers = scored_primary.loc[~scored_primary["tier_pass"], "confidence_tier"].tolist()

    backtest_per_place_detail = pd.concat(winner_detail_frames, ignore_index=True)
    detail_cols = [
        "window",
        "variant_id",
        "place_fips",
        "place_name",
        "county_fips",
        "tier",
        "MAPE",
        "MedAPE",
        "mean_error_pct",
        "MaxAPE",
        "AE_terminal",
        "flag",
        "tier_boundary",
        "multi_county_primary",
    ]
    existing_detail_cols = [col for col in detail_cols if col in backtest_per_place_detail.columns]
    backtest_per_place_detail = backtest_per_place_detail[existing_detail_cols].copy()
    backtest_per_place_detail = backtest_per_place_detail.sort_values(
        by=["window", "tier", "MAPE", "place_fips"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    backtest_prediction_intervals = pd.DataFrame(winner_pi_rows)
    if backtest_prediction_intervals.empty:
        raise ValueError("Prediction interval table is empty for winner variant.")
    backtest_prediction_intervals = _rank_tiers_for_output(backtest_prediction_intervals)

    winner_payload = {
        "winner_variant_id": winner,
        "window": "primary",
        "score": float(winner_row["score"]),
        "fitting_method": winner_row["fitting_method"],
        "constraint_method": winner_row["constraint_method"],
        "acceptance": {
            "all_scored_tiers_pass_primary": primary_all_pass,
            "failed_tiers_primary": failed_tiers,
            "scored_tiers": ["HIGH", "MODERATE", "LOWER"],
            "thresholds": S05_THRESHOLDS,
        },
    }

    if args.dry_run:
        logger.info("Dry run complete; winner would be: %s", winner_payload)
        if not primary_all_pass:
            logger.warning("Primary window scored tiers failed thresholds: %s", failed_tiers)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.output_dir / "backtest_variant_scores.csv", index=False)
    primary_summary.to_csv(args.output_dir / "backtest_summary_primary.csv", index=False)
    secondary_summary = winner_summary_by_window.get("secondary")
    if secondary_summary is not None:
        secondary_summary.to_csv(args.output_dir / "backtest_summary_secondary.csv", index=False)
    backtest_per_place_detail.to_csv(args.output_dir / "backtest_per_place_detail.csv", index=False)
    backtest_prediction_intervals.to_csv(args.output_dir / "backtest_prediction_intervals.csv", index=False)
    all_tiers.to_csv(args.output_dir / "backtest_tier_aggregates.csv", index=False)
    all_metrics.to_csv(args.output_dir / "backtest_per_place_metrics.csv", index=False)
    with open(args.output_dir / "backtest_winner.json", "w", encoding="utf-8") as file_handle:
        json.dump(winner_payload, file_handle, indent=2)

    if not primary_all_pass:
        logger.warning("Primary window scored tiers failed thresholds: %s", failed_tiers)

    logger.info(
        "Backtest complete: %d score rows, winner %s (score=%.4f).",
        len(all_scores),
        winner,
        float(winner_row["score"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
