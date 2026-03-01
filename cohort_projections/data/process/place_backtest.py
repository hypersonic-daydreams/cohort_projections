"""
Place share-trending backtest utilities for PP-003 Phase 3 (IMP-08).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.data.process.place_share_trending import trend_all_places_in_county
from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

TIER_HIGH = "HIGH"
TIER_MODERATE = "MODERATE"
TIER_LOWER = "LOWER"
SCORED_TIERS = {TIER_HIGH, TIER_MODERATE, TIER_LOWER}


def _normalize_fips(value: object, width: int) -> str:
    """Normalize FIPS-like values to zero-padded strings."""
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(width)[-width:]


def _parse_window(window: Sequence[int] | tuple[int, int]) -> tuple[int, int]:
    """Parse a `[start, end]` window definition into integer bounds."""
    if len(window) != 2:
        raise ValueError(f"Window must contain exactly two years, got: {window}")
    start = int(window[0])
    end = int(window[1])
    if start > end:
        raise ValueError(f"Window start must be <= end, got: {window}")
    return start, end


def _extract_actual_population(
    actual_source: pd.DataFrame,
) -> pd.DataFrame:
    """Extract actual place populations for backtest evaluation."""
    source = actual_source.copy()

    if "population" in source.columns:
        source["actual_population"] = pd.to_numeric(source["population"], errors="coerce")
    elif {"share_raw", "county_population"}.issubset(source.columns):
        source["share_raw"] = pd.to_numeric(source["share_raw"], errors="coerce")
        source["county_population"] = pd.to_numeric(source["county_population"], errors="coerce")
        source["actual_population"] = source["share_raw"] * source["county_population"]
    else:
        raise ValueError(
            "Actual backtest source must include either `population`, or `share_raw` + `county_population`."
        )

    cols = ["place_fips", "year", "actual_population"]
    if "county_fips" in source.columns:
        cols.append("county_fips")
    if "place_name" in source.columns:
        cols.append("place_name")

    actual = source[cols].dropna(subset=["place_fips", "year", "actual_population"]).copy()
    actual["place_fips"] = actual["place_fips"].map(lambda v: _normalize_fips(v, 7))
    if "county_fips" in actual.columns:
        actual["county_fips"] = actual["county_fips"].map(lambda v: _normalize_fips(v, 5))
    actual["year"] = pd.to_numeric(actual["year"], errors="coerce").astype("Int64")
    actual = actual.dropna(subset=["year"]).copy()
    actual["year"] = actual["year"].astype(int)

    if actual.duplicated(subset=["place_fips", "year"]).any():
        raise ValueError("Actual population input has duplicate place_fips/year rows.")
    return actual


def _normalize_county_population(
    county_pop: pd.DataFrame,
    test_window: tuple[int, int],
) -> pd.DataFrame:
    """Normalize county population input to long format."""
    data = county_pop.copy()
    test_start, test_end = test_window

    if {"county_fips", "year"}.issubset(data.columns):
        pop_col = next(
            (
                col
                for col in [
                    "county_population",
                    "population",
                    "total_population",
                    "pop_total",
                    "POPESTIMATE",
                ]
                if col in data.columns
            ),
            None,
        )
        if pop_col is None:
            raise ValueError(
                "county_pop long format must include one of: county_population, population, "
                "total_population, pop_total, POPESTIMATE."
            )

        normalized = data[["county_fips", "year", pop_col]].copy()
        normalized = normalized.rename(columns={pop_col: "county_population"})
        normalized["county_fips"] = normalized["county_fips"].map(lambda v: _normalize_fips(v, 5))
        normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
        normalized["county_population"] = pd.to_numeric(normalized["county_population"], errors="coerce")
        normalized = normalized.dropna(subset=["county_fips", "year", "county_population"]).copy()
        normalized["year"] = normalized["year"].astype(int)
        normalized = normalized[
            (normalized["year"] >= test_start) & (normalized["year"] <= test_end)
        ].copy()
        return normalized

    raise ValueError("county_pop must include county_fips/year columns in long format.")


def run_single_variant(
    variant_id: str,
    fitting_method: str,
    constraint_method: str,
    train_years: Sequence[int] | tuple[int, int],
    test_years: Sequence[int] | tuple[int, int],
    share_history: pd.DataFrame,
    county_pop: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Execute one backtest variant for one train/test window.

    Args:
        variant_id: Variant identifier (`A-I`, `A-II`, `B-I`, `B-II`).
        fitting_method: `ols` or `wls`.
        constraint_method: `proportional` or `cap_and_redistribute`.
        train_years: Training window `[start, end]`.
        test_years: Test window `[start, end]`.
        share_history: Historical place-share rows (expected from IMP-03 output).
        county_pop: County population history in long format.
        config: Project configuration.

    Returns:
        Dictionary with `projected` and `actual` DataFrames for the variant.
    """
    train_start, train_end = _parse_window(train_years)
    test_start, test_end = _parse_window(test_years)
    projection_years = list(range(test_start, test_end + 1))

    history = share_history.copy()
    if "year" not in history.columns:
        raise ValueError("share_history must include `year`.")
    if "place_fips" not in history.columns:
        raise ValueError("share_history must include `place_fips`.")
    if "county_fips" not in history.columns:
        raise ValueError("share_history must include `county_fips`.")
    if "share_raw" not in history.columns and "share" not in history.columns:
        raise ValueError("share_history must include `share_raw` or `share`.")

    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype("Int64")
    history = history.dropna(subset=["year"]).copy()
    history["year"] = history["year"].astype(int)
    history["county_fips"] = history["county_fips"].map(lambda v: _normalize_fips(v, 5))
    history["place_fips"] = history["place_fips"].map(
        lambda v: _normalize_fips(v, 7) if pd.notna(v) else pd.NA
    )

    if "row_type" in history.columns:
        place_history = history[history["row_type"].astype(str) == "place"].copy()
    else:
        place_history = history[history["place_fips"].notna()].copy()

    train_history = place_history[
        (place_history["year"] >= train_start) & (place_history["year"] <= train_end)
    ].copy()
    test_actual_source = place_history[
        (place_history["year"] >= test_start) & (place_history["year"] <= test_end)
    ].copy()

    if train_history.empty:
        raise ValueError(f"No training share rows found in window {train_start}-{train_end}.")
    if test_actual_source.empty:
        raise ValueError(f"No test actual rows found in window {test_start}-{test_end}.")

    county_population = _normalize_county_population(county_pop, (test_start, test_end))
    projected_frames: list[pd.DataFrame] = []

    model_cfg = config.get("place_projections", {}).get("model", {})
    trend_cfg = {
        "projection_years": projection_years,
        "fitting_method": fitting_method,
        "constraint_method": constraint_method,
        "epsilon": float(model_cfg.get("epsilon", 0.001)),
        "lambda_decay": float(model_cfg.get("lambda_decay", 0.9)),
        "reconciliation_flag_threshold": float(model_cfg.get("reconciliation_flag_threshold", 0.05)),
    }

    for county_fips, county_train in train_history.groupby("county_fips"):
        county_pop_rows = county_population[county_population["county_fips"] == county_fips].copy()
        if county_pop_rows.empty:
            raise ValueError(f"Missing county population rows for county {county_fips}.")

        county_projection = trend_all_places_in_county(
            place_share_history=county_train,
            county_pop_history=county_pop_rows[["year", "county_population"]],
            config=trend_cfg,
        )
        place_rows = county_projection[county_projection["row_type"] == "place"].copy()
        place_rows["variant_id"] = variant_id
        projected_frames.append(place_rows)

    projected = pd.concat(projected_frames, ignore_index=True)
    projected = projected.rename(
        columns={
            "projected_population": "projected_population",
            "projected_share": "projected_share",
        }
    )
    projected = projected[
        ["variant_id", "county_fips", "place_fips", "year", "projected_population", "projected_share"]
    ].copy()

    actual = _extract_actual_population(test_actual_source)

    logger.info(
        "Backtest variant %s complete: %d projected rows, %d actual rows.",
        variant_id,
        len(projected),
        len(actual),
    )
    return {"projected": projected, "actual": actual}


def compute_per_place_metrics(
    projected: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-place backtest metrics (S05 Section 3.1).

    Args:
        projected: Variant projections by place-year.
        actual: Actual place population by place-year.

    Returns:
        DataFrame with one row per place and columns:
        `MAPE`, `MedAPE`, `ME`, `MaxAPE`, `AE_terminal`.
    """
    required_proj = {"place_fips", "year", "projected_population"}
    required_actual = {"place_fips", "year", "actual_population"}
    if missing := (required_proj - set(projected.columns)):
        raise ValueError(f"projected missing required columns: {sorted(missing)}")
    if missing := (required_actual - set(actual.columns)):
        raise ValueError(f"actual missing required columns: {sorted(missing)}")

    merged = projected.merge(
        actual,
        on=["place_fips", "year"],
        how="inner",
        validate="one_to_one",
    ).copy()
    if merged.empty:
        raise ValueError("No overlapping projected/actual place-year rows for metrics.")

    merged["projected_population"] = pd.to_numeric(merged["projected_population"], errors="coerce")
    merged["actual_population"] = pd.to_numeric(merged["actual_population"], errors="coerce")
    merged = merged.dropna(subset=["projected_population", "actual_population"]).copy()

    error = merged["projected_population"] - merged["actual_population"]
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = merged["actual_population"].to_numpy(dtype=float)
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
    merged["AE"] = np.abs(error.to_numpy(dtype=float))

    terminal_year = int(merged["year"].max())
    terminal = merged[merged["year"] == terminal_year][["place_fips", "AE"]].rename(
        columns={"AE": "AE_terminal"}
    )
    if terminal.duplicated(subset=["place_fips"]).any():
        terminal = terminal.groupby("place_fips", as_index=False)["AE_terminal"].mean()

    summary = (
        merged.groupby("place_fips", as_index=False)
        .agg(
            MAPE=("APE", "mean"),
            MedAPE=("APE", "median"),
            ME=("PE", "mean"),
            MaxAPE=("APE", "max"),
            n_test_years=("year", "nunique"),
        )
        .merge(terminal, on="place_fips", how="left", validate="one_to_one")
    )

    if "county_fips" in merged.columns:
        county_map = merged.groupby("place_fips", as_index=False)["county_fips"].first()
        summary = summary.merge(county_map, on="place_fips", how="left", validate="one_to_one")
    if "place_name" in merged.columns:
        name_map = merged.groupby("place_fips", as_index=False)["place_name"].first()
        summary = summary.merge(name_map, on="place_fips", how="left", validate="one_to_one")

    return summary


def compute_tier_aggregates(
    place_metrics: pd.DataFrame,
    tier_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute tier-level aggregates (S05 Section 3.2).

    Args:
        place_metrics: Per-place metric output from `compute_per_place_metrics`.
        tier_assignments: Place-tier mapping with columns
            `place_fips`, `confidence_tier`, and optional `population_2024`.

    Returns:
        Tier-level DataFrame with MedAPE, mean ME, p90 MAPE, and tier max MAPE.
    """
    required_metrics = {"place_fips", "MAPE", "ME"}
    required_tiers = {"place_fips", "confidence_tier"}
    if missing := (required_metrics - set(place_metrics.columns)):
        raise ValueError(f"place_metrics missing required columns: {sorted(missing)}")
    if missing := (required_tiers - set(tier_assignments.columns)):
        raise ValueError(f"tier_assignments missing required columns: {sorted(missing)}")

    tiers = tier_assignments.copy()
    tiers["place_fips"] = tiers["place_fips"].map(lambda v: _normalize_fips(v, 7))
    tiers["confidence_tier"] = tiers["confidence_tier"].astype(str).str.upper()
    if "population_2024" in tiers.columns:
        tiers["population_2024"] = pd.to_numeric(tiers["population_2024"], errors="coerce")
    else:
        tiers["population_2024"] = np.nan

    metrics = place_metrics.copy()
    metrics["place_fips"] = metrics["place_fips"].map(lambda v: _normalize_fips(v, 7))

    merged = metrics.merge(
        tiers[["place_fips", "confidence_tier", "population_2024"]],
        on="place_fips",
        how="left",
        validate="one_to_one",
    )
    merged["confidence_tier"] = merged["confidence_tier"].fillna("UNASSIGNED")

    aggregates = (
        merged.groupby("confidence_tier", as_index=False)
        .agg(
            place_count=("place_fips", "nunique"),
            tier_medape=("MAPE", "median"),
            tier_mean_me=("ME", "mean"),
            tier_p90_mape=("MAPE", lambda s: float(np.nanpercentile(s, 90))),
            tier_max_mape=("MAPE", "max"),
            tier_population_2024=("population_2024", "sum"),
        )
        .sort_values("confidence_tier")
        .reset_index(drop=True)
    )
    return aggregates


def compute_variant_score(tier_aggregates: pd.DataFrame) -> float:
    """
    Compute population-weighted MedAPE score (S04 Section 5.3).

    Args:
        tier_aggregates: Tier-level metric table from `compute_tier_aggregates`.

    Returns:
        Scalar score where lower is better.
    """
    required = {"confidence_tier", "tier_medape"}
    if missing := (required - set(tier_aggregates.columns)):
        raise ValueError(f"tier_aggregates missing required columns: {sorted(missing)}")

    scored = tier_aggregates[
        tier_aggregates["confidence_tier"].astype(str).str.upper().isin(SCORED_TIERS)
    ].copy()
    if scored.empty:
        raise ValueError("No HIGH/MODERATE/LOWER rows found for variant scoring.")

    if "tier_population_2024" in scored.columns:
        weights = pd.to_numeric(scored["tier_population_2024"], errors="coerce").to_numpy(dtype=float)
    else:
        weights = np.ones(len(scored), dtype=float)

    medapes = pd.to_numeric(scored["tier_medape"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(medapes)
    medapes = medapes[valid]
    weights = weights[valid]

    if len(medapes) == 0:
        raise ValueError("No finite tier_medape values available for scoring.")

    if not np.isfinite(weights).all() or np.nansum(weights) <= 0:
        weights = np.ones(len(medapes), dtype=float)

    return float(np.average(medapes, weights=weights))


def _variant_rank(variant_id: str) -> tuple[int, int]:
    """Return tie-break complexity rank for variant IDs."""
    normalized = variant_id.upper().replace("_", "-")
    if "-" in normalized:
        fit_part, constraint_part = normalized.split("-", 1)
    else:
        fit_part, constraint_part = normalized, ""

    fit_rank = 0 if fit_part == "A" else 1 if fit_part == "B" else 9
    constraint_rank = 0 if constraint_part == "I" else 1 if constraint_part == "II" else 9
    return fit_rank, constraint_rank


def select_winner(variant_scores: pd.DataFrame | Mapping[str, float]) -> str:
    """
    Select winning variant by score with S04 tie-breaking rules.

    Tie-break sequence (when scores equal to 3 decimals):
    1. Prefer A over B.
    2. Prefer I over II.

    Args:
        variant_scores: Either DataFrame with columns `variant_id`, `score`,
            or a mapping from variant ID to score.

    Returns:
        Winning variant ID.
    """
    if isinstance(variant_scores, Mapping):
        score_df = pd.DataFrame(
            [{"variant_id": variant_id, "score": score} for variant_id, score in variant_scores.items()]
        )
    else:
        score_df = variant_scores.copy()

    if {"variant_id", "score"} - set(score_df.columns):
        raise ValueError("variant_scores must provide `variant_id` and `score`.")
    if score_df.empty:
        raise ValueError("variant_scores cannot be empty.")

    score_df["variant_id"] = score_df["variant_id"].astype(str)
    score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
    score_df = score_df.dropna(subset=["score"]).copy()
    if score_df.empty:
        raise ValueError("variant_scores has no finite score values.")

    score_df["score_3dp"] = score_df["score"].round(3)
    min_3dp = float(score_df["score_3dp"].min())
    tied = score_df[score_df["score_3dp"] == min_3dp].copy()
    tied["fit_rank"] = tied["variant_id"].map(lambda value: _variant_rank(value)[0])
    tied["constraint_rank"] = tied["variant_id"].map(lambda value: _variant_rank(value)[1])

    winner = tied.sort_values(
        ["fit_rank", "constraint_rank", "score", "variant_id"],
        ascending=[True, True, True, True],
    ).iloc[0]
    return str(winner["variant_id"])


__all__ = [
    "compute_per_place_metrics",
    "compute_tier_aggregates",
    "compute_variant_score",
    "run_single_variant",
    "select_winner",
]

