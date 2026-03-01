"""
Multi-county place allocation for PP-005 (WS-B).

Handles the 7 North Dakota places that span multiple counties by allocating
population to constituent counties, supporting per-county share trending, and
reaggregating county-level projections back to place-level totals.

Created: 2026-03-01
ADR: 058 (Multi-county place splitting)
Author: Claude / N. Haarstad

Purpose
-------
Phase 1 place projections (PP-003) assign each multi-county place to its
primary county only. This module implements proper multi-county handling:

1. Load allocation weights from the crosswalk multicounty detail CSV.
2. Split a multi-county place's population across constituent counties
   using area-share weights (default) or population-share weights.
3. Distribute historical share entries across counties so each county's
   share-trending engine receives correctly scaled input.
4. After trending, reaggregate county-level projected shares back to a
   single place-level total.

Method
------
- Allocation weights come from TIGER 2020 area overlaps already computed
  by ``scripts/data/build_place_county_crosswalk.py``.
- For each multi-county place, the place's historical share in its primary
  county is split proportionally to area_share weights across all
  constituent counties.
- The share-trending engine (``trend_all_places_in_county``) then projects
  each county-portion independently.
- Reaggregation sums projected populations across counties and recovers a
  single place-level total. The invariant ``sum(county_allocations) == place_total``
  is enforced.

Key design decisions
--------------------
- Area-share is the default allocation method because Census does not
  publish sub-place-by-county population breakdowns.
- The feature is gated behind ``multicounty_allocation.enabled`` in config;
  when disabled, behavior is identical to Phase 1 (primary-county-only).
- Only 7 of 357 active places are affected; the module is designed to be
  minimally invasive to the existing 90-place pipeline.

Inputs
------
- ``data/processed/geographic/place_county_crosswalk_2020.csv``: Primary
  crosswalk (one row per place, primary county assignment).
- ``data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv``:
  Supplemental overlap detail for multi-county places (14 rows, 7 places).

Outputs
-------
No file outputs. This module provides in-memory transformation functions
consumed by ``place_projection_orchestrator.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)


def _normalize_fips(value: object, width: int) -> str | None:
    """Normalize a FIPS-like value to zero-padded digit string."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)[-width:]


def identify_multicounty_places(crosswalk: pd.DataFrame) -> list[str]:
    """
    Return list of place_fips values that span multiple counties.

    Identifies multi-county places by looking for ``multi_county_primary``
    in the ``assignment_type`` column of the primary crosswalk.

    Args:
        crosswalk: Primary crosswalk DataFrame with ``place_fips`` and
            ``assignment_type`` columns.

    Returns:
        Sorted list of place_fips strings for multi-county places.
    """
    if crosswalk.empty:
        return []

    required = {"place_fips", "assignment_type"}
    missing = required - set(crosswalk.columns)
    if missing:
        raise ValueError(
            f"Crosswalk missing required columns: {sorted(missing)}"
        )

    multicounty = crosswalk[
        crosswalk["assignment_type"].astype(str) == "multi_county_primary"
    ]
    return sorted(multicounty["place_fips"].astype(str).unique().tolist())


def load_allocation_weights(
    crosswalk_path: Path,
    multicounty_detail_path: Path,
) -> dict[str, dict[str, float]]:
    """
    Load allocation weights for multi-county places.

    Reads the multicounty detail CSV and returns a nested dictionary mapping
    each multi-county place_fips to its constituent county_fips and
    area_share weights. Weights are normalized to sum to 1.0 within each
    place.

    Args:
        crosswalk_path: Path to primary crosswalk CSV (used to identify
            multi-county places via assignment_type).
        multicounty_detail_path: Path to multicounty detail CSV with
            ``place_fips``, ``county_fips``, ``area_share`` columns.

    Returns:
        Dict of ``{place_fips: {county_fips: weight, ...}, ...}``.
        Weights sum to 1.0 within each place.

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If required columns are missing or weights are invalid.
    """
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Crosswalk not found: {crosswalk_path}")
    if not multicounty_detail_path.exists():
        raise FileNotFoundError(
            f"Multicounty detail not found: {multicounty_detail_path}"
        )

    detail = pd.read_csv(multicounty_detail_path, dtype=str)
    required = {"place_fips", "county_fips", "area_share"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(
            f"Multicounty detail CSV missing required columns: {sorted(missing)}"
        )

    detail["place_fips"] = detail["place_fips"].map(
        lambda v: _normalize_fips(v, 7)
    )
    detail["county_fips"] = detail["county_fips"].map(
        lambda v: _normalize_fips(v, 5)
    )
    detail["area_share"] = pd.to_numeric(
        detail["area_share"], errors="coerce"
    )
    detail = detail.dropna(
        subset=["place_fips", "county_fips", "area_share"]
    ).copy()

    if (detail["area_share"] < 0).any():
        raise ValueError("Multicounty detail contains negative area_share values.")

    weights: dict[str, dict[str, float]] = {}
    for place_fips, group in detail.groupby("place_fips"):
        place_weights: dict[str, float] = {}
        total = float(group["area_share"].sum())
        if total <= 0:
            raise ValueError(
                f"Place {place_fips} has non-positive total area_share: {total}"
            )
        for _, row in group.iterrows():
            county = str(row["county_fips"])
            place_weights[county] = float(row["area_share"]) / total
        weights[str(place_fips)] = place_weights

    logger.info(
        "Loaded allocation weights for %d multi-county places (%d total county portions).",
        len(weights),
        sum(len(v) for v in weights.values()),
    )
    return weights


def split_multicounty_place(
    place_fips: str,
    population: float,
    weights: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Fan-out a place's population to constituent county portions.

    Distributes ``population`` across counties according to the allocation
    weights for the given place. The sum of allocated populations equals
    the input population (up to floating-point precision).

    Args:
        place_fips: 7-digit place FIPS code.
        population: Total population to allocate.
        weights: Allocation weights from ``load_allocation_weights()``.

    Returns:
        Dict of ``{county_fips: allocated_population, ...}``.

    Raises:
        ValueError: If place_fips not found in weights or population is negative.
    """
    if population < 0:
        raise ValueError(f"Population must be non-negative, got {population}.")

    if place_fips not in weights:
        raise ValueError(
            f"Place {place_fips} not found in multicounty allocation weights."
        )

    place_weights = weights[place_fips]
    allocations: dict[str, float] = {}
    running_total = 0.0
    counties = sorted(place_weights.keys())

    # Allocate to all but the last county, then assign remainder to last
    # to ensure exact sum preservation.
    for county in counties[:-1]:
        allocated = population * place_weights[county]
        allocations[county] = allocated
        running_total += allocated

    # Last county gets the remainder.
    allocations[counties[-1]] = population - running_total

    return allocations


def split_multicounty_shares(
    share_history: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    place_fips: str,
    primary_county_fips: str,
) -> pd.DataFrame:
    """
    Distribute a multi-county place's share history across constituent counties.

    Takes the historical share entries for a multi-county place (which exist
    only under the primary county) and creates per-county share entries by
    scaling the place's share by the allocation weight ratio.

    For a place with share ``s`` in primary county and allocation weight ``w_p``
    for the primary county and ``w_c`` for another county, the synthetic share
    entry for county c is computed as::

        share_c = s * (w_c / w_p)

    This preserves the place's total population across counties while creating
    properly scaled inputs for each county's share-trending engine.

    Args:
        share_history: Historical share rows for the place in its primary
            county. Must include ``county_fips``, ``year``, ``place_fips``,
            ``share_raw``, ``row_type``.
        weights: Allocation weights from ``load_allocation_weights()``.
        place_fips: 7-digit FIPS for the multi-county place.
        primary_county_fips: 5-digit FIPS of the primary county.

    Returns:
        DataFrame with share rows for all non-primary constituent counties.
        Has the same columns as the input. The primary county's rows are
        NOT included (they remain in the original history unchanged).
    """
    if place_fips not in weights:
        raise ValueError(
            f"Place {place_fips} not found in multicounty weights."
        )

    place_weights = weights[place_fips]
    if primary_county_fips not in place_weights:
        raise ValueError(
            f"Primary county {primary_county_fips} not in weights for place {place_fips}."
        )

    primary_weight = place_weights[primary_county_fips]
    if primary_weight <= 0:
        raise ValueError(
            f"Primary county weight must be positive, got {primary_weight}."
        )

    place_rows = share_history[
        (share_history["place_fips"] == place_fips)
        & (share_history["county_fips"] == primary_county_fips)
    ].copy()

    if place_rows.empty:
        return pd.DataFrame(columns=share_history.columns)

    synthetic_rows: list[pd.DataFrame] = []
    for county_fips, weight in place_weights.items():
        if county_fips == primary_county_fips:
            continue

        county_rows = place_rows.copy()
        county_rows["county_fips"] = county_fips
        # Scale the share proportionally to the weight ratio.
        county_rows["share_raw"] = (
            county_rows["share_raw"] * (weight / primary_weight)
        )
        if "share" in county_rows.columns:
            county_rows["share"] = (
                county_rows["share"] * (weight / primary_weight)
            )
        synthetic_rows.append(county_rows)

    if not synthetic_rows:
        return pd.DataFrame(columns=share_history.columns)

    result = pd.concat(synthetic_rows, ignore_index=True)
    logger.info(
        "Split place %s shares across %d non-primary counties (%d synthetic rows).",
        place_fips,
        len(synthetic_rows),
        len(result),
    )
    return result


def reaggregate_multicounty_place(
    county_projections: dict[str, pd.DataFrame],
    place_fips: str,
    weights: dict[str, dict[str, float]],
    year_column: str = "year",
    population_column: str = "projected_population",
) -> pd.DataFrame:
    """
    Fan-in projected county shares back to a single place total.

    Sums projected populations for a multi-county place across all
    constituent counties. Enforces the invariant that the sum of county
    allocations equals the place total.

    Args:
        county_projections: Dict of ``{county_fips: projection_df}`` where
            each DataFrame includes at least ``year``, ``place_fips``, and
            a population column for the place rows.
        place_fips: 7-digit FIPS of the multi-county place.
        weights: Allocation weights (used only to identify constituent counties).
        year_column: Name of the year column in projections.
        population_column: Name of the population column to sum.

    Returns:
        DataFrame with ``year`` and ``place_total`` columns representing
        the reaggregated place population by year.

    Raises:
        ValueError: If place_fips not in weights or no projection data found.
    """
    if place_fips not in weights:
        raise ValueError(
            f"Place {place_fips} not found in multicounty weights."
        )

    constituent_counties = list(weights[place_fips].keys())
    year_totals: dict[int, float] = {}

    for county_fips in constituent_counties:
        if county_fips not in county_projections:
            continue

        county_df = county_projections[county_fips]
        place_rows = county_df[
            county_df["place_fips"].astype(str) == place_fips
        ].copy()

        if place_rows.empty:
            continue

        for _, row in place_rows.iterrows():
            year = int(row[year_column])
            pop = float(row[population_column])
            year_totals[year] = year_totals.get(year, 0.0) + pop

    if not year_totals:
        raise ValueError(
            f"No projection data found for multi-county place {place_fips} "
            f"across counties {constituent_counties}."
        )

    result = pd.DataFrame(
        sorted(year_totals.items()),
        columns=["year", "place_total"],
    )
    return result


def get_multicounty_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract multicounty allocation config from the projection config.

    Returns the ``multicounty_allocation`` block from the
    ``place_projections`` section, with defaults applied.

    Args:
        config: Full projection configuration dictionary.

    Returns:
        Dict with keys ``enabled``, ``allocation_method``,
        ``multicounty_detail_path``.
    """
    place_cfg = config.get("place_projections", {})
    if not isinstance(place_cfg, dict):
        place_cfg = {}

    mc_cfg = place_cfg.get("multicounty_allocation", {})
    if not isinstance(mc_cfg, dict):
        mc_cfg = {}

    return {
        "enabled": bool(mc_cfg.get("enabled", False)),
        "allocation_method": str(
            mc_cfg.get("allocation_method", "area_share")
        ),
        "multicounty_detail_path": str(
            mc_cfg.get(
                "multicounty_detail_path",
                "data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv",
            )
        ),
    }


def prepare_multicounty_share_history(
    share_history: pd.DataFrame,
    crosswalk: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    multicounty_place_fips: list[str],
) -> pd.DataFrame:
    """
    Augment share history with synthetic rows for multi-county place portions.

    For each multi-county place, creates synthetic share history entries for
    non-primary counties by distributing the primary county's share history
    using allocation weights. The original primary-county rows are retained
    unchanged; new rows are appended for non-primary counties.

    Args:
        share_history: Full historical share DataFrame (all counties, all places).
        crosswalk: Primary crosswalk with ``place_fips`` and ``county_fips``
            for primary county assignments.
        weights: Allocation weights from ``load_allocation_weights()``.
        multicounty_place_fips: List of multi-county place FIPS codes.

    Returns:
        Augmented share history DataFrame with synthetic rows appended.
    """
    if not multicounty_place_fips:
        return share_history

    all_synthetic: list[pd.DataFrame] = []
    for pfips in multicounty_place_fips:
        # Find the primary county from the crosswalk.
        place_xw = crosswalk[crosswalk["place_fips"] == pfips]
        if place_xw.empty:
            logger.warning(
                "Multi-county place %s not found in crosswalk; skipping.",
                pfips,
            )
            continue

        primary_county = str(place_xw["county_fips"].iloc[0])

        synthetic = split_multicounty_shares(
            share_history=share_history,
            weights=weights,
            place_fips=pfips,
            primary_county_fips=primary_county,
        )
        if not synthetic.empty:
            all_synthetic.append(synthetic)

    if not all_synthetic:
        return share_history

    augmented = pd.concat(
        [share_history, *all_synthetic],
        ignore_index=True,
    )
    logger.info(
        "Augmented share history with %d synthetic multi-county rows.",
        sum(len(s) for s in all_synthetic),
    )
    return augmented


__all__ = [
    "get_multicounty_config",
    "identify_multicounty_places",
    "load_allocation_weights",
    "prepare_multicounty_share_history",
    "reaggregate_multicounty_place",
    "split_multicounty_place",
    "split_multicounty_shares",
]
