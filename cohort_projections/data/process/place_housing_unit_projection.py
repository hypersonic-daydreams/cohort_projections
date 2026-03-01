"""
Housing-unit method for place population projections (ADR-060).

This module implements a complementary short-term (5-10 year) place
projection method using:

    projected_population = housing_units x persons_per_household

It is NOT a replacement for the share-trending county-constrained
method in ``place_projection_orchestrator.py``.  Instead it provides an
independent cross-check for the nearest projection horizon.

Functions
---------
load_housing_data
    Load housing-unit and PPH history from CSV.
trend_housing_units
    Fit a trend model to housing-unit counts and project forward.
project_pph
    Project persons-per-household forward.
project_population_from_hu
    Multiply projected HU by projected PPH.
run_housing_unit_projections
    Orchestrate HU projections for all eligible places.
cross_validate_with_share_trending
    Compute divergence metrics between HU and share-trending methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_housing_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load housing-unit and PPH history from the configured CSV.

    The CSV is expected to have columns:
        place_fips, place_name, year, housing_units, avg_hh_size

    Args:
        config: Full projection config dict (must contain
                ``housing_unit_method.housing_data_path``).

    Returns:
        DataFrame with columns place_fips, place_name, year,
        housing_units, avg_hh_size.

    Raises:
        FileNotFoundError: If the housing data CSV does not exist.
        ValueError: If required columns are missing.
    """
    hu_cfg = config.get("housing_unit_method", {})
    raw_path = hu_cfg.get("housing_data_path", "data/raw/housing/nd_place_housing_units.csv")
    path = Path(raw_path)
    if not path.is_absolute():
        # Resolve relative to project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Housing data file not found: {path}")

    df = pd.read_csv(path, dtype={"place_fips": str})
    required = {"place_fips", "year", "housing_units", "avg_hh_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Housing data CSV missing required columns: {missing}")

    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["housing_units"] = pd.to_numeric(df["housing_units"], errors="coerce")
    df["avg_hh_size"] = pd.to_numeric(df["avg_hh_size"], errors="coerce")

    # Drop rows with null essentials
    df = df.dropna(subset=["place_fips", "year", "housing_units"])

    logger.info(
        "Loaded housing data: %d rows, %d places, years %s-%s",
        len(df),
        df["place_fips"].nunique(),
        df["year"].min(),
        df["year"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Trend models
# ---------------------------------------------------------------------------


def trend_housing_units(
    hu_history: pd.DataFrame,
    method: str = "log_linear",
    projection_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Fit a trend model to housing-unit counts and project forward.

    Supports two methods:

    * ``"linear"``: ordinary least-squares on (year, HU).
    * ``"log_linear"``: OLS on (year, ln(HU)), then exponentiate.
      More appropriate when housing growth is multiplicative.

    Args:
        hu_history: DataFrame with columns ``year`` and ``housing_units``
                    for a **single** place.
        method: ``"linear"`` or ``"log_linear"``.
        projection_years: List of future years to project to.

    Returns:
        DataFrame with columns ``year`` and ``hu_projected``.

    Raises:
        ValueError: If fewer than 1 observation or unknown method.
    """
    if hu_history.empty:
        raise ValueError("Cannot fit trend to empty housing-unit history.")

    years = hu_history["year"].values.astype(float)
    hu = hu_history["housing_units"].values.astype(float)

    if len(years) == 1:
        # Single observation: hold constant
        if projection_years is None:
            projection_years = [int(years[0]) + 5]
        return pd.DataFrame({
            "year": projection_years,
            "hu_projected": [float(hu[0])] * len(projection_years),
        })

    if projection_years is None:
        last_year = int(years.max())
        projection_years = [last_year + 5, last_year + 10]

    proj_years = np.array(projection_years, dtype=float)

    if method == "linear":
        coeffs = np.polyfit(years, hu, deg=1)
        projected = np.polyval(coeffs, proj_years)
        # Floor at zero -- negative housing units are not meaningful
        projected = np.maximum(projected, 0.0)

    elif method == "log_linear":
        # Guard against zero or negative HU
        hu_safe = np.where(hu > 0, hu, 1.0)
        coeffs = np.polyfit(years, np.log(hu_safe), deg=1)
        projected = np.exp(np.polyval(coeffs, proj_years))

    else:
        raise ValueError(f"Unknown trend method: {method!r}. Use 'linear' or 'log_linear'.")

    return pd.DataFrame({
        "year": [int(y) for y in proj_years],
        "hu_projected": projected.tolist(),
    })


def project_pph(
    pph_history: pd.DataFrame,
    method: str = "hold_last",
    projection_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Project persons-per-household (PPH) forward.

    Supported methods:

    * ``"hold_last"``: Use the most recent observed PPH for all future years.
    * ``"linear_trend"``: Fit a linear trend and extrapolate.

    Args:
        pph_history: DataFrame with columns ``year`` and ``avg_hh_size``
                     for a **single** place.
        method: ``"hold_last"`` or ``"linear_trend"``.
        projection_years: Years to project to.

    Returns:
        DataFrame with columns ``year`` and ``pph_projected``.
    """
    if pph_history.empty:
        raise ValueError("Cannot project PPH from empty history.")

    sorted_hist = pph_history.sort_values("year")
    last_pph = float(sorted_hist["avg_hh_size"].iloc[-1])

    if projection_years is None:
        last_year = int(sorted_hist["year"].max())
        projection_years = [last_year + 5, last_year + 10]

    if method == "hold_last" or len(sorted_hist) < 2:
        return pd.DataFrame({
            "year": projection_years,
            "pph_projected": [last_pph] * len(projection_years),
        })

    if method == "linear_trend":
        years = sorted_hist["year"].values.astype(float)
        pph = sorted_hist["avg_hh_size"].values.astype(float)
        coeffs = np.polyfit(years, pph, deg=1)
        proj = np.polyval(coeffs, np.array(projection_years, dtype=float))
        # Floor PPH at 1.0 (a household has at least 1 person)
        proj = np.maximum(proj, 1.0)
        return pd.DataFrame({
            "year": projection_years,
            "pph_projected": proj.tolist(),
        })

    raise ValueError(f"Unknown PPH method: {method!r}. Use 'hold_last' or 'linear_trend'.")


# ---------------------------------------------------------------------------
# Population projection
# ---------------------------------------------------------------------------


def project_population_from_hu(
    projected_hu: pd.DataFrame,
    projected_pph: pd.DataFrame,
) -> pd.DataFrame:
    """
    Multiply projected housing units by projected PPH to get population.

    Both inputs must share a ``year`` column.  Returns a merged DataFrame
    with columns ``year``, ``hu_projected``, ``pph_projected``,
    ``population_hu``.

    Args:
        projected_hu: Output of :func:`trend_housing_units`.
        projected_pph: Output of :func:`project_pph`.

    Returns:
        DataFrame with HU, PPH, and resulting population for each year.
    """
    merged = projected_hu.merge(projected_pph, on="year", how="inner")
    merged["population_hu"] = merged["hu_projected"] * merged["pph_projected"]
    return merged


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_housing_unit_projections(
    config: dict[str, Any],
    place_fips_list: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the housing-unit projection for all eligible places.

    Eligible = places with at least ``min_history_years`` vintages of
    housing-unit data.

    Args:
        config: Full projection config dict.
        place_fips_list: Optional whitelist of place FIPS codes to project.
                         If None, project all places with sufficient data.

    Returns:
        DataFrame with columns: place_fips, year, hu_projected,
        pph_projected, population_hu, method.
    """
    hu_cfg = config.get("housing_unit_method", {})
    trend_method = hu_cfg.get("trend_method", "log_linear")
    pph_method = hu_cfg.get("pph_method", "hold_last")
    min_history = hu_cfg.get("min_history_years", 3)
    proj_years = hu_cfg.get("projection_years", [2025, 2030, 2035])

    housing_df = load_housing_data(config)

    if place_fips_list is not None:
        housing_df = housing_df[housing_df["place_fips"].isin(place_fips_list)]

    results: list[pd.DataFrame] = []

    for place_fips, place_df in housing_df.groupby("place_fips"):
        n_years = place_df["year"].nunique()
        if n_years < min_history:
            logger.debug(
                "Skipping %s: only %d vintage(s) (need %d)",
                place_fips, n_years, min_history,
            )
            continue

        try:
            hu_proj = trend_housing_units(
                place_df[["year", "housing_units"]].drop_duplicates(),
                method=trend_method,
                projection_years=proj_years,
            )
            pph_proj = project_pph(
                place_df[["year", "avg_hh_size"]].dropna().drop_duplicates(),
                method=pph_method,
                projection_years=proj_years,
            )
            pop = project_population_from_hu(hu_proj, pph_proj)
            pop["place_fips"] = str(place_fips)
            pop["method"] = f"hu_{trend_method}"
            results.append(pop)
        except Exception:
            logger.warning("HU projection failed for %s", place_fips, exc_info=True)

    if not results:
        logger.warning("No places produced HU projections")
        return pd.DataFrame(
            columns=["place_fips", "year", "hu_projected", "pph_projected", "population_hu", "method"]
        )

    combined = pd.concat(results, ignore_index=True)
    combined = combined[["place_fips", "year", "hu_projected", "pph_projected", "population_hu", "method"]]
    combined = combined.sort_values(["place_fips", "year"]).reset_index(drop=True)

    logger.info(
        "HU projections: %d places, %d projection years",
        combined["place_fips"].nunique(),
        combined["year"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_with_share_trending(
    hu_projections: pd.DataFrame,
    share_trending_projections: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute divergence metrics between HU and share-trending projections.

    Both DataFrames must contain ``place_fips``, ``year``, and a population
    column (``population_hu`` for HU, ``population`` or ``total_population``
    for share-trending).

    Returns a DataFrame with columns:
        place_fips, year, pop_hu, pop_share, abs_diff, pct_diff

    ``pct_diff`` is (HU - share) / share * 100.  Positive means HU projects
    higher than share-trending.
    """
    # Determine the share-trending population column
    pop_col = "population"
    if pop_col not in share_trending_projections.columns:
        pop_col = "total_population"
    if pop_col not in share_trending_projections.columns:
        raise ValueError(
            "share_trending_projections must contain 'population' or 'total_population' column"
        )

    hu = hu_projections[["place_fips", "year", "population_hu"]].copy()
    st = share_trending_projections[["place_fips", "year", pop_col]].copy()
    st = st.rename(columns={pop_col: "pop_share"})

    merged = hu.merge(st, on=["place_fips", "year"], how="inner")
    merged = merged.rename(columns={"population_hu": "pop_hu"})
    merged["abs_diff"] = merged["pop_hu"] - merged["pop_share"]
    merged["pct_diff"] = np.where(
        merged["pop_share"] != 0,
        (merged["pop_hu"] - merged["pop_share"]) / merged["pop_share"] * 100,
        np.nan,
    )

    return merged.sort_values(["place_fips", "year"]).reset_index(drop=True)
