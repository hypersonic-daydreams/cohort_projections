"""Naive benchmark runner implementations for projection comparison.

Provides simple reference benchmark projections (carry-forward, linear trend,
average growth) and a component-swap helper for use with
:class:`BenchmarkComparisonModule`.

Each runner accepts a results DataFrame whose columns match
:class:`ProjectionResultRecord` and returns a new DataFrame in the same
schema with ``projected_value`` replaced by the benchmark projection.

References:
    - docs/plans/evaluation-blueprint.md, Module 4
    - cohort_projections/analysis/evaluation/benchmark_comparison.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that define a unique observation (mirrored from benchmark_comparison)
_JOIN_KEYS = [
    "geography",
    "geography_type",
    "year",
    "horizon",
    "sex",
    "age_group",
    "target",
]

# Minimum columns we require from input DataFrames
_REQUIRED_COLUMNS = {
    "geography",
    "year",
    "actual_value",
    "projected_value",
    "horizon",
    "target",
    "sex",
    "age_group",
}


def _check_required_columns(df: pd.DataFrame, label: str = "input") -> None:
    """Raise ``ValueError`` if *df* is missing required columns."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} DataFrame missing required columns: {sorted(missing)}"
        )


# ------------------------------------------------------------------
# Carry-forward benchmark
# ------------------------------------------------------------------


def carry_forward(results_df: pd.DataFrame, origin_year: int) -> pd.DataFrame:
    """Project the base-year actual value forward unchanged for all horizons.

    For each unique (geography, sex, age_group, target) series, the value at
    *origin_year* is carried forward as the ``projected_value`` for every
    subsequent year.

    Parameters
    ----------
    results_df : pd.DataFrame
        Projection results with at minimum the columns listed in
        ``_REQUIRED_COLUMNS``.
    origin_year : int
        The base year whose ``actual_value`` is carried forward.

    Returns
    -------
    pd.DataFrame
        Copy of *results_df* with ``projected_value`` replaced by the
        carry-forward projection and ``run_id`` set to
        ``"benchmark_carry_forward"``.
    """
    _check_required_columns(results_df, label="carry_forward input")

    group_cols = ["geography", "sex", "age_group", "target"]

    # Extract the base-year value for each series
    base_mask = results_df["year"] == origin_year
    if not base_mask.any():
        # Fall back: use the earliest year as proxy for origin
        base_mask = results_df["year"] == results_df["year"].min()

    base_values = (
        results_df.loc[base_mask, group_cols + ["actual_value"]]
        .drop_duplicates(subset=group_cols)
        .rename(columns={"actual_value": "_base_val"})
    )

    result = results_df.copy()
    result = result.merge(base_values, on=group_cols, how="left")
    result["projected_value"] = result["_base_val"]
    result.drop(columns=["_base_val"], inplace=True)
    result["run_id"] = "benchmark_carry_forward"
    return result


# ------------------------------------------------------------------
# Linear trend benchmark
# ------------------------------------------------------------------


def linear_trend(
    results_df: pd.DataFrame,
    origin_year: int,
    lookback: int = 5,
) -> pd.DataFrame:
    """Extrapolate a linear trend from historical data.

    For each unique (geography, sex, age_group, target) series, a simple
    linear regression is fit on ``actual_value`` for the *lookback* years
    ending at *origin_year*.  The fitted trend is then extrapolated forward.

    Parameters
    ----------
    results_df : pd.DataFrame
        Projection results.
    origin_year : int
        Last historical year (the trend is fitted up to this year).
    lookback : int
        Number of years of history to use for trend fitting.

    Returns
    -------
    pd.DataFrame
        Copy of *results_df* with ``projected_value`` replaced by the
        trend extrapolation and ``run_id`` set to
        ``"benchmark_linear_trend"``.
    """
    _check_required_columns(results_df, label="linear_trend input")

    group_cols = ["geography", "sex", "age_group", "target"]

    # Identify historical window
    min_year = origin_year - lookback + 1
    hist_mask = (results_df["year"] >= min_year) & (
        results_df["year"] <= origin_year
    )
    hist = results_df.loc[hist_mask].copy()

    # Fit per-series slopes and intercepts
    slopes: dict[tuple, float] = {}
    intercepts: dict[tuple, float] = {}

    for key, grp in hist.groupby(group_cols):
        years = grp["year"].values.astype(float)
        values = grp["actual_value"].values.astype(float)
        if len(years) < 2:
            # Not enough data for a trend — fall back to constant
            slopes[key] = 0.0
            intercepts[key] = float(values[-1]) if len(values) > 0 else 0.0
        else:
            coeffs = np.polyfit(years, values, 1)
            slopes[key] = float(coeffs[0])
            intercepts[key] = float(coeffs[1])

    result = results_df.copy()

    def _predict(row: pd.Series) -> float:
        key = tuple(row[col] for col in group_cols)
        m = slopes.get(key, 0.0)
        b = intercepts.get(key, row["actual_value"])
        return m * row["year"] + b

    result["projected_value"] = result.apply(_predict, axis=1)
    result["run_id"] = "benchmark_linear_trend"
    return result


# ------------------------------------------------------------------
# Average-growth benchmark
# ------------------------------------------------------------------


def average_growth(
    results_df: pd.DataFrame,
    origin_year: int,
    lookback: int = 5,
) -> pd.DataFrame:
    """Apply the average historical growth rate forward.

    For each unique (geography, sex, age_group, target) series, computes
    the mean year-over-year growth rate during the *lookback* period and
    compounds it forward from the *origin_year* actual value.

    Parameters
    ----------
    results_df : pd.DataFrame
        Projection results.
    origin_year : int
        Last historical year.
    lookback : int
        Number of years of history to compute growth from.

    Returns
    -------
    pd.DataFrame
        Copy of *results_df* with ``projected_value`` replaced by the
        average-growth projection and ``run_id`` set to
        ``"benchmark_average_growth"``.
    """
    _check_required_columns(results_df, label="average_growth input")

    group_cols = ["geography", "sex", "age_group", "target"]

    # Historical window
    min_year = origin_year - lookback + 1
    hist_mask = (results_df["year"] >= min_year) & (
        results_df["year"] <= origin_year
    )
    hist = results_df.loc[hist_mask].copy()

    # Compute mean growth rate and base value per series
    mean_rates: dict[tuple, float] = {}
    base_values: dict[tuple, float] = {}

    for key, grp in hist.groupby(group_cols):
        sorted_grp = grp.sort_values("year")
        values = sorted_grp["actual_value"].values.astype(float)

        # Base value at origin year
        origin_rows = sorted_grp[sorted_grp["year"] == origin_year]
        if not origin_rows.empty:
            base_values[key] = float(origin_rows["actual_value"].iloc[0])
        else:
            base_values[key] = float(values[-1]) if len(values) > 0 else 0.0

        # Compute year-over-year growth rates (avoid division by zero)
        if len(values) < 2:
            mean_rates[key] = 0.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.diff(values) / np.where(
                    values[:-1] != 0, values[:-1], np.nan
                )
            valid_rates = rates[np.isfinite(rates)]
            mean_rates[key] = (
                float(np.mean(valid_rates)) if len(valid_rates) > 0 else 0.0
            )

    result = results_df.copy()

    def _project(row: pd.Series) -> float:
        key = tuple(row[col] for col in group_cols)
        base = base_values.get(key, row["actual_value"])
        rate = mean_rates.get(key, 0.0)
        years_ahead = row["year"] - origin_year
        if years_ahead <= 0:
            return base
        return base * (1.0 + rate) ** years_ahead

    result["projected_value"] = result.apply(_project, axis=1)
    result["run_id"] = "benchmark_average_growth"
    return result


# ------------------------------------------------------------------
# Component-swap helper
# ------------------------------------------------------------------


def build_component_swap(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    components: list[str],
    label_a: str = "methodA",
    label_b: str = "methodB",
) -> pd.DataFrame:
    """Build a component-swapped result DataFrame.

    For the specified *components* (matched against the ``target`` column),
    uses ``method_b``'s ``projected_value``; for all other targets, uses
    ``method_a``'s values.

    Parameters
    ----------
    method_a : pd.DataFrame
        Base projection results.
    method_b : pd.DataFrame
        Donor projection results (provides values for swapped components).
    components : list[str]
        Target names (e.g. ``["births", "deaths"]``) to take from *method_b*.
    label_a : str
        Label for the base method (used in the composite ``run_id``).
    label_b : str
        Label for the donor method.

    Returns
    -------
    pd.DataFrame
        Combined result with ``run_id`` set to
        ``"{label_a}_swap_{components}_from_{label_b}"``.
    """
    _check_required_columns(method_a, label="method_a")
    _check_required_columns(method_b, label="method_b")

    # Split method_a into swapped and kept portions
    swap_mask_a = method_a["target"].isin(components)
    kept = method_a.loc[~swap_mask_a].copy()

    # Get the swapped components from method_b
    swap_mask_b = method_b["target"].isin(components)
    swapped = method_b.loc[swap_mask_b].copy()

    result = pd.concat([kept, swapped], ignore_index=True)

    comp_str = "_".join(sorted(components))
    result["run_id"] = f"{label_a}_swap_{comp_str}_from_{label_b}"
    return result
