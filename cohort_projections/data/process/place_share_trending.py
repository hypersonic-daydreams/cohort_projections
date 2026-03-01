"""
Place share-trending model utilities for PP-003 Phase 2 (IMP-05).

Implements the S04 logit-linear share-trending specification, including:
1. Logit/inverse-logit transforms with epsilon clamping.
2. OLS and recency-weighted WLS trend fitting on centered time.
3. Forward share projection on the logit scale.
4. Constraint mechanisms (proportional rescaling and cap-and-redistribute).
5. Balance-of-county reconciliation and county-level orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

BALANCE_KEY = "balance_of_county"
DEFAULT_EPSILON = 0.001
DEFAULT_LAMBDA_DECAY = 0.9
DEFAULT_RECONCILIATION_FLAG_THRESHOLD = 0.05


@dataclass(frozen=True)
class ReconciliationResult:
    """Container for county-share reconciliation outputs."""

    place_shares: dict[str, float]
    balance_share: float
    total_before_adjustment: float
    adjustment: float
    flagged: bool


def _coalesce_float(default: float, *values: object) -> float:
    """Return the first present value, coerced to float."""
    for value in values:
        if value is not None:
            value_any: Any = value
            return float(value_any)
    return default


def _coalesce_int(default: int, *values: object) -> int:
    """Return the first present value, coerced to int."""
    for value in values:
        if value is not None:
            value_any: Any = value
            return int(value_any)
    return default


def _coalesce_str(default: str, *values: object) -> str:
    """Return the first present value, coerced to str."""
    for value in values:
        if value is not None:
            return str(value)
    return default


def _as_1d_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert values to a finite, 1D NumPy float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _normalize_component_id(value: Any) -> str:
    """Normalize component identifiers to stable string keys."""
    if value is None or pd.isna(value):
        raise ValueError("Component identifier cannot be null.")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text.removesuffix(".0")
    return text


def _normalize_county_population_projection(
    county_pop_history: pd.DataFrame | None,
    projection_years: np.ndarray,
) -> dict[int, float]:
    """Normalize county population history to a year->population mapping."""
    if county_pop_history is None or county_pop_history.empty:
        return {}

    county = county_pop_history.copy()
    projection_year_set = {int(year) for year in projection_years}

    if "year" in county.columns:
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
                if col in county.columns
            ),
            None,
        )
        if pop_col is None:
            raise ValueError(
                "county_pop_history long format must include one of: "
                "county_population, population, total_population, pop_total, POPESTIMATE."
            )

        county["year"] = pd.to_numeric(county["year"], errors="coerce").astype("Int64")
        county[pop_col] = pd.to_numeric(county[pop_col], errors="coerce")
        county = county.dropna(subset=["year", pop_col]).copy()
        county["year"] = county["year"].astype(int)
        county = county[county["year"].isin(projection_year_set)].copy()
        if county.duplicated(subset=["year"]).any():
            raise ValueError("county_pop_history has duplicate year rows.")
        return dict(zip(county["year"], county[pop_col], strict=True))

    pop_cols = [
        col
        for col in county.columns
        if col.startswith("POPESTIMATE") and col[-4:].isdigit()
    ]
    if not pop_cols:
        raise ValueError(
            "county_pop_history must be long with year/pop columns or wide with POPESTIMATEYYYY."
        )

    melted = county.melt(value_vars=pop_cols, var_name="population_column", value_name="county_population")
    melted["year"] = melted["population_column"].str.extract(r"(\d{4})").astype(int)
    melted["county_population"] = pd.to_numeric(melted["county_population"], errors="coerce")
    melted = melted.dropna(subset=["county_population"]).copy()
    melted = melted[melted["year"].isin(projection_year_set)].copy()
    if melted.duplicated(subset=["year"]).any():
        raise ValueError("county_pop_history wide format resolves to duplicate year rows.")
    return dict(zip(melted["year"], melted["county_population"], strict=True))


def logit_transform(
    shares: Sequence[float] | np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Clamp shares and apply the logit transform.

    Args:
        shares: Share values in [0, 1].
        epsilon: Clamp bound. Shares are constrained to [epsilon, 1-epsilon].

    Returns:
        Logit-transformed values as a NumPy array.
    """
    if not (0 < epsilon < 0.5):
        raise ValueError("epsilon must be in (0, 0.5).")

    share_arr = np.asarray(shares, dtype=float)
    if not np.isfinite(share_arr).all():
        raise ValueError("shares contains non-finite values.")

    clamped = np.clip(share_arr, epsilon, 1.0 - epsilon)
    return np.log(clamped / (1.0 - clamped))


def inverse_logit(logit_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Apply the inverse logit transform.

    Args:
        logit_values: Values on the logit scale.

    Returns:
        Shares in (0, 1).
    """
    values = np.asarray(logit_values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("logit_values contains non-finite values.")

    positive = values >= 0
    out = np.empty_like(values, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def compute_recency_weights(
    years: Sequence[float] | np.ndarray,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
) -> np.ndarray:
    """
    Compute exponential recency weights for WLS fitting.

    Args:
        years: Year values in the fitting window.
        lambda_decay: Exponential decay parameter (0, 1].

    Returns:
        Weight vector where the latest year has weight 1.
    """
    if not (0 < lambda_decay <= 1):
        raise ValueError("lambda_decay must be in (0, 1].")

    year_arr = _as_1d_float_array(years, name="years")
    max_year = float(np.max(year_arr))
    return np.power(lambda_decay, max_year - year_arr)


def fit_share_trend(
    logit_shares: Sequence[float] | np.ndarray,
    years: Sequence[float] | np.ndarray,
    method: str = "ols",
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
) -> tuple[float, float]:
    """
    Fit logit-linear share trend coefficients.

    Time is centered at the midpoint (mean) of the fitting window:
    ``x = year - mean(year)``.

    Args:
        logit_shares: Shares transformed via ``logit_transform``.
        years: Matching year vector.
        method: ``"ols"`` or ``"wls"``.
        lambda_decay: Recency decay for WLS.

    Returns:
        Tuple of ``(intercept, slope)`` for the centered-time model.
    """
    y = _as_1d_float_array(logit_shares, name="logit_shares")
    t = _as_1d_float_array(years, name="years")

    if len(y) != len(t):
        raise ValueError("logit_shares and years must have the same length.")
    if len(y) < 2:
        raise ValueError("At least two observations are required to fit a trend.")

    center_year = float(np.mean(t))
    x = t - center_year
    design = np.column_stack([np.ones(len(x)), x])

    method_lower = method.lower()
    if method_lower == "ols":
        weighted_design = design
        weighted_y = y
    elif method_lower == "wls":
        weights = compute_recency_weights(t, lambda_decay=lambda_decay)
        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        weighted_y = y * sqrt_weights
    else:
        raise ValueError("method must be 'ols' or 'wls'.")

    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    intercept, slope = coefficients
    return float(intercept), float(slope)


def project_shares(
    intercept: float,
    slope: float,
    projection_years: Sequence[float] | np.ndarray,
    center_year: float,
) -> np.ndarray:
    """
    Project shares forward from fitted centered-time coefficients.

    Args:
        intercept: Trend intercept (at ``center_year``).
        slope: Trend slope per year.
        projection_years: Years to project.
        center_year: Time-centering year used in fitting.

    Returns:
        Projected shares in (0, 1).
    """
    years = _as_1d_float_array(projection_years, name="projection_years")
    linear_term = intercept + slope * (years - center_year)
    return inverse_logit(linear_term)


def apply_proportional_rescaling(shares_dict: Mapping[str, float]) -> dict[str, float]:
    """
    Rescale all shares proportionally to sum to 1.

    Args:
        shares_dict: Component shares to rescale.

    Returns:
        Rescaled shares with exact sum of 1 (up to floating-point precision).
    """
    if not shares_dict:
        return {}

    adjusted = {key: float(value) for key, value in shares_dict.items()}
    if any(value < 0 for value in adjusted.values()):
        raise ValueError("Shares must be non-negative for rescaling.")

    total = sum(adjusted.values())
    if total <= 0:
        raise ValueError("Share total must be positive for rescaling.")

    return {key: value / total for key, value in adjusted.items()}


def apply_cap_and_redistribute(
    shares_dict: Mapping[str, float],
    base_shares_dict: Mapping[str, float],
) -> dict[str, float]:
    """
    Apply cap-and-redistribute reconciliation.

    Declining/stable components remain unchanged; discrepancy is redistributed
    only among components that grew versus base-year shares.

    Args:
        shares_dict: Current projected shares.
        base_shares_dict: Base-year shares for the same keys.

    Returns:
        Adjusted share dictionary summing to 1.
    """
    if not shares_dict:
        return {}

    current = {key: float(value) for key, value in shares_dict.items()}
    base = {key: float(value) for key, value in base_shares_dict.items()}

    missing_base = set(current) - set(base)
    if missing_base:
        missing_text = ", ".join(sorted(missing_base))
        raise ValueError(f"base_shares_dict missing keys: {missing_text}")

    if any(value < 0 for value in current.values()):
        raise ValueError("Shares must be non-negative for cap-and-redistribute.")

    total = sum(current.values())
    if np.isclose(total, 1.0, atol=1e-12):
        return current

    growing = {key for key, value in current.items() if value > base[key]}
    if not growing:
        return apply_proportional_rescaling(current)
    if len(growing) == len(current):
        # S04 edge case: if all components grew, cap-and-redistribute collapses
        # to proportional rescaling.
        return apply_proportional_rescaling(current)

    adjusted = current.copy()
    active = set(growing)

    while active:
        discrepancy = sum(adjusted.values()) - 1.0
        if np.isclose(discrepancy, 0.0, atol=1e-12):
            break

        active_total = sum(adjusted[key] for key in active)
        if active_total <= 0:
            break

        for key in active:
            adjusted[key] -= discrepancy * (adjusted[key] / active_total)

        # Over-correction can only occur when removing excess (discrepancy > 0).
        if discrepancy <= 0:
            break

        clamped = {key for key in active if adjusted[key] < base[key]}
        if not clamped:
            break

        for key in clamped:
            adjusted[key] = base[key]
        active -= clamped

    adjusted = {key: max(0.0, value) for key, value in adjusted.items()}
    final_total = sum(adjusted.values())
    if not np.isclose(final_total, 1.0, atol=1e-10):
        adjusted = apply_proportional_rescaling(adjusted)

    return adjusted


def reconcile_county_shares(
    place_shares: Mapping[str, float],
    balance_share: float,
    constraint_method: str,
    base_shares: Mapping[str, float],
    flag_threshold: float = DEFAULT_RECONCILIATION_FLAG_THRESHOLD,
    tolerance: float = 1e-9,
) -> ReconciliationResult:
    """
    Reconcile place shares with balance-of-county share to sum to 1.

    Args:
        place_shares: Projected place shares by place identifier.
        balance_share: Independently projected balance-of-county share.
        constraint_method: ``"proportional"`` or ``"cap_and_redistribute"``.
        base_shares: Base-year shares used by cap-and-redistribute.
        flag_threshold: QA flag threshold for pre-reconciliation discrepancy.
        tolerance: No-adjustment tolerance around 1.0.

    Returns:
        ``ReconciliationResult`` with adjusted shares and QA metadata.
    """
    if not (0 <= balance_share <= 1):
        raise ValueError("balance_share must be in [0, 1].")
    if flag_threshold < 0:
        raise ValueError("flag_threshold must be non-negative.")

    place = {key: float(value) for key, value in place_shares.items()}
    if any(value < 0 for value in place.values()):
        raise ValueError("place_shares must be non-negative.")

    combined = dict(place)
    combined[BALANCE_KEY] = float(balance_share)

    total_before_adjustment = sum(combined.values())
    adjustment = abs(total_before_adjustment - 1.0)

    if adjustment <= tolerance:
        adjusted = combined
    else:
        method = constraint_method.lower()
        if method in {"proportional", "proportional_rescaling"}:
            adjusted = apply_proportional_rescaling(combined)
        elif method in {"cap_and_redistribute", "cap-and-redistribute"}:
            if not base_shares:
                raise ValueError("base_shares is required for cap_and_redistribute.")
            base_combined = {key: float(base_shares.get(key, combined[key])) for key in combined}
            adjusted = apply_cap_and_redistribute(combined, base_combined)
        else:
            raise ValueError(
                "constraint_method must be 'proportional' or 'cap_and_redistribute'."
            )

    adjusted_place_shares = {
        key: value for key, value in adjusted.items() if key != BALANCE_KEY
    }
    adjusted_balance_share = float(adjusted[BALANCE_KEY])
    flagged = adjustment > flag_threshold

    return ReconciliationResult(
        place_shares=adjusted_place_shares,
        balance_share=adjusted_balance_share,
        total_before_adjustment=float(total_before_adjustment),
        adjustment=float(adjustment),
        flagged=flagged,
    )


def trend_all_places_in_county(
    place_share_history: pd.DataFrame,
    county_pop_history: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Fit and project place shares for a single county, then reconcile to 1.0.

    Args:
        place_share_history: Historical share rows for one county. Must include
            ``year`` and either ``share_raw`` or ``share``. If ``row_type`` is
            present, balance rows should be labeled ``balance_of_county``.
        county_pop_history: Optional county population history/projections with
            ``year`` and one population column (or POPESTIMATEYYYY wide format).
        config: Model configuration. Supports both a nested
            ``place_projections`` block and direct overrides:
            ``epsilon``, ``lambda_decay``, ``fitting_method``,
            ``constraint_method``, ``projection_years``, ``base_year``,
            ``end_year``, ``reconciliation_flag_threshold``.

    Returns:
        Long DataFrame of projected shares for places and balance rows.
    """
    output_columns = [
        "county_fips",
        "year",
        "row_type",
        "place_fips",
        "projected_share_raw",
        "projected_share",
        "county_population",
        "projected_population",
        "base_share",
        "reconciliation_adjustment",
        "reconciliation_flag",
        "fitting_method",
        "constraint_method",
    ]
    if place_share_history.empty:
        return pd.DataFrame(columns=output_columns)

    place_config = config.get("place_projections", {})
    model_config = place_config.get("model", {}) if isinstance(place_config, Mapping) else {}
    output_config = place_config.get("output", {}) if isinstance(place_config, Mapping) else {}

    epsilon = _coalesce_float(
        DEFAULT_EPSILON,
        config.get("epsilon"),
        model_config.get("epsilon"),
    )
    lambda_decay = _coalesce_float(
        DEFAULT_LAMBDA_DECAY,
        config.get("lambda_decay"),
        model_config.get("lambda_decay"),
    )
    fitting_method = _coalesce_str(
        "ols",
        config.get("fitting_method"),
        config.get("method"),
        model_config.get("fitting_method"),
    )
    constraint_method = _coalesce_str(
        "proportional",
        config.get("constraint_method"),
        model_config.get("constraint_method"),
    )
    flag_threshold = _coalesce_float(
        DEFAULT_RECONCILIATION_FLAG_THRESHOLD,
        config.get("reconciliation_flag_threshold"),
        model_config.get("reconciliation_flag_threshold"),
    )

    projection_years_raw = config.get("projection_years")
    if projection_years_raw is None:
        base_year = _coalesce_int(
            2025,
            config.get("base_year"),
            output_config.get("base_year"),
        )
        end_year = _coalesce_int(
            base_year,
            config.get("end_year"),
            output_config.get("end_year"),
        )
        projection_years = np.arange(base_year, end_year + 1, dtype=float)
    else:
        projection_years = _as_1d_float_array(projection_years_raw, name="projection_years")
    if len(projection_years) == 0:
        raise ValueError("projection_years cannot be empty.")

    history = place_share_history.copy()
    if "year" not in history.columns:
        raise ValueError("place_share_history must include 'year'.")
    share_col = "share_raw" if "share_raw" in history.columns else "share"
    if share_col not in history.columns:
        raise ValueError("place_share_history must include 'share_raw' or 'share'.")

    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype("Int64")
    history[share_col] = pd.to_numeric(history[share_col], errors="coerce")
    history = history.dropna(subset=["year", share_col]).copy()
    history["year"] = history["year"].astype(int)

    if "row_type" in history.columns:
        balance_mask = history["row_type"].astype(str).eq(BALANCE_KEY)
    elif "place_fips" in history.columns:
        balance_mask = history["place_fips"].isna()
    else:
        raise ValueError(
            "place_share_history must include either 'row_type' or 'place_fips' for balance rows."
        )

    place_rows = history[~balance_mask].copy()
    if place_rows.empty:
        return pd.DataFrame(columns=output_columns)

    if "place_fips" not in place_rows.columns:
        raise ValueError("place_share_history place rows must include 'place_fips'.")
    place_rows["place_fips"] = place_rows["place_fips"].map(_normalize_component_id)

    balance_rows = history[balance_mask].copy()
    if balance_rows.empty:
        derived_balance = (
            place_rows.groupby("year", as_index=False)[share_col].sum().rename(columns={share_col: "place_sum"})
        )
        derived_balance[share_col] = 1.0 - derived_balance["place_sum"]
        balance_rows = derived_balance[["year", share_col]]
    else:
        balance_rows = (
            balance_rows.groupby("year", as_index=False)[share_col].sum().reset_index(drop=True)
        )

    county_fips = pd.NA
    if "county_fips" in history.columns:
        county_values = history["county_fips"].dropna().astype(str).unique()
        if len(county_values) == 1:
            county_fips = county_values[0]

    place_projection_raw: dict[str, np.ndarray] = {}
    for place_fips, group in place_rows.groupby("place_fips"):
        group = group.sort_values("year")
        years = group["year"].to_numpy(dtype=float)
        shares = group[share_col].to_numpy(dtype=float)
        logits = logit_transform(shares, epsilon=epsilon)
        intercept, slope = fit_share_trend(
            logit_shares=logits,
            years=years,
            method=fitting_method,
            lambda_decay=lambda_decay,
        )
        center_year = float(np.mean(years))
        place_projection_raw[place_fips] = project_shares(
            intercept=intercept,
            slope=slope,
            projection_years=projection_years,
            center_year=center_year,
        )

    balance_rows = balance_rows.sort_values("year")
    balance_years = balance_rows["year"].to_numpy(dtype=float)
    balance_shares = balance_rows[share_col].to_numpy(dtype=float)
    balance_logits = logit_transform(balance_shares, epsilon=epsilon)
    balance_intercept, balance_slope = fit_share_trend(
        logit_shares=balance_logits,
        years=balance_years,
        method=fitting_method,
        lambda_decay=lambda_decay,
    )
    balance_center_year = float(np.mean(balance_years))
    balance_projection_raw = project_shares(
        intercept=balance_intercept,
        slope=balance_slope,
        projection_years=projection_years,
        center_year=balance_center_year,
    )

    base_shares = {place_fips: float(values[0]) for place_fips, values in place_projection_raw.items()}
    base_shares[BALANCE_KEY] = float(balance_projection_raw[0])

    county_population_map = _normalize_county_population_projection(
        county_pop_history=county_pop_history,
        projection_years=projection_years,
    )

    records: list[dict[str, Any]] = []
    projection_year_int = projection_years.astype(int)
    for idx, year in enumerate(projection_year_int):
        raw_place_shares = {
            place_fips: float(values[idx]) for place_fips, values in place_projection_raw.items()
        }
        raw_balance_share = float(balance_projection_raw[idx])

        reconciliation = reconcile_county_shares(
            place_shares=raw_place_shares,
            balance_share=raw_balance_share,
            constraint_method=constraint_method,
            base_shares=base_shares,
            flag_threshold=flag_threshold,
        )

        county_population = county_population_map.get(int(year))

        for place_fips, share in reconciliation.place_shares.items():
            projected_population = np.nan
            if county_population is not None:
                projected_population = 0.0 if share < epsilon else share * county_population

            records.append(
                {
                    "county_fips": county_fips,
                    "year": int(year),
                    "row_type": "place",
                    "place_fips": place_fips,
                    "projected_share_raw": raw_place_shares[place_fips],
                    "projected_share": share,
                    "county_population": county_population,
                    "projected_population": projected_population,
                    "base_share": base_shares[place_fips],
                    "reconciliation_adjustment": reconciliation.adjustment,
                    "reconciliation_flag": reconciliation.flagged,
                    "fitting_method": fitting_method,
                    "constraint_method": constraint_method,
                }
            )

        balance_population = np.nan
        if county_population is not None:
            balance_population = reconciliation.balance_share * county_population

        records.append(
            {
                "county_fips": county_fips,
                "year": int(year),
                "row_type": BALANCE_KEY,
                "place_fips": pd.NA,
                "projected_share_raw": raw_balance_share,
                "projected_share": reconciliation.balance_share,
                "county_population": county_population,
                "projected_population": balance_population,
                "base_share": base_shares[BALANCE_KEY],
                "reconciliation_adjustment": reconciliation.adjustment,
                "reconciliation_flag": reconciliation.flagged,
                "fitting_method": fitting_method,
                "constraint_method": constraint_method,
            }
        )

    output = pd.DataFrame(records, columns=output_columns)
    output = output.sort_values(["year", "row_type", "place_fips"], na_position="last").reset_index(
        drop=True
    )

    logger.info(
        "Projected place shares for county %s across %d years (%d place rows, %d balance rows).",
        county_fips,
        len(projection_years),
        len(output[output["row_type"] == "place"]),
        len(output[output["row_type"] == BALANCE_KEY]),
    )
    return output


__all__ = [
    "BALANCE_KEY",
    "DEFAULT_EPSILON",
    "DEFAULT_LAMBDA_DECAY",
    "DEFAULT_RECONCILIATION_FLAG_THRESHOLD",
    "ReconciliationResult",
    "apply_cap_and_redistribute",
    "apply_proportional_rescaling",
    "compute_recency_weights",
    "fit_share_trend",
    "inverse_logit",
    "logit_transform",
    "project_shares",
    "reconcile_county_shares",
    "trend_all_places_in_county",
]
