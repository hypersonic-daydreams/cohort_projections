"""Module 1: Forecast Accuracy evaluation.

Computes accuracy, bias, rank-correlation, and directional metrics for
population projections, stratified by geography, horizon, age group, and
county-type group.

All public methods return DataFrames whose rows are compatible with
:class:`DiagnosticRecord`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .metrics import (
    decile_capture,
    directional_accuracy,
    mae,
    mape,
    mean_signed_error,
    mean_signed_percentage_error,
    median_absolute_percentage_error,
    rmse,
    spearman_rank_correlation,
    wape,
)

logger = logging.getLogger(__name__)

# Mapping from config-friendly names to (function, is_signed) pairs.
_ACCURACY_METRICS: dict[str, tuple[Any, str]] = {
    "mae": (mae, "accuracy"),
    "rmse": (rmse, "accuracy"),
    "mape": (mape, "accuracy"),
    "median_ape": (median_absolute_percentage_error, "accuracy"),
    "wape": (wape, "accuracy"),
    "mean_signed_error": (mean_signed_error, "bias"),
    "mean_signed_percentage_error": (mean_signed_percentage_error, "bias"),
}

# Required input columns (subset of ProjectionResultRecord fields).
_REQUIRED_COLUMNS: set[str] = {
    "run_id",
    "geography",
    "geography_type",
    "year",
    "horizon",
    "sex",
    "age_group",
    "target",
    "projected_value",
    "actual_value",
    "base_value",
}


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* is missing required columns."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {sorted(missing)}")


def _make_diagnostic_row(
    run_id: str,
    metric_name: str,
    metric_group: str,
    geography: str,
    geography_group: str,
    target: str,
    value: float,
    horizon: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a dict compatible with :class:`DiagnosticRecord`."""
    return {
        "run_id": run_id,
        "metric_name": metric_name,
        "metric_group": metric_group,
        "geography": geography,
        "geography_group": geography_group,
        "target": target,
        "horizon": horizon,
        "value": value,
        "comparison_run_id": "",
        "notes": notes,
    }


def _resolve_county_group(
    fips: str,
    county_groups: dict[str, list[str]],
) -> str:
    """Return the county-type group a FIPS code belongs to, or ``'rural'``."""
    for group_name, fips_list in county_groups.items():
        if fips in fips_list:
            return group_name
    return "rural"


class ForecastAccuracyModule:
    """Module 1 of the Evaluation Blueprint: Forecast Accuracy.

    Args:
        county_groups: Mapping of group name to list of FIPS codes.
            Expected groups: ``bakken``, ``reservation``, ``urban_college``.
            Counties not listed are assigned to ``rural``.
    """

    def __init__(self, county_groups: dict[str, list[str]] | None = None) -> None:
        self.county_groups: dict[str, list[str]] = county_groups or {}

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_all_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all accuracy analyses and return a unified diagnostics table.

        Args:
            df: Tidy projection-result DataFrame with columns matching
                :class:`ProjectionResultRecord`.

        Returns:
            DataFrame with columns matching :class:`DiagnosticRecord`.
        """
        _validate_dataframe(df)
        parts: list[pd.DataFrame] = [
            self.accuracy_by_geography_horizon(df, target="population"),
            self.accuracy_by_age_group(df),
            self.bias_summary(df),
            self.rank_direction_tests(df),
            self.weighted_vs_unweighted_comparison(df),
        ]
        result = pd.concat(parts, ignore_index=True)
        logger.info("ForecastAccuracyModule: computed %d diagnostic rows", len(result))
        return result

    # ------------------------------------------------------------------
    # 1 & 2. Total population accuracy by geography and horizon
    # ------------------------------------------------------------------

    def accuracy_by_geography_horizon(
        self,
        df: pd.DataFrame,
        target: str = "population",
    ) -> pd.DataFrame:
        """Accuracy metrics for total population by geography and horizon.

        Computes county-level and state-level metrics for every horizon
        present in *df*.

        Args:
            df: Projection-result DataFrame.
            target: Target variable to evaluate (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.
        """
        _validate_dataframe(df)
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
        ].copy()

        rows: list[dict[str, Any]] = []

        for (run_id, geography, horizon), grp in subset.groupby(
            ["run_id", "geography", "horizon"]
        ):
            geo_type = grp["geography_type"].iloc[0]
            geo_group = (
                "state"
                if geo_type == "state"
                else _resolve_county_group(str(geography), self.county_groups)
            )
            proj = grp["projected_value"]
            act = grp["actual_value"]

            for metric_name, (metric_fn, metric_group) in _ACCURACY_METRICS.items():
                val = metric_fn(proj, act)
                rows.append(
                    _make_diagnostic_row(
                        run_id=str(run_id),
                        metric_name=metric_name,
                        metric_group=metric_group,
                        geography=str(geography),
                        geography_group=geo_group,
                        target=target,
                        value=val,
                        horizon=int(horizon),
                    )
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3 & 4. Age-group accuracy
    # ------------------------------------------------------------------

    def accuracy_by_age_group(
        self,
        df: pd.DataFrame,
        target: str = "population",
    ) -> pd.DataFrame:
        """Accuracy metrics stratified by age group, geography, and horizon.

        Args:
            df: Projection-result DataFrame.
            target: Target variable (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.
        """
        _validate_dataframe(df)
        subset = df[
            (df["target"] == target)
            & (df["age_group"] != "total")
            & (df["sex"] == "total")
        ].copy()

        rows: list[dict[str, Any]] = []

        for (run_id, geography, horizon, age_group), grp in subset.groupby(
            ["run_id", "geography", "horizon", "age_group"]
        ):
            geo_type = grp["geography_type"].iloc[0]
            geo_group = (
                "state"
                if geo_type == "state"
                else _resolve_county_group(str(geography), self.county_groups)
            )
            proj = grp["projected_value"]
            act = grp["actual_value"]

            for metric_name, (metric_fn, metric_group) in _ACCURACY_METRICS.items():
                val = metric_fn(proj, act)
                rows.append(
                    _make_diagnostic_row(
                        run_id=str(run_id),
                        metric_name=f"{metric_name}__age_{age_group}",
                        metric_group=metric_group,
                        geography=str(geography),
                        geography_group=geo_group,
                        target=target,
                        value=val,
                        horizon=int(horizon),
                        notes=f"age_group={age_group}",
                    )
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Signed bias summary
    # ------------------------------------------------------------------

    def bias_summary(
        self,
        df: pd.DataFrame,
        target: str = "population",
    ) -> pd.DataFrame:
        """Signed bias metrics by geography and horizon.

        Reports ``mean_signed_error`` and ``mean_signed_percentage_error``
        for total population at the county and state level.

        Args:
            df: Projection-result DataFrame.
            target: Target variable (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.
        """
        _validate_dataframe(df)
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
        ].copy()

        bias_metrics = {
            k: v
            for k, v in _ACCURACY_METRICS.items()
            if v[1] == "bias"
        }

        rows: list[dict[str, Any]] = []

        for (run_id, geography, horizon), grp in subset.groupby(
            ["run_id", "geography", "horizon"]
        ):
            geo_type = grp["geography_type"].iloc[0]
            geo_group = (
                "state"
                if geo_type == "state"
                else _resolve_county_group(str(geography), self.county_groups)
            )
            proj = grp["projected_value"]
            act = grp["actual_value"]

            for metric_name, (metric_fn, metric_group) in bias_metrics.items():
                val = metric_fn(proj, act)
                rows.append(
                    _make_diagnostic_row(
                        run_id=str(run_id),
                        metric_name=metric_name,
                        metric_group=metric_group,
                        geography=str(geography),
                        geography_group=geo_group,
                        target=target,
                        value=val,
                        horizon=int(horizon),
                    )
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Rank and direction tests
    # ------------------------------------------------------------------

    def rank_direction_tests(
        self,
        df: pd.DataFrame,
        target: str = "population",
    ) -> pd.DataFrame:
        """Spearman rank correlation, directional accuracy, and decile capture.

        Growth rates are computed as ``(projected_value - base_value) /
        base_value`` for projected and ``(actual_value - base_value) /
        base_value`` for actual.  Only county-level rows with
        ``age_group == 'total'`` and ``sex == 'total'`` are used.

        Args:
            df: Projection-result DataFrame.
            target: Target variable (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.
        """
        _validate_dataframe(df)
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
            & (df["geography_type"] == "county")
        ].copy()

        # Compute growth rates relative to base
        subset = subset.copy()
        mask = subset["base_value"] != 0
        subset.loc[mask, "proj_growth"] = (
            (subset.loc[mask, "projected_value"] - subset.loc[mask, "base_value"])
            / subset.loc[mask, "base_value"]
        )
        subset.loc[mask, "act_growth"] = (
            (subset.loc[mask, "actual_value"] - subset.loc[mask, "base_value"])
            / subset.loc[mask, "base_value"]
        )
        subset = subset.dropna(subset=["proj_growth", "act_growth"])

        rows: list[dict[str, Any]] = []

        for (run_id, horizon), grp in subset.groupby(["run_id", "horizon"]):
            pg = grp["proj_growth"].values
            ag = grp["act_growth"].values

            # Spearman rank correlation
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="spearman_rank_correlation",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=spearman_rank_correlation(pg, ag),
                    horizon=int(horizon),
                )
            )

            # Directional accuracy
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="directional_accuracy",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=directional_accuracy(pg, ag),
                    horizon=int(horizon),
                )
            )

            # Top-decile capture
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="top_decile_capture",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=decile_capture(pg, ag, quantile=0.1, tail="top"),
                    horizon=int(horizon),
                )
            )

            # Bottom-decile capture
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="bottom_decile_capture",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=decile_capture(pg, ag, quantile=0.1, tail="bottom"),
                    horizon=int(horizon),
                )
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. Weighted vs unweighted comparison
    # ------------------------------------------------------------------

    def weighted_vs_unweighted_comparison(
        self,
        df: pd.DataFrame,
        target: str = "population",
    ) -> pd.DataFrame:
        """Compare WAPE (population-weighted) to MAPE (unweighted).

        Reports both metrics and their ratio for each run/horizon
        combination, using county-level total-population rows.

        Args:
            df: Projection-result DataFrame.
            target: Target variable (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.
        """
        _validate_dataframe(df)
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
            & (df["geography_type"] == "county")
        ].copy()

        rows: list[dict[str, Any]] = []

        for (run_id, horizon), grp in subset.groupby(["run_id", "horizon"]):
            proj = grp["projected_value"]
            act = grp["actual_value"]

            mape_val = mape(proj, act)
            wape_val = wape(proj, act)

            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="mape",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=mape_val,
                    horizon=int(horizon),
                    notes="unweighted (cross-county)",
                )
            )
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="wape",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=wape_val,
                    horizon=int(horizon),
                    notes="population-weighted (cross-county)",
                )
            )

            ratio = wape_val / mape_val if mape_val != 0 else float("nan")
            rows.append(
                _make_diagnostic_row(
                    run_id=str(run_id),
                    metric_name="wape_mape_ratio",
                    metric_group="accuracy",
                    geography="all_counties",
                    geography_group="all",
                    target=target,
                    value=ratio,
                    horizon=int(horizon),
                    notes="<1 means large counties more accurate",
                )
            )

        return pd.DataFrame(rows)
