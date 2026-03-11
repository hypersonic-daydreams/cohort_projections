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
    mape,
    spearman_rank_correlation,
    wape,
)
from .schemas import METRIC_REGISTRY, PROJECTION_RESULT_COLUMNS
from .utils import (
    compute_grouped_metrics,
    make_diagnostic_record,
    resolve_county_group,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


class ForecastAccuracyModule:
    """Module 1 of the Evaluation Blueprint: Forecast Accuracy.

    Args:
        county_groups: Mapping of group name to list of FIPS codes.
            Expected groups: ``bakken``, ``reservation``, ``urban_college``.
            Counties not listed are assigned to ``rural``.
        regimes: Mapping of regime name to ``{"start": int, "end": int}``
            defining historical year ranges.  If provided,
            :meth:`compute_all_metrics` will include regime-stratified
            accuracy rows.
    """

    def __init__(
        self,
        county_groups: dict[str, list[str]] | None = None,
        regimes: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.county_groups: dict[str, list[str]] = county_groups or {}
        self.regimes: dict[str, dict[str, int]] = regimes or {}

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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
        parts: list[pd.DataFrame] = [
            self.accuracy_by_geography_horizon(df, target="population"),
            self.accuracy_by_age_group(df),
            self.bias_summary(df),
            self.rank_direction_tests(df),
            self.weighted_vs_unweighted_comparison(df),
        ]
        if self.regimes:
            parts.append(self.accuracy_by_regime(df, self.regimes))
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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
        ].copy()

        rows = compute_grouped_metrics(
            subset,
            groupby_cols=["run_id", "geography", "horizon"],
            metrics=METRIC_REGISTRY,
            county_groups=self.county_groups,
            default_target=target,
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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
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
                else resolve_county_group(str(geography), self.county_groups)
            )
            proj = grp["projected_value"]
            act = grp["actual_value"]

            for metric_name, (metric_fn, metric_group) in METRIC_REGISTRY.items():
                val = metric_fn(proj, act)
                rows.append(
                    make_diagnostic_record(
                        run_id=str(run_id),
                        metric_name=f"{metric_name}__age_{age_group}",
                        metric_group=metric_group,
                        geography=str(geography),
                        target=target,
                        value=val,
                        geography_group=geo_group,
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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
        ].copy()

        bias_metrics = {
            k: v
            for k, v in METRIC_REGISTRY.items()
            if v[1] == "bias"
        }

        rows = compute_grouped_metrics(
            subset,
            groupby_cols=["run_id", "geography", "horizon"],
            metrics=bias_metrics,
            county_groups=self.county_groups,
            default_target=target,
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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
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
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="spearman_rank_correlation",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=spearman_rank_correlation(pg, ag),
                    geography_group="all",
                    horizon=int(horizon),
                )
            )

            # Directional accuracy
            rows.append(
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="directional_accuracy",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=directional_accuracy(pg, ag),
                    geography_group="all",
                    horizon=int(horizon),
                )
            )

            # Top-decile capture
            rows.append(
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="top_decile_capture",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=decile_capture(pg, ag, quantile=0.1, tail="top"),
                    geography_group="all",
                    horizon=int(horizon),
                )
            )

            # Bottom-decile capture
            rows.append(
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="bottom_decile_capture",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=decile_capture(pg, ag, quantile=0.1, tail="bottom"),
                    geography_group="all",
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
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
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
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="mape",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=mape_val,
                    geography_group="all",
                    horizon=int(horizon),
                    notes="unweighted (cross-county)",
                )
            )
            rows.append(
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="wape",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=wape_val,
                    geography_group="all",
                    horizon=int(horizon),
                    notes="population-weighted (cross-county)",
                )
            )

            ratio = wape_val / mape_val if mape_val != 0 else float("nan")
            rows.append(
                make_diagnostic_record(
                    run_id=str(run_id),
                    metric_name="wape_mape_ratio",
                    metric_group="accuracy",
                    geography="all_counties",
                    target=target,
                    value=ratio,
                    geography_group="all",
                    horizon=int(horizon),
                    notes="<1 means large counties more accurate",
                )
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 7. Regime-specific accuracy
    # ------------------------------------------------------------------

    def accuracy_by_regime(
        self,
        df: pd.DataFrame,
        regimes: dict[str, dict[str, int]] | None = None,
        target: str = "population",
    ) -> pd.DataFrame:
        """Accuracy metrics stratified by historical regime and geography group.

        Each projection year is assigned to the regime whose ``[start, end]``
        range contains it.  Rows whose year falls outside all regimes are
        silently excluded.

        Args:
            df: Projection-result DataFrame.
            regimes: Mapping of regime name to ``{"start": int, "end": int}``.
                If ``None``, falls back to ``self.regimes``.
            target: Target variable (default ``"population"``).

        Returns:
            DataFrame of :class:`DiagnosticRecord`-compatible rows.  Each
            row carries a ``notes`` field of the form ``regime=<name>``.
        """
        validate_dataframe(df, PROJECTION_RESULT_COLUMNS, label="input")
        regimes = regimes if regimes is not None else self.regimes
        if not regimes:
            return pd.DataFrame()

        subset = df[
            (df["target"] == target)
            & (df["age_group"] == "total")
            & (df["sex"] == "total")
        ].copy()

        # Tag each row with its regime
        def _year_to_regime(year: int) -> str | None:
            for name, span in regimes.items():
                if span["start"] <= year <= span["end"]:
                    return name
            return None

        subset["regime"] = subset["year"].apply(_year_to_regime)
        subset = subset.dropna(subset=["regime"])

        if subset.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []

        for (run_id, regime, geography), grp in subset.groupby(
            ["run_id", "regime", "geography"]
        ):
            geo_type = grp["geography_type"].iloc[0]
            geo_group = (
                "state"
                if geo_type == "state"
                else resolve_county_group(str(geography), self.county_groups)
            )
            proj = grp["projected_value"]
            act = grp["actual_value"]

            for metric_name, (metric_fn, metric_group) in METRIC_REGISTRY.items():
                val = metric_fn(proj, act)
                rows.append(
                    make_diagnostic_record(
                        run_id=str(run_id),
                        metric_name=f"{metric_name}__regime_{regime}",
                        metric_group=metric_group,
                        geography=str(geography),
                        target=target,
                        value=val,
                        geography_group=geo_group,
                        notes=f"regime={regime}",
                    )
                )

        return pd.DataFrame(rows)
