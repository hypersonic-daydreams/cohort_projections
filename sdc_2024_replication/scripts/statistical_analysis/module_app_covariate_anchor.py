#!/usr/bin/env python3
r"""
Module (Appendix): Covariate-Conditioned Near-Term Forecast Anchor (2025–2029)
=============================================================================

This module implements an appendix-only robustness check that conditions the
near-term forecast (2025–2029) of North Dakota \PEP net international migration
on auxiliary covariates:

- Refugee arrivals (RPC; FY totals, ND)
- LPR admissions (DHS; FY totals, ND)
- ACS moved-from-abroad proxy (B07007; optional sensitivity)

The design is explicitly scoped to *near-term anchoring* and does not replace
the manuscript's Moderate baseline scenario. See ADR-031 and the v0.8.6
runbook for scope and interpretation:

- ADR-031: docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md
- Spec: sdc_2024_replication/revisions/v0.8.6/covariate_forecasting_appendix_spec.md

Outputs:
- JSON bundle: sdc_2024_replication/scripts/statistical_analysis/results/module_app_covariate_anchor.json
- Forecast table (CSV): .../results/module_app_covariate_anchor_2025_2029.csv
- Figure (PNG/PDF): .../figures/fig_app_covariate_anchor_2025_2029.{png,pdf}
  (copied into journal_article/figures/ for LaTeX inclusion)

Usage:
    uv run python sdc_2024_replication/scripts/statistical_analysis/module_app_covariate_anchor.py
    uv run python sdc_2024_replication/scripts/statistical_analysis/module_app_covariate_anchor.py --data-source auto
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import data_loader


LOGGER = setup_logger(__name__)

DataSourceMode = Literal["auto", "db", "files"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _analysis_dir() -> Path:
    """Resolve the immigration analysis directory from configuration."""
    cfg = ConfigLoader().get_projection_config()
    processed_dir = (
        cfg.get("data_sources", {})
        .get("acs_moved_from_abroad", {})
        .get("processed_dir", "data/processed/immigration/analysis")
    )
    return _project_root() / processed_dir


def _results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def _figures_dir() -> Path:
    return Path(__file__).resolve().parent / "figures"


def _journal_figures_dir() -> Path:
    return Path(__file__).resolve().parent / "journal_article" / "figures"


@dataclass(frozen=True)
class CovariateSpec:
    """Specification for the appendix covariate-conditioned anchor."""

    tag: str
    description: str
    include_acs: bool


@dataclass(frozen=True)
class Standardization:
    """Z-score standardization metadata for exogenous regressors."""

    means: dict[str, float]
    stds: dict[str, float]

    def apply(self, exog: pd.DataFrame) -> pd.DataFrame:
        """Apply stored standardization to a new exog frame."""
        out = exog.copy()
        for col in out.columns:
            mean = self.means.get(col)
            std = self.stds.get(col)
            if mean is None or std is None:
                raise ValueError(f"Missing standardization parameters for column: {col}")
            if std == 0:
                out[col] = out[col] - mean
            else:
                out[col] = (out[col] - mean) / std
        return out


def _as_annual_period_index(index: pd.Index) -> pd.PeriodIndex:
    """Convert a year-like index to an annual PeriodIndex (stable in statsmodels)."""
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("Y")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("Y")
    years = [int(v) for v in index.to_list()]
    return pd.PeriodIndex([str(y) for y in years], freq="Y")


def _load_arima_baseline(results_dir: Path) -> pd.DataFrame:
    """Load ARIMA baseline forecast (2025–2029) from Module 2.1 outputs."""
    path = results_dir / "module_2_1_arima_model.json"
    with path.open() as f:
        payload = json.load(f)

    forecasts = payload.get("results", {}).get("forecasts", [])
    if not forecasts:
        raise ValueError(f"ARIMA baseline forecasts missing in {path}")

    rows: list[dict[str, Any]] = []
    for fc in forecasts:
        horizon = int(fc["horizon"])
        rows.append(
            {
                "year": 2024 + horizon,
                "baseline_point": float(fc["point"]),
                "baseline_se": float(fc["se"]),
                "baseline_ci80_lo": float(fc["ci_80"][0]),
                "baseline_ci80_hi": float(fc["ci_80"][1]),
                "baseline_ci95_lo": float(fc["ci_95"][0]),
                "baseline_ci95_hi": float(fc["ci_95"][1]),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _series_from_grouped_sum(
    df: pd.DataFrame, *, year_col: str, value_col: str, year_min: int, year_max: int
) -> pd.Series:
    """Build a year-indexed series with missing years filled as 0."""
    grouped = df.groupby(year_col, dropna=False)[value_col].sum(min_count=1)
    idx = pd.Index(range(year_min, year_max + 1), name=year_col)
    return grouped.reindex(idx, fill_value=0.0).astype(float)


def _extend_locf(series: pd.Series, *, end_year: int) -> pd.Series:
    """Extend a year-indexed series to `end_year` using last observation carried forward."""
    if series.empty:
        raise ValueError("Cannot extend an empty series.")

    start_year = int(series.index.min())
    idx = pd.Index(range(start_year, end_year + 1), name=series.index.name)
    out = series.reindex(idx)
    last_valid = out.dropna()
    if last_valid.empty:
        raise ValueError("Cannot LOCF-extend a fully-missing series.")
    out = out.fillna(float(last_valid.iloc[-1]))
    return out.astype(float)


def _extract_refugee_fy_series(
    refugees: pd.DataFrame, *, end_fy: int, state: str = "North Dakota"
) -> pd.Series:
    """Extract ND FY refugee arrivals series and LOCF-extend for forecasting."""
    nd = refugees[refugees["state"] == state].copy()
    if nd.empty:
        raise ValueError(f"No refugee arrivals found for state={state!r}")

    year_min = int(nd["fiscal_year"].min())
    year_max = int(nd["fiscal_year"].max())
    series = _series_from_grouped_sum(
        nd,
        year_col="fiscal_year",
        value_col="arrivals",
        year_min=year_min,
        year_max=year_max,
    )
    return _extend_locf(series, end_year=end_fy)


def _extract_lpr_fy_series(
    lpr_nd_share: pd.DataFrame,
    *,
    end_fy: int,
    value_col: Literal["nd_lpr_count", "nd_share_pct"] = "nd_lpr_count",
) -> pd.Series:
    """Extract ND FY LPR series from processed outputs and LOCF-extend."""
    if value_col not in lpr_nd_share.columns:
        raise ValueError(f"Missing expected LPR column: {value_col}")

    df = lpr_nd_share.copy()
    year_min = int(df["fiscal_year"].min())
    year_max = int(df["fiscal_year"].max())
    series = (
        df.set_index("fiscal_year")[value_col]
        .reindex(pd.Index(range(year_min, year_max + 1), name="fiscal_year"))
        .astype(float)
    )
    if series.isna().any():
        series = series.fillna(0.0)
    return _extend_locf(series, end_year=end_fy)


def _extract_acs_moved_from_abroad_series(
    acs: pd.DataFrame,
    *,
    end_year: int,
    state: str = "North Dakota",
    variable: str = "B07007_028E",
) -> pd.Series:
    """Extract ACS moved-from-abroad series (foreign born) and LOCF-extend."""
    df = acs[
        (acs["state_name"] == state) & (acs["variable"] == variable) & (acs["estimate"].notna())
    ].copy()
    if df.empty:
        raise ValueError(f"No ACS rows found for state={state!r} variable={variable!r}")

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    series = (
        df.groupby("year")["estimate"]
        .mean()
        .reindex(pd.Index(range(year_min, year_max + 1), name="year"))
        .astype(float)
    )
    return _extend_locf(series, end_year=end_year)


def _build_exog_frame(
    *,
    years: pd.Index,
    refugee_fy: pd.Series,
    lpr_fy: pd.Series,
    acs_year: pd.Series | None,
) -> pd.DataFrame:
    """Build a lag-1 exogenous regressor frame aligned to `years`."""
    exog = pd.DataFrame(index=years)
    exog["refugees_lag1"] = refugee_fy.rename("refugees").shift(1).reindex(years)
    exog["lpr_lag1"] = lpr_fy.rename("lpr").shift(1).reindex(years)
    if acs_year is not None:
        exog["acs_moved_from_abroad_lag1"] = (
            acs_year.rename("acs").shift(1).reindex(years)
        )
    return exog


def _standardize_exog(exog: pd.DataFrame) -> tuple[pd.DataFrame, Standardization]:
    """Z-score exog columns; return standardized frame and parameters."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    standardized = exog.copy()
    for col in standardized.columns:
        mean = float(standardized[col].mean())
        std = float(standardized[col].std(ddof=0))
        means[col] = mean
        stds[col] = std
        if std == 0:
            standardized[col] = standardized[col] - mean
        else:
            standardized[col] = (standardized[col] - mean) / std
    return standardized, Standardization(means=means, stds=stds)


def fit_local_level_regression(
    y: pd.Series, exog: pd.DataFrame, *, maxiter: int = 500
) -> Any:
    """
    Fit a local-level state-space regression.

    Args:
        y: Dependent series indexed by year.
        exog: Standardized exogenous regressors aligned to y.
        maxiter: Maximum optimizer iterations.

    Returns:
        Fitted statsmodels results object.
    """
    y_endog = y.astype(float).copy()
    y_endog.index = _as_annual_period_index(y_endog.index)

    exog_aligned = exog.astype(float).copy()
    exog_aligned.index = y_endog.index

    model = UnobservedComponents(
        endog=y_endog,
        level=True,
        stochastic_level=True,
        irregular=True,
        exog=exog_aligned,
    )
    return model.fit(disp=False, maxiter=maxiter)


def forecast_local_level_regression(
    fitted: Any, *, future_exog: pd.DataFrame, alpha_80: float = 0.2, alpha_95: float = 0.05
) -> pd.DataFrame:
    """Forecast and return point + 80/95% intervals in a tidy frame."""
    steps = len(future_exog)
    fc = fitted.get_forecast(steps=steps, exog=future_exog.to_numpy(dtype=float))
    mean = fc.predicted_mean.astype(float)

    ci80 = fc.conf_int(alpha=alpha_80).astype(float)
    ci95 = fc.conf_int(alpha=alpha_95).astype(float)

    # statsmodels uses column names like 'lower y', 'upper y' (or similar).
    lo80 = ci80.iloc[:, 0].rename("ci80_lo")
    hi80 = ci80.iloc[:, 1].rename("ci80_hi")
    lo95 = ci95.iloc[:, 0].rename("ci95_lo")
    hi95 = ci95.iloc[:, 1].rename("ci95_hi")

    out = pd.concat(
        [
            mean.rename("point").reset_index(drop=True),
            lo80.reset_index(drop=True),
            hi80.reset_index(drop=True),
            lo95.reset_index(drop=True),
            hi95.reset_index(drop=True),
        ],
        axis=1,
    )
    return out


def rolling_origin_backtest_one_step(
    y: pd.Series,
    exog: pd.DataFrame,
    *,
    start_train_year: int,
    start_test_year: int,
    end_test_year: int,
) -> dict[str, float]:
    """
    Rolling-origin one-step-ahead backtest for the covariate model.

    This is intended as a diagnostic check (appendix robustness). Metrics are
    descriptive given the very small sample.
    """
    test_years = list(range(start_test_year, end_test_year + 1))
    if not test_years:
        raise ValueError("Empty backtest horizon.")

    # MASE scale denominator computed on the initial training window.
    train_initial = y.loc[start_train_year : start_test_year - 1]
    if len(train_initial) < 3:
        raise ValueError("Insufficient observations for MASE denominator.")
    scale = float(train_initial.diff().abs().iloc[1:].mean())
    if scale == 0:
        scale = 1.0

    errors: list[float] = []
    abs_errors: list[float] = []
    abs_errors_naive: list[float] = []

    for year in test_years:
        train_end = year - 1
        train_y = y.loc[start_train_year:train_end]
        train_x = exog.loc[start_train_year:train_end]

        if train_y.isna().any() or train_x.isna().any().any():
            continue

        fitted = fit_local_level_regression(train_y, train_x, maxiter=300)

        point = forecast_local_level_regression(
            fitted, future_exog=exog.loc[[year]]
        ).iloc[0]["point"]
        actual = float(y.loc[year])
        err = actual - float(point)

        errors.append(err)
        abs_errors.append(abs(err))
        abs_errors_naive.append(abs(actual - float(y.loc[year - 1])))

    if not abs_errors:
        raise ValueError("Backtest produced no usable forecast errors.")

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mase = float(np.mean([ae / scale for ae in abs_errors]))
    naive_mae = float(np.mean(abs_errors_naive))

    return {
        "n_forecasts": int(len(abs_errors)),
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "naive_mae": naive_mae,
        "relative_mae_vs_naive": float(mae / naive_mae) if naive_mae else float("nan"),
    }


def _plot_overlay(
    historical: pd.Series,
    baseline: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    *,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Create the appendix overlay figure."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Historical (2010–2024)
    ax.plot(
        historical.index,
        historical.values,
        color="#333333",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Observed (PEP)",
    )

    # Baseline (ARIMA)
    ax.plot(
        baseline["year"],
        baseline["baseline_point"],
        color="#666666",
        linewidth=1.6,
        marker="o",
        markersize=4,
        linestyle="--",
        label="Baseline (ARIMA)",
    )
    ax.fill_between(
        baseline["year"],
        baseline["baseline_ci95_lo"],
        baseline["baseline_ci95_hi"],
        color="#999999",
        alpha=0.18,
        label="Baseline 95% PI",
    )

    # Covariate-conditioned specs (points + 95% intervals)
    palette = {
        "P0": ("#0072B2", "Covariates (Refugees + LPR)"),
        "S3": ("#D55E00", "Covariates (+ ACS sensitivity)"),
    }
    for tag, df_fc in forecasts.items():
        color, label = palette.get(tag, ("#009E73", f"Covariates ({tag})"))
        ax.plot(
            df_fc["year"],
            df_fc["point"],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=4,
            label=label,
        )
        ax.fill_between(
            df_fc["year"],
            df_fc["ci95_lo"],
            df_fc["ci95_hi"],
            color=color,
            alpha=0.12,
        )

    ax.set_title("Appendix Diagnostic: Covariate-Conditioned Near-Term Anchor (2025–2029)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net International Migration (persons; PEP)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=1)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def _build_comparison_table(
    baseline: pd.DataFrame, forecasts: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Combine baseline and covariate-conditioned forecasts into one table."""
    table = baseline[["year", "baseline_point", "baseline_ci95_lo", "baseline_ci95_hi"]].copy()
    for tag, df_fc in forecasts.items():
        table = table.merge(
            df_fc[["year", "point", "ci95_lo", "ci95_hi"]].rename(
                columns={
                    "point": f"{tag}_point",
                    "ci95_lo": f"{tag}_ci95_lo",
                    "ci95_hi": f"{tag}_ci95_hi",
                }
            ),
            on="year",
            how="left",
        )
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["auto", "db", "files"],
        default="files",
        help="Data source mode for statistical_analysis data_loader (default: files).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2029,
        help="Forecast end year for appendix anchor (default: 2029).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["SDC_ANALYSIS_DATA_SOURCE"] = str(args.data_source)

    analysis_dir = _analysis_dir()
    results_dir = _results_dir()
    figures_dir = _figures_dir()
    journal_figures_dir = _journal_figures_dir()

    out_json = results_dir / "module_app_covariate_anchor.json"
    out_csv = results_dir / "module_app_covariate_anchor_2025_2029.csv"
    out_png = figures_dir / "fig_app_covariate_anchor_2025_2029.png"
    out_pdf = figures_dir / "fig_app_covariate_anchor_2025_2029.pdf"

    inputs = [
        analysis_dir / "nd_migration_summary.csv",
        analysis_dir / "refugee_arrivals_by_state_nationality.parquet",
        analysis_dir / "dhs_lpr_nd_share_time.parquet",
        analysis_dir / "acs_moved_from_abroad_by_state.parquet",
        results_dir / "module_2_1_arima_model.json",
    ]
    outputs = [out_json, out_csv, out_png, out_pdf]

    covariate_specs = [
        CovariateSpec(
            tag="P0",
            description="Local level + lagged refugees (FY) and LPR (FY) regressors",
            include_acs=False,
        ),
        CovariateSpec(
            tag="S3",
            description="P0 + ACS moved-from-abroad (foreign born) regressor (sensitivity)",
            include_acs=True,
        ),
    ]

    with log_execution(
        __file__,
        parameters={
            "data_source": args.data_source,
            "forecast_end_year": args.end_year,
            "specs": [spec.tag for spec in covariate_specs],
        },
        inputs=inputs,
        outputs=outputs,
    ):
        baseline = _load_arima_baseline(results_dir=results_dir)
        y_df = data_loader.load_migration_summary()
        refugees = data_loader.load_refugee_arrivals()

        lpr_path = analysis_dir / "dhs_lpr_nd_share_time.parquet"
        acs_path = analysis_dir / "acs_moved_from_abroad_by_state.parquet"
        lpr = pd.read_parquet(lpr_path)
        acs = pd.read_parquet(acs_path)

        y = y_df.set_index("year")["nd_intl_migration"].astype(float)
        historical_years = pd.Index(range(int(y.index.min()), int(y.index.max()) + 1), name="year")
        y = y.reindex(historical_years).astype(float)

        end_year = int(args.end_year)
        if end_year < 2025:
            raise ValueError("end-year must be >= 2025 for this appendix anchor.")
        future_years = pd.Index(range(2025, end_year + 1), name="year")

        # Covariate series extended through end_year so that lag-1 values are
        # defined for the full forecast horizon (e.g., exog for 2029 uses 2028).
        refugee_fy = _extract_refugee_fy_series(refugees, end_fy=end_year)
        lpr_fy = _extract_lpr_fy_series(lpr, end_fy=end_year, value_col="nd_lpr_count")
        acs_year = _extract_acs_moved_from_abroad_series(acs, end_year=end_year)

        exog_all = {}
        forecasts_by_spec: dict[str, pd.DataFrame] = {}
        fit_summaries: dict[str, Any] = {}
        backtests: dict[str, Any] = {}

        for spec in covariate_specs:
            exog = _build_exog_frame(
                years=historical_years,
                refugee_fy=refugee_fy,
                lpr_fy=lpr_fy,
                acs_year=acs_year if spec.include_acs else None,
            )

            # Drop years with missing exog (e.g., ACS lag in 2010).
            mask = (~exog.isna().any(axis=1)) & (~y.isna())
            y_fit = y.loc[mask].copy()
            exog_fit = exog.loc[mask].copy()

            exog_fit_std, standardization = _standardize_exog(exog_fit)

            fitted = fit_local_level_regression(y_fit, exog_fit_std)

            # Build future exog (years 2025–end_year) with same construction and scaling.
            exog_future = _build_exog_frame(
                years=future_years,
                refugee_fy=refugee_fy,
                lpr_fy=lpr_fy,
                acs_year=acs_year if spec.include_acs else None,
            )
            exog_future_std = standardization.apply(exog_future)

            fc = forecast_local_level_regression(fitted, future_exog=exog_future_std)
            fc.insert(0, "year", future_years.astype(int).to_numpy())

            forecasts_by_spec[spec.tag] = fc
            exog_all[spec.tag] = {
                "exog_columns": list(exog_fit.columns),
                "n_obs_fit": int(len(y_fit)),
                "fit_year_start": int(y_fit.index.min()),
                "fit_year_end": int(y_fit.index.max()),
                "standardization": {
                    "means": standardization.means,
                    "stds": standardization.stds,
                },
            }
            fit_summaries[spec.tag] = {
                "aic": float(getattr(fitted, "aic", np.nan)),
                "bic": float(getattr(fitted, "bic", np.nan)),
                "llf": float(getattr(fitted, "llf", np.nan)),
                "param_names": list(getattr(fitted, "param_names", [])),
                "params": {
                    name: float(val)
                    for name, val in zip(
                        getattr(fitted, "param_names", []), getattr(fitted, "params", [])
                    )
                },
            }

            # Backtest (one-step ahead) for P0 and ACS sensitivity, on the maximal
            # common test window. This is descriptive only.
            try:
                start_train_year = int(y_fit.index.min())
                backtests[spec.tag] = rolling_origin_backtest_one_step(
                    y=y_fit,
                    exog=exog_fit_std,
                    start_train_year=start_train_year,
                    start_test_year=2017,
                    end_test_year=2024,
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                LOGGER.warning("Backtest failed for %s: %s", spec.tag, exc)
                backtests[spec.tag] = {"error": str(exc)}

        comparison_table = _build_comparison_table(baseline=baseline, forecasts=forecasts_by_spec)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_table.to_csv(out_csv, index=False)

        # Plot overlay and copy figure into journal_article/figures for LaTeX.
        _plot_overlay(historical=y, baseline=baseline, forecasts=forecasts_by_spec, out_png=out_png, out_pdf=out_pdf)
        journal_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_png, journal_figures_dir / out_png.name)
        shutil.copy2(out_pdf, journal_figures_dir / out_pdf.name)

        payload: dict[str, Any] = {
            "module": "APP-COVARIATE-ANCHOR",
            "analysis": "covariate_conditioned_near_term_anchor",
            "generated": datetime.now(UTC).isoformat(),
            "adr": "docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md",
            "runbook": "sdc_2024_replication/revisions/v0.8.6/covariate_forecasting_appendix_spec.md",
            "input_files": [str(p) for p in inputs],
            "outputs": {
                "results_json": str(out_json),
                "forecast_table_csv": str(out_csv),
                "figure_png": str(out_png),
                "figure_pdf": str(out_pdf),
                "journal_article_figure_png": str(journal_figures_dir / out_png.name),
                "journal_article_figure_pdf": str(journal_figures_dir / out_pdf.name),
            },
            "parameters": {
                "data_source": args.data_source,
                "forecast_years": [int(y) for y in future_years],
                "model_class": "statsmodels.UnobservedComponents(level=True, stochastic_level=True, irregular=True, exog=...)",
                "covariate_rules": {
                    "refugees": "ND FY total; missing FY treated as 0; post-FY2024 extended via LOCF",
                    "lpr": "ND FY total; post-FY2023 extended via LOCF",
                    "acs": "ND ACS B07007_028E estimate; post-2023 extended via LOCF (sensitivity only)",
                    "lag": "All covariates enter as lag-1 (x_{t-1})",
                },
            },
            "baseline": baseline.to_dict(orient="records"),
            "specs": {spec.tag: {"description": spec.description} for spec in covariate_specs},
            "fit_inputs": exog_all,
            "fit_summaries": fit_summaries,
            "forecasts": {k: v.to_dict(orient="records") for k, v in forecasts_by_spec.items()},
            "backtesting": backtests,
        }
        _write_json(out_json, payload)

        LOGGER.info("Wrote appendix covariate anchor outputs: %s", out_json)
        LOGGER.info("Forecast table: %s", out_csv)
        LOGGER.info("Figure: %s", out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
