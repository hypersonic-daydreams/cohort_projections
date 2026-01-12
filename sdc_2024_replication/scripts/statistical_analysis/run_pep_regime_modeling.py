#!/usr/bin/env python3
"""
Run v0.8.6 regime-aware modeling for the long-run (2000–2024) PEP series.

This script operationalizes the v0.8.6 spec grid documented in:
- `sdc_2024_replication/revisions/v0.8.6/pep_regime_modeling_spec_grid.md`

It produces:
- A single JSON results bundle under `sdc_2024_replication/scripts/statistical_analysis/results/`
- At least one figure under `sdc_2024_replication/scripts/statistical_analysis/figures/`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper

from cohort_projections.utils import ConfigLoader, setup_logger

from module_regime_aware.covid_intervention import create_covid_intervention
from module_regime_aware.robust_inference import estimate_regime_variances, estimate_wls_by_regime
from module_regime_aware.vintage_dummies import create_regime_dummies

logger = setup_logger(__name__)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


@dataclass(frozen=True)
class SpecConfig:
    tag: str
    description: str
    drop_years: tuple[int, ...] = ()
    include_covid_pulse: bool = False
    use_wls: bool = False
    cov_type: str = "HAC"
    maxlags: int = 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_long_run_components_path() -> Path:
    cfg = ConfigLoader().get_projection_config()
    census_cfg = cfg.get("data_sources", {}).get("census_components", {})
    output_dir = census_cfg.get("output_dir", "data/processed/immigration")
    output_file = census_cfg.get(
        "regime_output_file", "state_migration_components_2000_2024_with_regime.csv"
    )
    return _project_root() / output_dir / output_file


def _load_long_run_state_series(
    path: Path,
    state_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Filter to state-level rows and the named state.
    # The long-run file includes US totals, regions, divisions, etc.
    df = df.copy()

    def _as_int(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("Int64")

    if "SUMLEV" in df.columns:
        sumlev = _as_int(df["SUMLEV"])
        df = df[sumlev == 40]

    if "NAME" not in df.columns:
        raise ValueError(f"Expected `NAME` column in {path}")
    df = df[df["NAME"] == state_name].copy()

    required = {"year", "INTERNATIONALMIG", "vintage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["intl_migration"] = pd.to_numeric(df["INTERNATIONALMIG"], errors="coerce")
    df["vintage"] = pd.to_numeric(df["vintage"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["intl_migration", "vintage"]).copy()
    df["vintage"] = df["vintage"].astype(int)

    df = df.sort_values("year").reset_index(drop=True)
    return df[["year", "intl_migration", "vintage"]]


def _add_piecewise_design_matrix(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
) -> pd.DataFrame:
    df = df.copy()
    df = create_regime_dummies(df, year_col=year_col)

    min_year = int(df[year_col].min())
    df["t"] = df[year_col] - min_year

    df["trend_2000s"] = df["t"] * df["regime_2000s"]
    df["trend_2010s"] = ((df[year_col] - 2010) * df["regime_2010s"]).clip(lower=0)
    df["trend_2020s"] = ((df[year_col] - 2020) * df["regime_2020s"]).clip(lower=0)

    return df


def _fit_model(
    df: pd.DataFrame,
    *,
    y_col: str,
    X_cols: list[str],
    cov_type: str,
    maxlags: int,
) -> RegressionResultsWrapper:
    X = sm.add_constant(df[X_cols])
    y = df[y_col]

    if cov_type.upper() == "HAC":
        return OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    if cov_type.lower() == "nonrobust":
        return OLS(y, X).fit()
    return OLS(y, X).fit(cov_type=cov_type)


def _model_to_dict(model: RegressionResultsWrapper) -> dict[str, Any]:
    return {
        "params": {k: float(v) for k, v in model.params.to_dict().items()},
        "bse": {k: float(v) for k, v in model.bse.to_dict().items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.to_dict().items()},
        "rsquared": float(model.rsquared),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n_obs": int(model.nobs),
    }


def _f_test_equal_slopes(model: RegressionResultsWrapper) -> dict[str, Any] | None:
    try:
        f_test = model.f_test("trend_2000s = trend_2010s = trend_2020s")
        return {
            "f_statistic": float(f_test.fvalue),
            "p_value": float(f_test.pvalue),
            "df_num": int(f_test.df_num),
            "df_denom": float(f_test.df_denom),
            "reject_equality_at_05": bool(f_test.pvalue < 0.05),
        }
    except Exception:
        return None


def _run_spec(df: pd.DataFrame, spec: SpecConfig) -> dict[str, Any]:
    df_spec = df.copy()
    if spec.drop_years:
        df_spec = df_spec[~df_spec["year"].isin(spec.drop_years)].copy()

    df_spec = _add_piecewise_design_matrix(df_spec)

    if spec.include_covid_pulse:
        df_spec = create_covid_intervention(df_spec, year_col="year")

    base_X_cols = [
        "regime_2010s",
        "regime_2020s",
        "trend_2000s",
        "trend_2010s",
        "trend_2020s",
    ]
    X_cols = base_X_cols.copy()
    if spec.include_covid_pulse:
        X_cols.append("covid_pulse")

    if spec.use_wls:
        wls = estimate_wls_by_regime(
            df_spec,
            y_col="intl_migration",
            X_cols=X_cols,
            regime_col="vintage",
        )
        fitted = _compute_fitted_values(
            df_spec, wls["wls_params"], X_cols=X_cols, y_col="intl_migration"
        )
        return {
            "spec": asdict(spec),
            "inputs": {"n_obs": int(len(df_spec)), "years": [int(y) for y in df_spec["year"]]},
            "wls": wls,
            "fitted": fitted,
        }

    model = _fit_model(
        df_spec,
        y_col="intl_migration",
        X_cols=X_cols,
        cov_type=spec.cov_type,
        maxlags=spec.maxlags,
    )
    fitted = _compute_fitted_values(
        df_spec, model.params.to_dict(), X_cols=X_cols, y_col="intl_migration"
    )
    variance_diag = estimate_regime_variances(df_spec, y_col="intl_migration", regime_col="vintage")

    return {
        "spec": asdict(spec),
        "inputs": {"n_obs": int(len(df_spec)), "years": [int(y) for y in df_spec["year"]]},
        "model": _model_to_dict(model),
        "slope_equality_test": _f_test_equal_slopes(model),
        "regime_variance": asdict(variance_diag),
        "fitted": fitted,
    }


def _compute_fitted_values(
    df: pd.DataFrame,
    params: dict[str, Any],
    *,
    X_cols: list[str],
    y_col: str,
) -> list[dict[str, float]]:
    df = df.copy()
    X = sm.add_constant(df[X_cols])
    p = pd.Series(params).astype(float)
    fitted = (X * p.reindex(X.columns).fillna(0.0)).sum(axis=1)
    return [
        {"year": float(y), "actual": float(a), "fitted": float(f)}
        for y, a, f in zip(df["year"], df[y_col], fitted, strict=False)
    ]


def _plot_series_with_fit(
    df: pd.DataFrame,
    fitted: list[dict[str, float]],
    output_path: Path,
    *,
    title: str,
) -> None:
    df_plot = df.copy()
    fitted_df = pd.DataFrame(fitted)

    colors = {2009: "#e74c3c", 2020: "#3498db", 2024: "#2ecc71"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for v in sorted(df_plot["vintage"].unique()):
        subset = df_plot[df_plot["vintage"] == v]
        ax.scatter(
            subset["year"],
            subset["intl_migration"],
            s=70,
            color=colors.get(int(v), "gray"),
            label=f"Vintage {int(v)}",
            zorder=5,
        )

    ax.plot(
        fitted_df["year"],
        fitted_df["fitted"],
        color="black",
        linewidth=2.0,
        label="Fitted (P0)",
        zorder=4,
    )

    ax.axvline(x=2010, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(x=2020, color="gray", linestyle="--", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("PEP Net International Migration (persons)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_slopes_table(spec_results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slope_vars = ["trend_2000s", "trend_2010s", "trend_2020s"]
    for tag, payload in spec_results.items():
        if "model" in payload:
            model = payload["model"]
            params = model.get("params", {})
            bse = model.get("bse", {})
            pvalues = model.get("pvalues", {})
            for var in slope_vars:
                rows.append(
                    {
                        "spec": tag,
                        "estimator": payload["spec"].get("cov_type", "OLS"),
                        "term": var,
                        "slope": params.get(var),
                        "se": bse.get(var),
                        "p_value": pvalues.get(var),
                    }
                )
        elif "wls" in payload:
            wls = payload["wls"]
            params = wls.get("wls_params", {})
            bse = wls.get("wls_bse", {})
            pvalues = wls.get("wls_pvalues", {})
            for var in slope_vars:
                rows.append(
                    {
                        "spec": tag,
                        "estimator": "WLS",
                        "term": var,
                        "slope": params.get(var),
                        "se": bse.get(var),
                        "p_value": pvalues.get(var),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["slope"] = pd.to_numeric(df["slope"], errors="coerce")
        df["se"] = pd.to_numeric(df["se"], errors="coerce")
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run v0.8.6 long-run PEP regime-aware modeling (P0/S1/S2) and write artifacts."
    )
    parser.add_argument(
        "--no-journal-copy",
        action="store_true",
        help="Do not copy the P0 diagnostic figure into the journal article figures directory.",
    )
    parser.add_argument(
        "--journal-figure-path",
        default=None,
        help=(
            "Override path for the copied journal figure "
            "(default: sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/"
            "fig_app_pep_regime_diagnostic.png)."
        ),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    state_name = "North Dakota"
    input_path = _resolve_long_run_components_path()

    logger.info("Loading long-run PEP components series from %s", input_path)
    df = _load_long_run_state_series(input_path, state_name=state_name)

    # Basic validation: expect 2000–2024 inclusive => 25 points.
    years = df["year"].tolist()
    if years and (min(years) != 2000 or max(years) != 2024):
        logger.warning("Unexpected year coverage for %s: %s–%s", state_name, min(years), max(years))
    if len(df) < 20:
        logger.warning("Unexpectedly short long-run series for %s: n=%s", state_name, len(df))

    specs = [
        SpecConfig(
            tag="P0",
            description="Piecewise vintage regimes + COVID pulse + HAC inference",
            include_covid_pulse=True,
            use_wls=False,
            cov_type="HAC",
            maxlags=2,
        ),
        SpecConfig(
            tag="S1",
            description="Exclude 2020 entirely (no intervention term)",
            drop_years=(2020,),
            include_covid_pulse=False,
            use_wls=False,
            cov_type="HAC",
            maxlags=2,
        ),
        SpecConfig(
            tag="S2",
            description="WLS by regime variance (downweight high-variance regime)",
            include_covid_pulse=True,
            use_wls=True,
        ),
    ]

    results: dict[str, Any] = {
        "module": "module_B1_pep_regime_modeling",
        "generated": datetime.now(UTC).isoformat(),
        "inputs": {
            "state_name": state_name,
            "long_run_components_path": str(input_path),
            "n_obs": int(len(df)),
            "years": [int(y) for y in df["year"].tolist()],
            "vintage_counts": df["vintage"].value_counts().to_dict(),
        },
        "specs": {},
    }

    for spec in specs:
        logger.info("Running spec %s: %s", spec.tag, spec.description)
        results["specs"][spec.tag] = _run_spec(df, spec)

    # Export a compact slopes table for manuscript/appendix use.
    slopes_path = RESULTS_DIR / "module_B1_pep_regime_modeling_slopes.csv"
    slopes_df = _build_slopes_table(results["specs"])
    slopes_df.to_csv(slopes_path, index=False)

    # Primary figure uses P0 fitted values.
    fig_path = FIGURES_DIR / "module_B1_pep_regime_modeling__P0.png"
    _plot_series_with_fit(
        df,
        fitted=results["specs"]["P0"]["fitted"],
        output_path=fig_path,
        title="North Dakota PEP Net International Migration (2000–2024)\nPiecewise Vintage Regimes + COVID Pulse (P0)",
    )

    out_path = RESULTS_DIR / "module_B1_pep_regime_modeling.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    journal_copy_path = None
    if not args.no_journal_copy:
        if args.journal_figure_path:
            journal_copy_path = Path(args.journal_figure_path)
        else:
            journal_copy_path = (
                Path(__file__).resolve().parent
                / "journal_article"
                / "figures"
                / "fig_app_pep_regime_diagnostic.png"
            )
        journal_copy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil

            shutil.copy2(fig_path, journal_copy_path)
        except Exception as exc:
            logger.warning("Failed to copy P0 figure to journal figures (%s): %s", journal_copy_path, exc)
            journal_copy_path = None

    logger.info("Wrote results JSON: %s", out_path)
    logger.info("Wrote slopes CSV: %s", slopes_path)
    logger.info("Wrote figure: %s", fig_path)
    if journal_copy_path:
        logger.info("Copied journal figure: %s", journal_copy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
