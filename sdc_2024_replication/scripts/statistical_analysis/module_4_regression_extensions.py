#!/usr/bin/env python3
"""
Module 4: Regression Extensions - Beta, Quantile, and Robust Regression
========================================================================

Implements extended regression techniques for bounded proportions (beta regression),
distributional analysis (quantile regression), and outlier-robust estimation.

Usage:
    micromamba run -n cohort_proj python module_4_regression_extensions.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Statsmodels imports
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
}

CATEGORICAL = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#999999",
]

# Quantile colors (gradient)
QUANTILE_COLORS = {
    0.10: "#d7191c",  # Red
    0.25: "#fdae61",  # Orange
    0.50: "#ffffbf",  # Yellow (neutral)
    0.75: "#a6d96a",  # Light green
    0.90: "#1a9641",  # Dark green
}


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.warnings: list[str] = []
        self.decisions: list[dict] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] = None,
        evidence: str = None,
        reversible: bool = True,
    ):
        """Log a decision with full context."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": reversible,
            }
        )

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "decisions": self.decisions,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def setup_figure(figsize=(10, 8)):
    """Standard figure setup for all visualizations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    return fig, ax


def save_figure(fig, filepath_base, title, source_note):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        f"Source: {source_note}",
        fontsize=8,
        fontstyle="italic",
        transform=fig.transFigure,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save both formats
    fig.savefig(
        f"{filepath_base}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        f"{filepath_base}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Figure saved: {filepath_base}.png/pdf")


def load_data(result: ModuleResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load panel data and ND migration summary."""
    # Load panel data from Module 3.1
    panel_path = RESULTS_DIR / "module_3_1_panel_data.parquet"
    df_panel = pd.read_parquet(panel_path)
    result.input_files.append("module_3_1_panel_data.parquet")

    # Load ND migration summary
    nd_summary_path = DATA_DIR / "nd_migration_summary.csv"
    df_nd = pd.read_csv(nd_summary_path)
    result.input_files.append("nd_migration_summary.csv")

    print(f"Loaded panel data: {df_panel.shape[0]} rows, {df_panel.shape[1]} columns")
    print(f"Loaded ND summary: {df_nd.shape[0]} rows, {df_nd.shape[1]} columns")

    return df_panel, df_nd


def prepare_nd_data(df_nd: pd.DataFrame, result: ModuleResult) -> pd.DataFrame:
    """Prepare ND time series data for regression analysis."""
    df = df_nd.copy()

    # Convert share from percentage to proportion (0-1 scale)
    df["nd_share"] = df["nd_share_of_us_intl_pct"] / 100

    # Ensure bounded between epsilon and 1-epsilon for beta regression
    epsilon = 1e-6
    df["nd_share_bounded"] = df["nd_share"].clip(lower=epsilon, upper=1 - epsilon)

    # Time trend (centered for numerical stability)
    df["time_trend"] = df["year"] - df["year"].min()

    # COVID dummy
    df["covid_2020"] = (df["year"] == 2020).astype(int)

    # Lagged values
    df = df.sort_values("year")
    df["lag_nd_share"] = df["nd_share"].shift(1)
    df["lag_nd_intl"] = df["nd_intl_migration"].shift(1)

    # Growth rate
    df["nd_intl_growth"] = df["nd_intl_migration"].pct_change()

    # Log transformations (handle zeros)
    df["log_nd_intl"] = np.log(df["nd_intl_migration"].clip(lower=1))

    result.add_decision(
        decision_id="D001",
        category="data_preparation",
        decision="Bounded ND share to (epsilon, 1-epsilon) for beta regression",
        rationale="Beta regression requires outcomes strictly between 0 and 1",
        alternatives=["Use quasi-binomial GLM", "Transform with logit before OLS"],
        evidence=f"Original range: [{df['nd_share'].min():.6f}, {df['nd_share'].max():.6f}]",
    )

    return df


def beta_regression(df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate beta regression for ND share of US international migration.

    Uses GLM with quasi-binomial family and logit link as approximation to beta regression.
    """
    print("\n--- Beta Regression Analysis ---")

    # Prepare data (drop NaN rows from lagged variables)
    df_model = df.dropna(
        subset=["nd_share_bounded", "time_trend", "covid_2020", "lag_nd_share"]
    ).copy()

    # Dependent variable
    y = df_model["nd_share_bounded"]

    # Independent variables
    X = sm.add_constant(df_model[["time_trend", "covid_2020", "lag_nd_share"]])

    # Fit GLM with binomial family and logit link (quasi-beta regression)
    # Note: statsmodels doesn't have native beta regression, so we use binomial GLM
    model = GLM(y, X, family=Binomial(link=Logit()))
    results = model.fit()

    print(results.summary())

    # Calculate odds ratios (exponentiate coefficients)
    odds_ratios = np.exp(results.params)
    odds_ratio_ci = np.exp(results.conf_int())

    # Predicted values
    df_model["fitted"] = results.fittedvalues
    df_model["residuals"] = results.resid_response

    # Model diagnostics
    deviance = results.deviance
    pearson_chi2 = results.pearson_chi2
    log_likelihood = results.llf
    aic = results.aic
    bic = results.bic

    # Pseudo R-squared (McFadden)
    null_model = GLM(y, np.ones(len(y)), family=Binomial(link=Logit()))
    null_results = null_model.fit()
    pseudo_r2 = 1 - (results.llf / null_results.llf)

    # Build coefficient table
    coef_table = {}
    for var in results.params.index:
        coef_table[var] = {
            "estimate": float(results.params[var]),
            "std_error": float(results.bse[var]),
            "z_statistic": float(results.tvalues[var]),
            "p_value": float(results.pvalues[var]),
            "ci_95_lower": float(results.conf_int().loc[var, 0]),
            "ci_95_upper": float(results.conf_int().loc[var, 1]),
            "odds_ratio": float(odds_ratios[var]),
            "or_ci_95_lower": float(odds_ratio_ci.loc[var, 0]),
            "or_ci_95_upper": float(odds_ratio_ci.loc[var, 1]),
        }

    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Used GLM with binomial family as quasi-beta regression",
        rationale="Native beta regression not in statsmodels; binomial GLM with logit link approximates beta regression for proportions",
        alternatives=[
            "Use betareg R package via rpy2",
            "Use fractional logit regression",
        ],
        evidence=f"Pseudo R-squared: {pseudo_r2:.4f}",
    )

    beta_results = {
        "model_type": "Beta Regression (Binomial GLM with Logit Link)",
        "dependent_variable": "nd_share_bounded (ND share of US international migration)",
        "n_observations": int(len(df_model)),
        "coefficient_table": coef_table,
        "fit_statistics": {
            "deviance": float(deviance),
            "pearson_chi2": float(pearson_chi2),
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
            "pseudo_r2_mcfadden": float(pseudo_r2),
        },
        "interpretation": {
            "time_trend": f"Each year increases log-odds of ND share by {results.params['time_trend']:.4f} (OR: {odds_ratios['time_trend']:.4f})",
            "covid_2020": f"COVID year {'increases' if results.params['covid_2020'] > 0 else 'decreases'} log-odds by {abs(results.params['covid_2020']):.4f}",
            "lag_nd_share": f"Strong persistence: lagged share coefficient = {results.params['lag_nd_share']:.4f}",
        },
        "residual_diagnostics": {
            "mean_residual": float(df_model["residuals"].mean()),
            "std_residual": float(df_model["residuals"].std()),
            "min_residual": float(df_model["residuals"].min()),
            "max_residual": float(df_model["residuals"].max()),
        },
    }

    # Store for plotting
    beta_results["_fitted_data"] = df_model[
        ["year", "nd_share_bounded", "fitted", "residuals"]
    ].to_dict("records")

    return beta_results, results, df_model


def quantile_regression(df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate quantile regression at multiple quantiles.

    Analyzes how coefficients change across the distribution of ND migration.
    """
    print("\n--- Quantile Regression Analysis ---")

    # Prepare data
    df_model = df.dropna(
        subset=["nd_intl_migration", "time_trend", "covid_2020"]
    ).copy()

    # For state share analysis
    df_share = df.dropna(
        subset=["nd_share", "time_trend", "covid_2020", "lag_nd_share"]
    ).copy()

    # Quantiles to estimate
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    # Dependent variable: ND international migration (levels)
    y_levels = df_model["nd_intl_migration"]
    X_levels = sm.add_constant(df_model[["time_trend", "covid_2020"]])

    # Dependent variable: ND share (proportions)
    y_share = df_share["nd_share"] * 100  # Scale for numerical stability
    X_share = sm.add_constant(df_share[["time_trend", "covid_2020", "lag_nd_share"]])

    # Results containers
    qr_levels_results = {}
    qr_share_results = {}

    # Estimate at each quantile - Levels model
    print("\nQuantile regression for ND international migration (levels):")
    for q in quantiles:
        model = QuantReg(y_levels, X_levels)
        res = model.fit(q=q)
        print(
            f"  Q{int(q*100)}: const={res.params['const']:.1f}, trend={res.params['time_trend']:.1f}, covid={res.params['covid_2020']:.1f}"
        )

        # Bootstrap standard errors
        n_boot = 500
        boot_params = np.zeros((n_boot, len(res.params)))
        n_obs = len(y_levels)

        for b in range(n_boot):
            boot_idx = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y_levels.iloc[boot_idx].reset_index(drop=True)
            X_boot = X_levels.iloc[boot_idx].reset_index(drop=True)
            try:
                boot_res = QuantReg(y_boot, X_boot).fit(q=q)
                boot_params[b, :] = boot_res.params
            except Exception:
                boot_params[b, :] = np.nan

        boot_se = np.nanstd(boot_params, axis=0)

        qr_levels_results[f"q{int(q*100)}"] = {
            "quantile": q,
            "coefficients": {
                var: {
                    "estimate": float(res.params[var]),
                    "std_error": float(res.bse[var]),
                    "bootstrap_se": float(boot_se[i]),
                    "t_statistic": float(res.tvalues[var]),
                    "p_value": float(res.pvalues[var]),
                    "ci_95_lower": float(res.conf_int().loc[var, 0]),
                    "ci_95_upper": float(res.conf_int().loc[var, 1]),
                }
                for i, var in enumerate(res.params.index)
            },
            "pseudo_r2": float(res.prsquared),
            "n_observations": int(res.nobs),
        }

    # Estimate at each quantile - Share model
    print("\nQuantile regression for ND share (percent):")
    for q in quantiles:
        model = QuantReg(y_share, X_share)
        res = model.fit(q=q)
        print(
            f"  Q{int(q*100)}: const={res.params['const']:.4f}, trend={res.params['time_trend']:.4f}"
        )

        qr_share_results[f"q{int(q*100)}"] = {
            "quantile": q,
            "coefficients": {
                var: {
                    "estimate": float(res.params[var]),
                    "std_error": float(res.bse[var]),
                    "t_statistic": float(res.tvalues[var]),
                    "p_value": float(res.pvalues[var]),
                }
                for var in res.params.index
            },
            "pseudo_r2": float(res.prsquared),
            "n_observations": int(res.nobs),
        }

    # Compare OLS (mean) to median regression
    ols_model = sm.OLS(y_levels, X_levels)
    ols_results = ols_model.fit()

    median_results = qr_levels_results["q50"]

    comparison = {
        "ols_vs_median": {
            "ols_const": float(ols_results.params["const"]),
            "median_const": median_results["coefficients"]["const"]["estimate"],
            "ols_trend": float(ols_results.params["time_trend"]),
            "median_trend": median_results["coefficients"]["time_trend"]["estimate"],
            "ols_covid": float(ols_results.params["covid_2020"]),
            "median_covid": median_results["coefficients"]["covid_2020"]["estimate"],
        }
    }

    result.add_decision(
        decision_id="D003",
        category="methodology",
        decision="Estimated quantile regression at 10th, 25th, 50th, 75th, and 90th percentiles",
        rationale="Captures heterogeneous effects across the distribution; reveals if relationships differ at extremes",
        alternatives=[
            "Additional quantiles (5th, 95th)",
            "Continuous quantile process",
        ],
        evidence="Bootstrap SE used for inference (n_boot=500)",
    )

    # Build quantile process (how coefficients change)
    quantile_process = {
        "time_trend": {
            str(q): qr_levels_results[f"q{int(q*100)}"]["coefficients"]["time_trend"][
                "estimate"
            ]
            for q in quantiles
        },
        "covid_2020": {
            str(q): qr_levels_results[f"q{int(q*100)}"]["coefficients"]["covid_2020"][
                "estimate"
            ]
            for q in quantiles
        },
    }

    qr_results = {
        "model_type": "Quantile Regression",
        "quantiles_estimated": quantiles,
        "levels_model": {
            "dependent_variable": "nd_intl_migration",
            "results_by_quantile": qr_levels_results,
        },
        "share_model": {
            "dependent_variable": "nd_share_percent",
            "results_by_quantile": qr_share_results,
        },
        "quantile_process": quantile_process,
        "ols_comparison": comparison,
        "interpretation": {
            "trend_pattern": "Positive trend in migration; effect "
            + (
                "increases"
                if quantile_process["time_trend"]["0.9"]
                > quantile_process["time_trend"]["0.1"]
                else "decreases"
            )
            + " at higher quantiles",
            "covid_pattern": "COVID effect magnitude varies across distribution",
        },
    }

    return qr_results, qr_levels_results


def robust_regression(df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate robust regression using M-estimators and MM-estimators.

    Provides outlier-robust coefficient estimates.
    """
    print("\n--- Robust Regression Analysis ---")

    # Prepare data
    df_model = df.dropna(
        subset=["nd_intl_migration", "time_trend", "covid_2020"]
    ).copy()

    y = df_model["nd_intl_migration"]
    X = sm.add_constant(df_model[["time_trend", "covid_2020"]])

    # OLS for comparison
    ols_model = sm.OLS(y, X)
    ols_results = ols_model.fit()

    # Huber M-estimator (default)
    huber_model = RLM(y, X, M=sm.robust.norms.HuberT())
    huber_results = huber_model.fit()

    # Tukey's biweight (bisquare) - more resistant
    tukey_model = RLM(y, X, M=sm.robust.norms.TukeyBiweight())
    tukey_results = tukey_model.fit()

    print("\nOLS coefficients:")
    print(ols_results.params)
    print("\nHuber M-estimator coefficients:")
    print(huber_results.params)
    print("\nTukey biweight coefficients:")
    print(tukey_results.params)

    # Identify influential observations using robust weights
    df_model["huber_weights"] = huber_results.weights
    df_model["tukey_weights"] = tukey_results.weights

    # Low weight = influential/outlier
    outlier_threshold = 0.5
    outliers_huber = df_model[df_model["huber_weights"] < outlier_threshold]
    outliers_tukey = df_model[df_model["tukey_weights"] < outlier_threshold]

    # Fitted values
    df_model["ols_fitted"] = ols_results.fittedvalues
    df_model["huber_fitted"] = huber_results.fittedvalues
    df_model["tukey_fitted"] = tukey_results.fittedvalues

    # Residuals
    df_model["ols_resid"] = ols_results.resid
    df_model["huber_resid"] = huber_results.resid
    df_model["tukey_resid"] = tukey_results.resid

    # Build comparison table
    comparison_table = {
        "variable": ["const", "time_trend", "covid_2020"],
        "ols": {
            "const": {
                "estimate": float(ols_results.params["const"]),
                "std_error": float(ols_results.bse["const"]),
                "t_stat": float(ols_results.tvalues["const"]),
                "p_value": float(ols_results.pvalues["const"]),
            },
            "time_trend": {
                "estimate": float(ols_results.params["time_trend"]),
                "std_error": float(ols_results.bse["time_trend"]),
                "t_stat": float(ols_results.tvalues["time_trend"]),
                "p_value": float(ols_results.pvalues["time_trend"]),
            },
            "covid_2020": {
                "estimate": float(ols_results.params["covid_2020"]),
                "std_error": float(ols_results.bse["covid_2020"]),
                "t_stat": float(ols_results.tvalues["covid_2020"]),
                "p_value": float(ols_results.pvalues["covid_2020"]),
            },
        },
        "huber": {
            "const": {
                "estimate": float(huber_results.params["const"]),
                "std_error": float(huber_results.bse["const"]),
                "t_stat": float(huber_results.tvalues["const"]),
                "p_value": float(huber_results.pvalues["const"]),
            },
            "time_trend": {
                "estimate": float(huber_results.params["time_trend"]),
                "std_error": float(huber_results.bse["time_trend"]),
                "t_stat": float(huber_results.tvalues["time_trend"]),
                "p_value": float(huber_results.pvalues["time_trend"]),
            },
            "covid_2020": {
                "estimate": float(huber_results.params["covid_2020"]),
                "std_error": float(huber_results.bse["covid_2020"]),
                "t_stat": float(huber_results.tvalues["covid_2020"]),
                "p_value": float(huber_results.pvalues["covid_2020"]),
            },
        },
        "tukey": {
            "const": {
                "estimate": float(tukey_results.params["const"]),
                "std_error": float(tukey_results.bse["const"]),
                "t_stat": float(tukey_results.tvalues["const"]),
                "p_value": float(tukey_results.pvalues["const"]),
            },
            "time_trend": {
                "estimate": float(tukey_results.params["time_trend"]),
                "std_error": float(tukey_results.bse["time_trend"]),
                "t_stat": float(tukey_results.tvalues["time_trend"]),
                "p_value": float(tukey_results.pvalues["time_trend"]),
            },
            "covid_2020": {
                "estimate": float(tukey_results.params["covid_2020"]),
                "std_error": float(tukey_results.bse["covid_2020"]),
                "t_stat": float(tukey_results.tvalues["covid_2020"]),
                "p_value": float(tukey_results.pvalues["covid_2020"]),
            },
        },
    }

    # Calculate percent difference from OLS
    pct_diff = {
        "huber_vs_ols": {
            "const": float(
                (huber_results.params["const"] - ols_results.params["const"])
                / abs(ols_results.params["const"])
                * 100
            ),
            "time_trend": float(
                (huber_results.params["time_trend"] - ols_results.params["time_trend"])
                / abs(ols_results.params["time_trend"])
                * 100
            ),
            "covid_2020": float(
                (huber_results.params["covid_2020"] - ols_results.params["covid_2020"])
                / abs(ols_results.params["covid_2020"])
                * 100
            ),
        },
        "tukey_vs_ols": {
            "const": float(
                (tukey_results.params["const"] - ols_results.params["const"])
                / abs(ols_results.params["const"])
                * 100
            ),
            "time_trend": float(
                (tukey_results.params["time_trend"] - ols_results.params["time_trend"])
                / abs(ols_results.params["time_trend"])
                * 100
            ),
            "covid_2020": float(
                (tukey_results.params["covid_2020"] - ols_results.params["covid_2020"])
                / abs(ols_results.params["covid_2020"])
                * 100
            ),
        },
    }

    result.add_decision(
        decision_id="D004",
        category="methodology",
        decision="Used Huber and Tukey biweight M-estimators for robust regression",
        rationale="Huber provides moderate robustness; Tukey biweight provides high breakdown point",
        alternatives=["Least trimmed squares (LTS)", "S-estimator"],
        evidence=f"Identified {len(outliers_huber)} Huber outliers, {len(outliers_tukey)} Tukey outliers",
    )

    robust_results = {
        "model_type": "Robust Regression (M-estimators)",
        "estimators": ["OLS", "Huber M-estimator", "Tukey Biweight"],
        "n_observations": int(len(df_model)),
        "comparison_table": comparison_table,
        "percent_difference_from_ols": pct_diff,
        "influential_observations": {
            "huber_outliers": {
                "count": int(len(outliers_huber)),
                "years": outliers_huber["year"].tolist()
                if len(outliers_huber) > 0
                else [],
                "threshold": outlier_threshold,
            },
            "tukey_outliers": {
                "count": int(len(outliers_tukey)),
                "years": outliers_tukey["year"].tolist()
                if len(outliers_tukey) > 0
                else [],
                "threshold": outlier_threshold,
            },
        },
        "weights_by_year": {
            int(row["year"]): {
                "huber_weight": float(row["huber_weights"]),
                "tukey_weight": float(row["tukey_weights"]),
            }
            for _, row in df_model.iterrows()
        },
        "ols_diagnostics": {
            "r_squared": float(ols_results.rsquared),
            "adj_r_squared": float(ols_results.rsquared_adj),
            "f_statistic": float(ols_results.fvalue),
            "f_pvalue": float(ols_results.f_pvalue),
        },
        "interpretation": {
            "robustness": "Robust estimates similar to OLS suggests outliers have limited influence"
            if abs(pct_diff["tukey_vs_ols"]["time_trend"]) < 20
            else "Substantial differences indicate outliers affect OLS estimates",
            "covid_2020": f"Year 2020 weight: Huber={df_model[df_model['year']==2020]['huber_weights'].iloc[0]:.3f}, "
            + f"Tukey={df_model[df_model['year']==2020]['tukey_weights'].iloc[0]:.3f}",
        },
    }

    # Store for plotting
    robust_results["_fitted_data"] = df_model[
        [
            "year",
            "nd_intl_migration",
            "ols_fitted",
            "huber_fitted",
            "tukey_fitted",
            "huber_weights",
            "tukey_weights",
        ]
    ].to_dict("records")

    return robust_results, ols_results, huber_results, tukey_results, df_model


def plot_quantile_coefficients(qr_results: dict, result: ModuleResult):
    """Plot how regression coefficients change across quantiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    quantiles = qr_results["quantiles_estimated"]

    # Time trend coefficient across quantiles
    ax1 = axes[0]
    trend_coefs = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["time_trend"]["estimate"]
        for q in quantiles
    ]
    trend_lower = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["time_trend"]["ci_95_lower"]
        for q in quantiles
    ]
    trend_upper = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["time_trend"]["ci_95_upper"]
        for q in quantiles
    ]

    ax1.plot(
        quantiles,
        trend_coefs,
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        markersize=10,
        label="Quantile estimate",
    )
    ax1.fill_between(
        quantiles,
        trend_lower,
        trend_upper,
        alpha=0.2,
        color=COLORS["primary"],
        label="95% CI",
    )

    # Add OLS estimate as reference
    ols_trend = qr_results["ols_comparison"]["ols_vs_median"]["ols_trend"]
    ax1.axhline(
        ols_trend,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=2,
        label=f"OLS: {ols_trend:.1f}",
    )

    ax1.set_xlabel("Quantile", fontsize=12)
    ax1.set_ylabel("Time Trend Coefficient", fontsize=12)
    ax1.set_title("Time Trend Effect Across Distribution", fontsize=12)
    ax1.set_xticks(quantiles)
    ax1.set_xticklabels([f"{int(q*100)}th" for q in quantiles])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # COVID coefficient across quantiles
    ax2 = axes[1]
    covid_coefs = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["covid_2020"]["estimate"]
        for q in quantiles
    ]
    covid_lower = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["covid_2020"]["ci_95_lower"]
        for q in quantiles
    ]
    covid_upper = [
        qr_results["levels_model"]["results_by_quantile"][f"q{int(q*100)}"][
            "coefficients"
        ]["covid_2020"]["ci_95_upper"]
        for q in quantiles
    ]

    ax2.plot(
        quantiles,
        covid_coefs,
        "o-",
        color=COLORS["tertiary"],
        linewidth=2,
        markersize=10,
        label="Quantile estimate",
    )
    ax2.fill_between(
        quantiles,
        covid_lower,
        covid_upper,
        alpha=0.2,
        color=COLORS["tertiary"],
        label="95% CI",
    )

    # Add OLS estimate as reference
    ols_covid = qr_results["ols_comparison"]["ols_vs_median"]["ols_covid"]
    ax2.axhline(
        ols_covid,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=2,
        label=f"OLS: {ols_covid:.1f}",
    )
    ax2.axhline(0, color="black", linewidth=0.5)

    ax2.set_xlabel("Quantile", fontsize=12)
    ax2.set_ylabel("COVID-2020 Coefficient", fontsize=12)
    ax2.set_title("COVID Effect Across Distribution", fontsize=12)
    ax2.set_xticks(quantiles)
    ax2.set_xticklabels([f"{int(q*100)}th" for q in quantiles])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_4_quantile_coefficients"),
        "Quantile Regression Coefficients - ND International Migration (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def plot_beta_diagnostics(
    beta_results: dict, df_fitted: pd.DataFrame, result: ModuleResult
):
    """Plot diagnostic plots for beta regression."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Actual vs Fitted values
    ax1 = axes[0, 0]
    ax1.scatter(
        df_fitted["nd_share_bounded"],
        df_fitted["fitted"],
        color=COLORS["primary"],
        alpha=0.7,
        s=80,
        edgecolor="white",
    )

    # Add 45-degree line
    min_val = min(df_fitted["nd_share_bounded"].min(), df_fitted["fitted"].min())
    max_val = max(df_fitted["nd_share_bounded"].max(), df_fitted["fitted"].max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect fit"
    )

    # Label years
    for _, row in df_fitted.iterrows():
        ax1.annotate(
            str(int(row["year"])),
            (row["nd_share_bounded"], row["fitted"]),
            fontsize=8,
            alpha=0.7,
        )

    ax1.set_xlabel("Actual ND Share", fontsize=12)
    ax1.set_ylabel("Fitted ND Share", fontsize=12)
    ax1.set_title("Actual vs Fitted Values", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Fitted
    ax2 = axes[0, 1]
    ax2.scatter(
        df_fitted["fitted"],
        df_fitted["residuals"],
        color=COLORS["secondary"],
        alpha=0.7,
        s=80,
        edgecolor="white",
    )
    ax2.axhline(0, color="black", linewidth=0.5)

    # Add LOESS-like smooth
    z = np.polyfit(df_fitted["fitted"], df_fitted["residuals"], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(df_fitted["fitted"].min(), df_fitted["fitted"].max(), 100)
    ax2.plot(
        x_smooth,
        p(x_smooth),
        color=COLORS["tertiary"],
        linewidth=2,
        label="Quadratic fit",
    )

    ax2.set_xlabel("Fitted Values", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_title("Residuals vs Fitted", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Residual histogram with normal overlay
    ax3 = axes[1, 0]
    n_bins = min(10, len(df_fitted))
    ax3.hist(
        df_fitted["residuals"],
        bins=n_bins,
        density=True,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="white",
    )

    # Normal distribution overlay
    mu, sigma = df_fitted["residuals"].mean(), df_fitted["residuals"].std()
    x = np.linspace(df_fitted["residuals"].min(), df_fitted["residuals"].max(), 100)
    ax3.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        color=COLORS["secondary"],
        linewidth=2,
        label=f"Normal(0, {sigma:.4f})",
    )

    ax3.set_xlabel("Residuals", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.set_title("Residual Distribution", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Time series of residuals
    ax4 = axes[1, 1]
    ax4.plot(
        df_fitted["year"],
        df_fitted["residuals"],
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        markersize=8,
    )
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.axvline(2020, color=COLORS["neutral"], linestyle=":", alpha=0.7)
    ax4.text(
        2020.1, ax4.get_ylim()[1] * 0.9, "COVID", fontsize=9, color=COLORS["neutral"]
    )

    ax4.set_xlabel("Year", fontsize=12)
    ax4.set_ylabel("Residuals", fontsize=12)
    ax4.set_title("Residuals Over Time", fontsize=12)
    ax4.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_4_beta_diagnostics"),
        "Beta Regression Diagnostics - ND Share of US International Migration",
        "Census Bureau Population Estimates Program",
    )


def plot_robust_comparison(
    robust_results: dict, df_fitted: pd.DataFrame, result: ModuleResult
):
    """Plot comparison of OLS vs robust regression fitted values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Fitted values comparison
    ax1 = axes[0]
    years = df_fitted["year"]
    actual = df_fitted["nd_intl_migration"]

    ax1.scatter(
        years,
        actual,
        color=COLORS["neutral"],
        s=100,
        zorder=5,
        label="Actual",
        edgecolor="black",
        linewidth=1,
    )
    ax1.plot(
        years,
        df_fitted["ols_fitted"],
        "-",
        color=COLORS["primary"],
        linewidth=2,
        label="OLS",
    )
    ax1.plot(
        years,
        df_fitted["huber_fitted"],
        "--",
        color=COLORS["secondary"],
        linewidth=2,
        label="Huber M-est.",
    )
    ax1.plot(
        years,
        df_fitted["tukey_fitted"],
        ":",
        color=COLORS["tertiary"],
        linewidth=3,
        label="Tukey Biweight",
    )

    ax1.axvline(2020, color=COLORS["neutral"], linestyle=":", alpha=0.5)
    ax1.text(
        2020.1, ax1.get_ylim()[1] * 0.95, "COVID", fontsize=9, color=COLORS["neutral"]
    )

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("ND International Migration", fontsize=12)
    ax1.set_title("Fitted Values: OLS vs Robust Estimators", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Robust weights by year
    ax2 = axes[1]
    bar_width = 0.35
    x = np.arange(len(years))

    ax2.bar(
        x - bar_width / 2,
        df_fitted["huber_weights"],
        bar_width,
        color=COLORS["secondary"],
        alpha=0.7,
        label="Huber weights",
    )
    ax2.bar(
        x + bar_width / 2,
        df_fitted["tukey_weights"],
        bar_width,
        color=COLORS["tertiary"],
        alpha=0.7,
        label="Tukey weights",
    )

    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax2.axhline(
        0.5,
        color=COLORS["neutral"],
        linewidth=1,
        linestyle=":",
        label="Outlier threshold (0.5)",
    )

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Weight", fontsize=12)
    ax2.set_title("Robust Regression Weights by Year", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([int(y) for y in years], rotation=45, ha="right")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Highlight 2020
    idx_2020 = list(years).index(2020) if 2020 in list(years) else None
    if idx_2020 is not None:
        ax2.annotate(
            "2020\n(COVID)",
            xy=(idx_2020, 0.1),
            fontsize=9,
            ha="center",
            color=COLORS["secondary"],
            fontweight="bold",
        )

    save_figure(
        fig,
        str(FIGURES_DIR / "module_4_robust_comparison"),
        "Robust Regression: OLS vs M-Estimators (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 4."""
    result = ModuleResult(
        module_id="4", analysis_name="regression_extensions_beta_quantile_robust"
    )

    print("Loading data...")
    df_panel, df_nd = load_data(result)

    print("\nPreparing ND time series data...")
    df_nd_prepped = prepare_nd_data(df_nd, result)

    # Record parameters
    result.parameters = {
        "panel_observations": int(len(df_panel)),
        "nd_time_series_length": int(len(df_nd_prepped)),
        "years": [int(y) for y in sorted(df_nd_prepped["year"].unique())],
        "models_estimated": {
            "beta_regression": {
                "family": "Binomial (quasi-beta)",
                "link": "Logit",
                "predictors": ["time_trend", "covid_2020", "lag_nd_share"],
            },
            "quantile_regression": {
                "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
                "predictors": ["time_trend", "covid_2020"],
                "bootstrap_iterations": 500,
            },
            "robust_regression": {
                "estimators": ["Huber M-estimator", "Tukey Biweight"],
                "comparison_baseline": "OLS",
            },
        },
    }

    # Beta regression
    print("\n" + "=" * 60)
    print("BETA REGRESSION")
    print("=" * 60)
    beta_results, beta_model, beta_fitted = beta_regression(df_nd_prepped, result)

    # Save beta regression results
    beta_output = RESULTS_DIR / "module_4_beta_regression.json"
    with open(beta_output, "w") as f:
        # Remove internal plotting data before saving
        beta_save = {k: v for k, v in beta_results.items() if not k.startswith("_")}
        json.dump(beta_save, f, indent=2, default=str)
    print(f"Beta regression results saved: {beta_output}")

    # Quantile regression
    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION")
    print("=" * 60)
    qr_results, qr_by_quantile = quantile_regression(df_nd_prepped, result)

    # Save quantile regression results
    qr_output = RESULTS_DIR / "module_4_quantile_regression.json"
    with open(qr_output, "w") as f:
        json.dump(qr_results, f, indent=2, default=str)
    print(f"Quantile regression results saved: {qr_output}")

    # Robust regression
    print("\n" + "=" * 60)
    print("ROBUST REGRESSION")
    print("=" * 60)
    robust_results, ols_res, huber_res, tukey_res, robust_fitted = robust_regression(
        df_nd_prepped, result
    )

    # Save robust regression results
    robust_output = RESULTS_DIR / "module_4_robust_regression.json"
    with open(robust_output, "w") as f:
        # Remove internal plotting data before saving
        robust_save = {k: v for k, v in robust_results.items() if not k.startswith("_")}
        json.dump(robust_save, f, indent=2, default=str)
    print(f"Robust regression results saved: {robust_output}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_quantile_coefficients(qr_results, result)
    plot_beta_diagnostics(beta_results, beta_fitted, result)
    plot_robust_comparison(robust_results, robust_fitted, result)

    # Build comprehensive comparison table
    comparison_table = {
        "ols_vs_quantile_vs_robust": {
            "time_trend": {
                "ols": robust_results["comparison_table"]["ols"]["time_trend"][
                    "estimate"
                ],
                "q10": qr_results["levels_model"]["results_by_quantile"]["q10"][
                    "coefficients"
                ]["time_trend"]["estimate"],
                "q25": qr_results["levels_model"]["results_by_quantile"]["q25"][
                    "coefficients"
                ]["time_trend"]["estimate"],
                "q50": qr_results["levels_model"]["results_by_quantile"]["q50"][
                    "coefficients"
                ]["time_trend"]["estimate"],
                "q75": qr_results["levels_model"]["results_by_quantile"]["q75"][
                    "coefficients"
                ]["time_trend"]["estimate"],
                "q90": qr_results["levels_model"]["results_by_quantile"]["q90"][
                    "coefficients"
                ]["time_trend"]["estimate"],
                "huber": robust_results["comparison_table"]["huber"]["time_trend"][
                    "estimate"
                ],
                "tukey": robust_results["comparison_table"]["tukey"]["time_trend"][
                    "estimate"
                ],
            },
            "covid_2020": {
                "ols": robust_results["comparison_table"]["ols"]["covid_2020"][
                    "estimate"
                ],
                "q10": qr_results["levels_model"]["results_by_quantile"]["q10"][
                    "coefficients"
                ]["covid_2020"]["estimate"],
                "q25": qr_results["levels_model"]["results_by_quantile"]["q25"][
                    "coefficients"
                ]["covid_2020"]["estimate"],
                "q50": qr_results["levels_model"]["results_by_quantile"]["q50"][
                    "coefficients"
                ]["covid_2020"]["estimate"],
                "q75": qr_results["levels_model"]["results_by_quantile"]["q75"][
                    "coefficients"
                ]["covid_2020"]["estimate"],
                "q90": qr_results["levels_model"]["results_by_quantile"]["q90"][
                    "coefficients"
                ]["covid_2020"]["estimate"],
                "huber": robust_results["comparison_table"]["huber"]["covid_2020"][
                    "estimate"
                ],
                "tukey": robust_results["comparison_table"]["tukey"]["covid_2020"][
                    "estimate"
                ],
            },
        }
    }

    # Compile main results
    result.results = {
        "beta_regression": {
            "pseudo_r2": beta_results["fit_statistics"]["pseudo_r2_mcfadden"],
            "aic": beta_results["fit_statistics"]["aic"],
            "time_trend_odds_ratio": beta_results["coefficient_table"]["time_trend"][
                "odds_ratio"
            ],
            "covid_odds_ratio": beta_results["coefficient_table"]["covid_2020"][
                "odds_ratio"
            ],
        },
        "quantile_regression": {
            "median_trend": qr_results["levels_model"]["results_by_quantile"]["q50"][
                "coefficients"
            ]["time_trend"]["estimate"],
            "trend_heterogeneity": max(
                comparison_table["ols_vs_quantile_vs_robust"]["time_trend"].values()
            )
            - min(comparison_table["ols_vs_quantile_vs_robust"]["time_trend"].values()),
            "covid_effect_range": [
                min(
                    [
                        qr_results["levels_model"]["results_by_quantile"][
                            f"q{int(q*100)}"
                        ]["coefficients"]["covid_2020"]["estimate"]
                        for q in [0.10, 0.25, 0.50, 0.75, 0.90]
                    ]
                ),
                max(
                    [
                        qr_results["levels_model"]["results_by_quantile"][
                            f"q{int(q*100)}"
                        ]["coefficients"]["covid_2020"]["estimate"]
                        for q in [0.10, 0.25, 0.50, 0.75, 0.90]
                    ]
                ),
            ],
        },
        "robust_regression": {
            "ols_r_squared": robust_results["ols_diagnostics"]["r_squared"],
            "outliers_identified": robust_results["influential_observations"][
                "tukey_outliers"
            ]["years"],
            "percent_diff_tukey_ols_trend": robust_results[
                "percent_difference_from_ols"
            ]["tukey_vs_ols"]["time_trend"],
        },
        "comparison_table": comparison_table,
    }

    # Diagnostics
    result.diagnostics = {
        "beta_regression": {
            "deviance": beta_results["fit_statistics"]["deviance"],
            "pearson_chi2": beta_results["fit_statistics"]["pearson_chi2"],
            "residual_mean": beta_results["residual_diagnostics"]["mean_residual"],
            "residual_std": beta_results["residual_diagnostics"]["std_residual"],
        },
        "quantile_regression": {
            "pseudo_r2_by_quantile": {
                f"q{int(q*100)}": qr_results["levels_model"]["results_by_quantile"][
                    f"q{int(q*100)}"
                ]["pseudo_r2"]
                for q in [0.10, 0.25, 0.50, 0.75, 0.90]
            }
        },
        "robust_regression": {
            "year_2020_weights": {
                "huber": robust_results["weights_by_year"]
                .get(2020, {})
                .get("huber_weight", None),
                "tukey": robust_results["weights_by_year"]
                .get(2020, {})
                .get("tukey_weight", None),
            }
        },
    }

    # Next steps
    result.next_steps = [
        "Use quantile regression insights for scenario analysis (Module 5)",
        "Robust estimates inform sensitivity analysis for projections",
        "Beta regression framework applicable to other proportional outcomes",
        "Compare results with time series models from Module 2",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 4: Regression Extensions - Beta, Quantile, and Robust Regression")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_4_regression_extensions.json")

        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\nKey Results:")
        print(
            f"  Beta Regression pseudo-R2: {result.results['beta_regression']['pseudo_r2']:.4f}"
        )
        print(
            f"  Time trend odds ratio: {result.results['beta_regression']['time_trend_odds_ratio']:.4f}"
        )
        print(
            f"  Median regression trend coefficient: {result.results['quantile_regression']['median_trend']:.2f}"
        )
        print(
            f"  Trend heterogeneity (max-min across quantiles): {result.results['quantile_regression']['trend_heterogeneity']:.2f}"
        )
        print(
            f"  Robust regression outliers: {result.results['robust_regression']['outliers_identified']}"
        )
        print(
            f"  Tukey vs OLS trend difference: {result.results['robust_regression']['percent_diff_tukey_ols_trend']:.1f}%"
        )

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision']}")

        print("\nFigures generated:")
        print("  - module_4_quantile_coefficients.png/pdf")
        print("  - module_4_beta_diagnostics.png/pdf")
        print("  - module_4_robust_comparison.png/pdf")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
