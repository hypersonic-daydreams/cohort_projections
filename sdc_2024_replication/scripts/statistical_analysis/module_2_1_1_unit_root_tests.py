#!/usr/bin/env python3
"""
Module 2.1.1: Unit Root Testing
===============================

Performs comprehensive unit root and stationarity tests on immigration time series
to determine integration order for subsequent time series modeling.

Tests Performed:
- Augmented Dickey-Fuller (ADF) test - H0: Unit root present (non-stationary)
- KPSS test - H0: Series is stationary
- Phillips-Perron test - H0: Unit root present
- Zivot-Andrews test - H0: Unit root with no break; HA: trend-stationary with one break

Variables Tested:
- nd_share_of_us_intl_pct (primary)
- nd_intl_migration
- us_intl_migration

Usage:
    micromamba run -n cohort_proj python module_2_1_1_unit_root_tests.py
"""

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

# Suppress warnings during execution
warnings.filterwarnings("ignore")

from data_loader import load_migration_summary

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Colorblind-safe palette
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
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
        self.decisions: list[dict] = []
        self.warnings: list[str] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        evidence: str | None = None,
        reversible: bool = True,
    ):
        """Add a documented decision to the log."""
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
            "decisions": self.decisions,
            "warnings": self.warnings,
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


def save_figure(
    fig, filepath_base, title, source_note="Census Bureau Population Estimates Program"
):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
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


def load_data(filename: str) -> pd.DataFrame:
    """Load data file from analysis directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def phillips_perron_test(series, regression="c"):
    """
    Perform Phillips-Perron test using ADF with lag=0 and Newey-West correction.

    The PP test is similar to ADF but uses non-parametric correction for serial
    correlation. For small samples, we approximate using ADF with minimal lags.

    Parameters:
    -----------
    series : array-like
        Time series data
    regression : str
        'c' for constant only, 'ct' for constant and trend

    Returns:
    --------
    dict with test statistic, p-value, critical values
    """
    # Use ADF with maxlag=0 as base (equivalent to Dickey-Fuller)
    # In small samples, PP test approximates this
    result = adfuller(series, maxlag=0, regression=regression, autolag=None)

    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": 0,
        "n_obs_used": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "note": "PP test approximated using DF test (maxlag=0) for small sample",
    }


def run_adf_test(series, series_name, maxlag=None, regression="c"):
    """
    Run Augmented Dickey-Fuller test with full diagnostics.

    Parameters:
    -----------
    series : array-like
        Time series data
    series_name : str
        Name of the series for reporting
    maxlag : int, optional
        Maximum lag to consider
    regression : str
        'c' for constant only, 'ct' for constant and trend

    Returns:
    --------
    dict with complete ADF test results
    """
    # Run ADF test with automatic lag selection (AIC)
    result = adfuller(series, maxlag=maxlag, regression=regression, autolag="AIC")

    # Extract results
    adf_result = {
        "series": series_name,
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "n_obs_used": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "ic_best": float(result[5]) if result[5] is not None else None,
        "regression_type": regression,
    }

    # Get regression coefficients by running OLS manually
    # This provides the actual coefficient estimates
    len(series)
    y = np.diff(series)
    y_lag = series[:-1]

    # Build regressor matrix based on regression type
    if regression == "c":
        X = np.column_stack([np.ones(len(y)), y_lag])
        coef_names = ["constant", "rho_minus_1"]
    elif regression == "ct":
        trend = np.arange(1, len(y) + 1)
        X = np.column_stack([np.ones(len(y)), trend, y_lag])
        coef_names = ["constant", "trend", "rho_minus_1"]
    else:
        X = y_lag.reshape(-1, 1)
        coef_names = ["rho_minus_1"]

    # Add lagged differences if lags > 0
    for i in range(1, result[2] + 1):
        if i < len(y):
            lagged_diff = np.zeros(len(y))
            lagged_diff[i:] = np.diff(series)[:-i]
            X = np.column_stack([X, lagged_diff])
            coef_names.append(f"diff_lag_{i}")

    # Ensure dimensions match
    if X.shape[0] != len(y):
        # Trim y to match X
        y = y[: X.shape[0]]

    try:
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        # Standard errors
        sigma2 = np.sum(residuals**2) / (len(y) - X.shape[1])
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        t_stats = beta / se

        # Build regression output
        regression_output = {}
        for i, name in enumerate(coef_names):
            regression_output[name] = {
                "coef": float(beta[i]),
                "se": float(se[i]),
                "t": float(t_stats[i]),
            }

        adf_result["regression"] = regression_output

    except Exception as e:
        adf_result["regression"] = {"error": str(e)}

    return adf_result


def run_kpss_test(series, series_name, regression="c"):
    """
    Run KPSS test with full diagnostics.

    Parameters:
    -----------
    series : array-like
        Time series data
    series_name : str
        Name of the series for reporting
    regression : str
        'c' for level stationarity, 'ct' for trend stationarity

    Returns:
    --------
    dict with complete KPSS test results
    """
    # Run KPSS test
    result = kpss(series, regression=regression, nlags="auto")

    kpss_result = {
        "series": series_name,
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": int(result[2]),
        "critical_values": {k: float(v) for k, v in result[3].items()},
        "regression_type": "level" if regression == "c" else "trend",
    }

    return kpss_result


def run_zivot_andrews_test(
    series,
    series_name,
    years=None,
    regression="c",
    trim=0.15,
    maxlag=1,
    autolag="AIC",
):
    """
    Run Zivot-Andrews single-break unit root test (break-robust).

    Parameters:
    -----------
    series : array-like
        Time series data
    series_name : str
        Name of the series for reporting
    years : list[int] | None
        Optional list of calendar years aligned to series indices
    regression : str
        'c' for break in intercept, 'ct' for break in trend and intercept
    trim : float
        Fraction to trim from each end when searching for a break
    maxlag : int, optional
        Maximum lag to consider
    autolag : str
        Autolag selection criterion

    Returns:
    --------
    dict with Zivot-Andrews test results
    """
    try:
        statistic, p_value, critical_values, used_lag, break_index = zivot_andrews(
            series,
            trim=trim,
            maxlag=maxlag,
            regression=regression,
            autolag=autolag,
        )
        break_year = None
        if years is not None:
            try:
                break_year = int(years[int(break_index)])
            except (IndexError, TypeError, ValueError):
                break_year = None

        return {
            "series": series_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "used_lag": int(used_lag),
            "break_index": int(break_index),
            "break_year": break_year,
            "regression_type": "level" if regression == "c" else "trend",
            "trim": trim,
            "autolag": autolag,
        }
    except Exception as exc:
        return {
            "series": series_name,
            "error": str(exc),
            "regression_type": "level" if regression == "c" else "trend",
            "trim": trim,
            "autolag": autolag,
        }


def determine_integration_order(adf_level, kpss_level, adf_diff, kpss_diff, alpha=0.05):
    """
    Determine integration order based on ADF and KPSS tests.

    Decision logic:
    - If ADF rejects H0 (no unit root) AND KPSS fails to reject H0 (stationary): I(0)
    - If ADF fails to reject H0 (unit root) AND KPSS rejects H0 (not stationary): I(1) or higher
    - If tests disagree: consider as borderline, lean toward I(1) for conservative approach

    Parameters:
    -----------
    adf_level : dict
        ADF test results for level series
    kpss_level : dict
        KPSS test results for level series
    adf_diff : dict
        ADF test results for first-differenced series
    kpss_diff : dict
        KPSS test results for first-differenced series
    alpha : float
        Significance level

    Returns:
    --------
    dict with integration order conclusion and evidence
    """
    # Level series conclusions
    adf_level_rejects = adf_level["p_value"] < alpha  # Rejects unit root
    kpss_level_rejects = kpss_level["p_value"] < alpha  # Rejects stationarity

    # Differenced series conclusions
    adf_diff_rejects = adf_diff["p_value"] < alpha
    kpss_diff_rejects = kpss_diff["p_value"] < alpha

    evidence = []

    # Case 1: Level series appears stationary (I(0))
    if adf_level_rejects and not kpss_level_rejects:
        order = 0
        conclusion = "I(0) - Level stationary"
        evidence.append(
            f"ADF rejects unit root at {alpha*100}% (p={adf_level['p_value']:.4f})"
        )
        evidence.append(
            f"KPSS fails to reject stationarity (p={kpss_level['p_value']:.4f})"
        )

    # Case 2: Level has unit root, difference is stationary (I(1))
    elif not adf_level_rejects and adf_diff_rejects:
        order = 1
        conclusion = "I(1) - First-difference stationary"
        evidence.append(
            f"ADF fails to reject unit root in level (p={adf_level['p_value']:.4f})"
        )
        evidence.append(
            f"ADF rejects unit root in first difference (p={adf_diff['p_value']:.4f})"
        )

    # Case 3: Tests disagree on level (borderline)
    elif adf_level_rejects and kpss_level_rejects:
        order = 1  # Conservative choice
        conclusion = (
            "Borderline - ADF and KPSS disagree, treated as I(1) conservatively"
        )
        evidence.append(f"ADF rejects unit root (p={adf_level['p_value']:.4f})")
        evidence.append(
            f"KPSS also rejects stationarity (p={kpss_level['p_value']:.4f})"
        )
        evidence.append("Conservative approach: treat as I(1)")

    elif not adf_level_rejects and not kpss_level_rejects:
        order = 1  # Conservative choice
        conclusion = (
            "Borderline - ADF and KPSS disagree, treated as I(1) conservatively"
        )
        evidence.append(f"ADF fails to reject unit root (p={adf_level['p_value']:.4f})")
        evidence.append(
            f"KPSS fails to reject stationarity (p={kpss_level['p_value']:.4f})"
        )
        evidence.append("Conservative approach: treat as I(1)")

    # Case 4: Both level and difference appear non-stationary (I(2) or data issues)
    elif not adf_diff_rejects:
        order = 2
        conclusion = (
            "Possibly I(2) or data issues - both level and difference non-stationary"
        )
        evidence.append(
            f"ADF fails to reject unit root in level (p={adf_level['p_value']:.4f})"
        )
        evidence.append(
            f"ADF fails to reject unit root in first difference (p={adf_diff['p_value']:.4f})"
        )

    else:
        order = 1
        conclusion = "I(1) - Default assumption"
        evidence.append("Unable to determine clearly, defaulting to I(1)")

    return {
        "integration_order": order,
        "conclusion": conclusion,
        "evidence": evidence,
        "level_tests": {
            "adf_rejects_unit_root": adf_level_rejects,
            "kpss_rejects_stationarity": kpss_level_rejects,
        },
        "differenced_tests": {
            "adf_rejects_unit_root": adf_diff_rejects,
            "kpss_rejects_stationarity": kpss_diff_rejects,
        },
        "significance_level": alpha,
    }


def plot_original_series(df, variables, year_col="year"):
    """Plot original time series for all variables."""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))

    if n_vars == 1:
        axes = [axes]

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    for i, (var, ax) in enumerate(zip(variables, axes, strict=False)):
        ax.plot(
            df[year_col],
            df[var],
            marker="o",
            linewidth=2,
            color=colors[i % len(colors)],
            markersize=6,
        )
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(var.replace("_", " ").title(), fontsize=11)
        ax.set_title(f"{var}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark 2020 COVID year
        if 2020 in df[year_col].values:
            covid_idx = df[df[year_col] == 2020].index[0]
            ax.axvline(x=2020, color=COLORS["neutral"], linestyle="--", alpha=0.5)
            ax.annotate(
                "COVID",
                (2020, df.loc[covid_idx, var]),
                textcoords="offset points",
                xytext=(5, 10),
                fontsize=9,
                color=COLORS["neutral"],
            )

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_1_original_series")
    fig.suptitle(
        "Original Immigration Time Series (2010-2024)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    fig.savefig(f"{filepath}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return f"{filepath}.png"


def plot_differenced_series(df, variables, year_col="year"):
    """Plot first-differenced time series for all variables."""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))

    if n_vars == 1:
        axes = [axes]

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    for i, (var, ax) in enumerate(zip(variables, axes, strict=False)):
        diff_series = df[var].diff().dropna()
        years = df[year_col].iloc[1:]  # Skip first year (no diff)

        ax.plot(
            years,
            diff_series,
            marker="o",
            linewidth=2,
            color=colors[i % len(colors)],
            markersize=6,
        )
        ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.5)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(f"d({var})", fontsize=11)
        ax.set_title(f"First Difference: {var}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_1_differenced_series")
    fig.suptitle(
        "First-Differenced Immigration Time Series",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    fig.savefig(f"{filepath}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return f"{filepath}.png"


def plot_acf_series(series, series_name, is_differenced=False):
    """Plot ACF with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate maximum lags (rule of thumb: n/4 or 10, whichever is smaller)
    max_lags = min(len(series) // 4, 10)
    if max_lags < 2:
        max_lags = 2

    plot_acf(
        series,
        ax=ax,
        lags=max_lags,
        alpha=0.05,
        color=COLORS["primary"],
        vlines_kwargs={"color": COLORS["primary"]},
    )

    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.grid(True, alpha=0.3)

    suffix = "_differenced" if is_differenced else "_original"
    series_label = "First-Differenced" if is_differenced else "Original"

    title = f"Autocorrelation Function: {series_name} ({series_label})"

    filepath = str(FIGURES_DIR / f"module_2_1_1_acf{suffix}")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    plt.tight_layout()
    fig.savefig(f"{filepath}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return f"{filepath}.png"


def run_analysis() -> ModuleResult:
    """
    Main analysis function - perform unit root tests.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(
        module_id="2.1.1",
        analysis_name="unit_root_tests",
    )

    # Load required data
    print("\nLoading data...")
    # Load required data
    print("\nLoading data...")
    # df = load_data("nd_migration_summary.csv")
    df = load_migration_summary()
    result.input_files.append("census.state_components (PostgreSQL)")

    print(
        f"  Loaded {len(df)} observations (years {df['year'].min()}-{df['year'].max()})"
    )

    # Define variables to test
    variables_to_test = [
        "nd_share_of_us_intl_pct",
        "nd_intl_migration",
        "us_intl_migration",
    ]

    # Record parameters
    result.parameters = {
        "variables_tested": variables_to_test,
        "n_observations": len(df),
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "significance_level": 0.05,
        "adf_autolag": "AIC",
        "kpss_nlags": "auto",
        "zivot_andrews_regression": "c",
        "zivot_andrews_trim": 0.15,
        "zivot_andrews_maxlag": 1,
    }

    # Document decision about COVID year
    result.add_decision(
        decision_id="D001",
        category="data_handling",
        decision="Include 2020 COVID year in analysis",
        rationale="With only 15 observations, excluding any year significantly reduces test power. "
        "2020 anomaly is noted but retained for completeness. Sensitivity without 2020 "
        "could be performed in follow-up analysis.",
        alternatives=["Exclude 2020 entirely", "Replace 2020 with interpolated value"],
        evidence=f"2020 nd_intl_migration={df[df['year']==2020]['nd_intl_migration'].values[0]} vs mean={df['nd_intl_migration'].mean():.0f}",
        reversible=True,
    )

    # Document short series limitation
    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Proceed with unit root tests despite short series",
        rationale="With n=15, asymptotic distributions may not hold well. Tests have reduced power. "
        "Results should be interpreted with caution.",
        alternatives=["Skip unit root testing", "Use only visual inspection"],
        evidence="Standard recommendation is n>=25 for reliable ADF tests",
        reversible=False,
    )

    result.warnings.append(
        "Short time series (n=15) limits test power and may affect asymptotic validity of tests."
    )
    result.warnings.append(
        "2020 COVID year shows anomalous values that may affect test results."
    )
    result.warnings.append(
        "Zivot-Andrews single-break test is low power in n=15; interpret as diagnostic only."
    )

    # Store results for each variable
    result.results = {"variables": {}, "integration_order_summary": {}}

    # Run tests for each variable
    for var in variables_to_test:
        print(f"\nTesting variable: {var}")
        series = df[var].values

        var_results = {
            "n_observations": len(series),
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
        }

        # Level series tests
        print("  Running ADF test on level series...")
        adf_level = run_adf_test(series, var, regression="c")
        var_results["adf_level"] = adf_level
        print(
            f"    ADF statistic: {adf_level['statistic']:.4f}, p-value: {adf_level['p_value']:.4f}"
        )

        print("  Running KPSS test on level series...")
        kpss_level = run_kpss_test(series, var, regression="c")
        var_results["kpss_level"] = kpss_level
        print(
            f"    KPSS statistic: {kpss_level['statistic']:.4f}, p-value: {kpss_level['p_value']:.4f}"
        )

        print("  Running Phillips-Perron test on level series...")
        pp_level = phillips_perron_test(series, regression="c")
        var_results["phillips_perron_level"] = pp_level
        print(
            f"    PP statistic: {pp_level['statistic']:.4f}, p-value: {pp_level['p_value']:.4f}"
        )

        zivot_level = run_zivot_andrews_test(
            series,
            var,
            years=df["year"].tolist(),
            regression="c",
            maxlag=1,
        )
        var_results["zivot_andrews_level"] = zivot_level

        # First-differenced series tests
        diff_series = np.diff(series)

        print("  Running ADF test on differenced series...")
        adf_diff = run_adf_test(diff_series, f"d({var})", regression="c")
        var_results["adf_differenced"] = adf_diff
        print(
            f"    ADF statistic: {adf_diff['statistic']:.4f}, p-value: {adf_diff['p_value']:.4f}"
        )

        print("  Running KPSS test on differenced series...")
        kpss_diff = run_kpss_test(diff_series, f"d({var})", regression="c")
        var_results["kpss_differenced"] = kpss_diff
        print(
            f"    KPSS statistic: {kpss_diff['statistic']:.4f}, p-value: {kpss_diff['p_value']:.4f}"
        )

        # Determine integration order
        integration = determine_integration_order(
            adf_level, kpss_level, adf_diff, kpss_diff
        )
        var_results["integration_order"] = integration

        result.results["variables"][var] = var_results
        result.results["integration_order_summary"][var] = {
            "order": integration["integration_order"],
            "conclusion": integration["conclusion"],
        }

        print(
            f"  Integration order: I({integration['integration_order']}) - {integration['conclusion']}"
        )

    # Generate figures
    print("\nGenerating figures...")

    # Original series plot
    fig1 = plot_original_series(df, variables_to_test)
    print(f"  Saved: {fig1}")

    # Differenced series plot
    fig2 = plot_differenced_series(df, variables_to_test)
    print(f"  Saved: {fig2}")

    # ACF plots for primary variable
    primary_var = "nd_share_of_us_intl_pct"
    fig3 = plot_acf_series(df[primary_var].values, primary_var, is_differenced=False)
    print(f"  Saved: {fig3}")

    fig4 = plot_acf_series(
        np.diff(df[primary_var].values), primary_var, is_differenced=True
    )
    print(f"  Saved: {fig4}")

    # Store diagnostics
    result.diagnostics = {
        "figures_generated": [
            "module_2_1_1_original_series.png",
            "module_2_1_1_differenced_series.png",
            "module_2_1_1_acf_original.png",
            "module_2_1_1_acf_differenced.png",
        ],
        "test_interpretation": {
            "adf": "H0: Unit root present (non-stationary). Reject if p < 0.05.",
            "kpss": "H0: Series is stationary. Reject if p < 0.05.",
            "phillips_perron": "H0: Unit root present. Reject if p < 0.05.",
            "zivot_andrews": "H0: Unit root with no break. Reject if p < 0.05 (single-break alternative).",
        },
        "sample_size_concern": "n=15 is below recommended minimum of 25 for reliable unit root tests",
    }

    # Suggest next steps
    result.next_steps = [
        "Use integration order results for ARIMA model specification (Module 2.1)",
        "Consider differencing before structural break analysis (Module 2.1.2)",
        "Apply appropriate differencing in VAR/VECM models (Module 2.2)",
        "Sensitivity analysis excluding 2020 COVID year if results are borderline",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 2.1.1: Unit Root Testing")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_2_1_1_unit_root_tests.json")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")

        if result.decisions:
            print(f"\nDecisions made ({len(result.decisions)}):")
            for d in result.decisions:
                print(f"  - {d['decision_id']}: {d['decision']}")

        # Print summary
        print("\n" + "-" * 60)
        print("Integration Order Summary:")
        for var, summary in result.results["integration_order_summary"].items():
            print(f"  {var}: I({summary['order']}) - {summary['conclusion']}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
