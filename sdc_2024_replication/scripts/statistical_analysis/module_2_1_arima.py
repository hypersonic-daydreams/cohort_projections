#!/usr/bin/env python3
"""
Module 2.1: ARIMA Time Series Modeling
======================================

Performs ARIMA time series forecasting for ND international migration using
unit root test results from Module 2.1.1 to inform model specification.

Analyses Performed:
- Auto-ARIMA model selection using pmdarima
- Model estimation with full coefficient extraction
- Diagnostic testing (Ljung-Box, Jarque-Bera, ACF/PACF residuals)
- 5-year forecasting (2025-2029) with prediction intervals
- Comprehensive SPSS-style outputs

Dependencies:
- Module 2.1.1 (unit root tests) - provides integration order

Usage:
    micromamba run -n cohort_proj python module_2_1_arima.py
"""

import json
import sys
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pmdarima for auto_arima
import pmdarima as pm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings during execution
warnings.filterwarnings("ignore")

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
        self.dependencies: dict = {}

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
            "dependencies": self.dependencies,
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


def load_dependency(filename: str) -> dict:
    """Load results from a dependency module."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dependency file not found: {filepath}")

    with open(filepath) as f:
        return json.load(f)


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


def run_auto_arima(
    series: np.ndarray, d: int = None, seasonal: bool = False, m: int = 1
) -> dict:
    """
    Run auto_arima to find optimal ARIMA specification.

    Parameters:
    -----------
    series : np.ndarray
        Time series data
    d : int, optional
        Pre-specified differencing order from unit root tests
    seasonal : bool
        Whether to test for seasonal components
    m : int
        Seasonal period (if seasonal=True)

    Returns:
    --------
    dict with model selection results
    """
    # Run auto_arima
    auto_model = pm.auto_arima(
        series,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=d,  # Use pre-determined d from unit root tests
        start_P=0,
        max_P=2,
        start_Q=0,
        max_Q=2,
        D=0 if not seasonal else None,
        m=m,
        seasonal=seasonal,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
        with_intercept=True,
        n_fits=50,
    )

    # Extract results
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order if seasonal else None

    return {
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": float(auto_model.aic()),
        "bic": float(auto_model.bic()),
        "aicc": float(auto_model.aicc()) if hasattr(auto_model, "aicc") else None,
        "model": auto_model,
    }


def estimate_arima_model(
    series: np.ndarray, order: tuple, exog: np.ndarray = None
) -> dict:
    """
    Estimate ARIMA model and extract comprehensive statistics.

    Parameters:
    -----------
    series : np.ndarray
        Time series data
    order : tuple
        ARIMA order (p, d, q)
    exog : np.ndarray, optional
        Exogenous variables

    Returns:
    --------
    dict with full model estimation results
    """
    p, d, q = order

    # Fit ARIMA model
    model = ARIMA(series, order=order, exog=exog)
    results = model.fit()

    # Extract parameters - handle both pandas Series and numpy array cases
    params = {}

    # Get parameter names - may be index attribute or need to reconstruct
    try:
        param_names = list(results.params.index)
    except AttributeError:
        # For simple ARIMA models, params may be numpy array
        # Reconstruct names based on model specification
        param_names = []
        for i in range(1, p + 1):
            param_names.append(f"ar.L{i}")
        for i in range(1, q + 1):
            param_names.append(f"ma.L{i}")
        param_names.append("sigma2")

    # Convert params to dict-like access
    param_values = results.params
    bse_values = results.bse
    tvalues = results.tvalues if hasattr(results, "tvalues") else None
    pvalues = results.pvalues if hasattr(results, "pvalues") else None

    # Try to get confidence intervals
    try:
        conf_int = results.conf_int()
    except Exception:
        conf_int = None

    # Get AR coefficients
    for i in range(1, p + 1):
        param_name = f"ar.L{i}"
        if param_name in param_names:
            idx = param_names.index(param_name)
            param_entry = {
                "coef": float(param_values[idx])
                if hasattr(param_values, "__getitem__")
                else float(param_values),
                "se": float(bse_values[idx])
                if bse_values is not None and idx < len(bse_values)
                else None,
            }
            if tvalues is not None and idx < len(tvalues):
                param_entry["z"] = float(tvalues[idx])
            if pvalues is not None and idx < len(pvalues):
                param_entry["p_value"] = float(pvalues[idx])
            if conf_int is not None:
                try:
                    param_entry["ci_95"] = [
                        float(conf_int.iloc[idx, 0]),
                        float(conf_int.iloc[idx, 1]),
                    ]
                except Exception:
                    param_entry["ci_95"] = None
            params[f"ar_{i}"] = param_entry

    # Get MA coefficients
    for i in range(1, q + 1):
        param_name = f"ma.L{i}"
        if param_name in param_names:
            idx = param_names.index(param_name)
            param_entry = {
                "coef": float(param_values[idx])
                if hasattr(param_values, "__getitem__")
                else float(param_values),
                "se": float(bse_values[idx])
                if bse_values is not None and idx < len(bse_values)
                else None,
            }
            if tvalues is not None and idx < len(tvalues):
                param_entry["z"] = float(tvalues[idx])
            if pvalues is not None and idx < len(pvalues):
                param_entry["p_value"] = float(pvalues[idx])
            if conf_int is not None:
                try:
                    param_entry["ci_95"] = [
                        float(conf_int.iloc[idx, 0]),
                        float(conf_int.iloc[idx, 1]),
                    ]
                except Exception:
                    param_entry["ci_95"] = None
            params[f"ma_{i}"] = param_entry

    # Get sigma2 (variance of residuals)
    if "sigma2" in param_names:
        idx = param_names.index("sigma2")
        params["sigma2"] = {
            "coef": float(param_values[idx])
            if hasattr(param_values, "__getitem__")
            else float(param_values),
            "se": float(bse_values[idx])
            if bse_values is not None and idx < len(bse_values)
            else None,
        }
    elif len(param_values) > 0:
        # For simple random walk (0,1,0), sigma2 is the only parameter
        params["sigma2"] = {
            "coef": float(param_values[-1])
            if hasattr(param_values, "__getitem__")
            else float(param_values),
            "se": float(bse_values[-1])
            if bse_values is not None and len(bse_values) > 0
            else None,
        }

    # Fit statistics
    fit_stats = {
        "aic": float(results.aic),
        "bic": float(results.bic),
        "hqic": float(results.hqic) if hasattr(results, "hqic") else None,
        "log_likelihood": float(results.llf),
        "n_observations": int(results.nobs),
    }

    return {
        "order": order,
        "parameters": params,
        "fit_statistics": fit_stats,
        "results_object": results,
    }


def run_diagnostics(results, series: np.ndarray) -> dict:
    """
    Run comprehensive model diagnostics.

    Parameters:
    -----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results
    series : np.ndarray
        Original time series

    Returns:
    --------
    dict with diagnostic test results
    """
    residuals = results.resid

    # Ljung-Box test for serial correlation at various lags
    # Adjust lags based on sample size
    n = len(residuals)
    max_lag = min(10, n // 2)

    ljung_box_results = {}
    for lag in [5, 10, 15, 20]:
        if lag <= max_lag:
            try:
                lb_result = acorr_ljungbox(residuals, lags=[lag], return_df=True)
                ljung_box_results[f"lag_{lag}"] = {
                    "statistic": float(lb_result["lb_stat"].values[0]),
                    "df": lag - (results.model.order[0] + results.model.order[2]),
                    "p_value": float(lb_result["lb_pvalue"].values[0]),
                }
            except Exception as e:
                ljung_box_results[f"lag_{lag}"] = {"error": str(e)}
        else:
            ljung_box_results[f"lag_{lag}"] = {
                "skipped": f"Lag {lag} exceeds max_lag {max_lag} for sample size {n}"
            }

    # Residual statistics
    resid_mean = float(np.mean(residuals))
    resid_std = float(np.std(residuals))
    resid_skew = float(stats.skew(residuals))
    resid_kurt = float(stats.kurtosis(residuals))

    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(residuals)

    # Shapiro-Wilk test for normality (better for small samples)
    sw_stat, sw_pval = stats.shapiro(residuals)

    residual_diagnostics = {
        "mean": resid_mean,
        "std": resid_std,
        "skewness": resid_skew,
        "kurtosis": resid_kurt,
        "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_pval)},
        "shapiro_wilk": {"statistic": float(sw_stat), "p_value": float(sw_pval)},
    }

    return {
        "ljung_box": ljung_box_results,
        "residual_diagnostics": residual_diagnostics,
    }


def generate_forecasts(results, n_periods: int = 5, alpha: float = 0.05) -> list:
    """
    Generate forecasts with prediction intervals.

    Parameters:
    -----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results
    n_periods : int
        Number of periods to forecast
    alpha : float
        Significance level for prediction intervals

    Returns:
    --------
    list of forecast dictionaries
    """
    # Generate forecasts
    forecast = results.get_forecast(steps=n_periods)
    forecast_mean = forecast.predicted_mean
    conf_int_95 = forecast.conf_int(alpha=0.05)
    conf_int_80 = forecast.conf_int(alpha=0.20)
    se_mean = forecast.se_mean

    # Handle both pandas Series and numpy array cases
    def get_value(arr, idx):
        """Get value from array/series at index."""
        try:
            return float(arr.iloc[idx])
        except (AttributeError, TypeError):
            return float(arr[idx])

    def get_row(arr, idx, col):
        """Get value from 2D array/dataframe at position."""
        try:
            return float(arr.iloc[idx, col])
        except (AttributeError, TypeError):
            return float(arr[idx, col])

    forecasts = []
    for i in range(n_periods):
        forecasts.append(
            {
                "horizon": i + 1,
                "point": get_value(forecast_mean, i),
                "se": get_value(se_mean, i),
                "ci_80": [get_row(conf_int_80, i, 0), get_row(conf_int_80, i, 1)],
                "ci_95": [get_row(conf_int_95, i, 0), get_row(conf_int_95, i, 1)],
            }
        )

    return forecasts


def plot_series_with_fitted(
    df: pd.DataFrame, results, series_col: str, year_col: str = "year"
):
    """Plot original series with fitted values overlay."""
    fig, ax = setup_figure(figsize=(12, 7))

    years = df[year_col].values
    actual = df[series_col].values
    fitted = results.fittedvalues

    # Plot actual values
    ax.plot(
        years,
        actual,
        marker="o",
        linewidth=2,
        color=COLORS["primary"],
        label="Actual",
        markersize=8,
    )

    # Plot fitted values (align with actual)
    fitted_years = years[: len(fitted)]
    ax.plot(
        fitted_years,
        fitted,
        linestyle="--",
        linewidth=2,
        color=COLORS["secondary"],
        label="Fitted",
        alpha=0.8,
    )

    # Mark 2020 COVID year
    if 2020 in years:
        covid_idx = list(years).index(2020)
        ax.axvline(x=2020, color=COLORS["neutral"], linestyle=":", alpha=0.5)
        ax.annotate(
            "COVID",
            (2020, actual[covid_idx]),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=9,
            color=COLORS["neutral"],
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("International Migration (persons)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)

    filepath = str(FIGURES_DIR / "module_2_1_series_with_fitted")
    save_figure(
        fig, filepath, "ND International Migration: Actual vs Fitted Values (2010-2024)"
    )

    return f"{filepath}.png"


def plot_acf_pacf_original(series: np.ndarray, series_name: str):
    """Plot ACF and PACF for original series."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_lags = min(len(series) // 4, 10)
    if max_lags < 2:
        max_lags = 2

    # ACF
    plot_acf(
        series,
        ax=axes[0],
        lags=max_lags,
        alpha=0.05,
        color=COLORS["primary"],
        vlines_kwargs={"color": COLORS["primary"]},
    )
    axes[0].set_xlabel("Lag", fontsize=12)
    axes[0].set_ylabel("Autocorrelation", fontsize=12)
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # PACF
    plot_pacf(
        series,
        ax=axes[1],
        lags=max_lags,
        alpha=0.05,
        method="ywm",
        color=COLORS["secondary"],
        vlines_kwargs={"color": COLORS["secondary"]},
    )
    axes[1].set_xlabel("Lag", fontsize=12)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=12)
    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_acf_pacf_original")
    fig.suptitle(f"ACF and PACF: {series_name}", fontsize=14, fontweight="bold", y=1.02)
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


def plot_acf_pacf_residuals(residuals: np.ndarray):
    """Plot ACF and PACF for model residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_lags = min(len(residuals) // 4, 10)
    if max_lags < 2:
        max_lags = 2

    # ACF
    plot_acf(
        residuals,
        ax=axes[0],
        lags=max_lags,
        alpha=0.05,
        color=COLORS["primary"],
        vlines_kwargs={"color": COLORS["primary"]},
    )
    axes[0].set_xlabel("Lag", fontsize=12)
    axes[0].set_ylabel("Autocorrelation", fontsize=12)
    axes[0].set_title("ACF of Residuals", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # PACF
    plot_pacf(
        residuals,
        ax=axes[1],
        lags=max_lags,
        alpha=0.05,
        method="ywm",
        color=COLORS["secondary"],
        vlines_kwargs={"color": COLORS["secondary"]},
    )
    axes[1].set_xlabel("Lag", fontsize=12)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=12)
    axes[1].set_title("PACF of Residuals", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_acf_pacf_residuals")
    fig.suptitle(
        "Model Residuals: ACF and PACF", fontsize=14, fontweight="bold", y=1.02
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


def plot_residual_diagnostics(residuals: np.ndarray):
    """Plot residual diagnostic plots (histogram, Q-Q, time series)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residual time series
    ax = axes[0, 0]
    ax.plot(residuals, marker="o", linewidth=1.5, color=COLORS["primary"], markersize=5)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.7)
    ax.axhline(
        y=2 * np.std(residuals), color=COLORS["secondary"], linestyle="--", alpha=0.5
    )
    ax.axhline(
        y=-2 * np.std(residuals), color=COLORS["secondary"], linestyle="--", alpha=0.5
    )
    ax.set_xlabel("Observation", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title("Residuals Over Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Histogram with normal curve
    ax = axes[0, 1]
    n_bins = min(len(residuals) // 2, 10)
    ax.hist(
        residuals,
        bins=n_bins,
        density=True,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="white",
    )

    # Overlay normal distribution
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(
        x,
        stats.norm.pdf(x, np.mean(residuals), np.std(residuals)),
        color=COLORS["secondary"],
        linewidth=2,
        label="Normal",
    )
    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_color(COLORS["primary"])
    ax.get_lines()[0].set_markersize(8)
    ax.get_lines()[1].set_color(COLORS["secondary"])
    ax.set_title("Q-Q Plot (Normal)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Residuals vs fitted (if available)
    ax = axes[1, 1]
    ax.scatter(
        range(len(residuals)), residuals, color=COLORS["primary"], alpha=0.7, s=50
    )
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.7)
    ax.set_xlabel("Fitted Value Index", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title("Residuals vs Index", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_residual_diagnostics")
    fig.suptitle(
        "ARIMA Model Residual Diagnostics", fontsize=14, fontweight="bold", y=1.02
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


def plot_forecast_fan_chart(
    df: pd.DataFrame, results, forecasts: list, series_col: str, year_col: str = "year"
):
    """Plot forecast fan chart with prediction intervals."""
    fig, ax = setup_figure(figsize=(14, 8))

    years = df[year_col].values
    actual = df[series_col].values
    last_year = int(years[-1])

    # Plot historical data
    ax.plot(
        years,
        actual,
        marker="o",
        linewidth=2,
        color=COLORS["primary"],
        label="Historical",
        markersize=8,
    )

    # Forecast years
    forecast_years = [last_year + f["horizon"] for f in forecasts]
    forecast_values = [f["point"] for f in forecasts]
    ci_95_lower = [f["ci_95"][0] for f in forecasts]
    ci_95_upper = [f["ci_95"][1] for f in forecasts]
    ci_80_lower = [f["ci_80"][0] for f in forecasts]
    ci_80_upper = [f["ci_80"][1] for f in forecasts]

    # Connect last historical point to first forecast
    connect_years = [last_year] + forecast_years
    connect_values = [actual[-1]] + forecast_values

    # Plot forecast line
    ax.plot(
        connect_years,
        connect_values,
        marker="s",
        linewidth=2,
        color=COLORS["secondary"],
        label="Forecast",
        markersize=8,
    )

    # Plot 95% CI (lighter)
    ax.fill_between(
        [last_year] + forecast_years,
        [actual[-1]] + ci_95_lower,
        [actual[-1]] + ci_95_upper,
        alpha=0.2,
        color=COLORS["secondary"],
        label="95% CI",
    )

    # Plot 80% CI (darker)
    ax.fill_between(
        [last_year] + forecast_years,
        [actual[-1]] + ci_80_lower,
        [actual[-1]] + ci_80_upper,
        alpha=0.3,
        color=COLORS["secondary"],
        label="80% CI",
    )

    # Vertical line at forecast start
    ax.axvline(x=last_year + 0.5, color=COLORS["neutral"], linestyle="--", alpha=0.7)
    ax.text(
        last_year + 0.6,
        ax.get_ylim()[1] * 0.95,
        "Forecast",
        fontsize=10,
        color=COLORS["neutral"],
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("International Migration (persons)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)

    # Set x-axis to show all years
    all_years = list(years) + forecast_years
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(int(y)) for y in all_years], rotation=45)

    filepath = str(FIGURES_DIR / "module_2_1_forecast_fan_chart")
    save_figure(
        fig, filepath, "ND International Migration: 5-Year Forecast (2025-2029)"
    )

    return f"{filepath}.png"


def run_analysis() -> ModuleResult:
    """
    Main analysis function - perform ARIMA modeling.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(module_id="2.1", analysis_name="arima_time_series_modeling")

    # Load dependency - unit root test results
    print("\nLoading dependency: Module 2.1.1 (unit root tests)...")
    try:
        unit_root_results = load_dependency("module_2_1_1_unit_root_tests.json")
        result.dependencies = {
            "module_2_1_1": {
                "status": "loaded",
                "file": "results/module_2_1_1_unit_root_tests.json",
                "key_results_used": ["integration_order"],
            }
        }
        # Get integration order for nd_intl_migration
        integration_order = unit_root_results["results"]["integration_order_summary"][
            "nd_intl_migration"
        ]["order"]
        print(f"  Integration order for nd_intl_migration: I({integration_order})")
    except FileNotFoundError:
        print("  WARNING: Module 2.1.1 results not found, defaulting to d=1")
        integration_order = 1
        result.dependencies = {
            "module_2_1_1": {
                "status": "not_found",
                "fallback": "Using d=1 based on typical migration series properties",
            }
        }
        result.warnings.append(
            "Unit root test results not found. Using default d=1 for differencing."
        )

    # Load required data
    print("\nLoading data...")
    df = load_data("nd_migration_summary.csv")
    result.input_files.append("nd_migration_summary.csv")
    print(
        f"  Loaded {len(df)} observations (years {df['year'].min()}-{df['year'].max()})"
    )

    # Target variable
    target_var = "nd_intl_migration"
    series = df[target_var].values

    # Record parameters
    result.parameters = {
        "target_variable": target_var,
        "n_observations": len(df),
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "integration_order_from_unit_root": integration_order,
        "forecast_horizon": 5,
        "forecast_years": "2025-2029",
        "confidence_levels": [0.80, 0.95],
    }

    # Document decision about using unit root results
    result.add_decision(
        decision_id="D001",
        category="model_specification",
        decision=f"Use d={integration_order} from unit root tests",
        rationale="Module 2.1.1 determined nd_intl_migration is I(1), requiring first differencing",
        alternatives=["Let auto_arima determine d", "Use d=0 for levels"],
        evidence="ADF test on level: p=0.5564 (fail to reject unit root), ADF on diff: p=0.0025 (reject)",
        reversible=True,
    )

    # Run auto_arima for model selection
    print("\nRunning auto_arima for model selection...")
    try:
        # First try without seasonal
        auto_result = run_auto_arima(series, d=integration_order, seasonal=False)
        selected_order = auto_result["order"]
        print(f"  Selected ARIMA order: {selected_order}")
        print(f"  AIC: {auto_result['aic']:.2f}, BIC: {auto_result['bic']:.2f}")

        # Document model selection
        result.add_decision(
            decision_id="D002",
            category="model_selection",
            decision=f"Selected ARIMA{selected_order} via auto_arima",
            rationale=f"Minimum AIC={auto_result['aic']:.2f} among candidate models with d={integration_order}",
            alternatives=[
                "Manual model selection via ACF/PACF",
                "Grid search over all orders",
            ],
            evidence="auto_arima with stepwise search, max_p=3, max_q=3",
            reversible=True,
        )

        # Try with seasonal component to compare
        print("\n  Testing seasonal ARIMA...")
        try:
            auto_seasonal = run_auto_arima(
                series, d=integration_order, seasonal=True, m=1
            )
            print(f"  Seasonal model AIC: {auto_seasonal['aic']:.2f}")

            # Compare AIC
            if auto_seasonal["aic"] < auto_result["aic"] - 2:
                result.warnings.append(
                    f"Seasonal model has lower AIC ({auto_seasonal['aic']:.2f} vs {auto_result['aic']:.2f}), "
                    "but m=1 (annual) makes seasonal unlikely. Using non-seasonal model."
                )
        except Exception as e:
            print(f"  Seasonal model failed: {e}")

    except Exception as e:
        print(f"  auto_arima failed: {e}")
        # Fallback to simple ARIMA(1,1,0)
        selected_order = (1, integration_order, 0)
        result.warnings.append(
            f"auto_arima failed ({e}), using fallback ARIMA{selected_order}"
        )
        result.add_decision(
            decision_id="D002",
            category="model_selection",
            decision=f"Fallback to ARIMA{selected_order}",
            rationale="auto_arima failed, using simple specification",
            alternatives=["Manual grid search"],
            evidence=str(e),
            reversible=True,
        )

    # Estimate final model
    print(f"\nEstimating ARIMA{selected_order} model...")
    estimation_result = estimate_arima_model(series, selected_order)
    arima_results = estimation_result["results_object"]

    print(
        f"  Log-likelihood: {estimation_result['fit_statistics']['log_likelihood']:.2f}"
    )
    print(f"  AIC: {estimation_result['fit_statistics']['aic']:.2f}")
    print(f"  BIC: {estimation_result['fit_statistics']['bic']:.2f}")

    # Store model results
    result.results["model"] = f"ARIMA{selected_order}"
    result.results["model_selection"] = {
        "method": "auto_arima",
        "order": selected_order,
        "seasonal_order": None,
        "selection_criteria": {
            "aic": auto_result["aic"]
            if "auto_result" in dir()
            else estimation_result["fit_statistics"]["aic"],
            "bic": auto_result["bic"]
            if "auto_result" in dir()
            else estimation_result["fit_statistics"]["bic"],
        },
    }
    result.results["parameters"] = estimation_result["parameters"]
    result.results["fit_statistics"] = estimation_result["fit_statistics"]

    # Run diagnostics
    print("\nRunning model diagnostics...")
    diagnostics = run_diagnostics(arima_results, series)
    result.results["ljung_box"] = diagnostics["ljung_box"]
    result.results["residual_diagnostics"] = diagnostics["residual_diagnostics"]

    # Interpret Ljung-Box results
    lb_interpretation = []
    for lag, lb in diagnostics["ljung_box"].items():
        if "p_value" in lb:
            if lb["p_value"] > 0.05:
                lb_interpretation.append(
                    f"{lag}: p={lb['p_value']:.3f} (no autocorrelation)"
                )
            else:
                lb_interpretation.append(
                    f"{lag}: p={lb['p_value']:.3f} (autocorrelation detected)"
                )
                result.warnings.append(
                    f"Ljung-Box test at {lag} suggests residual autocorrelation (p={lb['p_value']:.3f})"
                )

    print(f"  Residual mean: {diagnostics['residual_diagnostics']['mean']:.2f}")
    print(f"  Residual std: {diagnostics['residual_diagnostics']['std']:.2f}")
    print(f"  Ljung-Box: {'; '.join(lb_interpretation[:2])}")

    # Generate forecasts
    print("\nGenerating 5-year forecasts (2025-2029)...")
    forecasts = generate_forecasts(arima_results, n_periods=5)
    result.results["forecasts"] = forecasts

    for f in forecasts:
        year = int(df["year"].max()) + f["horizon"]
        print(
            f"  {year}: {f['point']:.0f} (95% CI: [{f['ci_95'][0]:.0f}, {f['ci_95'][1]:.0f}])"
        )

    # Document short series limitation
    result.add_decision(
        decision_id="D003",
        category="methodology",
        decision="Proceed with ARIMA despite short series (n=15)",
        rationale="Sample size is small for reliable ARIMA, but this is all available data. "
        "Results should be interpreted with caution. Wide prediction intervals reflect uncertainty.",
        alternatives=["Use simpler models only", "Skip ARIMA entirely"],
        evidence="Standard recommendation is n>=30 for ARIMA",
        reversible=False,
    )

    result.warnings.append(
        "Short time series (n=15) limits model reliability. Prediction intervals are wide."
    )
    result.warnings.append(
        "2020 COVID year (value=30) may distort model estimates and forecasts."
    )

    # Generate visualizations
    print("\nGenerating figures...")

    # Series with fitted values
    fig1 = plot_series_with_fitted(df, arima_results, target_var)
    print(f"  Saved: {fig1}")

    # ACF/PACF of original series
    fig2 = plot_acf_pacf_original(series, target_var)
    print(f"  Saved: {fig2}")

    # ACF/PACF of residuals
    fig3 = plot_acf_pacf_residuals(arima_results.resid)
    print(f"  Saved: {fig3}")

    # Residual diagnostics
    fig4 = plot_residual_diagnostics(arima_results.resid)
    print(f"  Saved: {fig4}")

    # Forecast fan chart
    fig5 = plot_forecast_fan_chart(df, arima_results, forecasts, target_var)
    print(f"  Saved: {fig5}")

    # Store diagnostics metadata
    result.diagnostics = {
        "figures_generated": [
            "module_2_1_series_with_fitted.png",
            "module_2_1_acf_pacf_original.png",
            "module_2_1_acf_pacf_residuals.png",
            "module_2_1_residual_diagnostics.png",
            "module_2_1_forecast_fan_chart.png",
        ],
        "model_adequacy": {
            "ljung_box_interpretation": lb_interpretation,
            "residual_normality": "acceptable"
            if diagnostics["residual_diagnostics"]["shapiro_wilk"]["p_value"] > 0.05
            else "questionable",
        },
        "forecast_quality_note": "Wide intervals due to small sample and COVID volatility",
    }

    # Save forecasts to parquet for downstream use
    forecast_df = pd.DataFrame(
        [
            {
                "year": int(df["year"].max()) + f["horizon"],
                "horizon": f["horizon"],
                "forecast": f["point"],
                "se": f["se"],
                "ci_80_lower": f["ci_80"][0],
                "ci_80_upper": f["ci_80"][1],
                "ci_95_lower": f["ci_95"][0],
                "ci_95_upper": f["ci_95"][1],
            }
            for f in forecasts
        ]
    )
    forecast_path = RESULTS_DIR / "module_2_1_arima_forecasts.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    print(f"\nForecasts saved to: {forecast_path}")

    # Next steps
    result.next_steps = [
        "Use forecasts for scenario modeling (Module 9)",
        "Compare ARIMA forecasts with VAR model forecasts (Module 2.2)",
        "Incorporate structural breaks if identified (Module 2.1.2)",
        "Consider policy scenario adjustments based on immigration regime changes",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 2.1: ARIMA Time Series Modeling")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_2_1_arima_model.json")

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

        # Print model summary
        print("\n" + "-" * 60)
        print("Model Summary:")
        print(f"  Model: {result.results['model']}")
        print(f"  AIC: {result.results['fit_statistics']['aic']:.2f}")
        print(f"  BIC: {result.results['fit_statistics']['bic']:.2f}")
        print(
            f"  Log-likelihood: {result.results['fit_statistics']['log_likelihood']:.2f}"
        )

        print("\n5-Year Forecast Summary:")
        for f in result.results["forecasts"]:
            year = 2024 + f["horizon"]
            print(
                f"  {year}: {f['point']:.0f} persons (95% CI: [{f['ci_95'][0]:.0f}, {f['ci_95'][1]:.0f}])"
            )

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
