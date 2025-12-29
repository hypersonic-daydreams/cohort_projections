#!/usr/bin/env python3
"""
Module 1.1: Descriptive Statistics
===================================

Comprehensive descriptive statistics analysis for North Dakota immigration data.
Produces SPSS-style output with all required statistics and visualizations.

Usage:
    micromamba run -n cohort_proj python module_1_1_descriptive_statistics.py
"""

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.filters.hp_filter import hpfilter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    ):
        """Add a documented decision."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": True,
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


def setup_figure(figsize=(10, 8)):
    """Standard figure setup for all visualizations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    return fig, ax


def save_figure(fig, filepath_base: str, title: str, source_note: str):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.02,
        0.02,
        f"Source: {source_note}",
        fontsize=8,
        fontstyle="italic",
        transform=fig.transFigure,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save both formats
    png_path = f"{filepath_base}.png"
    pdf_path = f"{filepath_base}.pdf"
    fig.savefig(
        png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    fig.savefig(
        pdf_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def calculate_mode(series: pd.Series):
    """Calculate mode(s) of a series, return None if no repeated values."""
    mode_result = series.mode()
    if len(mode_result) == 0:
        return None
    elif len(mode_result) == len(series):
        return None  # All values unique
    else:
        return float(mode_result.iloc[0])


def bootstrap_ci(series: pd.Series, stat_func, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for a statistic."""
    np.random.seed(42)  # Reproducibility
    bootstrap_stats = []
    n = len(series)
    for _ in range(n_bootstrap):
        sample = series.sample(n=n, replace=True)
        try:
            bootstrap_stats.append(stat_func(sample))
        except Exception:
            continue

    if len(bootstrap_stats) < 100:
        return [None, None]

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    return [float(lower), float(upper)]


def identify_outliers(series: pd.Series):
    """Identify mild and extreme outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # Mild outliers: 1.5 * IQR
    mild_lower = q1 - 1.5 * iqr
    mild_upper = q3 + 1.5 * iqr

    # Extreme outliers: 3 * IQR
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr

    mild_outliers = series[(series < mild_lower) | (series > mild_upper)]
    extreme_outliers = series[(series < extreme_lower) | (series > extreme_upper)]

    # Remove extreme from mild
    mild_only = mild_outliers[~mild_outliers.index.isin(extreme_outliers.index)]

    return {
        "mild": [
            {"index": int(idx), "value": float(v)} for idx, v in mild_only.items()
        ],
        "extreme": [
            {"index": int(idx), "value": float(v)}
            for idx, v in extreme_outliers.items()
        ],
        "iqr_bounds": {
            "mild_lower": float(mild_lower),
            "mild_upper": float(mild_upper),
            "extreme_lower": float(extreme_lower),
            "extreme_upper": float(extreme_upper),
        },
    }


def compute_descriptive_stats(series: pd.Series, variable_name: str) -> dict:
    """Compute comprehensive SPSS-style descriptive statistics."""
    n = len(series)
    mean = series.mean()
    se_mean = series.sem()
    median = series.median()
    mode = calculate_mode(series)
    sd = series.std()
    variance = series.var()

    # Skewness and kurtosis with standard errors
    skewness = series.skew()
    kurtosis = series.kurtosis()
    # SE for skewness: sqrt(6/n) for normal distribution
    se_skewness = np.sqrt(6 / n) if n > 0 else None
    # SE for kurtosis: sqrt(24/n) for normal distribution
    se_kurtosis = np.sqrt(24 / n) if n > 0 else None

    # Range statistics
    range_val = series.max() - series.min()
    min_val = series.min()
    max_val = series.max()
    iqr = series.quantile(0.75) - series.quantile(0.25)

    # Coefficient of variation
    cv = sd / mean if mean != 0 else None

    # Percentiles
    percentiles = {
        "5": float(series.quantile(0.05)),
        "10": float(series.quantile(0.10)),
        "25": float(series.quantile(0.25)),
        "50": float(series.quantile(0.50)),
        "75": float(series.quantile(0.75)),
        "90": float(series.quantile(0.90)),
        "95": float(series.quantile(0.95)),
    }

    # Confidence intervals
    ci_mean_95 = [
        float(mean - 1.96 * se_mean),
        float(mean + 1.96 * se_mean),
    ]
    ci_median_95 = bootstrap_ci(series, np.median, n_bootstrap=1000, ci=0.95)

    # Normality tests
    # Shapiro-Wilk (for n < 50)
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(series)
    else:
        shapiro_stat, shapiro_p = None, None

    # Kolmogorov-Smirnov test against normal distribution
    if n >= 3:
        ks_stat, ks_p = stats.kstest(series, "norm", args=(mean, sd))
    else:
        ks_stat, ks_p = None, None

    # Outliers
    outliers = identify_outliers(series)

    return {
        "variable": variable_name,
        "n": int(n),
        "mean": float(mean),
        "se_mean": float(se_mean),
        "ci_mean_95": ci_mean_95,
        "median": float(median),
        "ci_median_95": ci_median_95,
        "mode": mode,
        "sd": float(sd),
        "variance": float(variance),
        "skewness": float(skewness),
        "se_skewness": float(se_skewness) if se_skewness else None,
        "kurtosis": float(kurtosis),
        "se_kurtosis": float(se_kurtosis) if se_kurtosis else None,
        "range": float(range_val),
        "min": float(min_val),
        "max": float(max_val),
        "iqr": float(iqr),
        "cv": float(cv) if cv is not None else None,
        "percentiles": percentiles,
        "normality": {
            "shapiro_wilk": {
                "statistic": float(shapiro_stat) if shapiro_stat else None,
                "p_value": float(shapiro_p) if shapiro_p else None,
            },
            "kolmogorov_smirnov": {
                "statistic": float(ks_stat) if ks_stat else None,
                "p_value": float(ks_p) if ks_p else None,
            },
        },
        "outliers": outliers,
    }


def hp_decompose(series: pd.Series, lamb: float = 6.25) -> dict:
    """
    Hodrick-Prescott filter decomposition.

    Args:
        series: Time series data
        lamb: Smoothing parameter (6.25 for annual data)

    Returns:
        Dictionary with trend and cycle components
    """
    cycle, trend = hpfilter(series, lamb=lamb)
    return {
        "trend": trend.tolist(),
        "cycle": cycle.tolist(),
        "lambda": lamb,
        "trend_stats": {
            "mean": float(trend.mean()),
            "sd": float(trend.std()),
            "min": float(trend.min()),
            "max": float(trend.max()),
        },
        "cycle_stats": {
            "mean": float(cycle.mean()),
            "sd": float(cycle.std()),
            "min": float(cycle.min()),
            "max": float(cycle.max()),
        },
    }


def compute_first_differences(series: pd.Series, variable_name: str) -> dict:
    """Compute first differences and log-differences."""
    diff = series.diff().dropna()

    # Log-differences (requires positive values)
    if (series > 0).all():
        log_diff = np.log(series).diff().dropna()
        log_diff_stats = compute_descriptive_stats(
            log_diff, f"{variable_name}_log_diff"
        )
    else:
        log_diff = None
        log_diff_stats = None

    return {
        "variable": variable_name,
        "first_difference": {
            "values": diff.tolist(),
            "stats": compute_descriptive_stats(diff, f"{variable_name}_diff"),
        },
        "log_difference": {
            "values": log_diff.tolist() if log_diff is not None else None,
            "stats": log_diff_stats,
        },
    }


def create_time_series_plot(df: pd.DataFrame, result: ModuleResult):
    """Create time series plot for ND share."""
    fig, ax = setup_figure(figsize=(12, 6))

    # Plot ND share of US international migration
    ax.plot(
        df["year"],
        df["nd_share_of_us_intl_pct"],
        color=COLORS["primary"],
        linewidth=2,
        marker="o",
        markersize=6,
        label="ND Share of US Intl Migration",
    )

    # Add horizontal line for mean
    mean_val = df["nd_share_of_us_intl_pct"].mean()
    ax.axhline(
        y=mean_val,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean ({mean_val:.3f}%)",
    )

    # Mark 2020 COVID year
    covid_idx = df[df["year"] == 2020].index
    if len(covid_idx) > 0:
        ax.scatter(
            [2020],
            df.loc[covid_idx, "nd_share_of_us_intl_pct"],
            color=COLORS["secondary"],
            s=100,
            zorder=5,
            label="2020 (COVID)",
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of US International Migration (%)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(2009.5, 2024.5)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_1_1_time_series_nd_share"),
        "North Dakota Share of US International Migration (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def create_histogram_plot(series: pd.Series, variable_name: str, result: ModuleResult):
    """Create histogram with normal curve overlay."""
    fig, ax = setup_figure(figsize=(10, 7))

    # Histogram
    n, bins, patches = ax.hist(
        series,
        bins="auto",
        density=True,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="white",
        label="Observed",
    )

    # Normal curve overlay
    mu, sigma = series.mean(), series.std()
    x = np.linspace(series.min() - sigma, series.max() + sigma, 100)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax.plot(
        x,
        normal_curve,
        color=COLORS["secondary"],
        linewidth=2,
        label=f"Normal (mu={mu:.3f}, sigma={sigma:.3f})",
    )

    # Add vertical lines for mean and median
    ax.axvline(
        mu,
        color=COLORS["tertiary"],
        linewidth=2,
        linestyle="-",
        label=f"Mean: {mu:.3f}",
    )
    ax.axvline(
        series.median(),
        color=COLORS["quaternary"],
        linewidth=2,
        linestyle="--",
        label=f"Median: {series.median():.3f}",
    )

    ax.set_xlabel(variable_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_1_1_histogram_nd_share"),
        f"Distribution of {variable_name.replace('_', ' ').title()}",
        "Census Bureau Population Estimates Program",
    )


def create_qq_plot(series: pd.Series, variable_name: str, result: ModuleResult):
    """Create Q-Q plot."""
    fig, ax = setup_figure(figsize=(8, 8))

    # Q-Q plot
    stats.probplot(series, dist="norm", plot=ax)

    # Style the plot
    ax.get_lines()[0].set_markerfacecolor(COLORS["primary"])
    ax.get_lines()[0].set_markeredgecolor(COLORS["primary"])
    ax.get_lines()[0].set_markersize(8)
    ax.get_lines()[1].set_color(COLORS["secondary"])
    ax.get_lines()[1].set_linewidth(2)

    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_1_1_qq_plot_nd_share"),
        f"Q-Q Plot: {variable_name.replace('_', ' ').title()}",
        "Census Bureau Population Estimates Program",
    )


def create_boxplot(
    series: pd.Series, variable_name: str, years: pd.Series, result: ModuleResult
):
    """Create box plot with outlier identification."""
    fig, ax = setup_figure(figsize=(8, 8))

    # Box plot
    ax.boxplot(
        series,
        vert=True,
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": COLORS["primary"], "alpha": 0.7},
        medianprops={"color": COLORS["secondary"], "linewidth": 2},
        whiskerprops={"color": COLORS["primary"], "linewidth": 1.5},
        capprops={"color": COLORS["primary"], "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markerfacecolor": COLORS["secondary"],
            "markersize": 10,
            "markeredgecolor": COLORS["secondary"],
        },
    )

    # Add individual points
    jittered_x = np.random.normal(1, 0.04, len(series))
    ax.scatter(jittered_x, series, alpha=0.5, color=COLORS["tertiary"], s=50, zorder=3)

    # Label outliers with year
    outliers = identify_outliers(series)
    for outlier in outliers["mild"] + outliers["extreme"]:
        idx = outlier["index"]
        year = years.iloc[idx]
        val = outlier["value"]
        ax.annotate(
            str(year),
            xy=(1, val),
            xytext=(1.2, val),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": COLORS["neutral"]},
        )

    ax.set_ylabel(variable_name.replace("_", " ").title(), fontsize=12)
    ax.set_xticklabels(["ND Share of US Intl Migration"])

    save_figure(
        fig,
        str(FIGURES_DIR / "module_1_1_boxplot_nd_share"),
        f"Box Plot: {variable_name.replace('_', ' ').title()}",
        "Census Bureau Population Estimates Program",
    )


def create_trend_decomposition_plot(
    df: pd.DataFrame, hp_result: dict, result: ModuleResult
):
    """Create Hodrick-Prescott trend decomposition plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    years = df["year"].values
    original = df["nd_share_of_us_intl_pct"].values
    trend = np.array(hp_result["trend"])
    cycle = np.array(hp_result["cycle"])

    # Original series
    axes[0].plot(years, original, color=COLORS["primary"], linewidth=2, marker="o")
    axes[0].set_ylabel("Original", fontsize=12)
    axes[0].set_title("ND Share of US International Migration", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Trend component
    axes[1].plot(years, trend, color=COLORS["tertiary"], linewidth=2)
    axes[1].set_ylabel("Trend", fontsize=12)
    axes[1].set_title(f"HP Trend (lambda={hp_result['lambda']})", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Cycle component
    axes[2].plot(years, cycle, color=COLORS["secondary"], linewidth=2)
    axes[2].axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    axes[2].fill_between(
        years, cycle, 0, where=(cycle >= 0), color=COLORS["tertiary"], alpha=0.3
    )
    axes[2].fill_between(
        years, cycle, 0, where=(cycle < 0), color=COLORS["secondary"], alpha=0.3
    )
    axes[2].set_ylabel("Cycle", fontsize=12)
    axes[2].set_xlabel("Year", fontsize=12)
    axes[2].set_title("HP Cycle (Deviations from Trend)", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure(
        fig,
        str(FIGURES_DIR / "module_1_1_trend_decomposition"),
        "Hodrick-Prescott Trend-Cycle Decomposition",
        "Census Bureau Population Estimates Program",
    )


def run_analysis() -> ModuleResult:
    """
    Main analysis function for Module 1.1: Descriptive Statistics.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(
        module_id="1.1",
        analysis_name="descriptive_statistics",
    )

    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    # Load required data
    nd_migration = load_data("nd_migration_summary.csv")
    result.input_files.append("nd_migration_summary.csv")
    print(f"  Loaded nd_migration_summary.csv: {len(nd_migration)} rows")

    # Record parameters
    result.parameters = {
        "hp_filter_lambda": 6.25,
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95,
        "outlier_method": "IQR",
        "random_seed": 42,
    }

    # Key variables to analyze
    key_variables = [
        "nd_intl_migration",
        "us_intl_migration",
        "nd_share_of_us_intl_pct",
        "nd_share_of_us_pop_pct",
    ]

    print("\n" + "=" * 60)
    print("Computing descriptive statistics...")
    print("=" * 60)

    # Compute descriptive statistics for each variable
    descriptive_stats = {}
    for var in key_variables:
        if var in nd_migration.columns:
            print(f"  Processing: {var}")
            descriptive_stats[var] = compute_descriptive_stats(nd_migration[var], var)

    result.results["descriptive_statistics"] = descriptive_stats

    # Document COVID-2020 handling decision
    covid_value = nd_migration.loc[
        nd_migration["year"] == 2020, "nd_intl_migration"
    ].values[0]
    mean_val = nd_migration["nd_intl_migration"].mean()
    sd_val = nd_migration["nd_intl_migration"].std()
    z_score = (covid_value - mean_val) / sd_val

    result.add_decision(
        decision_id="D001",
        category="data_handling",
        decision="Retained year 2020 in descriptive statistics analysis",
        rationale="Descriptive analysis should show full data including anomalies. "
        "Exclusion decisions deferred to time series modeling modules.",
        alternatives=[
            "Exclude 2020",
            "Winsorize to 1st percentile",
            "Replace with interpolation",
        ],
        evidence=f"2020 value ({covid_value}) is {z_score:.1f} standard deviations below mean",
    )
    result.warnings.append(
        f"Year 2020 (COVID) shows anomalous international migration value ({covid_value}), "
        f"which is {z_score:.1f} SD below mean"
    )

    print("\n" + "=" * 60)
    print("Computing trend decomposition (HP filter)...")
    print("=" * 60)

    # HP Filter decomposition
    hp_result = hp_decompose(nd_migration["nd_share_of_us_intl_pct"], lamb=6.25)
    result.results["hp_decomposition"] = {
        "variable": "nd_share_of_us_intl_pct",
        "years": nd_migration["year"].tolist(),
        **hp_result,
    }

    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Used HP filter lambda=6.25 for annual data",
        rationale="Standard recommendation for annual data (Ravn and Uhlig, 2002). "
        "Lambda=6.25 balances trend smoothness with cycle responsiveness.",
        alternatives=[
            "lambda=100 (quarterly)",
            "lambda=1600 (monthly)",
            "Baxter-King filter",
        ],
        evidence="Ravn, M.O. and Uhlig, H. (2002). On adjusting the Hodrick-Prescott filter.",
    )
    print("  HP decomposition complete")

    print("\n" + "=" * 60)
    print("Computing first differences and log-differences...")
    print("=" * 60)

    # First differences and log-differences
    differences_results = {}
    for var in ["nd_share_of_us_intl_pct", "nd_intl_migration"]:
        if var in nd_migration.columns:
            print(f"  Processing differences: {var}")
            differences_results[var] = compute_first_differences(nd_migration[var], var)

    result.results["differences"] = differences_results

    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)

    # Create visualizations
    print("\n1. Time series plot...")
    create_time_series_plot(nd_migration, result)

    print("\n2. Histogram with normal curve...")
    create_histogram_plot(
        nd_migration["nd_share_of_us_intl_pct"], "nd_share_of_us_intl_pct", result
    )

    print("\n3. Q-Q plot...")
    create_qq_plot(
        nd_migration["nd_share_of_us_intl_pct"], "nd_share_of_us_intl_pct", result
    )

    print("\n4. Box plot...")
    create_boxplot(
        nd_migration["nd_share_of_us_intl_pct"],
        "nd_share_of_us_intl_pct",
        nd_migration["year"],
        result,
    )

    print("\n5. Trend decomposition plot...")
    create_trend_decomposition_plot(nd_migration, hp_result, result)

    # Store diagnostics
    result.diagnostics = {
        "data_quality": {
            "n_observations": len(nd_migration),
            "year_range": [
                int(nd_migration["year"].min()),
                int(nd_migration["year"].max()),
            ],
            "missing_values": nd_migration[key_variables].isnull().sum().to_dict(),
            "complete_cases": int(nd_migration[key_variables].dropna().shape[0]),
        },
        "normality_assessment": {
            "nd_share_of_us_intl_pct": {
                "shapiro_wilk_p": descriptive_stats["nd_share_of_us_intl_pct"][
                    "normality"
                ]["shapiro_wilk"]["p_value"],
                "interpretation": (
                    "Cannot reject normality (p > 0.05)"
                    if descriptive_stats["nd_share_of_us_intl_pct"]["normality"][
                        "shapiro_wilk"
                    ]["p_value"]
                    > 0.05
                    else "Reject normality at 5% level"
                ),
            }
        },
        "outlier_summary": {
            "nd_share_of_us_intl_pct": {
                "n_mild": len(
                    descriptive_stats["nd_share_of_us_intl_pct"]["outliers"]["mild"]
                ),
                "n_extreme": len(
                    descriptive_stats["nd_share_of_us_intl_pct"]["outliers"]["extreme"]
                ),
            }
        },
    }

    # Next steps
    result.next_steps = [
        "Module 2.1.1: Conduct unit root tests to determine stationarity",
        "Module 1.2: Analyze geographic concentration patterns",
        "Module 2.1.2: Test for structural breaks (2015 Bakken, 2017 Travel Ban, 2020 COVID)",
        "Consider special handling of 2020 in time series models",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 1.1: Descriptive Statistics")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()

        # Save main results
        output_file = result.save("module_1_1_summary_statistics.json")

        # Save trend decomposition separately
        trend_output = RESULTS_DIR / "module_1_1_trend_decomposition.json"
        with open(trend_output, "w") as f:
            json.dump(result.results["hp_decomposition"], f, indent=2, default=str)
        print(f"Trend decomposition saved to: {trend_output}")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
        print("\nOutputs:")
        print(f"  - {output_file}")
        print(f"  - {trend_output}")
        print(f"\nFigures saved to: {FIGURES_DIR}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")

        if result.decisions:
            print(f"\nDecisions logged ({len(result.decisions)}):")
            for d in result.decisions:
                print(f"  - {d['decision_id']}: {d['decision']}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
