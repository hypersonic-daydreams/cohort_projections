#!/usr/bin/env python3
"""
Module 2.1.2: Structural Break Detection
=========================================

Performs structural break detection on ND international migration time series
to identify significant regime changes corresponding to policy events.

Methods Implemented:
- Bai-Perron test (via ruptures Pelt algorithm with BIC)
- CUSUM test for parameter stability
- Chow test at known break dates (2017 Travel Ban, 2020 COVID)

Policy Events Tested:
- 2017: Travel ban executive orders
- 2020: COVID-19 pandemic
- 2021-2022: Post-COVID surge

Usage:
    micromamba run -n cohort_proj python module_2_1_2_structural_breaks.py
"""

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

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
    "break_line": "#E31A1C",  # Red for break lines
}

# Known policy event dates
POLICY_EVENTS = {
    2017: {
        "name": "Travel Ban Executive Orders",
        "description": "Executive Order 13769 (Jan 2017), EO 13780 (Mar 2017)",
        "expected_effect": "negative",
    },
    2020: {
        "name": "COVID-19 Pandemic",
        "description": "March 2020 travel restrictions, embassy closures",
        "expected_effect": "negative",
    },
    2021: {
        "name": "Post-COVID Recovery",
        "description": "Vaccination rollout, travel reopening",
        "expected_effect": "positive",
    },
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


def bai_perron_test(series, years, min_size=2, pen_value=None):
    """
    Perform Bai-Perron style multiple breakpoint detection using PELT algorithm.

    Parameters:
    -----------
    series : array-like
        Time series data
    years : array-like
        Corresponding years for the series
    min_size : int
        Minimum segment length between breaks
    pen_value : float, optional
        Penalty value for BIC. If None, will be estimated.

    Returns:
    --------
    dict with detected break points, segment statistics, and BIC values
    """
    signal = np.array(series).reshape(-1, 1)
    n = len(signal)

    # Use Pelt algorithm with rbf (radial basis function) kernel
    # This is robust to different types of changes in the series
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
    algo.fit(signal)

    # Calculate BIC-optimal penalty
    # BIC penalty = log(n) * k where k is number of parameters per segment
    # For mean shift detection, k = 1 (mean of each segment)
    if pen_value is None:
        # Standard BIC penalty
        pen_value = np.log(n) * 2  # 2 parameters per segment (mean + variance)

    # Detect breakpoints
    breakpoints = algo.predict(pen=pen_value)

    # Remove the last breakpoint (end of series)
    if breakpoints and breakpoints[-1] == n:
        breakpoints = breakpoints[:-1]

    # Calculate segment statistics
    segments = []
    start = 0
    for bp in breakpoints + [n]:  # Include end of series
        segment_data = series[start:bp]
        segment_years = years[start:bp]
        segments.append(
            {
                "start_index": int(start),
                "end_index": int(bp - 1),
                "start_year": int(segment_years.iloc[0])
                if hasattr(segment_years, "iloc")
                else int(segment_years[0]),
                "end_year": int(segment_years.iloc[-1])
                if hasattr(segment_years, "iloc")
                else int(segment_years[-1]),
                "n_observations": int(bp - start),
                "mean": float(np.mean(segment_data)),
                "std": float(np.std(segment_data)),
                "min": float(np.min(segment_data)),
                "max": float(np.max(segment_data)),
            }
        )
        start = bp

    # Calculate regime shift magnitudes
    regime_shifts = []
    for i in range(len(segments) - 1):
        shift = {
            "break_index": breakpoints[i] if i < len(breakpoints) else None,
            "break_year": int(years.iloc[breakpoints[i]])
            if i < len(breakpoints)
            else None,
            "pre_regime_mean": segments[i]["mean"],
            "post_regime_mean": segments[i + 1]["mean"],
            "absolute_shift": segments[i + 1]["mean"] - segments[i]["mean"],
            "percent_shift": (
                (segments[i + 1]["mean"] - segments[i]["mean"])
                / abs(segments[i]["mean"])
                * 100
            )
            if segments[i]["mean"] != 0
            else None,
        }
        regime_shifts.append(shift)

    # Calculate BIC for model selection
    # BIC = n * ln(RSS/n) + k * ln(n)
    residuals = []
    for seg in segments:
        seg_data = series[seg["start_index"] : seg["end_index"] + 1]
        residuals.extend((seg_data - seg["mean"]).tolist())
    rss = np.sum(np.array(residuals) ** 2)
    k = 2 * len(segments)  # mean and variance per segment
    bic = n * np.log(rss / n + 1e-10) + k * np.log(n)

    # Test alternative models with different numbers of breaks
    bic_alternatives = {}
    for n_breaks in range(min(5, n // min_size)):
        if n_breaks == 0:
            # No break model
            rss_alt = np.sum((series - np.mean(series)) ** 2)
            k_alt = 2
        else:
            try:
                algo_alt = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
                algo_alt.fit(signal)
                # Use higher penalty to get fewer breaks
                pen_alt = pen_value * (n_breaks + 1)
                bp_alt = algo_alt.predict(pen=pen_alt)
                if bp_alt and bp_alt[-1] == n:
                    bp_alt = bp_alt[:-1]

                rss_alt = 0
                start = 0
                for bp in bp_alt + [n]:
                    seg_data = series[start:bp]
                    rss_alt += np.sum((seg_data - np.mean(seg_data)) ** 2)
                    start = bp
                k_alt = 2 * (len(bp_alt) + 1)
            except Exception:
                continue

        bic_alt = n * np.log(rss_alt / n + 1e-10) + k_alt * np.log(n)
        bic_alternatives[f"{n_breaks}_breaks"] = float(bic_alt)

    return {
        "method": "Bai-Perron (PELT with RBF kernel)",
        "n_observations": n,
        "penalty_value": float(pen_value),
        "n_breaks_detected": len(breakpoints),
        "break_indices": [int(b) for b in breakpoints],
        "break_years": [int(years.iloc[b]) for b in breakpoints] if breakpoints else [],
        "segments": segments,
        "regime_shifts": regime_shifts,
        "model_fit": {
            "bic": float(bic),
            "rss": float(rss),
            "bic_alternatives": bic_alternatives,
        },
        "interpretation": f"Detected {len(breakpoints)} structural break(s) in the series",
    }


def cusum_test(series, years):
    """
    Perform CUSUM test for parameter stability.

    The CUSUM test checks if regression parameters are stable over time.
    It calculates cumulative sums of recursive residuals.

    Parameters:
    -----------
    series : array-like
        Time series data
    years : array-like
        Corresponding years

    Returns:
    --------
    dict with CUSUM statistics, bounds, and stability conclusion
    """
    y = np.array(series)
    n = len(y)

    # Create simple time trend regression
    X = np.column_stack([np.ones(n), np.arange(n)])

    # OLS estimation
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    sigma = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Calculate recursive residuals
    recursive_residuals = []
    for t in range(3, n + 1):  # Need at least 3 observations to start
        X_t = X[:t]
        y_t = y[:t]
        beta_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
        y_pred = X_t[-1] @ beta_t
        e_t = y[t - 1] - y_pred

        # Variance of recursive residual
        f_t = 1 + X_t[-1] @ np.linalg.inv(X_t.T @ X_t) @ X_t[-1].T
        w_t = e_t / (sigma * np.sqrt(f_t))
        recursive_residuals.append(w_t)

    recursive_residuals = np.array(recursive_residuals)

    # Calculate CUSUM
    cusum = np.cumsum(recursive_residuals)

    # Calculate 5% significance bounds
    # Bounds: +/- a * sqrt(n - k) where a depends on significance level
    # For 5% level, a = 0.948
    n_rec = len(recursive_residuals)
    time_index = np.arange(1, n_rec + 1)

    # Brown, Durbin, Evans (1975) bounds
    a = 0.948  # 5% significance level
    upper_bound = a * np.sqrt(n_rec) + 2 * a * time_index / np.sqrt(n_rec)
    lower_bound = -upper_bound

    # Check for crossing
    crossings = []
    for i, (c, ub, lb) in enumerate(zip(cusum, upper_bound, lower_bound, strict=False)):
        if c > ub or c < lb:
            year_idx = i + 3  # Account for initial observations
            crossings.append(
                {
                    "index": year_idx,
                    "year": int(years.iloc[year_idx])
                    if year_idx < len(years)
                    else None,
                    "cusum_value": float(c),
                    "upper_bound": float(ub),
                    "lower_bound": float(lb),
                }
            )

    # Calculate max deviation from bounds (for significance assessment)
    max_cusum = np.max(np.abs(cusum))
    max_bound = np.max(upper_bound)
    stability_ratio = max_cusum / max_bound

    # Conclusion
    if len(crossings) > 0:
        conclusion = f"CUSUM crosses bounds at {len(crossings)} point(s) - parameter instability detected"
        stable = False
    else:
        conclusion = "CUSUM stays within bounds - parameters appear stable"
        stable = True

    return {
        "method": "CUSUM Test (Brown, Durbin, Evans 1975)",
        "null_hypothesis": "Parameters are stable over time",
        "significance_level": 0.05,
        "n_observations": n,
        "n_recursive_residuals": n_rec,
        "cusum_values": cusum.tolist(),
        "upper_bounds": upper_bound.tolist(),
        "lower_bounds": lower_bound.tolist(),
        "years_for_cusum": [
            int(years.iloc[i + 3]) for i in range(n_rec) if i + 3 < len(years)
        ],
        "max_cusum": float(max_cusum),
        "max_bound": float(max_bound),
        "stability_ratio": float(stability_ratio),
        "crossings": crossings,
        "stable": stable,
        "conclusion": conclusion,
    }


def chow_test(series, years, break_year):
    """
    Perform Chow test for structural break at known date.

    Tests whether regression parameters differ before and after break point.

    Parameters:
    -----------
    series : array-like
        Time series data
    years : array-like
        Corresponding years
    break_year : int
        Year to test for structural break

    Returns:
    --------
    dict with F-statistic, p-value, and break significance
    """
    y = np.array(series)
    t = np.arange(len(y))
    years_arr = np.array(years)

    # Find break index
    break_indices = np.where(years_arr == break_year)[0]
    if len(break_indices) == 0:
        # If exact year not found, find closest
        break_idx = np.argmin(np.abs(years_arr - break_year))
    else:
        break_idx = break_indices[0]

    n = len(y)
    k = 2  # Number of parameters (constant + trend)

    # Check minimum sample sizes
    n1 = break_idx
    n2 = n - break_idx

    if n1 < k or n2 < k:
        return {
            "method": "Chow Test",
            "break_year": int(break_year),
            "error": f"Insufficient observations: n1={n1}, n2={n2}, need at least k={k} in each subsample",
            "significant": None,
            "conclusion": "Cannot perform test - insufficient observations in one or both subsamples",
        }

    # Pooled regression (restricted model)
    X = np.column_stack([np.ones(n), t])
    beta_pooled = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_pooled = np.sum((y - X @ beta_pooled) ** 2)

    # Pre-break regression
    X1 = np.column_stack([np.ones(n1), t[:n1]])
    y1 = y[:n1]
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    rss1 = np.sum((y1 - X1 @ beta1) ** 2)

    # Post-break regression
    X2 = np.column_stack([np.ones(n2), t[n1:] - t[n1]])  # Reset time index
    y2 = y[n1:]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    rss2 = np.sum((y2 - X2 @ beta2) ** 2)

    # Unrestricted RSS
    rss_unrestricted = rss1 + rss2

    # F-statistic
    numerator = (rss_pooled - rss_unrestricted) / k
    denominator = rss_unrestricted / (n - 2 * k)

    if denominator == 0:
        return {
            "method": "Chow Test",
            "break_year": int(break_year),
            "error": "Zero denominator in F-statistic calculation",
            "significant": None,
            "conclusion": "Cannot perform test - degenerate case",
        }

    f_stat = numerator / denominator
    df1 = k
    df2 = n - 2 * k

    # P-value
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Critical values
    critical_values = {
        "10%": float(stats.f.ppf(0.90, df1, df2)),
        "5%": float(stats.f.ppf(0.95, df1, df2)),
        "1%": float(stats.f.ppf(0.99, df1, df2)),
    }

    significant = p_value < 0.05

    # Calculate regime statistics for interpretation
    pre_mean = float(np.mean(y1))
    post_mean = float(np.mean(y2))
    shift = post_mean - pre_mean
    shift_pct = (shift / abs(pre_mean) * 100) if pre_mean != 0 else None

    if significant:
        direction = "increase" if shift > 0 else "decrease"
        conclusion = (
            f"Significant structural break at {break_year} (p={p_value:.4f}). "
            f"Mean {direction}d from {pre_mean:.1f} to {post_mean:.1f} "
            f"({shift_pct:+.1f}% change)"
        )
    else:
        conclusion = (
            f"No significant structural break at {break_year} (p={p_value:.4f})"
        )

    return {
        "method": "Chow Test",
        "null_hypothesis": "No structural break at specified date",
        "break_year": int(break_year),
        "break_index": int(break_idx),
        "f_statistic": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "p_value": float(p_value),
        "critical_values": critical_values,
        "significant_at_5pct": significant,
        "rss_pooled": float(rss_pooled),
        "rss_unrestricted": float(rss_unrestricted),
        "pre_break": {
            "n_obs": int(n1),
            "mean": pre_mean,
            "std": float(np.std(y1)),
            "coefficients": {"constant": float(beta1[0]), "trend": float(beta1[1])},
        },
        "post_break": {
            "n_obs": int(n2),
            "mean": post_mean,
            "std": float(np.std(y2)),
            "coefficients": {"constant": float(beta2[0]), "trend": float(beta2[1])},
        },
        "regime_shift": {"absolute_change": float(shift), "percent_change": shift_pct},
        "conclusion": conclusion,
    }


def plot_structural_breaks(df, series_name, break_results, year_col="year"):
    """Plot time series with detected structural breaks marked."""
    fig, ax = plt.subplots(figsize=(12, 7))

    years = df[year_col]
    series = df[series_name]

    # Plot original series
    ax.plot(
        years,
        series,
        marker="o",
        linewidth=2,
        markersize=8,
        color=COLORS["primary"],
        label="ND International Migration",
        zorder=3,
    )

    # Mark detected breaks from Bai-Perron
    if break_results.get("bai_perron", {}).get("break_years"):
        for i, break_year in enumerate(break_results["bai_perron"]["break_years"]):
            ax.axvline(
                x=break_year,
                color=COLORS["break_line"],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Detected Break" if i == 0 else "",
            )
            # Add annotation
            y_pos = series.max() * 0.95
            ax.annotate(
                f"Break: {break_year}",
                (break_year, y_pos),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=10,
                color=COLORS["break_line"],
                fontweight="bold",
            )

    # Mark known policy events
    policy_colors = {
        "2017": COLORS["secondary"],
        "2020": COLORS["tertiary"],
        "2021": COLORS["quaternary"],
    }
    for event_year, event_info in POLICY_EVENTS.items():
        if event_year in years.values:
            color = policy_colors.get(str(event_year), COLORS["neutral"])
            ax.axvline(
                x=event_year, color=color, linestyle=":", linewidth=1.5, alpha=0.7
            )
            y_pos = series.min() + (series.max() - series.min()) * 0.1
            ax.annotate(
                event_info["name"],
                (event_year, y_pos),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=8,
                color=color,
                rotation=90,
                va="bottom",
            )

    # Add segment means if available
    if break_results.get("bai_perron", {}).get("segments"):
        for seg in break_results["bai_perron"]["segments"]:
            years[(years >= seg["start_year"]) & (years <= seg["end_year"])]
            ax.hlines(
                y=seg["mean"],
                xmin=seg["start_year"],
                xmax=seg["end_year"],
                colors=COLORS["neutral"],
                linestyles="-",
                linewidth=1.5,
                alpha=0.5,
            )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("ND International Migration", fontsize=12)
    ax.set_title(
        "ND International Migration with Structural Breaks",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add legend for policy events
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color=COLORS["break_line"], linestyle="--", lw=2),
        Line2D([0], [0], color=COLORS["secondary"], linestyle=":", lw=1.5),
        Line2D([0], [0], color=COLORS["tertiary"], linestyle=":", lw=1.5),
    ]
    ax.legend(
        custom_lines,
        ["Detected Break", "Travel Ban (2017)", "COVID (2020)"],
        loc="upper left",
        fontsize=9,
    )

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_2_structural_breaks")
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


def plot_cusum(cusum_results, df, year_col="year"):
    """Plot CUSUM statistic with confidence bounds."""
    fig, ax = plt.subplots(figsize=(12, 7))

    years = cusum_results.get("years_for_cusum", [])
    cusum = cusum_results.get("cusum_values", [])
    upper = cusum_results.get("upper_bounds", [])
    lower = cusum_results.get("lower_bounds", [])

    if not years or not cusum:
        print("  Warning: No CUSUM data to plot")
        return None

    # Ensure same length
    n = min(len(years), len(cusum), len(upper), len(lower))
    years = years[:n]
    cusum = cusum[:n]
    upper = upper[:n]
    lower = lower[:n]

    # Plot CUSUM
    ax.plot(
        years,
        cusum,
        marker="o",
        linewidth=2,
        markersize=6,
        color=COLORS["primary"],
        label="CUSUM Statistic",
        zorder=3,
    )

    # Plot bounds
    ax.fill_between(
        years,
        lower,
        upper,
        alpha=0.2,
        color=COLORS["tertiary"],
        label="95% Confidence Bounds",
    )
    ax.plot(years, upper, linestyle="--", color=COLORS["tertiary"], linewidth=1.5)
    ax.plot(years, lower, linestyle="--", color=COLORS["tertiary"], linewidth=1.5)

    # Mark crossings
    crossings = cusum_results.get("crossings", [])
    for cross in crossings:
        if cross.get("year"):
            ax.axvline(
                x=cross["year"],
                color=COLORS["break_line"],
                linestyle=":",
                linewidth=2,
                alpha=0.8,
            )
            ax.annotate(
                f"Crossing: {cross['year']}",
                (cross["year"], cross["cusum_value"]),
                textcoords="offset points",
                xytext=(5, 10),
                fontsize=9,
                color=COLORS["break_line"],
            )

    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=1, alpha=0.5)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("CUSUM Statistic", fontsize=12)
    ax.set_title("CUSUM Test for Parameter Stability", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add stability conclusion
    cusum_results.get("conclusion", "")
    stability_text = "Stable" if cusum_results.get("stable") else "Unstable"
    ax.text(
        0.98,
        0.02,
        f"Conclusion: {stability_text}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_2_cusum_test")
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


def plot_segmented_regression(df, series_name, break_results, year_col="year"):
    """Plot segmented regression showing regime changes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    years = df[year_col].values
    series = df[series_name].values

    # Plot original data
    ax.scatter(
        years,
        series,
        s=80,
        color=COLORS["primary"],
        label="Observed",
        zorder=4,
        edgecolors="white",
        linewidth=1,
    )

    # Get breaks
    break_years = break_results.get("bai_perron", {}).get("break_years", [])
    segments = break_results.get("bai_perron", {}).get("segments", [])

    # Fit and plot segmented regression
    colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["tertiary"],
        COLORS["quaternary"],
        COLORS["neutral"],
    ]

    break_indices = (
        [0] + [np.argmin(np.abs(years - by)) for by in break_years] + [len(years)]
    )

    for i in range(len(break_indices) - 1):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1]

        seg_years = years[start_idx:end_idx]
        seg_series = series[start_idx:end_idx]

        if len(seg_years) >= 2:
            # Fit linear trend for segment
            t = np.arange(len(seg_years))
            X = np.column_stack([np.ones(len(t)), t])
            beta = np.linalg.lstsq(X, seg_series, rcond=None)[0]
            fitted = X @ beta

            color = colors[i % len(colors)]
            ax.plot(
                seg_years,
                fitted,
                linestyle="-",
                linewidth=2.5,
                color=color,
                alpha=0.8,
                label=f"Regime {i + 1}",
            )

            # Add mean line
            ax.hlines(
                y=np.mean(seg_series),
                xmin=seg_years[0],
                xmax=seg_years[-1],
                colors=color,
                linestyles="--",
                linewidth=1.5,
                alpha=0.5,
            )

    # Mark break points
    for break_year in break_years:
        ax.axvline(
            x=break_year,
            color=COLORS["break_line"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="_nolegend_",
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("ND International Migration", fontsize=12)
    ax.set_title(
        "Segmented Regression: Regime Changes in ND Immigration",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add regime summary
    if segments:
        summary_text = "Regime Means:\n"
        for i, seg in enumerate(segments):
            summary_text += (
                f"  {seg['start_year']}-{seg['end_year']}: {seg['mean']:.0f}\n"
            )
        ax.text(
            0.98,
            0.98,
            summary_text.strip(),
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            family="monospace",
        )

    plt.tight_layout()

    filepath = str(FIGURES_DIR / "module_2_1_2_segmented_regression")
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


def run_analysis() -> ModuleResult:
    """
    Main analysis function - perform structural break detection.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(
        module_id="2.1.2",
        analysis_name="structural_break_detection",
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
    years = df["year"]
    series = df[target_var]

    # Record parameters
    result.parameters = {
        "target_variable": target_var,
        "n_observations": len(df),
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "policy_events_tested": list(POLICY_EVENTS.keys()),
        "bai_perron_settings": {
            "algorithm": "PELT",
            "model": "rbf",
            "min_segment_size": 2,
            "penalty": "BIC",
        },
        "cusum_settings": {
            "significance_level": 0.05,
            "bounds_method": "Brown-Durbin-Evans (1975)",
        },
    }

    # Document decisions
    result.add_decision(
        decision_id="D001",
        category="methodology",
        decision="Use Bai-Perron (PELT) for data-driven break detection",
        rationale="PELT algorithm is computationally efficient and optimal under BIC criterion. "
        "RBF kernel is robust to various types of changes (level shifts, trend changes).",
        alternatives=[
            "Binary segmentation",
            "Window-based detection",
            "ARIMA with intervention",
        ],
        evidence="PELT has O(n) complexity and is consistent for multiple break detection",
        reversible=True,
    )

    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Set minimum segment size to 2 observations",
        rationale="With only 15 total observations, requiring larger segments would limit "
        "ability to detect recent breaks (e.g., 2020 COVID, 2022 surge).",
        alternatives=[
            "min_size=3 for more stable estimates",
            "min_size=4 for traditional Bai-Perron",
        ],
        evidence="n=15 limits flexibility; 2-observation minimum allows detection of COVID break",
        reversible=True,
    )

    result.warnings.append(
        "Short time series (n=15) limits statistical power for structural break tests."
    )
    result.warnings.append(
        "Results should be interpreted alongside known policy events, not purely data-driven."
    )

    # Store all break detection results
    break_results = {}

    # 1. Bai-Perron Test (PELT)
    print("\n1. Running Bai-Perron test (PELT algorithm)...")
    bai_perron = bai_perron_test(series.values, years, min_size=2)
    break_results["bai_perron"] = bai_perron
    print(f"   Detected {bai_perron['n_breaks_detected']} break(s)")
    if bai_perron["break_years"]:
        print(f"   Break years: {bai_perron['break_years']}")
    for shift in bai_perron.get("regime_shifts", []):
        print(
            f"   Regime shift at {shift['break_year']}: {shift['absolute_shift']:+.0f} ({shift['percent_shift']:+.1f}%)"
        )

    # 2. CUSUM Test
    print("\n2. Running CUSUM test for parameter stability...")
    cusum = cusum_test(series.values, years)
    break_results["cusum"] = cusum
    print(f"   {cusum['conclusion']}")
    print(f"   Stability ratio (max CUSUM / bound): {cusum['stability_ratio']:.3f}")

    # 3. Chow Tests at known policy dates
    print("\n3. Running Chow tests at known policy event dates...")
    chow_results = {}
    for event_year, event_info in POLICY_EVENTS.items():
        print(f"   Testing break at {event_year} ({event_info['name']})...")
        chow = chow_test(series.values, years.values, event_year)
        chow_results[str(event_year)] = chow
        if chow.get("error"):
            print(f"     Could not test: {chow['error']}")
        else:
            sig_text = (
                "SIGNIFICANT" if chow["significant_at_5pct"] else "not significant"
            )
            print(
                f"     F={chow['f_statistic']:.3f}, p={chow['p_value']:.4f} ({sig_text})"
            )

    break_results["chow_tests"] = chow_results

    # Store results
    result.results = {
        "bai_perron": bai_perron,
        "cusum": cusum,
        "chow_tests": chow_results,
        "summary": {
            "data_driven_breaks": bai_perron["break_years"],
            "n_breaks_detected": bai_perron["n_breaks_detected"],
            "cusum_stable": cusum["stable"],
            "significant_policy_breaks": [
                int(year)
                for year, test in chow_results.items()
                if test.get("significant_at_5pct")
            ],
        },
    }

    # Generate figures
    print("\nGenerating figures...")

    # Plot structural breaks
    fig1 = plot_structural_breaks(df, target_var, break_results)
    print(f"  Saved: {fig1}")

    # Plot CUSUM
    fig2 = plot_cusum(cusum, df)
    if fig2:
        print(f"  Saved: {fig2}")

    # Plot segmented regression
    fig3 = plot_segmented_regression(df, target_var, break_results)
    print(f"  Saved: {fig3}")

    # Store diagnostics
    result.diagnostics = {
        "figures_generated": [
            "module_2_1_2_structural_breaks.png",
            "module_2_1_2_cusum_test.png",
            "module_2_1_2_segmented_regression.png",
        ],
        "test_interpretation": {
            "bai_perron": "Data-driven detection of optimal breakpoints using BIC criterion",
            "cusum": "H0: Parameters stable over time. If CUSUM crosses bounds, reject stability.",
            "chow": "H0: No structural break at specified date. Reject if p < 0.05.",
        },
        "policy_event_alignment": {
            year: {
                "event": info["name"],
                "chow_significant": chow_results.get(str(year), {}).get(
                    "significant_at_5pct"
                ),
                "detected_by_pelt": year in bai_perron["break_years"],
            }
            for year, info in POLICY_EVENTS.items()
        },
    }

    # Suggest next steps
    result.next_steps = [
        "Use detected break dates for intervention analysis in time series models",
        "Incorporate regime-specific parameters in forecasting models",
        "Consider structural breaks when specifying VAR/VECM models (Module 2.2)",
        "Compare break detection with policy implementation dates for causal inference",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 2.1.2: Structural Break Detection")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_2_1_2_structural_breaks.json")

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
        print("Structural Break Detection Summary:")
        print(
            f"  Data-driven breaks (Bai-Perron): {result.results['summary']['data_driven_breaks']}"
        )
        print(
            f"  CUSUM parameter stability: {'Stable' if result.results['summary']['cusum_stable'] else 'Unstable'}"
        )
        print(
            f"  Significant policy breaks (Chow): {result.results['summary']['significant_policy_breaks']}"
        )

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
