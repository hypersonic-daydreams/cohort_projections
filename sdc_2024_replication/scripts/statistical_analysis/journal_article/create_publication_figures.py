#!/usr/bin/env python3
"""
Create Publication-Quality Figures for Journal Article

This script generates 9 publication-quality figures for an academic journal article
on forecasting international migration to North Dakota.

Figures:
1. Time Series Overview (fig_01_timeseries.pdf)
2. Geographic Concentration (fig_02_concentration.pdf)
3. Unit Root Diagnostics (fig_03_acf_pacf.pdf)
4. Structural Breaks (fig_04_structural_breaks.pdf)
5. Gravity Model Results (fig_05_gravity.pdf)
6. Event Study (fig_06_event_study.pdf)
7. Survival Curves (fig_07_survival.pdf)
8. Cox Model Hazard Ratios (fig_08_cox.pdf)
9. Forecast Scenarios (fig_08_scenarios.pdf)

Author: Generated with Claude Code
Date: 2025-12-29
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

# =============================================================================
# Publication Style Configuration
# =============================================================================

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6.5, 4.5),
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Color-blind friendly palette (Tableau 10)
COLORS = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#B07AA1",
    "brown": "#9C755F",
    "pink": "#FF9DA7",
    "gray": "#BAB0AC",
    "olive": "#76B7B2",
    "cyan": "#EDC948",
}

# Paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filename: str) -> dict[str, Any]:
    """Load a JSON file from the results directory."""
    filepath = RESULTS_DIR / filename
    with open(filepath) as f:
        return json.load(f)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure in both PDF and PNG formats."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"Saved: {pdf_path.name}, {png_path.name}")
    plt.close(fig)


def _resolve_duration_results_filename() -> str:
    """Resolve the duration results filename for the survival figure."""
    duration_tag = os.environ.get("SDC_DURATION_TAG")
    if duration_tag:
        duration_tag = duration_tag.strip() or None

    if not duration_tag:
        try:
            scenario_data = load_json("module_9_scenario_modeling.json")
            duration_tag = (
                scenario_data.get("parameters", {}).get("duration_model_tag") or None
            )
        except Exception:
            duration_tag = None

    if (
        not duration_tag
        and (RESULTS_DIR / "module_8_duration_analysis__P0.json").exists()
    ):
        duration_tag = "P0"

    if duration_tag:
        tagged_filename = f"module_8_duration_analysis__{duration_tag}.json"
        if (RESULTS_DIR / tagged_filename).exists():
            return tagged_filename

    return "module_8_duration_analysis.json"


# =============================================================================
# Figure 1: Time Series Overview
# =============================================================================


def create_figure_01_timeseries() -> None:
    """Create Figure 1: Time Series Overview with HP filter trend."""
    # Load data
    summary_stats = load_json("module_1_1_summary_statistics.json")
    load_json("module_2_1_2_structural_breaks.json")

    # Load raw time series data
    nd_data = pd.read_csv(DATA_DIR / "nd_migration_summary.csv")

    # Extract HP decomposition
    hp_data = summary_stats["results"]["hp_decomposition"]
    years = np.array(hp_data["years"])
    trend = np.array(hp_data["trend"])

    # Get ND share values (original series) - need to reconstruct from trend + cycle
    original = np.array(trend) + np.array(hp_data["cycle"])

    # Get migration values (raw counts)
    migration = nd_data["nd_intl_migration"].values
    migration_years = nd_data["year"].values

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)

    # Panel A: Absolute migration flows
    ax1.plot(
        migration_years,
        migration,
        "o-",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=4,
        label="Net international migration",
    )

    # Add trend line for migration (simple linear for visual)
    z = np.polyfit(migration_years, migration, 2)
    p = np.poly1d(z)
    ax1.plot(
        migration_years,
        p(migration_years),
        "--",
        color=COLORS["orange"],
        linewidth=1.5,
        label="Quadratic trend",
    )

    # Mark structural breaks
    for break_year in [2020, 2021]:
        ax1.axvline(
            x=break_year, color=COLORS["red"], linestyle=":", linewidth=1.0, alpha=0.7
        )

    ax1.set_ylabel("Net int'l migration (persons)")
    ax1.set_title(
        "(A) Annual Net International Migration to North Dakota",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # Break year labels and COVID annotation (avoid long cross-plot arrow)
    ax1.text(2020.2, ax1.get_ylim()[1] * 0.95, "2020", fontsize=8, color=COLORS["red"])
    ax1.text(2021.2, ax1.get_ylim()[1] * 0.95, "2021", fontsize=8, color=COLORS["red"])
    ax1.text(
        2020.05,
        ax1.get_ylim()[1] * 0.75,
        "COVID-19",
        fontsize=8,
        color=COLORS["gray"],
        rotation=90,
        va="center",
        ha="left",
    )

    # Panel B: ND Share with HP trend
    # Note: original/trend values are already in percentage form (0.173 = 0.173%)
    # Do NOT multiply by 100 again
    ax2.plot(
        years,
        original,
        "o-",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=4,
        label="Observed share",
    )
    ax2.plot(
        years,
        trend,
        "--",
        color=COLORS["orange"],
        linewidth=1.5,
        label=r"HP trend ($\lambda = 6.25$)",
    )

    # Mark structural breaks
    for break_year in [2020, 2021]:
        ax2.axvline(
            x=break_year, color=COLORS["red"], linestyle=":", linewidth=1.0, alpha=0.7
        )

    ax2.set_xlabel("Year")
    ax2.set_ylabel("ND share of U.S. net int'l migration")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax2.set_title(
        "(B) North Dakota Share of U.S. Net International Migration",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.set_xlim(2009.5, 2024.5)

    # Add break year labels
    ax2.text(2020.2, ax2.get_ylim()[1] * 0.95, "2020", fontsize=8, color=COLORS["red"])
    ax2.text(2021.2, ax2.get_ylim()[1] * 0.95, "2021", fontsize=8, color=COLORS["red"])

    plt.tight_layout()
    save_figure(fig, "fig_01_timeseries")


# =============================================================================
# Figure 2: Geographic Concentration
# =============================================================================


def create_figure_02_concentration() -> None:
    """Create Figure 2: Geographic Concentration - Location Quotients."""
    # Load data
    lq_data = load_json("module_1_2_location_quotients.json")

    # Get top countries by LQ
    top_lq = lq_data["results"]["location_quotients"]["top_20_lq_countries"][:12]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5))

    countries = [c["country"] for c in top_lq]
    lq_values = [c["location_quotient"] for c in top_lq]
    nd_pops = [c["nd_foreign_born"] for c in top_lq]

    # Shorten country names
    name_map = {
        "Other Australian and New Zealand Subregion": "Oceania (Other)",
        "Other Eastern Africa": "E. Africa (Other)",
        "Other Middle Africa": "Middle Africa (Other)",
        "Other Western Africa": "W. Africa (Other)",
        "Other Northern Africa": "N. Africa (Other)",
        "United Kingdom (inc. Crown Dependencies)": "United Kingdom",
        "Czechoslovakia (includes Czech Republic and Slovakia)": "Czechia/Slovakia",
        "Democratic Republic of Congo (Zaire)": "DR Congo",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    }
    countries = [name_map.get(c, c) for c in countries]

    # Create horizontal bar chart
    y_pos = np.arange(len(countries))
    bars = ax.barh(y_pos, lq_values, color=COLORS["blue"], alpha=0.8, height=0.7)

    # Color bars by population size (darker = larger)
    max_pop = max(nd_pops)
    for bar, pop in zip(bars, nd_pops):
        intensity = 0.3 + 0.7 * (pop / max_pop)
        bar.set_alpha(intensity)

    # Add ND foreign-born counts as end labels.
    for bar, pop in zip(bars, nd_pops):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(pop):,}",
            fontsize=8,
            color="#555555",
            verticalalignment="center",
            horizontalalignment="left",
        )

    # Add reference line at LQ = 1
    ax.axvline(x=1, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.text(
        1.05,
        -0.01,
        "LQ = 1 (national average)",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        color=COLORS["red"],
        verticalalignment="top",
        horizontalalignment="left",
        clip_on=False,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries)
    ax.tick_params(axis="y", pad=3)
    ax.set_xlabel("Location Quotient (LQ)")
    ax.set_title(
        "Top Origins by Concentration in North Dakota (2023)",
        fontsize=11,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color=COLORS["gray"], alpha=0.6)
    ax.set_xlim(0, math.ceil((max(lq_values) + 6) / 5) * 5)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "fig_02_concentration")


# =============================================================================
# Figure 3: Unit Root Diagnostics (ACF/PACF)
# =============================================================================


def create_figure_03_acf_pacf() -> None:
    """Create Figure 3: ACF/PACF plots for unit root diagnostics."""
    # Load raw data
    nd_data = pd.read_csv(DATA_DIR / "nd_migration_summary.csv")
    migration = nd_data["nd_intl_migration"].values

    # Calculate ACF and PACF manually
    from statsmodels.tsa.stattools import acf, pacf

    n_lags = 10

    # Original series
    acf_orig = acf(migration, nlags=n_lags, fft=False)
    try:
        pacf_orig = pacf(migration, nlags=min(n_lags, len(migration) // 2 - 1))
    except Exception:
        pacf_orig = acf_orig  # Fallback

    # Differenced series
    diff_migration = np.diff(migration)
    acf_diff = acf(diff_migration, nlags=n_lags, fft=False)
    try:
        pacf_diff = pacf(
            diff_migration, nlags=min(n_lags, len(diff_migration) // 2 - 1)
        )
    except Exception:
        pacf_diff = acf_diff

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))

    # Confidence bounds (95%)
    n = len(migration)
    conf_bound = 1.96 / np.sqrt(n)

    lags_acf = np.arange(len(acf_orig))
    lags_pacf = np.arange(len(pacf_orig))
    lags_acf_diff = np.arange(len(acf_diff))
    lags_pacf_diff = np.arange(len(pacf_diff))

    # Panel A: ACF - Original
    ax1 = axes[0, 0]
    ax1.bar(lags_acf, acf_orig, color=COLORS["blue"], alpha=0.8, width=0.5)
    ax1.axhline(y=conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax1.axhline(y=-conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("ACF")
    ax1.set_title("(A) ACF: Level Series", fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.5, 1.1)

    # Panel B: PACF - Original
    ax2 = axes[0, 1]
    ax2.bar(lags_pacf, pacf_orig, color=COLORS["blue"], alpha=0.8, width=0.5)
    ax2.axhline(y=conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax2.axhline(y=-conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("PACF")
    ax2.set_title("(B) PACF: Level Series", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.5, 1.1)

    # Panel C: ACF - Differenced
    ax3 = axes[1, 0]
    ax3.bar(lags_acf_diff, acf_diff, color=COLORS["green"], alpha=0.8, width=0.5)
    ax3.axhline(y=conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax3.axhline(y=-conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")
    ax3.set_title("(C) ACF: First Difference", fontsize=10, fontweight="bold")
    ax3.set_ylim(-0.8, 1.1)

    # Panel D: PACF - Differenced
    ax4 = axes[1, 1]
    ax4.bar(lags_pacf_diff, pacf_diff, color=COLORS["green"], alpha=0.8, width=0.5)
    ax4.axhline(y=conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax4.axhline(y=-conf_bound, color=COLORS["red"], linestyle="--", linewidth=0.8)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("PACF")
    ax4.set_title("(D) PACF: First Difference", fontsize=10, fontweight="bold")
    ax4.set_ylim(-0.8, 1.1)

    # Add overall title
    fig.suptitle(
        "Autocorrelation Diagnostics for ND International Migration",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    save_figure(fig, "fig_03_acf_pacf")


# =============================================================================
# Figure 4: Structural Breaks
# =============================================================================


def create_figure_04_structural_breaks() -> None:
    """Create Figure 4: Structural breaks with segmented regression."""
    # Load data
    breaks_data = load_json("module_2_1_2_structural_breaks.json")
    nd_data = pd.read_csv(DATA_DIR / "nd_migration_summary.csv")

    years = nd_data["year"].values
    migration = nd_data["nd_intl_migration"].values

    # Get Chow test results
    chow_2020 = breaks_data["results"]["chow_tests"]["2020"]
    breaks_data["results"]["chow_tests"]["2021"]

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6))

    # Panel A: Time series with break points
    ax1.plot(
        years,
        migration,
        "o-",
        color=COLORS["blue"],
        linewidth=1.5,
        markersize=5,
        label="Observed",
    )

    # Pre-2020 trend line
    pre_mask = years < 2020
    pre_years = years[pre_mask]
    z_pre = np.polyfit(years[pre_mask], migration[pre_mask], 1)
    p_pre = np.poly1d(z_pre)
    ax1.plot(
        years[pre_mask],
        p_pre(years[pre_mask]),
        "--",
        color=COLORS["orange"],
        linewidth=1.5,
        label=f"Pre-break trend ({pre_years.min()}-{pre_years.max()})",
    )

    # Post-2020 trend line
    post_mask = years >= 2020
    post_years = years[post_mask]
    z_post = np.polyfit(years[post_mask], migration[post_mask], 1)
    p_post = np.poly1d(z_post)
    ax1.plot(
        years[post_mask],
        p_post(years[post_mask]),
        "--",
        color=COLORS["green"],
        linewidth=1.5,
        label=f"Post-break trend ({post_years.min()}-{post_years.max()})",
    )

    # Mark break points
    ax1.axvline(
        x=2020,
        color=COLORS["red"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Candidate break (2020)",
    )

    # Add annotation
    ax1.annotate(
        "Chow test:\n"
        f"F = {chow_2020['f_statistic']:.2f}\n"
        f"p = {chow_2020['p_value']:.4f}",
        xy=(2020, migration[years == 2020][0] if 2020 in years else 0),
        xytext=(2017.0, 3900),
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.8),
        fontsize=8,
        ha="left",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=COLORS["gray"], alpha=0.9
        ),
    )

    ax1.set_ylabel("International migration (persons)")
    ax1.set_title(
        "(A) Structural Break Detection: Segmented Regression",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xlim(2009.5, 2024.5)
    ax1.set_ylim(bottom=0)

    # Panel B: CUSUM plot
    cusum_data = breaks_data["results"]["cusum"]
    cusum_values = np.array(cusum_data["cusum_values"])
    upper_bounds = np.array(cusum_data["upper_bounds"])
    lower_bounds = np.array(cusum_data["lower_bounds"])

    # Generate x-values for CUSUM (recursive residuals start later)
    n_cusum = len(cusum_values)
    cusum_years = years[2 : 2 + n_cusum] if len(years) >= n_cusum + 2 else None
    cusum_x = cusum_years if cusum_years is not None else np.arange(1, n_cusum + 1)

    ax2.plot(
        cusum_x,
        cusum_values,
        "-",
        color=COLORS["blue"],
        linewidth=1.5,
        label="CUSUM statistic",
    )
    ax2.plot(
        cusum_x,
        upper_bounds,
        "--",
        color=COLORS["red"],
        linewidth=1.0,
        label="95% bounds (α=0.05)",
    )
    ax2.plot(cusum_x, lower_bounds, "--", color=COLORS["red"], linewidth=1.0)
    ax2.fill_between(
        cusum_x, lower_bounds, upper_bounds, color=COLORS["red"], alpha=0.1
    )
    ax2.axhline(y=0, color="black", linewidth=0.5)

    if cusum_years is not None:
        ax2.set_xlabel("Year (recursive residual)")
        ax2.set_xticks(cusum_years[::2])
    else:
        ax2.set_xlabel("Recursive residual index")
    ax2.set_ylabel("CUSUM")
    ax2.set_title(
        "(B) CUSUM Test for Parameter Stability", fontsize=11, fontweight="bold"
    )
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, "fig_04_structural_breaks")


# =============================================================================
# Figure 5: Gravity Model Results
# =============================================================================


def create_figure_05_gravity() -> None:
    """Create Figure 5: Gravity model coefficient plot."""
    # Load data
    gravity_data = load_json("module_5_gravity_model.json")

    # Extract coefficients from full gravity model
    full_model = gravity_data["model_2_full_gravity"]
    coeffs = full_model["coefficients"]

    # Prepare data for coefficient plot
    var_names = ["Network (log diaspora)", "Origin mass", "Destination mass"]
    var_keys = ["log_diaspora", "log_origin_total", "log_state_total"]

    estimates = [coeffs[k]["estimate"] for k in var_keys]
    ci_lower = [coeffs[k]["ci_95_lower"] for k in var_keys]
    ci_upper = [coeffs[k]["ci_95_upper"] for k in var_keys]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 4))

    # Panel A: Coefficient plot
    y_pos = np.arange(len(var_names))
    ax1.errorbar(
        estimates,
        y_pos,
        xerr=[
            np.array(estimates) - np.array(ci_lower),
            np.array(ci_upper) - np.array(estimates),
        ],
        fmt="o",
        color=COLORS["blue"],
        markersize=8,
        capsize=4,
        capthick=1.5,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(var_names)
    ax1.set_xlabel("Coefficient (95% CI)")
    ax1.set_title("(A) Gravity Model Coefficients", fontsize=10, fontweight="bold")
    x_min = min(min(ci_lower), 0)
    x_max = max(max(ci_upper), 0)
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.axvline(x=0, color=COLORS["gray"], linestyle="--", linewidth=1, alpha=0.9)

    # Panel B: Network elasticity comparison across models
    models = ["Simple", "Full\nGravity", "State\nFE"]
    model_1_diaspora = gravity_data["model_1_simple_network"]["coefficients"][
        "log_diaspora"
    ]
    model_2_diaspora = gravity_data["model_2_full_gravity"]["coefficients"][
        "log_diaspora"
    ]
    model_3_diaspora = gravity_data["model_3_state_fixed_effects"]["network_elasticity"]

    elasticities = [
        model_1_diaspora["estimate"],
        model_2_diaspora["estimate"],
        model_3_diaspora["estimate"],
    ]
    ci_lower_b = [
        model_1_diaspora["ci_95_lower"],
        model_2_diaspora["ci_95_lower"],
        model_3_diaspora["ci_95_lower"],
    ]
    ci_upper_b = [
        model_1_diaspora["ci_95_upper"],
        model_2_diaspora["ci_95_upper"],
        model_3_diaspora["ci_95_upper"],
    ]

    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]
    x_pos = np.arange(len(models))
    ax2.errorbar(
        x_pos,
        elasticities,
        yerr=[
            np.array(elasticities) - np.array(ci_lower_b),
            np.array(ci_upper_b) - np.array(elasticities),
        ],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=2,
    )
    ax2.scatter(x_pos, elasticities, color=colors, s=80, alpha=0.9, zorder=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Network elasticity")
    ax2.set_title(
        "(B) Diaspora Effect by Specification", fontsize=10, fontweight="bold"
    )
    y_min = min(min(ci_lower_b), 0)
    y_max = max(max(ci_upper_b), 0)
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    ax2.axhline(y=0, color=COLORS["gray"], linestyle="--", linewidth=1, alpha=0.9)

    # Add value labels
    for x, val in zip(x_pos, elasticities):
        ax2.text(
            x,
            val + 0.03 * (y_max - y_min),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    save_figure(fig, "fig_05_gravity")


# =============================================================================
# Figure 6: Event Study
# =============================================================================


def create_figure_06_event_study() -> None:
    """Create Figure 6: DiD Event Study for Travel Ban."""
    # Load data
    causal_data = load_json("module_7_causal_inference.json")
    event_study = causal_data["results"]["event_study"]

    # Extract coefficients
    coeffs = event_study["coefficients"]

    rel_times = [c["rel_time"] for c in coeffs]
    estimates = [c["coefficient"] for c in coeffs]
    ci_lower = [c["ci_lower"] for c in coeffs]
    ci_upper = [c["ci_upper"] for c in coeffs]
    is_ref = [c.get("is_reference", False) for c in coeffs]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Plot confidence intervals as error bars
    for i, (t, est, low, up, ref) in enumerate(
        zip(rel_times, estimates, ci_lower, ci_upper, is_ref)
    ):
        if ref:
            # Reference period - plot as open circle
            ax.plot(
                t,
                est,
                "o",
                color=COLORS["gray"],
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
            )
        elif t < 0:
            # Pre-treatment
            ax.errorbar(
                t,
                est,
                yerr=[[est - low], [up - est]],
                fmt="o",
                color=COLORS["blue"],
                markersize=6,
                capsize=3,
                capthick=1,
            )
        else:
            # Post-treatment
            ax.errorbar(
                t,
                est,
                yerr=[[est - low], [up - est]],
                fmt="s",
                color=COLORS["red"],
                markersize=6,
                capsize=3,
                capthick=1,
            )

    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=-0.5, color=COLORS["gray"], linestyle="--", linewidth=1.0, alpha=0.7)

    # Shade post-treatment period
    ax.axvspan(-0.5, max(rel_times) + 0.5, alpha=0.1, color=COLORS["red"])

    # Labels
    min_rel = int(min(rel_times))
    max_rel = int(max(rel_times))
    tick_step = 2 if (max_rel - min_rel) >= 12 else 1
    x_ticks = list(range(min_rel, max_rel + 1, tick_step))
    if max_rel not in x_ticks:
        x_ticks.append(max_rel)
    if 0 not in x_ticks:
        x_ticks.append(0)
    ax.set_xticks(sorted(set(x_ticks)))

    ax.set_xlabel("Years relative to Travel Ban (t = 0 is FY2018, first full post year)")
    ax.set_ylabel("Difference in log refugee arrivals (treated − control)")
    ax.set_title(
        "Event Study: Treated–Control Divergence in Refugee Arrivals",
        fontsize=11,
        fontweight="bold",
    )

    # Legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["blue"],
            markersize=8,
            label="Pre-treatment",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLORS["red"],
            markersize=8,
            label="Post-treatment",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["gray"],
            markerfacecolor="white",
            markersize=8,
            markeredgewidth=2,
            label="Reference (FY2017)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9)

    # Add pre-trend test annotation
    pre_trend = event_study["pre_trend_test"]
    pre_trend_p = pre_trend.get("p_value")
    if pre_trend_p is None:
        pre_trend_p_label = "p = NA"
    elif pre_trend_p < 0.001:
        pre_trend_p_label = "p < 0.001"
    else:
        pre_trend_p_label = f"p = {pre_trend_p:.3f}"
    ax.text(
        0.95,
        0.95,
        f"Pre-treatment test: F = {pre_trend['f_statistic']:.2f}\n{pre_trend_p_label}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=COLORS["gray"], alpha=0.9
        ),
    )

    plt.tight_layout()
    save_figure(fig, "fig_06_event_study")


# =============================================================================
# Appendix Figure: Extended Event Study (Supplemental)
# =============================================================================


def create_fig_app_event_study_extended() -> None:
    """Create Appendix Figure: extended event study through FY2024 (supplemental; descriptive)."""
    causal_data = load_json("module_7_causal_inference.json")
    event_study = causal_data["results"]["event_study_extended"]

    coef_df = pd.DataFrame(event_study["coefficients"]).sort_values("year")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Regime shading (visual aid only; not causal identification)
    ax.axvspan(2017.5, 2019.5, alpha=0.08, color=COLORS["red"])
    ax.axvspan(2019.5, 2021.5, alpha=0.08, color=COLORS["blue"])
    ax.axvspan(2021.5, 2024.5, alpha=0.08, color=COLORS["green"])

    # Reference line at zero
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)

    # Plot coefficients with 95% CIs, split pre/post for styling
    non_ref = coef_df[~coef_df["is_reference"]].copy()
    pre = non_ref[non_ref["year"] < 2018]
    post = non_ref[non_ref["year"] >= 2018]

    for group, marker, color, label in [
        (pre, "o", COLORS["blue"], "Pre-treatment"),
        (post, "s", COLORS["red"], "Post (includes multiple regimes)"),
    ]:
        if group.empty:
            continue
        ax.vlines(
            group["year"],
            group["ci_lower"],
            group["ci_upper"],
            color=color,
            linewidth=1.2,
            alpha=0.8,
        )
        ax.scatter(
            group["year"],
            group["coefficient"],
            marker=marker,
            color=color,
            s=30,
            label=label,
            zorder=3,
        )

    # Reference period marker (t = -1, FY2017)
    ref = coef_df[coef_df["is_reference"]]
    if not ref.empty:
        ax.scatter(
            ref["year"],
            ref["coefficient"],
            marker="o",
            facecolors="white",
            edgecolors=COLORS["gray"],
            linewidths=1.5,
            s=40,
            label="Reference (FY2017)",
            zorder=4,
        )

    # Regime markers
    ax.axvline(2017.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axvline(2019.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axvline(2020.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(2018.0, ax.get_ylim()[1] * 0.92, "FY2018", fontsize=7, color=COLORS["gray"])
    ax.text(2020.0, ax.get_ylim()[1] * 0.92, "FY2020", fontsize=7, color=COLORS["gray"])
    ax.text(2021.0, ax.get_ylim()[1] * 0.92, "FY2021", fontsize=7, color=COLORS["gray"])

    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Event-study coefficient (log arrivals)")
    ax.set_title(
        "Extended Event Study: Treated vs Control Refugee Arrivals (FY2002–FY2024)\n"
        "(Supplemental; descriptive regime dynamics)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(coef_df["year"].min() - 0.5, 2024.5)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    pre_trend = event_study.get("pre_trend_test", {})
    p_val = pre_trend.get("p_value")
    p_label = (
        "<0.001"
        if p_val is not None and p_val < 0.001
        else f"{p_val:.3f}"
        if p_val is not None
        else "NA"
    )
    ax.text(
        0.98,
        0.02,
        "Not a causal Travel Ban estimand post-2019.\n"
        "Post-2020 overlaps COVID, rescission, and USRAP rebuild.\n"
        f"Joint pre-trend test: p = {p_label}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=COLORS["gray"], alpha=0.9
        ),
    )

    plt.tight_layout()
    save_figure(fig, "fig_app_event_study_extended")


# =============================================================================
# Figure 7: Survival Curves
# =============================================================================


def create_figure_07_survival() -> None:
    """Create Figure 7: Kaplan-Meier survival curves by intensity quartile."""
    duration_filename = _resolve_duration_results_filename()

    # Load data
    duration_data = load_json(duration_filename)

    # Get KM data by intensity
    km_intensity = duration_data["results"]["kaplan_meier_by_intensity"]
    groups = km_intensity["groups"]

    # Get overall life table for x-axis
    life_table = duration_data["results"]["kaplan_meier"]["life_table"]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Plot survival curves for each quartile
    quartile_colors = {
        "Q1 (Low)": COLORS["red"],
        "Q2": COLORS["orange"],
        "Q3": COLORS["olive"],
        "Q4 (High)": COLORS["blue"],
    }

    # Get time points from life table
    times = [row["time_years"] for row in life_table]

    # Overall survival curve
    surv_probs = [row["survival_probability"] for row in life_table]

    # Plot overall curve
    ax.step(
        times,
        surv_probs,
        where="post",
        color=COLORS["gray"],
        linewidth=2,
        linestyle="-",
        label="Overall",
        alpha=0.5,
    )

    quartile_order = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    for quartile in quartile_order:
        info = groups.get(quartile)
        if not info:
            continue

        median = info["median_survival"]
        life_table_q = info.get("life_table")
        if not life_table_q:
            raise KeyError(
                "Missing per-group Kaplan-Meier life table for "
                f"{quartile!r} in {duration_filename}. "
                "Regenerate duration results using module_8_duration_analysis.py."
            )

        times_q = [row["time_years"] for row in life_table_q]
        surv_q = [row["survival_probability"] for row in life_table_q]

        ax.step(
            times_q,
            surv_q,
            where="post",
            linewidth=1.8,
            color=quartile_colors[quartile],
            label=f"{quartile} (n={info['n_subjects']}, med={median:.1f}yr)",
        )

    ax.set_xlabel("Duration (years)")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        "Kaplan-Meier Curves: Immigration Wave Duration by Intensity",
        fontsize=11,
        fontweight="bold",
    )
    max_time = max(times) if times else 1
    ax.set_xlim(1, max_time)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Add log-rank test result
    log_rank = km_intensity["log_rank_test"]
    p_value = log_rank.get("p_value")
    if isinstance(p_value, (int, float)) and p_value > 0:
        if p_value < 0.001:
            power = int(math.floor(-math.log10(p_value)))
            p_label = rf"p < $10^{{-{power}}}$"
        else:
            p_label = f"p = {p_value:.3f}"
    else:
        p_label = "p = NA"
    ax.text(
        0.05,
        0.05,
        rf"Log-rank test: $\chi^2$ = {log_rank['test_statistic']:.1f}"
        "\n"
        f"{p_label}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=COLORS["gray"], alpha=0.9
        ),
    )

    # Add median survival line
    ax.axhline(y=0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(max_time - 0.5, 0.52, "Median", fontsize=7, color="black", ha="right")

    plt.tight_layout()
    save_figure(fig, "fig_07_survival")


# =============================================================================
# Figure 8: Cox Model Hazard Ratios
# =============================================================================


def create_figure_08_cox() -> None:
    """Create Figure 8: Cox proportional hazards model hazard ratios."""
    duration_filename = _resolve_duration_results_filename()
    duration_data = load_json(duration_filename)

    cox_results = duration_data["results"]["cox_proportional_hazards"]
    coef_table = cox_results["coefficient_table"]

    variables = [
        "log_intensity",
        "high_intensity",
        "early_wave",
        "peak_arrivals",
        "nationality_region_Americas",
        "nationality_region_Asia",
        "nationality_region_Europe",
        "nationality_region_Middle East",
        "nationality_region_Other",
    ]
    variables = [v for v in variables if v in coef_table]
    if not variables:
        raise KeyError(
            f"No expected Cox coefficient_table entries found in {duration_filename}."
        )

    label_map = {
        "log_intensity": "Log intensity",
        "high_intensity": "High intensity (ratio > 5)",
        "early_wave": "Early wave (start ≤ 2010)",
        "peak_arrivals": "Peak arrivals (per 1,000)",
        "nationality_region_Americas": "Origin: Americas (vs Africa)",
        "nationality_region_Asia": "Origin: Asia (vs Africa)",
        "nationality_region_Europe": "Origin: Europe (vs Africa)",
        "nationality_region_Middle East": "Origin: Middle East (vs Africa)",
        "nationality_region_Other": "Origin: Other (vs Africa)",
    }

    fig_height = max(4.0, 0.45 * len(variables) + 1.5)
    fig, ax = plt.subplots(figsize=(6.75, fig_height))

    y_positions = np.arange(len(variables))
    for idx, var in enumerate(variables):
        entry = coef_table[var]
        hr = float(entry["hazard_ratio"])
        ci_lower = float(entry["hr_ci_95_lower"])
        ci_upper = float(entry["hr_ci_95_upper"])
        p_value = float(entry["p_value"])

        color = COLORS["blue"] if p_value < 0.05 else COLORS["gray"]
        ax.errorbar(
            hr,
            idx,
            xerr=[[hr - ci_lower], [ci_upper - hr]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            markersize=6,
        )

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label_map.get(v, v.replace("_", " ")) for v in variables])
    ax.invert_yaxis()

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    ci_bounds = [
        float(coef_table[v]["hr_ci_95_lower"]) for v in variables
    ] + [float(coef_table[v]["hr_ci_95_upper"]) for v in variables]
    x_min = max(0.05, min(ci_bounds) * 0.85)
    x_max = max(ci_bounds) * 1.15
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Hazard ratio for wave termination (95% CI)")
    ax.set_title(
        "Cox Proportional Hazards Model: Predictors of Wave Persistence",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, which="major", axis="x", alpha=0.25)

    plt.tight_layout()
    save_figure(fig, "fig_08_cox")


# =============================================================================
# Figure 9: Forecast Scenarios
# =============================================================================


def create_figure_08_scenarios() -> None:
    """Create Figure 9: Forecast scenarios with uncertainty bands."""
    # Load data
    scenario_data = load_json("module_9_scenario_modeling.json")
    scenario_projections = pd.read_parquet(
        RESULTS_DIR / "module_9_scenario_projections.parquet"
    )
    nd_data = pd.read_csv(DATA_DIR / "nd_migration_summary.csv")

    # Historical data
    hist_years = nd_data["year"].values
    hist_migration = nd_data["nd_intl_migration"].values
    hist_last_year = int(hist_years.max())
    hist_last_value = float(hist_migration[hist_years == hist_last_year][0])

    # Projection parameters
    proj_years = np.arange(2025, 2046)

    # Extract scenario parameters
    mc_results = scenario_data["results"]["monte_carlo"]
    baseline_mc = pd.read_parquet(RESULTS_DIR / "module_9_monte_carlo_baseline.parquet")
    wave_mc = pd.read_parquet(RESULTS_DIR / "module_9_monte_carlo.parquet")

    # Create figure
    fig, ax = plt.subplots(figsize=(7.25, 5.25))

    # Plot historical data
    scenario_handles: list[Any] = []
    scenario_labels: list[str] = []
    (hist_handle,) = ax.plot(
        hist_years,
        hist_migration,
        "o-",
        color="black",
        linewidth=1.5,
        markersize=4,
        zorder=6,
    )
    scenario_handles.append(hist_handle)
    scenario_labels.append("Historical")

    # Note: the "zero" counterfactual is a flat line at y=0 and provides little
    # additional information in this fan chart. It is intentionally omitted for
    # readability (it remains documented and tabulated elsewhere in the manuscript).
    scenario_order = ["cbo_full", "moderate", "immigration_policy", "pre_2020_trend"]
    scenario_colors = {
        "cbo_full": COLORS["blue"],
        "moderate": COLORS["green"],
        "immigration_policy": COLORS["cyan"],
        "zero": COLORS["gray"],
        "pre_2020_trend": COLORS["orange"],
    }
    scenario_display_names = {
        "immigration_policy": "Restrictive policy (0.65× Moderate)",
    }

    for scenario_code in scenario_order:
        scenario_subset = scenario_projections[
            scenario_projections["scenario"] == scenario_code
        ].sort_values("year")
        if scenario_subset.empty:
            continue

        # A few scenarios are intentionally anchored away from the final historical year
        # (e.g., CBO scaling, pre-2020 counterfactual, and immediate policy multipliers).
        # Draw a subtle connector to make the discontinuity visually explicit.
        first_year = int(scenario_subset["year"].iloc[0])
        first_value = float(scenario_subset["value"].iloc[0])
        if first_year == hist_last_year + 1 and scenario_code in {
            "cbo_full",
            "immigration_policy",
            "pre_2020_trend",
        }:
            ax.plot(
                [hist_last_year, first_year],
                [hist_last_value, first_value],
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                color=scenario_colors.get(scenario_code, COLORS["gray"]),
                label="_nolegend_",
                zorder=4,
            )

        (scenario_handle,) = ax.plot(
            scenario_subset["year"].values,
            scenario_subset["value"].values,
            "-",
            color=scenario_colors.get(scenario_code, COLORS["gray"]),
            linewidth=1.5,
            zorder=5,
        )
        scenario_handles.append(scenario_handle)
        scenario_labels.append(
            scenario_display_names.get(
                scenario_code, str(scenario_subset["scenario_name"].iloc[0])
            )
        )

    # Two-band Monte Carlo uncertainty (year-specific percentiles):
    # - baseline-only PI (inner band)
    # - wave-adjusted envelope (outer band)
    mc_median_baseline = mc_results["baseline_only"]["median_2045"]
    mc_median_wave = mc_results["wave_adjusted"]["median_2045"]
    baseline_mc = baseline_mc.set_index("year").sort_index()
    wave_mc = wave_mc.set_index("year").sort_index()

    missing_baseline_years = [
        int(y) for y in proj_years if int(y) not in baseline_mc.index
    ]
    missing_wave_years = [int(y) for y in proj_years if int(y) not in wave_mc.index]
    if missing_baseline_years:
        raise ValueError(
            "Baseline Monte Carlo percentiles missing years: "
            f"{missing_baseline_years} (expected {int(proj_years[0])}--{int(proj_years[-1])})"
        )
    if missing_wave_years:
        raise ValueError(
            "Wave-adjusted Monte Carlo percentiles missing years: "
            f"{missing_wave_years} (expected {int(proj_years[0])}--{int(proj_years[-1])})"
        )

    # Baseline-only bands: 50% = [p25, p75], 95% = [p5, p95]
    lower_50 = baseline_mc.loc[proj_years, "p25"].to_numpy()
    upper_50 = baseline_mc.loc[proj_years, "p75"].to_numpy()
    lower_95 = baseline_mc.loc[proj_years, "p5"].to_numpy()
    upper_95 = baseline_mc.loc[proj_years, "p95"].to_numpy()
    median_50 = baseline_mc.loc[proj_years, "p50"].to_numpy()

    # Wave-adjusted envelope: 95% = [p5, p95]
    lower_95_wave = wave_mc.loc[proj_years, "p5"].to_numpy()
    upper_95_wave = wave_mc.loc[proj_years, "p95"].to_numpy()

    # Plot uncertainty as a standard fan chart:
    # - baseline-only pointwise prediction intervals (50% and 95%)
    # - baseline median line
    # - wave-adjusted 95% envelope as dashed bounds
    band_95_color = "#D9D9D9"
    band_50_color = "#BDBDBD"
    band_95_alpha = 0.55
    band_50_alpha = 0.75
    median_color = "#4D4D4D"
    wave_color = "#7F7F7F"

    ax.fill_between(
        proj_years,
        lower_95,
        upper_95,
        alpha=band_95_alpha,
        color=band_95_color,
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        proj_years,
        lower_50,
        upper_50,
        alpha=band_50_alpha,
        color=band_50_color,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        proj_years,
        median_50,
        color=median_color,
        linewidth=2,
        zorder=3,
    )
    ax.plot(
        proj_years,
        lower_95_wave,
        color=wave_color,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )
    ax.plot(
        proj_years,
        upper_95_wave,
        color=wave_color,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )

    # Add vertical line at forecast start
    ax.axvline(x=2024.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        2024.7, ax.get_ylim()[1] * 0.95, "Forecast", fontsize=8, rotation=90, va="top"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("International migration (persons)")
    ax.set_title(
        "International Migration Scenarios for North Dakota, 2010-2045",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(2009, 2046)
    ax.set_ylim(bottom=0)
    uncertainty_handles = [
        Line2D(
            [0],
            [0],
            color=median_color,
            linewidth=2,
            label="Baseline median (MC)",
        ),
        Patch(
            facecolor=band_50_color,
            edgecolor="none",
            alpha=band_50_alpha,
            label="50% PI (baseline)",
        ),
        Patch(
            facecolor=band_95_color,
            edgecolor="none",
            alpha=band_95_alpha,
            label="95% PI (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=wave_color,
            linestyle="--",
            linewidth=1.2,
            label="Wave-adjusted 95% envelope",
        ),
    ]

    # Add 2045 endpoint annotations
    endpoints = (
        scenario_projections[scenario_projections["year"] == 2045]
        .set_index("scenario")["value"]
        .to_dict()
    )
    for scenario_code in scenario_order:
        if scenario_code not in endpoints:
            continue
        ax.scatter(
            [2045],
            [endpoints[scenario_code]],
            color=scenario_colors.get(scenario_code, COLORS["gray"]),
            s=30,
            zorder=7,
        )

    # Add Monte Carlo note
    ax.text(
        0.02,
        0.98,
        f"Monte Carlo: n = {mc_results['n_draws']:,}\n"
        f"2045 median (baseline): {mc_median_baseline:,.0f}\n"
        f"2045 median (wave-adj): {mc_median_wave:,.0f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=COLORS["gray"], alpha=0.9
        ),
    )

    combined_handles = scenario_handles + uncertainty_handles
    combined_labels = scenario_labels + [h.get_label() for h in uncertainty_handles]
    ax.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, "fig_08_scenarios")


# =============================================================================
# LaTeX Figure Captions
# =============================================================================


def create_figure_captions() -> None:
    """Create LaTeX file with figure captions."""
    captions = r"""% Figure Captions for Journal Article
% Forecasting International Migration to North Dakota
% Generated: 2025-12-29

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_01_timeseries.pdf}
\caption{International migration to North Dakota, 2010--2024. Panel~(A) shows annual net international migration (persons) with a quadratic trend line. Panel~(B) displays North Dakota's share of total U.S.\ net international migration, with the solid line showing observed values and the dashed line indicating the HP filter trend component ($\lambda = 6.25$). Vertical dotted lines mark candidate break years tested via Chow tests at 2020 (COVID-19 pandemic) and 2021 (post-pandemic period), both significant at $p < 0.01$.}
\label{fig:timeseries}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_02_concentration.pdf}
\caption{Geographic concentration of foreign-born population in North Dakota by origin (2023). Bars show location quotients (LQ), defined as the ratio of an origin's share of North Dakota's foreign-born population to its share of the U.S.\ foreign-born population: $\mathrm{LQ}_i = (FB_{i,\mathrm{ND}}/FB_{\mathrm{ND}})\,/\,(FB_{i,\mathrm{US}}/FB_{\mathrm{US}})$. Values $> 1$ indicate overrepresentation relative to the national average; the dashed vertical line marks LQ $= 1$. Origins are sorted by LQ value; bar intensity reflects population size (darker indicates larger foreign-born population in North Dakota). Numbers at bar ends show the origin-specific foreign-born population in North Dakota (persons). Liberia, Ivory Coast, Somalia, and Tanzania show the highest concentration relative to their national presence.}
\label{fig:concentration}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/fig_03_acf_pacf.pdf}
\caption{Autocorrelation function (ACF) and partial autocorrelation function (PACF) for North Dakota international migration series. Panels~(A) and (B) show the level series, which exhibits slow decay in the ACF consistent with nonstationarity. Panels~(C) and (D) show the first-differenced series, where autocorrelations fall within the 95\% confidence bounds (dashed lines), supporting the characterization of the series as I(1). Sample size $n = 15$.}
\label{fig:acfpacf}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_04_structural_breaks.pdf}
\caption{Structural break diagnostics for North Dakota international migration. Panel~(A) shows annual net international migration with separate linear fits for 2010--2019 and 2020--2024; the vertical dotted line marks the candidate break year (2020). The Chow test rejects equality of pre/post trend parameters at 2020 ($F = 16.01$, $p = 0.0006$). Panel~(B) displays the CUSUM test for parameter stability; the statistic remains within the 95\% bounds (dashed lines), so we do not reject stability at $\alpha = 0.05$. Because the series contains only 15 annual observations, these diagnostics have limited power and should be interpreted cautiously.}
\label{fig:breaks}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_05_gravity.pdf}
\caption{Gravity model estimation results. Panel~(A) shows coefficient estimates with 95\% confidence intervals from the full gravity specification: diaspora stock (log size of an origin's existing community in the destination state), origin mass (log origin stock in the U.S.), and destination mass (log total foreign-born population in the destination state). Panel~(B) compares the estimated diaspora elasticity across model specifications (95\% confidence intervals). Adding mass controls reduces the diaspora point estimate from roughly 0.45 in the diaspora-only model to about 0.14 in the full model; in the full model, a 1\% increase in diaspora stock is associated with an estimated 0.14\% increase in new admissions, but the estimate is imprecise (the 95\% confidence interval includes zero).}
\label{fig:gravity}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_06_event_study.pdf}
\caption{Event study estimates of treated--control divergence around the Travel Ban in refugee arrivals. Coefficients represent the difference in $\log(\mathrm{arrivals} + 1)$ between treated countries (Iran, Iraq, Libya, Somalia, Sudan, Syria, Yemen) and control countries in each year, relative to the reference period ($t = -1$, FY2017); $t = 0$ corresponds to FY2018 (first full post year). Blue circles show pre-treatment estimates; red squares show post-treatment estimates; vertical bars indicate 95\% confidence intervals. The joint pre-treatment test rejects parallel trends over the full pre-period ($F = 4.22$, $p < 0.001$), so post-treatment deviations are interpreted as policy-associated divergence rather than definitive causal effects. A post-period difference-in-differences estimate implies treated-country arrivals were about 75\% lower than controls in the first full post years (ATT $= -1.39$, conventional $p = 0.031$), but this estimate should be interpreted cautiously given the pre-trend violation.}
\label{fig:eventstudy}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_07_survival.pdf}
\caption{Kaplan--Meier survival curves for immigration wave duration by initial intensity quartile. Waves are defined as periods where arrivals exceed 50\% above baseline for at least two consecutive years. Q1 (lowest intensity) waves have median duration of 2 years, while Q4 (highest intensity) waves persist for a median of 6 years. The log-rank test strongly rejects equality across groups ($\chi^2 = 633.0$, $p < 10^{-136}$), indicating that higher-intensity immigration flows are significantly more persistent.}
\label{fig:survival}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_08_cox.pdf}
\caption{Cox proportional hazards model estimates of wave termination risk. Points show hazard ratios and bars show 95\% confidence intervals. Values below 1 indicate lower hazard of termination (more persistent waves), while values above 1 indicate faster termination. Covariates include wave intensity measures, timing, peak arrivals, and origin-region indicators (reference: Africa). State-region controls are included in estimation but omitted from the figure for readability.}
\label{fig:cox}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_08_scenarios.pdf}
\caption{Projection scenarios for North Dakota international migration, 2025--2045. The figure shows four non-zero scenario paths: CBO (Jan 2026) Baseline (CBO national net immigration projection scaled by North Dakota's mean share of U.S.\ net international migration in the \PEP series), Moderate (dampened historical trend), Restrictive policy (0.65$\times$ Moderate), and Pre-2020 Trend (anchored to 2019 with the 2010--2019 slope). The Zero counterfactual (no international migration) is a flat line at $y=0$ and is omitted from the plot for readability but remains part of the scenario set reported in Table~\ref{tab:scenarios}. Shaded bands show baseline-only pointwise prediction intervals from Monte Carlo simulations: the dark band is the central 50\% interval (25th--75th percentiles) and the light band is the central 95\% interval (5th--95th percentiles). The solid gray line is the baseline median. Dashed lines show a wave-adjusted 95\% envelope that adds hazard-based wave persistence draws; this envelope is conservative because wave-driven variation may already be reflected in baseline forecast uncertainty. Because some scenarios are intentionally anchored away from the final historical year (e.g., external CBO scaling, pre-2020 counterfactual anchoring, or immediate policy multipliers), paths may be discontinuous at 2025; dotted connectors indicate the change from the 2024 observation to each scenario's 2025 starting value. Pointwise intervals should not be interpreted as guaranteeing that 95\% of full simulated trajectories lie entirely within the band across all horizons. Historical data (2010--2024) shown with black circles. The vertical dashed line separates historical data from projections.}
\label{fig:scenarios}
\end{figure}
"""

    # Keep captions synchronized with the scenario engine's Monte Carlo draw count.
    try:
        scenario_data = load_json("module_9_scenario_modeling.json")
        n_draws = (
            scenario_data.get("results", {}).get("monte_carlo", {}).get("n_draws", 1000)
        )
        captions = captions.replace(
            "1,000 Monte Carlo simulations", f"{int(n_draws):,} Monte Carlo simulations"
        )
    except Exception:
        # Fall back to the legacy caption if results are not available.
        pass

    output_path = FIGURES_DIR / "figure_captions.tex"
    with open(output_path, "w") as f:
        f.write(captions)
    print(f"Saved: {output_path.name}")


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Generate all publication figures."""
    print("=" * 60)
    print("Creating Publication-Quality Figures")
    print("=" * 60)
    print()

    print("Figure 1: Time Series Overview")
    create_figure_01_timeseries()

    print("\nFigure 2: Geographic Concentration")
    create_figure_02_concentration()

    print("\nFigure 3: ACF/PACF Diagnostics")
    create_figure_03_acf_pacf()

    print("\nFigure 4: Structural Breaks")
    create_figure_04_structural_breaks()

    print("\nFigure 5: Gravity Model")
    create_figure_05_gravity()

    print("\nFigure 6: Event Study")
    create_figure_06_event_study()

    print("\nAppendix Figure: Extended Event Study (Supplemental)")
    create_fig_app_event_study_extended()

    print("\nFigure 7: Survival Curves")
    create_figure_07_survival()

    print("\nFigure 8: Cox Model Hazard Ratios")
    create_figure_08_cox()

    print("\nFigure 9: Forecast Scenarios")
    create_figure_08_scenarios()

    print("\nLaTeX Captions")
    create_figure_captions()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
