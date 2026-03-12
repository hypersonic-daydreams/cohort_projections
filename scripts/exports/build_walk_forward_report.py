#!/usr/bin/env python3
"""
Build interactive HTML report for walk-forward validation results.

Compares two population projection methodologies — SDC 2024 ("sdc_2024")
and our 2026 cohort-component method ("m2026") — using walk-forward
validation from multiple origin years (2005, 2010, 2015, 2020).  Each
method projects forward in 5-year steps, with annual interpolation,
and is compared against actual Census/PEP population data through 2024.

Output: data/analysis/walk_forward/walk_forward_report.html

Usage:
    python scripts/exports/build_walk_forward_report.py
    python scripts/exports/build_walk_forward_report.py --output data/analysis/walk_forward/custom.html

Data Sources:
    Annual resolution (preferred, created by parallel agent):
        - data/analysis/walk_forward/annual_state_results.csv
        - data/analysis/walk_forward/annual_county_detail.csv
        - data/analysis/walk_forward/annual_horizon_summary.csv
        - data/analysis/walk_forward/annual_method_comparison.csv
        - data/analysis/walk_forward/projection_curves.csv
    5-year resolution (fallback):
        - data/analysis/walk_forward/state_results.csv
        - data/analysis/walk_forward/county_detail.csv
        - data/analysis/walk_forward/horizon_summary.csv
        - data/analysis/walk_forward/method_comparison.csv

Processing Steps:
    1. Load annual data files (fall back to 5-year if annual not available)
    2. Build projection curves chart with Bakken boom annotation (Chart 1)
    3. Build state-level error trajectories (Chart 2)
    4. Build average error by horizon (Chart 3)
    5. Build county MAPE heatmap (Chart 4)
    5b. Build county MAPE difference heatmap (Chart 4b)
    6. Build method comparison dashboard with winner cards (Chart 5)
    7. Build direction-of-error analysis (Chart 6)
    8. Build error scatter: SDC vs 2026 per county (Chart 7)
    9. Build county deep-dive with Plotly dropdown selector (Chart 8)
    10. Build method delta visualization (Chart 9)
    11. Build error distribution histograms (Chart 10)
    12. Build key takeaways and methodology summary
    13. Build sortable/filterable raw data tables
    14. Assemble self-contained HTML with 13 tabs and JavaScript interactivity

Key ADRs:
    ADR-057: Rolling-origin backtests

SOP-002 Metadata:
    Author:        Projection team
    Date created:  2026-03-03
    Last modified: 2026-03-03
    Input files:   data/analysis/walk_forward/*.csv
    Output files:  data/analysis/walk_forward/walk_forward_report.html
    Dependencies:  pandas, plotly (CDN)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")

DATA_DIR = PROJECT_ROOT / "data" / "analysis" / "walk_forward"
DEFAULT_OUTPUT = DATA_DIR / "walk_forward_report.html"

# Method display config
METHOD_CONFIG = {
    "sdc_2024": {"label": "SDC 2024", "color": "#1f77b4"},
    "m2026": {"label": "2026 Method", "color": "#d62728"},
}
ACTUAL_COLOR = "#000000"

ORIGIN_YEARS = [2005, 2010, 2015, 2020]
LAST_ACTUAL_YEAR = 2024

# Bakken boom annotation config
BAKKEN_START = 2007
BAKKEN_END = 2014

# Theme constants (matching project convention)
NAVY = "#1F3864"
BLUE = "#0563C1"
DARK_GRAY = "#595959"
MID_GRAY = "#D9D9D9"
LIGHT_GRAY = "#F2F2F2"
WHITE = "#FFFFFF"
GREEN = "#00B050"
ORANGE = "#FF8C00"
FONT_FAMILY = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Oil patch counties strongly affected by Bakken boom
BAKKEN_COUNTIES = [
    "McKenzie", "Williams", "Mountrail", "Dunn", "Stark",
    "Divide", "Burke", "Bowman", "Billings", "Golden Valley",
]


# ===================================================================
# Plotly template
# ===================================================================

def _get_plotly_template() -> str:
    """Register and return the walk-forward report template name."""
    template_name = "walk_forward"
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT_FAMILY, "size": 12, "color": DARK_GRAY},
        title={
            "font": {"family": FONT_FAMILY, "size": 16, "color": NAVY},
            "x": 0.0,
            "xanchor": "left",
        },
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        xaxis={
            "showgrid": True,
            "gridcolor": LIGHT_GRAY,
            "linecolor": MID_GRAY,
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": LIGHT_GRAY,
            "linecolor": MID_GRAY,
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": MID_GRAY,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hoverlabel={
            "bgcolor": WHITE,
            "font_size": 12,
            "font_family": FONT_FAMILY,
        },
        colorway=[
            METHOD_CONFIG["sdc_2024"]["color"],
            METHOD_CONFIG["m2026"]["color"],
            ACTUAL_COLOR,
        ],
    )
    pio.templates[template_name] = template
    return template_name


# ===================================================================
# Formatting helpers
# ===================================================================

def _fmt_pop(n: float | int) -> str:
    """Format a population number with commas."""
    return f"{int(round(n)):,}"


def _fmt_pct(p: float, decimals: int = 1) -> str:
    """Format a percentage with +/- sign."""
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.{decimals}f}%"


def _method_label(method: str) -> str:
    """Return display label for a method key."""
    return METHOD_CONFIG.get(method, {}).get("label", method)


def _method_color(method: str) -> str:
    """Return display color for a method key."""
    return METHOD_CONFIG.get(method, {}).get("color", "#808080")


# ===================================================================
# Data Loading
# ===================================================================

def _load_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    """Load a CSV file, returning None on failure."""
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        logger.exception("Failed to read CSV: %s", path)
        return None


def _load_fips_names() -> dict[int, str]:
    """Load FIPS-to-county-name mapping.

    Returns dict like {38001: 'Adams', 38003: 'Barnes', ...}.
    """
    path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    df = _load_csv_safe(path)
    if df is None:
        return {}
    return dict(zip(df["county_fips"].astype(int), df["county_name"].astype(str)))


def load_data() -> dict[str, pd.DataFrame | None]:
    """Load all walk-forward validation data files.

    Prefers annual-resolution files; falls back to 5-year-interval files.

    Returns
    -------
    dict with keys:
        state_results, county_detail, horizon_summary, method_comparison,
        projection_curves, data_mode ('annual' or '5year')
    """
    data: dict[str, Any] = {}

    # Try annual files first
    annual_state = _load_csv_safe(DATA_DIR / "annual_state_results.csv")
    annual_county = _load_csv_safe(DATA_DIR / "annual_county_detail.csv")
    annual_horizon = _load_csv_safe(DATA_DIR / "annual_horizon_summary.csv")
    annual_comparison = _load_csv_safe(DATA_DIR / "annual_method_comparison.csv")

    if annual_state is not None:
        logger.info("Using annual-resolution data files.")
        data["state_results"] = annual_state
        data["county_detail"] = annual_county
        data["horizon_summary"] = annual_horizon
        data["method_comparison"] = annual_comparison
        data["data_mode"] = "annual"
    else:
        logger.info(
            "Annual data files not found; falling back to 5-year-interval data."
        )
        data["state_results"] = _load_csv_safe(DATA_DIR / "state_results.csv")
        data["county_detail"] = _load_csv_safe(DATA_DIR / "county_detail.csv")
        data["horizon_summary"] = _load_csv_safe(DATA_DIR / "horizon_summary.csv")
        data["method_comparison"] = _load_csv_safe(DATA_DIR / "method_comparison.csv")
        data["data_mode"] = "5year"

    # Projection curves (only available if annual agent has run)
    data["projection_curves"] = _load_csv_safe(DATA_DIR / "projection_curves.csv")

    # FIPS mapping
    data["fips_names"] = _load_fips_names()

    return data


# ===================================================================
# Chart 1: Projection Curves vs Actuals (State Level)
# ===================================================================

def _build_projection_curves(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 1: Projection curves vs actual population for each origin year.

    If projection_curves.csv is available, uses those full 50-year curves.
    Otherwise, plots the validated data points from state_results.
    """
    parts = ["<h2>Projection Curves vs. Actual Population (State Level)</h2>"]
    parts.append(
        '<p class="note">Each panel shows projections from a different origin year. '
        "Solid lines show the validated portion (through 2024); "
        "dashed lines show future projections. "
        "The black dashed line is actual Census/PEP population.</p>"
    )

    state_df = data.get("state_results")
    curves_df = data.get("projection_curves")

    if state_df is None:
        parts.append('<p class="placeholder">State results data not available.</p>')
        return "\n".join(parts)

    origins = sorted(state_df["origin_year"].unique())
    n_origins = len(origins)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Origin: {yr}" for yr in origins]
        + [""] * (4 - n_origins),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, origin in enumerate(origins):
        row = idx // 2 + 1
        col = idx % 2 + 1
        show_legend = idx == 0

        # Get actual population values from state_results
        origin_state = state_df[state_df["origin_year"] == origin]

        # Plot actual population line
        actual_points = (
            origin_state[origin_state["method"] == origin_state["method"].iloc[0]]
            .sort_values("validation_year")
        )
        actual_years = [origin] + list(actual_points["validation_year"])
        # For origin year, we can infer actual from projected - error at origin
        # but more simply, the actual population at origin is the starting pop
        actual_vals_from_data = list(actual_points["actual_state"])
        # Get actual at origin: projected - error for first point projected backward,
        # or just use the actual from the first validation year data
        if curves_df is not None:
            # Try to get the origin year actual from curves
            origin_curve = curves_df[
                (curves_df["origin_year"] == origin)
                & (curves_df["method"] == "sdc_2024")
            ]
            if not origin_curve.empty:
                origin_pop = origin_curve[
                    origin_curve["year"] == origin
                ]["projected_state"]
                if not origin_pop.empty:
                    actual_years_full = [origin] + list(actual_points["validation_year"])
                    actual_vals_full = [float(origin_pop.iloc[0])] + actual_vals_from_data
                else:
                    actual_years_full = list(actual_points["validation_year"])
                    actual_vals_full = actual_vals_from_data
            else:
                actual_years_full = list(actual_points["validation_year"])
                actual_vals_full = actual_vals_from_data
        else:
            actual_years_full = list(actual_points["validation_year"])
            actual_vals_full = actual_vals_from_data

        fig.add_trace(
            go.Scatter(
                x=actual_years_full,
                y=actual_vals_full,
                name="Actual (Census/PEP)",
                mode="lines+markers",
                line={"color": ACTUAL_COLOR, "width": 2, "dash": "dash"},
                marker={"size": 6, "symbol": "circle"},
                legendgroup="actual",
                showlegend=show_legend,
                hovertemplate=(
                    "<b>Actual</b><br>Year: %{x}<br>"
                    "Population: %{y:,.0f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Plot projection curves for each method
        for method in ["sdc_2024", "m2026"]:
            color = _method_color(method)
            label = _method_label(method)

            if curves_df is not None:
                # Use full projection curves
                method_curve = curves_df[
                    (curves_df["origin_year"] == origin)
                    & (curves_df["method"] == method)
                ].sort_values("year")

                if not method_curve.empty:
                    # Split into validated (through 2024) and future
                    validated = method_curve[
                        method_curve["year"] <= LAST_ACTUAL_YEAR
                    ]
                    future = method_curve[
                        method_curve["year"] >= LAST_ACTUAL_YEAR
                    ]

                    # Validated portion (solid)
                    if not validated.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=list(validated["year"]),
                                y=list(validated["projected_state"]),
                                name=f"{label}",
                                mode="lines",
                                line={"color": color, "width": 2.5},
                                legendgroup=method,
                                showlegend=show_legend,
                                hovertemplate=(
                                    f"<b>{label}</b><br>Year: %{{x}}<br>"
                                    f"Projected: %{{y:,.0f}}<extra></extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )

                    # Future portion (dashed)
                    if not future.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=list(future["year"]),
                                y=list(future["projected_state"]),
                                name=f"{label} (future)",
                                mode="lines",
                                line={
                                    "color": color,
                                    "width": 2,
                                    "dash": "dot",
                                },
                                legendgroup=method,
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{label} (projection)</b><br>"
                                    f"Year: %{{x}}<br>"
                                    f"Projected: %{{y:,.0f}}<extra></extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )
                    continue

            # Fallback: use state_results data points
            method_data = origin_state[
                origin_state["method"] == method
            ].sort_values("validation_year")

            if not method_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=list(method_data["validation_year"]),
                        y=list(method_data["projected_state"]),
                        name=label,
                        mode="lines+markers",
                        line={"color": color, "width": 2.5},
                        marker={"size": 5},
                        legendgroup=method,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{label}</b><br>Year: %{{x}}<br>"
                            f"Projected: %{{y:,.0f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    # Add Bakken boom shaded region to each subplot
    for idx, origin in enumerate(origins):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.add_vrect(
            x0=BAKKEN_START,
            x1=BAKKEN_END,
            fillcolor="rgba(255, 165, 0, 0.08)",
            line_width=0,
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=(BAKKEN_START + BAKKEN_END) / 2,
            y=1.0,
            yref=f"y{idx + 1 if idx > 0 else ''} domain",
            text="Bakken Boom",
            showarrow=False,
            font={"size": 9, "color": "#CC7700"},
            xanchor="center",
            yanchor="top",
            row=row,
            col=col,
        )

    fig.update_layout(
        template=template_name,
        height=700,
        title_text="Walk-Forward Projection Curves vs. Actual Population",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.update_yaxes(tickformat=",", title_text="Population")
    fig.update_xaxes(title_text="Year")

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 2: State-Level Error Trajectory by Origin
# ===================================================================

def _build_error_trajectory(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 2: % error trajectory by horizon for each origin year."""
    parts = ["<h2>State-Level Error Trajectory by Origin Year</h2>"]
    parts.append(
        '<p class="note">Signed percentage error at the state level over increasing '
        "forecast horizons. The shaded band shows the +/-5% acceptable zone. "
        "Negative values indicate under-projection.</p>"
    )

    state_df = data.get("state_results")
    if state_df is None:
        parts.append('<p class="placeholder">State results data not available.</p>')
        return "\n".join(parts)

    origins = sorted(state_df["origin_year"].unique())
    n_origins = len(origins)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Origin: {yr}" for yr in origins]
        + [""] * (4 - n_origins),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, origin in enumerate(origins):
        row = idx // 2 + 1
        col = idx % 2 + 1
        show_legend = idx == 0

        origin_data = state_df[state_df["origin_year"] == origin]
        max_horizon = int(origin_data["horizon"].max()) if not origin_data.empty else 20

        # Add shaded acceptable zone (+/-5%)
        fig.add_trace(
            go.Scatter(
                x=list(range(0, max_horizon + 2)),
                y=[5] * (max_horizon + 2),
                fill=None,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(0, max_horizon + 2)),
                y=[-5] * (max_horizon + 2),
                fill="tonexty",
                mode="lines",
                line={"width": 0},
                fillcolor="rgba(0, 176, 80, 0.08)",
                showlegend=show_legend,
                name="Acceptable (+/-5%)",
                legendgroup="acceptable",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add zero reference line
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color=MID_GRAY,
            line_width=1,
            row=row,
            col=col,
        )

        # Plot error lines for each method
        for method in ["sdc_2024", "m2026"]:
            method_data = origin_data[
                origin_data["method"] == method
            ].sort_values("horizon")
            if method_data.empty:
                continue

            color = _method_color(method)
            label = _method_label(method)

            fig.add_trace(
                go.Scatter(
                    x=list(method_data["horizon"]),
                    y=list(method_data["pct_error"]),
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.5},
                    marker={"size": 6},
                    legendgroup=method,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{label}</b><br>Horizon: %{{x}} yrs<br>"
                        f"Error: %{{y:.1f}}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        template=template_name,
        height=700,
        title_text="State-Level Percentage Error by Forecast Horizon",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.update_yaxes(title_text="% Error (signed)")
    fig.update_xaxes(title_text="Forecast Horizon (years)")

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 3: Average Error by Forecast Horizon
# ===================================================================

def _build_avg_error_by_horizon(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 3: Mean Absolute Percentage Error by horizon, averaged across origins."""
    parts = ["<h2>Average Error by Forecast Horizon</h2>"]
    parts.append(
        '<p class="note">Mean Absolute Percentage Error (MAPE) at the state level, '
        "averaged across all applicable origin years for each horizon. "
        "Lower is better.</p>"
    )

    horizon_df = data.get("horizon_summary")
    if horizon_df is None:
        parts.append(
            '<p class="placeholder">Horizon summary data not available.</p>'
        )
        return "\n".join(parts)

    fig = go.Figure()

    for method in ["sdc_2024", "m2026"]:
        method_data = horizon_df[horizon_df["method"] == method].sort_values("horizon")
        if method_data.empty:
            continue

        color = _method_color(method)
        label = _method_label(method)

        fig.add_trace(
            go.Bar(
                x=list(method_data["horizon"]),
                y=list(method_data["mean_state_ape"]),
                name=label,
                marker_color=color,
                opacity=0.85,
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yrs<br>"
                    f"MAPE: %{{y:.2f}}%<br>"
                    f"Origins: %{{customdata[0]}}<extra></extra>"
                ),
                customdata=list(
                    zip(method_data["n_origins"])
                ),
            )
        )

    # Determine winner annotations
    horizons_all = sorted(horizon_df["horizon"].unique())
    annotations = []
    for h in horizons_all:
        h_data = horizon_df[horizon_df["horizon"] == h]
        sdc_val = h_data[h_data["method"] == "sdc_2024"]["mean_state_ape"]
        m26_val = h_data[h_data["method"] == "m2026"]["mean_state_ape"]
        if not sdc_val.empty and not m26_val.empty:
            sdc_v = float(sdc_val.iloc[0])
            m26_v = float(m26_val.iloc[0])
            winner = "SDC" if sdc_v < m26_v else "2026"
            diff = abs(sdc_v - m26_v)
            max_val = max(sdc_v, m26_v)
            annotations.append(
                {
                    "x": h,
                    "y": max_val + 1.5,
                    "text": f"{winner}",
                    "showarrow": False,
                    "font": {
                        "size": 9,
                        "color": _method_color(
                            "sdc_2024" if winner == "SDC" else "m2026"
                        ),
                    },
                }
            )

    fig.update_layout(
        template=template_name,
        title="Mean Absolute Percentage Error by Forecast Horizon",
        xaxis_title="Forecast Horizon (years)",
        yaxis_title="State-Level MAPE (%)",
        barmode="group",
        height=450,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        annotations=annotations,
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 4: County MAPE Heatmap
# ===================================================================

def _build_county_heatmap(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 4: County MAPE heatmaps (SDC vs 2026), top 20 counties."""
    parts = ["<h2>County-Level Error Heatmap</h2>"]
    parts.append(
        '<p class="note">Mean Absolute Percentage Error for each county across '
        "origin/horizon combinations.  Only the 20 counties with the highest "
        "average error are shown. Counties are sorted by their overall MAPE "
        "(worst at top).</p>"
    )

    county_df = data.get("county_detail")
    fips_names = data.get("fips_names", {})

    if county_df is None:
        parts.append(
            '<p class="placeholder">County detail data not available.</p>'
        )
        return "\n".join(parts)

    # Add county names
    county_df = county_df.copy()
    county_df["county_name"] = county_df["county_fips"].map(fips_names)
    county_df["county_name"] = county_df["county_name"].fillna(
        county_df["county_fips"].astype(str)
    )
    county_df["abs_pct_error"] = county_df["pct_error"].abs()

    # Build pivot: rows=county, cols=horizon, values=mean abs pct error
    # We'll make two side-by-side heatmaps
    methods = ["sdc_2024", "m2026"]

    # Rank counties by overall mean APE
    county_avg = (
        county_df.groupby("county_name")["abs_pct_error"]
        .mean()
        .sort_values(ascending=True)
    )
    top_counties = list(county_avg.tail(20).index)  # worst 20

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{_method_label(m)} - County MAPE by Horizon" for m in methods
        ],
        horizontal_spacing=0.12,
    )

    for col_idx, method in enumerate(methods, 1):
        method_data = county_df[
            (county_df["method"] == method)
            & (county_df["county_name"].isin(top_counties))
        ]

        pivot = method_data.pivot_table(
            index="county_name",
            columns="horizon",
            values="abs_pct_error",
            aggfunc="mean",
        )

        # Reindex to sorted order (worst at top)
        sorted_counties = [c for c in reversed(top_counties) if c in pivot.index]
        pivot = pivot.reindex(sorted_counties)

        horizons = sorted(pivot.columns)
        counties = list(pivot.index)

        z_vals = []
        hover_text = []
        for county in counties:
            row_vals = []
            row_hover = []
            for h in horizons:
                val = pivot.loc[county, h] if h in pivot.columns else None
                if pd.notna(val):
                    row_vals.append(round(float(val), 1))
                    row_hover.append(
                        f"{county}<br>Horizon: {h} yrs<br>MAPE: {val:.1f}%"
                    )
                else:
                    row_vals.append(None)
                    row_hover.append(f"{county}<br>Horizon: {h} yrs<br>No data")
            z_vals.append(row_vals)
            hover_text.append(row_hover)

        fig.add_trace(
            go.Heatmap(
                z=z_vals,
                x=[str(h) for h in horizons],
                y=counties,
                colorscale="YlOrRd",
                zmin=0,
                zmax=40,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorbar={
                    "title": "MAPE (%)",
                    "x": 0.45 if col_idx == 1 else 1.0,
                    "len": 0.8,
                },
                showscale=col_idx == 2,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template=template_name,
        height=max(500, 25 * len(top_counties) + 150),
        title_text="County MAPE Heatmap: SDC 2024 vs 2026 Method (Top 20 Counties)",
    )
    fig.update_xaxes(title_text="Forecast Horizon (years)")

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 4b: County MAPE Difference Heatmap
# ===================================================================

def _build_county_diff_heatmap(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 4b: Difference heatmap (m2026 MAPE minus SDC MAPE).

    Shows which method has lower error for each county/horizon combination
    using a diverging colorscale: blue = m2026 better (negative difference),
    red = SDC better (positive difference), white = no difference.
    """
    parts = ["<h2>County Error Difference Heatmap</h2>"]
    parts.append(
        '<p class="note">Difference in Mean Absolute Percentage Error: '
        "<b>m2026 MAPE &minus; SDC MAPE</b>.  "
        '<span style="color:#1f77b4;font-weight:600;">Blue</span> = 2026 method '
        "has lower error (better), "
        '<span style="color:#d62728;font-weight:600;">Red</span> = SDC 2024 has '
        "lower error (better), "
        "White = no meaningful difference.  "
        "Counties are sorted by overall average difference "
        "(most improved by m2026 at top).</p>"
    )

    county_df = data.get("county_detail")
    fips_names = data.get("fips_names", {})

    if county_df is None:
        parts.append(
            '<p class="placeholder">County detail data not available.</p>'
        )
        return "\n".join(parts)

    county_df = county_df.copy()
    county_df["county_name"] = county_df["county_fips"].map(fips_names)
    county_df["county_name"] = county_df["county_name"].fillna(
        county_df["county_fips"].astype(str)
    )
    county_df["abs_pct_error"] = county_df["pct_error"].abs()

    # Pivot each method separately: rows=county, cols=horizon, values=mean APE
    methods = ["sdc_2024", "m2026"]
    pivots: dict[str, pd.DataFrame] = {}
    for method in methods:
        mdf = county_df[county_df["method"] == method]
        pivots[method] = mdf.pivot_table(
            index="county_name",
            columns="horizon",
            values="abs_pct_error",
            aggfunc="mean",
        )

    # Compute difference: positive means SDC is better, negative means m2026 is better
    diff = pivots["m2026"].reindex_like(pivots["sdc_2024"]) - pivots["sdc_2024"]

    # Sort counties by overall mean difference (most negative = m2026 wins most at top)
    county_mean_diff = diff.mean(axis=1).sort_values(ascending=False)
    sorted_counties = list(county_mean_diff.index)

    diff = diff.reindex(sorted_counties)
    horizons = sorted(diff.columns)
    counties = list(diff.index)

    # Build z-values and hover text
    z_vals: list[list[float | None]] = []
    hover_text: list[list[str]] = []
    for county in counties:
        row_vals: list[float | None] = []
        row_hover: list[str] = []
        for h in horizons:
            d_val = diff.loc[county, h] if h in diff.columns else None
            m2026_val = (
                pivots["m2026"].loc[county, h]
                if county in pivots["m2026"].index and h in pivots["m2026"].columns
                else None
            )
            sdc_val = (
                pivots["sdc_2024"].loc[county, h]
                if county in pivots["sdc_2024"].index and h in pivots["sdc_2024"].columns
                else None
            )

            if pd.notna(d_val):
                row_vals.append(round(float(d_val), 2))
                winner = (
                    "2026 Method better"
                    if d_val < -0.5
                    else "SDC 2024 better"
                    if d_val > 0.5
                    else "Similar"
                )
                hover_parts = [
                    f"<b>{county}</b>",
                    f"Horizon: {h} yrs",
                    f"m2026 MAPE: {m2026_val:.1f}%" if pd.notna(m2026_val) else "m2026: N/A",
                    f"SDC MAPE: {sdc_val:.1f}%" if pd.notna(sdc_val) else "SDC: N/A",
                    f"Difference: {d_val:+.1f}pp",
                    f"<i>{winner}</i>",
                ]
                row_hover.append("<br>".join(hover_parts))
            else:
                row_vals.append(None)
                row_hover.append(f"<b>{county}</b><br>Horizon: {h} yrs<br>No data")

        z_vals.append(row_vals)
        hover_text.append(row_hover)

    # Determine symmetric color range
    flat_vals = [v for row in z_vals for v in row if v is not None]
    if flat_vals:
        max_abs = max(abs(min(flat_vals)), abs(max(flat_vals)))
        # Cap at 30 for readability, round up to nearest 5
        max_abs = min(max_abs, 30.0)
        max_abs = max(max_abs, 1.0)
    else:
        max_abs = 10.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=[str(h) for h in horizons],
            y=counties,
            colorscale=[
                [0.0, "#08519c"],     # strong blue (m2026 much better)
                [0.25, "#6baed6"],    # light blue
                [0.5, "#ffffff"],     # white (no difference)
                [0.75, "#fb6a4a"],    # light red
                [1.0, "#a50f15"],     # strong red (SDC much better)
            ],
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={
                "title": {
                    "text": "MAPE Diff (pp)<br>m2026 − SDC",
                    "font": {"size": 11},
                },
                "ticksuffix": "pp",
                "len": 0.8,
            },
        ),
    )

    fig.update_layout(
        template=template_name,
        height=max(600, 22 * len(counties) + 150),
        title_text=(
            "County MAPE Difference: m2026 − SDC 2024"
            "<br><sup>Blue = 2026 Method better | Red = SDC better</sup>"
        ),
        xaxis_title="Forecast Horizon (years)",
        yaxis={"dtick": 1},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 5: Method Comparison Dashboard
# ===================================================================

def _build_comparison_dashboard(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 5: Summary dashboard with winner cards and comparison table."""
    parts = ["<h2>Method Comparison Dashboard</h2>"]
    parts.append(
        '<p class="note">Summary of which method performs better across '
        "different forecast horizon ranges, plus identification of "
        "horizons where both methods struggle (>10% state error).</p>"
    )

    horizon_df = data.get("horizon_summary")
    comparison_df = data.get("method_comparison")

    if horizon_df is None:
        parts.append(
            '<p class="placeholder">Horizon summary data not available.</p>'
        )
        return "\n".join(parts)

    # Classify horizons into short/medium/long
    horizon_ranges = {
        "Short (1-5 years)": (1, 5),
        "Medium (6-10 years)": (6, 10),
        "Long (11+ years)": (11, 50),
    }

    cards_html = ['<div class="dashboard-cards">']

    for range_label, (lo, hi) in horizon_ranges.items():
        range_data = horizon_df[
            (horizon_df["horizon"] >= lo) & (horizon_df["horizon"] <= hi)
        ]
        if range_data.empty:
            cards_html.append(
                f'<div class="card">'
                f'<div class="card-title">{range_label}</div>'
                f'<div class="card-value">No data</div>'
                f"</div>"
            )
            continue

        sdc_avg = range_data[range_data["method"] == "sdc_2024"][
            "mean_state_ape"
        ].mean()
        m26_avg = range_data[range_data["method"] == "m2026"][
            "mean_state_ape"
        ].mean()

        if pd.isna(sdc_avg) or pd.isna(m26_avg):
            winner = "Insufficient data"
            winner_color = DARK_GRAY
        elif sdc_avg < m26_avg:
            winner = "SDC 2024"
            winner_color = _method_color("sdc_2024")
        elif m26_avg < sdc_avg:
            winner = "2026 Method"
            winner_color = _method_color("m2026")
        else:
            winner = "Tie"
            winner_color = DARK_GRAY

        sdc_str = f"{sdc_avg:.1f}%" if pd.notna(sdc_avg) else "N/A"
        m26_str = f"{m26_avg:.1f}%" if pd.notna(m26_avg) else "N/A"

        cards_html.append(
            f'<div class="card">'
            f'<div class="card-title">{range_label}</div>'
            f'<div class="card-value" style="color:{winner_color};">{winner}</div>'
            f'<div class="card-detail">'
            f'SDC: {sdc_str} &nbsp;|&nbsp; 2026: {m26_str}'
            f"</div></div>"
        )

    # Struggle card: identify horizons where both methods exceed 10% state MAPE
    if comparison_df is not None and "both_struggle" in comparison_df.columns:
        # Non-annual format with both_struggle column
        struggle_rows = comparison_df[comparison_df["both_struggle"] == True]  # noqa: E712
        n_struggle = len(struggle_rows)
        total = len(comparison_df)
        cards_html.append(
            f'<div class="card card-warning">'
            f'<div class="card-title">Both Struggle (&gt;10% state error)</div>'
            f'<div class="card-value">{n_struggle} of {total} cases</div>'
            f'<div class="card-detail">'
            f"Predominantly from early origins (pre-Bakken boom)"
            f"</div></div>"
        )
    elif comparison_df is not None and "sdc_state_ape" in comparison_df.columns:
        # Annual format — compute struggle from APE columns
        struggle_mask = (
            (comparison_df["sdc_state_ape"] > 10)
            & (comparison_df["m2026_state_ape"] > 10)
        )
        n_struggle = int(struggle_mask.sum())
        total = len(comparison_df)
        cards_html.append(
            f'<div class="card card-warning">'
            f'<div class="card-title">Both Struggle (&gt;10% state error)</div>'
            f'<div class="card-value">{n_struggle} of {total} horizons</div>'
            f'<div class="card-detail">'
            f"Predominantly at longer horizons from early origins"
            f"</div></div>"
        )

    cards_html.append("</div>")
    parts.append("\n".join(cards_html))

    # Build the comparison table — handle both annual and non-annual formats
    if comparison_df is not None:
        if "origin_year" in comparison_df.columns:
            # Non-annual format: one row per origin x validation_year
            parts.append("<h3>Detailed Comparison by Origin and Horizon</h3>")
            parts.append('<div class="table-container">')
            parts.append('<table class="data-table" id="tbl-comparison">')
            parts.append(
                "<thead><tr>"
                "<th onclick=\"sortTable('tbl-comparison',0)\">Origin</th>"
                "<th onclick=\"sortTable('tbl-comparison',1)\">Validation</th>"
                "<th onclick=\"sortTable('tbl-comparison',2)\">Horizon</th>"
                '<th class="num" onclick="sortTable(\'tbl-comparison\',3)">SDC State %Err</th>'
                '<th class="num" onclick="sortTable(\'tbl-comparison\',4)">2026 State %Err</th>'
                '<th class="num" onclick="sortTable(\'tbl-comparison\',5)">SDC County MAPE</th>'
                '<th class="num" onclick="sortTable(\'tbl-comparison\',6)">2026 County MAPE</th>'
                "<th onclick=\"sortTable('tbl-comparison',7)\">Both Struggle?</th>"
                "</tr></thead><tbody>"
            )

            for _, row in comparison_df.iterrows():
                origin = int(row["origin_year"])
                val_yr = int(row["validation_year"])
                horizon = int(row["horizon"])

                sdc_err = float(row.get("sdc_state_pct_error", 0.0))
                m26_err = float(row.get("m2026_state_pct_error", 0.0))
                sdc_mape = float(row.get("sdc_county_mape", 0.0))
                m26_mape = float(row.get("m2026_county_mape", 0.0))
                struggle = bool(row.get("both_struggle", False))

                sdc_cls = "negative" if sdc_err < -5 else ("positive" if sdc_err > 5 else "")
                m26_cls = "negative" if m26_err < -5 else ("positive" if m26_err > 5 else "")
                struggle_text = "Yes" if struggle else "No"
                struggle_cls = "negative" if struggle else ""

                parts.append(
                    f"<tr>"
                    f"<td>{origin}</td>"
                    f"<td>{val_yr}</td>"
                    f"<td>{horizon}</td>"
                    f'<td class="num {sdc_cls}">{_fmt_pct(sdc_err)}</td>'
                    f'<td class="num {m26_cls}">{_fmt_pct(m26_err)}</td>'
                    f'<td class="num">{sdc_mape:.1f}%</td>'
                    f'<td class="num">{m26_mape:.1f}%</td>'
                    f'<td class="{struggle_cls}">{struggle_text}</td>'
                    f"</tr>"
                )
            parts.append("</tbody></table></div>")

        else:
            # Annual format: one row per horizon (aggregated across origins)
            parts.append("<h3>Comparison by Forecast Horizon</h3>")
            parts.append('<div class="table-container">')
            parts.append('<table class="data-table" id="tbl-horizon-comp">')
            parts.append(
                "<thead><tr>"
                "<th onclick=\"sortTable('tbl-horizon-comp',0)\">Horizon</th>"
                "<th onclick=\"sortTable('tbl-horizon-comp',1)\">Origins</th>"
                '<th class="num" onclick="sortTable(\'tbl-horizon-comp\',2)">SDC State MAPE</th>'
                '<th class="num" onclick="sortTable(\'tbl-horizon-comp\',3)">2026 State MAPE</th>'
                '<th class="num" onclick="sortTable(\'tbl-horizon-comp\',4)">SDC County MAPE</th>'
                '<th class="num" onclick="sortTable(\'tbl-horizon-comp\',5)">2026 County MAPE</th>'
                "<th onclick=\"sortTable('tbl-horizon-comp',6)\">State Winner</th>"
                "<th onclick=\"sortTable('tbl-horizon-comp',7)\">County Winner</th>"
                "</tr></thead><tbody>"
            )

            for _, row in comparison_df.sort_values("horizon").iterrows():
                horizon = int(row["horizon"])
                n_origins = int(row.get("n_origins", 0))
                sdc_ape = float(row.get("sdc_state_ape", 0.0))
                m26_ape = float(row.get("m2026_state_ape", 0.0))
                sdc_mape = float(row.get("sdc_county_mape", 0.0))
                m26_mape = float(row.get("m2026_county_mape", 0.0))
                state_winner = str(row.get("winner_state", ""))
                county_winner = str(row.get("winner_county", ""))

                state_w_label = _method_label(state_winner) if state_winner in METHOD_CONFIG else state_winner.title()
                county_w_label = _method_label(county_winner) if county_winner in METHOD_CONFIG else county_winner.title()
                state_w_color = _method_color(state_winner) if state_winner in METHOD_CONFIG else DARK_GRAY
                county_w_color = _method_color(county_winner) if county_winner in METHOD_CONFIG else DARK_GRAY

                parts.append(
                    f"<tr>"
                    f"<td>{horizon}</td>"
                    f"<td>{n_origins}</td>"
                    f'<td class="num">{sdc_ape:.2f}%</td>'
                    f'<td class="num">{m26_ape:.2f}%</td>'
                    f'<td class="num">{sdc_mape:.2f}%</td>'
                    f'<td class="num">{m26_mape:.2f}%</td>'
                    f'<td style="color:{state_w_color}; font-weight:600;">{state_w_label}</td>'
                    f'<td style="color:{county_w_color}; font-weight:600;">{county_w_label}</td>'
                    f"</tr>"
                )
            parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Chart 6: Direction of Error Analysis
# ===================================================================

def _build_direction_analysis(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 6: Mean Percentage Error (signed) showing systematic bias."""
    parts = ["<h2>Direction of Error Analysis</h2>"]
    parts.append(
        '<p class="note">Mean Percentage Error (signed, averaged across origins) '
        "showing whether methods systematically over- or under-project. "
        "Negative values mean under-projection. "
        "This chart reveals systematic bias in both methods.</p>"
    )

    horizon_df = data.get("horizon_summary")
    if horizon_df is None:
        parts.append(
            '<p class="placeholder">Horizon summary data not available.</p>'
        )
        return "\n".join(parts)

    fig = go.Figure()

    # Zero reference line
    horizons_all = sorted(horizon_df["horizon"].unique())
    fig.add_hline(y=0, line_dash="solid", line_color=MID_GRAY, line_width=1)

    # Add shaded zone
    fig.add_hrect(
        y0=-5,
        y1=5,
        fillcolor="rgba(0, 176, 80, 0.08)",
        line_width=0,
        annotation_text="Acceptable range",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#999",
    )

    for method in ["sdc_2024", "m2026"]:
        method_data = horizon_df[horizon_df["method"] == method].sort_values("horizon")
        if method_data.empty:
            continue

        color = _method_color(method)
        label = _method_label(method)

        fig.add_trace(
            go.Scatter(
                x=list(method_data["horizon"]),
                y=list(method_data["mean_state_mpe"]),
                name=f"{label} (State MPE)",
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yrs<br>"
                    f"State MPE: %{{y:.1f}}%<extra></extra>"
                ),
            )
        )

        # Also add county MPE as lighter line
        if "mean_county_mpe" in method_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(method_data["horizon"]),
                    y=list(method_data["mean_county_mpe"]),
                    name=f"{label} (County avg MPE)",
                    mode="lines+markers",
                    line={"color": color, "width": 1.5, "dash": "dot"},
                    marker={"size": 5, "symbol": "diamond"},
                    hovertemplate=(
                        f"<b>{label} Counties</b><br>Horizon: %{{x}} yrs<br>"
                        f"County avg MPE: %{{y:.1f}}%<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template=template_name,
        title="Direction of Error: Mean Percentage Error by Forecast Horizon",
        xaxis_title="Forecast Horizon (years)",
        yaxis_title="Mean Percentage Error (%, signed)",
        height=500,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # Add interpretation note
    parts.append(
        '<div class="interpretation">'
        "<h3>Interpretation</h3>"
        "<ul>"
        "<li><strong>Both methods under-project</strong> (negative MPE) at all "
        "horizons, reflecting the unexpected Bakken oil boom that drove rapid "
        "population growth from 2005-2015.</li>"
        "<li>The bias grows with horizon length, reaching -25% or more at 19-year "
        "horizons from the 2005 origin.</li>"
        "<li>For more recent origins (2015, 2020), the bias is dramatically "
        "smaller, suggesting both methods perform well when not confronted with "
        "unprecedented structural economic shifts.</li>"
        "</ul></div>"
    )

    return "\n".join(parts)


# ===================================================================
# Chart 7: Scatter Plot — SDC Error vs 2026 Error per County
# ===================================================================

def _build_scatter_comparison(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 7: Scatter plot comparing error of each method per county x horizon.

    Points above the diagonal mean 2026 method had higher error; below means SDC had
    higher error.  Color-coded by origin year.
    """
    parts = ["<h2>Method Comparison: County Error Scatter</h2>"]
    parts.append(
        '<p class="note">Each point is one county at one origin-year and horizon. '
        "Points <b>below</b> the diagonal mean the 2026 method had lower absolute error "
        "than SDC; points above mean SDC was better. "
        "Use the origin-year buttons to filter.</p>"
    )

    county_df = data.get("county_detail")
    if county_df is None:
        parts.append('<p class="placeholder">County detail data not available.</p>')
        return "\n".join(parts)

    df = county_df.copy()
    df["abs_pct_error"] = df["pct_error"].abs()

    # Pivot so we have sdc and m2026 side by side per county x origin x horizon
    sdc = df[df["method"] == "sdc_2024"][
        ["origin_year", "horizon", "county_name", "abs_pct_error"]
    ].rename(columns={"abs_pct_error": "sdc_ape"})
    m26 = df[df["method"] == "m2026"][
        ["origin_year", "horizon", "county_name", "abs_pct_error"]
    ].rename(columns={"abs_pct_error": "m2026_ape"})

    merged = pd.merge(
        sdc, m26,
        on=["origin_year", "horizon", "county_name"],
        how="inner",
    )

    if merged.empty:
        parts.append('<p class="placeholder">Insufficient data for scatter plot.</p>')
        return "\n".join(parts)

    origin_colors = {
        2005: "#e41a1c",
        2010: "#377eb8",
        2015: "#4daf4a",
        2020: "#984ea3",
    }

    fig = go.Figure()

    # Add diagonal reference line
    max_val = max(merged["sdc_ape"].max(), merged["m2026_ape"].max()) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line={"color": MID_GRAY, "dash": "dash", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add shaded regions with labels
    fig.add_annotation(
        x=max_val * 0.85,
        y=max_val * 0.15,
        text="2026 Better",
        showarrow=False,
        font={"size": 12, "color": _method_color("m2026")},
        opacity=0.5,
    )
    fig.add_annotation(
        x=max_val * 0.15,
        y=max_val * 0.85,
        text="SDC Better",
        showarrow=False,
        font={"size": 12, "color": _method_color("sdc_2024")},
        opacity=0.5,
    )

    origins = sorted(merged["origin_year"].unique())
    for origin in origins:
        od = merged[merged["origin_year"] == origin]
        is_bakken = od["county_name"].isin(BAKKEN_COUNTIES)
        color = origin_colors.get(origin, "#808080")

        fig.add_trace(
            go.Scatter(
                x=list(od["sdc_ape"]),
                y=list(od["m2026_ape"]),
                mode="markers",
                marker={
                    "size": 5,
                    "color": color,
                    "opacity": 0.6,
                    "symbol": [
                        "diamond" if b else "circle" for b in is_bakken
                    ],
                },
                name=f"Origin {origin}",
                customdata=list(
                    zip(
                        od["county_name"],
                        od["horizon"],
                        od["sdc_ape"].round(1),
                        od["m2026_ape"].round(1),
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Origin: " + str(origin) + " | Horizon: %{customdata[1]} yrs<br>"
                    "SDC |%%Error|: %{customdata[2]:.1f}%<br>"
                    "2026 |%%Error|: %{customdata[3]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=template_name,
        title="SDC 2024 vs 2026 Method: Absolute Error per County",
        xaxis_title="SDC 2024 |% Error|",
        yaxis_title="2026 Method |% Error|",
        height=550,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # Add summary statistics
    n_2026_better = int((merged["m2026_ape"] < merged["sdc_ape"]).sum())
    n_sdc_better = int((merged["sdc_ape"] < merged["m2026_ape"]).sum())
    n_tie = len(merged) - n_2026_better - n_sdc_better
    parts.append(
        '<div class="interpretation">'
        "<h3>Summary</h3>"
        "<ul>"
        f"<li><b>2026 Method wins</b> in {n_2026_better} of {len(merged)} county-horizon "
        f"combinations ({100 * n_2026_better / len(merged):.0f}%)</li>"
        f"<li><b>SDC 2024 wins</b> in {n_sdc_better} combinations "
        f"({100 * n_sdc_better / len(merged):.0f}%)</li>"
        f"<li><b>Ties</b>: {n_tie}</li>"
        f'<li>Diamond markers indicate Bakken oil-patch counties</li>'
        "</ul></div>"
    )

    return "\n".join(parts)


# ===================================================================
# Chart 8: County Deep-Dive (dropdown-based)
# ===================================================================

def _build_county_deep_dive(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 8: County deep-dive with a dropdown to select county.

    Shows projected vs actual population trajectory and error by horizon
    for a selected county, with all origin years overlaid.
    """
    parts = ["<h2>County Deep-Dive Explorer</h2>"]
    parts.append(
        '<p class="note">Select a county from the dropdown to see its projection '
        "trajectories and error patterns across all origin years. "
        "The left panel shows projected vs actual population; "
        "the right panel shows signed percentage error by horizon.</p>"
    )

    county_df = data.get("county_detail")
    if county_df is None:
        parts.append(
            '<p class="placeholder">County detail data not available.</p>'
        )
        return "\n".join(parts)

    df = county_df.copy()
    counties = sorted(df["county_name"].unique())

    if not counties:
        parts.append('<p class="placeholder">No county data available.</p>')
        return "\n".join(parts)

    # Build one figure with all counties as traces, using visibility toggling
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Projected vs Actual Population", "% Error by Horizon"],
        horizontal_spacing=0.1,
    )

    # We'll create traces for each county; all hidden except first
    traces_per_county = []
    origin_colors_map = {
        2005: "#e41a1c",
        2010: "#377eb8",
        2015: "#4daf4a",
        2020: "#984ea3",
    }

    for county in counties:
        cdf = df[df["county_name"] == county]
        county_traces = []

        for origin in sorted(cdf["origin_year"].unique()):
            origin_data = cdf[cdf["origin_year"] == origin]
            ocolor = origin_colors_map.get(origin, "#808080")

            # Actual line (only once per origin — use first method)
            first_method_data = origin_data[
                origin_data["method"] == origin_data["method"].iloc[0]
            ].sort_values("validation_year")

            fig.add_trace(
                go.Scatter(
                    x=list(first_method_data["validation_year"]),
                    y=list(first_method_data["actual"]),
                    mode="lines+markers",
                    line={"color": ACTUAL_COLOR, "width": 1.5, "dash": "dot"},
                    marker={"size": 4},
                    name=f"Actual ({origin})",
                    legendgroup=f"actual_{origin}",
                    showlegend=False,
                    visible=False,
                    hovertemplate=(
                        f"<b>Actual</b> (Origin {origin})<br>"
                        "Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            county_traces.append(len(fig.data) - 1)

            for method in ["sdc_2024", "m2026"]:
                mdata = origin_data[origin_data["method"] == method].sort_values(
                    "validation_year"
                )
                if mdata.empty:
                    continue

                mcolor = _method_color(method)
                mlabel = _method_label(method)
                dash = "solid" if method == "m2026" else "dash"

                # Population trajectory
                fig.add_trace(
                    go.Scatter(
                        x=list(mdata["validation_year"]),
                        y=list(mdata["projected"]),
                        mode="lines+markers",
                        line={"color": mcolor, "width": 2, "dash": dash},
                        marker={"size": 4},
                        name=f"{mlabel} ({origin})",
                        legendgroup=f"{method}_{origin}",
                        showlegend=False,
                        visible=False,
                        hovertemplate=(
                            f"<b>{mlabel}</b> (Origin {origin})<br>"
                            "Year: %{x}<br>Projected: %{y:,.0f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                county_traces.append(len(fig.data) - 1)

                # Error by horizon
                fig.add_trace(
                    go.Scatter(
                        x=list(mdata["horizon"]),
                        y=list(mdata["pct_error"]),
                        mode="lines+markers",
                        line={"color": mcolor, "width": 2, "dash": dash},
                        marker={"size": 5},
                        name=f"{mlabel} ({origin})",
                        legendgroup=f"{method}_{origin}",
                        showlegend=False,
                        visible=False,
                        hovertemplate=(
                            f"<b>{mlabel}</b> (Origin {origin})<br>"
                            "Horizon: %{x} yrs<br>"
                            "Error: %{y:.1f}%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
                )
                county_traces.append(len(fig.data) - 1)

        traces_per_county.append(county_traces)

    # Make first county visible
    if traces_per_county:
        for tidx in traces_per_county[0]:
            fig.data[tidx].visible = True

    # Build dropdown menu
    buttons = []
    for i, county in enumerate(counties):
        visibility = [False] * len(fig.data)
        for tidx in traces_per_county[i]:
            visibility[tidx] = True
        buttons.append(
            {
                "label": county,
                "method": "update",
                "args": [{"visible": visibility}],
            }
        )

    fig.update_layout(
        template=template_name,
        height=500,
        title_text=f"County Deep-Dive: {counties[0]}",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": WHITE,
                "bordercolor": MID_GRAY,
                "font": {"size": 12},
            }
        ],
    )
    fig.update_yaxes(tickformat=",", title_text="Population", row=1, col=1)
    fig.update_yaxes(title_text="% Error (signed)", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Forecast Horizon (years)", row=1, col=2)

    # Add zero line on error panel
    fig.add_hline(y=0, line_dash="solid", line_color=MID_GRAY, line_width=1, row=1, col=2)

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 9: Method Delta Visualization
# ===================================================================

def _build_method_delta(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 9: Difference between methods (2026 MAPE minus SDC MAPE).

    Positive means SDC was better; negative means 2026 was better.
    Shows both state-level and county-level deltas.
    """
    parts = ["<h2>Method Delta: Where Does 2026 Improve Over SDC?</h2>"]
    parts.append(
        '<p class="note">Difference in Mean Absolute Percentage Error: '
        "(2026 Method MAPE) minus (SDC 2024 MAPE). "
        "<b>Negative values</b> (below zero) mean the 2026 method has lower error. "
        "<b>Positive values</b> mean SDC 2024 has lower error.</p>"
    )

    comparison_df = data.get("method_comparison")
    if comparison_df is None:
        parts.append(
            '<p class="placeholder">Method comparison data not available.</p>'
        )
        return "\n".join(parts)

    comp = comparison_df.copy().sort_values("horizon")
    comp["state_delta"] = comp["m2026_state_ape"] - comp["sdc_state_ape"]
    comp["county_delta"] = comp["m2026_county_mape"] - comp["sdc_county_mape"]

    fig = go.Figure()

    # State-level delta
    fig.add_trace(
        go.Bar(
            x=list(comp["horizon"]),
            y=list(comp["state_delta"]),
            name="State-Level Delta",
            marker_color=[
                GREEN if v < 0 else ORANGE for v in comp["state_delta"]
            ],
            opacity=0.85,
            hovertemplate=(
                "Horizon: %{x} yrs<br>"
                "Delta: %{y:+.2f} pp<br>"
                "<extra>State Level</extra>"
            ),
        )
    )

    # County-level delta as line overlay
    fig.add_trace(
        go.Scatter(
            x=list(comp["horizon"]),
            y=list(comp["county_delta"]),
            name="County-Level Avg Delta",
            mode="lines+markers",
            line={"color": NAVY, "width": 2.5},
            marker={"size": 7},
            hovertemplate=(
                "Horizon: %{x} yrs<br>"
                "Delta: %{y:+.2f} pp<br>"
                "<extra>County Average</extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=DARK_GRAY,
        line_width=1.5,
        annotation_text="No difference",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    fig.update_layout(
        template=template_name,
        title="MAPE Delta: 2026 Method minus SDC 2024",
        xaxis_title="Forecast Horizon (years)",
        yaxis_title="Delta (percentage points)",
        height=450,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        annotations=[
            {
                "x": 0.02,
                "y": 0.98,
                "xref": "paper",
                "yref": "paper",
                "text": "Below zero = 2026 better",
                "showarrow": False,
                "font": {"size": 10, "color": GREEN},
                "bgcolor": "rgba(255,255,255,0.8)",
            },
            {
                "x": 0.02,
                "y": 0.90,
                "xref": "paper",
                "yref": "paper",
                "text": "Above zero = SDC better",
                "showarrow": False,
                "font": {"size": 10, "color": ORANGE},
                "bgcolor": "rgba(255,255,255,0.8)",
            },
        ],
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Chart 10: Error Distribution Histograms
# ===================================================================

def _build_error_distributions(
    data: dict[str, Any],
    template_name: str,
) -> str:
    """Build Chart 10: Distribution of county-level errors for each method."""
    parts = ["<h2>County Error Distributions</h2>"]
    parts.append(
        '<p class="note">Histogram of county-level absolute percentage errors '
        "across all origin-year and horizon combinations. "
        "A tighter, left-shifted distribution indicates better performance. "
        "Use the dropdown to filter by horizon range.</p>"
    )

    county_df = data.get("county_detail")
    if county_df is None:
        parts.append(
            '<p class="placeholder">County detail data not available.</p>'
        )
        return "\n".join(parts)

    df = county_df.copy()
    df["abs_pct_error"] = df["pct_error"].abs()

    horizon_groups = {
        "All Horizons": (0, 100),
        "Short (1-5 yrs)": (1, 5),
        "Medium (6-10 yrs)": (6, 10),
        "Long (11+ yrs)": (11, 100),
    }

    fig = go.Figure()
    buttons = []

    for group_idx, (group_label, (lo, hi)) in enumerate(horizon_groups.items()):
        gdf = df[(df["horizon"] >= lo) & (df["horizon"] <= hi)]

        for method in ["sdc_2024", "m2026"]:
            mdata = gdf[gdf["method"] == method]["abs_pct_error"]
            visible = group_idx == 0  # only "All Horizons" visible initially

            fig.add_trace(
                go.Histogram(
                    x=list(mdata),
                    name=f"{_method_label(method)}",
                    marker_color=_method_color(method),
                    opacity=0.6,
                    nbinsx=30,
                    visible=visible,
                    hovertemplate=(
                        f"<b>{_method_label(method)}</b><br>"
                        "|%Error| range: %{x}<br>"
                        "Count: %{y}<extra></extra>"
                    ),
                )
            )

    # Build visibility buttons
    n_methods = 2
    for group_idx, group_label in enumerate(horizon_groups):
        visibility = [False] * (len(horizon_groups) * n_methods)
        for j in range(n_methods):
            visibility[group_idx * n_methods + j] = True
        buttons.append(
            {
                "label": group_label,
                "method": "update",
                "args": [{"visible": visibility}],
            }
        )

    fig.update_layout(
        template=template_name,
        title="Distribution of County |% Error|",
        xaxis_title="Absolute Percentage Error (%)",
        yaxis_title="Count (county x origin x horizon)",
        barmode="overlay",
        height=450,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": WHITE,
                "bordercolor": MID_GRAY,
                "font": {"size": 12},
            }
        ],
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # Add summary statistics table
    parts.append("<h3>Distribution Summary Statistics</h3>")
    parts.append('<div class="table-container">')
    parts.append('<table class="data-table" id="tbl-dist-stats">')
    parts.append(
        "<thead><tr>"
        "<th onclick=\"sortTable('tbl-dist-stats',0)\">Horizon Range</th>"
        "<th onclick=\"sortTable('tbl-dist-stats',1)\">Method</th>"
        '<th class="num" onclick="sortTable(\'tbl-dist-stats\',2)">Median |%Err|</th>'
        '<th class="num" onclick="sortTable(\'tbl-dist-stats\',3)">Mean |%Err|</th>'
        '<th class="num" onclick="sortTable(\'tbl-dist-stats\',4)">Std Dev</th>'
        '<th class="num" onclick="sortTable(\'tbl-dist-stats\',5)">90th Pctl</th>'
        '<th class="num" onclick="sortTable(\'tbl-dist-stats\',6)">N</th>'
        "</tr></thead><tbody>"
    )

    for group_label, (lo, hi) in horizon_groups.items():
        gdf = df[(df["horizon"] >= lo) & (df["horizon"] <= hi)]
        for method in ["sdc_2024", "m2026"]:
            mdata = gdf[gdf["method"] == method]["abs_pct_error"]
            if mdata.empty:
                continue
            parts.append(
                f"<tr>"
                f"<td>{group_label}</td>"
                f"<td>{_method_label(method)}</td>"
                f'<td class="num">{mdata.median():.1f}%</td>'
                f'<td class="num">{mdata.mean():.1f}%</td>'
                f'<td class="num">{mdata.std():.1f}</td>'
                f'<td class="num">{mdata.quantile(0.9):.1f}%</td>'
                f'<td class="num">{len(mdata):,}</td>'
                f"</tr>"
            )

    parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Key Takeaways & Methodology Summary
# ===================================================================

def _build_key_takeaways(data: dict[str, Any]) -> str:
    """Build a Key Takeaways section derived from the actual data."""
    parts = ['<h2>Key Takeaways</h2>']

    # Compute takeaway metrics from available data
    horizon_df = data.get("horizon_summary")
    comparison_df = data.get("method_comparison")
    county_df = data.get("county_detail")
    state_df = data.get("state_results")

    takeaways = []

    # 1. Overall winner
    if comparison_df is not None:
        sdc_wins = int((comparison_df.get("winner_state", pd.Series()) == "sdc_2024").sum())
        m26_wins = int((comparison_df.get("winner_state", pd.Series()) == "m2026").sum())
        total_h = len(comparison_df)
        if sdc_wins > m26_wins:
            takeaways.append(
                f"<b>SDC 2024 wins at the state level</b> in {sdc_wins} of {total_h} "
                f"forecast horizons, though margins are often narrow."
            )
        elif m26_wins > sdc_wins:
            takeaways.append(
                f"<b>2026 Method wins at the state level</b> in {m26_wins} of {total_h} "
                f"forecast horizons."
            )
        else:
            takeaways.append(
                f"<b>Methods are evenly matched at the state level</b> "
                f"({sdc_wins} vs {m26_wins} horizon wins out of {total_h})."
            )

    # 2. Short vs long horizon pattern
    if horizon_df is not None:
        short = horizon_df[(horizon_df["horizon"] <= 5)]
        long_h = horizon_df[(horizon_df["horizon"] >= 10)]
        if not short.empty and not long_h.empty:
            short_sdc = short[short["method"] == "sdc_2024"]["mean_state_ape"].mean()
            short_m26 = short[short["method"] == "m2026"]["mean_state_ape"].mean()
            long_sdc = long_h[long_h["method"] == "sdc_2024"]["mean_state_ape"].mean()
            long_m26 = long_h[long_h["method"] == "m2026"]["mean_state_ape"].mean()
            takeaways.append(
                f"<b>Short-horizon accuracy (1-5 years)</b>: SDC {short_sdc:.1f}% MAPE vs "
                f"2026 {short_m26:.1f}% MAPE. "
                f"<b>Long-horizon</b> (10+ years): SDC {long_sdc:.1f}% vs 2026 {long_m26:.1f}%."
            )

    # 3. Bias direction
    if horizon_df is not None:
        all_mpe = horizon_df["mean_state_mpe"]
        if (all_mpe < 0).all():
            takeaways.append(
                "<b>Both methods systematically under-project</b> North Dakota's population "
                "at all horizons, driven by the unpredicted Bakken oil boom "
                f"({BAKKEN_START}-{BAKKEN_END})."
            )

    # 4. County-level assessment
    if county_df is not None:
        df_c = county_df.copy()
        df_c["abs_pct_error"] = df_c["pct_error"].abs()
        avg_by_method = df_c.groupby("method")["abs_pct_error"].mean()
        sdc_avg = avg_by_method.get("sdc_2024", 0)
        m26_avg = avg_by_method.get("m2026", 0)
        takeaways.append(
            f"<b>County-level average |% error|</b>: SDC {sdc_avg:.1f}% vs "
            f"2026 {m26_avg:.1f}%. "
            + (
                "The 2026 method has a modest advantage at the county level."
                if m26_avg < sdc_avg
                else "SDC has a modest advantage at the county level."
                if sdc_avg < m26_avg
                else "Methods perform similarly at the county level."
            )
        )

    # 5. Most recent origin
    if state_df is not None:
        recent = state_df[state_df["origin_year"] == 2020]
        if not recent.empty:
            sdc_recent = recent[recent["method"] == "sdc_2024"]["abs_pct_error"].mean()
            m26_recent = recent[recent["method"] == "m2026"]["abs_pct_error"].mean()
            takeaways.append(
                f"<b>Most recent origin (2020)</b>: SDC MAPE {sdc_recent:.1f}% vs "
                f"2026 MAPE {m26_recent:.1f}% over a 4-year forecast window, "
                f"reflecting performance under stable conditions."
            )

    if takeaways:
        parts.append('<div class="interpretation">')
        parts.append("<ul>")
        for t in takeaways:
            parts.append(f"<li>{t}</li>")
        parts.append("</ul></div>")
    else:
        parts.append('<p class="placeholder">Insufficient data for takeaways.</p>')

    # Methodology summary
    parts.append("<h2>Methodology Summary</h2>")
    parts.append(
        '<div class="interpretation">'
        "<h3>SDC 2024 Method</h3>"
        "<ul>"
        "<li>5-year cohort-component projection steps</li>"
        "<li>Uses census-to-census residual migration rates</li>"
        "<li>Static fertility and mortality assumptions</li>"
        "<li>Simpler approach with fewer parameters</li>"
        "</ul>"
        "<h3>2026 Enhanced Method</h3>"
        "<ul>"
        "<li>Annual projection steps with convergence schedule</li>"
        "<li>PEP-recalibrated residual migration with GQ correction (ADR-055)</li>"
        "<li>Lee-Carter mortality improvement model</li>"
        "<li>BEBR-style multi-period migration averaging (ADR-036)</li>"
        "<li>Ward County floor constraint (ADR-052)</li>"
        "<li>Reservation county PEP anchoring (ADR-045)</li>"
        "</ul>"
        "<h3>Walk-Forward Validation Design</h3>"
        "<ul>"
        "<li>Origins: 2005, 2010, 2015, 2020 — each projects forward through 2024</li>"
        "<li>Evaluation: compare projected population to actual Census/PEP data</li>"
        "<li>Metrics: Mean Absolute Percentage Error (MAPE) and Mean Percentage Error (MPE)</li>"
        "<li>Levels: state-level (53 counties summed) and individual county-level</li>"
        "</ul>"
        "</div>"
    )

    return "\n".join(parts)


# ===================================================================
# Data Table section
# ===================================================================

def _error_class(abs_pct: float) -> str:
    """Return CSS class for color-coding error magnitude."""
    if abs_pct < 3:
        return "err-low"
    if abs_pct < 10:
        return "err-mid"
    return "err-high"


def _build_data_table(data: dict[str, Any]) -> str:
    """Build a raw data table section with sortable columns and filtering."""
    parts = ["<h2>Raw Comparison Data</h2>"]
    parts.append(
        '<p class="note">Complete walk-forward validation results. '
        "All percentage errors are signed (negative = under-projection). "
        "Click any column header to sort. Use the filters below to narrow rows.</p>"
    )

    state_df = data.get("state_results")
    if state_df is None:
        parts.append('<p class="placeholder">State results data not available.</p>')
        return "\n".join(parts)

    # Filter controls
    origins = sorted(state_df["origin_year"].unique())
    parts.append('<div class="table-filter">')
    parts.append('<label>Origin:</label>')
    parts.append(
        '<select id="filter-origin" '
        "onchange=\"filterTableByColumn('tbl-raw',this.id,0)\">"
        '<option value="all">All Origins</option>'
    )
    for o in origins:
        parts.append(f'<option value="{int(o)}">{int(o)}</option>')
    parts.append("</select>")

    parts.append('<label>Method:</label>')
    parts.append(
        '<select id="filter-method" '
        "onchange=\"filterTableByColumn('tbl-raw',this.id,1)\">"
        '<option value="all">All Methods</option>'
        f'<option value="{_method_label("sdc_2024")}">{_method_label("sdc_2024")}</option>'
        f'<option value="{_method_label("m2026")}">{_method_label("m2026")}</option>'
        "</select>"
    )

    parts.append('<label>Search:</label>')
    parts.append(
        '<input type="text" id="filter-raw-text" placeholder="Type to filter..." '
        "oninput=\"filterTable('tbl-raw','filter-raw-text')\">"
    )
    parts.append('<span class="row-count" id="tbl-raw-count"></span>')
    parts.append("</div>")

    parts.append('<div class="table-container">')
    parts.append('<table class="data-table" id="tbl-raw">')
    parts.append(
        "<thead><tr>"
        "<th onclick=\"sortTable('tbl-raw',0)\">Origin</th>"
        "<th onclick=\"sortTable('tbl-raw',1)\">Method</th>"
        "<th onclick=\"sortTable('tbl-raw',2)\">Validation Year</th>"
        "<th onclick=\"sortTable('tbl-raw',3)\">Horizon</th>"
        '<th class="num" onclick="sortTable(\'tbl-raw\',4)">Projected</th>'
        '<th class="num" onclick="sortTable(\'tbl-raw\',5)">Actual</th>'
        '<th class="num" onclick="sortTable(\'tbl-raw\',6)">Error</th>'
        '<th class="num" onclick="sortTable(\'tbl-raw\',7)">% Error</th>'
        '<th class="num" onclick="sortTable(\'tbl-raw\',8)">|% Error|</th>'
        "</tr></thead><tbody>"
    )

    sorted_df = state_df.sort_values(
        ["origin_year", "validation_year", "method"]
    )

    for _, row in sorted_df.iterrows():
        pct_err = float(row["pct_error"])
        abs_err = abs(pct_err)
        err_cls = "negative" if pct_err < -5 else ("positive" if pct_err > 5 else "")
        abs_cls = _error_class(abs_err)

        parts.append(
            f"<tr>"
            f"<td>{int(row['origin_year'])}</td>"
            f"<td>{_method_label(row['method'])}</td>"
            f"<td>{int(row['validation_year'])}</td>"
            f"<td>{int(row['horizon'])}</td>"
            f'<td class="num">{_fmt_pop(row["projected_state"])}</td>'
            f'<td class="num">{_fmt_pop(row["actual_state"])}</td>'
            f'<td class="num">{_fmt_pop(row["error"])}</td>'
            f'<td class="num {err_cls}">{_fmt_pct(pct_err)}</td>'
            f'<td class="num {abs_cls}">{abs_err:.1f}%</td>'
            f"</tr>"
        )

    parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# HTML Assembly
# ===================================================================

def _build_html(
    tab1: str,
    tab2: str,
    tab3: str,
    tab4: str,
    tab4b: str,
    tab5: str,
    tab6: str,
    tab_scatter: str,
    tab_county_deep: str,
    tab_delta: str,
    tab_distributions: str,
    tab_takeaways: str,
    data_table: str,
    data_mode: str,
) -> str:
    """Assemble the full self-contained HTML document."""
    subtitle = (
        f"SDC 2024 vs. 2026 Cohort-Component Method | "
        f"Origins: {', '.join(str(y) for y in ORIGIN_YEARS)} | "
        f"Data: {'Annual' if data_mode == 'annual' else '5-Year'} Resolution | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walk-Forward Validation Report - {TODAY.isoformat()}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: {FONT_FAMILY};
            color: {DARK_GRAY};
            background: #f5f5f5;
            line-height: 1.5;
        }}

        .header {{
            background: linear-gradient(135deg, #1F3864 0%, #0563C1 100%);
            color: white;
            padding: 30px 40px;
            margin-bottom: 0;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .header .subtitle {{
            font-size: 14px;
            opacity: 0.85;
        }}

        .tab-nav {{
            background: white;
            border-bottom: 2px solid {MID_GRAY};
            padding: 0 20px;
            display: flex;
            gap: 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-wrap: wrap;
        }}

        .tab-btn {{
            padding: 12px 14px;
            background: none;
            border: none;
            font-family: {FONT_FAMILY};
            font-size: 12px;
            font-weight: 500;
            color: {DARK_GRAY};
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: {BLUE};
            background: {LIGHT_GRAY};
        }}

        .tab-btn.active {{
            color: {BLUE};
            border-bottom-color: {BLUE};
            font-weight: 600;
        }}

        .tab-content {{
            display: none;
            padding: 30px 40px;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .tab-content.active {{
            display: block;
        }}

        h2 {{
            color: {NAVY};
            font-size: 22px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid {LIGHT_GRAY};
        }}

        h3 {{
            color: {NAVY};
            font-size: 17px;
            margin: 24px 0 12px 0;
        }}

        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .table-container {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .data-table th {{
            background: {NAVY};
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}

        .data-table th:hover {{
            background: #2a4a7a;
        }}

        .data-table th::after {{
            content: ' \\2195';
            font-size: 10px;
            opacity: 0.5;
        }}

        .data-table th.sort-asc::after {{
            content: ' \\2191';
            opacity: 1;
        }}

        .data-table th.sort-desc::after {{
            content: ' \\2193';
            opacity: 1;
        }}

        .data-table th.num {{
            text-align: right;
        }}

        .data-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid {LIGHT_GRAY};
        }}

        .data-table td.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .data-table tbody tr:hover {{
            background: #f0f6ff;
        }}

        .data-table tbody tr:nth-child(even) {{
            background: #fafafa;
        }}

        .data-table tbody tr:nth-child(even):hover {{
            background: #f0f6ff;
        }}

        /* Color-coded error cells */
        .err-low {{ color: {GREEN}; }}
        .err-mid {{ color: {ORANGE}; }}
        .err-high {{ color: #FF0000; font-weight: 600; }}

        .positive {{ color: #00B050; font-weight: 600; }}
        .negative {{ color: #FF0000; font-weight: 600; }}

        .note {{
            font-size: 13px;
            color: #888;
            font-style: italic;
            margin-bottom: 16px;
        }}

        .placeholder {{
            font-style: italic;
            color: #999;
            padding: 20px;
        }}

        .interpretation {{
            background: #f8f9fa;
            border-left: 4px solid {BLUE};
            padding: 16px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}

        .interpretation h3 {{
            margin-top: 0;
        }}

        .interpretation ul {{
            margin: 8px 0 0 20px;
        }}

        .interpretation li {{
            margin-bottom: 6px;
        }}

        /* Dashboard cards */
        .dashboard-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 4px solid {BLUE};
        }}

        .card-warning {{
            border-top-color: #FF8C00;
        }}

        .card-success {{
            border-top-color: {GREEN};
        }}

        .card-title {{
            font-size: 13px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .card-value {{
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
        }}

        .card-detail {{
            font-size: 12px;
            color: #888;
        }}

        .legend-strip {{
            display: flex;
            gap: 20px;
            padding: 12px 40px;
            background: white;
            border-bottom: 1px solid {LIGHT_GRAY};
            font-size: 13px;
            align-items: center;
        }}

        .legend-strip .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-strip .legend-swatch {{
            width: 24px;
            height: 4px;
            border-radius: 2px;
        }}

        .legend-strip .legend-swatch.dashed {{
            background: repeating-linear-gradient(
                90deg,
                var(--color) 0, var(--color) 6px,
                transparent 6px, transparent 10px
            );
        }}

        /* Table filter controls */
        .table-filter {{
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .table-filter label {{
            font-size: 13px;
            font-weight: 600;
            color: {NAVY};
        }}

        .table-filter select, .table-filter input {{
            padding: 6px 10px;
            border: 1px solid {MID_GRAY};
            border-radius: 4px;
            font-size: 13px;
            font-family: {FONT_FAMILY};
        }}

        .table-filter input[type="text"] {{
            width: 200px;
        }}

        .row-count {{
            font-size: 12px;
            color: #888;
            margin-left: auto;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #999;
            border-top: 1px solid {MID_GRAY};
            margin-top: 40px;
        }}

        @media print {{
            .tab-nav {{ display: none; }}
            .tab-content {{ display: block !important; page-break-inside: avoid; }}
            .header {{ background: {NAVY} !important; -webkit-print-color-adjust: exact; }}
        }}

        @media (max-width: 900px) {{
            .dashboard-cards {{ grid-template-columns: 1fr; }}
            .header {{ padding: 20px; }}
            .tab-nav {{ padding: 0 10px; }}
            .tab-btn {{ padding: 10px 8px; font-size: 11px; }}
            .tab-content {{ padding: 16px; }}
            .legend-strip {{ padding: 12px 10px; flex-wrap: wrap; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Walk-Forward Validation: SDC 2024 vs. 2026 Method</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="legend-strip">
        <span style="font-weight:600;">Legend:</span>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{METHOD_CONFIG['sdc_2024']['color']};"></div>
            <span>SDC 2024</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{METHOD_CONFIG['m2026']['color']};"></div>
            <span>2026 Method</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch dashed" style="--color:{ACTUAL_COLOR}; background:{ACTUAL_COLOR};"></div>
            <span>Actual (Census/PEP)</span>
        </div>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('takeaways')">Summary</button>
        <button class="tab-btn" onclick="switchTab('curves')">Projection Curves</button>
        <button class="tab-btn" onclick="switchTab('errors')">Error Trajectories</button>
        <button class="tab-btn" onclick="switchTab('horizon')">Error by Horizon</button>
        <button class="tab-btn" onclick="switchTab('delta')">Method Delta</button>
        <button class="tab-btn" onclick="switchTab('scatter')">Error Scatter</button>
        <button class="tab-btn" onclick="switchTab('distributions')">Distributions</button>
        <button class="tab-btn" onclick="switchTab('county')">County Heatmap</button>
        <button class="tab-btn" onclick="switchTab('county_diff')">Diff Heatmap</button>
        <button class="tab-btn" onclick="switchTab('county_deep')">County Deep-Dive</button>
        <button class="tab-btn" onclick="switchTab('dashboard')">Dashboard</button>
        <button class="tab-btn" onclick="switchTab('direction')">Error Direction</button>
        <button class="tab-btn" onclick="switchTab('data')">Raw Data</button>
    </div>

    <div id="tab-takeaways" class="tab-content active">
        {tab_takeaways}
    </div>

    <div id="tab-curves" class="tab-content">
        {tab1}
    </div>

    <div id="tab-errors" class="tab-content">
        {tab2}
    </div>

    <div id="tab-horizon" class="tab-content">
        {tab3}
    </div>

    <div id="tab-delta" class="tab-content">
        {tab_delta}
    </div>

    <div id="tab-scatter" class="tab-content">
        {tab_scatter}
    </div>

    <div id="tab-distributions" class="tab-content">
        {tab_distributions}
    </div>

    <div id="tab-county" class="tab-content">
        {tab4}
    </div>

    <div id="tab-county_diff" class="tab-content">
        {tab4b}
    </div>

    <div id="tab-county_deep" class="tab-content">
        {tab_county_deep}
    </div>

    <div id="tab-dashboard" class="tab-content">
        {tab5}
    </div>

    <div id="tab-direction" class="tab-content">
        {tab6}
    </div>

    <div id="tab-data" class="tab-content">
        {data_table}
    </div>

    <div class="footer">
        North Dakota Population Projections |
        Walk-Forward Validation Report |
        Generated {TODAY.strftime('%B %d, %Y')} |
        Data: Census PEP 2025 Vintage
    </div>

    <script>
        /* ---- Tab switching ---- */
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(function(el) {{
                el.classList.remove('active');
            }});
            document.querySelectorAll('.tab-btn').forEach(function(el) {{
                el.classList.remove('active');
            }});

            document.getElementById('tab-' + tabId).classList.add('active');

            /* Find and activate the matching button */
            var buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(function(btn) {{
                if (btn.getAttribute('onclick').indexOf("'" + tabId + "'") !== -1) {{
                    btn.classList.add('active');
                }}
            }});

            /* Trigger Plotly resize for charts in hidden tabs */
            setTimeout(function() {{
                var plots = document.getElementById('tab-' + tabId)
                    .querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 50);
        }}

        /* ---- Sortable tables ---- */
        function sortTable(tableId, colIdx) {{
            var table = document.getElementById(tableId);
            if (!table) return;
            var tbody = table.querySelector('tbody');
            if (!tbody) return;
            var rows = Array.from(tbody.querySelectorAll('tr'));

            /* Determine sort direction */
            var th = table.querySelectorAll('thead th')[colIdx];
            var ascending = !th.classList.contains('sort-asc');

            /* Clear all sort indicators */
            table.querySelectorAll('thead th').forEach(function(h) {{
                h.classList.remove('sort-asc', 'sort-desc');
            }});
            th.classList.add(ascending ? 'sort-asc' : 'sort-desc');

            rows.sort(function(a, b) {{
                var aText = a.cells[colIdx].textContent.trim();
                var bText = b.cells[colIdx].textContent.trim();

                /* Strip % and , for numeric comparison */
                var aClean = aText.replace(/[%,+]/g, '');
                var bClean = bText.replace(/[%,+]/g, '');
                var aVal = parseFloat(aClean);
                var bVal = parseFloat(bClean);

                if (!isNaN(aVal) && !isNaN(bVal)) {{
                    return ascending ? aVal - bVal : bVal - aVal;
                }}
                /* String comparison fallback */
                return ascending
                    ? aText.localeCompare(bText)
                    : bText.localeCompare(aText);
            }});

            rows.forEach(function(row) {{ tbody.appendChild(row); }});

            /* Update row count if present */
            updateRowCount(tableId);
        }}

        /* ---- Table text filter ---- */
        function filterTable(tableId, inputId) {{
            var filter = document.getElementById(inputId).value.toLowerCase();
            var table = document.getElementById(tableId);
            if (!table) return;
            var rows = table.querySelectorAll('tbody tr');
            var shown = 0;
            rows.forEach(function(row) {{
                var text = row.textContent.toLowerCase();
                var match = text.indexOf(filter) !== -1;
                row.style.display = match ? '' : 'none';
                if (match) shown++;
            }});
            updateRowCount(tableId, shown, rows.length);
        }}

        /* ---- Dropdown filter for specific column ---- */
        function filterTableByColumn(tableId, selectId, colIdx) {{
            var val = document.getElementById(selectId).value;
            var table = document.getElementById(tableId);
            if (!table) return;
            var rows = table.querySelectorAll('tbody tr');
            var shown = 0;
            rows.forEach(function(row) {{
                if (val === 'all' || row.cells[colIdx].textContent.trim() === val) {{
                    row.style.display = '';
                    shown++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});
            updateRowCount(tableId, shown, rows.length);
        }}

        function updateRowCount(tableId, shown, total) {{
            var counter = document.getElementById(tableId + '-count');
            if (!counter) return;
            if (shown === undefined) {{
                var table = document.getElementById(tableId);
                var rows = table.querySelectorAll('tbody tr');
                total = rows.length;
                shown = 0;
                rows.forEach(function(r) {{
                    if (r.style.display !== 'none') shown++;
                }});
            }}
            counter.textContent = shown + ' of ' + total + ' rows';
        }}
    </script>
</body>
</html>"""


# ===================================================================
# Report Builder
# ===================================================================

def build_report(output_path: Path) -> Path:
    """Build the complete walk-forward validation HTML report.

    Parameters
    ----------
    output_path : Path
        Full path to the output HTML file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    logger.info("=" * 60)
    logger.info("Walk-Forward Validation Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    # Register Plotly template
    template_name = _get_plotly_template()

    # Load data
    logger.info("Loading walk-forward validation data...")
    data = load_data()
    data_mode = data.get("data_mode", "5year")
    logger.info("  Data mode: %s", data_mode)

    # Build sections — original charts
    logger.info("Building Chart 1: Projection Curves...")
    tab1 = _build_projection_curves(data, template_name)

    logger.info("Building Chart 2: Error Trajectories...")
    tab2 = _build_error_trajectory(data, template_name)

    logger.info("Building Chart 3: Average Error by Horizon...")
    tab3 = _build_avg_error_by_horizon(data, template_name)

    logger.info("Building Chart 4: County MAPE Heatmap...")
    tab4 = _build_county_heatmap(data, template_name)

    logger.info("Building Chart 4b: County MAPE Difference Heatmap...")
    tab4b = _build_county_diff_heatmap(data, template_name)

    logger.info("Building Chart 5: Comparison Dashboard...")
    tab5 = _build_comparison_dashboard(data, template_name)

    logger.info("Building Chart 6: Direction of Error...")
    tab6 = _build_direction_analysis(data, template_name)

    # Build sections — new charts
    logger.info("Building Chart 7: Error Scatter Comparison...")
    tab_scatter = _build_scatter_comparison(data, template_name)

    logger.info("Building Chart 8: County Deep-Dive...")
    tab_county_deep = _build_county_deep_dive(data, template_name)

    logger.info("Building Chart 9: Method Delta...")
    tab_delta = _build_method_delta(data, template_name)

    logger.info("Building Chart 10: Error Distributions...")
    tab_distributions = _build_error_distributions(data, template_name)

    logger.info("Building Key Takeaways & Methodology...")
    tab_takeaways = _build_key_takeaways(data)

    logger.info("Building Raw Data Table...")
    data_table = _build_data_table(data)

    # Assemble HTML
    logger.info("Assembling HTML report...")
    html = _build_html(
        tab1,
        tab2,
        tab3,
        tab4,
        tab4b,
        tab5,
        tab6,
        tab_scatter,
        tab_county_deep,
        tab_delta,
        tab_distributions,
        tab_takeaways,
        data_table,
        data_mode,
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Report generated successfully!")
    logger.info("  Output: %s", output_path)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  Data mode: %s", data_mode)
    logger.info(
        "  Tabs: 13 (Summary, Curves, Errors, Horizon, Delta, Scatter, "
        "Distributions, County Heatmap, Diff Heatmap, County Deep-Dive, "
        "Dashboard, Direction, Data)"
    )
    logger.info("=" * 60)

    return output_path


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build interactive HTML report for walk-forward validation "
            "comparing SDC 2024 and 2026 cohort-component projection methods."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML file path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the walk-forward report builder."""
    args = parse_args()

    try:
        output_path = build_report(output_path=args.output)
        logger.info("Done. Report at: %s", output_path)
        return 0
    except Exception:
        logger.exception("Failed to build walk-forward validation report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
