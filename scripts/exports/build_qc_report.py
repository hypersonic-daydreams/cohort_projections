#!/usr/bin/env python3
"""Build unified QC analysis HTML report from walk-forward validation outputs.

Created: 2026-03-03
Author: Claude Code (automated)
SOP-002 compliant: Yes

Purpose
-------
Merge the sensitivity analysis, uncertainty quantification, and QC diagnostics
results into a single tabbed HTML report.  Reads pre-computed CSV outputs from
``data/analysis/walk_forward/`` and produces a self-contained interactive HTML
file with Plotly charts, sortable tables, and tab navigation.

Inputs
------
- data/analysis/walk_forward/sensitivity_results.csv
- data/analysis/walk_forward/sensitivity_tornado.csv
- data/analysis/walk_forward/prediction_intervals.csv
- data/analysis/walk_forward/uncertainty_bands.csv
- data/analysis/walk_forward/error_decomposition.csv
- data/analysis/walk_forward/bias_analysis.csv
- data/analysis/walk_forward/residual_diagnostics.csv
- data/analysis/walk_forward/outlier_flags.csv
- data/analysis/walk_forward/county_report_cards.csv

Outputs
-------
- data/analysis/walk_forward/qc_analysis_report.html

Usage
-----
    python scripts/exports/build_qc_report.py
    python scripts/exports/build_qc_report.py --output path/to/output.html

Dependencies
------------
pandas, plotly (loaded via CDN in output HTML)

Key ADRs
--------
ADR-057: Rolling-origin backtests

SOP-002 Metadata
-----------------
    Author:        Projection team
    Date created:  2026-03-03
    Last modified: 2026-03-03
    Input files:   data/analysis/walk_forward/*.csv
    Output files:  data/analysis/walk_forward/qc_analysis_report.html
    Dependencies:  pandas, plotly (CDN)
"""

from __future__ import annotations

import argparse
import html as html_lib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
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
DEFAULT_OUTPUT = DATA_DIR / "qc_analysis_report.html"

# Method display config
METHOD_CONFIG = {
    "sdc_2024": {"label": "SDC 2024", "color": "#1f77b4"},
    "m2026": {"label": "2026 Method", "color": "#d62728"},
}

# Theme constants (matching walk-forward report)
NAVY = "#1F3864"
BLUE = "#0563C1"
DARK_GRAY = "#595959"
MID_GRAY = "#D9D9D9"
LIGHT_GRAY = "#F2F2F2"
WHITE = "#FFFFFF"
GREEN = "#00B050"
ORANGE = "#FF8C00"
RED = "#FF0000"
FONT_FAMILY = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Grade colors
GRADE_COLORS = {
    "A": GREEN,
    "B": BLUE,
    "C": ORANGE,
    "D": RED,
}

# Category colors
CATEGORY_COLORS = {
    "Rural": "#4daf4a",
    "Urban/College": "#377eb8",
    "Bakken": "#e41a1c",
    "Reservation": "#984ea3",
}


# ===================================================================
# Plotly template
# ===================================================================

def _get_plotly_template() -> str:
    """Register and return the QC report template name."""
    template_name = "qc_report"
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
        ],
    )
    pio.templates[template_name] = template
    return template_name


# ===================================================================
# Helpers
# ===================================================================

def _load_csv(name: str, **kwargs: Any) -> pd.DataFrame | None:
    """Load a CSV from the walk-forward data directory."""
    path = DATA_DIR / name
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        logger.exception("Failed to read CSV: %s", path)
        return None


def _method_label(method: str) -> str:
    return METHOD_CONFIG.get(method, {}).get("label", method)


def _method_color(method: str) -> str:
    return METHOD_CONFIG.get(method, {}).get("color", "#808080")


def _fmt_pct(p: float, decimals: int = 1) -> str:
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.{decimals}f}%"


def _grade_color(grade: str) -> str:
    return GRADE_COLORS.get(grade, DARK_GRAY)


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(str(text))


def _chart_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to embeddable HTML (no plotly.js include)."""
    return (
        '<div class="chart-container">'
        + pio.to_html(fig, include_plotlyjs=False, full_html=False)
        + "</div>"
    )


# ===================================================================
# Data loading
# ===================================================================

def load_all_data() -> dict[str, pd.DataFrame | None]:
    """Load all CSV files needed for the report."""
    return {
        "sensitivity_results": _load_csv("sensitivity_results.csv"),
        "sensitivity_tornado": _load_csv("sensitivity_tornado.csv"),
        "prediction_intervals": _load_csv("prediction_intervals.csv"),
        "uncertainty_bands": _load_csv("uncertainty_bands.csv"),
        "error_decomposition": _load_csv("error_decomposition.csv"),
        "bias_analysis": _load_csv("bias_analysis.csv"),
        "residual_diagnostics": _load_csv("residual_diagnostics.csv"),
        "outlier_flags": _load_csv("outlier_flags.csv"),
        "county_report_cards": _load_csv("county_report_cards.csv"),
    }


# ===================================================================
# Tab 1: Overview / Summary
# ===================================================================

def _build_overview(data: dict[str, Any]) -> str:
    """Build the overview / summary tab with key findings from all analyses."""
    parts: list[str] = []
    parts.append("<h2>QC Analysis Overview</h2>")
    parts.append(
        '<p class="note">Key findings from sensitivity analysis, uncertainty '
        "quantification, and QC diagnostics for the walk-forward validation of "
        "SDC 2024 and 2026 projection methods.</p>"
    )

    # Dashboard cards
    parts.append('<div class="dashboard-cards">')

    # Report cards summary
    rc = data.get("county_report_cards")
    if rc is not None:
        for method in ["m2026", "sdc_2024"]:
            mrc = rc[rc["method"] == method]
            grade_counts = mrc["grade"].value_counts().to_dict()
            a_b = grade_counts.get("A", 0) + grade_counts.get("B", 0)
            total = len(mrc)
            avg_mape = mrc["mape"].mean()
            label = _method_label(method)
            cls = "card-success" if a_b > total / 2 else "card-warning"
            parts.append(
                f'<div class="card {cls}">'
                f'<div class="card-title">{_esc(label)} Report Card</div>'
                f'<div class="card-value">{a_b}/{total} A/B</div>'
                f'<div class="card-detail">Avg MAPE: {avg_mape:.1f}% | '
                f'A:{grade_counts.get("A", 0)} B:{grade_counts.get("B", 0)} '
                f'C:{grade_counts.get("C", 0)} D:{grade_counts.get("D", 0)}</div>'
                f"</div>"
            )

    # Sensitivity: top parameter
    tornado = data.get("sensitivity_tornado")
    if tornado is not None:
        for method in ["m2026", "sdc_2024"]:
            mt = tornado[tornado["method"] == method]
            if not mt.empty:
                top = mt.sort_values("swing_state_error", ascending=False).iloc[0]
                label = _method_label(method)
                parts.append(
                    f'<div class="card">'
                    f'<div class="card-title">{_esc(label)} Top Sensitivity</div>'
                    f'<div class="card-value">{_esc(str(top["parameter"]))}</div>'
                    f'<div class="card-detail">State error swing: '
                    f'{top["swing_state_error"]:.1f} pp</div>'
                    f"</div>"
                )

    # Uncertainty: 95% interval width at 10yr horizon
    pi = data.get("prediction_intervals")
    if pi is not None:
        state_pi = pi[(pi["level"] == "state") & (pi["horizon"] == 10)]
        for method in ["m2026", "sdc_2024"]:
            mpi = state_pi[state_pi["method"] == method]
            if not mpi.empty:
                row = mpi.iloc[0]
                width = row["p95"] - row["p5"]
                label = _method_label(method)
                parts.append(
                    f'<div class="card">'
                    f'<div class="card-title">{_esc(label)} 10yr 95% PI Width</div>'
                    f'<div class="card-value">{width:.1f} pp</div>'
                    f'<div class="card-detail">P5: {row["p5"]:.1f}% | '
                    f'P95: {row["p95"]:.1f}%</div>'
                    f"</div>"
                )

    parts.append("</div>")  # dashboard-cards

    # Key findings narrative
    parts.append(
        '<div class="interpretation">'
        "<h3>Key Findings</h3>"
        "<ul>"
    )

    # Sensitivity finding
    if tornado is not None and not tornado.empty:
        top_all = tornado.sort_values("swing_state_error", ascending=False).iloc[0]
        parts.append(
            f"<li><strong>Sensitivity:</strong> "
            f"{_esc(str(top_all['parameter']))} is the most influential parameter "
            f"for {_esc(_method_label(str(top_all['method'])))} "
            f"(state error swing of {top_all['swing_state_error']:.1f} pp). "
            f"Migration rates dominate uncertainty in both methods.</li>"
        )

    # Uncertainty finding
    if pi is not None:
        state_pi_long = pi[(pi["level"] == "state") & (pi["horizon"] == 19)]
        if not state_pi_long.empty:
            max_width = (state_pi_long["p95"] - state_pi_long["p5"]).max()
            parts.append(
                f"<li><strong>Uncertainty:</strong> "
                f"At 19-year horizons the 95% prediction interval width reaches "
                f"{max_width:.0f} pp, driven largely by the Bakken boom era. "
                f"Intervals from recent origins (2015, 2020) are substantially "
                f"narrower.</li>"
            )

    # QC finding
    if rc is not None:
        d_counties = rc[rc["grade"] == "D"]["county_name"].unique()
        if len(d_counties) > 0:
            d_list = ", ".join(sorted(d_counties)[:8])
            parts.append(
                f"<li><strong>QC Diagnostics:</strong> "
                f"{len(d_counties)} counties receive D grades "
                f"(worst-case |error| above 25%): {_esc(d_list)}. "
                f"These are predominantly Bakken oil-patch counties where "
                f"the boom created unpredictable migration surges.</li>"
            )

    # Residual diagnostics
    rd = data.get("residual_diagnostics")
    if rd is not None:
        non_normal = rd[~rd["normal_at_05"]]
        if len(non_normal) == len(rd):
            parts.append(
                "<li><strong>Residuals:</strong> Error distributions are non-normal "
                "across all horizon buckets for both methods, with moderate positive "
                "lag-1 autocorrelation (0.5-0.7). This supports using empirical "
                "percentile-based prediction intervals rather than Gaussian "
                "assumptions.</li>"
            )

    parts.append("</ul></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 2: Sensitivity — Tornado Diagrams
# ===================================================================

def _build_sensitivity_tornado(data: dict[str, Any], tmpl: str) -> str:
    """Build tornado diagrams for both methods."""
    parts: list[str] = []
    parts.append("<h2>Sensitivity Analysis: Tornado Diagrams</h2>")
    parts.append(
        '<p class="note">Each bar shows the deviation in state-level error '
        "from baseline when a single parameter is perturbed to its extreme "
        "low and high values. Longer bars indicate higher sensitivity. "
        "Parameters are ranked by total swing magnitude.</p>"
    )

    tornado = data.get("sensitivity_tornado")
    if tornado is None:
        parts.append('<p class="placeholder">Tornado data not available.</p>')
        return "\n".join(parts)

    for method in ["m2026", "sdc_2024"]:
        mt = tornado[tornado["method"] == method].copy()
        if mt.empty:
            continue
        mt = mt.sort_values("swing_state_error", ascending=True)
        label = _method_label(method)
        color = _method_color(method)

        fig = go.Figure()

        # Low side (negative deviation)
        fig.add_trace(go.Bar(
            y=mt["parameter"],
            x=mt["low_deviation"],
            orientation="h",
            name="Low perturbation",
            marker_color="rgba(31, 119, 180, 0.7)",
            customdata=list(mt["low_label"]),
            hovertemplate=(
                "<b>%{y}</b><br>Perturbation: %{customdata}<br>"
                "State error deviation: %{x:.2f} pp<extra></extra>"
            ),
        ))

        # High side (positive deviation)
        fig.add_trace(go.Bar(
            y=mt["parameter"],
            x=mt["high_deviation"],
            orientation="h",
            name="High perturbation",
            marker_color="rgba(214, 39, 40, 0.7)",
            customdata=list(mt["high_label"]),
            hovertemplate=(
                "<b>%{y}</b><br>Perturbation: %{customdata}<br>"
                "State error deviation: %{x:.2f} pp<extra></extra>"
            ),
        ))

        # Zero reference
        fig.add_vline(x=0, line_dash="solid", line_color=MID_GRAY, line_width=1)

        fig.update_layout(
            template=tmpl,
            title=f"Tornado Diagram: {label} (State % Error Deviation from Baseline)",
            xaxis_title="Deviation in State % Error (pp)",
            yaxis_title="",
            height=350,
            barmode="overlay",
            legend={"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center"},
        )
        parts.append(_chart_html(fig))

        # Also show MAPE tornado
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=mt["parameter"],
            x=mt["mape_low_deviation"],
            orientation="h",
            name="Low perturbation",
            marker_color="rgba(31, 119, 180, 0.5)",
            customdata=list(mt["low_label"]),
            hovertemplate=(
                "<b>%{y}</b><br>Perturbation: %{customdata}<br>"
                "MAPE deviation: %{x:.2f} pp<extra></extra>"
            ),
        ))
        fig2.add_trace(go.Bar(
            y=mt["parameter"],
            x=mt["mape_high_deviation"],
            orientation="h",
            name="High perturbation",
            marker_color="rgba(214, 39, 40, 0.5)",
            customdata=list(mt["high_label"]),
            hovertemplate=(
                "<b>%{y}</b><br>Perturbation: %{customdata}<br>"
                "MAPE deviation: %{x:.2f} pp<extra></extra>"
            ),
        ))
        fig2.add_vline(x=0, line_dash="solid", line_color=MID_GRAY, line_width=1)
        fig2.update_layout(
            template=tmpl,
            title=f"Tornado Diagram: {label} (County MAPE Deviation from Baseline)",
            xaxis_title="Deviation in County MAPE (pp)",
            yaxis_title="",
            height=350,
            barmode="overlay",
            legend={"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center"},
        )
        parts.append(_chart_html(fig2))

    return "\n".join(parts)


# ===================================================================
# Tab 3: Sensitivity — Parameter Sweeps
# ===================================================================

def _build_sensitivity_sweeps(data: dict[str, Any], tmpl: str) -> str:
    """Build parameter sweep line charts and radar/spider charts."""
    parts: list[str] = []
    parts.append("<h2>Sensitivity Analysis: Parameter Sweeps</h2>")
    parts.append(
        '<p class="note">Line charts showing how state-level error and county '
        "MAPE change as each parameter is swept through its range. Each line "
        "represents one method.</p>"
    )

    sr = data.get("sensitivity_results")
    if sr is None:
        parts.append('<p class="placeholder">Sensitivity results not available.</p>')
        return "\n".join(parts)

    # Average across origin years for each method x parameter x perturbation
    avg = sr.groupby(["method", "parameter", "perturbation_level"]).agg(
        state_pct_error=("state_pct_error", "mean"),
        county_mape=("county_mape", "mean"),
    ).reset_index()

    parameters = sorted(avg["parameter"].unique())

    for param in parameters:
        pdata = avg[avg["parameter"] == param]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["State % Error", "County MAPE"],
            horizontal_spacing=0.12,
        )

        for method in ["sdc_2024", "m2026"]:
            md = pdata[pdata["method"] == method]
            if md.empty:
                continue
            color = _method_color(method)
            label = _method_label(method)

            # Sort perturbation levels logically
            md = md.copy()
            # Try numeric sort first
            try:
                md["sort_key"] = md["perturbation_level"].str.extract(
                    r"([-+]?\d*\.?\d+)"
                )[0].astype(float)
            except Exception:
                md["sort_key"] = range(len(md))
            md = md.sort_values("sort_key")

            x_labels = list(md["perturbation_level"])

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=list(md["state_pct_error"]),
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "width": 2},
                    marker={"size": 7},
                    legendgroup=method,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{label}</b><br>Level: %{{x}}<br>"
                        f"State Error: %{{y:.2f}}%<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=list(md["county_mape"]),
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "width": 2},
                    marker={"size": 7},
                    legendgroup=method,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>Level: %{{x}}<br>"
                        f"County MAPE: %{{y:.2f}}%<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

        param_display = param.replace("_", " ").title()
        fig.update_layout(
            template=tmpl,
            title=f"Parameter Sweep: {param_display}",
            height=400,
            legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center"},
        )
        fig.update_xaxes(tickangle=45)
        parts.append(_chart_html(fig))

    # Radar / spider chart: MAPE swing per parameter
    tornado = data.get("sensitivity_tornado")
    if tornado is not None:
        parts.append("<h3>Sensitivity Spider Chart: MAPE Swing by Parameter</h3>")
        parts.append(
            '<p class="note">The radial distance shows the total MAPE swing '
            "(high minus low perturbation effect) for each parameter.</p>"
        )
        fig = go.Figure()
        for method in ["m2026", "sdc_2024"]:
            mt = tornado[tornado["method"] == method]
            if mt.empty:
                continue
            categories = list(mt["parameter"])
            values = list(mt["mape_swing"])
            # Close the polygon
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=_method_label(method),
                line_color=_method_color(method),
                opacity=0.6,
            ))
        fig.update_layout(
            template=tmpl,
            title="Sensitivity Spider: County MAPE Swing by Parameter",
            polar={"radialaxis": {"visible": True, "range": [0, None]}},
            height=500,
        )
        parts.append(_chart_html(fig))

    return "\n".join(parts)


# ===================================================================
# Tab 4: Uncertainty — Fan Charts
# ===================================================================

def _build_uncertainty_fan_charts(data: dict[str, Any], tmpl: str) -> str:
    """Build state + county fan charts with dropdown for all 53 counties."""
    parts: list[str] = []
    parts.append("<h2>Uncertainty: Fan Charts</h2>")
    parts.append(
        '<p class="note">Fan charts show projected population with 50%, 80%, '
        "and 95% prediction interval bands derived from historical walk-forward "
        "errors. Use the dropdown to select a county (all 53 included).</p>"
    )

    ub = data.get("uncertainty_bands")
    if ub is None:
        parts.append('<p class="placeholder">Uncertainty bands data not available.</p>')
        return "\n".join(parts)

    # --- State fan chart ---
    parts.append("<h3>State-Level Fan Charts</h3>")
    state_ub = ub[ub["level"] == "state"]

    for method in ["m2026", "sdc_2024"]:
        ms = state_ub[state_ub["method"] == method].sort_values("year")
        if ms.empty:
            continue
        label = _method_label(method)
        color = _method_color(method)
        years = list(ms["year"])
        proj = list(ms["projected"])

        fig = go.Figure()

        # 95% band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=list(ms["upper_95"]) + list(ms["lower_95"])[::-1],
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)",
            line={"width": 0},
            name="95% PI",
            hoverinfo="skip",
        ))
        # 80% band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=list(ms["upper_80"]) + list(ms["lower_80"])[::-1],
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.2)",
            line={"width": 0},
            name="80% PI",
            hoverinfo="skip",
        ))
        # 50% band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=list(ms["upper_50"]) + list(ms["lower_50"])[::-1],
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.3)",
            line={"width": 0},
            name="50% PI",
            hoverinfo="skip",
        ))
        # Central projection
        fig.add_trace(go.Scatter(
            x=years, y=proj,
            mode="lines",
            name="Projected",
            line={"color": color, "width": 2.5},
            hovertemplate="Year: %{x}<br>Projected: %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            template=tmpl,
            title=f"State Population Fan Chart: {label}",
            xaxis_title="Year",
            yaxis_title="Population",
            height=450,
            yaxis_tickformat=",",
        )
        parts.append(_chart_html(fig))

    # --- County fan charts with dropdown ---
    parts.append("<h3>County-Level Fan Charts (All 53 Counties)</h3>")
    county_ub = ub[ub["level"] == "county"]
    counties = sorted(county_ub["geography"].unique())

    if not counties:
        parts.append('<p class="placeholder">No county uncertainty data.</p>')
        return "\n".join(parts)

    # Build one figure per method with county dropdown
    for method in ["m2026", "sdc_2024"]:
        label = _method_label(method)
        color = _method_color(method)
        mc = county_ub[county_ub["method"] == method]
        if mc.empty:
            continue

        fig = go.Figure()
        traces_per_county: list[list[int]] = []

        for county in counties:
            cc = mc[mc["geography"] == county].sort_values("year")
            if cc.empty:
                traces_per_county.append([])
                continue
            county_traces: list[int] = []
            years = list(cc["year"])
            proj = list(cc["projected"])

            # 95% band
            fig.add_trace(go.Scatter(
                x=years + years[::-1],
                y=list(cc["upper_95"]) + list(cc["lower_95"])[::-1],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)",
                line={"width": 0},
                name="95% PI",
                showlegend=False,
                visible=False,
                hoverinfo="skip",
            ))
            county_traces.append(len(fig.data) - 1)

            # 80% band
            fig.add_trace(go.Scatter(
                x=years + years[::-1],
                y=list(cc["upper_80"]) + list(cc["lower_80"])[::-1],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.2)",
                line={"width": 0},
                name="80% PI",
                showlegend=False,
                visible=False,
                hoverinfo="skip",
            ))
            county_traces.append(len(fig.data) - 1)

            # 50% band
            fig.add_trace(go.Scatter(
                x=years + years[::-1],
                y=list(cc["upper_50"]) + list(cc["lower_50"])[::-1],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.3)",
                line={"width": 0},
                name="50% PI",
                showlegend=False,
                visible=False,
                hoverinfo="skip",
            ))
            county_traces.append(len(fig.data) - 1)

            # Central line
            fig.add_trace(go.Scatter(
                x=years, y=proj,
                mode="lines",
                name="Projected",
                line={"color": color, "width": 2.5},
                showlegend=False,
                visible=False,
                hovertemplate=(
                    f"<b>{_esc(county)}</b><br>"
                    "Year: %{x}<br>Projected: %{y:,.0f}<extra></extra>"
                ),
            ))
            county_traces.append(len(fig.data) - 1)

            traces_per_county.append(county_traces)

        # Make first county visible
        if traces_per_county and traces_per_county[0]:
            for tidx in traces_per_county[0]:
                fig.data[tidx].visible = True

        # Build dropdown buttons
        buttons = []
        for i, county in enumerate(counties):
            vis = [False] * len(fig.data)
            for tidx in traces_per_county[i]:
                vis[tidx] = True
            buttons.append({
                "label": county,
                "method": "update",
                "args": [
                    {"visible": vis},
                    {"title": f"County Fan Chart: {county} ({label})"},
                ],
            })

        fig.update_layout(
            template=tmpl,
            title=f"County Fan Chart: {counties[0]} ({label})",
            xaxis_title="Year",
            yaxis_title="Population",
            height=500,
            yaxis_tickformat=",",
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }],
        )
        parts.append(_chart_html(fig))

    return "\n".join(parts)


# ===================================================================
# Tab 5: Uncertainty — Prediction Intervals
# ===================================================================

def _build_uncertainty_intervals(data: dict[str, Any], tmpl: str) -> str:
    """Build PI width growth charts and error distributions."""
    parts: list[str] = []
    parts.append("<h2>Uncertainty: Prediction Intervals</h2>")
    parts.append(
        '<p class="note">Prediction interval widths grow with forecast horizon. '
        "These charts show how the 50%, 80%, and 95% interval widths expand as "
        "the horizon lengthens, for both state and county levels.</p>"
    )

    pi = data.get("prediction_intervals")
    if pi is None:
        parts.append('<p class="placeholder">Prediction interval data not available.</p>')
        return "\n".join(parts)

    for level_name, level_label in [("state", "State-Level"), ("county", "County-Level")]:
        lpi = pi[pi["level"] == level_name]
        if lpi.empty:
            continue

        fig = go.Figure()
        for method in ["sdc_2024", "m2026"]:
            mp = lpi[lpi["method"] == method].sort_values("horizon")
            if mp.empty:
                continue
            label = _method_label(method)
            color = _method_color(method)
            horizons = list(mp["horizon"])

            # 95% width
            width_95 = list(mp["p95"] - mp["p5"])
            fig.add_trace(go.Scatter(
                x=horizons, y=width_95,
                mode="lines+markers",
                name=f"{label} (90% width: P5-P95)",
                line={"color": color, "width": 2.5},
                marker={"size": 6},
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yr<br>"
                    f"90% width: %{{y:.1f}} pp<extra></extra>"
                ),
            ))

            # 80% width
            width_80 = list(mp["p90"] - mp["p10"])
            fig.add_trace(go.Scatter(
                x=horizons, y=width_80,
                mode="lines+markers",
                name=f"{label} (80% width: P10-P90)",
                line={"color": color, "width": 1.5, "dash": "dash"},
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yr<br>"
                    f"80% width: %{{y:.1f}} pp<extra></extra>"
                ),
            ))

        fig.update_layout(
            template=tmpl,
            title=f"Prediction Interval Width Growth: {level_label}",
            xaxis_title="Forecast Horizon (years)",
            yaxis_title="Interval Width (pp)",
            height=450,
            legend={"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center"},
        )
        parts.append(_chart_html(fig))

    # Error distribution by horizon (using prediction_intervals stats)
    parts.append("<h3>Error Distribution Statistics by Horizon</h3>")
    state_pi = pi[pi["level"] == "state"].copy()
    if not state_pi.empty:
        parts.append('<div class="table-container">')
        parts.append('<table class="data-table" id="tbl-pi-state">')
        parts.append(
            "<thead><tr>"
            "<th onclick=\"sortTable('tbl-pi-state',0)\">Method</th>"
            "<th onclick=\"sortTable('tbl-pi-state',1)\">Horizon</th>"
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',2)">Mean Error</th>'
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',3)">Std Dev</th>'
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',4)">Median (P50)</th>'
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',5)">P5</th>'
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',6)">P95</th>'
            '<th class="num" onclick="sortTable(\'tbl-pi-state\',7)">N</th>'
            "</tr></thead><tbody>"
        )
        for _, row in state_pi.sort_values(["method", "horizon"]).iterrows():
            parts.append(
                f"<tr>"
                f"<td>{_esc(_method_label(str(row['method'])))}</td>"
                f"<td>{int(row['horizon'])}</td>"
                f'<td class="num">{row["mean"]:.2f}%</td>'
                f'<td class="num">{row["std"]:.2f}</td>'
                f'<td class="num">{row["p50"]:.2f}%</td>'
                f'<td class="num">{row["p5"]:.2f}%</td>'
                f'<td class="num">{row["p95"]:.2f}%</td>'
                f'<td class="num">{int(row["n_obs"])}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 6: Uncertainty — Decomposition
# ===================================================================

def _build_uncertainty_decomposition(data: dict[str, Any], tmpl: str) -> str:
    """Build error variance decomposition and normality checks."""
    parts: list[str] = []
    parts.append("<h2>Uncertainty: Error Decomposition</h2>")
    parts.append(
        '<p class="note">Decomposition of total projection error variance into '
        "systematic bias, geographic variation, horizon effect, and residual "
        "components. Also includes normality test results.</p>"
    )

    ed = data.get("error_decomposition")
    if ed is None:
        parts.append('<p class="placeholder">Error decomposition data not available.</p>')
        return "\n".join(parts)

    # Summary rows (those with n_observations filled)
    summary = ed[ed["n_observations"].notna()].copy()

    if not summary.empty:
        parts.append("<h3>Variance Decomposition</h3>")

        fig = go.Figure()
        for _, row in summary.iterrows():
            method = str(row["method"])
            label = _method_label(method)
            components = ["Bias", "County", "Horizon", "Residual"]
            fractions = [
                float(row.get("frac_bias", 0) or 0),
                float(row.get("frac_county", 0) or 0),
                float(row.get("frac_horizon", 0) or 0),
                float(row.get("frac_residual", 0) or 0),
            ]
            fig.add_trace(go.Bar(
                x=components,
                y=[f * 100 for f in fractions],
                name=label,
                marker_color=_method_color(method),
                hovertemplate=(
                    f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"
                ),
            ))

        fig.update_layout(
            template=tmpl,
            title="Error Variance Decomposition",
            yaxis_title="Fraction of Total Variance (%)",
            height=400,
            barmode="group",
        )
        parts.append(_chart_html(fig))

        # Summary table
        parts.append("<h3>Decomposition Summary</h3>")
        parts.append('<div class="table-container">')
        parts.append('<table class="data-table" id="tbl-decomp">')
        parts.append(
            "<thead><tr>"
            "<th>Method</th>"
            '<th class="num">N Obs</th>'
            '<th class="num">Mean Error %</th>'
            '<th class="num">Total Variance</th>'
            '<th class="num">Bias^2</th>'
            '<th class="num">Random Std</th>'
            '<th class="num">Horizon R^2</th>'
            '<th class="num">Geographic Std</th>'
            "</tr></thead><tbody>"
        )
        for _, row in summary.iterrows():
            parts.append(
                f"<tr>"
                f"<td>{_esc(_method_label(str(row['method'])))}</td>"
                f'<td class="num">{int(row["n_observations"]):,}</td>'
                f'<td class="num">{row["mean_error_pct"]:.2f}%</td>'
                f'<td class="num">{row["total_variance"]:.1f}</td>'
                f'<td class="num">{row["bias_squared"]:.1f}</td>'
                f'<td class="num">{row["random_std"]:.2f}</td>'
                f'<td class="num">{row["horizon_r2"]:.3f}</td>'
                f'<td class="num">{row["geographic_std"]:.2f}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    # Horizon-level error stats (rows without n_observations but with horizon)
    horizon_rows = ed[ed["n_observations"].isna() & ed["horizon"].notna()].copy()
    if not horizon_rows.empty:
        parts.append("<h3>Error Statistics by Horizon</h3>")

        fig = go.Figure()
        for method in ["sdc_2024", "m2026"]:
            mh = horizon_rows[horizon_rows["method"] == method].sort_values("horizon")
            if mh.empty:
                continue
            label = _method_label(method)
            color = _method_color(method)

            fig.add_trace(go.Scatter(
                x=list(mh["horizon"]),
                y=list(mh["rmse_pct"]),
                mode="lines+markers",
                name=f"{label} RMSE",
                line={"color": color, "width": 2},
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yr<br>"
                    f"RMSE: %{{y:.1f}}%<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x=list(mh["horizon"]),
                y=list(mh["mae_pct"]),
                mode="lines+markers",
                name=f"{label} MAE",
                line={"color": color, "width": 1.5, "dash": "dash"},
                hovertemplate=(
                    f"<b>{label}</b><br>Horizon: %{{x}} yr<br>"
                    f"MAE: %{{y:.1f}}%<extra></extra>"
                ),
            ))

        fig.update_layout(
            template=tmpl,
            title="Error Statistics by Horizon (RMSE and MAE)",
            xaxis_title="Forecast Horizon (years)",
            yaxis_title="Error (%)",
            height=450,
            legend={"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center"},
        )
        parts.append(_chart_html(fig))

    # Normality checks
    rd = data.get("residual_diagnostics")
    if rd is not None:
        parts.append("<h3>Normality and Residual Distribution Checks</h3>")
        parts.append('<div class="table-container">')
        parts.append('<table class="data-table" id="tbl-normality">')
        parts.append(
            "<thead><tr>"
            "<th>Method</th>"
            "<th>Horizon Bucket</th>"
            '<th class="num">N Obs</th>'
            '<th class="num">Shapiro W</th>'
            '<th class="num">Shapiro p</th>'
            "<th>Normal?</th>"
            '<th class="num">Skew</th>'
            '<th class="num">Kurtosis</th>'
            '<th class="num">Mean Autocorr</th>'
            "</tr></thead><tbody>"
        )
        for _, row in rd.sort_values(["method", "horizon_bucket"]).iterrows():
            normal_cls = "positive" if row["normal_at_05"] else "negative"
            normal_text = "Yes" if row["normal_at_05"] else "No"
            parts.append(
                f"<tr>"
                f"<td>{_esc(_method_label(str(row['method'])))}</td>"
                f"<td>{_esc(str(row['horizon_bucket']))}</td>"
                f'<td class="num">{int(row["n_obs"]):,}</td>'
                f'<td class="num">{row["shapiro_w"]:.4f}</td>'
                f'<td class="num">{row["shapiro_p"]:.4f}</td>'
                f'<td class="{normal_cls}">{normal_text}</td>'
                f'<td class="num">{row["error_skew"]:.3f}</td>'
                f'<td class="num">{row["error_kurtosis"]:.2f}</td>'
                f'<td class="num">{row["mean_autocorr_lag1"]:.3f}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 7: QC — Bias Analysis
# ===================================================================

def _build_qc_bias(data: dict[str, Any], tmpl: str) -> str:
    """Build bias heatmaps and direction analysis by county type."""
    parts: list[str] = []
    parts.append("<h2>QC: Bias Analysis</h2>")
    parts.append(
        '<p class="note">Systematic bias patterns by county category and '
        "forecast horizon. Negative values indicate under-projection. "
        "Filtered to 'all' origin years for aggregate patterns.</p>"
    )

    ba = data.get("bias_analysis")
    if ba is None:
        parts.append('<p class="placeholder">Bias analysis data not available.</p>')
        return "\n".join(parts)

    # Heatmap: mean signed error by category x horizon (origin=all)
    ba_all = ba[ba["origin_year"] == "all"]

    for method in ["m2026", "sdc_2024"]:
        mb = ba_all[ba_all["method"] == method]
        if mb.empty:
            continue
        label = _method_label(method)

        categories = sorted(mb["category"].unique())
        horizons = sorted(mb["horizon"].unique())

        z = []
        hover_text = []
        for cat in categories:
            row_z = []
            row_h = []
            for h in horizons:
                cell = mb[(mb["category"] == cat) & (mb["horizon"] == h)]
                if not cell.empty:
                    val = float(cell["mean_signed_pct_error"].iloc[0])
                    n = int(cell["n_counties"].iloc[0])
                    row_z.append(val)
                    row_h.append(
                        f"{cat}<br>Horizon: {h} yr<br>"
                        f"Mean error: {val:+.1f}%<br>N counties: {n}"
                    )
                else:
                    row_z.append(np.nan)
                    row_h.append("")
            z.append(row_z)
            hover_text.append(row_h)

        fig = go.Figure(go.Heatmap(
            z=z,
            x=horizons,
            y=categories,
            colorscale="RdBu_r",
            zmid=0,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={"title": "Mean % Error"},
        ))
        fig.update_layout(
            template=tmpl,
            title=f"Bias Heatmap: {label} (Mean Signed % Error by Category x Horizon)",
            xaxis_title="Forecast Horizon (years)",
            yaxis_title="County Category",
            height=350,
        )
        parts.append(_chart_html(fig))

    # Bias direction by origin year
    parts.append("<h3>Bias by Origin Year</h3>")
    parts.append(
        '<p class="note">How systematic bias changes depending on the '
        "projection origin year.</p>"
    )

    ba_origins = ba[ba["origin_year"] != "all"]
    if not ba_origins.empty:
        for method in ["m2026", "sdc_2024"]:
            mb = ba_origins[ba_origins["method"] == method]
            if mb.empty:
                continue
            label = _method_label(method)

            fig = go.Figure()
            for cat in sorted(mb["category"].unique()):
                mc = mb[mb["category"] == cat]
                cat_color = CATEGORY_COLORS.get(cat, "#808080")
                for origin in sorted(mc["origin_year"].unique()):
                    mo = mc[mc["origin_year"] == origin].sort_values("horizon")
                    if mo.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=list(mo["horizon"]),
                        y=list(mo["mean_signed_pct_error"]),
                        mode="lines",
                        name=f"{cat} ({origin})",
                        line={"color": cat_color, "width": 1.5},
                        legendgroup=cat,
                        showlegend=(origin == sorted(mc["origin_year"].unique())[0]),
                        opacity=0.7,
                        hovertemplate=(
                            f"<b>{cat} (Origin {origin})</b><br>"
                            f"Horizon: %{{x}}<br>"
                            f"Mean error: %{{y:.1f}}%<extra></extra>"
                        ),
                    ))

            fig.add_hline(y=0, line_dash="dash", line_color=MID_GRAY, line_width=1)
            fig.update_layout(
                template=tmpl,
                title=f"Bias Trajectories by Category & Origin: {label}",
                xaxis_title="Forecast Horizon (years)",
                yaxis_title="Mean Signed % Error",
                height=500,
            )
            parts.append(_chart_html(fig))

    return "\n".join(parts)


# ===================================================================
# Tab 8: QC — County Report Cards (ALL 53 counties)
# ===================================================================

def _build_qc_report_cards(data: dict[str, Any], tmpl: str) -> str:
    """Build sortable report card table for ALL 53 counties."""
    parts: list[str] = []
    parts.append("<h2>QC: County Report Cards</h2>")
    parts.append(
        '<p class="note">Letter grades (A-D) for all 53 North Dakota counties, '
        "based on MAPE, worst-case error, and bias direction. "
        "Click column headers to sort. All 53 counties are shown for each method.</p>"
    )

    rc = data.get("county_report_cards")
    if rc is None:
        parts.append('<p class="placeholder">Report card data not available.</p>')
        return "\n".join(parts)

    # Grade distribution chart
    parts.append("<h3>Grade Distribution</h3>")
    fig = go.Figure()
    for method in ["m2026", "sdc_2024"]:
        mrc = rc[rc["method"] == method]
        grade_counts = mrc["grade"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
        fig.add_trace(go.Bar(
            x=list(grade_counts.index),
            y=list(grade_counts.values),
            name=_method_label(method),
            marker_color=_method_color(method),
            hovertemplate=(
                f"<b>{_method_label(method)}</b><br>"
                "Grade: %{x}<br>Count: %{y}<extra></extra>"
            ),
        ))
    fig.update_layout(
        template=tmpl,
        title="Grade Distribution by Method",
        xaxis_title="Grade",
        yaxis_title="Number of Counties",
        height=350,
        barmode="group",
    )
    parts.append(_chart_html(fig))

    # Tables — one per method
    for method in ["m2026", "sdc_2024"]:
        mrc = rc[rc["method"] == method].sort_values("mape")
        label = _method_label(method)
        tbl_id = f"tbl-rc-{method.replace('.', '')}"

        parts.append(f"<h3>{_esc(label)} Report Cards (All 53 Counties)</h3>")
        parts.append(
            f'<div class="table-filter">'
            f'<label>Search:</label> '
            f'<input type="text" id="filter-{tbl_id}" '
            f"oninput=\"filterTable('{tbl_id}', 'filter-{tbl_id}')\" "
            f'placeholder="Type county name...">'
            f'<label>Grade:</label> '
            f'<select id="grade-{tbl_id}" '
            f"onchange=\"filterTableByColumn('{tbl_id}', 'grade-{tbl_id}', 4)\">"
            f'<option value="all">All</option>'
            f'<option value="A">A</option>'
            f'<option value="B">B</option>'
            f'<option value="C">C</option>'
            f'<option value="D">D</option>'
            f"</select>"
            f'<span class="row-count" id="{tbl_id}-count">'
            f"{len(mrc)} of {len(mrc)} rows</span>"
            f"</div>"
        )

        parts.append(f'<div class="table-container">')
        parts.append(f'<table class="data-table" id="{tbl_id}">')
        parts.append(
            "<thead><tr>"
            f"<th onclick=\"sortTable('{tbl_id}',0)\">County</th>"
            f"<th onclick=\"sortTable('{tbl_id}',1)\">FIPS</th>"
            f"<th onclick=\"sortTable('{tbl_id}',2)\">Category</th>"
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',3)">MAPE (%)</th>'
            f"<th onclick=\"sortTable('{tbl_id}',4)\">Grade</th>"
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',5)">Mean Signed Error</th>'
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',6)">Median |Error|</th>'
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',7)">Std Error</th>'
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',8)">Worst Case</th>'
            f"<th onclick=\"sortTable('{tbl_id}',9)\">Bias Direction</th>"
            f'<th class="num" onclick="sortTable(\'{tbl_id}\',10)">N Validations</th>'
            "</tr></thead><tbody>"
        )
        for _, row in mrc.iterrows():
            grade = str(row["grade"])
            grade_clr = _grade_color(grade)
            cat_clr = CATEGORY_COLORS.get(str(row["category"]), DARK_GRAY)
            mape_val = float(row["mape"])
            err_cls = (
                "err-low" if mape_val < 5
                else "err-mid" if mape_val < 10
                else "err-high"
            )
            parts.append(
                f"<tr>"
                f"<td><strong>{_esc(str(row['county_name']))}</strong></td>"
                f"<td>{int(row['county_fips'])}</td>"
                f'<td style="color:{cat_clr}; font-weight:600;">'
                f"{_esc(str(row['category']))}</td>"
                f'<td class="num {err_cls}">{mape_val:.1f}%</td>'
                f'<td style="color:{grade_clr}; font-weight:700; font-size:16px;">'
                f"{grade}</td>"
                f'<td class="num">{row["mean_signed_error"]:+.1f}%</td>'
                f'<td class="num">{row["median_abs_error"]:.1f}%</td>'
                f'<td class="num">{row["std_error"]:.1f}</td>'
                f'<td class="num">{row["worst_case_abs_error"]:.1f}%</td>'
                f"<td>{_esc(str(row['bias_direction']))}</td>"
                f'<td class="num">{int(row["n_validations"])}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 9: QC — Outliers & Residuals
# ===================================================================

def _build_qc_outliers(data: dict[str, Any], tmpl: str) -> str:
    """Build outlier scatter, autocorrelation, and heteroscedasticity visuals."""
    parts: list[str] = []
    parts.append("<h2>QC: Outliers & Residuals</h2>")
    parts.append(
        '<p class="note">Outliers are county-year observations with |z-score| > 2 '
        "relative to the horizon-specific error distribution. Residual diagnostics "
        "check for autocorrelation and heteroscedasticity.</p>"
    )

    of = data.get("outlier_flags")
    rd = data.get("residual_diagnostics")

    if of is None:
        parts.append('<p class="placeholder">Outlier data not available.</p>')
        return "\n".join(parts)

    # Outlier scatter: z_score vs pct_error, colored by category
    parts.append("<h3>Outlier Scatter: Z-Score vs % Error</h3>")
    for method in ["m2026", "sdc_2024"]:
        mo = of[of["method"] == method]
        if mo.empty:
            continue
        label = _method_label(method)

        fig = go.Figure()
        for cat in sorted(mo["category"].unique()):
            mc = mo[mo["category"] == cat]
            cat_color = CATEGORY_COLORS.get(cat, "#808080")
            fig.add_trace(go.Scatter(
                x=list(mc["pct_error"]),
                y=list(mc["z_score"]),
                mode="markers",
                name=cat,
                marker={"color": cat_color, "size": 6, "opacity": 0.7},
                customdata=list(zip(
                    mc["county_name"],
                    mc["origin_year"],
                    mc["validation_year"],
                    mc["horizon"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Origin: %{customdata[1]} | Val: %{customdata[2]}<br>"
                    "Horizon: %{customdata[3]} yr<br>"
                    "Error: %{x:.1f}%<br>"
                    "Z-score: %{y:.2f}<extra></extra>"
                ),
            ))

        fig.add_hline(y=2, line_dash="dash", line_color=RED, line_width=1,
                       annotation_text="|z|=2")
        fig.add_hline(y=-2, line_dash="dash", line_color=RED, line_width=1)

        fig.update_layout(
            template=tmpl,
            title=f"Outlier Scatter: {label}",
            xaxis_title="% Error",
            yaxis_title="Z-Score",
            height=450,
        )
        parts.append(_chart_html(fig))

    # Outlier frequency by county (all 53 or however many have outliers)
    parts.append("<h3>Outlier Frequency by County</h3>")
    for method in ["m2026", "sdc_2024"]:
        mo = of[of["method"] == method]
        if mo.empty:
            continue
        label = _method_label(method)
        county_freq = mo.groupby(["county_name", "category"]).size().reset_index(name="n_outliers")
        county_freq = county_freq.sort_values("n_outliers", ascending=True)

        fig = go.Figure()
        for cat in sorted(county_freq["category"].unique()):
            mc = county_freq[county_freq["category"] == cat]
            fig.add_trace(go.Bar(
                y=list(mc["county_name"]),
                x=list(mc["n_outliers"]),
                orientation="h",
                name=cat,
                marker_color=CATEGORY_COLORS.get(cat, "#808080"),
                hovertemplate=(
                    "<b>%{y}</b><br>Outlier count: %{x}<extra></extra>"
                ),
            ))

        fig.update_layout(
            template=tmpl,
            title=f"Outlier Counts by County: {label}",
            xaxis_title="Number of Outlier Observations",
            yaxis_title="",
            height=max(400, len(county_freq) * 22),
            barmode="stack",
        )
        parts.append(_chart_html(fig))

    # Autocorrelation and heteroscedasticity summary
    if rd is not None:
        parts.append("<h3>Residual Diagnostics Summary</h3>")
        parts.append('<div class="table-container">')
        parts.append('<table class="data-table" id="tbl-resid">')
        parts.append(
            "<thead><tr>"
            "<th>Method</th>"
            "<th>Horizon Bucket</th>"
            '<th class="num">N Obs</th>'
            '<th class="num">Lag-1 Autocorr</th>'
            '<th class="num">Het R^2</th>'
            '<th class="num">Het p-value</th>'
            "<th>Het Significant?</th>"
            '<th class="num">Error Mean</th>'
            '<th class="num">Error Std</th>'
            "</tr></thead><tbody>"
        )
        for _, row in rd.sort_values(["method", "horizon_bucket"]).iterrows():
            het_cls = "negative" if row["het_significant"] else "positive"
            het_text = "Yes" if row["het_significant"] else "No"
            parts.append(
                f"<tr>"
                f"<td>{_esc(_method_label(str(row['method'])))}</td>"
                f"<td>{_esc(str(row['horizon_bucket']))}</td>"
                f'<td class="num">{int(row["n_obs"]):,}</td>'
                f'<td class="num">{row["mean_autocorr_lag1"]:.3f}</td>'
                f'<td class="num">{row["het_r2_vs_pop_size"]:.4f}</td>'
                f'<td class="num">{row["het_p_value"]:.4f}</td>'
                f'<td class="{het_cls}">{het_text}</td>'
                f'<td class="num">{row["error_mean"]:.2f}%</td>'
                f'<td class="num">{row["error_std"]:.2f}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 10: QC — Structural Breaks
# ===================================================================

def _build_qc_structural_breaks(data: dict[str, Any], tmpl: str) -> str:
    """Build pre-boom vs post-boom and paired method comparison."""
    parts: list[str] = []
    parts.append("<h2>QC: Structural Breaks</h2>")
    parts.append(
        '<p class="note">Comparison of projection accuracy before and after '
        "the Bakken oil boom. Pre-boom origins (2005, 2010) faced unpredictable "
        "structural change; post-boom origins (2015, 2020) project from a more "
        "stable baseline.</p>"
    )

    ba = data.get("bias_analysis")
    if ba is None:
        parts.append('<p class="placeholder">Bias analysis data not available.</p>')
        return "\n".join(parts)

    ba_by_origin = ba[ba["origin_year"] != "all"].copy()
    if ba_by_origin.empty:
        parts.append('<p class="placeholder">No origin-specific bias data.</p>')
        return "\n".join(parts)

    # Categorize origins as pre-boom vs post-boom
    ba_by_origin["era"] = ba_by_origin["origin_year"].apply(
        lambda x: "Pre-Boom (2005, 2010)" if str(x) in ("2005", "2010") else "Post-Boom (2015, 2020)"
    )

    for method in ["m2026", "sdc_2024"]:
        mb = ba_by_origin[ba_by_origin["method"] == method]
        if mb.empty:
            continue
        label = _method_label(method)

        # Aggregate across categories within each era x horizon
        era_agg = mb.groupby(["era", "horizon"]).agg(
            mean_error=("mean_signed_pct_error", "mean"),
            std_error=("std_pct_error", "mean"),
        ).reset_index()

        fig = go.Figure()
        era_colors = {
            "Pre-Boom (2005, 2010)": "#e41a1c",
            "Post-Boom (2015, 2020)": "#4daf4a",
        }

        for era in era_agg["era"].unique():
            ed = era_agg[era_agg["era"] == era].sort_values("horizon")
            fig.add_trace(go.Scatter(
                x=list(ed["horizon"]),
                y=list(ed["mean_error"]),
                mode="lines+markers",
                name=era,
                line={"color": era_colors.get(era, "#808080"), "width": 2.5},
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{era}</b><br>"
                    "Horizon: %{x} yr<br>"
                    "Mean error: %{y:.1f}%<extra></extra>"
                ),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color=MID_GRAY, line_width=1)
        fig.update_layout(
            template=tmpl,
            title=f"Pre-Boom vs Post-Boom Mean Error: {label}",
            xaxis_title="Forecast Horizon (years)",
            yaxis_title="Mean Signed % Error",
            height=450,
        )
        parts.append(_chart_html(fig))

    # Paired method comparison: SDC vs 2026 bias differential by era
    parts.append("<h3>Paired Method Comparison: Bias Differential by Era</h3>")

    ba_all = ba[ba["origin_year"] == "all"]
    methods_pivot = ba_all.pivot_table(
        values="mean_signed_pct_error",
        index=["category", "horizon"],
        columns="method",
        aggfunc="first",
    ).reset_index()

    if "sdc_2024" in methods_pivot.columns and "m2026" in methods_pivot.columns:
        methods_pivot["delta"] = methods_pivot["m2026"] - methods_pivot["sdc_2024"]

        fig = go.Figure()
        for cat in sorted(methods_pivot["category"].unique()):
            mc = methods_pivot[methods_pivot["category"] == cat].sort_values("horizon")
            fig.add_trace(go.Scatter(
                x=list(mc["horizon"]),
                y=list(mc["delta"]),
                mode="lines+markers",
                name=cat,
                line={"color": CATEGORY_COLORS.get(cat, "#808080"), "width": 2},
                hovertemplate=(
                    f"<b>{cat}</b><br>"
                    "Horizon: %{x} yr<br>"
                    "2026 - SDC delta: %{y:.1f} pp<extra></extra>"
                ),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color=MID_GRAY, line_width=1,
                       annotation_text="Methods equal")
        fig.update_layout(
            template=tmpl,
            title="Method Bias Differential: 2026 minus SDC (positive = 2026 more optimistic)",
            xaxis_title="Forecast Horizon (years)",
            yaxis_title="Bias Differential (pp)",
            height=450,
        )
        parts.append(_chart_html(fig))

    return "\n".join(parts)


# ===================================================================
# Tab 11: QC — County Deep-Dive (box plots by category, all counties)
# ===================================================================

def _build_qc_county_deep_dive(data: dict[str, Any], tmpl: str) -> str:
    """Build box plots by category and county-level detail for all 53 counties."""
    parts: list[str] = []
    parts.append("<h2>QC: County Deep-Dive</h2>")
    parts.append(
        '<p class="note">Box plots showing the distribution of MAPE within each '
        "county category, plus detailed county-level comparisons. All 53 counties "
        "are included.</p>"
    )

    rc = data.get("county_report_cards")
    if rc is None:
        parts.append('<p class="placeholder">Report card data not available.</p>')
        return "\n".join(parts)

    # Box plots by category
    parts.append("<h3>MAPE Distribution by Category</h3>")
    fig = go.Figure()
    for method in ["m2026", "sdc_2024"]:
        mrc = rc[rc["method"] == method]
        label = _method_label(method)
        fig.add_trace(go.Box(
            x=list(mrc["category"]),
            y=list(mrc["mape"]),
            name=label,
            marker_color=_method_color(method),
            boxmean=True,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Category: %{x}<br>"
                "MAPE: %{y:.1f}%<extra></extra>"
            ),
        ))
    fig.update_layout(
        template=tmpl,
        title="MAPE Distribution by County Category (All 53 Counties)",
        yaxis_title="MAPE (%)",
        height=450,
        boxmode="group",
    )
    parts.append(_chart_html(fig))

    # MAPE bar chart: all 53 counties, sorted
    parts.append("<h3>MAPE by County (All 53 Counties)</h3>")
    for method in ["m2026", "sdc_2024"]:
        mrc = rc[rc["method"] == method].sort_values("mape", ascending=True)
        label = _method_label(method)

        fig = go.Figure()
        colors = [CATEGORY_COLORS.get(str(cat), "#808080") for cat in mrc["category"]]
        fig.add_trace(go.Bar(
            y=list(mrc["county_name"]),
            x=list(mrc["mape"]),
            orientation="h",
            marker_color=colors,
            customdata=list(zip(
                mrc["category"],
                mrc["grade"],
                mrc["worst_case_abs_error"],
                mrc["bias_direction"],
            )),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Category: %{customdata[0]}<br>"
                "MAPE: %{x:.1f}%<br>"
                "Grade: %{customdata[1]}<br>"
                "Worst case: %{customdata[2]:.1f}%<br>"
                "Bias: %{customdata[3]}<extra></extra>"
            ),
        ))

        # Grade threshold lines
        fig.add_vline(x=5, line_dash="dash", line_color=GREEN, line_width=1,
                       annotation_text="A/B boundary")
        fig.add_vline(x=10, line_dash="dash", line_color=ORANGE, line_width=1,
                       annotation_text="B/C boundary")

        fig.update_layout(
            template=tmpl,
            title=f"MAPE by County: {label} (All 53 Counties)",
            xaxis_title="MAPE (%)",
            yaxis_title="",
            height=max(600, len(mrc) * 20),
        )
        parts.append(_chart_html(fig))

    # Grade comparison: side-by-side
    parts.append("<h3>Grade Comparison: 2026 vs SDC</h3>")
    m2026_rc = rc[rc["method"] == "m2026"][["county_name", "grade", "mape"]].rename(
        columns={"grade": "grade_m2026", "mape": "mape_m2026"}
    )
    sdc_rc = rc[rc["method"] == "sdc_2024"][["county_name", "grade", "mape"]].rename(
        columns={"grade": "grade_sdc", "mape": "mape_sdc"}
    )
    merged = pd.merge(m2026_rc, sdc_rc, on="county_name", how="outer")
    merged = merged.sort_values("county_name")

    tbl_id = "tbl-grade-compare"
    parts.append(
        f'<div class="table-filter">'
        f'<label>Search:</label> '
        f'<input type="text" id="filter-{tbl_id}" '
        f"oninput=\"filterTable('{tbl_id}', 'filter-{tbl_id}')\" "
        f'placeholder="Type county name...">'
        f'<span class="row-count" id="{tbl_id}-count">'
        f"{len(merged)} of {len(merged)} rows</span>"
        f"</div>"
    )
    parts.append(f'<div class="table-container">')
    parts.append(f'<table class="data-table" id="{tbl_id}">')
    parts.append(
        "<thead><tr>"
        f"<th onclick=\"sortTable('{tbl_id}',0)\">County</th>"
        f"<th onclick=\"sortTable('{tbl_id}',1)\">2026 Grade</th>"
        f'<th class="num" onclick="sortTable(\'{tbl_id}\',2)">2026 MAPE</th>'
        f"<th onclick=\"sortTable('{tbl_id}',3)\">SDC Grade</th>"
        f'<th class="num" onclick="sortTable(\'{tbl_id}\',4)">SDC MAPE</th>'
        f"<th onclick=\"sortTable('{tbl_id}',5)\">Better Method</th>"
        "</tr></thead><tbody>"
    )
    for _, row in merged.iterrows():
        g2026 = str(row.get("grade_m2026", ""))
        gsdc = str(row.get("grade_sdc", ""))
        mape2026 = float(row.get("mape_m2026", 0))
        mapesdc = float(row.get("mape_sdc", 0))
        if mape2026 < mapesdc:
            better = "2026 Method"
            better_color = _method_color("m2026")
        elif mapesdc < mape2026:
            better = "SDC 2024"
            better_color = _method_color("sdc_2024")
        else:
            better = "Tie"
            better_color = DARK_GRAY
        parts.append(
            f"<tr>"
            f"<td><strong>{_esc(str(row['county_name']))}</strong></td>"
            f'<td style="color:{_grade_color(g2026)}; font-weight:700;">{g2026}</td>'
            f'<td class="num">{mape2026:.1f}%</td>'
            f'<td style="color:{_grade_color(gsdc)}; font-weight:700;">{gsdc}</td>'
            f'<td class="num">{mapesdc:.1f}%</td>'
            f'<td style="color:{better_color}; font-weight:600;">{better}</td>'
            f"</tr>"
        )
    parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# Tab 12: Raw Data Tables
# ===================================================================

def _build_raw_data_tables(data: dict[str, Any]) -> str:
    """Build sortable/filterable raw data tables for key datasets."""
    parts: list[str] = []
    parts.append("<h2>Raw Data Tables</h2>")
    parts.append(
        '<p class="note">Sortable and filterable tables of the underlying data. '
        "Click column headers to sort; use the search box to filter.</p>"
    )

    # --- Sensitivity Results ---
    sr = data.get("sensitivity_results")
    if sr is not None:
        tbl_id = "tbl-raw-sensitivity"
        parts.append("<h3>Sensitivity Results</h3>")
        parts.append(
            f'<div class="table-filter">'
            f'<label>Search:</label> '
            f'<input type="text" id="filter-{tbl_id}" '
            f"oninput=\"filterTable('{tbl_id}', 'filter-{tbl_id}')\" "
            f'placeholder="Filter...">'
            f'<span class="row-count" id="{tbl_id}-count">'
            f"{len(sr)} of {len(sr)} rows</span>"
            f"</div>"
        )
        parts.append(f'<div class="table-container">')
        parts.append(f'<table class="data-table" id="{tbl_id}">')
        parts.append(
            "<thead><tr>"
            + "".join(
                f"<th onclick=\"sortTable('{tbl_id}',{i})\">{_esc(col)}</th>"
                for i, col in enumerate(sr.columns)
            )
            + "</tr></thead><tbody>"
        )
        for _, row in sr.iterrows():
            parts.append("<tr>" + "".join(
                f'<td class="num">{v:.4f}</td>' if isinstance(v, float)
                else f"<td>{_esc(str(v))}</td>"
                for v in row
            ) + "</tr>")
        parts.append("</tbody></table></div>")

    # --- Tornado Data ---
    tornado = data.get("sensitivity_tornado")
    if tornado is not None:
        tbl_id = "tbl-raw-tornado"
        parts.append("<h3>Sensitivity Tornado Data</h3>")
        parts.append(f'<div class="table-container">')
        parts.append(f'<table class="data-table" id="{tbl_id}">')
        parts.append(
            "<thead><tr>"
            + "".join(
                f"<th onclick=\"sortTable('{tbl_id}',{i})\">{_esc(col)}</th>"
                for i, col in enumerate(tornado.columns)
            )
            + "</tr></thead><tbody>"
        )
        for _, row in tornado.iterrows():
            parts.append("<tr>" + "".join(
                f'<td class="num">{v:.4f}</td>' if isinstance(v, float)
                else f"<td>{_esc(str(v))}</td>"
                for v in row
            ) + "</tr>")
        parts.append("</tbody></table></div>")

    # --- County Report Cards ---
    rc = data.get("county_report_cards")
    if rc is not None:
        tbl_id = "tbl-raw-rc"
        parts.append("<h3>County Report Cards (All 53 Counties x 2 Methods)</h3>")
        parts.append(
            f'<div class="table-filter">'
            f'<label>Search:</label> '
            f'<input type="text" id="filter-{tbl_id}" '
            f"oninput=\"filterTable('{tbl_id}', 'filter-{tbl_id}')\" "
            f'placeholder="Filter...">'
            f'<span class="row-count" id="{tbl_id}-count">'
            f"{len(rc)} of {len(rc)} rows</span>"
            f"</div>"
        )
        parts.append(f'<div class="table-container">')
        parts.append(f'<table class="data-table" id="{tbl_id}">')
        parts.append(
            "<thead><tr>"
            + "".join(
                f"<th onclick=\"sortTable('{tbl_id}',{i})\">{_esc(col)}</th>"
                for i, col in enumerate(rc.columns)
            )
            + "</tr></thead><tbody>"
        )
        for _, row in rc.iterrows():
            parts.append("<tr>" + "".join(
                f'<td class="num">{v:.2f}</td>' if isinstance(v, float)
                else f"<td>{_esc(str(v))}</td>"
                for v in row
            ) + "</tr>")
        parts.append("</tbody></table></div>")

    # --- Outlier Flags ---
    of = data.get("outlier_flags")
    if of is not None:
        tbl_id = "tbl-raw-outliers"
        parts.append("<h3>Outlier Flags</h3>")
        parts.append(
            f'<div class="table-filter">'
            f'<label>Search:</label> '
            f'<input type="text" id="filter-{tbl_id}" '
            f"oninput=\"filterTable('{tbl_id}', 'filter-{tbl_id}')\" "
            f'placeholder="Filter...">'
            f'<span class="row-count" id="{tbl_id}-count">'
            f"{len(of)} of {len(of)} rows</span>"
            f"</div>"
        )
        parts.append(f'<div class="table-container">')
        parts.append(f'<table class="data-table" id="{tbl_id}">')
        parts.append(
            "<thead><tr>"
            + "".join(
                f"<th onclick=\"sortTable('{tbl_id}',{i})\">{_esc(col)}</th>"
                for i, col in enumerate(of.columns)
            )
            + "</tr></thead><tbody>"
        )
        for _, row in of.iterrows():
            parts.append("<tr>" + "".join(
                f'<td class="num">{v:.4f}</td>' if isinstance(v, float)
                else f"<td>{_esc(str(v))}</td>"
                for v in row
            ) + "</tr>")
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ===================================================================
# HTML Assembly
# ===================================================================

TAB_DEFS = [
    ("overview", "Overview"),
    ("tornado", "Sensitivity: Tornado"),
    ("sweeps", "Sensitivity: Sweeps"),
    ("fan", "Uncertainty: Fan Charts"),
    ("intervals", "Uncertainty: Intervals"),
    ("decomposition", "Uncertainty: Decomposition"),
    ("bias", "QC: Bias Analysis"),
    ("report_cards", "QC: Report Cards"),
    ("outliers", "QC: Outliers & Residuals"),
    ("structural", "QC: Structural Breaks"),
    ("county_deep", "QC: County Deep-Dive"),
    ("raw_data", "Raw Data"),
]


def _build_html(tabs: dict[str, str]) -> str:
    """Assemble the full self-contained HTML document."""
    subtitle = (
        f"Sensitivity + Uncertainty + QC Diagnostics | "
        f"SDC 2024 vs. 2026 Method | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    # Build tab nav buttons
    tab_buttons = []
    for i, (tab_id, tab_label) in enumerate(TAB_DEFS):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-btn{active}" '
            f"onclick=\"switchTab('{tab_id}')\">{tab_label}</button>"
        )
    tab_nav_html = "\n        ".join(tab_buttons)

    # Build tab content divs
    tab_content_html = []
    for i, (tab_id, _) in enumerate(TAB_DEFS):
        active = " active" if i == 0 else ""
        content = tabs.get(tab_id, '<p class="placeholder">No content available.</p>')
        tab_content_html.append(
            f'    <div id="tab-{tab_id}" class="tab-content{active}">\n'
            f"        {content}\n"
            f"    </div>"
        )
    tab_divs_html = "\n\n".join(tab_content_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QC Analysis Report - {TODAY.isoformat()}</title>
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
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QC Analysis Report: Sensitivity, Uncertainty & Diagnostics</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="tab-nav">
        {tab_nav_html}
    </div>

{tab_divs_html}

    <div class="footer">
        North Dakota Population Projections |
        QC Analysis Report |
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

            var th = table.querySelectorAll('thead th')[colIdx];
            var ascending = !th.classList.contains('sort-asc');

            table.querySelectorAll('thead th').forEach(function(h) {{
                h.classList.remove('sort-asc', 'sort-desc');
            }});
            th.classList.add(ascending ? 'sort-asc' : 'sort-desc');

            rows.sort(function(a, b) {{
                var aText = a.cells[colIdx].textContent.trim();
                var bText = b.cells[colIdx].textContent.trim();

                var aClean = aText.replace(/[%,+]/g, '');
                var bClean = bText.replace(/[%,+]/g, '');
                var aVal = parseFloat(aClean);
                var bVal = parseFloat(bClean);

                if (!isNaN(aVal) && !isNaN(bVal)) {{
                    return ascending ? aVal - bVal : bVal - aVal;
                }}
                return ascending
                    ? aText.localeCompare(bText)
                    : bText.localeCompare(aText);
            }});

            rows.forEach(function(row) {{ tbody.appendChild(row); }});
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
    """Build the complete QC analysis HTML report.

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
    logger.info("QC Analysis Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    # Register Plotly template
    tmpl = _get_plotly_template()

    # Load data
    logger.info("Loading analysis data...")
    data = load_all_data()
    for key, df in data.items():
        if df is not None:
            logger.info("  %s: %d rows x %d cols", key, len(df), len(df.columns))
        else:
            logger.warning("  %s: NOT FOUND", key)

    # Build tabs
    logger.info("Building Tab 1: Overview...")
    tab_overview = _build_overview(data)

    logger.info("Building Tab 2: Sensitivity Tornado...")
    tab_tornado = _build_sensitivity_tornado(data, tmpl)

    logger.info("Building Tab 3: Sensitivity Sweeps...")
    tab_sweeps = _build_sensitivity_sweeps(data, tmpl)

    logger.info("Building Tab 4: Uncertainty Fan Charts...")
    tab_fan = _build_uncertainty_fan_charts(data, tmpl)

    logger.info("Building Tab 5: Uncertainty Intervals...")
    tab_intervals = _build_uncertainty_intervals(data, tmpl)

    logger.info("Building Tab 6: Uncertainty Decomposition...")
    tab_decomposition = _build_uncertainty_decomposition(data, tmpl)

    logger.info("Building Tab 7: QC Bias Analysis...")
    tab_bias = _build_qc_bias(data, tmpl)

    logger.info("Building Tab 8: QC Report Cards...")
    tab_report_cards = _build_qc_report_cards(data, tmpl)

    logger.info("Building Tab 9: QC Outliers & Residuals...")
    tab_outliers = _build_qc_outliers(data, tmpl)

    logger.info("Building Tab 10: QC Structural Breaks...")
    tab_structural = _build_qc_structural_breaks(data, tmpl)

    logger.info("Building Tab 11: QC County Deep-Dive...")
    tab_county_deep = _build_qc_county_deep_dive(data, tmpl)

    logger.info("Building Tab 12: Raw Data Tables...")
    tab_raw_data = _build_raw_data_tables(data)

    # Assemble HTML
    logger.info("Assembling HTML report...")
    tabs = {
        "overview": tab_overview,
        "tornado": tab_tornado,
        "sweeps": tab_sweeps,
        "fan": tab_fan,
        "intervals": tab_intervals,
        "decomposition": tab_decomposition,
        "bias": tab_bias,
        "report_cards": tab_report_cards,
        "outliers": tab_outliers,
        "structural": tab_structural,
        "county_deep": tab_county_deep,
        "raw_data": tab_raw_data,
    }
    html = _build_html(tabs)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Report generated successfully!")
    logger.info("  Output: %s", output_path)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  Tabs: %d (%s)", len(TAB_DEFS), ", ".join(t[1] for t in TAB_DEFS))
    logger.info("=" * 60)

    return output_path


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build unified QC analysis HTML report from walk-forward "
            "validation sensitivity, uncertainty, and QC diagnostic outputs."
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
    """Entry point for the QC analysis report builder."""
    args = parse_args()

    try:
        output_path = build_report(output_path=args.output)
        logger.info("Done. Report at: %s", output_path)
        return 0
    except Exception:
        logger.exception("Failed to build QC analysis report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
