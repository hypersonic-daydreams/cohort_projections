#!/usr/bin/env python3
"""
Build interactive HTML report comparing SDC 2024 projections with our cohort-component projections.

Creates a single self-contained HTML file with Plotly charts and tabbed navigation
across four sections: State-Level Comparison, County-Level Comparison,
Population Pyramids, and Growth Rate Analysis.

Output: data/exports/nd_sdc_comparison_{datestamp}.html

Usage:
    python scripts/exports/build_sdc_comparison_report.py
    python scripts/exports/build_sdc_comparison_report.py --output-dir data/exports/

Data Sources:
    Our projections:
        - data/exports/{scenario}/summaries/state_total_population_by_year.csv
        - data/exports/{scenario}/summaries/county_total_population_by_year.csv
        - data/exports/{scenario}/summaries/state_age_distribution_by_year.csv
    SDC 2024 projections:
        - data/raw/nd_sdc_2024_projections/state_projections.csv
        - data/raw/nd_sdc_2024_projections/county_projections.csv
        - data/raw/nd_sdc_2024_projections/state_age_sex_total.csv
        - data/raw/nd_sdc_2024_projections/state_age_sex_male.csv
        - data/raw/nd_sdc_2024_projections/state_age_sex_female.csv
    County FIPS mapping:
        - data/raw/population/nd_county_population.csv

Methodology:
    Comparison is limited to the overlap period 2025-2050 (SDC horizon ends at 2050).
    CAGR = (P_end / P_start)^(1 / years) - 1.
    County matching uses FIPS codes; SDC county names are stripped of " County" suffix
    and matched via the nd_county_population.csv crosswalk.

Key ADRs:
    ADR-037: CBO-grounded scenario methodology
    ADR-054: State-county aggregation

Processing Steps:
    1. Load SDC state and county projections
    2. Load our projection exports (baseline, high_growth, restricted_growth)
    3. Build FIPS-to-county-name crosswalk
    4. Generate four tabbed sections with Plotly charts
    5. Assemble into self-contained HTML with embedded CSS/JS

SOP-002 Metadata:
    Author: Claude Code (automated)
    Created: 2026-03-02
    Input files: See Data Sources above
    Output files: data/exports/nd_sdc_comparison_{datestamp}.html
    Dependencies: pandas, plotly
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

# Scenario definitions
SCENARIOS = {
    "baseline": "Baseline",
    "high_growth": "High Growth",
    "restricted_growth": "Restricted Growth",
}

SCENARIO_COLORS = {
    "baseline": "#0563C1",
    "high_growth": "#00B050",
    "restricted_growth": "#FF0000",
}

SDC_COLOR = "#FF8C00"

# Theme constants
NAVY = "#1F3864"
BLUE = "#0563C1"
DARK_GRAY = "#595959"
MID_GRAY = "#D9D9D9"
LIGHT_GRAY = "#F2F2F2"
WHITE = "#FFFFFF"
FONT_FAMILY = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Overlap years between SDC (2025-2050) and ours (2025-2055)
SDC_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
KEY_YEARS = [2030, 2040, 2050]

# SDC age groups from their data files
SDC_AGE_GROUPS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85+",
]

# Our broader age groups from state_age_distribution_by_year.csv
OUR_AGE_GROUPS = ["0-4", "5-17", "18-24", "25-44", "45-64", "65-74", "75-84", "85+"]


# ===================================================================
# Plotly template
# ===================================================================

def _get_plotly_template() -> str:
    """Register and return the SDC comparison template name."""
    template_name = "sdc_comparison"
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
        colorway=[BLUE, SDC_COLOR, "#00B050", "#FF0000", "#1F3864"],
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


def _fmt_change(n: float | int) -> str:
    """Format an absolute change with +/- sign and commas."""
    val = int(round(n))
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,}"


def _cagr(start: float, end: float, years: int) -> float:
    """Compound annual growth rate."""
    if start <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


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


def load_fips_mapping() -> dict[str, str]:
    """Build a mapping from county name (without ' County') to FIPS code.

    Returns dict like {'Adams': '38001', 'Barnes': '38003', ...}
    """
    path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    df = _load_csv_safe(path)
    if df is None:
        logger.error("Cannot load FIPS mapping from %s", path)
        return {}

    mapping = {}
    for _, row in df.iterrows():
        fips = str(int(row["county_fips"])).zfill(5)
        name = str(row["county_name"]).strip()
        mapping[name] = fips

    return mapping


def load_sdc_data() -> dict[str, Any]:
    """Load all SDC 2024 projection data."""
    sdc_dir = PROJECT_ROOT / "data" / "raw" / "nd_sdc_2024_projections"
    data: dict[str, Any] = {}

    # State totals: columns year, total_population
    data["state_totals"] = _load_csv_safe(sdc_dir / "state_projections.csv")

    # County totals: columns county_name, 2020, 2025, ..., 2050
    data["county_totals"] = _load_csv_safe(sdc_dir / "county_projections.csv")

    # Age/sex data
    data["age_sex_total"] = _load_csv_safe(sdc_dir / "state_age_sex_total.csv")
    data["age_sex_male"] = _load_csv_safe(sdc_dir / "state_age_sex_male.csv")
    data["age_sex_female"] = _load_csv_safe(sdc_dir / "state_age_sex_female.csv")

    return data


def load_our_data() -> dict[str, dict[str, pd.DataFrame | None]]:
    """Load our projection exports for all three scenarios."""
    exports_dir = PROJECT_ROOT / "data" / "exports"
    all_data: dict[str, dict[str, pd.DataFrame | None]] = {}

    for scenario in SCENARIOS:
        sdir = exports_dir / scenario / "summaries"
        sdata: dict[str, pd.DataFrame | None] = {}

        sdata["state_totals"] = _load_csv_safe(
            sdir / "state_total_population_by_year.csv"
        )
        sdata["county_totals"] = _load_csv_safe(
            sdir / "county_total_population_by_year.csv"
        )
        sdata["state_age_dist"] = _load_csv_safe(
            sdir / "state_age_distribution_by_year.csv"
        )

        all_data[scenario] = sdata

    return all_data


def _our_state_series(our_data: dict, scenario: str) -> pd.Series | None:
    """Extract a year->population Series from our state totals for a scenario.

    Returns a Series indexed by integer year, filtered to SDC overlap years.
    """
    df = our_data.get(scenario, {}).get("state_totals")
    if df is None or df.empty:
        return None

    # Data format: fips, 2025, 2026, ..., 2055 (columns are year strings)
    row = df.iloc[0]  # Single state row
    year_cols = [str(y) for y in range(2025, 2056)]
    series = pd.Series(dtype=float)
    for yc in year_cols:
        if yc in row.index:
            series[int(yc)] = float(row[yc])
    return series


def _our_county_dict(our_data: dict, scenario: str) -> dict[str, pd.Series]:
    """Extract per-county year->population Series from our county totals.

    Returns dict keyed by FIPS code (str), each value a Series indexed by year.
    """
    df = our_data.get(scenario, {}).get("county_totals")
    if df is None or df.empty:
        return {}

    result = {}
    year_cols = [str(y) for y in range(2025, 2056)]
    for _, row in df.iterrows():
        fips = str(int(row["fips"])).zfill(5)
        series = pd.Series(dtype=float)
        for yc in year_cols:
            if yc in row.index:
                series[int(yc)] = float(row[yc])
        result[fips] = series
    return result


def _sdc_state_series(sdc_data: dict) -> pd.Series | None:
    """Extract a year->population Series from SDC state totals."""
    df = sdc_data.get("state_totals")
    if df is None or df.empty:
        return None
    return pd.Series(df["total_population"].values, index=df["year"].astype(int).values)


def _sdc_county_dict(sdc_data: dict, fips_map: dict[str, str]) -> dict[str, pd.Series]:
    """Extract per-county year->population Series from SDC county totals.

    Returns dict keyed by FIPS code, each value a Series indexed by year.
    """
    df = sdc_data.get("county_totals")
    if df is None or df.empty:
        return {}

    year_cols = ["2020", "2025", "2030", "2035", "2040", "2045", "2050"]
    result = {}
    for _, row in df.iterrows():
        # SDC uses "Adams County", "Barnes County" etc.
        name = str(row["county_name"]).replace(" County", "").strip()
        fips = fips_map.get(name)
        if fips is None:
            logger.warning("No FIPS mapping for SDC county: %s", name)
            continue
        series = pd.Series(dtype=float)
        for yc in year_cols:
            if yc in row.index:
                series[int(yc)] = float(row[yc])
        result[fips] = series

    return result


# ===================================================================
# Section Builders
# ===================================================================

def _build_state_comparison(
    sdc_data: dict, our_data: dict, template_name: str,
) -> str:
    """Build Tab 1: State-Level Comparison."""
    parts = ['<h2>State-Level Comparison</h2>']

    sdc_state = _sdc_state_series(sdc_data)
    our_baseline = _our_state_series(our_data, "baseline")
    our_high = _our_state_series(our_data, "high_growth")
    our_restricted = _our_state_series(our_data, "restricted_growth")

    if sdc_state is None or our_baseline is None:
        parts.append('<p class="placeholder">State data not available.</p>')
        return "\n".join(parts)

    # --- Trend Lines ---
    fig = go.Figure()

    # SDC
    sdc_years = sorted(sdc_state.index)
    # Filter to 2025+
    sdc_plot = sdc_state[sdc_state.index >= 2025].sort_index()
    fig.add_trace(go.Scatter(
        x=list(sdc_plot.index),
        y=list(sdc_plot.values),
        name="SDC 2024",
        mode="lines+markers",
        line={"color": SDC_COLOR, "width": 3, "dash": "dash"},
        marker={"size": 8, "symbol": "diamond"},
        hovertemplate="<b>SDC 2024</b><br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>",
    ))

    # Our scenarios
    for scenario, label, color in [
        ("baseline", "Our Baseline", SCENARIO_COLORS["baseline"]),
        ("high_growth", "Our High Growth", SCENARIO_COLORS["high_growth"]),
        ("restricted_growth", "Our Restricted", SCENARIO_COLORS["restricted_growth"]),
    ]:
        series = _our_state_series(our_data, scenario)
        if series is None:
            continue
        # Filter to overlap period
        overlap = series[(series.index >= 2025) & (series.index <= 2050)]
        fig.add_trace(go.Scatter(
            x=list(overlap.index),
            y=list(overlap.values),
            name=label,
            mode="lines",
            line={"color": color, "width": 2.5},
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Population: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        template=template_name,
        title="State Population: SDC 2024 vs Our Projections (2025-2050)",
        xaxis_title="Year",
        yaxis_title="Total Population",
        yaxis_tickformat=",",
        height=480,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Growth Rate Comparison Table ---
    parts.append("<h3>Growth Rate Comparison (2025-2050)</h3>")

    sdc_2025 = float(sdc_state.get(2025, 0))
    sdc_2050 = float(sdc_state.get(2050, 0))
    sdc_growth_pct = ((sdc_2050 - sdc_2025) / sdc_2025 * 100) if sdc_2025 > 0 else 0

    rows_html = []
    rows_html.append(f"""
        <tr>
            <td><span class="color-dot" style="background:{SDC_COLOR};"></span>SDC 2024</td>
            <td class="num">{_fmt_pop(sdc_2025)}</td>
            <td class="num">{_fmt_pop(sdc_2050)}</td>
            <td class="num">{_fmt_change(sdc_2050 - sdc_2025)}</td>
            <td class="num">{_fmt_pct(sdc_growth_pct)}</td>
            <td class="num">{_fmt_pct(_cagr(sdc_2025, sdc_2050, 25) * 100, 2)}</td>
        </tr>
    """)

    for scenario, label, color in [
        ("baseline", "Our Baseline", SCENARIO_COLORS["baseline"]),
        ("high_growth", "Our High Growth", SCENARIO_COLORS["high_growth"]),
        ("restricted_growth", "Our Restricted", SCENARIO_COLORS["restricted_growth"]),
    ]:
        series = _our_state_series(our_data, scenario)
        if series is None:
            continue
        p2025 = float(series.get(2025, 0))
        p2050 = float(series.get(2050, 0))
        growth_pct = ((p2050 - p2025) / p2025 * 100) if p2025 > 0 else 0
        rows_html.append(f"""
            <tr>
                <td><span class="color-dot" style="background:{color};"></span>{label}</td>
                <td class="num">{_fmt_pop(p2025)}</td>
                <td class="num">{_fmt_pop(p2050)}</td>
                <td class="num">{_fmt_change(p2050 - p2025)}</td>
                <td class="num">{_fmt_pct(growth_pct)}</td>
                <td class="num">{_fmt_pct(_cagr(p2025, p2050, 25) * 100, 2)}</td>
            </tr>
        """)

    parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Source</th>
                    <th class="num">2025</th>
                    <th class="num">2050</th>
                    <th class="num">Change</th>
                    <th class="num">Growth %</th>
                    <th class="num">CAGR</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        </div>
    """)

    # --- Difference Table (Our Baseline - SDC) ---
    parts.append("<h3>Population Difference: Our Baseline minus SDC</h3>")

    diff_rows = []
    for yr in SDC_YEARS:
        sdc_val = float(sdc_state.get(yr, 0))
        our_val = float(our_baseline.get(yr, 0))
        diff = our_val - sdc_val
        diff_pct = (diff / sdc_val * 100) if sdc_val > 0 else 0
        diff_rows.append(f"""
            <tr>
                <td class="num">{yr}</td>
                <td class="num">{_fmt_pop(sdc_val)}</td>
                <td class="num">{_fmt_pop(our_val)}</td>
                <td class="num {'positive' if diff >= 0 else 'negative'}">{_fmt_change(diff)}</td>
                <td class="num {'positive' if diff >= 0 else 'negative'}">{_fmt_pct(diff_pct)}</td>
            </tr>
        """)

    parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th class="num">Year</th>
                    <th class="num">SDC 2024</th>
                    <th class="num">Our Baseline</th>
                    <th class="num">Difference</th>
                    <th class="num">Diff %</th>
                </tr>
            </thead>
            <tbody>
                {"".join(diff_rows)}
            </tbody>
        </table>
        </div>
    """)

    # --- Bar chart at key years ---
    fig2 = go.Figure()

    abs_diffs = []
    pct_diffs = []
    for yr in KEY_YEARS:
        sdc_val = float(sdc_state.get(yr, 0))
        our_val = float(our_baseline.get(yr, 0))
        abs_diffs.append(our_val - sdc_val)
        pct_diffs.append((our_val - sdc_val) / sdc_val * 100 if sdc_val > 0 else 0)

    bar_colors = ["#00B050" if d >= 0 else "#FF0000" for d in abs_diffs]

    fig2.add_trace(go.Bar(
        x=[str(y) for y in KEY_YEARS],
        y=abs_diffs,
        name="Absolute Difference",
        marker_color=bar_colors,
        text=[_fmt_change(d) for d in abs_diffs],
        textposition="outside",
        hovertemplate="Year: %{x}<br>Difference: %{y:,.0f}<extra></extra>",
    ))

    fig2.update_layout(
        template=template_name,
        title="Our Baseline vs SDC: Absolute Difference at Key Years",
        xaxis_title="Year",
        yaxis_title="Population Difference (Our - SDC)",
        yaxis_tickformat=",",
        height=380,
        showlegend=False,
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig2, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # Percentage difference bar chart
    fig3 = go.Figure()
    pct_colors = ["#00B050" if d >= 0 else "#FF0000" for d in pct_diffs]

    fig3.add_trace(go.Bar(
        x=[str(y) for y in KEY_YEARS],
        y=pct_diffs,
        name="% Difference",
        marker_color=pct_colors,
        text=[f"{d:+.1f}%" for d in pct_diffs],
        textposition="outside",
        hovertemplate="Year: %{x}<br>Difference: %{y:.1f}%<extra></extra>",
    ))

    fig3.update_layout(
        template=template_name,
        title="Our Baseline vs SDC: Percentage Difference at Key Years",
        xaxis_title="Year",
        yaxis_title="Difference (%)",
        height=380,
        showlegend=False,
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig3, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    return "\n".join(parts)


def _build_county_comparison(
    sdc_data: dict,
    our_data: dict,
    fips_map: dict[str, str],
    template_name: str,
) -> str:
    """Build Tab 2: County-Level Comparison."""
    parts = ['<h2>County-Level Comparison</h2>']

    # Build reverse mapping: FIPS -> county name
    name_map = {v: k for k, v in fips_map.items()}

    sdc_counties = _sdc_county_dict(sdc_data, fips_map)
    our_counties = _our_county_dict(our_data, "baseline")
    our_counties_hg = _our_county_dict(our_data, "high_growth")
    our_counties_rg = _our_county_dict(our_data, "restricted_growth")

    if not sdc_counties or not our_counties:
        parts.append('<p class="placeholder">County data not available.</p>')
        return "\n".join(parts)

    # Get sorted list of all counties with both datasets
    common_fips = sorted(set(sdc_counties.keys()) & set(our_counties.keys()))

    # --- Per-county dropdown chart ---
    # Build chart data for each county as separate traces, toggled by JS dropdown
    county_options = []
    for fips in common_fips:
        cname = name_map.get(fips, fips)
        county_options.append({"fips": fips, "name": cname})

    # Build a figure with dropdown
    fig = go.Figure()

    # Add traces for each county (all invisible except the first)
    first_fips = common_fips[0] if common_fips else None

    for i, fips in enumerate(common_fips):
        visible = (i == 0)
        cname = name_map.get(fips, fips)

        # SDC trace
        sdc_s = sdc_counties[fips]
        sdc_plot = sdc_s[sdc_s.index >= 2025].sort_index()
        fig.add_trace(go.Scatter(
            x=list(sdc_plot.index),
            y=list(sdc_plot.values),
            name="SDC 2024",
            mode="lines+markers",
            line={"color": SDC_COLOR, "width": 3, "dash": "dash"},
            marker={"size": 8, "symbol": "diamond"},
            visible=visible,
            legendgroup="sdc",
            showlegend=(i == 0),
            hovertemplate=f"<b>SDC 2024 - {cname}</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>",
        ))

        # Our scenarios
        for scenario, label, color, lg in [
            ("baseline", "Our Baseline", SCENARIO_COLORS["baseline"], "baseline"),
            ("high_growth", "Our High Growth", SCENARIO_COLORS["high_growth"], "high"),
            ("restricted_growth", "Our Restricted", SCENARIO_COLORS["restricted_growth"], "restricted"),
        ]:
            county_dict = _our_county_dict(our_data, scenario)
            if fips not in county_dict:
                continue
            our_s = county_dict[fips]
            our_plot = our_s[(our_s.index >= 2025) & (our_s.index <= 2050)].sort_index()
            fig.add_trace(go.Scatter(
                x=list(our_plot.index),
                y=list(our_plot.values),
                name=label,
                mode="lines",
                line={"color": color, "width": 2},
                visible=visible,
                legendgroup=lg,
                showlegend=(i == 0),
                hovertemplate=f"<b>{label} - {cname}</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>",
            ))

    # Build dropdown buttons
    n_traces_per_county = 4  # SDC + 3 scenarios
    buttons = []
    for i, fips in enumerate(common_fips):
        cname = name_map.get(fips, fips)
        visibility = [False] * (len(common_fips) * n_traces_per_county)
        for j in range(n_traces_per_county):
            visibility[i * n_traces_per_county + j] = True
        buttons.append({
            "method": "update",
            "label": cname,
            "args": [
                {"visible": visibility},
                {"title": f"County Projection Comparison: {cname} County"},
            ],
        })

    fig.update_layout(
        template=template_name,
        title=f"County Projection Comparison: {name_map.get(first_fips, '')} County",
        xaxis_title="Year",
        yaxis_title="Total Population",
        yaxis_tickformat=",",
        height=500,
        updatemenus=[{
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
        }],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Scatter plot: SDC 2050 vs Our Baseline 2050 ---
    scatter_x = []  # SDC
    scatter_y = []  # Our
    scatter_names = []

    for fips in common_fips:
        sdc_s = sdc_counties[fips]
        our_s = our_counties.get(fips)
        if our_s is None:
            continue
        sdc_val = sdc_s.get(2050)
        our_val = our_s.get(2050)
        if sdc_val is not None and our_val is not None:
            scatter_x.append(float(sdc_val))
            scatter_y.append(float(our_val))
            scatter_names.append(name_map.get(fips, fips))

    fig_scatter = go.Figure()

    # 45-degree reference line
    max_val = max(max(scatter_x, default=0), max(scatter_y, default=0)) * 1.05
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line={"color": MID_GRAY, "width": 1, "dash": "dot"},
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_scatter.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode="markers",
        marker={"color": BLUE, "size": 8, "line": {"width": 1, "color": NAVY}},
        text=scatter_names,
        hovertemplate="<b>%{text}</b><br>SDC 2050: %{x:,.0f}<br>Our 2050: %{y:,.0f}<extra></extra>",
        name="Counties",
    ))

    fig_scatter.update_layout(
        template=template_name,
        title="2050 Projection Comparison: Our Baseline vs SDC (County-Level)",
        xaxis_title="SDC 2024 Projection (2050)",
        yaxis_title="Our Baseline Projection (2050)",
        xaxis_tickformat=",",
        yaxis_tickformat=",",
        height=500,
        showlegend=False,
    )

    # Add annotation: above line = we project higher, below = SDC projects higher
    fig_scatter.add_annotation(
        x=max_val * 0.3,
        y=max_val * 0.75,
        text="We project higher",
        showarrow=False,
        font={"size": 12, "color": BLUE},
    )
    fig_scatter.add_annotation(
        x=max_val * 0.7,
        y=max_val * 0.3,
        text="SDC projects higher",
        showarrow=False,
        font={"size": 12, "color": SDC_COLOR},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig_scatter, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Heatmap-style bar chart: difference at 2050 for all counties ---
    diff_data = []
    for fips in common_fips:
        sdc_val = float(sdc_counties[fips].get(2050, 0))
        our_val = float(our_counties.get(fips, pd.Series()).get(2050, 0))
        if sdc_val > 0:
            diff_pct = (our_val - sdc_val) / sdc_val * 100
        else:
            diff_pct = 0
        diff_data.append({
            "fips": fips,
            "name": name_map.get(fips, fips),
            "sdc": sdc_val,
            "ours": our_val,
            "diff": our_val - sdc_val,
            "diff_pct": diff_pct,
        })

    diff_df = pd.DataFrame(diff_data).sort_values("diff_pct", ascending=True)

    fig_bar = go.Figure()
    bar_colors = ["#00B050" if d >= 0 else "#FF0000" for d in diff_df["diff_pct"]]

    fig_bar.add_trace(go.Bar(
        y=diff_df["name"],
        x=diff_df["diff_pct"],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Diff: %{x:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig_bar.update_layout(
        template=template_name,
        title="Projection Difference at 2050: (Our Baseline - SDC) / SDC",
        xaxis_title="Difference (%)",
        yaxis_title="",
        height=max(500, len(common_fips) * 20),
        margin={"l": 120},
        showlegend=False,
    )

    # Add vertical zero line
    fig_bar.add_vline(x=0, line_width=1, line_color=DARK_GRAY)

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig_bar, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Top 10 higher / lower tables ---
    sorted_by_diff = sorted(diff_data, key=lambda d: d["diff_pct"], reverse=True)

    higher = sorted_by_diff[:10]
    lower = sorted_by_diff[-10:][::-1]  # Reverse to show most negative first

    def _county_table(items: list[dict], title: str) -> str:
        rows = ""
        for item in items:
            css = "positive" if item["diff_pct"] >= 0 else "negative"
            rows += f"""
                <tr>
                    <td>{item['name']}</td>
                    <td class="num">{_fmt_pop(item['sdc'])}</td>
                    <td class="num">{_fmt_pop(item['ours'])}</td>
                    <td class="num {css}">{_fmt_change(item['diff'])}</td>
                    <td class="num {css}">{_fmt_pct(item['diff_pct'])}</td>
                </tr>
            """
        return f"""
            <h3>{title}</h3>
            <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>County</th>
                        <th class="num">SDC 2050</th>
                        <th class="num">Our 2050</th>
                        <th class="num">Difference</th>
                        <th class="num">Diff %</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            </div>
        """

    parts.append('<div class="two-col">')
    parts.append(_county_table(higher, "Top 10: We Project Higher than SDC"))
    parts.append(_county_table(lower, "Top 10: SDC Projects Higher than Us"))
    parts.append("</div>")

    return "\n".join(parts)


def _build_pyramid_comparison(
    sdc_data: dict, our_data: dict, template_name: str,
) -> str:
    """Build Tab 3: Population Pyramids comparison.

    SDC has 5-year age groups with separate male/female files.
    Our data has broader groups (0-4, 5-17, 18-24, 25-44, 45-64, 65-74, 75-84, 85+).
    We show them side-by-side using SDC's fine-grained pyramids and our broader pyramids.
    """
    parts = ['<h2>Population Pyramid Comparison</h2>']

    sdc_male = sdc_data.get("age_sex_male")
    sdc_female = sdc_data.get("age_sex_female")
    our_age = our_data.get("baseline", {}).get("state_age_dist")

    if sdc_male is None or sdc_female is None:
        parts.append('<p class="placeholder">SDC age/sex data not available.</p>')
        return "\n".join(parts)

    if our_age is None or our_age.empty:
        parts.append('<p class="placeholder">Our age distribution data not available.</p>')
        return "\n".join(parts)

    # Pyramid comparison years (must be in both datasets)
    pyramid_years = [2025, 2035, 2050]

    # --- SDC Pyramids with slider ---
    fig_sdc = go.Figure()

    initial_yr = pyramid_years[0]
    yr_str = str(initial_yr)

    # Helper to build SDC pyramid data for a given year
    def _sdc_pyramid_data(year: int) -> tuple[list, list, list]:
        yr_s = str(year)
        male_vals = []
        female_vals = []
        age_labels = []
        for _, row in sdc_male.iterrows():
            ag = str(row["age_group"])
            male_vals.append(-float(row[yr_s]))  # Negative for left side
            age_labels.append(ag)
        for _, row in sdc_female.iterrows():
            female_vals.append(float(row[yr_s]))
        return age_labels, male_vals, female_vals

    age_labels, male_vals, female_vals = _sdc_pyramid_data(initial_yr)

    fig_sdc.add_trace(go.Bar(
        y=age_labels, x=male_vals, name="Male",
        orientation="h", marker_color="#0563C1",
        hovertemplate="<b>Male</b><br>Age: %{y}<br>Pop: %{customdata:,.0f}<extra></extra>",
        customdata=[-v for v in male_vals],
    ))
    fig_sdc.add_trace(go.Bar(
        y=age_labels, x=female_vals, name="Female",
        orientation="h", marker_color="#C00000",
        hovertemplate="<b>Female</b><br>Age: %{y}<br>Pop: %{customdata:,.0f}<extra></extra>",
        customdata=female_vals,
    ))

    # Build frames for slider
    frames = []
    steps = []
    all_vals = []
    for yr in pyramid_years:
        _, m, f = _sdc_pyramid_data(yr)
        all_vals.extend([-v for v in m])
        all_vals.extend(f)

    x_extent = max(abs(v) for v in all_vals) * 1.1 if all_vals else 50000

    for yr in pyramid_years:
        ag, m, f = _sdc_pyramid_data(yr)
        frames.append(go.Frame(
            data=[
                go.Bar(y=ag, x=m, customdata=[-v for v in m]),
                go.Bar(y=ag, x=f, customdata=f),
            ],
            name=str(yr),
            layout=go.Layout(title=f"SDC 2024 Population Pyramid ({yr})"),
        ))
        steps.append({
            "method": "animate",
            "args": [[str(yr)], {"mode": "immediate", "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 200}}],
            "label": str(yr),
        })

    fig_sdc.frames = frames

    fig_sdc.update_layout(
        template=template_name,
        title=f"SDC 2024 Population Pyramid ({initial_yr})",
        xaxis={"title": "Population", "range": [-x_extent, x_extent], "tickformat": ","},
        yaxis={"title": "Age Group", "categoryorder": "array", "categoryarray": SDC_AGE_GROUPS},
        barmode="overlay",
        bargap=0.05,
        height=520,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Year: ", "font": {"size": 14}},
            "pad": {"t": 40},
            "steps": steps,
        }],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    parts.append("<h3>SDC 2024 Projections</h3>")
    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig_sdc, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Our Baseline Pyramid (broader groups) ---
    fig_ours = go.Figure()

    our_state = our_age[our_age["fips"] == 38].copy()
    pop_cols = [c for c in our_state.columns if c.startswith("pop_")]
    age_group_names = [c.replace("pop_", "") for c in pop_cols]

    def _our_pyramid_data(year: int) -> tuple[list, list]:
        row = our_state[our_state["year"] == year]
        if row.empty:
            return [], []
        vals = [float(row[c].iloc[0]) for c in pop_cols]
        return age_group_names, vals

    initial_ag, initial_vals = _our_pyramid_data(initial_yr)
    if initial_ag:
        fig_ours.add_trace(go.Bar(
            y=initial_ag,
            x=initial_vals,
            name="Our Baseline",
            orientation="h",
            marker_color=BLUE,
            hovertemplate="<b>Our Baseline</b><br>Age: %{y}<br>Pop: %{x:,.0f}<extra></extra>",
        ))

        our_frames = []
        our_steps = []
        for yr in pyramid_years:
            ag, vals = _our_pyramid_data(yr)
            our_frames.append(go.Frame(
                data=[go.Bar(y=ag, x=vals)],
                name=str(yr),
                layout=go.Layout(title=f"Our Baseline Age Distribution ({yr})"),
            ))
            our_steps.append({
                "method": "animate",
                "args": [[str(yr)], {"mode": "immediate", "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 200}}],
                "label": str(yr),
            })

        fig_ours.frames = our_frames

        max_our = max(
            max(
                (float(our_state[our_state["year"] == yr][c].iloc[0])
                 for c in pop_cols if not our_state[our_state["year"] == yr].empty),
                default=0,
            )
            for yr in pyramid_years
        ) * 1.1

        fig_ours.update_layout(
            template=template_name,
            title=f"Our Baseline Age Distribution ({initial_yr})",
            xaxis={"title": "Population", "tickformat": ",", "range": [0, max_our]},
            yaxis={"title": "Age Group", "categoryorder": "array", "categoryarray": age_group_names},
            height=420,
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Year: ", "font": {"size": 14}},
                "pad": {"t": 40},
                "steps": our_steps,
            }],
            showlegend=False,
        )

        parts.append("<h3>Our Baseline Projections</h3>")
        parts.append('<p class="note">Note: Our projections use broader age groups (0-4, 5-17, 18-24, 25-44, 45-64, 65-74, 75-84, 85+) while SDC uses standard 5-year groups.</p>')
        parts.append('<div class="chart-container">')
        parts.append(pio.to_html(fig_ours, include_plotlyjs=False, full_html=False))
        parts.append("</div>")

    # --- Overlay comparison at 2050: aggregate SDC to our broader groups ---
    parts.append("<h3>Age Structure Comparison at 2050 (Aggregated to Common Groups)</h3>")

    # Map SDC 5-year groups to our broader groups
    sdc_to_our_map = {
        "0-4": "0-4",
        "5-9": "5-17", "10-14": "5-17", "15-17_part": "5-17",
        "15-19": "mixed_15_19",  # Split: 15-17 goes to 5-17, 18-19 goes to 18-24
        "20-24": "18-24",
        "25-29": "25-44", "30-34": "25-44", "35-39": "25-44", "40-44": "25-44",
        "45-49": "45-64", "50-54": "45-64", "55-59": "45-64", "60-64": "45-64",
        "65-69": "65-74", "70-74": "65-74",
        "75-79": "75-84", "80-84": "75-84",
        "85+": "85+",
    }

    # Approximate aggregation for SDC (split 15-19 proportionally: 3/5 to 5-17, 2/5 to 18-24)
    sdc_total = sdc_data.get("age_sex_total")
    if sdc_total is not None and "2050" in sdc_total.columns:
        sdc_agg = {g: 0.0 for g in OUR_AGE_GROUPS}
        for _, row in sdc_total.iterrows():
            ag = str(row["age_group"])
            val = float(row["2050"])
            if ag == "0-4":
                sdc_agg["0-4"] += val
            elif ag in ("5-9", "10-14"):
                sdc_agg["5-17"] += val
            elif ag == "15-19":
                # Approximate: 3 years in 5-17 (ages 15,16,17), 2 years in 18-24 (18,19)
                sdc_agg["5-17"] += val * 3 / 5
                sdc_agg["18-24"] += val * 2 / 5
            elif ag == "20-24":
                sdc_agg["18-24"] += val
            elif ag in ("25-29", "30-34", "35-39", "40-44"):
                sdc_agg["25-44"] += val
            elif ag in ("45-49", "50-54", "55-59", "60-64"):
                sdc_agg["45-64"] += val
            elif ag in ("65-69", "70-74"):
                sdc_agg["65-74"] += val
            elif ag in ("75-79", "80-84"):
                sdc_agg["75-84"] += val
            elif ag == "85+":
                sdc_agg["85+"] += val

        # Our data at 2050
        our_2050_row = our_state[our_state["year"] == 2050]
        if not our_2050_row.empty:
            our_agg = {ag: float(our_2050_row[f"pop_{ag}"].iloc[0]) for ag in OUR_AGE_GROUPS}

            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Bar(
                x=OUR_AGE_GROUPS,
                y=[sdc_agg[g] for g in OUR_AGE_GROUPS],
                name="SDC 2024",
                marker_color=SDC_COLOR,
                opacity=0.7,
                hovertemplate="<b>SDC 2024</b><br>Age: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
            ))
            fig_overlay.add_trace(go.Bar(
                x=OUR_AGE_GROUPS,
                y=[our_agg[g] for g in OUR_AGE_GROUPS],
                name="Our Baseline",
                marker_color=BLUE,
                opacity=0.7,
                hovertemplate="<b>Our Baseline</b><br>Age: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
            ))

            fig_overlay.update_layout(
                template=template_name,
                title="Age Group Comparison at 2050: SDC vs Our Baseline",
                xaxis_title="Age Group",
                yaxis_title="Population",
                yaxis_tickformat=",",
                barmode="group",
                height=420,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            )

            parts.append('<div class="chart-container">')
            parts.append(pio.to_html(fig_overlay, include_plotlyjs=False, full_html=False))
            parts.append("</div>")

            # Difference table for age groups at 2050
            parts.append("<h3>Age Group Difference at 2050</h3>")
            ag_rows = ""
            for ag in OUR_AGE_GROUPS:
                sdc_v = sdc_agg[ag]
                our_v = our_agg[ag]
                diff = our_v - sdc_v
                diff_pct = (diff / sdc_v * 100) if sdc_v > 0 else 0
                css = "positive" if diff >= 0 else "negative"
                ag_rows += f"""
                    <tr>
                        <td>{ag}</td>
                        <td class="num">{_fmt_pop(sdc_v)}</td>
                        <td class="num">{_fmt_pop(our_v)}</td>
                        <td class="num {css}">{_fmt_change(diff)}</td>
                        <td class="num {css}">{_fmt_pct(diff_pct)}</td>
                    </tr>
                """
            parts.append(f"""
                <div class="table-container">
                <table class="data-table">
                    <thead><tr>
                        <th>Age Group</th>
                        <th class="num">SDC 2050</th>
                        <th class="num">Our 2050</th>
                        <th class="num">Difference</th>
                        <th class="num">Diff %</th>
                    </tr></thead>
                    <tbody>{ag_rows}</tbody>
                </table>
                </div>
            """)

    return "\n".join(parts)


def _build_growth_analysis(
    sdc_data: dict,
    our_data: dict,
    fips_map: dict[str, str],
    template_name: str,
) -> str:
    """Build Tab 4: Growth Rate Analysis (CAGR comparison)."""
    parts = ['<h2>Growth Rate Analysis</h2>']

    name_map = {v: k for k, v in fips_map.items()}

    sdc_state = _sdc_state_series(sdc_data)
    our_baseline = _our_state_series(our_data, "baseline")

    if sdc_state is None or our_baseline is None:
        parts.append('<p class="placeholder">State data not available.</p>')
        return "\n".join(parts)

    # --- State-level CAGR comparison ---
    periods = [
        ("2025-2030", 2025, 2030, 5),
        ("2030-2040", 2030, 2040, 10),
        ("2040-2050", 2040, 2050, 10),
    ]

    sdc_cagrs = []
    our_cagrs_bl = []
    our_cagrs_hg = []
    our_cagrs_rg = []
    period_labels = []

    for label, start_yr, end_yr, n_yrs in periods:
        period_labels.append(label)

        sdc_start = float(sdc_state.get(start_yr, 0))
        sdc_end = float(sdc_state.get(end_yr, 0))
        sdc_cagrs.append(_cagr(sdc_start, sdc_end, n_yrs) * 100)

        for scenario, cagr_list in [
            ("baseline", our_cagrs_bl),
            ("high_growth", our_cagrs_hg),
            ("restricted_growth", our_cagrs_rg),
        ]:
            series = _our_state_series(our_data, scenario)
            if series is not None:
                s = float(series.get(start_yr, 0))
                e = float(series.get(end_yr, 0))
                cagr_list.append(_cagr(s, e, n_yrs) * 100)
            else:
                cagr_list.append(0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=period_labels, y=sdc_cagrs, name="SDC 2024",
        marker_color=SDC_COLOR,
        text=[f"{v:.2f}%" for v in sdc_cagrs],
        textposition="outside",
        hovertemplate="<b>SDC 2024</b><br>Period: %{x}<br>CAGR: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=period_labels, y=our_cagrs_bl, name="Our Baseline",
        marker_color=SCENARIO_COLORS["baseline"],
        text=[f"{v:.2f}%" for v in our_cagrs_bl],
        textposition="outside",
        hovertemplate="<b>Our Baseline</b><br>Period: %{x}<br>CAGR: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=period_labels, y=our_cagrs_hg, name="Our High Growth",
        marker_color=SCENARIO_COLORS["high_growth"],
        text=[f"{v:.2f}%" for v in our_cagrs_hg],
        textposition="outside",
        hovertemplate="<b>Our High Growth</b><br>Period: %{x}<br>CAGR: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=period_labels, y=our_cagrs_rg, name="Our Restricted",
        marker_color=SCENARIO_COLORS["restricted_growth"],
        text=[f"{v:.2f}%" for v in our_cagrs_rg],
        textposition="outside",
        hovertemplate="<b>Our Restricted</b><br>Period: %{x}<br>CAGR: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template=template_name,
        title="Compound Annual Growth Rate by Period: State Level",
        xaxis_title="Period",
        yaxis_title="CAGR (%)",
        barmode="group",
        height=440,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- State CAGR table ---
    parts.append("<h3>State-Level CAGR Summary</h3>")
    cagr_header = "<tr><th>Source</th>"
    for pl in period_labels:
        cagr_header += f'<th class="num">{pl}</th>'
    cagr_header += "</tr>"

    def _cagr_row(name: str, vals: list[float], color: str) -> str:
        row = f'<tr><td><span class="color-dot" style="background:{color};"></span>{name}</td>'
        for v in vals:
            row += f'<td class="num">{v:.2f}%</td>'
        row += "</tr>"
        return row

    parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead>{cagr_header}</thead>
            <tbody>
                {_cagr_row("SDC 2024", sdc_cagrs, SDC_COLOR)}
                {_cagr_row("Our Baseline", our_cagrs_bl, SCENARIO_COLORS["baseline"])}
                {_cagr_row("Our High Growth", our_cagrs_hg, SCENARIO_COLORS["high_growth"])}
                {_cagr_row("Our Restricted", our_cagrs_rg, SCENARIO_COLORS["restricted_growth"])}
            </tbody>
        </table>
        </div>
    """)

    # --- Top 10 counties CAGR comparison (2025-2050) ---
    parts.append("<h3>Top 10 Counties by Population: CAGR Comparison (2025-2050)</h3>")

    sdc_counties = _sdc_county_dict(sdc_data, fips_map)
    our_counties = _our_county_dict(our_data, "baseline")

    common_fips = sorted(set(sdc_counties.keys()) & set(our_counties.keys()))

    # Find top 10 by baseline 2025 population
    county_pops = []
    for fips in common_fips:
        our_s = our_counties.get(fips)
        if our_s is not None:
            p2025 = float(our_s.get(2025, 0))
            county_pops.append((fips, p2025))

    county_pops.sort(key=lambda x: x[1], reverse=True)
    top10_fips = [f for f, _ in county_pops[:10]]

    top10_names = []
    top10_sdc_cagrs = []
    top10_our_cagrs = []

    for fips in top10_fips:
        cname = name_map.get(fips, fips)
        top10_names.append(cname)

        sdc_s = sdc_counties.get(fips)
        if sdc_s is not None:
            s25 = float(sdc_s.get(2025, 0))
            s50 = float(sdc_s.get(2050, 0))
            top10_sdc_cagrs.append(_cagr(s25, s50, 25) * 100)
        else:
            top10_sdc_cagrs.append(0)

        our_s = our_counties.get(fips)
        if our_s is not None:
            o25 = float(our_s.get(2025, 0))
            o50 = float(our_s.get(2050, 0))
            top10_our_cagrs.append(_cagr(o25, o50, 25) * 100)
        else:
            top10_our_cagrs.append(0)

    fig_county = go.Figure()
    fig_county.add_trace(go.Bar(
        x=top10_names, y=top10_sdc_cagrs, name="SDC 2024",
        marker_color=SDC_COLOR,
        text=[f"{v:.2f}%" for v in top10_sdc_cagrs],
        textposition="outside",
    ))
    fig_county.add_trace(go.Bar(
        x=top10_names, y=top10_our_cagrs, name="Our Baseline",
        marker_color=SCENARIO_COLORS["baseline"],
        text=[f"{v:.2f}%" for v in top10_our_cagrs],
        textposition="outside",
    ))

    fig_county.update_layout(
        template=template_name,
        title="Top 10 Counties: 25-Year CAGR Comparison (2025-2050)",
        xaxis_title="County",
        yaxis_title="CAGR (%)",
        barmode="group",
        height=440,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig_county, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- County CAGR by period table for top 10 ---
    parts.append("<h3>Top 10 Counties: Period CAGR Breakdown</h3>")

    county_table_rows = ""
    for fips in top10_fips:
        cname = name_map.get(fips, fips)
        sdc_s = sdc_counties.get(fips, pd.Series(dtype=float))
        our_s = our_counties.get(fips, pd.Series(dtype=float))

        row_cells = f"<td><b>{cname}</b></td>"
        for _, start_yr, end_yr, n_yrs in periods:
            sdc_c = _cagr(
                float(sdc_s.get(start_yr, 0)),
                float(sdc_s.get(end_yr, 0)),
                n_yrs,
            ) * 100
            our_c = _cagr(
                float(our_s.get(start_yr, 0)),
                float(our_s.get(end_yr, 0)),
                n_yrs,
            ) * 100
            row_cells += f"""
                <td class="num">
                    <span style="color:{SDC_COLOR};">{sdc_c:.2f}%</span> /
                    <span style="color:{BLUE};">{our_c:.2f}%</span>
                </td>
            """
        county_table_rows += f"<tr>{row_cells}</tr>"

    period_headers = "".join(f'<th class="num">{pl}<br><small style="color:{SDC_COLOR};">SDC</small> / <small style="color:{BLUE};">Ours</small></th>' for pl, _, _, _ in periods)

    parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead><tr><th>County</th>{period_headers}</tr></thead>
            <tbody>{county_table_rows}</tbody>
        </table>
        </div>
    """)

    return "\n".join(parts)


# ===================================================================
# HTML Assembly
# ===================================================================

def _build_html(
    tab1: str, tab2: str, tab3: str, tab4: str,
) -> str:
    """Assemble the complete HTML report with tabbed navigation."""
    subtitle = (
        f"SDC 2024 vs Cohort-Component Projections | "
        f"Overlap Period 2025-2050 | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ND SDC Comparison Report - {TODAY.isoformat()}</title>
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
            padding: 0 40px;
            display: flex;
            gap: 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .tab-btn {{
            padding: 14px 24px;
            background: none;
            border: none;
            font-family: {FONT_FAMILY};
            font-size: 14px;
            font-weight: 500;
            color: {DARK_GRAY};
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
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

        .positive {{ color: #00B050; font-weight: 600; }}
        .negative {{ color: #FF0000; font-weight: 600; }}

        .color-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        .note {{
            font-size: 13px;
            color: #888;
            font-style: italic;
            margin-bottom: 12px;
        }}

        .placeholder {{
            font-style: italic;
            color: #999;
            padding: 20px;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #999;
            border-top: 1px solid {MID_GRAY};
            margin-top: 40px;
        }}

        .legend-strip {{
            display: flex;
            gap: 20px;
            padding: 12px 20px;
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

        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .header {{ padding: 20px; }}
            .tab-nav {{ padding: 0 10px; }}
            .tab-btn {{ padding: 10px 14px; font-size: 13px; }}
            .tab-content {{ padding: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>North Dakota Population Projections: SDC 2024 Comparison</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="legend-strip">
        <span style="font-weight:600;">Legend:</span>
        <div class="legend-item">
            <div class="legend-swatch dashed" style="--color:{SDC_COLOR}; background: {SDC_COLOR};"></div>
            <span>SDC 2024 (dashed)</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{SCENARIO_COLORS['baseline']};"></div>
            <span>Our Baseline</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{SCENARIO_COLORS['high_growth']};"></div>
            <span>Our High Growth</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{SCENARIO_COLORS['restricted_growth']};"></div>
            <span>Our Restricted</span>
        </div>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('state')">State Comparison</button>
        <button class="tab-btn" onclick="switchTab('county')">County Comparison</button>
        <button class="tab-btn" onclick="switchTab('pyramids')">Population Pyramids</button>
        <button class="tab-btn" onclick="switchTab('growth')">Growth Rate Analysis</button>
    </div>

    <div id="tab-state" class="tab-content active">
        {tab1}
    </div>

    <div id="tab-county" class="tab-content">
        {tab2}
    </div>

    <div id="tab-pyramids" class="tab-content">
        {tab3}
    </div>

    <div id="tab-growth" class="tab-content">
        {tab4}
    </div>

    <div class="footer">
        North Dakota Cohort-Component Population Projections |
        SDC Comparison Report |
        Generated {TODAY.strftime('%B %d, %Y')} |
        Data: Census PEP 2025 Vintage, SDC 2024 Projections
    </div>

    <script>
        function switchTab(tabId) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

            // Show selected tab
            document.getElementById('tab-' + tabId).classList.add('active');

            // Activate button
            const buttons = document.querySelectorAll('.tab-btn');
            const tabMap = {{'state': 0, 'county': 1, 'pyramids': 2, 'growth': 3}};
            buttons[tabMap[tabId]].classList.add('active');

            // Trigger Plotly resize for proper rendering of charts in hidden tabs
            setTimeout(function() {{
                const plots = document.getElementById('tab-' + tabId).querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 50);
        }}
    </script>
</body>
</html>"""


# ===================================================================
# Main entry point
# ===================================================================

def build_report(output_dir: Path) -> Path:
    """Build the complete SDC comparison HTML report.

    Parameters
    ----------
    output_dir : Path
        Directory to write the output HTML file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    logger.info("=" * 60)
    logger.info("ND SDC Comparison Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    # Register Plotly template
    template_name = _get_plotly_template()

    # Load data
    logger.info("Loading FIPS mapping...")
    fips_map = load_fips_mapping()
    logger.info("  Mapped %d counties", len(fips_map))

    logger.info("Loading SDC 2024 data...")
    sdc_data = load_sdc_data()

    logger.info("Loading our projection data...")
    our_data = load_our_data()

    # Build sections
    logger.info("Building Tab 1: State Comparison...")
    tab1 = _build_state_comparison(sdc_data, our_data, template_name)

    logger.info("Building Tab 2: County Comparison...")
    tab2 = _build_county_comparison(sdc_data, our_data, fips_map, template_name)

    logger.info("Building Tab 3: Population Pyramids...")
    tab3 = _build_pyramid_comparison(sdc_data, our_data, template_name)

    logger.info("Building Tab 4: Growth Rate Analysis...")
    tab4 = _build_growth_analysis(sdc_data, our_data, fips_map, template_name)

    # Assemble HTML
    logger.info("Assembling HTML report...")
    html = _build_html(tab1, tab2, tab3, tab4)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nd_sdc_comparison_{DATE_STAMP}.html"
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Report generated successfully!")
    logger.info("  Output: %s", output_path)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  Sections: 4 tabs")
    logger.info("=" * 60)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build interactive HTML report comparing SDC 2024 and our projections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports",
        help="Output directory (default: data/exports/)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the SDC comparison report builder."""
    args = parse_args()

    try:
        output_path = build_report(output_dir=args.output_dir)
        logger.info("Done. Report at: %s", output_path)
        return 0
    except Exception:
        logger.exception("Failed to build SDC comparison report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
