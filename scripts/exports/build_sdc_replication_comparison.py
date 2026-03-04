#!/usr/bin/env python3
"""
Build interactive HTML report comparing three sets of ND population projections:
SDC 2024 Original, SDC Method + New Data, and Our Model (baseline/high/restricted).

Creates a single self-contained HTML file with Plotly charts and five tabbed sections:
State-Level Comparison, County Comparison, Data Impact Analysis,
Population Pyramids, and Methods Summary.

Output: data/exports/nd_sdc_replication_comparison_{datestamp}.html

Usage:
    python scripts/exports/build_sdc_replication_comparison.py
    python scripts/exports/build_sdc_replication_comparison.py --output-dir data/exports/

Data Sources:
    SDC 2024 Original:
        - data/raw/nd_sdc_2024_projections/state_projections.csv
          columns: year, total_population
        - data/raw/nd_sdc_2024_projections/county_projections.csv
          columns: county_name, 2020, 2025, 2030, 2035, 2040, 2045, 2050
        - data/raw/nd_sdc_2024_projections/state_age_sex_male.csv
          columns: age_group, 2020, 2025, ..., 2050
        - data/raw/nd_sdc_2024_projections/state_age_sex_female.csv
          columns: age_group, 2020, 2025, ..., 2050

    SDC Method + New Data (companion script output):
        - data/exports/sdc_method_new_data/state_population_by_year.csv
          columns: year, total_population
        - data/exports/sdc_method_new_data/county_population_by_year.csv
          columns: county_fips, county_name, 2025, 2030, ..., 2055
        - data/exports/sdc_method_new_data/state_age_sex_by_year.csv
          columns: year, age_group, sex, population

    Our Model (3 scenarios):
        - data/exports/{scenario}/summaries/state_total_population_by_year.csv
          columns: fips, 2025, 2026, ..., 2055
        - data/exports/{scenario}/summaries/county_total_population_by_year.csv
          columns: fips, 2025, 2026, ..., 2055
        - data/exports/{scenario}/summaries/state_age_distribution_by_year.csv
          columns: fips, year, total_population, pop_0-4, pct_0-4, ...

    County FIPS mapping:
        - data/raw/population/nd_county_population.csv
          columns: county_fips, county_name, ...

Methodology:
    This report decomposes the difference between SDC 2024 and our projections into
    two components: (1) data-driven differences (SDC Original vs SDC+New Data), and
    (2) method-driven differences (SDC+New Data vs Our Model).  SDC projections end
    at 2050; our projections extend to 2055.

Key ADRs:
    ADR-037: CBO-grounded scenario methodology
    ADR-054: State-county aggregation
    ADR-055: Group quarters separation

Processing Steps:
    1. Load SDC 2024 original state, county, and age/sex projections
    2. Load SDC Method + New Data companion output
    3. Load our projection exports (baseline, high_growth, restricted_growth)
    4. Build FIPS-to-county-name crosswalk
    5. Generate five tabbed sections with Plotly charts
    6. Assemble into self-contained HTML with embedded CSS/JS

SOP-002 Metadata:
    Author: Claude Code (automated)
    Created: 2026-03-02
    Input files: See Data Sources above
    Output files: data/exports/nd_sdc_replication_comparison_{datestamp}.html
    Dependencies: pandas, plotly
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Path setup & theme import
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _report_theme import (  # noqa: E402
    BLUE,
    DARK_GRAY,
    FONT_FAMILY,
    LIGHT_GRAY,
    MID_GRAY,
    NAVY,
    SCENARIO_COLORS,
    WHITE,
    format_change,
    format_percent,
    format_population,
    get_plotly_template,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")

SCENARIOS = {
    "baseline": "Baseline",
    "high_growth": "High Growth",
    "restricted_growth": "Restricted Growth",
}

# SDC 2024 Original color (dashed)
SDC_COLOR = "#FF8C00"
# SDC Method + New Data color (solid, darker orange)
SDC_NEW_DATA_COLOR = "#D4760A"

# SDC age groups from their 5-year data files
SDC_AGE_GROUPS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85+",
]

# Our broader age groups from state_age_distribution_by_year.csv
OUR_AGE_GROUPS = ["0-4", "5-17", "18-24", "25-44", "45-64", "65-74", "75-84", "85+"]

# Key comparison years
SDC_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
KEY_YEARS = [2025, 2030, 2040, 2050]

# Featured counties for small-multiples (FIPS codes)
FEATURED_COUNTIES = {
    "38017": "Cass",
    "38015": "Burleigh",
    "38035": "Grand Forks",
    "38101": "Ward",
    "38105": "Williams",
    "38053": "McKenzie",
}


# ===================================================================
# Plotly template
# ===================================================================

def _register_template() -> str:
    """Register and return the comparison report template name."""
    template_name = "sdc_replication_comparison"
    template = get_plotly_template()
    # Override colorway for this report
    template.layout.colorway = [
        SDC_COLOR, SDC_NEW_DATA_COLOR,
        SCENARIO_COLORS["baseline"],
        SCENARIO_COLORS["high_growth"],
        SCENARIO_COLORS["restricted_growth"],
    ]
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


def load_fips_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """Build mappings between county name and FIPS code.

    Returns
    -------
    name_to_fips : dict
        e.g. {'Adams': '38001', 'Barnes': '38003', ...}
    fips_to_name : dict
        e.g. {'38001': 'Adams', '38003': 'Barnes', ...}
    """
    path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    df = _load_csv_safe(path)
    if df is None:
        logger.error("Cannot load FIPS mapping from %s", path)
        return {}, {}

    name_to_fips: dict[str, str] = {}
    fips_to_name: dict[str, str] = {}
    for _, row in df.iterrows():
        fips = str(int(row["county_fips"])).zfill(5)
        name = str(row["county_name"]).strip()
        name_to_fips[name] = fips
        fips_to_name[fips] = name

    return name_to_fips, fips_to_name


def load_sdc_original() -> dict[str, Any]:
    """Load SDC 2024 original projection data."""
    sdc_dir = PROJECT_ROOT / "data" / "raw" / "nd_sdc_2024_projections"
    data: dict[str, Any] = {}

    # State totals: columns year, total_population
    data["state_totals"] = _load_csv_safe(sdc_dir / "state_projections.csv")

    # County totals: columns county_name, 2020, 2025, ..., 2050
    data["county_totals"] = _load_csv_safe(sdc_dir / "county_projections.csv")

    # Age/sex data
    data["age_sex_male"] = _load_csv_safe(sdc_dir / "state_age_sex_male.csv")
    data["age_sex_female"] = _load_csv_safe(sdc_dir / "state_age_sex_female.csv")

    return data


def load_sdc_new_data() -> dict[str, Any]:
    """Load SDC Method + New Data companion output."""
    sdc_dir = PROJECT_ROOT / "data" / "exports" / "sdc_method_new_data"
    data: dict[str, Any] = {}

    # State totals: columns year, total_population
    data["state_totals"] = _load_csv_safe(sdc_dir / "state_population_by_year.csv")

    # County totals: columns county_fips, county_name, 2025, 2030, ..., 2055
    data["county_totals"] = _load_csv_safe(sdc_dir / "county_population_by_year.csv")

    # Age/sex data: columns year, age_group, sex, population
    data["age_sex"] = _load_csv_safe(sdc_dir / "state_age_sex_by_year.csv")

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


# ===================================================================
# Series extraction helpers
# ===================================================================

def _sdc_orig_state_series(sdc_data: dict) -> pd.Series | None:
    """Extract year->population Series from SDC 2024 original state totals."""
    df = sdc_data.get("state_totals")
    if df is None or df.empty:
        return None
    return pd.Series(
        df["total_population"].values,
        index=df["year"].astype(int).values,
    )


def _sdc_orig_county_dict(
    sdc_data: dict, name_to_fips: dict[str, str],
) -> dict[str, pd.Series]:
    """Extract per-county year->population from SDC 2024 original county totals."""
    df = sdc_data.get("county_totals")
    if df is None or df.empty:
        return {}

    year_cols = ["2020", "2025", "2030", "2035", "2040", "2045", "2050"]
    result: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        name = str(row["county_name"]).replace(" County", "").strip()
        fips = name_to_fips.get(name)
        if fips is None:
            logger.warning("No FIPS mapping for SDC county: %s", name)
            continue
        series = pd.Series(dtype=float)
        for yc in year_cols:
            if yc in row.index:
                series[int(yc)] = float(row[yc])
        result[fips] = series

    return result


def _sdc_new_state_series(sdc_new: dict) -> pd.Series | None:
    """Extract year->population Series from SDC Method + New Data state totals."""
    df = sdc_new.get("state_totals")
    if df is None or df.empty:
        return None
    return pd.Series(
        df["total_population"].values,
        index=df["year"].astype(int).values,
    )


def _sdc_new_county_dict(sdc_new: dict) -> dict[str, pd.Series]:
    """Extract per-county year->population from SDC Method + New Data county totals."""
    df = sdc_new.get("county_totals")
    if df is None or df.empty:
        return {}

    # Columns: county_fips, county_name, 2025, 2030, ..., 2055
    year_cols = [c for c in df.columns if c.isdigit()]
    result: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        fips = str(int(row["county_fips"])).zfill(5)
        series = pd.Series(dtype=float)
        for yc in year_cols:
            series[int(yc)] = float(row[yc])
        result[fips] = series

    return result


def _our_state_series(our_data: dict, scenario: str) -> pd.Series | None:
    """Extract year->population Series from our state totals for a scenario."""
    df = our_data.get(scenario, {}).get("state_totals")
    if df is None or df.empty:
        return None

    row = df.iloc[0]
    year_cols = [str(y) for y in range(2025, 2056)]
    series = pd.Series(dtype=float)
    for yc in year_cols:
        if yc in row.index:
            series[int(yc)] = float(row[yc])
    return series


def _our_county_dict(our_data: dict, scenario: str) -> dict[str, pd.Series]:
    """Extract per-county year->population from our county totals for a scenario."""
    df = our_data.get(scenario, {}).get("county_totals")
    if df is None or df.empty:
        return {}

    result: dict[str, pd.Series] = {}
    year_cols = [str(y) for y in range(2025, 2056)]
    for _, row in df.iterrows():
        fips = str(int(row["fips"])).zfill(5)
        series = pd.Series(dtype=float)
        for yc in year_cols:
            if yc in row.index:
                series[int(yc)] = float(row[yc])
        result[fips] = series
    return result


# ===================================================================
# Tab 1: State-Level Comparison
# ===================================================================

def _build_state_comparison(
    sdc_orig: dict,
    sdc_new: dict,
    our_data: dict,
    template_name: str,
) -> str:
    """Build Tab 1: State-Level Comparison."""
    parts: list[str] = ["<h2>State-Level Comparison</h2>"]

    sdc_orig_state = _sdc_orig_state_series(sdc_orig)
    sdc_new_state = _sdc_new_state_series(sdc_new)
    our_baseline = _our_state_series(our_data, "baseline")
    our_high = _our_state_series(our_data, "high_growth")
    our_restricted = _our_state_series(our_data, "restricted_growth")

    if sdc_orig_state is None and our_baseline is None:
        parts.append('<p class="placeholder">State data not available.</p>')
        return "\n".join(parts)

    # --- Main Line Chart ---
    fig = go.Figure()

    # SDC 2024 Original (dashed)
    if sdc_orig_state is not None:
        plot_data = sdc_orig_state[sdc_orig_state.index >= 2025].sort_index()
        fig.add_trace(go.Scatter(
            x=list(plot_data.index),
            y=list(plot_data.values),
            name="SDC 2024 Original",
            mode="lines+markers",
            line={"color": SDC_COLOR, "width": 3, "dash": "dash"},
            marker={"size": 7, "symbol": "diamond"},
            hovertemplate=(
                "<b>SDC 2024 Original</b><br>"
                "Year: %{x}<br>Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    # SDC Method + New Data (solid)
    if sdc_new_state is not None:
        plot_data = sdc_new_state[sdc_new_state.index >= 2025].sort_index()
        fig.add_trace(go.Scatter(
            x=list(plot_data.index),
            y=list(plot_data.values),
            name="SDC Method + New Data",
            mode="lines+markers",
            line={"color": SDC_NEW_DATA_COLOR, "width": 3},
            marker={"size": 7, "symbol": "square"},
            hovertemplate=(
                "<b>SDC Method + New Data</b><br>"
                "Year: %{x}<br>Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    # Our scenarios
    scenario_traces = [
        ("baseline", "Our Baseline", SCENARIO_COLORS["baseline"], "solid"),
        ("high_growth", "Our High Growth", SCENARIO_COLORS["high_growth"], "dash"),
        ("restricted_growth", "Our Restricted", SCENARIO_COLORS["restricted_growth"], "dash"),
    ]
    for scenario, label, color, dash in scenario_traces:
        series = _our_state_series(our_data, scenario)
        if series is None:
            continue
        # Show full range 2025-2055
        plot_s = series[(series.index >= 2025)].sort_index()
        fig.add_trace(go.Scatter(
            x=list(plot_s.index),
            y=list(plot_s.values),
            name=label,
            mode="lines",
            line={"color": color, "width": 2.5, "dash": dash},
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Year: %{x}<br>Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="State Total Population: Three Projection Sets (2025-2055)",
        xaxis_title="Year",
        yaxis_title="Total Population",
        yaxis_tickformat=",",
        height=500,
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
        },
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # --- Summary Table at Key Years ---
    parts.append("<h3>Population at Key Years</h3>")

    # Build rows for each projection
    def _get_val(series: pd.Series | None, yr: int) -> str:
        if series is None:
            return "N/A"
        val = series.get(yr)
        if val is None or pd.isna(val):
            return "N/A"
        return _fmt_pop(val)

    header_cols = "".join(f'<th class="num">{yr}</th>' for yr in KEY_YEARS)
    table_rows: list[str] = []

    # SDC Original
    if sdc_orig_state is not None:
        cells = "".join(
            f'<td class="num">{_get_val(sdc_orig_state, yr)}</td>'
            for yr in KEY_YEARS
        )
        table_rows.append(
            f'<tr><td><span class="color-dot" style="background:{SDC_COLOR};"></span>'
            f"SDC 2024 Original</td>{cells}</tr>"
        )

    # SDC New Data
    if sdc_new_state is not None:
        cells = "".join(
            f'<td class="num">{_get_val(sdc_new_state, yr)}</td>'
            for yr in KEY_YEARS
        )
        table_rows.append(
            f'<tr><td><span class="color-dot" style="background:{SDC_NEW_DATA_COLOR};"></span>'
            f"SDC Method + New Data</td>{cells}</tr>"
        )

    # Our scenarios
    for scenario, label, color, _ in scenario_traces:
        series = _our_state_series(our_data, scenario)
        cells = "".join(
            f'<td class="num">{_get_val(series, yr)}</td>'
            for yr in KEY_YEARS
        )
        table_rows.append(
            f'<tr><td><span class="color-dot" style="background:{color};"></span>'
            f"{label}</td>{cells}</tr>"
        )

    parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Projection</th>
                    {header_cols}
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
        </div>
    """)

    # --- Key Insight Callout ---
    if sdc_orig_state is not None and sdc_new_state is not None:
        orig_2050 = float(sdc_orig_state.get(2050, 0))
        new_2050 = float(sdc_new_state.get(2050, 0))
        data_diff = new_2050 - orig_2050
        data_diff_pct = (data_diff / orig_2050 * 100) if orig_2050 > 0 else 0

        parts.append(f"""
        <div class="callout-box">
            <h3>Key Insight: Data Impact on SDC Methodology</h3>
            <p>Updating the SDC methodology from 2020 Census base year to 2025 PEP data
            changes the 2050 state population projection by
            <strong>{_fmt_change(data_diff)}</strong>
            (<strong>{_fmt_pct(data_diff_pct)}</strong>).</p>
            <p>SDC 2024 Original at 2050: <strong>{_fmt_pop(orig_2050)}</strong> |
            SDC Method + New Data at 2050: <strong>{_fmt_pop(new_2050)}</strong></p>
        </div>
        """)

    return "\n".join(parts)


# ===================================================================
# Tab 2: County Comparison
# ===================================================================

def _build_county_comparison(
    sdc_orig: dict,
    sdc_new: dict,
    our_data: dict,
    name_to_fips: dict[str, str],
    fips_to_name: dict[str, str],
    template_name: str,
) -> str:
    """Build Tab 2: County Comparison with dropdown and small multiples."""
    parts: list[str] = ["<h2>County Comparison</h2>"]

    # Extract county data
    sdc_orig_counties = _sdc_orig_county_dict(sdc_orig, name_to_fips)
    sdc_new_counties = _sdc_new_county_dict(sdc_new)
    our_bl_counties = _our_county_dict(our_data, "baseline")
    our_hg_counties = _our_county_dict(our_data, "high_growth")
    our_rg_counties = _our_county_dict(our_data, "restricted_growth")

    # Get list of all counties sorted by name
    all_fips = sorted(
        set(our_bl_counties.keys()) | set(sdc_orig_counties.keys()),
        key=lambda f: fips_to_name.get(f, f),
    )

    if not all_fips:
        parts.append('<p class="placeholder">County data not available.</p>')
        return "\n".join(parts)

    # --- Build per-county data for JS-driven dropdown chart ---
    county_chart_data: dict[str, dict[str, Any]] = {}
    for fips in all_fips:
        cname = fips_to_name.get(fips, fips)
        entry: dict[str, Any] = {"name": cname}

        # SDC Original
        s = sdc_orig_counties.get(fips)
        if s is not None:
            s_filtered = s[s.index >= 2025].sort_index()
            entry["sdc_orig_years"] = [int(y) for y in s_filtered.index]
            entry["sdc_orig_vals"] = [float(v) for v in s_filtered.values]
        else:
            entry["sdc_orig_years"] = []
            entry["sdc_orig_vals"] = []

        # SDC New Data
        s = sdc_new_counties.get(fips)
        if s is not None:
            s_filtered = s[s.index >= 2025].sort_index()
            entry["sdc_new_years"] = [int(y) for y in s_filtered.index]
            entry["sdc_new_vals"] = [float(v) for v in s_filtered.values]
        else:
            entry["sdc_new_years"] = []
            entry["sdc_new_vals"] = []

        # Our scenarios
        for key, counties in [
            ("our_bl", our_bl_counties),
            ("our_hg", our_hg_counties),
            ("our_rg", our_rg_counties),
        ]:
            s = counties.get(fips)
            if s is not None:
                s_filtered = s[s.index >= 2025].sort_index()
                entry[f"{key}_years"] = [int(y) for y in s_filtered.index]
                entry[f"{key}_vals"] = [float(v) for v in s_filtered.values]
            else:
                entry[f"{key}_years"] = []
                entry[f"{key}_vals"] = []

        county_chart_data[fips] = entry

    # Dropdown selector + chart container
    options_html = "\n".join(
        f'<option value="{fips}">{fips_to_name.get(fips, fips)} ({fips})</option>'
        for fips in all_fips
    )

    parts.append(f"""
    <div class="county-selector">
        <label for="county-select"><strong>Select County:</strong></label>
        <select id="county-select" onchange="updateCountyChart(this.value)">
            {options_html}
        </select>
    </div>
    <div id="county-chart-container" class="chart-container" style="min-height:460px;">
        <div id="county-chart"></div>
    </div>
    """)

    # Embed county data as JSON for JavaScript
    parts.append(f"""
    <script>
        var countyData = {json.dumps(county_chart_data)};

        function updateCountyChart(fips) {{
            var d = countyData[fips];
            if (!d) return;

            var traces = [];

            // SDC 2024 Original
            if (d.sdc_orig_years.length > 0) {{
                traces.push({{
                    x: d.sdc_orig_years, y: d.sdc_orig_vals,
                    name: 'SDC 2024 Original',
                    mode: 'lines+markers',
                    line: {{color: '{SDC_COLOR}', width: 3, dash: 'dash'}},
                    marker: {{size: 7, symbol: 'diamond'}},
                    hovertemplate: '<b>SDC 2024 Original</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>'
                }});
            }}

            // SDC Method + New Data
            if (d.sdc_new_years.length > 0) {{
                traces.push({{
                    x: d.sdc_new_years, y: d.sdc_new_vals,
                    name: 'SDC Method + New Data',
                    mode: 'lines+markers',
                    line: {{color: '{SDC_NEW_DATA_COLOR}', width: 3}},
                    marker: {{size: 7, symbol: 'square'}},
                    hovertemplate: '<b>SDC Method + New Data</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>'
                }});
            }}

            // Our Baseline
            if (d.our_bl_years.length > 0) {{
                traces.push({{
                    x: d.our_bl_years, y: d.our_bl_vals,
                    name: 'Our Baseline',
                    mode: 'lines',
                    line: {{color: '{SCENARIO_COLORS["baseline"]}', width: 2.5}},
                    hovertemplate: '<b>Our Baseline</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>'
                }});
            }}

            // Our High Growth
            if (d.our_hg_years.length > 0) {{
                traces.push({{
                    x: d.our_hg_years, y: d.our_hg_vals,
                    name: 'Our High Growth',
                    mode: 'lines',
                    line: {{color: '{SCENARIO_COLORS["high_growth"]}', width: 2.5, dash: 'dash'}},
                    hovertemplate: '<b>Our High Growth</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>'
                }});
            }}

            // Our Restricted
            if (d.our_rg_years.length > 0) {{
                traces.push({{
                    x: d.our_rg_years, y: d.our_rg_vals,
                    name: 'Our Restricted',
                    mode: 'lines',
                    line: {{color: '{SCENARIO_COLORS["restricted_growth"]}', width: 2.5, dash: 'dash'}},
                    hovertemplate: '<b>Our Restricted</b><br>Year: %{{x}}<br>Pop: %{{y:,.0f}}<extra></extra>'
                }});
            }}

            var layout = {{
                title: d.name + ' County Population Projections',
                xaxis: {{title: 'Year'}},
                yaxis: {{title: 'Population', tickformat: ','}},
                height: 440,
                font: {{family: "{FONT_FAMILY}", size: 12, color: "{DARK_GRAY}"}},
                plot_bgcolor: '{WHITE}',
                paper_bgcolor: '{WHITE}',
                legend: {{orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1}},
                margin: {{l: 70, r: 30, t: 60, b: 50}}
            }};

            Plotly.newPlot('county-chart', traces, layout);
        }}

        // Initialize with first county
        document.addEventListener('DOMContentLoaded', function() {{
            var select = document.getElementById('county-select');
            if (select && select.value) {{
                updateCountyChart(select.value);
            }}
        }});
    </script>
    """)

    # --- Small Multiples Panel ---
    parts.append("<h3>Featured Counties</h3>")
    parts.append('<div class="small-multiples-grid">')

    for fips, cname in FEATURED_COUNTIES.items():
        fig = go.Figure()

        # SDC Original
        s = sdc_orig_counties.get(fips)
        if s is not None:
            s_plot = s[s.index >= 2025].sort_index()
            fig.add_trace(go.Scatter(
                x=list(s_plot.index), y=list(s_plot.values),
                name="SDC Original", mode="lines",
                line={"color": SDC_COLOR, "width": 2, "dash": "dash"},
                showlegend=False,
            ))

        # SDC New Data
        s = sdc_new_counties.get(fips)
        if s is not None:
            s_plot = s[s.index >= 2025].sort_index()
            fig.add_trace(go.Scatter(
                x=list(s_plot.index), y=list(s_plot.values),
                name="SDC + New Data", mode="lines",
                line={"color": SDC_NEW_DATA_COLOR, "width": 2},
                showlegend=False,
            ))

        # Our Baseline
        s = our_bl_counties.get(fips)
        if s is not None:
            s_plot = s[s.index >= 2025].sort_index()
            fig.add_trace(go.Scatter(
                x=list(s_plot.index), y=list(s_plot.values),
                name="Our Baseline", mode="lines",
                line={"color": SCENARIO_COLORS["baseline"], "width": 2},
                showlegend=False,
            ))

        # Our High Growth
        s = our_hg_counties.get(fips)
        if s is not None:
            s_plot = s[s.index >= 2025].sort_index()
            fig.add_trace(go.Scatter(
                x=list(s_plot.index), y=list(s_plot.values),
                name="Our High", mode="lines",
                line={"color": SCENARIO_COLORS["high_growth"], "width": 1.5, "dash": "dash"},
                showlegend=False,
            ))

        # Our Restricted
        s = our_rg_counties.get(fips)
        if s is not None:
            s_plot = s[s.index >= 2025].sort_index()
            fig.add_trace(go.Scatter(
                x=list(s_plot.index), y=list(s_plot.values),
                name="Our Restricted", mode="lines",
                line={"color": SCENARIO_COLORS["restricted_growth"], "width": 1.5, "dash": "dash"},
                showlegend=False,
            ))

        fig.update_layout(
            template=template_name,
            title={"text": f"{cname} ({fips})", "font": {"size": 13}},
            xaxis={"dtick": 10, "tickfont": {"size": 9}},
            yaxis={"tickformat": ",", "tickfont": {"size": 9}},
            height=260,
            margin={"l": 55, "r": 10, "t": 35, "b": 30},
            showlegend=False,
        )

        parts.append('<div class="small-multiple">')
        parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        parts.append("</div>")

    parts.append("</div>")  # close small-multiples-grid

    return "\n".join(parts)


# ===================================================================
# Tab 3: Data Impact Analysis
# ===================================================================

def _build_data_impact(
    sdc_orig: dict,
    sdc_new: dict,
    name_to_fips: dict[str, str],
    fips_to_name: dict[str, str],
    template_name: str,
) -> str:
    """Build Tab 3: Data Impact Analysis (SDC Original vs SDC+New Data)."""
    parts: list[str] = ["<h2>Data Impact Analysis</h2>"]
    parts.append(
        '<p class="note">This tab isolates the impact of updating the data while '
        "keeping the SDC cohort-component methodology constant.</p>"
    )

    sdc_orig_state = _sdc_orig_state_series(sdc_orig)
    sdc_new_state = _sdc_new_state_series(sdc_new)

    if sdc_orig_state is None or sdc_new_state is None:
        parts.append(
            '<p class="placeholder">Both SDC Original and SDC Method + New Data '
            "are required for this tab.</p>"
        )
        return "\n".join(parts)

    # --- State-Level Bar Chart ---
    parts.append("<h3>State-Level Difference: SDC Original vs SDC + New Data</h3>")

    compare_years = [2030, 2040, 2050]
    orig_vals = []
    new_vals = []
    diffs = []
    for yr in compare_years:
        ov = float(sdc_orig_state.get(yr, 0))
        nv = float(sdc_new_state.get(yr, 0))
        orig_vals.append(ov)
        new_vals.append(nv)
        diffs.append(nv - ov)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[str(y) for y in compare_years],
        y=orig_vals,
        name="SDC 2024 Original",
        marker_color=SDC_COLOR,
        text=[_fmt_pop(v) for v in orig_vals],
        textposition="outside",
        hovertemplate="<b>SDC Original</b><br>Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=[str(y) for y in compare_years],
        y=new_vals,
        name="SDC Method + New Data",
        marker_color=SDC_NEW_DATA_COLOR,
        text=[_fmt_pop(v) for v in new_vals],
        textposition="outside",
        hovertemplate=(
            "<b>SDC + New Data</b><br>Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>"
        ),
    ))

    fig_bar.update_layout(
        template=template_name,
        title="State Population: SDC Original vs SDC + New Data",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        barmode="group",
        height=440,
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
        },
    )

    parts.append('<div class="chart-container">')
    parts.append(pio.to_html(fig_bar, include_plotlyjs=False, full_html=False))
    parts.append("</div>")

    # Difference annotations
    parts.append('<div class="diff-summary">')
    for i, yr in enumerate(compare_years):
        pct = (diffs[i] / orig_vals[i] * 100) if orig_vals[i] > 0 else 0
        css_class = "positive" if diffs[i] >= 0 else "negative"
        parts.append(
            f'<div class="diff-card">'
            f"<div class=\"diff-year\">{yr}</div>"
            f'<div class="diff-value {css_class}">{_fmt_change(diffs[i])}</div>'
            f'<div class="diff-pct {css_class}">{_fmt_pct(pct)}</div>'
            f"</div>"
        )
    parts.append("</div>")

    # --- County-Level Difference Table at 2050 ---
    parts.append("<h3>County-Level Differences at 2050</h3>")

    sdc_orig_counties = _sdc_orig_county_dict(sdc_orig, name_to_fips)
    sdc_new_counties = _sdc_new_county_dict(sdc_new)

    common_fips = sorted(
        set(sdc_orig_counties.keys()) & set(sdc_new_counties.keys()),
    )

    if common_fips:
        county_diffs: list[tuple[str, str, float, float, float, float]] = []
        for fips in common_fips:
            cname = fips_to_name.get(fips, fips)
            orig_s = sdc_orig_counties[fips]
            new_s = sdc_new_counties[fips]
            ov = float(orig_s.get(2050, 0))
            nv = float(new_s.get(2050, 0))
            diff = nv - ov
            diff_pct = (diff / ov * 100) if ov > 0 else 0
            county_diffs.append((fips, cname, ov, nv, diff, diff_pct))

        # Sort by absolute difference descending
        county_diffs.sort(key=lambda x: abs(x[4]), reverse=True)

        county_rows = ""
        for fips, cname, ov, nv, diff, diff_pct in county_diffs:
            css = "positive" if diff >= 0 else "negative"
            county_rows += f"""
                <tr>
                    <td>{cname}</td>
                    <td class="num">{_fmt_pop(ov)}</td>
                    <td class="num">{_fmt_pop(nv)}</td>
                    <td class="num {css}">{_fmt_change(diff)}</td>
                    <td class="num {css}">{_fmt_pct(diff_pct)}</td>
                </tr>
            """

        parts.append(f"""
        <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>County</th>
                    <th class="num">SDC Original (2050)</th>
                    <th class="num">SDC + New Data (2050)</th>
                    <th class="num">Difference</th>
                    <th class="num">Diff %</th>
                </tr>
            </thead>
            <tbody>
                {county_rows}
            </tbody>
        </table>
        </div>
        """)

        # --- Scatter Plot ---
        parts.append(
            "<h3>County Scatter: SDC Original vs SDC + New Data at 2050</h3>"
        )

        scatter_x = [cd[2] for cd in county_diffs]
        scatter_y = [cd[3] for cd in county_diffs]
        scatter_names = [cd[1] for cd in county_diffs]

        # 45-degree reference line
        max_val = max(max(scatter_x), max(scatter_y)) * 1.05
        min_val = min(min(scatter_x), min(scatter_y)) * 0.95

        fig_scatter = go.Figure()

        # Reference line
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": MID_GRAY, "width": 1.5, "dash": "dash"},
            name="Equal",
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_scatter.add_trace(go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers+text",
            marker={"color": SDC_NEW_DATA_COLOR, "size": 10, "opacity": 0.8},
            text=scatter_names,
            textposition="top center",
            textfont={"size": 9},
            name="Counties",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "SDC Original: %{x:,.0f}<br>"
                "SDC + New Data: %{y:,.0f}<extra></extra>"
            ),
        ))

        fig_scatter.update_layout(
            template=template_name,
            title="County Population at 2050: SDC Original (x) vs SDC + New Data (y)",
            xaxis_title="SDC 2024 Original Population (2050)",
            yaxis_title="SDC Method + New Data Population (2050)",
            xaxis_tickformat=",",
            yaxis_tickformat=",",
            height=550,
            showlegend=False,
        )

        # Add annotation about the line
        fig_scatter.add_annotation(
            x=max_val * 0.7,
            y=max_val * 0.65,
            text="Above line = higher with new data<br>Below line = lower with new data",
            showarrow=False,
            font={"size": 11, "color": DARK_GRAY},
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=MID_GRAY,
            borderwidth=1,
            borderpad=6,
        )

        parts.append('<div class="chart-container">')
        parts.append(pio.to_html(fig_scatter, include_plotlyjs=False, full_html=False))
        parts.append("</div>")

    else:
        parts.append(
            '<p class="placeholder">No common counties found between both SDC datasets.</p>'
        )

    return "\n".join(parts)


# ===================================================================
# Tab 4: Population Pyramids
# ===================================================================

def _build_pyramids(
    sdc_new: dict,
    our_data: dict,
    template_name: str,
) -> str:
    """Build Tab 4: Population Pyramids (SDC+New Data vs Our Baseline)."""
    parts: list[str] = ["<h2>Population Pyramids</h2>"]

    sdc_age_sex = sdc_new.get("age_sex")
    our_age = our_data.get("baseline", {}).get("state_age_dist")

    if sdc_age_sex is None and our_age is None:
        parts.append(
            '<p class="placeholder">Age/sex data not available for pyramid charts.</p>'
        )
        return "\n".join(parts)

    pyramid_years = [2025, 2030, 2040, 2050]

    # --- SDC Method + New Data Pyramid (18 five-year groups, male/female) ---
    if sdc_age_sex is not None:
        # Build pyramid data from state_age_sex_by_year.csv
        # columns: year, age_group, sex, population
        sdc_pyramid_data: dict[int, dict[str, list[float]]] = {}
        for yr in pyramid_years:
            yr_df = sdc_age_sex[sdc_age_sex["year"] == yr]
            if yr_df.empty:
                continue
            male_vals: list[float] = []
            female_vals: list[float] = []
            for ag in SDC_AGE_GROUPS:
                m_row = yr_df[
                    (yr_df["age_group"] == ag) & (yr_df["sex"].str.lower() == "male")
                ]
                f_row = yr_df[
                    (yr_df["age_group"] == ag) & (yr_df["sex"].str.lower() == "female")
                ]
                male_vals.append(float(m_row["population"].iloc[0]) if not m_row.empty else 0)
                female_vals.append(float(f_row["population"].iloc[0]) if not f_row.empty else 0)
            sdc_pyramid_data[yr] = {"male": male_vals, "female": female_vals}

        if sdc_pyramid_data:
            # Build figure with slider
            available_yrs = sorted(sdc_pyramid_data.keys())
            initial_yr = available_yrs[0]

            fig_sdc = go.Figure()

            # Initial data
            init = sdc_pyramid_data[initial_yr]
            fig_sdc.add_trace(go.Bar(
                y=SDC_AGE_GROUPS,
                x=[-v for v in init["male"]],
                name="Male",
                orientation="h",
                marker_color="#0563C1",
                hovertemplate="<b>Male</b><br>Age: %{y}<br>Pop: %{customdata:,.0f}<extra></extra>",
                customdata=init["male"],
            ))
            fig_sdc.add_trace(go.Bar(
                y=SDC_AGE_GROUPS,
                x=init["female"],
                name="Female",
                orientation="h",
                marker_color="#C00000",
                hovertemplate="<b>Female</b><br>Age: %{y}<br>Pop: %{customdata:,.0f}<extra></extra>",
                customdata=init["female"],
            ))

            # Build frames & slider steps
            frames = []
            steps = []
            all_abs_vals: list[float] = []
            for yr in available_yrs:
                d = sdc_pyramid_data[yr]
                all_abs_vals.extend(d["male"])
                all_abs_vals.extend(d["female"])

            x_extent = max(all_abs_vals) * 1.1 if all_abs_vals else 50000

            for yr in available_yrs:
                d = sdc_pyramid_data[yr]
                frames.append(go.Frame(
                    data=[
                        go.Bar(
                            y=SDC_AGE_GROUPS,
                            x=[-v for v in d["male"]],
                            customdata=d["male"],
                        ),
                        go.Bar(
                            y=SDC_AGE_GROUPS,
                            x=d["female"],
                            customdata=d["female"],
                        ),
                    ],
                    name=str(yr),
                    layout=go.Layout(
                        title=f"SDC Method + New Data: Population Pyramid ({yr})",
                    ),
                ))
                steps.append({
                    "method": "animate",
                    "args": [
                        [str(yr)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 300, "redraw": True},
                            "transition": {"duration": 200},
                        },
                    ],
                    "label": str(yr),
                })

            fig_sdc.frames = frames

            fig_sdc.update_layout(
                template=template_name,
                title=f"SDC Method + New Data: Population Pyramid ({initial_yr})",
                xaxis={
                    "title": "Population",
                    "range": [-x_extent, x_extent],
                    "tickformat": ",",
                },
                yaxis={
                    "title": "Age Group",
                    "categoryorder": "array",
                    "categoryarray": SDC_AGE_GROUPS,
                },
                barmode="overlay",
                bargap=0.05,
                height=520,
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Year: ", "font": {"size": 14}},
                    "pad": {"t": 40},
                    "steps": steps,
                }],
                legend={
                    "orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1,
                },
            )

            parts.append("<h3>SDC Method + New Data</h3>")
            parts.append('<div class="chart-container">')
            parts.append(pio.to_html(fig_sdc, include_plotlyjs=False, full_html=False))
            parts.append("</div>")

    # --- Our Baseline Pyramid (broader age groups) ---
    if our_age is not None:
        our_state = our_age[our_age["fips"] == 38].copy()
        pop_cols = [c for c in our_state.columns if c.startswith("pop_")]
        age_group_names = [c.replace("pop_", "") for c in pop_cols]

        if pop_cols and not our_state.empty:
            available_yrs_ours = sorted(
                yr for yr in pyramid_years
                if not our_state[our_state["year"] == yr].empty
            )

            if available_yrs_ours:
                initial_yr_o = available_yrs_ours[0]

                fig_ours = go.Figure()

                # Initial data
                init_row = our_state[our_state["year"] == initial_yr_o].iloc[0]
                init_vals = [float(init_row[c]) for c in pop_cols]

                fig_ours.add_trace(go.Bar(
                    y=age_group_names,
                    x=init_vals,
                    name="Our Baseline",
                    orientation="h",
                    marker_color=BLUE,
                    hovertemplate=(
                        "<b>Our Baseline</b><br>Age: %{y}<br>"
                        "Pop: %{x:,.0f}<extra></extra>"
                    ),
                ))

                # Frames & slider
                our_frames = []
                our_steps = []
                max_our = 0.0
                for yr in available_yrs_ours:
                    row = our_state[our_state["year"] == yr]
                    if row.empty:
                        continue
                    vals = [float(row[c].iloc[0]) for c in pop_cols]
                    max_our = max(max_our, max(vals))
                    our_frames.append(go.Frame(
                        data=[go.Bar(y=age_group_names, x=vals)],
                        name=str(yr),
                        layout=go.Layout(
                            title=f"Our Baseline: Age Distribution ({yr})",
                        ),
                    ))
                    our_steps.append({
                        "method": "animate",
                        "args": [
                            [str(yr)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 300, "redraw": True},
                                "transition": {"duration": 200},
                            },
                        ],
                        "label": str(yr),
                    })

                fig_ours.frames = our_frames

                fig_ours.update_layout(
                    template=template_name,
                    title=f"Our Baseline: Age Distribution ({initial_yr_o})",
                    xaxis={
                        "title": "Population",
                        "tickformat": ",",
                        "range": [0, max_our * 1.1],
                    },
                    yaxis={
                        "title": "Age Group",
                        "categoryorder": "array",
                        "categoryarray": age_group_names,
                    },
                    height=420,
                    sliders=[{
                        "active": 0,
                        "currentvalue": {"prefix": "Year: ", "font": {"size": 14}},
                        "pad": {"t": 40},
                        "steps": our_steps,
                    }],
                    showlegend=False,
                )

                parts.append("<h3>Our Baseline</h3>")
                parts.append(
                    '<p class="note">Note: Our projections use broader age groups '
                    "(0-4, 5-17, 18-24, 25-44, 45-64, 65-74, 75-84, 85+) while "
                    "the SDC method uses standard 5-year groups. The two pyramids are "
                    "displayed with their native age groupings for accuracy.</p>"
                )
                parts.append('<div class="chart-container">')
                parts.append(
                    pio.to_html(fig_ours, include_plotlyjs=False, full_html=False)
                )
                parts.append("</div>")

    return "\n".join(parts)


# ===================================================================
# Tab 5: Methods Summary
# ===================================================================

def _build_methods_summary() -> str:
    """Build Tab 5: Methods Summary."""
    parts: list[str] = ["<h2>Methods Summary</h2>"]

    # --- Methodology Comparison Table ---
    parts.append("<h3>Methodology Comparison</h3>")
    parts.append("""
    <div class="table-container">
    <table class="data-table methods-table">
        <thead>
            <tr>
                <th>Aspect</th>
                <th>SDC 2024 Original</th>
                <th>SDC Method + New Data</th>
                <th>Our Model</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Base year</strong></td>
                <td>2020 Census</td>
                <td>2025 PEP</td>
                <td>2025 PEP</td>
            </tr>
            <tr>
                <td><strong>Migration periods</strong></td>
                <td>2000-2020 (4 periods)</td>
                <td>2000-2025 (5 periods)</td>
                <td>2000-2025 (5 periods)</td>
            </tr>
            <tr>
                <td><strong>Bakken dampening</strong></td>
                <td>~60%</td>
                <td>60%</td>
                <td>40-50% period-specific</td>
            </tr>
            <tr>
                <td><strong>GQ separation</strong></td>
                <td>No</td>
                <td>No</td>
                <td>Yes (ADR-055)</td>
            </tr>
            <tr>
                <td><strong>Mortality improvement</strong></td>
                <td>No</td>
                <td>No</td>
                <td>Yes (0.5%/yr)</td>
            </tr>
            <tr>
                <td><strong>Race/ethnicity</strong></td>
                <td>No</td>
                <td>No</td>
                <td>6 categories</td>
            </tr>
            <tr>
                <td><strong>Age resolution</strong></td>
                <td>5-year groups</td>
                <td>5-year groups</td>
                <td>Single year</td>
            </tr>
            <tr>
                <td><strong>Projection step</strong></td>
                <td>5-year</td>
                <td>5-year</td>
                <td>Annual</td>
            </tr>
            <tr>
                <td><strong>Convergence</strong></td>
                <td>No</td>
                <td>No</td>
                <td>Yes (5-10-5)</td>
            </tr>
            <tr>
                <td><strong>College-age smoothing</strong></td>
                <td>No</td>
                <td>No</td>
                <td>Yes</td>
            </tr>
            <tr>
                <td><strong>Rate caps</strong></td>
                <td>No</td>
                <td>No</td>
                <td>Yes</td>
            </tr>
            <tr>
                <td><strong>Scenarios</strong></td>
                <td>1 (single projection)</td>
                <td>1 (single projection)</td>
                <td>3 (baseline, high, restricted)</td>
            </tr>
        </tbody>
    </table>
    </div>
    """)

    # --- Narrative Explanation ---
    parts.append("<h3>What This Comparison Reveals</h3>")
    parts.append("""
    <div class="narrative">
        <div class="narrative-section">
            <h4>Decomposing the Difference</h4>
            <p>This report decomposes the total difference between the SDC 2024 projections
            and our projections into two distinct components:</p>
            <ol>
                <li><strong>Data Effect</strong> (SDC Original vs SDC + New Data):
                    What happens when you update the data from 2020 Census to 2025 PEP
                    while keeping the SDC methodology constant? This isolates the impact
                    of five additional years of observed population change, including
                    post-pandemic recovery, Bakken oil region dynamics, and recent
                    migration patterns.</li>
                <li><strong>Method Effect</strong> (SDC + New Data vs Our Model):
                    What do our methodological enhancements add beyond the data update?
                    This captures the impact of GQ separation, mortality improvement,
                    single-year age resolution, annual projection steps, convergence,
                    college-age smoothing, rate caps, and scenario analysis.</li>
            </ol>
        </div>

        <div class="narrative-section">
            <h4>Why This Matters</h4>
            <p>When our projections differ from the published SDC 2024 projections,
            stakeholders naturally ask: <em>"Why are these different?"</em> This report
            provides a rigorous answer by showing exactly how much of the difference comes
            from using newer data versus using improved methods.</p>
            <p>If the <strong>data effect</strong> dominates, it means most of the
            difference is simply because we have five more years of observed population
            data. If the <strong>method effect</strong> dominates, it means our
            methodological enhancements are the primary driver of differences.</p>
        </div>

        <div class="narrative-section">
            <h4>Reading the Results</h4>
            <ul>
                <li><strong>Tab 1 (State)</strong>: Shows all projection lines together.
                    The gap between SDC Original (dashed orange) and SDC + New Data
                    (solid darker orange) is the data effect. The gap between SDC + New
                    Data and our Baseline (blue) is the method effect.</li>
                <li><strong>Tab 2 (County)</strong>: Reveals that data vs. method effects
                    vary significantly by county, especially for fast-growing Bakken
                    counties where additional observed data captures recent growth trends.</li>
                <li><strong>Tab 3 (Data Impact)</strong>: Quantifies and visualizes the
                    data effect in isolation, showing which counties are most affected
                    by the data update.</li>
                <li><strong>Tab 4 (Pyramids)</strong>: Shows age-structure differences
                    between the SDC method approach and our model.</li>
            </ul>
        </div>
    </div>
    """)

    return "\n".join(parts)


# ===================================================================
# HTML Assembly
# ===================================================================

def _assemble_html(
    tab1: str,
    tab2: str,
    tab3: str,
    tab4: str,
    tab5: str,
) -> str:
    """Assemble the five tab sections into a complete HTML document."""
    subtitle = (
        f"SDC 2024 Original | SDC Method + New Data | Our Model (3 Scenarios) | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ND SDC Replication Comparison Report - {TODAY.isoformat()}</title>
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
            background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
            color: white;
            padding: 30px 40px;
            margin-bottom: 0;
        }}

        .header h1 {{
            font-size: 26px;
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
            padding: 14px 22px;
            background: none;
            border: none;
            font-family: {FONT_FAMILY};
            font-size: 13px;
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

        h4 {{
            color: {NAVY};
            font-size: 15px;
            margin: 16px 0 8px 0;
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

        /* Methods table specific styles */
        .methods-table td:first-child {{
            min-width: 180px;
        }}
        .methods-table td {{
            vertical-align: top;
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

        /* County selector */
        .county-selector {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .county-selector select {{
            font-family: {FONT_FAMILY};
            font-size: 14px;
            padding: 8px 12px;
            border: 1px solid {MID_GRAY};
            border-radius: 4px;
            margin-left: 10px;
            min-width: 250px;
        }}

        /* Small multiples */
        .small-multiples-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 16px;
        }}

        .small-multiple {{
            background: white;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        /* Callout box */
        .callout-box {{
            background: #f0f6ff;
            border-left: 4px solid {BLUE};
            padding: 16px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}

        .callout-box h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 15px;
        }}

        .callout-box p {{
            margin: 4px 0;
            font-size: 14px;
        }}

        /* Diff summary cards */
        .diff-summary {{
            display: flex;
            gap: 20px;
            margin: 16px 0 24px 0;
        }}

        .diff-card {{
            background: white;
            border-radius: 8px;
            padding: 16px 24px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            flex: 1;
        }}

        .diff-year {{
            font-size: 15px;
            font-weight: 600;
            color: {NAVY};
            margin-bottom: 6px;
        }}

        .diff-value {{
            font-size: 20px;
            font-weight: 700;
        }}

        .diff-pct {{
            font-size: 14px;
            margin-top: 2px;
        }}

        /* Narrative sections */
        .narrative {{
            background: white;
            border-radius: 8px;
            padding: 24px 28px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }}

        .narrative-section {{
            margin-bottom: 20px;
        }}

        .narrative-section:last-child {{
            margin-bottom: 0;
        }}

        .narrative p {{
            font-size: 14px;
            line-height: 1.6;
            margin: 8px 0;
        }}

        .narrative ol, .narrative ul {{
            font-size: 14px;
            line-height: 1.6;
            margin: 8px 0 8px 20px;
        }}

        .narrative li {{
            margin-bottom: 8px;
        }}

        /* Legend strip */
        .legend-strip {{
            display: flex;
            gap: 18px;
            padding: 12px 40px;
            background: white;
            border-bottom: 1px solid {LIGHT_GRAY};
            font-size: 12px;
            align-items: center;
            flex-wrap: wrap;
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
            .tab-content {{ display: block !important; page-break-after: always; }}
            .legend-strip {{ display: none; }}
            body {{ background: white; }}
        }}

        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .small-multiples-grid {{ grid-template-columns: 1fr 1fr; }}
            .header {{ padding: 20px; }}
            .tab-nav {{ padding: 0 10px; overflow-x: auto; }}
            .tab-btn {{ padding: 10px 12px; font-size: 12px; }}
            .tab-content {{ padding: 16px; }}
            .diff-summary {{ flex-direction: column; }}
        }}

        @media (max-width: 600px) {{
            .small-multiples-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>North Dakota Population Projections: SDC Method Replication Comparison</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="legend-strip">
        <span style="font-weight:600;">Legend:</span>
        <div class="legend-item">
            <div class="legend-swatch dashed" style="--color:{SDC_COLOR}; background:{SDC_COLOR};"></div>
            <span>SDC 2024 Original (dashed)</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{SDC_NEW_DATA_COLOR};"></div>
            <span>SDC Method + New Data</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:{SCENARIO_COLORS['baseline']};"></div>
            <span>Our Baseline</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch dashed" style="--color:{SCENARIO_COLORS['high_growth']}; background:{SCENARIO_COLORS['high_growth']};"></div>
            <span>Our High Growth (dashed)</span>
        </div>
        <div class="legend-item">
            <div class="legend-swatch dashed" style="--color:{SCENARIO_COLORS['restricted_growth']}; background:{SCENARIO_COLORS['restricted_growth']};"></div>
            <span>Our Restricted (dashed)</span>
        </div>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('state')">State-Level Comparison</button>
        <button class="tab-btn" onclick="switchTab('county')">County Comparison</button>
        <button class="tab-btn" onclick="switchTab('impact')">Data Impact Analysis</button>
        <button class="tab-btn" onclick="switchTab('pyramids')">Population Pyramids</button>
        <button class="tab-btn" onclick="switchTab('methods')">Methods Summary</button>
    </div>

    <div id="tab-state" class="tab-content active">
        {tab1}
    </div>

    <div id="tab-county" class="tab-content">
        {tab2}
    </div>

    <div id="tab-impact" class="tab-content">
        {tab3}
    </div>

    <div id="tab-pyramids" class="tab-content">
        {tab4}
    </div>

    <div id="tab-methods" class="tab-content">
        {tab5}
    </div>

    <div class="footer">
        North Dakota Cohort-Component Population Projections |
        SDC Method Replication Comparison Report |
        Generated on {TODAY.strftime('%B %d, %Y')} |
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
            var tabMap = {{'state': 0, 'county': 1, 'impact': 2, 'pyramids': 3, 'methods': 4}};
            var buttons = document.querySelectorAll('.tab-btn');
            buttons[tabMap[tabId]].classList.add('active');

            // Trigger Plotly resize for proper chart rendering in hidden tabs
            setTimeout(function() {{
                var plots = document.getElementById('tab-' + tabId).querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 50);
        }}

        // Initialize county chart after DOM loaded
        document.addEventListener('DOMContentLoaded', function() {{
            var select = document.getElementById('county-select');
            if (select && select.value) {{
                updateCountyChart(select.value);
            }}
        }});
    </script>
</body>
</html>"""


# ===================================================================
# Main entry point
# ===================================================================

def build_report(output_dir: Path) -> Path:
    """Build the complete SDC replication comparison HTML report.

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
    logger.info("ND SDC Replication Comparison Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    # --- Check for required data files ---
    sdc_new_dir = PROJECT_ROOT / "data" / "exports" / "sdc_method_new_data"
    required_files = [
        sdc_new_dir / "state_population_by_year.csv",
        sdc_new_dir / "county_population_by_year.csv",
        sdc_new_dir / "state_age_sex_by_year.csv",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        logger.error(
            "SDC Method + New Data companion files not found. "
            "Run the companion script first to generate these files."
        )
        for f in missing:
            logger.error("  Missing: %s", f)
        sys.exit(1)

    # Register Plotly template
    template_name = _register_template()

    # Load data
    logger.info("Loading FIPS mapping...")
    name_to_fips, fips_to_name = load_fips_mapping()
    logger.info("  Mapped %d counties", len(name_to_fips))

    logger.info("Loading SDC 2024 original data...")
    sdc_orig = load_sdc_original()

    logger.info("Loading SDC Method + New Data...")
    sdc_new = load_sdc_new_data()

    logger.info("Loading our projection data...")
    our_data = load_our_data()

    # Build tabs
    logger.info("Building Tab 1: State-Level Comparison...")
    tab1 = _build_state_comparison(sdc_orig, sdc_new, our_data, template_name)

    logger.info("Building Tab 2: County Comparison...")
    tab2 = _build_county_comparison(
        sdc_orig, sdc_new, our_data, name_to_fips, fips_to_name, template_name,
    )

    logger.info("Building Tab 3: Data Impact Analysis...")
    tab3 = _build_data_impact(
        sdc_orig, sdc_new, name_to_fips, fips_to_name, template_name,
    )

    logger.info("Building Tab 4: Population Pyramids...")
    tab4 = _build_pyramids(sdc_new, our_data, template_name)

    logger.info("Building Tab 5: Methods Summary...")
    tab5 = _build_methods_summary()

    # Assemble HTML
    logger.info("Assembling HTML report...")
    html = _assemble_html(tab1, tab2, tab3, tab4, tab5)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nd_sdc_replication_comparison_{DATE_STAMP}.html"
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Report written to: %s (%.1f MB)", output_path, file_size_mb)
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build SDC Replication Comparison HTML report",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports",
        help="Directory to write the output HTML file (default: data/exports/)",
    )
    args = parser.parse_args()

    build_report(args.output_dir)


if __name__ == "__main__":
    main()
