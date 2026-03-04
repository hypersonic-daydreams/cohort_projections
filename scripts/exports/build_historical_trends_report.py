#!/usr/bin/env python3
"""
Build interactive HTML report showing historical population trends extended by projections.

Created: 2026-03-02
Author: Claude Code

Purpose
-------
Provides a visual narrative of North Dakota population from 2000 through 2055,
combining Census PEP historical estimates (2000-2025) with cohort-component
projections (2025-2055). This makes it easy to see how projected growth extends
or diverges from recent historical trends, and to compare scenario assumptions
against actual experience.

Method
------
1. Assemble county-level annual population time series from three Census PEP
   vintages:
   - co-est2009-alldata (2000-2009, postcensal/revised)
   - cc-est2020int-alldata (2010-2019, intercensal revised)
   - co-est2024-alldata (2020-2024, postcensal)
   - nd_county_population.csv (2025, Vintage 2025 pre-release)
2. Load projection summary CSVs for baseline, high_growth, and restricted_growth
   scenarios (2025-2055).
3. Stitch historical and projected series at the 2025 base year.
4. Generate four Plotly-based interactive sections:
   - Section 1: State-level historical + projected trend with scenario cone
   - Section 2: County-level explorer with dropdown and small multiples
   - Section 3: Growth by decade (grouped bar)
   - Section 4: Annualized growth rates over time
5. Render as a self-contained HTML file with tabbed navigation.

Key design decisions
--------------------
- **Three-vintage stitching**: Uses co-est2009 for 2000-2009, cc-est2020int for
  2010-2019, and co-est2024 for 2020-2024 to get the best-available (revised)
  estimates for each decade. Intercensal revisions are preferred over postcensal
  where available.
- **Shared-data parquet files**: Reads from ~/workspace/shared-data/census/popest/
  parquet directory (managed by scripts/data/download_census_pep.py) with fallback
  to CSV files in data/raw/population/ for the 2020-2024 vintage.
- **2025 from nd_county_population.csv**: The PEP Vintage 2025 pre-release
  provides the authoritative 2025 base year population used by projections.
- **Standalone HTML**: Keeps this report independent from the main interactive
  report to avoid disrupting the existing 5-tab structure.

Inputs
------
- ~/workspace/shared-data/census/popest/parquet/2000-2009/county/co-est2009-alldata.parquet
- ~/workspace/shared-data/census/popest/parquet/2010-2020/county/cc-est2020int-alldata.parquet
- data/raw/population/co-est2024-alldata.csv (fallback for 2020-2024)
- data/raw/population/nd_county_population.csv (2025 population)
- data/exports/{scenario}/summaries/state_total_population_by_year.csv
- data/exports/{scenario}/summaries/county_total_population_by_year.csv

Outputs
-------
- data/exports/nd_historical_trends_{datestamp}.html

Usage:
    python scripts/exports/build_historical_trends_report.py
    python scripts/exports/build_historical_trends_report.py --no-plotly-js
    python scripts/exports/build_historical_trends_report.py --output-dir data/exports/
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
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import _report_theme as theme  # noqa: E402
from project_utils import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, log_level="INFO")
TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")

# Shared Census PEP data directory
SHARED_POPEST_DIR = Path.home() / "workspace" / "shared-data" / "census" / "popest"

# Scenario definitions
SCENARIOS = ["baseline", "high_growth", "restricted_growth"]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "high_growth": "High Growth",
    "restricted_growth": "Restricted Growth",
}
SCENARIO_COLORS = {
    "baseline": "#0563C1",
    "high_growth": "#00B050",
    "restricted_growth": "#FF0000",
}
SCENARIO_DASHES = {
    "baseline": "solid",
    "high_growth": "dash",
    "restricted_growth": "dash",
}

# Historical line style
HISTORICAL_COLOR = "#1F3864"  # Navy
PROJECTION_CONE_FILL = "rgba(5, 99, 193, 0.1)"

# Key counties for small multiples
KEY_COUNTIES = {
    "38017": "Cass (Fargo)",
    "38015": "Burleigh (Bismarck)",
    "38035": "Grand Forks",
    "38101": "Ward (Minot)",
    "38105": "Williams (Williston)",
    "38021": "Dickey (declining)",
}

# FIPS for the state row
STATE_FIPS = "38"

# cc-est2020int YEAR code -> calendar year mapping
# YEAR 1 = 4/1/2010 Census, YEAR 2 = 7/1/2010, ..., YEAR 12 = 7/1/2020
INTERCENSAL_YEAR_MAP = {str(i): 2008 + i for i in range(2, 12)}
# YEAR 2 -> 2010, YEAR 3 -> 2011, ..., YEAR 11 -> 2019


# ===================================================================
# Data Loading: Historical Population
# ===================================================================


def _load_historical_2000_2009() -> pd.DataFrame:
    """Load county population estimates 2000-2009 from co-est2009 parquet.

    Returns DataFrame with columns: [fips, year, population, county_name].
    """
    pq_path = (
        SHARED_POPEST_DIR / "parquet" / "2000-2009" / "county"
        / "co-est2009-alldata.parquet"
    )

    if not pq_path.exists():
        logger.warning("2000-2009 parquet not found at %s", pq_path)
        return pd.DataFrame(columns=["fips", "year", "population", "county_name"])

    df = pd.read_parquet(pq_path)
    nd = df[df["STNAME"] == "North Dakota"].copy()

    pop_cols = [f"POPESTIMATE{y}" for y in range(2000, 2010)]
    records = []
    for _, row in nd.iterrows():
        county_str = str(row["COUNTY"]).zfill(3)
        fips = f"38{county_str}"
        name = str(row["CTYNAME"])
        for y in range(2000, 2010):
            col = f"POPESTIMATE{y}"
            if col in nd.columns:
                records.append({
                    "fips": fips,
                    "year": y,
                    "population": int(row[col]),
                    "county_name": name,
                })

    result = pd.DataFrame(records)
    logger.info(
        "Loaded 2000-2009 historical: %d rows, %d counties",
        len(result), result["fips"].nunique(),
    )
    return result


def _load_historical_2010_2019() -> pd.DataFrame:
    """Load county population estimates 2010-2019 from cc-est2020int parquet.

    Uses intercensal (revised) estimates which are more accurate than postcensal.
    Returns DataFrame with columns: [fips, year, population, county_name].
    """
    pq_path = (
        SHARED_POPEST_DIR / "parquet" / "2010-2020" / "county"
        / "cc-est2020int-alldata.parquet"
    )

    if not pq_path.exists():
        logger.warning("2010-2020 intercensal parquet not found at %s", pq_path)
        return pd.DataFrame(columns=["fips", "year", "population", "county_name"])

    df = pd.read_parquet(pq_path)
    nd = df[df["STNAME"] == "North Dakota"].copy()

    # Filter to total population (AGEGRP = '0') and years 2-11 (2010-2019)
    totals = nd[(nd["AGEGRP"] == "0") & (nd["YEAR"].isin(list(INTERCENSAL_YEAR_MAP.keys())))].copy()

    records = []
    for _, row in totals.iterrows():
        county_str = str(row["COUNTY"]).zfill(3)
        fips = f"38{county_str}"
        cal_year = INTERCENSAL_YEAR_MAP[str(row["YEAR"])]
        records.append({
            "fips": fips,
            "year": cal_year,
            "population": int(row["TOT_POP"]),
            "county_name": str(row["CTYNAME"]),
        })

    result = pd.DataFrame(records)
    logger.info(
        "Loaded 2010-2019 historical: %d rows, %d counties",
        len(result), result["fips"].nunique(),
    )
    return result


def _load_historical_2020_2024() -> pd.DataFrame:
    """Load county population estimates 2020-2024 from co-est2024.

    Returns DataFrame with columns: [fips, year, population, county_name].
    """
    # Try parquet first, then CSV
    pq_path = (
        SHARED_POPEST_DIR / "parquet" / "2020-2024" / "county"
        / "co-est2024-alldata.parquet"
    )
    csv_path = PROJECT_ROOT / "data" / "raw" / "population" / "co-est2024-alldata.csv"

    df = None
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, encoding="latin1")
    else:
        logger.warning("No 2020-2024 data found")
        return pd.DataFrame(columns=["fips", "year", "population", "county_name"])

    # Filter ND - handle both string and int STATE column
    state_col = df["STATE"]
    if state_col.dtype == object:
        nd = df[state_col.astype(str).str.strip() == "38"].copy()
    else:
        nd = df[state_col == 38].copy()

    pop_cols = [f"POPESTIMATE{y}" for y in range(2020, 2025)]
    records = []
    for _, row in nd.iterrows():
        county_val = str(row["COUNTY"]).zfill(3)
        fips = f"38{county_val}"
        name = str(row["CTYNAME"])
        for y in range(2020, 2025):
            col = f"POPESTIMATE{y}"
            if col in nd.columns:
                records.append({
                    "fips": fips,
                    "year": y,
                    "population": int(row[col]),
                    "county_name": name,
                })

    result = pd.DataFrame(records)
    logger.info(
        "Loaded 2020-2024 historical: %d rows, %d counties",
        len(result), result["fips"].nunique(),
    )
    return result


def _load_historical_2025() -> pd.DataFrame:
    """Load 2025 population from nd_county_population.csv (PEP Vintage 2025).

    Returns DataFrame with columns: [fips, year, population, county_name].
    """
    csv_path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    if not csv_path.exists():
        logger.warning("nd_county_population.csv not found")
        return pd.DataFrame(columns=["fips", "year", "population", "county_name"])

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        fips = str(int(row["county_fips"])).zfill(5)
        records.append({
            "fips": fips,
            "year": 2025,
            "population": int(row["population_2025"]),
            "county_name": str(row["county_name"]),
        })

    result = pd.DataFrame(records)
    logger.info("Loaded 2025 population: %d counties", len(result))
    return result


def load_historical_population() -> pd.DataFrame:
    """Assemble complete county-level historical population 2000-2025.

    Returns DataFrame with columns: [fips, year, population, county_name]
    sorted by fips and year.
    """
    parts = [
        _load_historical_2000_2009(),
        _load_historical_2010_2019(),
        _load_historical_2020_2024(),
        _load_historical_2025(),
    ]

    combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    if combined.empty:
        logger.error("No historical population data loaded!")
        return combined

    # De-duplicate: prefer later vintages (they appear later in concat)
    combined = combined.drop_duplicates(subset=["fips", "year"], keep="last")
    combined = combined.sort_values(["fips", "year"]).reset_index(drop=True)

    # Build state totals by summing counties (exclude state-level row if present)
    county_data = combined[combined["fips"] != "38000"].copy()
    state_totals = (
        county_data.groupby("year")["population"]
        .sum()
        .reset_index()
    )
    state_totals["fips"] = "38"
    state_totals["county_name"] = "North Dakota"

    # Add state totals
    combined = pd.concat([combined, state_totals], ignore_index=True)
    combined = combined.sort_values(["fips", "year"]).reset_index(drop=True)

    logger.info(
        "Historical population assembled: %d rows, years %d-%d, %d geographies",
        len(combined),
        combined["year"].min(),
        combined["year"].max(),
        combined["fips"].nunique(),
    )

    return combined


# ===================================================================
# Data Loading: Projections
# ===================================================================


def load_projection_data() -> dict[str, pd.DataFrame]:
    """Load projection summary data for all scenarios.

    Returns dict keyed by scenario name, each value a DataFrame with
    columns: [fips, year, population] in long format.
    """
    result = {}
    for scenario in SCENARIOS:
        state_path = (
            PROJECT_ROOT / "data" / "exports" / scenario / "summaries"
            / "state_total_population_by_year.csv"
        )
        county_path = (
            PROJECT_ROOT / "data" / "exports" / scenario / "summaries"
            / "county_total_population_by_year.csv"
        )

        dfs = []
        for path in [state_path, county_path]:
            if not path.exists():
                logger.warning("Projection file not found: %s", path)
                continue
            df = pd.read_csv(path)
            # Wide -> long: fips column + year columns
            id_col = "fips"
            year_cols = [c for c in df.columns if c != id_col and c.isdigit()]
            long = df.melt(
                id_vars=[id_col],
                value_vars=year_cols,
                var_name="year",
                value_name="population",
            )
            long["year"] = long["year"].astype(int)
            long["fips"] = long["fips"].astype(str)
            dfs.append(long)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            result[scenario] = combined
            logger.info(
                "Loaded %s projections: %d rows, years %d-%d",
                scenario, len(combined),
                combined["year"].min(), combined["year"].max(),
            )

    return result


# ===================================================================
# County Name Mapping
# ===================================================================


def build_county_name_map(historical: pd.DataFrame) -> dict[str, str]:
    """Build FIPS -> county name mapping from historical data."""
    name_map = {}
    for _, row in historical[["fips", "county_name"]].drop_duplicates().iterrows():
        name_map[str(row["fips"])] = str(row["county_name"])
    # Ensure state is labeled
    name_map["38"] = "North Dakota"
    return name_map


# ===================================================================
# Section 1: State-Level Historical + Projected
# ===================================================================


def build_state_trend_chart(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
) -> str:
    """Build state-level chart: historical 2000-2025 + projected 2025-2055."""
    template_name = theme.register_template()
    fig = go.Figure()

    # Historical state data
    state_hist = historical[historical["fips"] == "38"].sort_values("year")

    if not state_hist.empty:
        fig.add_trace(go.Scatter(
            x=state_hist["year"],
            y=state_hist["population"],
            name="Historical (PEP)",
            mode="lines+markers",
            line={"color": HISTORICAL_COLOR, "width": 3},
            marker={"size": 4, "color": HISTORICAL_COLOR},
            hovertemplate=(
                "<b>Historical</b><br>"
                "Year: %{x}<br>"
                "Population: %{y:,.0f}<extra></extra>"
            ),
        ))

        # Census year markers (2000, 2010, 2020)
        census_years = [2000, 2010, 2020]
        census_data = state_hist[state_hist["year"].isin(census_years)]
        if not census_data.empty:
            fig.add_trace(go.Scatter(
                x=census_data["year"],
                y=census_data["population"],
                name="Census Years",
                mode="markers+text",
                marker={
                    "size": 12,
                    "color": HISTORICAL_COLOR,
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "white"},
                },
                text=[f"{int(p):,}" for p in census_data["population"]],
                textposition="top center",
                textfont={"size": 10, "color": HISTORICAL_COLOR},
                hovertemplate=(
                    "<b>Census %{x}</b><br>"
                    "Population: %{y:,.0f}<extra></extra>"
                ),
                showlegend=True,
            ))

    # Projection cone (shaded area between high and restricted)
    if "high_growth" in projections and "restricted_growth" in projections:
        high = projections["high_growth"]
        restricted = projections["restricted_growth"]
        high_state = high[high["fips"] == "38"].sort_values("year")
        rest_state = restricted[restricted["fips"] == "38"].sort_values("year")

        if not high_state.empty and not rest_state.empty:
            # Merge on year for aligned fill
            merged = high_state.merge(
                rest_state[["year", "population"]],
                on="year",
                suffixes=("_high", "_low"),
            )
            # Only show cone from 2026 onward (2025 is shared starting point)
            cone = merged[merged["year"] >= 2026]

            if not cone.empty:
                fig.add_trace(go.Scatter(
                    x=pd.concat([cone["year"], cone["year"][::-1]]),
                    y=pd.concat([cone["population_high"], cone["population_low"][::-1]]),
                    fill="toself",
                    fillcolor=PROJECTION_CONE_FILL,
                    line={"color": "rgba(0,0,0,0)"},
                    name="Projection Range",
                    showlegend=True,
                    hoverinfo="skip",
                ))

    # Projection scenario lines
    for scenario in SCENARIOS:
        if scenario not in projections:
            continue
        proj = projections[scenario]
        state_proj = proj[proj["fips"] == "38"].sort_values("year")
        # Start from 2025 for continuity
        state_proj = state_proj[state_proj["year"] >= 2025]

        if state_proj.empty:
            continue

        label = SCENARIO_LABELS[scenario]
        color = SCENARIO_COLORS[scenario]
        dash = SCENARIO_DASHES[scenario]

        fig.add_trace(go.Scatter(
            x=state_proj["year"],
            y=state_proj["population"],
            name=f"{label} Projection",
            mode="lines",
            line={"color": color, "width": 2.5, "dash": dash},
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Year: %{x}<br>"
                "Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    # Vertical line at base year 2025
    fig.add_vline(
        x=2025, line_dash="dot", line_color="#808080", line_width=1.5,
        annotation_text="Base Year 2025",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#808080",
    )

    # Growth rate annotations
    if not state_hist.empty:
        pop_2000 = state_hist[state_hist["year"] == 2000]["population"].iloc[0] if 2000 in state_hist["year"].values else None
        pop_2025 = state_hist[state_hist["year"] == 2025]["population"].iloc[0] if 2025 in state_hist["year"].values else None

        if pop_2000 is not None and pop_2025 is not None:
            hist_growth = (pop_2025 - pop_2000) / pop_2000 * 100
            fig.add_annotation(
                x=2012, y=pop_2000 + (pop_2025 - pop_2000) * 0.3,
                text=f"Historical: +{hist_growth:.1f}% (2000-2025)",
                showarrow=False,
                font={"size": 11, "color": HISTORICAL_COLOR},
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=HISTORICAL_COLOR,
                borderwidth=1,
                borderpad=4,
            )

    if "baseline" in projections:
        bl = projections["baseline"]
        bl_state = bl[bl["fips"] == "38"].sort_values("year")
        if not bl_state.empty:
            pop_2025_proj = bl_state[bl_state["year"] == 2025]["population"]
            pop_2055_proj = bl_state[bl_state["year"] == 2055]["population"]
            if not pop_2025_proj.empty and not pop_2055_proj.empty:
                p25 = pop_2025_proj.iloc[0]
                p55 = pop_2055_proj.iloc[0]
                proj_growth = (p55 - p25) / p25 * 100
                fig.add_annotation(
                    x=2042, y=p55 * 0.97,
                    text=f"Baseline: +{proj_growth:.1f}% (2025-2055)",
                    showarrow=False,
                    font={"size": 11, "color": SCENARIO_COLORS["baseline"]},
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=SCENARIO_COLORS["baseline"],
                    borderwidth=1,
                    borderpad=4,
                )

    fig.update_layout(
        template=template_name,
        title="North Dakota Population: Historical Trends & Projections (2000-2055)",
        xaxis_title="Year",
        yaxis_title="Total Population",
        yaxis_tickformat=",",
        height=500,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        xaxis={"range": [1999, 2056], "dtick": 5},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="state-trend")


# ===================================================================
# Section 2: County Explorer
# ===================================================================


def build_county_dropdown_chart(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
    name_map: dict[str, str],
) -> str:
    """Build county-level chart with JavaScript dropdown for county selection.

    Returns HTML+JS for an interactive chart where users can select any county.
    """
    template_name = theme.register_template()

    # Get list of all county FIPS (exclude state)
    all_fips = sorted(
        [f for f in historical["fips"].unique() if f != "38" and f != "38000"]
    )

    # Prepare data as JSON for JavaScript
    county_data: dict[str, dict[str, Any]] = {}
    for fips in all_fips:
        name = name_map.get(fips, fips)
        hist = historical[historical["fips"] == fips].sort_values("year")

        entry: dict[str, Any] = {
            "name": name,
            "hist_years": hist["year"].tolist(),
            "hist_pop": hist["population"].tolist(),
        }

        for scenario in SCENARIOS:
            if scenario in projections:
                proj = projections[scenario]
                county_proj = proj[proj["fips"].astype(str) == fips].sort_values("year")
                county_proj = county_proj[county_proj["year"] >= 2025]
                entry[f"{scenario}_years"] = county_proj["year"].tolist()
                entry[f"{scenario}_pop"] = [round(p, 0) for p in county_proj["population"].tolist()]

        county_data[fips] = entry

    import json
    data_json = json.dumps(county_data)

    # Build dropdown options HTML
    options_html = ""
    for fips in all_fips:
        name = name_map.get(fips, fips)
        selected = ' selected' if fips == "38017" else ''  # Default to Cass
        options_html += f'<option value="{fips}"{selected}>{name} ({fips})</option>\n'

    html = f"""
    <div style="margin-bottom: 15px;">
        <label for="county-select" style="font-weight: 600; font-size: 14px; color: {HISTORICAL_COLOR};">
            Select County:
        </label>
        <select id="county-select" onchange="updateCountyChart()"
                style="padding: 8px 12px; font-size: 14px; border: 1px solid #D9D9D9;
                       border-radius: 4px; margin-left: 10px; min-width: 250px;">
            {options_html}
        </select>
    </div>
    <div id="county-chart" style="width: 100%; height: 480px;"></div>
    <script>
    var countyData = {data_json};

    function updateCountyChart() {{
        var fips = document.getElementById('county-select').value;
        var d = countyData[fips];
        if (!d) return;

        var traces = [];

        // Historical line
        traces.push({{
            x: d.hist_years,
            y: d.hist_pop,
            name: 'Historical (PEP)',
            mode: 'lines+markers',
            line: {{color: '{HISTORICAL_COLOR}', width: 3}},
            marker: {{size: 4}},
            hovertemplate: '<b>Historical</b><br>Year: %{{x}}<br>Population: %{{y:,.0f}}<extra></extra>'
        }});

        // Census markers
        var censusYears = [2000, 2010, 2020];
        var cx = [], cy = [], ct = [];
        for (var i = 0; i < d.hist_years.length; i++) {{
            if (censusYears.indexOf(d.hist_years[i]) >= 0) {{
                cx.push(d.hist_years[i]);
                cy.push(d.hist_pop[i]);
                ct.push(d.hist_pop[i].toLocaleString());
            }}
        }}
        if (cx.length > 0) {{
            traces.push({{
                x: cx, y: cy,
                name: 'Census Years',
                mode: 'markers+text',
                marker: {{size: 10, color: '{HISTORICAL_COLOR}', symbol: 'diamond',
                         line: {{width: 2, color: 'white'}}}},
                text: ct,
                textposition: 'top center',
                textfont: {{size: 9, color: '{HISTORICAL_COLOR}'}},
                hovertemplate: '<b>Census %{{x}}</b><br>Population: %{{y:,.0f}}<extra></extra>'
            }});
        }}

        // Projection cone
        if (d.high_growth_years && d.restricted_growth_years) {{
            var hYears = d.high_growth_years.slice(1);
            var hPop = d.high_growth_pop.slice(1);
            var rPop = d.restricted_growth_pop.slice(1);
            var coneX = hYears.concat(hYears.slice().reverse());
            var coneY = hPop.concat(rPop.slice().reverse());
            traces.push({{
                x: coneX, y: coneY,
                fill: 'toself',
                fillcolor: '{PROJECTION_CONE_FILL}',
                line: {{color: 'rgba(0,0,0,0)'}},
                name: 'Projection Range',
                hoverinfo: 'skip'
            }});
        }}

        // Scenario lines
        var scenarios = [
            ['baseline', 'Baseline', '{SCENARIO_COLORS["baseline"]}', 'solid'],
            ['high_growth', 'High Growth', '{SCENARIO_COLORS["high_growth"]}', 'dash'],
            ['restricted_growth', 'Restricted Growth', '{SCENARIO_COLORS["restricted_growth"]}', 'dash']
        ];
        for (var s = 0; s < scenarios.length; s++) {{
            var key = scenarios[s][0];
            var label = scenarios[s][1];
            var color = scenarios[s][2];
            var dashStyle = scenarios[s][3];
            if (d[key + '_years']) {{
                traces.push({{
                    x: d[key + '_years'],
                    y: d[key + '_pop'],
                    name: label + ' Projection',
                    mode: 'lines',
                    line: {{color: color, width: 2.5, dash: dashStyle}},
                    hovertemplate: '<b>' + label + '</b><br>Year: %{{x}}<br>Population: %{{y:,.0f}}<extra></extra>'
                }});
            }}
        }}

        var layout = {{
            title: d.name + ' Population: Historical & Projected',
            xaxis: {{title: 'Year', range: [1999, 2056], dtick: 5}},
            yaxis: {{title: 'Total Population', tickformat: ','}},
            font: {{family: "{theme.FONT_FAMILY}"}},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            height: 480,
            legend: {{orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'center', x: 0.5}},
            shapes: [{{
                type: 'line', x0: 2025, x1: 2025, y0: 0, y1: 1,
                yref: 'paper', line: {{color: '#808080', width: 1.5, dash: 'dot'}}
            }}],
            annotations: [{{
                x: 2025, y: 1.05, yref: 'paper', text: 'Base Year 2025',
                showarrow: false, font: {{size: 11, color: '#808080'}}
            }}]
        }};

        Plotly.newPlot('county-chart', traces, layout, {{responsive: true}});
    }}

    // Initialize
    document.addEventListener('DOMContentLoaded', function() {{ updateCountyChart(); }});
    </script>
    """
    return html


def build_small_multiples(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
    name_map: dict[str, str],
) -> str:
    """Build small multiples grid for key counties."""
    template_name = theme.register_template()
    from plotly.subplots import make_subplots

    n_counties = len(KEY_COUNTIES)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v for v in KEY_COUNTIES.values()],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (fips, display_name) in enumerate(KEY_COUNTIES.items()):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Historical
        hist = historical[historical["fips"] == fips].sort_values("year")
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["year"], y=hist["population"],
                    mode="lines",
                    line={"color": HISTORICAL_COLOR, "width": 2},
                    showlegend=(idx == 0),
                    name="Historical",
                    hovertemplate="Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
                ),
                row=row, col=col,
            )

        # Baseline projection
        if "baseline" in projections:
            proj = projections["baseline"]
            county_proj = proj[proj["fips"].astype(str) == fips].sort_values("year")
            county_proj = county_proj[county_proj["year"] >= 2025]
            if not county_proj.empty:
                fig.add_trace(
                    go.Scatter(
                        x=county_proj["year"], y=county_proj["population"],
                        mode="lines",
                        line={"color": SCENARIO_COLORS["baseline"], "width": 2},
                        showlegend=(idx == 0),
                        name="Baseline",
                        hovertemplate="Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
                    ),
                    row=row, col=col,
                )

        # High growth
        if "high_growth" in projections:
            proj = projections["high_growth"]
            county_proj = proj[proj["fips"].astype(str) == fips].sort_values("year")
            county_proj = county_proj[county_proj["year"] >= 2025]
            if not county_proj.empty:
                fig.add_trace(
                    go.Scatter(
                        x=county_proj["year"], y=county_proj["population"],
                        mode="lines",
                        line={"color": SCENARIO_COLORS["high_growth"], "width": 1.5, "dash": "dash"},
                        showlegend=(idx == 0),
                        name="High Growth",
                        hovertemplate="Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
                    ),
                    row=row, col=col,
                )

        # Restricted
        if "restricted_growth" in projections:
            proj = projections["restricted_growth"]
            county_proj = proj[proj["fips"].astype(str) == fips].sort_values("year")
            county_proj = county_proj[county_proj["year"] >= 2025]
            if not county_proj.empty:
                fig.add_trace(
                    go.Scatter(
                        x=county_proj["year"], y=county_proj["population"],
                        mode="lines",
                        line={"color": SCENARIO_COLORS["restricted_growth"], "width": 1.5, "dash": "dash"},
                        showlegend=(idx == 0),
                        name="Restricted",
                        hovertemplate="Year: %{x}<br>Pop: %{y:,.0f}<extra></extra>",
                    ),
                    row=row, col=col,
                )

        # Base year marker
        fig.add_vline(
            x=2025, line_dash="dot", line_color="#808080", line_width=1,
            row=row, col=col,
        )

    fig.update_layout(
        template=template_name,
        title="Key County Trends: Historical & Projected",
        height=600,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "center", "x": 0.5},
    )

    # Format all y-axes
    for i in range(1, n_counties + 1):
        fig.update_yaxes(tickformat=",", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="small-multiples")


# ===================================================================
# Section 3: Growth Era Analysis
# ===================================================================


def build_growth_era_chart(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
) -> str:
    """Build grouped bar chart showing population change by decade."""
    template_name = theme.register_template()

    # Define eras
    eras = [
        ("2000-2010", 2000, 2010, "historical"),
        ("2010-2020", 2010, 2020, "historical"),
        ("2020-2025", 2020, 2025, "historical"),
        ("2025-2035", 2025, 2035, "projected"),
        ("2035-2045", 2035, 2045, "projected"),
        ("2045-2055", 2045, 2055, "projected"),
    ]

    # State-level data
    state_hist = historical[historical["fips"] == "38"].set_index("year")
    baseline_proj = None
    if "baseline" in projections:
        bp = projections["baseline"]
        baseline_proj = bp[bp["fips"] == "38"].set_index("year")

    era_labels = []
    era_changes = []
    era_colors = []

    for label, start, end, source in eras:
        change = None
        if source == "historical":
            if start in state_hist.index and end in state_hist.index:
                change = int(state_hist.loc[end, "population"]) - int(state_hist.loc[start, "population"])
        else:
            if baseline_proj is not None:
                if start in baseline_proj.index and end in baseline_proj.index:
                    change = baseline_proj.loc[end, "population"] - baseline_proj.loc[start, "population"]

        era_labels.append(label)
        era_changes.append(round(change) if change is not None else 0)
        era_colors.append(HISTORICAL_COLOR if source == "historical" else SCENARIO_COLORS["baseline"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=era_labels,
        y=era_changes,
        marker_color=era_colors,
        text=[f"{c:+,}" for c in era_changes],
        textposition="outside",
        textfont={"size": 12},
        hovertemplate="<b>%{x}</b><br>Population Change: %{y:+,.0f}<extra></extra>",
    ))

    fig.update_layout(
        template=template_name,
        title="North Dakota Population Change by Period",
        xaxis_title="Period",
        yaxis_title="Population Change",
        yaxis_tickformat=",",
        height=420,
        showlegend=False,
    )

    # Add a legend-like annotation
    fig.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text=(
            f'<span style="color:{HISTORICAL_COLOR};">&#9632;</span> Historical &nbsp;&nbsp;'
            f'<span style="color:{SCENARIO_COLORS["baseline"]};">&#9632;</span> Projected (Baseline)'
        ),
        showarrow=False,
        font={"size": 12},
        align="center",
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="growth-era")


def build_county_growth_era_chart(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
    name_map: dict[str, str],
) -> str:
    """Build grouped bar chart for top counties by decade."""
    template_name = theme.register_template()

    # Select top counties by 2025 population
    pop_2025 = historical[(historical["year"] == 2025) & (historical["fips"] != "38") & (historical["fips"] != "38000")]
    top_counties = pop_2025.nlargest(6, "population")["fips"].tolist()

    eras = [
        ("2000-2010", 2000, 2010, "historical"),
        ("2010-2020", 2010, 2020, "historical"),
        ("2020-2025", 2020, 2025, "historical"),
        ("2025-2035", 2025, 2035, "projected"),
        ("2035-2045", 2035, 2045, "projected"),
        ("2045-2055", 2045, 2055, "projected"),
    ]

    era_palette = ["#1F3864", "#2E5090", "#4472C4", "#0563C1", "#00B0F0", "#70C5E8"]

    fig = go.Figure()

    for era_idx, (label, start, end, source) in enumerate(eras):
        changes = []
        county_labels = []

        for fips in top_counties:
            name = name_map.get(fips, fips)
            # Shorten name
            short = name.replace(" County", "")
            county_labels.append(short)

            change = 0
            hist = historical[historical["fips"] == fips].set_index("year")
            if source == "historical":
                if start in hist.index and end in hist.index:
                    change = int(hist.loc[end, "population"]) - int(hist.loc[start, "population"])
            else:
                if "baseline" in projections:
                    bp = projections["baseline"]
                    proj = bp[bp["fips"].astype(str) == fips].set_index("year")
                    if start in proj.index and end in proj.index:
                        change = round(proj.loc[end, "population"] - proj.loc[start, "population"])
            changes.append(change)

        fig.add_trace(go.Bar(
            x=county_labels,
            y=changes,
            name=label,
            marker_color=era_palette[era_idx],
            hovertemplate=f"<b>{label}</b><br>" + "%{x}<br>Change: %{y:+,.0f}<extra></extra>",
        ))

    fig.update_layout(
        template=template_name,
        title="Population Change by Period: Top 6 Counties",
        xaxis_title="County",
        yaxis_title="Population Change",
        yaxis_tickformat=",",
        barmode="group",
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="county-growth-era")


# ===================================================================
# Section 4: Annualized Growth Rates
# ===================================================================


def build_growth_rate_chart(
    historical: pd.DataFrame,
    projections: dict[str, pd.DataFrame],
) -> str:
    """Build line chart of annualized growth rates for state."""
    template_name = theme.register_template()
    fig = go.Figure()

    # Historical growth rates
    state_hist = historical[historical["fips"] == "38"].sort_values("year")
    if len(state_hist) > 1:
        hist_years = state_hist["year"].values
        hist_pop = state_hist["population"].values
        growth_years = hist_years[1:]
        growth_rates = [(hist_pop[i] - hist_pop[i - 1]) / hist_pop[i - 1] * 100
                        for i in range(1, len(hist_pop))]

        fig.add_trace(go.Scatter(
            x=growth_years.tolist(),
            y=growth_rates,
            name="Historical Growth Rate",
            mode="lines+markers",
            line={"color": HISTORICAL_COLOR, "width": 2},
            marker={"size": 3},
            hovertemplate=(
                "<b>Historical</b><br>"
                "Year: %{x}<br>"
                "Growth Rate: %{y:.2f}%<extra></extra>"
            ),
        ))

    # Projected growth rates per scenario
    for scenario in SCENARIOS:
        if scenario not in projections:
            continue
        proj = projections[scenario]
        state_proj = proj[proj["fips"] == "38"].sort_values("year")
        state_proj = state_proj[state_proj["year"] >= 2025]

        if len(state_proj) > 1:
            proj_years = state_proj["year"].values
            proj_pop = state_proj["population"].values
            g_years = proj_years[1:]
            g_rates = [(proj_pop[i] - proj_pop[i - 1]) / proj_pop[i - 1] * 100
                       for i in range(1, len(proj_pop))]

            label = SCENARIO_LABELS[scenario]
            color = SCENARIO_COLORS[scenario]
            dash = SCENARIO_DASHES[scenario]

            fig.add_trace(go.Scatter(
                x=g_years.tolist(),
                y=g_rates,
                name=f"{label} Growth Rate",
                mode="lines",
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Year: %{x}<br>"
                    "Growth Rate: %{y:.2f}%<extra></extra>"
                ),
            ))

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#D9D9D9", line_width=1)

    # Base year marker
    fig.add_vline(
        x=2025, line_dash="dot", line_color="#808080", line_width=1.5,
        annotation_text="Base Year",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#808080",
    )

    fig.update_layout(
        template=template_name,
        title="North Dakota Annual Population Growth Rate (2001-2055)",
        xaxis_title="Year",
        yaxis_title="Annual Growth Rate (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        xaxis={"range": [2000, 2056], "dtick": 5},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="growth-rates")


# ===================================================================
# Plotly.js Bundling
# ===================================================================


def get_plotly_js_tag(inline: bool = True) -> str:
    """Return the Plotly.js <script> tag, either inline or CDN."""
    if not inline:
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    try:
        import plotly
        plotly_dir = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if plotly_dir.exists():
            js_content = plotly_dir.read_text(encoding="utf-8")
            return f"<script>{js_content}</script>"
    except Exception:
        pass

    logger.warning("Falling back to CDN for Plotly.js")
    return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


# ===================================================================
# HTML Assembly
# ===================================================================


def build_html_report(
    state_trend_html: str,
    county_dropdown_html: str,
    small_multiples_html: str,
    growth_era_html: str,
    county_growth_era_html: str,
    growth_rate_html: str,
    plotly_js_tag: str,
) -> str:
    """Assemble the complete standalone HTML report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>North Dakota Historical Population Trends &amp; Projections</title>
    {plotly_js_tag}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: {theme.FONT_FAMILY};
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}

        .report-header {{
            background: linear-gradient(135deg, #1F3864 0%, #0563C1 100%);
            color: white;
            padding: 30px 40px;
            text-align: center;
        }}
        .report-header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .report-header .subtitle {{
            font-size: 14px;
            opacity: 0.85;
        }}

        .tab-nav {{
            display: flex;
            background: white;
            border-bottom: 2px solid #e0e0e0;
            padding: 0 20px;
            overflow-x: auto;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .tab-btn {{
            padding: 14px 24px;
            cursor: pointer;
            border: none;
            background: none;
            font-family: {theme.FONT_FAMILY};
            font-size: 14px;
            font-weight: 500;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab-btn:hover {{ color: #0563C1; }}
        .tab-btn.active {{
            color: #0563C1;
            border-bottom-color: #0563C1;
            font-weight: 600;
        }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .section {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 40px;
        }}
        .section h2 {{
            color: #1F3864;
            font-size: 22px;
            margin-bottom: 8px;
        }}
        .section .section-desc {{
            color: #666;
            font-size: 14px;
            margin-bottom: 24px;
            line-height: 1.5;
        }}

        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}

        .methodology-note {{
            background: #f0f4f8;
            border-left: 4px solid #0563C1;
            padding: 16px 20px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
            font-size: 13px;
            color: #444;
        }}
        .methodology-note strong {{
            color: #1F3864;
        }}

        .data-source-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 13px;
        }}
        .data-source-table th {{
            background: #1F3864;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
        }}
        .data-source-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .data-source-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #999;
            border-top: 1px solid #e0e0e0;
            margin-top: 40px;
        }}
    </style>
</head>
<body>

<div class="report-header">
    <h1>North Dakota Population: Historical Trends &amp; Projections</h1>
    <div class="subtitle">
        Census PEP Estimates 2000-2025 | Cohort-Component Projections 2025-2055 |
        Generated {TODAY.strftime("%B %d, %Y")}
    </div>
</div>

<div class="tab-nav">
    <button class="tab-btn active" onclick="showTab('state-trends')">State Trends</button>
    <button class="tab-btn" onclick="showTab('county-explorer')">County Explorer</button>
    <button class="tab-btn" onclick="showTab('growth-eras')">Growth by Period</button>
    <button class="tab-btn" onclick="showTab('growth-rates')">Growth Rates</button>
    <button class="tab-btn" onclick="showTab('data-sources')">Data Sources</button>
</div>

<!-- Section 1: State Trends -->
<div id="state-trends" class="tab-content active">
    <div class="section">
        <h2>State-Level Population Trends</h2>
        <p class="section-desc">
            North Dakota's population trajectory from the 2000 Census through 2055 projections.
            The solid navy line shows historical Census PEP estimates, while the colored lines
            show three projection scenarios diverging from the 2025 base year. The shaded band
            indicates the range between high growth and restricted growth scenarios.
        </p>
        <div class="chart-container">
            {state_trend_html}
        </div>
        <div class="methodology-note">
            <strong>Reading this chart:</strong> Diamond markers indicate decennial Census
            counts (2000, 2010, 2020). The vertical dashed line marks the 2025 base year
            where projections begin. Historical data comes from Census PEP revised
            intercensal estimates; projections use the cohort-component method with
            three CBO-grounded scenarios (ADR-037).
        </div>
    </div>
</div>

<!-- Section 2: County Explorer -->
<div id="county-explorer" class="tab-content">
    <div class="section">
        <h2>County-Level Explorer</h2>
        <p class="section-desc">
            Select any of North Dakota's 53 counties to see its historical population trend
            extended by projection scenarios. The small multiples below show six key counties
            for quick comparison.
        </p>
        <div class="chart-container">
            {county_dropdown_html}
        </div>
        <div class="chart-container">
            {small_multiples_html}
        </div>
    </div>
</div>

<!-- Section 3: Growth Eras -->
<div id="growth-eras" class="tab-content">
    <div class="section">
        <h2>Population Change by Period</h2>
        <p class="section-desc">
            Absolute population change broken into historical periods (2000-2010, 2010-2020,
            2020-2025) and projected periods (2025-2035, 2035-2045, 2045-2055) using the
            baseline scenario. This reveals the acceleration and deceleration of growth over time.
        </p>
        <div class="chart-container">
            {growth_era_html}
        </div>
        <div class="chart-container">
            {county_growth_era_html}
        </div>
    </div>
</div>

<!-- Section 4: Growth Rates -->
<div id="growth-rates" class="tab-content">
    <div class="section">
        <h2>Annualized Growth Rates</h2>
        <p class="section-desc">
            Year-over-year population growth rates show how growth has evolved historically
            and how each projection scenario envisions the trajectory. Rates above 0% indicate
            population growth; below 0% indicates decline.
        </p>
        <div class="chart-container">
            {growth_rate_html}
        </div>
        <div class="methodology-note">
            <strong>Note:</strong> Historical growth rates show year-over-year percentage
            changes derived from Census PEP estimates. The 2010-2015 growth spike reflects
            the Bakken oil boom's impact on statewide population. Projected rates gradually
            moderate as the cohort-component model converges toward long-run demographic
            equilibrium.
        </div>
    </div>
</div>

<!-- Section 5: Data Sources -->
<div id="data-sources" class="tab-content">
    <div class="section">
        <h2>Data Sources &amp; Methodology</h2>
        <p class="section-desc">
            This report combines multiple Census Bureau Population Estimates Program (PEP)
            vintages with cohort-component projections to present a continuous population
            narrative from 2000 through 2055.
        </p>

        <h3 style="color: #1F3864; margin: 20px 0 12px;">Historical Data Sources</h3>
        <table class="data-source-table">
            <tr>
                <th>Period</th>
                <th>Source</th>
                <th>Type</th>
                <th>Notes</th>
            </tr>
            <tr>
                <td>2000-2009</td>
                <td>co-est2009-alldata</td>
                <td>Postcensal (final)</td>
                <td>Census PEP county estimates with 2010 Census benchmark</td>
            </tr>
            <tr>
                <td>2010-2019</td>
                <td>cc-est2020int-alldata</td>
                <td>Intercensal (revised)</td>
                <td>Revised estimates incorporating 2020 Census results; more accurate than postcensal</td>
            </tr>
            <tr>
                <td>2020-2024</td>
                <td>co-est2024-alldata</td>
                <td>Postcensal</td>
                <td>Current vintage; will be revised after 2030 Census</td>
            </tr>
            <tr>
                <td>2025</td>
                <td>nd_county_population.csv</td>
                <td>PEP Vintage 2025 (pre-release)</td>
                <td>Base year for projections; stcoreview extract</td>
            </tr>
        </table>

        <h3 style="color: #1F3864; margin: 20px 0 12px;">Projection Scenarios</h3>
        <table class="data-source-table">
            <tr>
                <th>Scenario</th>
                <th>Description</th>
                <th>Key Assumptions</th>
            </tr>
            <tr>
                <td style="color: #0563C1; font-weight: 600;">Baseline</td>
                <td>Trend continuation</td>
                <td>Recent migration and fertility trends continue; CBO economic trajectory</td>
            </tr>
            <tr>
                <td style="color: #00B050; font-weight: 600;">High Growth</td>
                <td>Elevated immigration</td>
                <td>Higher net in-migration; robust economic conditions</td>
            </tr>
            <tr>
                <td style="color: #FF0000; font-weight: 600;">Restricted Growth</td>
                <td>CBO policy-adjusted</td>
                <td>Reduced immigration; policy-constrained growth</td>
            </tr>
        </table>

        <div class="methodology-note" style="margin-top: 24px;">
            <strong>Data Assembly Method:</strong> County-level annual population estimates
            were stitched from three PEP vintages, preferring intercensal revisions where
            available. State totals are computed bottom-up as the sum of 53 county
            populations (per ADR-054). The 2025 base year population serves as the junction
            point between historical estimates and forward projections. All projections
            use the cohort-component method (ADR-012, ADR-037).
        </div>
    </div>
</div>

<div class="footer">
    North Dakota Population Projections | State Data Center |
    Generated {TODAY.strftime("%B %d, %Y")} |
    Historical data: Census PEP 2000-2025 | Projections: Cohort-Component 2025-2055
</div>

<script>
function showTab(tabId) {{
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(function(el) {{
        el.classList.remove('active');
    }});
    document.querySelectorAll('.tab-btn').forEach(function(el) {{
        el.classList.remove('active');
    }});

    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');

    // Trigger resize on Plotly charts to fix layout
    window.dispatchEvent(new Event('resize'));

    // Initialize county chart if switching to county explorer for first time
    if (tabId === 'county-explorer' && typeof updateCountyChart === 'function') {{
        setTimeout(function() {{ updateCountyChart(); }}, 100);
    }}
}}
</script>

</body>
</html>"""


# ===================================================================
# Main Pipeline
# ===================================================================


def build_report(
    output_dir: Path,
    inline_plotly: bool = True,
) -> Path:
    """Build the complete historical trends report.

    Parameters
    ----------
    output_dir : Path
        Directory to write the output HTML file.
    inline_plotly : bool
        Whether to inline Plotly.js or use CDN.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    logger.info("=" * 60)
    logger.info("ND Historical Trends Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    # --- Load Data ---
    logger.info("Loading historical population data...")
    historical = load_historical_population()

    if historical.empty:
        logger.error("No historical data loaded. Cannot build report.")
        sys.exit(1)

    logger.info("Loading projection data...")
    projections = load_projection_data()

    name_map = build_county_name_map(historical)

    # --- Build Sections ---
    logger.info("Building Section 1: State trend chart...")
    state_trend_html = build_state_trend_chart(historical, projections)

    logger.info("Building Section 2: County explorer...")
    county_dropdown_html = build_county_dropdown_chart(historical, projections, name_map)
    small_multiples_html = build_small_multiples(historical, projections, name_map)

    logger.info("Building Section 3: Growth era charts...")
    growth_era_html = build_growth_era_chart(historical, projections)
    county_growth_era_html = build_county_growth_era_chart(historical, projections, name_map)

    logger.info("Building Section 4: Growth rate chart...")
    growth_rate_html = build_growth_rate_chart(historical, projections)

    # --- Assemble HTML ---
    logger.info("Assembling HTML...")
    plotly_js_tag = get_plotly_js_tag(inline=inline_plotly)

    html_output = build_html_report(
        state_trend_html=state_trend_html,
        county_dropdown_html=county_dropdown_html,
        small_multiples_html=small_multiples_html,
        growth_era_html=growth_era_html,
        county_growth_era_html=county_growth_era_html,
        growth_rate_html=growth_rate_html,
        plotly_js_tag=plotly_js_tag,
    )

    # --- Write Output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nd_historical_trends_{DATE_STAMP}.html"
    output_path.write_text(html_output, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Report generated successfully!")
    logger.info("  Output: %s", output_path)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  Plotly.js: %s", "inline" if inline_plotly else "CDN")
    logger.info("  Sections: 5 tabs")
    logger.info("=" * 60)

    return output_path


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build interactive HTML report showing historical population trends "
            "extended by cohort-component projections for North Dakota."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports",
        help="Output directory (default: data/exports/)",
    )
    parser.add_argument(
        "--no-plotly-js",
        action="store_true",
        help="Use CDN Plotly.js instead of inline (smaller file, requires internet)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    try:
        output_path = build_report(
            output_dir=args.output_dir,
            inline_plotly=not args.no_plotly_js,
        )
        logger.info("Done. Report at: %s", output_path)
        return 0
    except Exception:
        logger.exception("Failed to build historical trends report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
