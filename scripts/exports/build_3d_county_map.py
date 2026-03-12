#!/usr/bin/env python3
"""
Build a standalone interactive 3D extruded county population map for North Dakota.

Loads TIGER 2020 county boundaries and cohort-component projection results,
merges them into a single GeoJSON structure, and generates a standalone HTML
file that uses deck.gl (via CDN) with MapLibre GL JS for a 3D extruded
polygon visualization.

Features:
    - Year slider (2025-2055) with play/pause animation
    - County selection/isolation with dynamic z-axis rescaling
    - Linear / log scale toggle for extrusion heights
    - Hover tooltips with county name, population, and growth rate
    - Semi-transparent extrusion sides with distinct top-face coloring
    - SDC brand color palette

Output:
    data/exports/nd_county_3d_population_map.html

Data sources:
    - TIGER 2020 county boundaries: data/interim/geographic/tiger2020/
    - Baseline county projections: data/projections/baseline/county/
    - County summary CSV: data/projections/baseline/county/countys_summary.csv

Usage:
    python scripts/exports/build_3d_county_map.py

Metadata:
    Author: SDC / Claude Code
    Created: 2026-03-01
    SOP: SOP-002
    ADR: ADR-059 (TIGER geospatial exports)
    Data vintage: Vintage 2025 PEP, TIGER 2020
    Processing: Aggregates age/sex/race detail to total population per county-year
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from project_utils import setup_logger  # noqa: E402

logger = setup_logger(__name__, level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIGER_SHP = PROJECT_ROOT / "data/interim/geographic/tiger2020/tl_2020_us_county.shp"
PROJECTIONS_DIR = PROJECT_ROOT / "data/projections/baseline/county"
SUMMARY_CSV = PROJECTIONS_DIR / "countys_summary.csv"
OUTPUT_HTML = PROJECT_ROOT / "data/exports/nd_county_3d_population_map.html"

# ---------------------------------------------------------------------------
# SDC Brand Colors (from _report_theme.py)
# ---------------------------------------------------------------------------
NAVY = "#1F3864"
BLUE = "#0563C1"
TEAL = "#00B0F0"
GROWTH_COLOR = "#00B050"
DECLINE_COLOR = "#FF0000"

# North Dakota centroid and camera defaults
ND_CENTER_LON = -100.47
ND_CENTER_LAT = 47.45


def load_nd_counties() -> gpd.GeoDataFrame:
    """Load TIGER 2020 county boundaries filtered to North Dakota (FIPS 38)."""
    logger.info("Loading TIGER 2020 county shapefile: %s", TIGER_SHP)
    gdf = gpd.read_file(TIGER_SHP)
    nd = gdf[gdf["STATEFP"] == "38"].copy()
    logger.info("Found %d North Dakota counties", len(nd))

    # Reproject to WGS84 (EPSG:4326) for deck.gl compatibility
    nd = nd.to_crs(epsg=4326)

    # Keep only needed columns
    nd = nd[["GEOID", "NAME", "INTPTLAT", "INTPTLON", "geometry"]].copy()
    nd["INTPTLAT"] = nd["INTPTLAT"].astype(float)
    nd["INTPTLON"] = nd["INTPTLON"].astype(float)

    return nd


def load_projection_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all county projection parquets and aggregate to total population per year.

    Returns a tuple of:
        - pop_df: DataFrame with columns: fips, year, population
        - age_df: DataFrame with columns: fips, year, mean_age
    """
    logger.info("Loading county projection parquets from: %s", PROJECTIONS_DIR)

    pop_records = []
    age_records = []
    parquet_files = sorted(PROJECTIONS_DIR.glob("nd_county_38*_projection_*.parquet"))
    logger.info("Found %d county projection files", len(parquet_files))

    for pf in parquet_files:
        # Extract FIPS from filename: nd_county_38XXX_projection_...
        fips = pf.stem.split("_")[2]
        df = pd.read_parquet(pf)

        # Total population per year
        yearly_pop = df.groupby("year")["population"].sum().reset_index()
        yearly_pop["fips"] = fips
        pop_records.append(yearly_pop)

        # Mean age per year: sum(age * population) / sum(population)
        df["age_x_pop"] = df["age"] * df["population"]
        yearly_age = df.groupby("year").agg(
            total_pop=("population", "sum"),
            weighted_age=("age_x_pop", "sum"),
        ).reset_index()
        yearly_age["mean_age"] = yearly_age["weighted_age"] / yearly_age["total_pop"]
        yearly_age["fips"] = fips
        age_records.append(yearly_age[["fips", "year", "mean_age"]])

    pop_result = pd.concat(pop_records, ignore_index=True)
    age_result = pd.concat(age_records, ignore_index=True)
    logger.info(
        "Loaded population data: %d county-year records, years %d-%d",
        len(pop_result),
        pop_result["year"].min(),
        pop_result["year"].max(),
    )
    logger.info(
        "Mean age range: %.1f - %.1f across all counties and years",
        age_result["mean_age"].min(),
        age_result["mean_age"].max(),
    )
    return pop_result, age_result


def load_county_names() -> dict[str, str]:
    """Load county names from the summary CSV."""
    df = pd.read_csv(SUMMARY_CSV)
    # Strip " County" suffix for cleaner labels
    names = {}
    for _, row in df.iterrows():
        fips = str(row["fips"])
        name = str(row["name"]).removesuffix(" County")
        names[fips] = name
    return names


def build_geojson_with_data(
    counties: gpd.GeoDataFrame,
    pop_data: pd.DataFrame,
    age_data: pd.DataFrame,
    county_names: dict[str, str],
) -> dict:
    """Merge geometry + population + age data into a GeoJSON FeatureCollection.

    Each feature has properties:
        - fips: 5-digit FIPS code
        - name: county name (without " County")
        - centroid_lat, centroid_lon: label position
        - pop_YYYY: population for each year
        - age_YYYY: mean age for each year
        - growth_rate: overall growth rate 2025-2055
    """
    logger.info("Building merged GeoJSON with population data")

    # Pivot population data: one column per year
    pop_pivot = pop_data.pivot(index="fips", columns="year", values="population")
    pop_pivot.columns = [f"pop_{int(y)}" for y in pop_pivot.columns]
    pop_pivot = pop_pivot.reset_index()

    # Pivot mean age data: one column per year
    age_pivot = age_data.pivot(index="fips", columns="year", values="mean_age")
    age_pivot.columns = [f"age_{int(y)}" for y in age_pivot.columns]
    age_pivot = age_pivot.reset_index()

    features = []
    years = sorted(pop_data["year"].unique())

    for _, row in counties.iterrows():
        geoid = row["GEOID"]
        geom = row["geometry"]

        # Find matching population data
        pop_row = pop_pivot[pop_pivot["fips"] == geoid]
        if pop_row.empty:
            logger.warning("No projection data for county FIPS %s, skipping", geoid)
            continue

        pop_row = pop_row.iloc[0]

        # Find matching age data
        age_row = age_pivot[age_pivot["fips"] == geoid]
        has_age = not age_row.empty
        if has_age:
            age_row = age_row.iloc[0]

        # Calculate growth rate
        base_pop = pop_row.get("pop_2025", 0)
        final_pop = pop_row.get("pop_2055", 0)
        growth_rate = (final_pop - base_pop) / base_pop if base_pop > 0 else 0

        # Build properties
        props: dict = {
            "fips": geoid,
            "name": county_names.get(geoid, row["NAME"]),
            "centroid_lat": row["INTPTLAT"],
            "centroid_lon": row["INTPTLON"],
            "growth_rate": round(growth_rate, 4),
        }

        # Add population and mean age for each year
        for year in years:
            pop_col = f"pop_{year}"
            props[pop_col] = round(float(pop_row.get(pop_col, 0)), 1)

            age_col = f"age_{year}"
            if has_age and age_col in age_row.index:
                props[age_col] = round(float(age_row[age_col]), 2)

        # Convert geometry to GeoJSON
        geom_json = json.loads(gpd.GeoSeries([geom]).to_json())
        geom_feature = geom_json["features"][0]["geometry"]

        features.append({"type": "Feature", "properties": props, "geometry": geom_feature})

    geojson = {"type": "FeatureCollection", "features": features}
    logger.info("Built GeoJSON with %d features", len(features))
    return geojson


def generate_html(geojson_data: dict, years: list[int]) -> str:
    """Generate the standalone HTML file with deck.gl 3D visualization.

    Uses deck.gl (via CDN) for 3D polygon extrusion and MapLibre GL JS
    for the basemap. All data is embedded inline for offline capability.
    """
    logger.info("Generating standalone HTML visualization")

    geojson_str = json.dumps(geojson_data)
    years_json = json.dumps(years)

    # Calculate max population across all years for scale reference
    max_pop = 0
    for feature in geojson_data["features"]:
        for year in years:
            pop = feature["properties"].get(f"pop_{year}", 0)
            if pop > max_pop:
                max_pop = pop

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>North Dakota County Population Projections - 3D Map</title>
<script src="https://unpkg.com/deck.gl@9.1.4/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    overflow: hidden;
  }}
  #map-container {{
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: #0f0f1e;
  }}

  /* Control Panel */
  #controls {{
    position: absolute;
    top: 16px;
    left: 16px;
    background: rgba(31, 56, 100, 0.92);
    border-radius: 10px;
    padding: 18px 22px;
    z-index: 10;
    min-width: 320px;
    max-width: 380px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid rgba(0,176,240,0.3);
  }}
  #controls h1 {{
    font-size: 16px;
    font-weight: 700;
    color: #00B0F0;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
  }}
  #controls .subtitle {{
    font-size: 11px;
    color: #a0b4cc;
    margin-bottom: 14px;
  }}

  .control-group {{
    margin-bottom: 12px;
  }}
  .control-group label {{
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: #80b0d0;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  #year-slider {{
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    border-radius: 3px;
    background: #2a3f5f;
    outline: none;
    cursor: pointer;
  }}
  #year-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: #00B0F0;
    cursor: pointer;
    box-shadow: 0 0 6px rgba(0,176,240,0.5);
  }}
  #year-slider::-moz-range-thumb {{
    width: 18px; height: 18px;
    border-radius: 50%;
    background: #00B0F0;
    cursor: pointer;
    border: none;
  }}

  .year-display {{
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin: 4px 0 2px;
    font-variant-numeric: tabular-nums;
  }}
  .pop-display {{
    text-align: center;
    font-size: 13px;
    color: #a0b4cc;
    margin-bottom: 8px;
  }}

  .btn-row {{
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .btn {{
    flex: 1;
    padding: 6px 10px;
    border: 1px solid rgba(0,176,240,0.4);
    border-radius: 6px;
    background: rgba(0,176,240,0.12);
    color: #00B0F0;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
  }}
  .btn:hover {{
    background: rgba(0,176,240,0.25);
    border-color: #00B0F0;
  }}
  .btn.active {{
    background: rgba(0,176,240,0.35);
    border-color: #00B0F0;
    color: #fff;
  }}
  .btn-play {{
    flex: 0 0 48px;
  }}
  .btn-step {{
    flex: 0 0 36px;
    font-size: 10px;
  }}

  /* Toggle switches */
  .toggle-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .toggle-row span {{
    font-size: 12px;
    color: #c0d0e0;
  }}
  .toggle {{
    position: relative;
    width: 36px; height: 20px;
    background: #2a3f5f;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s;
  }}
  .toggle.on {{
    background: #0563C1;
  }}
  .toggle::after {{
    content: '';
    position: absolute;
    top: 2px; left: 2px;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #e0e0e0;
    transition: transform 0.2s;
  }}
  .toggle.on::after {{
    transform: translateX(16px);
  }}

  /* County list panel */
  #county-panel {{
    position: absolute;
    top: 16px;
    right: 16px;
    background: rgba(31, 56, 100, 0.92);
    border-radius: 10px;
    padding: 14px 16px;
    z-index: 10;
    width: 220px;
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid rgba(0,176,240,0.3);
  }}
  #county-panel h2 {{
    font-size: 13px;
    font-weight: 700;
    color: #00B0F0;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  #county-panel .panel-actions {{
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
  }}
  #county-panel .panel-actions .btn {{
    font-size: 10px;
    padding: 3px 8px;
  }}
  .county-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 0;
    cursor: pointer;
    font-size: 12px;
    color: #c0d0e0;
    transition: color 0.15s;
  }}
  .county-item:hover {{
    color: #ffffff;
  }}
  .county-item input[type="checkbox"] {{
    accent-color: #00B0F0;
    cursor: pointer;
  }}
  .county-item .pop-badge {{
    margin-left: auto;
    font-size: 10px;
    color: #708090;
    font-variant-numeric: tabular-nums;
  }}

  /* Scrollbar styling for county panel */
  #county-panel::-webkit-scrollbar {{
    width: 6px;
  }}
  #county-panel::-webkit-scrollbar-track {{
    background: transparent;
  }}
  #county-panel::-webkit-scrollbar-thumb {{
    background: rgba(0,176,240,0.3);
    border-radius: 3px;
  }}

  /* Tooltip */
  #tooltip {{
    position: absolute;
    z-index: 20;
    pointer-events: none;
    background: rgba(31, 56, 100, 0.95);
    border: 1px solid rgba(0,176,240,0.5);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #e0e0e0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    display: none;
    max-width: 260px;
  }}
  #tooltip .tt-name {{
    font-size: 14px;
    font-weight: 700;
    color: #00B0F0;
    margin-bottom: 4px;
  }}
  #tooltip .tt-row {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin: 2px 0;
  }}
  #tooltip .tt-label {{
    color: #80b0d0;
  }}
  #tooltip .tt-value {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .tt-growth-positive {{ color: #00B050; }}
  .tt-growth-negative {{ color: #FF4444; }}

  /* Legend */
  #legend {{
    position: absolute;
    bottom: 16px;
    left: 16px;
    background: rgba(31, 56, 100, 0.88);
    border-radius: 8px;
    padding: 10px 14px;
    z-index: 10;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    border: 1px solid rgba(0,176,240,0.2);
  }}
  #legend .legend-title {{
    font-size: 11px;
    font-weight: 600;
    color: #80b0d0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }}
  .legend-bar {{
    width: 200px;
    height: 12px;
    border-radius: 3px;
    margin-bottom: 4px;
  }}
  .legend-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #a0b4cc;
  }}

  /* Collapse toggle for county panel */
  .collapse-btn {{
    background: none;
    border: none;
    color: #80b0d0;
    cursor: pointer;
    font-size: 14px;
    float: right;
    padding: 0 4px;
  }}
  .collapse-btn:hover {{ color: #00B0F0; }}

  /* Navigation help */
  #nav-help-btn {{
    position: absolute;
    bottom: 16px;
    right: 16px;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(31, 56, 100, 0.88);
    border: 1px solid rgba(0,176,240,0.4);
    color: #00B0F0;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    z-index: 15;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: all 0.2s;
  }}
  #nav-help-btn:hover {{
    background: rgba(0,176,240,0.25);
    border-color: #00B0F0;
  }}
  #nav-help {{
    position: absolute;
    bottom: 56px;
    right: 16px;
    background: rgba(31, 56, 100, 0.94);
    border-radius: 10px;
    padding: 14px 18px;
    z-index: 15;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid rgba(0,176,240,0.3);
    display: none;
    min-width: 220px;
  }}
  #nav-help h3 {{
    font-size: 12px;
    font-weight: 700;
    color: #00B0F0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 10px;
  }}
  .nav-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #c0d0e0;
  }}
  .nav-key {{
    flex: 0 0 auto;
    min-width: 90px;
    font-weight: 600;
    color: #80b0d0;
    font-size: 11px;
  }}
  .nav-sep {{
    border-top: 1px solid rgba(0,176,240,0.15);
    margin: 6px 0;
  }}
</style>
</head>
<body>

<div id="map-container"></div>

<!-- Tooltip -->
<div id="tooltip"></div>

<!-- Controls -->
<div id="controls">
  <h1>ND Population Projections</h1>
  <div class="subtitle">3D Extruded County Map &middot; 2025&ndash;2055 Baseline</div>

  <div class="control-group">
    <div class="year-display" id="year-display">2025</div>
    <div class="pop-display" id="pop-display">State Total: --</div>
    <input type="range" id="year-slider" min="0" max="30" value="0" step="1">
  </div>

  <div class="btn-row">
    <button class="btn btn-play" id="btn-play" title="Play/Pause (Space)">&#9654;</button>
    <button class="btn btn-step" id="btn-back" title="Previous year (&#8592;)">&#9664;</button>
    <button class="btn btn-step" id="btn-fwd" title="Next year (&#8594;)">&#9654;</button>
    <button class="btn" id="btn-speed" title="Animation speed">1x</button>
    <button class="btn" id="btn-reset" title="Reset view (R)">Reset View</button>
  </div>

  <div class="control-group">
    <div class="toggle-row">
      <div class="toggle" id="toggle-log" title="Toggle logarithmic scale"></div>
      <span>Log scale</span>
    </div>
    <div class="toggle-row">
      <div class="toggle on" id="toggle-labels" title="Toggle county labels"></div>
      <span>County labels</span>
    </div>
  </div>

  <div class="control-group">
    <div style="font-size: 11px; color: #80b0d0; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 600;">Color by</div>
    <div class="btn-row">
      <button class="btn active" id="btn-color-growth">Growth</button>
      <button class="btn" id="btn-color-age">Mean Age</button>
    </div>
  </div>
</div>

<!-- County Panel -->
<div id="county-panel">
  <button class="collapse-btn" id="panel-collapse" title="Collapse">&times;</button>
  <h2>Counties</h2>
  <div class="panel-actions">
    <button class="btn" id="btn-select-all">All</button>
    <button class="btn" id="btn-select-none">None</button>
    <button class="btn" id="btn-select-top10">Top 10</button>
  </div>
  <div id="county-list"></div>
</div>

<!-- Navigation help -->
<button id="nav-help-btn" title="Navigation controls">?</button>
<div id="nav-help">
  <h3>Map Navigation</h3>
  <div class="nav-row"><span class="nav-key">Left drag</span> Rotate</div>
  <div class="nav-row"><span class="nav-key">Right drag</span> Pan</div>
  <div class="nav-row"><span class="nav-key">Scroll wheel</span> Zoom</div>
  <div class="nav-row"><span class="nav-key">Ctrl + drag</span> Tilt (pitch)</div>
  <div class="nav-sep"></div>
  <div class="nav-row"><span class="nav-key">Click county</span> Toggle selection</div>
  <div class="nav-row"><span class="nav-key">Space</span> Play / Pause</div>
  <div class="nav-row"><span class="nav-key">&larr; / &rarr;</span> Step year</div>
  <div class="nav-row"><span class="nav-key">R</span> Reset view</div>
</div>

<!-- Legend -->
<div id="legend">
  <div class="legend-title" id="legend-title">Growth from 2025</div>
  <div class="legend-bar" style="background: linear-gradient(to right, #C81E1E, #DC6458, #96969B, #329646, #00BE3C);"></div>
  <div class="legend-labels">
    <span id="legend-min">-24%</span>
    <span id="legend-mid">0%</span>
    <span id="legend-max">+53%</span>
  </div>
</div>

<script>
// =========================================================================
// DATA
// =========================================================================
const GEOJSON_DATA = {geojson_str};
const YEARS = {years_json};
const MAX_POP = {max_pop};

// SDC brand colors
const NAVY = [31, 56, 100];
const BLUE = [5, 99, 193];
const TEAL = [0, 176, 240];

// =========================================================================
// STATE
// =========================================================================
let currentYearIndex = 0;
let isPlaying = false;
let animSpeed = 1; // 1x, 2x, 4x
let animInterval = null;
let useLogScale = false;
let showLabels = true;
let selectedCounties = new Set();
let panelCollapsed = false;
let colorMode = 'growth'; // 'growth' or 'age'

// Initialize: all counties selected
GEOJSON_DATA.features.forEach(f => selectedCounties.add(f.properties.fips));

// Sort features by base population (descending) for county list
const sortedFeatures = [...GEOJSON_DATA.features].sort(
  (a, b) => b.properties.pop_2025 - a.properties.pop_2025
);

// =========================================================================
// HELPERS
// =========================================================================
function getCurrentYear() {{
  return YEARS[currentYearIndex];
}}

function getPopulation(props, year) {{
  return props['pop_' + year] || 0;
}}

function formatPop(n) {{
  return Math.round(n).toLocaleString('en-US');
}}

function formatPercent(r) {{
  const pct = (r * 100).toFixed(1);
  return (r >= 0 ? '+' : '') + pct + '%';
}}

function getMeanAge(props, year) {{
  return props['age_' + year] || 0;
}}

function ageToColor(age) {{
  // Map mean age to color: green (young, ~30) -> yellow (~40) -> red (old, ~55+)
  // Clamp to [28, 55] range
  const clamped = Math.max(28, Math.min(55, age));
  const t = (clamped - 28) / 27; // normalize to [0, 1]

  let r, g, b;
  if (t < 0.35) {{
    // Green to yellow-green (young counties, ~28-37)
    const s = t / 0.35;
    r = Math.round(40 + 160 * s);
    g = Math.round(180 - 10 * s);
    b = Math.round(60 - 20 * s);
  }} else if (t < 0.55) {{
    // Yellow-green to amber (~37-43)
    const s = (t - 0.35) / 0.20;
    r = Math.round(200 + 30 * s);
    g = Math.round(170 - 30 * s);
    b = Math.round(40 - 10 * s);
  }} else {{
    // Amber to deep red (older counties, ~43-55)
    const s = (t - 0.55) / 0.45;
    r = Math.round(230 - 30 * s);
    g = Math.round(140 - 100 * s);
    b = Math.round(30 + 10 * s);
  }}
  return [r, g, b];
}}

function getMaxSelectedPop() {{
  const year = getCurrentYear();
  let maxPop = 0;
  GEOJSON_DATA.features.forEach(f => {{
    if (selectedCounties.has(f.properties.fips)) {{
      const pop = getPopulation(f.properties, year);
      if (pop > maxPop) maxPop = pop;
    }}
  }});
  return maxPop || 1;
}}

function getElevation(props) {{
  const year = getCurrentYear();
  const pop = getPopulation(props, year);
  if (!selectedCounties.has(props.fips)) return 0;

  // Dynamic max based on selected counties
  const maxSelected = getMaxSelectedPop();

  // Target max elevation in meters (visual scale)
  const MAX_ELEVATION = 120000;

  if (useLogScale) {{
    const logPop = pop > 0 ? Math.log10(pop) : 0;
    const logMax = maxSelected > 0 ? Math.log10(maxSelected) : 1;
    return (logPop / logMax) * MAX_ELEVATION;
  }} else {{
    return (pop / maxSelected) * MAX_ELEVATION;
  }}
}}

function growthToColor(rate) {{
  // Map growth rate to a color gradient: red (decline) -> narrow grey (neutral) -> green (growth)
  // Clamp rate to [-0.3, 0.6] range
  const clamped = Math.max(-0.3, Math.min(0.6, rate));
  const t = (clamped + 0.3) / 0.9; // normalize to [0, 1]

  // Narrow grey band (0.30–0.37), saturated endpoints for max contrast
  // Deep red [200,30,30] -> grey [150,150,155] -> vivid green [0,170,60]
  let r, g, b;
  if (t < 0.30) {{
    // Deep red to muted red-orange
    const s = t / 0.30;
    r = Math.round(200 + 20 * s);
    g = Math.round(30 + 70 * s);
    b = Math.round(30 + 60 * s);
  }} else if (t < 0.37) {{
    // Muted red-orange to grey (narrow neutral band)
    const s = (t - 0.30) / 0.07;
    r = Math.round(220 - 70 * s);
    g = Math.round(100 + 50 * s);
    b = Math.round(90 + 65 * s);
  }} else if (t < 0.50) {{
    // Grey to muted green
    const s = (t - 0.37) / 0.13;
    r = Math.round(150 - 100 * s);
    g = Math.round(150 + 10 * s);
    b = Math.round(155 - 85 * s);
  }} else {{
    // Muted green to vivid green
    const s = (t - 0.50) / 0.50;
    r = Math.round(50 - 50 * s);
    g = Math.round(160 + 30 * s);
    b = Math.round(70 - 20 * s);
  }}
  return [r, g, b];
}}

function getColorForProps(props) {{
  // Returns base [r, g, b] based on current color mode
  const year = getCurrentYear();
  if (colorMode === 'age') {{
    return ageToColor(getMeanAge(props, year));
  }}
  // Default: growth mode
  const basePop = getPopulation(props, 2025);
  const curPop = getPopulation(props, year);
  const yearGrowth = basePop > 0 ? (curPop - basePop) / basePop : 0;
  return growthToColor(yearGrowth);
}}

function getColor(props) {{
  if (!selectedCounties.has(props.fips)) {{
    return [60, 60, 80, 40]; // dimmed unselected
  }}
  const color = getColorForProps(props);
  return [...color, 200]; // semi-transparent
}}

function getSideColor(props) {{
  if (!selectedCounties.has(props.fips)) {{
    return [40, 40, 60, 20];
  }}
  const color = getColorForProps(props);
  // Darker side
  return [
    Math.round(color[0] * 0.6),
    Math.round(color[1] * 0.6),
    Math.round(color[2] * 0.6),
    180
  ];
}}

// =========================================================================
// DECK.GL SETUP
// =========================================================================
const INITIAL_VIEW_STATE = {{
  longitude: {ND_CENTER_LON},
  latitude: {ND_CENTER_LAT},
  zoom: 6.0,
  pitch: 50,
  bearing: -15,
  minZoom: 4,
  maxZoom: 12,
}};

function buildLayers() {{
  const layers = [];

  // GeoJSON polygon layer with extrusion
  layers.push(
    new deck.GeoJsonLayer({{
      id: 'counties-3d',
      data: GEOJSON_DATA,
      extruded: true,
      filled: true,
      wireframe: true,
      getElevation: f => getElevation(f.properties),
      getFillColor: f => getColor(f.properties),
      getLineColor: [15, 15, 30, 160],
      getLineWidth: 0,
      elevationScale: 1,
      material: {{
        ambient: 0.65,
        diffuse: 0.8,
        shininess: 8,
        specularColor: [120, 140, 160],
      }},
      pickable: true,
      autoHighlight: true,
      highlightColor: [0, 176, 240, 100],
      onHover: onHover,
      onClick: onCountyClick,
      updateTriggers: {{
        getElevation: [currentYearIndex, useLogScale, Array.from(selectedCounties).join(',')],
        getFillColor: [currentYearIndex, colorMode, Array.from(selectedCounties).join(',')],
      }},
      transitions: {{
        getElevation: {{ duration: isPlaying ? 0 : 300, easing: t => t * (2 - t) }},
      }},
    }})
  );

  // Flat boundary outlines (separate layer avoids z-fighting with extruded faces)
  layers.push(
    new deck.GeoJsonLayer({{
      id: 'county-outlines',
      data: GEOJSON_DATA,
      extruded: false,
      filled: false,
      stroked: true,
      getLineColor: [15, 15, 30, 180],
      getLineWidth: 120,
      lineWidthMinPixels: 1,
      lineWidthMaxPixels: 3,
    }})
  );

  // Text labels on top of each county
  if (showLabels) {{
    const labelData = GEOJSON_DATA.features
      .filter(f => selectedCounties.has(f.properties.fips))
      .map(f => ({{
        position: [f.properties.centroid_lon, f.properties.centroid_lat, getElevation(f.properties) + 3000],
        text: f.properties.name,
        size: 11,
      }}));

    layers.push(
      new deck.TextLayer({{
        id: 'county-labels',
        data: labelData,
        getPosition: d => d.position,
        getText: d => d.text,
        getSize: d => d.size,
        getColor: [255, 255, 255, 240],
        getAngle: 0,
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'bottom',
        fontFamily: "'Segoe UI', Roboto, Arial, sans-serif",
        fontWeight: 700,
        outlineWidth: 4,
        outlineColor: [10, 10, 20, 230],
        billboard: true,
        sizeScale: 1,
        sizeMinPixels: 8,
        sizeMaxPixels: 18,
        updateTriggers: {{
          getPosition: [currentYearIndex, useLogScale, Array.from(selectedCounties).join(',')],
        }},
      }})
    );
  }}

  return layers;
}}

// Tooltip handling
function onHover(info) {{
  const tooltip = document.getElementById('tooltip');
  if (!info.object) {{
    tooltip.style.display = 'none';
    return;
  }}

  const props = info.object.properties;
  const year = getCurrentYear();
  const pop = getPopulation(props, year);
  const basePop = getPopulation(props, 2025);
  const meanAge = getMeanAge(props, year);
  const yearGrowth = basePop > 0 ? (pop - basePop) / basePop : 0;
  const growthClass = yearGrowth >= 0 ? 'tt-growth-positive' : 'tt-growth-negative';
  const overallClass = props.growth_rate >= 0 ? 'tt-growth-positive' : 'tt-growth-negative';

  tooltip.innerHTML = `
    <div class="tt-name">${{props.name}} County</div>
    <div class="tt-row">
      <span class="tt-label">FIPS:</span>
      <span class="tt-value">${{props.fips}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-label">${{year}} Pop:</span>
      <span class="tt-value">${{formatPop(pop)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-label">Mean Age:</span>
      <span class="tt-value">${{meanAge.toFixed(1)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-label">2025 Base:</span>
      <span class="tt-value">${{formatPop(basePop)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-label">Change from 2025:</span>
      <span class="tt-value ${{growthClass}}">${{formatPercent(yearGrowth)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-label">2025-2055 Growth:</span>
      <span class="tt-value ${{overallClass}}">${{formatPercent(props.growth_rate)}}</span>
    </div>
  `;
  tooltip.style.display = 'block';
  tooltip.style.left = (info.x + 12) + 'px';
  tooltip.style.top = (info.y + 12) + 'px';

  // Keep tooltip within viewport
  const rect = tooltip.getBoundingClientRect();
  if (rect.right > window.innerWidth) {{
    tooltip.style.left = (info.x - rect.width - 12) + 'px';
  }}
  if (rect.bottom > window.innerHeight) {{
    tooltip.style.top = (info.y - rect.height - 12) + 'px';
  }}
}}

function onCountyClick(info) {{
  if (!info.object) return;
  const fips = info.object.properties.fips;
  const cb = document.getElementById('cb-' + fips);
  if (cb) {{
    cb.checked = !cb.checked;
    toggleCounty(fips, cb.checked);
  }}
}}

// =========================================================================
// DECK INSTANCE
// =========================================================================
const deckgl = new deck.DeckGL({{
  container: 'map-container',
  initialViewState: INITIAL_VIEW_STATE,
  controller: true,
  layers: buildLayers(),
  getTooltip: null,
}});

function updateDeck() {{
  deckgl.setProps({{ layers: buildLayers() }});
  updateDisplays();
}}

// =========================================================================
// DISPLAY UPDATES
// =========================================================================
function updateDisplays() {{
  const year = getCurrentYear();
  document.getElementById('year-display').textContent = year;
  document.getElementById('year-slider').value = currentYearIndex;

  // State total
  let stateTotal = 0;
  GEOJSON_DATA.features.forEach(f => {{
    stateTotal += getPopulation(f.properties, year);
  }});
  document.getElementById('pop-display').textContent = 'State Total: ' + formatPop(stateTotal);

  // Update county list badges
  sortedFeatures.forEach(f => {{
    const badge = document.getElementById('badge-' + f.properties.fips);
    if (badge) {{
      badge.textContent = formatPop(getPopulation(f.properties, year));
    }}
  }});

  // Update legend based on color mode
  const legendBar = document.querySelector('.legend-bar');
  if (colorMode === 'age') {{
    // Mean age legend
    let minAge = 99, maxAge = 0;
    GEOJSON_DATA.features.forEach(f => {{
      const a = getMeanAge(f.properties, year);
      if (a < minAge) minAge = a;
      if (a > maxAge) maxAge = a;
    }});
    document.getElementById('legend-min').textContent = minAge.toFixed(1);
    document.getElementById('legend-mid').textContent = ((minAge + maxAge) / 2).toFixed(0);
    document.getElementById('legend-max').textContent = maxAge.toFixed(1);
    document.getElementById('legend-title').textContent = 'Mean Age \u2013 ' + year;
    legendBar.style.background =
      'linear-gradient(to right, #28B43C, #C8AA28, #E08C1E, #C82828)';
  }} else {{
    // Growth legend
    let minGr = 0, maxGr = 0;
    GEOJSON_DATA.features.forEach(f => {{
      const basePop = getPopulation(f.properties, 2025);
      const curPop = getPopulation(f.properties, year);
      const gr = basePop > 0 ? (curPop - basePop) / basePop : 0;
      if (gr < minGr) minGr = gr;
      if (gr > maxGr) maxGr = gr;
    }});
    document.getElementById('legend-min').textContent = formatPercent(minGr);
    document.getElementById('legend-mid').textContent = '0%';
    document.getElementById('legend-max').textContent = formatPercent(maxGr);
    document.getElementById('legend-title').textContent =
      year === 2025 ? 'Growth from 2025' : 'Growth 2025\u2013' + year;
    legendBar.style.background =
      'linear-gradient(to right, #C81E1E, #DC6458, #96969B, #329646, #00BE3C)';
  }}
}}

// =========================================================================
// CONTROLS
// =========================================================================

// Year slider
document.getElementById('year-slider').addEventListener('input', function() {{
  currentYearIndex = parseInt(this.value);
  updateDeck();
}});

// Play/Pause
function togglePlay() {{
  isPlaying = !isPlaying;
  const btn = document.getElementById('btn-play');
  btn.textContent = isPlaying ? '\u23F8' : '\u25B6';
  btn.classList.toggle('active', isPlaying);

  if (isPlaying) {{
    startAnimation();
  }} else {{
    stopAnimation();
  }}
}}

let lastAnimTime = 0;
function startAnimation() {{
  stopAnimation();
  lastAnimTime = 0;
  animInterval = true;  // flag that we're animating
  requestAnimationFrame(animFrame);
}}

function animFrame(timestamp) {{
  if (!animInterval) return;
  if (!lastAnimTime) lastAnimTime = timestamp;
  const elapsed = timestamp - lastAnimTime;
  const interval = 800 / animSpeed;
  if (elapsed >= interval) {{
    lastAnimTime = timestamp;
    currentYearIndex++;
    if (currentYearIndex >= YEARS.length) {{
      currentYearIndex = 0;
    }}
    updateDeck();
  }}
  requestAnimationFrame(animFrame);
}}

function stopAnimation() {{
  animInterval = null;
}}

document.getElementById('btn-play').addEventListener('click', togglePlay);

// Step back / forward
document.getElementById('btn-back').addEventListener('click', () => {{
  if (currentYearIndex > 0) {{
    currentYearIndex--;
    updateDeck();
  }}
}});
document.getElementById('btn-fwd').addEventListener('click', () => {{
  if (currentYearIndex < YEARS.length - 1) {{
    currentYearIndex++;
    updateDeck();
  }}
}});

// Speed toggle
document.getElementById('btn-speed').addEventListener('click', function() {{
  const speeds = [1, 2, 4];
  const idx = speeds.indexOf(animSpeed);
  animSpeed = speeds[(idx + 1) % speeds.length];
  this.textContent = animSpeed + 'x';
  if (isPlaying) startAnimation();
}});

// Reset view
document.getElementById('btn-reset').addEventListener('click', function() {{
  deckgl.setProps({{
    initialViewState: {{
      ...INITIAL_VIEW_STATE,
      transitionDuration: 800,
      transitionInterpolator: new deck.FlyToInterpolator(),
    }},
  }});
}});

// Navigation help toggle
document.getElementById('nav-help-btn').addEventListener('click', function() {{
  const panel = document.getElementById('nav-help');
  panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
}});

// Color mode toggle
document.getElementById('btn-color-growth').addEventListener('click', function() {{
  colorMode = 'growth';
  this.classList.add('active');
  document.getElementById('btn-color-age').classList.remove('active');
  updateDeck();
}});
document.getElementById('btn-color-age').addEventListener('click', function() {{
  colorMode = 'age';
  this.classList.add('active');
  document.getElementById('btn-color-growth').classList.remove('active');
  updateDeck();
}});

// Log scale toggle
document.getElementById('toggle-log').addEventListener('click', function() {{
  useLogScale = !useLogScale;
  this.classList.toggle('on', useLogScale);
  updateDeck();
}});

// Labels toggle
document.getElementById('toggle-labels').addEventListener('click', function() {{
  showLabels = !showLabels;
  this.classList.toggle('on', showLabels);
  updateDeck();
}});

// =========================================================================
// COUNTY PANEL
// =========================================================================
function buildCountyList() {{
  const container = document.getElementById('county-list');
  container.innerHTML = '';

  sortedFeatures.forEach(f => {{
    const props = f.properties;
    const div = document.createElement('div');
    div.className = 'county-item';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = 'cb-' + props.fips;
    cb.checked = selectedCounties.has(props.fips);
    cb.addEventListener('change', () => toggleCounty(props.fips, cb.checked));

    const label = document.createElement('span');
    label.textContent = props.name;

    const badge = document.createElement('span');
    badge.className = 'pop-badge';
    badge.id = 'badge-' + props.fips;
    badge.textContent = formatPop(getPopulation(props, getCurrentYear()));

    div.appendChild(cb);
    div.appendChild(label);
    div.appendChild(badge);
    container.appendChild(div);
  }});
}}

function toggleCounty(fips, checked) {{
  if (checked) {{
    selectedCounties.add(fips);
  }} else {{
    selectedCounties.delete(fips);
  }}
  updateDeck();
}}

// Select All / None / Top 10
document.getElementById('btn-select-all').addEventListener('click', () => {{
  GEOJSON_DATA.features.forEach(f => selectedCounties.add(f.properties.fips));
  document.querySelectorAll('#county-list input').forEach(cb => cb.checked = true);
  updateDeck();
}});

document.getElementById('btn-select-none').addEventListener('click', () => {{
  selectedCounties.clear();
  document.querySelectorAll('#county-list input').forEach(cb => cb.checked = false);
  updateDeck();
}});

document.getElementById('btn-select-top10').addEventListener('click', () => {{
  selectedCounties.clear();
  sortedFeatures.slice(0, 10).forEach(f => {{
    selectedCounties.add(f.properties.fips);
  }});
  document.querySelectorAll('#county-list input').forEach(cb => {{
    const fips = cb.id.replace('cb-', '');
    cb.checked = selectedCounties.has(fips);
  }});
  updateDeck();
}});

// Collapse panel
document.getElementById('panel-collapse').addEventListener('click', function() {{
  panelCollapsed = !panelCollapsed;
  const panel = document.getElementById('county-panel');
  if (panelCollapsed) {{
    panel.style.width = '40px';
    panel.style.overflow = 'hidden';
    this.textContent = '\u2630';
    document.getElementById('county-list').style.display = 'none';
    document.querySelector('#county-panel h2').style.display = 'none';
    document.querySelector('#county-panel .panel-actions').style.display = 'none';
  }} else {{
    panel.style.width = '220px';
    panel.style.overflow = 'auto';
    this.textContent = '\u00D7';
    document.getElementById('county-list').style.display = '';
    document.querySelector('#county-panel h2').style.display = '';
    document.querySelector('#county-panel .panel-actions').style.display = '';
  }}
}});

// =========================================================================
// KEYBOARD SHORTCUTS
// =========================================================================
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  switch(e.key) {{
    case ' ':
    case 'p':
      e.preventDefault();
      togglePlay();
      break;
    case 'ArrowRight':
      e.preventDefault();
      if (currentYearIndex < YEARS.length - 1) {{
        currentYearIndex++;
        updateDeck();
      }}
      break;
    case 'ArrowLeft':
      e.preventDefault();
      if (currentYearIndex > 0) {{
        currentYearIndex--;
        updateDeck();
      }}
      break;
    case 'l':
      document.getElementById('toggle-log').click();
      break;
    case 'r':
      document.getElementById('btn-reset').click();
      break;
  }}
}});

// =========================================================================
// INITIALIZE
// =========================================================================
buildCountyList();
updateDisplays();

// Compute legend min/max
let minGrowth = Infinity, maxGrowth = -Infinity;
GEOJSON_DATA.features.forEach(f => {{
  const gr = f.properties.growth_rate;
  if (gr < minGrowth) minGrowth = gr;
  if (gr > maxGrowth) maxGrowth = gr;
}});
document.getElementById('legend-min').textContent = formatPercent(minGrowth);
document.getElementById('legend-max').textContent = formatPercent(maxGrowth);

</script>
</body>
</html>"""

    return html


def main() -> None:
    """Main entry point: load data, build GeoJSON, generate HTML."""
    logger.info("=" * 60)
    logger.info("Building 3D County Population Map")
    logger.info("=" * 60)

    # Load data
    counties = load_nd_counties()
    pop_data, age_data = load_projection_data()
    county_names = load_county_names()

    # Build merged GeoJSON
    geojson = build_geojson_with_data(counties, pop_data, age_data, county_names)

    # Get year list
    years = sorted(pop_data["year"].unique().tolist())

    # Generate HTML
    html_content = generate_html(geojson, years)

    # Write output
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    logger.info("Output written to: %s", OUTPUT_HTML)
    logger.info("File size: %.1f KB", OUTPUT_HTML.stat().st_size / 1024)
    logger.info("Done!")


if __name__ == "__main__":
    main()
