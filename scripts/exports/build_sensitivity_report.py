#!/usr/bin/env python3
"""
Build interactive HTML report visualizing sensitivity analysis results for ND population projections.

Creates a single self-contained HTML file with Plotly charts and six tabbed sections:
Executive Summary, Parameter Sweeps, County Impact, Decomposition Detail,
Methodology Notes, and Raw Data.

Output: data/exports/nd_sensitivity_analysis_{datestamp}.html

Usage:
    python scripts/exports/build_sensitivity_report.py
    python scripts/exports/build_sensitivity_report.py --output-dir data/exports/

Data Sources:
    Per-sweep files (one per sensitivity analysis):
        - data/exports/sensitivity/sweep_migration_weighting.csv
        - data/exports/sensitivity/sweep_convergence.csv
        - data/exports/sensitivity/sweep_mortality_improvement.csv
        - data/exports/sensitivity/sweep_bakken_dampening.csv
        - data/exports/sensitivity/sweep_gq_correction.csv
        - data/exports/sensitivity/sweep_rate_caps.csv
        - data/exports/sensitivity/sweep_college_smoothing.csv
        - data/exports/sensitivity/sweep_fertility.csv
        - data/exports/sensitivity/sweep_age_resolution.csv (optional)
      Format: variant, year, fips, population

    Summary files:
        - data/exports/sensitivity/decomposition_waterfall.csv
          columns: step, feature_name, pop_2050, delta_from_previous, cumulative_delta
        - data/exports/sensitivity/tornado_summary.csv
          columns: parameter, parameter_label, min_variant, max_variant,
                   min_pop_2050, max_pop_2050, baseline_pop_2050, range
        - data/exports/sensitivity/county_sensitivity_summary.csv
          columns: county_fips, county_name, pop_2025, baseline_2050, min_2050,
                   max_2050, range_abs, range_pct, most_sensitive_parameter

    Reference projections:
        - data/raw/nd_sdc_2024_projections/state_projections.csv
          columns: year, total_population
        - data/exports/baseline/summaries/state_total_population_by_year.csv
          columns: fips, 2025, 2026, ..., 2055
        - data/exports/sdc_method_new_data/state_population_by_year.csv
          columns: year, total_population
        - data/raw/population/nd_county_population.csv (FIPS mapping)

Methodology:
    Each sensitivity analysis holds all other parameters at a baseline setting and
    varies one parameter through a range of plausible values. The decomposition
    waterfall shows cumulative feature contributions from the SDC base to our model.

Key ADRs:
    ADR-036: Migration averaging (BEBR multiperiod)
    ADR-045: Reservation counties
    ADR-051: Oil county dampening (rejected)
    ADR-052: Ward County floor
    ADR-054: State-county aggregation
    ADR-055: Group quarters separation

Processing Steps:
    1. Load all sweep CSVs and summary CSVs from data/exports/sensitivity/
    2. Load reference projections (SDC 2024, Our Baseline, SDC Method + New Data)
    3. Build Tab 1: Executive Summary (waterfall, tornado, key findings)
    4. Build Tab 2: Parameter Sweeps (line charts per sweep)
    5. Build Tab 3: County Impact (heatmap, fan charts, sensitivity table)
    6. Build Tab 4: Decomposition Detail (feature contribution charts)
    7. Build Tab 5: Methodology Notes (static content)
    8. Build Tab 6: Raw Data (formatted tables)
    9. Assemble into self-contained HTML with embedded CSS/JS

SOP-002 Metadata:
    Author: Claude Code (automated)
    Created: 2026-03-02
    Input files: See Data Sources above
    Output files: data/exports/nd_sensitivity_analysis_{datestamp}.html
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
    RED_ACCENT,
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

# SDC orange for reference lines
SDC_COLOR = "#FF8C00"

# Colors for positive/negative deltas
POSITIVE_COLOR = "#00B050"
NEGATIVE_COLOR = "#FF0000"
TOTAL_BAR_COLOR = BLUE

# Sweep display configuration: (file_stem, display_title, expected_variants)
SWEEP_CONFIG: list[tuple[str, str, int | None]] = [
    ("sweep_migration_weighting", "Migration Window Weighting", 7),
    ("sweep_convergence", "Convergence Schedule", 4),
    ("sweep_mortality_improvement", "Mortality Improvement", 5),
    ("sweep_bakken_dampening", "Bakken Dampening", 6),
    ("sweep_gq_correction", "GQ Correction", 2),
    ("sweep_rate_caps", "Rate Caps", 4),
    ("sweep_college_smoothing", "College-Age Smoothing", 5),
    ("sweep_fertility", "Fertility Adjustment", 7),
    ("sweep_age_resolution", "Age Resolution", 2),
]

# Featured counties for fan charts (FIPS -> name)
FEATURED_COUNTIES = {
    "38017": "Cass",
    "38015": "Burleigh",
    "38035": "Grand Forks",
    "38101": "Ward",
    "38105": "Williams",
    "38053": "McKenzie",
}

# Qualitative color palette for sweep variants
SWEEP_PALETTE = [
    "#0563C1",  # Blue (baseline - prominent)
    "#00B050",  # Green
    "#ED7D31",  # Orange
    "#C00000",  # Dark Red
    "#7030A0",  # Purple
    "#00B0F0",  # Teal
    "#FFC000",  # Gold
    "#808080",  # Gray
    "#1F3864",  # Navy
    "#FF0000",  # Red
]


# ===================================================================
# Plotly template
# ===================================================================

def _register_template() -> str:
    """Register and return the sensitivity report template name."""
    template_name = "sensitivity_report"
    template = get_plotly_template()
    template.layout.colorway = SWEEP_PALETTE
    pio.templates[template_name] = template
    return template_name


# ===================================================================
# Formatting helpers
# ===================================================================

def _fmt_pop(n: float | int) -> str:
    """Format a population number with commas."""
    return f"{int(round(n)):,}"


def _fmt_change(n: float | int) -> str:
    """Format an absolute change with +/- sign and commas."""
    val = int(round(n))
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,}"


def _fmt_pct(p: float, decimals: int = 1) -> str:
    """Format a percentage with +/- sign."""
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.{decimals}f}%"


def _delta_class(val: float) -> str:
    """Return CSS class for positive/negative styling."""
    if val > 0:
        return "positive"
    if val < 0:
        return "negative"
    return ""


def _chart_json(fig: go.Figure) -> str:
    """Convert a Plotly figure to embedded JSON for HTML rendering."""
    return fig.to_json()


# ===================================================================
# Data Loading
# ===================================================================

def _load_sweep_file(path: Path) -> pd.DataFrame | None:
    """Load a single sweep CSV. Returns None if file missing."""
    if not path.exists():
        logger.warning("Sweep file not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        logger.exception("Error loading sweep file: %s", path)
        return None


def _load_summary_csv(path: Path) -> pd.DataFrame | None:
    """Load a summary CSV. Returns None if file missing."""
    if not path.exists():
        logger.warning("Summary file not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        logger.exception("Error loading summary: %s", path)
        return None


def _load_reference_projections() -> dict[str, pd.DataFrame | None]:
    """Load reference projection data for overlay on charts.

    Returns dict with keys: 'sdc_original', 'our_baseline', 'sdc_new_data'.
    Each value is a DataFrame with columns [year, population] or None if missing.
    """
    refs: dict[str, pd.DataFrame | None] = {}

    # SDC 2024 Original
    sdc_path = PROJECT_ROOT / "data" / "raw" / "nd_sdc_2024_projections" / "state_projections.csv"
    if sdc_path.exists():
        try:
            df = pd.read_csv(sdc_path)
            df.columns = [c.strip().lower() for c in df.columns]
            refs["sdc_original"] = df.rename(
                columns={"total_population": "population"},
            )[["year", "population"]]
        except Exception:
            logger.exception("Error loading SDC original projections")
            refs["sdc_original"] = None
    else:
        logger.warning("SDC original projections not found: %s", sdc_path)
        refs["sdc_original"] = None

    # Our Baseline
    baseline_path = (
        PROJECT_ROOT / "data" / "exports" / "baseline" / "summaries"
        / "state_total_population_by_year.csv"
    )
    if baseline_path.exists():
        try:
            df = pd.read_csv(baseline_path)
            df.columns = [c.strip().lower() for c in df.columns]
            # Wide format: fips, 2025, 2026, ..., 2055 — filter state row (fips=38)
            state_row = df[df["fips"] == 38]
            if len(state_row) == 0:
                state_row = df.iloc[[0]]
            year_cols = [c for c in state_row.columns if c.isdigit()]
            records = []
            for yc in year_cols:
                records.append({"year": int(yc), "population": float(state_row[yc].iloc[0])})
            refs["our_baseline"] = pd.DataFrame(records)
        except Exception:
            logger.exception("Error loading our baseline projections")
            refs["our_baseline"] = None
    else:
        logger.warning("Our baseline projections not found: %s", baseline_path)
        refs["our_baseline"] = None

    # SDC Method + New Data
    sdc_new_path = (
        PROJECT_ROOT / "data" / "exports" / "sdc_method_new_data"
        / "state_population_by_year.csv"
    )
    if sdc_new_path.exists():
        try:
            df = pd.read_csv(sdc_new_path)
            df.columns = [c.strip().lower() for c in df.columns]
            refs["sdc_new_data"] = df.rename(
                columns={"total_population": "population"},
            )[["year", "population"]]
        except Exception:
            logger.exception("Error loading SDC new data projections")
            refs["sdc_new_data"] = None
    else:
        logger.warning("SDC Method + New Data projections not found: %s", sdc_new_path)
        refs["sdc_new_data"] = None

    return refs


def _load_fips_mapping() -> dict[str, str]:
    """Load county FIPS -> county name mapping.

    Returns dict mapping FIPS string (e.g. '38001') to county name.
    """
    path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    if not path.exists():
        logger.warning("County FIPS mapping not found: %s", path)
        return {}
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        fips_col = "county_fips" if "county_fips" in df.columns else "fips"
        name_col = "county_name" if "county_name" in df.columns else "county"
        mapping = {}
        for _, row in df.iterrows():
            fips_str = str(int(row[fips_col])).zfill(5)
            mapping[fips_str] = str(row[name_col]).replace(" County", "")
        return mapping
    except Exception:
        logger.exception("Error loading FIPS mapping")
        return {}


# ===================================================================
# Tab 1: Executive Summary
# ===================================================================

def _build_waterfall_chart(
    waterfall_df: pd.DataFrame,
    refs: dict[str, pd.DataFrame | None],
    template_name: str,
) -> str:
    """Build the decomposition waterfall chart HTML."""
    if waterfall_df is None or waterfall_df.empty:
        return '<p class="placeholder">Decomposition waterfall data not available.</p>'

    # Extract step data
    steps = waterfall_df.sort_values("step")
    names = steps["feature_name"].tolist()
    pop_2050 = steps["pop_2050"].tolist()
    deltas = steps["delta_from_previous"].tolist()

    # Build waterfall bars
    # First bar is the base (total type), last bar is the total, intermediates are relative
    measures = []
    texts = []
    colors = []

    for i, (name, delta, pop) in enumerate(zip(names, deltas, pop_2050)):
        if i == 0:
            # Starting bar
            measures.append("absolute")
            texts.append(_fmt_pop(pop))
            colors.append(TOTAL_BAR_COLOR)
        elif i == len(names) - 1:
            # Final total bar
            measures.append("total")
            texts.append(_fmt_pop(pop))
            colors.append(TOTAL_BAR_COLOR)
        else:
            # Incremental bar
            measures.append("relative")
            texts.append(_fmt_change(delta))
            colors.append(POSITIVE_COLOR if delta >= 0 else NEGATIVE_COLOR)

    fig = go.Figure(go.Waterfall(
        name="Feature Decomposition",
        orientation="v",
        measure=measures,
        x=names,
        textposition="outside",
        text=texts,
        y=[pop_2050[0]] + deltas[1:],
        connector={"line": {"color": MID_GRAY, "width": 1}},
        increasing={"marker": {"color": POSITIVE_COLOR}},
        decreasing={"marker": {"color": NEGATIVE_COLOR}},
        totals={"marker": {"color": TOTAL_BAR_COLOR}},
    ))

    # Add reference lines
    shapes = []
    annotations = []
    if refs.get("sdc_original") is not None:
        sdc_2050 = refs["sdc_original"]
        sdc_2050_val = sdc_2050[sdc_2050["year"] == 2050]
        if len(sdc_2050_val) > 0:
            val = float(sdc_2050_val["population"].iloc[0])
            shapes.append(
                dict(type="line", x0=-0.5, x1=len(names) - 0.5, y0=val, y1=val,
                     line=dict(color=SDC_COLOR, width=2, dash="dash"))
            )
            annotations.append(
                dict(x=len(names) - 1, y=val, text=f"SDC 2024: {_fmt_pop(val)}",
                     showarrow=False, xanchor="right", yshift=12,
                     font=dict(color=SDC_COLOR, size=11))
            )

    if refs.get("our_baseline") is not None:
        bl = refs["our_baseline"]
        bl_2050 = bl[bl["year"] == 2050]
        if len(bl_2050) > 0:
            val = float(bl_2050["population"].iloc[0])
            shapes.append(
                dict(type="line", x0=-0.5, x1=len(names) - 0.5, y0=val, y1=val,
                     line=dict(color=SCENARIO_COLORS["baseline"], width=2, dash="dash"))
            )
            annotations.append(
                dict(x=len(names) - 1, y=val,
                     text=f"Our Baseline: {_fmt_pop(val)}",
                     showarrow=False, xanchor="right", yshift=-12,
                     font=dict(color=SCENARIO_COLORS["baseline"], size=11))
            )

    fig.update_layout(
        template=template_name,
        title="Feature Decomposition: SDC Base to Our Model (2050 Population)",
        yaxis_title="2050 State Population",
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        height=500,
        margin=dict(b=120),
        xaxis=dict(tickangle=-30),
    )

    div_id = "waterfall-chart"
    chart_json = _chart_json(fig)
    return f"""
    <div class="chart-container">
        <div id="{div_id}" style="width:100%;height:500px;"></div>
        <script>
            Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                {{responsive: true, displayModeBar: false}});
        </script>
    </div>
    """


def _build_tornado_chart(
    tornado_df: pd.DataFrame,
    template_name: str,
) -> str:
    """Build the tornado chart HTML."""
    if tornado_df is None or tornado_df.empty:
        return '<p class="placeholder">Tornado summary data not available.</p>'

    # Sort by range (widest at top)
    df = tornado_df.sort_values("range", ascending=True).reset_index(drop=True)

    baseline_pop = float(df["baseline_pop_2050"].iloc[0])

    fig = go.Figure()

    # Low side (below baseline) — blue
    fig.add_trace(go.Bar(
        name="Below Baseline",
        y=df["parameter_label"],
        x=df["min_pop_2050"] - baseline_pop,
        base=baseline_pop,
        orientation="h",
        marker_color=BLUE,
        text=df["min_variant"],
        textposition="inside",
        textfont=dict(size=10, color="white"),
        hovertemplate=(
            "%{y}<br>"
            "Variant: %{text}<br>"
            "Population: %{customdata:,}<br>"
            "<extra></extra>"
        ),
        customdata=df["min_pop_2050"],
    ))

    # High side (above baseline) — orange
    fig.add_trace(go.Bar(
        name="Above Baseline",
        y=df["parameter_label"],
        x=df["max_pop_2050"] - baseline_pop,
        base=baseline_pop,
        orientation="h",
        marker_color=SDC_COLOR,
        text=df["max_variant"],
        textposition="inside",
        textfont=dict(size=10, color="white"),
        hovertemplate=(
            "%{y}<br>"
            "Variant: %{text}<br>"
            "Population: %{customdata:,}<br>"
            "<extra></extra>"
        ),
        customdata=df["max_pop_2050"],
    ))

    # Baseline reference line
    fig.add_vline(
        x=baseline_pop, line_width=2, line_color=NAVY, line_dash="solid",
        annotation_text=f"Baseline: {_fmt_pop(baseline_pop)}",
        annotation_position="top",
        annotation_font_size=11,
        annotation_font_color=NAVY,
    )

    # Range annotations on right
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row["max_pop_2050"],
            y=row["parameter_label"],
            text=f"Range: {_fmt_pop(row['range'])}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(size=10, color=DARK_GRAY),
        )

    fig.update_layout(
        template=template_name,
        title="Tornado Chart: Parameter Sensitivity on 2050 State Population",
        xaxis_title="2050 State Population",
        barmode="overlay",
        height=max(350, len(df) * 55 + 120),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=200, r=120),
    )

    div_id = "tornado-chart"
    chart_json = _chart_json(fig)
    return f"""
    <div class="chart-container">
        <div id="{div_id}" style="width:100%;height:{max(350, len(df) * 55 + 120)}px;"></div>
        <script>
            Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                {{responsive: true, displayModeBar: false}});
        </script>
    </div>
    """


def _build_key_findings(
    tornado_df: pd.DataFrame | None,
    waterfall_df: pd.DataFrame | None,
    refs: dict[str, pd.DataFrame | None],
) -> str:
    """Generate the key findings callout box dynamically from data."""
    bullets: list[str] = []

    # Most sensitive parameter
    if tornado_df is not None and not tornado_df.empty:
        top = tornado_df.sort_values("range", ascending=False).iloc[0]
        bullets.append(
            f"<strong>{top['parameter_label']}</strong> is the most sensitive parameter, "
            f"producing a population range of {_fmt_pop(top['min_pop_2050'])} to "
            f"{_fmt_pop(top['max_pop_2050'])} ({_fmt_pop(top['range'])} spread) in 2050."
        )

        # Second most sensitive
        if len(tornado_df) > 1:
            second = tornado_df.sort_values("range", ascending=False).iloc[1]
            bullets.append(
                f"<strong>{second['parameter_label']}</strong> is the second most "
                f"sensitive, with a range of {_fmt_pop(second['range'])}."
            )

    # Decomposition insight
    if waterfall_df is not None and not waterfall_df.empty:
        steps = waterfall_df.sort_values("step")
        if len(steps) >= 2:
            # Find the largest absolute delta
            mid_steps = steps.iloc[1:-1] if len(steps) > 2 else steps.iloc[1:]
            if not mid_steps.empty:
                max_row = mid_steps.loc[mid_steps["delta_from_previous"].abs().idxmax()]
                direction = "increases" if max_row["delta_from_previous"] > 0 else "decreases"
                bullets.append(
                    f"In the decomposition, <strong>{max_row['feature_name']}</strong> "
                    f"{direction} the projection by {_fmt_change(max_row['delta_from_previous'])} "
                    f"people, the largest single-feature effect."
                )

            # Total gap explanation
            start_pop = float(steps.iloc[0]["pop_2050"])
            end_pop = float(steps.iloc[-1]["pop_2050"])
            gap = end_pop - start_pop
            bullets.append(
                f"The total gap between the SDC base methodology and all features combined "
                f"is {_fmt_change(gap)} at 2050 "
                f"({start_pop:,.0f} vs. {end_pop:,.0f})."
            )

    # Reference comparison
    if refs.get("sdc_original") is not None and refs.get("our_baseline") is not None:
        sdc_2050 = refs["sdc_original"][refs["sdc_original"]["year"] == 2050]
        bl_2050 = refs["our_baseline"][refs["our_baseline"]["year"] == 2050]
        if len(sdc_2050) > 0 and len(bl_2050) > 0:
            sdc_val = float(sdc_2050["population"].iloc[0])
            bl_val = float(bl_2050["population"].iloc[0])
            diff = bl_val - sdc_val
            pct = diff / sdc_val * 100
            bullets.append(
                f"Our baseline projects {_fmt_pop(bl_val)} at 2050, which is "
                f"{_fmt_change(diff)} ({pct:+.1f}%) relative to the SDC 2024 "
                f"projection of {_fmt_pop(sdc_val)}."
            )

    if not bullets:
        bullets.append("Key findings will be generated once sensitivity data is available.")

    items = "\n".join(f"            <li>{b}</li>" for b in bullets)
    return f"""
    <div class="callout-box">
        <h3>Key Findings</h3>
        <ul style="margin: 8px 0 0 20px; font-size: 14px; line-height: 1.8;">
{items}
        </ul>
    </div>
    """


def _build_tab_executive_summary(
    waterfall_df: pd.DataFrame | None,
    tornado_df: pd.DataFrame | None,
    refs: dict[str, pd.DataFrame | None],
    template_name: str,
) -> str:
    """Build Tab 1: Executive Summary."""
    html_parts: list[str] = []
    html_parts.append("<h2>Executive Summary</h2>")

    # Key findings callout
    html_parts.append(_build_key_findings(tornado_df, waterfall_df, refs))

    # Waterfall chart
    html_parts.append("<h3>Feature Decomposition Waterfall</h3>")
    html_parts.append(_build_waterfall_chart(waterfall_df, refs, template_name))

    # Tornado chart
    html_parts.append("<h3>Parameter Sensitivity Tornado</h3>")
    html_parts.append(_build_tornado_chart(tornado_df, template_name))

    return "\n".join(html_parts)


# ===================================================================
# Tab 2: Parameter Sweeps
# ===================================================================

def _build_sweep_chart(
    sweep_name: str,
    sweep_title: str,
    sweep_df: pd.DataFrame,
    refs: dict[str, pd.DataFrame | None],
    template_name: str,
    chart_index: int,
) -> str:
    """Build a single sweep line chart + summary table."""
    # Filter to state total (fips=38 or fips='38')
    df = sweep_df.copy()
    df["fips"] = df["fips"].astype(str).str.strip()
    state_df = df[df["fips"].isin(["38", "38.0"])]
    if state_df.empty:
        # Try integer comparison
        try:
            state_df = df[df["fips"].astype(float).astype(int) == 38]
        except (ValueError, TypeError):
            return f'<p class="placeholder">No state-level data found for {sweep_title}.</p>'

    if state_df.empty:
        return f'<p class="placeholder">No state-level data found for {sweep_title}.</p>'

    variants = sorted(state_df["variant"].unique())

    fig = go.Figure()

    # Determine which variant is the baseline/default (bold)
    baseline_variants = {"bebr", "baseline", "default", "current", "none", "no_caps", "no_smoothing"}
    bold_variant = None
    for v in variants:
        if str(v).lower() in baseline_variants:
            bold_variant = v
            break

    for i, variant in enumerate(variants):
        vdf = state_df[state_df["variant"] == variant].sort_values("year")
        is_bold = (variant == bold_variant)
        color = SWEEP_PALETTE[i % len(SWEEP_PALETTE)]
        fig.add_trace(go.Scatter(
            x=vdf["year"],
            y=vdf["population"],
            mode="lines",
            name=str(variant),
            line=dict(
                color=color,
                width=3 if is_bold else 1.5,
            ),
            hovertemplate=f"{variant}<br>Year: %{{x}}<br>Population: %{{y:,.0f}}<extra></extra>",
        ))

    # Reference lines
    if refs.get("sdc_original") is not None:
        sdc = refs["sdc_original"]
        fig.add_trace(go.Scatter(
            x=sdc["year"], y=sdc["population"],
            mode="lines", name="SDC 2024 Original",
            line=dict(color=SDC_COLOR, width=2, dash="dash"),
            hovertemplate="SDC 2024<br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>",
        ))

    if refs.get("our_baseline") is not None:
        bl = refs["our_baseline"]
        fig.add_trace(go.Scatter(
            x=bl["year"], y=bl["population"],
            mode="lines", name="Our Baseline",
            line=dict(color=SCENARIO_COLORS["baseline"], width=2, dash="dot"),
            hovertemplate="Our Baseline<br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        template=template_name,
        title=sweep_title,
        xaxis_title="Year",
        yaxis_title="State Population",
        height=420,
        legend=dict(font=dict(size=10)),
    )

    div_id = f"sweep-chart-{chart_index}"
    chart_json = _chart_json(fig)

    # Summary table: variant -> 2050 pop -> diff from baseline variant
    table_rows = []
    baseline_2050 = None
    summary_data: list[tuple[str, float]] = []

    for variant in variants:
        vdf = state_df[state_df["variant"] == variant]
        pop_2050_rows = vdf[vdf["year"] == 2050]
        if len(pop_2050_rows) > 0:
            pop_val = float(pop_2050_rows["population"].iloc[0])
        else:
            # Use last available year
            vdf_sorted = vdf.sort_values("year")
            pop_val = float(vdf_sorted["population"].iloc[-1]) if len(vdf_sorted) > 0 else 0
        summary_data.append((str(variant), pop_val))
        if variant == bold_variant:
            baseline_2050 = pop_val

    if baseline_2050 is None and summary_data:
        baseline_2050 = summary_data[0][1]

    for variant_name, pop_val in summary_data:
        diff = pop_val - baseline_2050 if baseline_2050 else 0
        diff_str = _fmt_change(diff) if baseline_2050 else "N/A"
        css_class = _delta_class(diff)
        bold = " style='font-weight:600;'" if variant_name == str(bold_variant) else ""
        table_rows.append(
            f"<tr{bold}>"
            f"<td>{variant_name}</td>"
            f'<td class="num">{_fmt_pop(pop_val)}</td>'
            f'<td class="num {css_class}">{diff_str}</td>'
            f"</tr>"
        )

    table_html = "\n".join(table_rows)

    return f"""
    <div style="margin-bottom: 32px;">
        <div class="chart-container">
            <div id="{div_id}" style="width:100%;height:420px;"></div>
            <script>
                Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                    {{responsive: true, displayModeBar: false}});
            </script>
        </div>
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Variant</th>
                        <th class="num">2050 Population</th>
                        <th class="num">Diff from Baseline</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>
    </div>
    """


def _build_tab_parameter_sweeps(
    sweeps: dict[str, pd.DataFrame],
    refs: dict[str, pd.DataFrame | None],
    template_name: str,
) -> str:
    """Build Tab 2: Parameter Sweeps."""
    html_parts: list[str] = []
    html_parts.append("<h2>Parameter Sweeps</h2>")
    html_parts.append(
        '<p class="note">Each chart shows the state population trajectory (2025-2055) '
        "under different settings for a single parameter, holding all others at baseline. "
        "Dashed reference lines show SDC 2024 Original and Our Baseline for context.</p>"
    )

    chart_index = 0
    for stem, title, _expected in SWEEP_CONFIG:
        if stem in sweeps:
            html_parts.append(
                _build_sweep_chart(stem, title, sweeps[stem], refs, template_name, chart_index)
            )
            chart_index += 1
        else:
            if stem == "sweep_age_resolution":
                html_parts.append(
                    f'<div style="margin-bottom:32px;">'
                    f"<h3>{title}</h3>"
                    f'<p class="note">Age resolution sweep data not available (optional analysis).</p>'
                    f"</div>"
                )
            else:
                html_parts.append(
                    f'<div style="margin-bottom:32px;">'
                    f"<h3>{title}</h3>"
                    f'<p class="placeholder">Sweep data file not found: {stem}.csv</p>'
                    f"</div>"
                )

    return "\n".join(html_parts)


# ===================================================================
# Tab 3: County Impact
# ===================================================================

def _build_county_heatmap(
    sweeps: dict[str, pd.DataFrame],
    fips_map: dict[str, str],
    template_name: str,
) -> str:
    """Build county sensitivity heatmap (top 20 counties by total sensitivity)."""
    if not sweeps:
        return '<p class="placeholder">No sweep data available for county heatmap.</p>'

    # For each county and sweep, compute range of 2050 population as % of 2025 base
    county_sweep_ranges: dict[str, dict[str, float]] = {}

    for stem, sweep_df in sweeps.items():
        df = sweep_df.copy()
        df["fips"] = df["fips"].astype(str).str.strip()
        # Remove ".0" suffix if present
        df["fips"] = df["fips"].str.replace(r"\.0$", "", regex=True)

        # Filter to county FIPS (38xxx, not state 38)
        county_df = df[df["fips"].str.match(r"^38\d{3}$")]
        if county_df.empty:
            continue

        for fips_code, grp in county_df.groupby("fips"):
            fips_str = str(fips_code)
            if fips_str not in county_sweep_ranges:
                county_sweep_ranges[fips_str] = {}

            # Get 2025 base population
            base_rows = grp[grp["year"] == 2025]
            if base_rows.empty:
                base_pop = grp.sort_values("year")["population"].iloc[0]
            else:
                base_pop = base_rows["population"].mean()

            # Get 2050 range
            pop_2050 = grp[grp["year"] == 2050]["population"]
            if pop_2050.empty:
                pop_2050 = grp[grp["year"] == grp["year"].max()]["population"]

            if len(pop_2050) > 0 and base_pop > 0:
                range_pct = (pop_2050.max() - pop_2050.min()) / base_pop * 100
                sweep_label = stem.replace("sweep_", "").replace("_", " ").title()
                county_sweep_ranges[fips_str][sweep_label] = range_pct

    if not county_sweep_ranges:
        return '<p class="placeholder">Could not compute county sensitivity ranges.</p>'

    # Build DataFrame
    range_df = pd.DataFrame.from_dict(county_sweep_ranges, orient="index").fillna(0)

    # Total sensitivity = sum of ranges
    range_df["total"] = range_df.sum(axis=1)
    range_df = range_df.sort_values("total", ascending=True)

    # Top 20
    top_counties = range_df.tail(20).copy()
    top_counties = top_counties.drop(columns=["total"])

    # Map FIPS to county names
    y_labels = [fips_map.get(f, f) for f in top_counties.index]

    fig = go.Figure(data=go.Heatmap(
        z=top_counties.values,
        x=top_counties.columns.tolist(),
        y=y_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Range (% of Base)"),
        hovertemplate=(
            "County: %{y}<br>"
            "Parameter: %{x}<br>"
            "Range: %{z:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template=template_name,
        title="County Sensitivity by Parameter (Top 20 Most Sensitive Counties)",
        xaxis_title="Sensitivity Parameter",
        yaxis_title="County",
        height=max(450, len(top_counties) * 28 + 120),
        margin=dict(l=140, b=100),
        xaxis=dict(tickangle=-30),
    )

    div_id = "county-heatmap"
    chart_json = _chart_json(fig)
    return f"""
    <div class="chart-container">
        <div id="{div_id}" style="width:100%;height:{max(450, len(top_counties) * 28 + 120)}px;"></div>
        <script>
            Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                {{responsive: true, displayModeBar: false}});
        </script>
    </div>
    """


def _build_county_fan_charts(
    sweeps: dict[str, pd.DataFrame],
    fips_map: dict[str, str],
    template_name: str,
) -> str:
    """Build county uncertainty fan charts for featured counties."""
    if not sweeps:
        return '<p class="placeholder">No sweep data available for county fan charts.</p>'

    # Combine all sweep data for county-level min/max envelope
    all_county_data: list[pd.DataFrame] = []
    for sweep_df in sweeps.values():
        df = sweep_df.copy()
        df["fips"] = df["fips"].astype(str).str.strip()
        df["fips"] = df["fips"].str.replace(r"\.0$", "", regex=True)
        all_county_data.append(df)

    if not all_county_data:
        return '<p class="placeholder">No county data found in sweep files.</p>'

    combined = pd.concat(all_county_data, ignore_index=True)

    charts_html: list[str] = []

    for fips_code, county_name in FEATURED_COUNTIES.items():
        county_df = combined[combined["fips"] == fips_code]
        if county_df.empty:
            charts_html.append(
                f'<div class="small-multiple">'
                f'<p class="placeholder">{county_name}: No data available</p></div>'
            )
            continue

        # Compute min/max/baseline at each year
        envelope = county_df.groupby("year")["population"].agg(["min", "max", "median"])
        envelope = envelope.sort_index()

        fig = go.Figure()

        # Shaded band (min to max)
        fig.add_trace(go.Scatter(
            x=list(envelope.index) + list(envelope.index)[::-1],
            y=list(envelope["max"]) + list(envelope["min"])[::-1],
            fill="toself",
            fillcolor="rgba(5, 99, 193, 0.15)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Range",
        ))

        # Max line (thin)
        fig.add_trace(go.Scatter(
            x=envelope.index, y=envelope["max"],
            mode="lines",
            line=dict(color=BLUE, width=1, dash="dot"),
            showlegend=False,
            hovertemplate="Max: %{y:,.0f}<extra></extra>",
        ))

        # Min line (thin)
        fig.add_trace(go.Scatter(
            x=envelope.index, y=envelope["min"],
            mode="lines",
            line=dict(color=BLUE, width=1, dash="dot"),
            showlegend=False,
            hovertemplate="Min: %{y:,.0f}<extra></extra>",
        ))

        # Median line (bold)
        fig.add_trace(go.Scatter(
            x=envelope.index, y=envelope["median"],
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            showlegend=False,
            hovertemplate="Median: %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            template=template_name,
            title=dict(text=county_name, font=dict(size=14)),
            height=260,
            margin=dict(l=50, r=20, t=40, b=30),
            xaxis=dict(title=""),
            yaxis=dict(title=""),
        )

        div_id = f"fan-chart-{fips_code}"
        chart_json = _chart_json(fig)
        charts_html.append(f"""
        <div class="small-multiple">
            <div id="{div_id}" style="width:100%;height:260px;"></div>
            <script>
                Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                    {{responsive: true, displayModeBar: false}});
            </script>
        </div>
        """)

    return '<div class="small-multiples-grid">' + "\n".join(charts_html) + "</div>"


def _build_county_sensitivity_table(
    county_summary: pd.DataFrame | None,
) -> str:
    """Build the sortable county sensitivity table."""
    if county_summary is None or county_summary.empty:
        return '<p class="placeholder">County sensitivity summary data not available.</p>'

    df = county_summary.sort_values("range_pct", ascending=False).reset_index(drop=True)

    rows: list[str] = []
    for _, row in df.iterrows():
        range_pct_val = float(row["range_pct"])
        highlight = ' style="background:#fffde7;"' if range_pct_val > 20 else ""
        rows.append(
            f"<tr{highlight}>"
            f"<td>{row['county_name']}</td>"
            f'<td class="num">{_fmt_pop(row["pop_2025"])}</td>'
            f'<td class="num">{_fmt_pop(row["baseline_2050"])}</td>'
            f'<td class="num">{_fmt_pop(row["min_2050"])}</td>'
            f'<td class="num">{_fmt_pop(row["max_2050"])}</td>'
            f'<td class="num">{range_pct_val:.1f}%</td>'
            f"<td>{row['most_sensitive_parameter']}</td>"
            f"</tr>"
        )

    table_rows = "\n".join(rows)

    # Table ID for sortable JS
    return f"""
    <div class="table-container">
        <table class="data-table" id="county-sensitivity-table">
            <thead>
                <tr>
                    <th onclick="sortTable('county-sensitivity-table', 0, 'str')"
                        style="cursor:pointer;">County</th>
                    <th class="num" onclick="sortTable('county-sensitivity-table', 1, 'num')"
                        style="cursor:pointer;">Pop 2025</th>
                    <th class="num" onclick="sortTable('county-sensitivity-table', 2, 'num')"
                        style="cursor:pointer;">Baseline 2050</th>
                    <th class="num" onclick="sortTable('county-sensitivity-table', 3, 'num')"
                        style="cursor:pointer;">Min 2050</th>
                    <th class="num" onclick="sortTable('county-sensitivity-table', 4, 'num')"
                        style="cursor:pointer;">Max 2050</th>
                    <th class="num" onclick="sortTable('county-sensitivity-table', 5, 'num')"
                        style="cursor:pointer;">Range %</th>
                    <th onclick="sortTable('county-sensitivity-table', 6, 'str')"
                        style="cursor:pointer;">Most Sensitive</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        <p class="note" style="margin-top:8px;">
            Click column headers to sort. Rows highlighted in yellow have sensitivity range &gt; 20%.
        </p>
    </div>
    """


def _build_tab_county_impact(
    sweeps: dict[str, pd.DataFrame],
    county_summary: pd.DataFrame | None,
    fips_map: dict[str, str],
    template_name: str,
) -> str:
    """Build Tab 3: County Impact."""
    html_parts: list[str] = []
    html_parts.append("<h2>County Impact</h2>")

    # Heatmap
    html_parts.append("<h3>County Sensitivity Heatmap</h3>")
    html_parts.append(
        '<p class="note">Sensitivity expressed as population range (max minus min across all '
        "variants) as a percentage of the county's 2025 base population. Top 20 most sensitive "
        "counties shown.</p>"
    )
    html_parts.append(_build_county_heatmap(sweeps, fips_map, template_name))

    # Fan charts
    html_parts.append("<h3>County Uncertainty Bands (Key Counties)</h3>")
    html_parts.append(
        '<p class="note">Shaded bands show the min-max population range across all '
        "sensitivity variants. Bold line is the median. Dotted lines are min/max bounds.</p>"
    )
    html_parts.append(_build_county_fan_charts(sweeps, fips_map, template_name))

    # Sensitivity table
    html_parts.append("<h3>Most Sensitive Parameter by County</h3>")
    html_parts.append(_build_county_sensitivity_table(county_summary))

    return "\n".join(html_parts)


# ===================================================================
# Tab 4: Decomposition Detail
# ===================================================================

def _build_feature_contribution_bar(
    waterfall_df: pd.DataFrame | None,
    template_name: str,
) -> str:
    """Build horizontal bar chart showing each feature's absolute contribution."""
    if waterfall_df is None or waterfall_df.empty:
        return '<p class="placeholder">Decomposition data not available.</p>'

    steps = waterfall_df.sort_values("step")

    # Exclude the first (base) and last (total) rows
    if len(steps) > 2:
        features = steps.iloc[1:-1].copy()
    else:
        features = steps.iloc[1:].copy()

    if features.empty:
        return '<p class="placeholder">No intermediate decomposition steps found.</p>'

    # Sort by absolute delta descending
    features = features.reindex(
        features["delta_from_previous"].abs().sort_values(ascending=True).index
    )

    colors = [
        POSITIVE_COLOR if d >= 0 else NEGATIVE_COLOR
        for d in features["delta_from_previous"]
    ]

    fig = go.Figure(go.Bar(
        y=features["feature_name"],
        x=features["delta_from_previous"],
        orientation="h",
        marker_color=colors,
        text=[_fmt_change(d) for d in features["delta_from_previous"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Delta: %{x:,.0f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_width=1, line_color=DARK_GRAY)

    fig.update_layout(
        template=template_name,
        title="Feature Contributions to 2050 Population (Absolute Delta)",
        xaxis_title="Population Change",
        height=max(300, len(features) * 50 + 120),
        showlegend=False,
        margin=dict(l=200, r=80),
    )

    div_id = "feature-contribution-bar"
    chart_json = _chart_json(fig)
    return f"""
    <div class="chart-container">
        <div id="{div_id}" style="width:100%;height:{max(300, len(features) * 50 + 120)}px;"></div>
        <script>
            Plotly.newPlot('{div_id}', {chart_json}.data, {chart_json}.layout,
                {{responsive: true, displayModeBar: false}});
        </script>
    </div>
    """


def _build_decomposition_table(waterfall_df: pd.DataFrame | None) -> str:
    """Build a detailed decomposition table with step data."""
    if waterfall_df is None or waterfall_df.empty:
        return '<p class="placeholder">Decomposition data not available.</p>'

    steps = waterfall_df.sort_values("step")

    rows: list[str] = []
    for _, row in steps.iterrows():
        delta = float(row["delta_from_previous"])
        cum_delta = float(row["cumulative_delta"])
        delta_str = _fmt_change(delta) if row["step"] > 0 else "---"
        cum_str = _fmt_change(cum_delta) if row["step"] > 0 else "---"
        css = _delta_class(delta) if row["step"] > 0 else ""
        rows.append(
            f"<tr>"
            f"<td>{int(row['step'])}</td>"
            f"<td>{row['feature_name']}</td>"
            f'<td class="num">{_fmt_pop(row["pop_2050"])}</td>'
            f'<td class="num {css}">{delta_str}</td>'
            f'<td class="num {css}">{cum_str}</td>'
            f"</tr>"
        )

    table_rows = "\n".join(rows)
    return f"""
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Feature</th>
                    <th class="num">2050 Population</th>
                    <th class="num">Delta from Previous</th>
                    <th class="num">Cumulative Delta</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """


def _build_tab_decomposition_detail(
    waterfall_df: pd.DataFrame | None,
    refs: dict[str, pd.DataFrame | None],
    template_name: str,
) -> str:
    """Build Tab 4: Decomposition Detail."""
    html_parts: list[str] = []
    html_parts.append("<h2>Decomposition Detail</h2>")
    html_parts.append(
        '<p class="note">The decomposition shows how starting from the SDC base methodology '
        "(equal-weight averaging, constant rates) and progressively adding our model's features "
        "changes the 2050 projection. Note: feature effects may not sum exactly due to interaction "
        "effects.</p>"
    )

    # Larger waterfall chart (same data, bigger)
    html_parts.append("<h3>Decomposition Waterfall</h3>")
    html_parts.append(_build_waterfall_chart(waterfall_df, refs, template_name))

    # Detailed step table
    html_parts.append("<h3>Step-by-Step Details</h3>")
    html_parts.append(_build_decomposition_table(waterfall_df))

    # Feature contribution bar chart
    html_parts.append("<h3>Feature Contributions (Absolute)</h3>")
    html_parts.append(_build_feature_contribution_bar(waterfall_df, template_name))

    return "\n".join(html_parts)


# ===================================================================
# Tab 5: Methodology Notes
# ===================================================================

def _build_tab_methodology() -> str:
    """Build Tab 5: Methodology Notes (static content)."""
    return """
    <h2>Methodology Notes</h2>

    <div class="narrative">
        <div class="narrative-section">
            <h3>What This Analysis Tests</h3>
            <p>Each sensitivity analysis holds all other parameters at a baseline setting and
            varies one parameter through a range of plausible values. This isolates the individual
            impact of each methodological choice on the population projection.</p>
        </div>

        <div class="narrative-section">
            <h3>Baseline Settings</h3>
            <ul>
                <li><strong>Migration:</strong> BEBR multi-period averaging</li>
                <li><strong>Bakken dampening:</strong> 60% (keep 60% of historical rates)</li>
                <li><strong>Convergence:</strong> None (constant rates)</li>
                <li><strong>Mortality improvement:</strong> None (constant survival rates)</li>
                <li><strong>Fertility:</strong> No adjustment (constant rates)</li>
                <li><strong>Rate caps:</strong> None</li>
                <li><strong>College-age smoothing:</strong> None</li>
                <li><strong>GQ separation:</strong> Current (GQ-corrected rates)</li>
            </ul>
        </div>

        <div class="narrative-section">
            <h3>The Decomposition Framework</h3>
            <p>The waterfall chart shows how starting from the SDC base methodology
            (equal-weight averaging, constant everything) and progressively adding our
            model's features changes the projection. This decomposes the total difference
            between "SDC Method + New Data" and "Our Model" into feature-specific
            contributions.</p>
            <p>Note that the sum of individual feature effects may not exactly equal the
            total difference due to <strong>interaction effects</strong> (features can amplify
            or dampen each other).</p>
        </div>

        <div class="narrative-section">
            <h3>Sensitivity Parameters Tested</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Description</th>
                        <th>Variants</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Migration Window Weighting</td>
                        <td>How historical migration periods are weighted when computing
                            average migration rates. Equal-weight treats all periods the same;
                            BEBR (our default) gives more weight to recent periods.</td>
                        <td>7 weighting schemes</td>
                    </tr>
                    <tr>
                        <td>Convergence Schedule</td>
                        <td>Whether and how quickly county-specific rates converge toward
                            state or national norms over the projection horizon.</td>
                        <td>4 schedules</td>
                    </tr>
                    <tr>
                        <td>Mortality Improvement</td>
                        <td>Annual rate of mortality improvement (declining death rates).
                            Higher improvement = more survivors = larger population.</td>
                        <td>5 rates (0% to 1%/yr)</td>
                    </tr>
                    <tr>
                        <td>Bakken Dampening</td>
                        <td>How much oil-patch county migration rates are attenuated
                            to reflect structural economic shifts away from peak Bakken
                            in-migration.</td>
                        <td>6 levels (0% to 100%)</td>
                    </tr>
                    <tr>
                        <td>GQ Correction</td>
                        <td>Whether group quarters (dorms, barracks, prisons) population
                            is separated from the household population before computing
                            migration rates.</td>
                        <td>2 (with/without)</td>
                    </tr>
                    <tr>
                        <td>Rate Caps</td>
                        <td>Maximum allowable net migration rates per age-sex cohort.
                            Prevents extreme rates from distorting projections.</td>
                        <td>4 cap levels</td>
                    </tr>
                    <tr>
                        <td>College-Age Smoothing</td>
                        <td>Whether 18-24 age cohort migration is smoothed to reduce
                            volatility from college enrollment fluctuations.</td>
                        <td>5 smoothing levels</td>
                    </tr>
                    <tr>
                        <td>Fertility Adjustment</td>
                        <td>Scaling of age-specific fertility rates (e.g., to model
                            recovery toward replacement or continued decline).</td>
                        <td>7 scaling factors</td>
                    </tr>
                    <tr>
                        <td>Age Resolution</td>
                        <td>Whether the projection uses single-year or 5-year age groups.
                            (Optional analysis; may not be available.)</td>
                        <td>2 (1-year / 5-year)</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="narrative-section">
            <h3>Important Caveats</h3>
            <ul>
                <li><strong>GQ correction effect is approximated</strong> &mdash; base population
                    adjustment only, not re-derived migration rates.</li>
                <li><strong>The decomposition is path-dependent</strong> &mdash; adding features in a
                    different order would produce different per-step deltas.</li>
                <li><strong>Approximate engine</strong> &mdash; our production model uses annual steps
                    and single-year ages; the sensitivity engine uses 5-year steps and groups, so
                    results are approximate.</li>
                <li><strong>Not prediction intervals</strong> &mdash; county-level uncertainty bands
                    represent sensitivity to parameter choices, not statistical prediction
                    intervals.</li>
            </ul>
        </div>
    </div>
    """


# ===================================================================
# Tab 6: Raw Data
# ===================================================================

def _build_tab_raw_data(
    tornado_df: pd.DataFrame | None,
    county_summary: pd.DataFrame | None,
) -> str:
    """Build Tab 6: Raw Data (formatted tables)."""
    html_parts: list[str] = []
    html_parts.append("<h2>Raw Data</h2>")
    html_parts.append(
        '<p class="note">Summary tables from the sensitivity analysis. Full sweep data '
        "is available as CSV files in <code>data/exports/sensitivity/</code>.</p>"
    )

    # Tornado summary table
    html_parts.append("<h3>Tornado Summary</h3>")
    if tornado_df is not None and not tornado_df.empty:
        rows: list[str] = []
        for _, row in tornado_df.sort_values("range", ascending=False).iterrows():
            rows.append(
                "<tr>"
                f"<td>{row['parameter_label']}</td>"
                f"<td>{row['min_variant']}</td>"
                f"<td>{row['max_variant']}</td>"
                f'<td class="num">{_fmt_pop(row["min_pop_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["max_pop_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["baseline_pop_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["range"])}</td>'
                "</tr>"
            )
        table_rows = "\n".join(rows)
        html_parts.append(f"""
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Min Variant</th>
                        <th>Max Variant</th>
                        <th class="num">Min Pop 2050</th>
                        <th class="num">Max Pop 2050</th>
                        <th class="num">Baseline Pop 2050</th>
                        <th class="num">Range</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """)
    else:
        html_parts.append('<p class="placeholder">Tornado summary data not available.</p>')

    # County sensitivity summary table
    html_parts.append("<h3>County Sensitivity Summary</h3>")
    if county_summary is not None and not county_summary.empty:
        rows = []
        for _, row in county_summary.sort_values("range_pct", ascending=False).iterrows():
            range_pct_val = float(row["range_pct"])
            rows.append(
                "<tr>"
                f"<td>{row.get('county_fips', '')}</td>"
                f"<td>{row['county_name']}</td>"
                f'<td class="num">{_fmt_pop(row["pop_2025"])}</td>'
                f'<td class="num">{_fmt_pop(row["baseline_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["min_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["max_2050"])}</td>'
                f'<td class="num">{_fmt_pop(row["range_abs"])}</td>'
                f'<td class="num">{range_pct_val:.1f}%</td>'
                f"<td>{row['most_sensitive_parameter']}</td>"
                "</tr>"
            )
        table_rows = "\n".join(rows)
        html_parts.append(f"""
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>FIPS</th>
                        <th>County</th>
                        <th class="num">Pop 2025</th>
                        <th class="num">Baseline 2050</th>
                        <th class="num">Min 2050</th>
                        <th class="num">Max 2050</th>
                        <th class="num">Range (Abs)</th>
                        <th class="num">Range %</th>
                        <th>Most Sensitive</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """)
    else:
        html_parts.append(
            '<p class="placeholder">County sensitivity summary data not available.</p>'
        )

    return "\n".join(html_parts)


# ===================================================================
# HTML Assembly
# ===================================================================

def _assemble_html(
    tab1: str,
    tab2: str,
    tab3: str,
    tab4: str,
    tab5: str,
    tab6: str,
) -> str:
    """Assemble the six tab sections into a complete HTML document."""
    subtitle = (
        f"Sensitivity Analysis | 10 Parameter Sweeps | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ND Sensitivity Analysis Report - {TODAY.isoformat()}</title>
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

        .callout-box p, .callout-box li {{
            margin: 4px 0;
            font-size: 14px;
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

        .narrative ul, .narrative ol {{
            font-size: 14px;
            line-height: 1.6;
            margin: 8px 0 8px 20px;
        }}

        .narrative li {{
            margin-bottom: 8px;
        }}

        /* Small multiples grid */
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
            body {{ background: white; }}
        }}

        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .small-multiples-grid {{ grid-template-columns: 1fr 1fr; }}
            .header {{ padding: 20px; }}
            .tab-nav {{ padding: 0 10px; overflow-x: auto; }}
            .tab-btn {{ padding: 10px 12px; font-size: 12px; }}
            .tab-content {{ padding: 16px; }}
        }}

        @media (max-width: 600px) {{
            .small-multiples-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>North Dakota Population Projections: Sensitivity Analysis</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('summary')">Executive Summary</button>
        <button class="tab-btn" onclick="switchTab('sweeps')">Parameter Sweeps</button>
        <button class="tab-btn" onclick="switchTab('county')">County Impact</button>
        <button class="tab-btn" onclick="switchTab('decomposition')">Decomposition Detail</button>
        <button class="tab-btn" onclick="switchTab('methodology')">Methodology Notes</button>
        <button class="tab-btn" onclick="switchTab('rawdata')">Raw Data</button>
    </div>

    <div id="tab-summary" class="tab-content active">
        {tab1}
    </div>

    <div id="tab-sweeps" class="tab-content">
        {tab2}
    </div>

    <div id="tab-county" class="tab-content">
        {tab3}
    </div>

    <div id="tab-decomposition" class="tab-content">
        {tab4}
    </div>

    <div id="tab-methodology" class="tab-content">
        {tab5}
    </div>

    <div id="tab-rawdata" class="tab-content">
        {tab6}
    </div>

    <div class="footer">
        North Dakota Cohort-Component Population Projections |
        Sensitivity Analysis Report |
        Generated on {TODAY.strftime('%B %d, %Y')} |
        Data: Census PEP 2025 Vintage
    </div>

    <script>
        function switchTab(tabId) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(
                function(el) {{ el.classList.remove('active'); }}
            );
            document.querySelectorAll('.tab-btn').forEach(
                function(el) {{ el.classList.remove('active'); }}
            );

            // Show selected tab
            document.getElementById('tab-' + tabId).classList.add('active');

            // Activate button
            var tabMap = {{
                'summary': 0, 'sweeps': 1, 'county': 2,
                'decomposition': 3, 'methodology': 4, 'rawdata': 5
            }};
            var buttons = document.querySelectorAll('.tab-btn');
            buttons[tabMap[tabId]].classList.add('active');

            // Trigger Plotly resize for proper chart rendering in hidden tabs
            setTimeout(function() {{
                var plots = document.getElementById('tab-' + tabId)
                    .querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 50);
        }}

        // Sortable table implementation
        function sortTable(tableId, colIndex, colType) {{
            var table = document.getElementById(tableId);
            if (!table) return;
            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));

            // Determine current sort direction
            var currentDir = table.getAttribute('data-sort-dir-' + colIndex) || 'asc';
            var newDir = (currentDir === 'asc') ? 'desc' : 'asc';
            table.setAttribute('data-sort-dir-' + colIndex, newDir);

            rows.sort(function(a, b) {{
                var aText = a.cells[colIndex].textContent.trim();
                var bText = b.cells[colIndex].textContent.trim();

                if (colType === 'num') {{
                    // Parse numeric values (remove commas, %, +)
                    var aVal = parseFloat(aText.replace(/[,%+]/g, '')) || 0;
                    var bVal = parseFloat(bText.replace(/[,%+]/g, '')) || 0;
                    return newDir === 'asc' ? aVal - bVal : bVal - aVal;
                }} else {{
                    return newDir === 'asc'
                        ? aText.localeCompare(bText)
                        : bText.localeCompare(aText);
                }}
            }});

            rows.forEach(function(row) {{ tbody.appendChild(row); }});
        }}
    </script>
</body>
</html>"""


# ===================================================================
# Main entry point
# ===================================================================

def build_report(output_dir: Path) -> Path:
    """Build the complete sensitivity analysis HTML report.

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
    logger.info("ND Sensitivity Analysis Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("=" * 60)

    sensitivity_dir = PROJECT_ROOT / "data" / "exports" / "sensitivity"

    # --- Load sweep data ---
    sweeps: dict[str, pd.DataFrame] = {}
    for stem, title, _expected in SWEEP_CONFIG:
        path = sensitivity_dir / f"{stem}.csv"
        df = _load_sweep_file(path)
        if df is not None:
            sweeps[stem] = df
            logger.info("  Loaded %s: %d rows, %d variants",
                        stem, len(df), df["variant"].nunique())
        else:
            logger.info("  Skipped %s (not found)", stem)

    logger.info("Loaded %d of %d sweep files", len(sweeps), len(SWEEP_CONFIG))

    # --- Load summary files ---
    waterfall_df = _load_summary_csv(sensitivity_dir / "decomposition_waterfall.csv")
    tornado_df = _load_summary_csv(sensitivity_dir / "tornado_summary.csv")
    county_summary = _load_summary_csv(sensitivity_dir / "county_sensitivity_summary.csv")

    # --- Load reference projections ---
    logger.info("Loading reference projections...")
    refs = _load_reference_projections()
    for key, val in refs.items():
        if val is not None:
            logger.info("  %s: %d rows", key, len(val))
        else:
            logger.info("  %s: not found", key)

    # --- Load FIPS mapping ---
    fips_map = _load_fips_mapping()
    logger.info("FIPS mapping: %d counties", len(fips_map))

    # --- Register Plotly template ---
    template_name = _register_template()

    # --- Build tabs ---
    logger.info("Building Tab 1: Executive Summary...")
    tab1 = _build_tab_executive_summary(waterfall_df, tornado_df, refs, template_name)

    logger.info("Building Tab 2: Parameter Sweeps...")
    tab2 = _build_tab_parameter_sweeps(sweeps, refs, template_name)

    logger.info("Building Tab 3: County Impact...")
    tab3 = _build_tab_county_impact(sweeps, county_summary, fips_map, template_name)

    logger.info("Building Tab 4: Decomposition Detail...")
    tab4 = _build_tab_decomposition_detail(waterfall_df, refs, template_name)

    logger.info("Building Tab 5: Methodology Notes...")
    tab5 = _build_tab_methodology()

    logger.info("Building Tab 6: Raw Data...")
    tab6 = _build_tab_raw_data(tornado_df, county_summary)

    # --- Assemble HTML ---
    logger.info("Assembling HTML report...")
    html = _assemble_html(tab1, tab2, tab3, tab4, tab5, tab6)

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nd_sensitivity_analysis_{DATE_STAMP}.html"
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Report written to: %s (%.1f MB)", output_path, file_size_mb)
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build ND Sensitivity Analysis HTML report",
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
