"""Build an interactive HTML dashboard comparing experiment results.

Created: 2026-03-12
Author: Claude Code / N. Haarstad

Purpose:
    Generates a self-contained interactive HTML dashboard for tracking and
    comparing A/B experiment results from the benchmarking pipeline. Designed
    to give a single-page overview of what has been tested, what remains, and
    how experiment variants compare across multiple dimensions.

Method:
    1. Parse the experiment catalog (Markdown) into structured entries.
    2. Load the experiment log (CSV) for completed experiment outcomes.
    3. Join to benchmark history artifacts via run_id (scorecards, projection
       curves, horizon summaries, county metrics).
    4. Build 5 interactive Plotly/HTML tabs:
       a. Experiment Tracker — catalog status, outcomes, KPIs
       b. Spaghetti Plot — overlaid state projection curves (ensemble view)
       c. Scorecard Comparison — side-by-side metrics with delta bar chart
       d. Horizon Analysis — accuracy degradation by forecast horizon
       e. County Detail — heatmap of county-level performance
    5. Assemble into a single self-contained HTML file with embedded CSS/JS.

Inputs:
    - ``docs/plans/experiment-catalog.md`` — experiment definitions
    - ``data/analysis/experiments/experiment_log.csv`` — experiment outcomes
    - ``data/analysis/benchmark_history/index.csv`` — benchmark run registry
    - ``data/analysis/benchmark_history/<run_id>/`` — per-run artifacts
    - ``config/benchmark_evaluation_policy.yaml`` — gate thresholds

Outputs:
    - ``data/analysis/experiments/experiment_dashboard.html`` — interactive dashboard

Usage::

    python scripts/analysis/build_experiment_dashboard.py [--output path] [--no-plotly-js]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root and sibling exports dir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "exports"))

from _report_theme import (  # noqa: E402
    BLUE,
    DARK_GRAY,
    MID_GRAY,
    NAVY,
    WHITE,
    get_plotly_template,
)

from cohort_projections.analysis.benchmarking import (  # noqa: E402
    DEFAULT_HISTORY_DIR,
)
from cohort_projections.analysis.experiment_log import (  # noqa: E402
    read_experiment_log,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CATALOG = PROJECT_ROOT / "docs" / "plans" / "experiment-catalog.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments"
DEFAULT_VARIANT_CATALOG_PATH = PROJECT_ROOT / "config" / "observatory_variants.yaml"

# Color palette for dynamic assignment when catalog-based colors are used
_COLOR_PALETTE: list[str] = [
    "#0563C1",  # Blue
    "#00B050",  # Green
    "#7030A0",  # Purple
    "#9DC3E6",  # Light blue
    "#ED7D31",  # Orange
    "#A9D18E",  # Light green
    "#BF8F00",  # Dark gold
    "#548235",  # Dark green
    "#C00000",  # Red
    "#00B0F0",  # Teal
    "#FFC000",  # Gold
    "#FF6699",  # Pink
    "#336699",  # Steel blue
    "#669933",  # Olive
    "#993366",  # Plum
]

# Fallback hardcoded color mapping — used when variant catalog is unavailable
_FALLBACK_EXPERIMENT_COLORS: dict[str, str] = {
    "EXP-A": "#0563C1",  # Blue
    "EXP-B": "#00B050",  # Green (the winner)
    "EXP-C": "#7030A0",  # Purple
    "EXP-D": "#9DC3E6",  # Light blue
    "EXP-E": "#ED7D31",  # Orange
    "EXP-F": "#A9D18E",  # Light green
    "EXP-G": "#BF8F00",  # Dark gold
    "EXP-H": "#548235",  # Dark green
    "EXP-I": "#C00000",  # Red
    "EXP-J": "#00B0F0",  # Teal
    "EXP-K": "#FFC000",  # Gold
}

# Fallback hardcoded slug-to-catalog-ID mapping
_FALLBACK_SLUG_TO_EXP: dict[str, str] = {
    "convergence-medium-hold-5": "EXP-A",
    "college-blend-70": "EXP-B",
    "gq-fraction-75": "EXP-C",
    "rate-cap-general-6pct": "EXP-D",
    "boom-dampening-peak-30": "EXP-E",
    "recent-period-2020-2025": "EXP-F",
    "mortality-improvement-03pct": "EXP-G",
    "billings-bakken-fips": "EXP-H",
    "college-blend-30": "EXP-I",
    "convergence-medium-hold-4": "EXP-J",
    "boom-dampening-peak-35": "EXP-K",
}


def _load_catalog_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Build SLUG_TO_EXP and EXPERIMENT_COLORS from the variant catalog YAML.

    Falls back to hardcoded mappings if the variant catalog cannot be loaded.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        (slug_to_exp mapping, experiment_colors mapping)
    """
    try:
        from cohort_projections.analysis.observatory.variant_catalog import (
            VariantCatalog,
        )

        if not DEFAULT_VARIANT_CATALOG_PATH.exists():
            raise FileNotFoundError(DEFAULT_VARIANT_CATALOG_PATH)

        cat = VariantCatalog(
            catalog_path=DEFAULT_VARIANT_CATALOG_PATH,
            experiment_log=pd.DataFrame(),
        )
        variants_df = cat.list_variants()

        slug_to_exp: dict[str, str] = {}
        colors: dict[str, str] = {}

        for i, (_, row) in enumerate(variants_df.iterrows()):
            vid = str(row["variant_id"])
            # Use the variant_id as the experiment ID — uppercase for display
            exp_id = vid.upper()
            # Use the slug from the catalog for matching
            slug_val = str(row.get("name", vid)).lower().replace(" ", "-")
            # Check if there's a slug in the raw catalog definition
            vdef = cat.get_variant(vid)
            slug_val = vdef.get("slug", slug_val)

            slug_to_exp[slug_val] = exp_id
            colors[exp_id] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]

        if slug_to_exp:
            return slug_to_exp, colors

    except Exception:
        pass

    return dict(_FALLBACK_SLUG_TO_EXP), dict(_FALLBACK_EXPERIMENT_COLORS)


# Initialize from catalog (with fallback)
SLUG_TO_EXP, EXPERIMENT_COLORS = _load_catalog_mappings()

CHAMPION_COLOR = NAVY
CANDIDATE_COLOR = BLUE

# Key metrics for comparison
DELTA_METRICS = [
    ("county_mape_overall", "Overall MAPE"),
    ("county_mape_urban_college", "College MAPE"),
    ("county_mape_rural", "Rural MAPE"),
    ("county_mape_bakken", "Bakken MAPE"),
    ("state_ape_recent_short", "State APE (short)"),
    ("state_ape_recent_medium", "State APE (medium)"),
]

SENTINEL_LABELS = {
    "sentinel_cass_mape": "Cass",
    "sentinel_grand_forks_mape": "Grand Forks",
    "sentinel_ward_mape": "Ward",
    "sentinel_burleigh_mape": "Burleigh",
    "sentinel_williams_mape": "Williams",
    "sentinel_mckenzie_mape": "McKenzie",
}


# ---------------------------------------------------------------------------
# Catalog parser
# ---------------------------------------------------------------------------
def parse_experiment_catalog(catalog_path: Path) -> list[dict[str, object]]:
    """Parse experiment-catalog.md into structured experiment entries.

    Args:
        catalog_path: Path to the experiment catalog Markdown file.

    Returns:
        List of dicts, each with keys: ``exp_id``, ``name``, ``tier``,
        ``slug``, ``parameter``, ``hypothesis``, ``results_text``,
        ``config_only``.
    """
    text = catalog_path.read_text(encoding="utf-8")
    entries: list[dict] = []
    current_tier: int | None = None

    # Split into blocks by H3 EXP- headings
    parts = re.split(r"(?=^### EXP-)", text, flags=re.MULTILINE)

    for part in parts:
        # Check for tier headers in non-EXP blocks (preamble text)
        if not part.startswith("### EXP-"):
            if "## Tier 1" in part:
                current_tier = 1
            if "## Tier 2" in part:
                current_tier = 2
            if "## Tier 3" in part:
                current_tier = 3
            continue

        # For EXP blocks, record the entry with current tier, then check
        # if this block's trailing text changes the tier for the next entry
        entry_tier = current_tier

        heading_match = re.match(r"### (EXP-\w+):\s*(.+)", part)
        if not heading_match:
            continue

        exp_id = heading_match.group(1)
        exp_name = heading_match.group(2).strip()

        # Extract table fields
        fields: dict[str, str] = {}
        for row_match in re.finditer(
            r"\|\s*([A-Za-z][\w\s]*?)\s*\|\s*(.+?)\s*\|", part
        ):
            key = row_match.group(1).strip().lower()
            val = row_match.group(2).strip()
            if key not in ("field", "---", "-----"):
                fields[key] = val

        slug = fields.get("slug", "").strip("`")
        notes = fields.get("notes", "")
        results = fields.get("results", "")

        entries.append(
            {
                "exp_id": exp_id,
                "name": exp_name,
                "tier": entry_tier,
                "slug": slug,
                "parameter": fields.get("parameter", "").strip("`"),
                "hypothesis": fields.get("hypothesis", ""),
                "results_text": results,
                "config_only": "Not benchmark-testable" not in notes,
            }
        )

        # Check if this block's trailing text sets the tier for the next entry
        if "## Tier 2" in part:
            current_tier = 2
        if "## Tier 3" in part:
            current_tier = 3

    # Parse generated experiment ideas table
    gen_section = re.search(
        r"### Generated Experiment Ideas\s*\n((?:\|.+\n)+)", text
    )
    if gen_section:
        for row in gen_section.group(1).strip().split("\n"):
            cells = [c.strip() for c in row.split("|")]
            # Filter out empty strings from split
            cells = [c for c in cells if c]
            if len(cells) >= 4 and cells[0].startswith("EXP-"):
                entries.append(
                    {
                        "exp_id": cells[0],
                        "name": cells[2],
                        "tier": 1,
                        "slug": cells[1].strip("`"),
                        "parameter": "",
                        "hypothesis": cells[2],
                        "results_text": "",
                        "config_only": True,
                    }
                )

    return entries


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _find_run_dir(run_id: str) -> Path | None:
    """Find the benchmark run directory for a given run_id."""
    d = DEFAULT_HISTORY_DIR / run_id
    return d if d.is_dir() else None


def _load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def load_dashboard_data(
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, object]:
    """Load all data needed for the experiment dashboard.

    Scans the experiment catalog, experiment log, benchmark index, and
    per-run artifacts to build a unified data structure for rendering.

    Args:
        catalog_path: Path to the experiment catalog Markdown file.

    Returns:
        Dict with keys: ``catalog``, ``exp_outcomes``, ``scorecards``,
        ``comparisons``, ``projection_curves``, ``horizon_summaries``,
        ``county_metrics``, ``state_metrics``, ``champion_scorecard``.
    """
    catalog = parse_experiment_catalog(catalog_path)

    # Load experiment log
    exp_log = read_experiment_log()

    # Load benchmark index
    index_path = DEFAULT_HISTORY_DIR / "index.csv"
    index_df = pd.read_csv(index_path, dtype=str) if index_path.exists() else pd.DataFrame()

    # Build run_id -> benchmark_label mapping from index
    label_to_run: dict[str, str] = {}
    if not index_df.empty:
        # Take one row per run_id (they share same benchmark_label)
        for _, row in index_df.drop_duplicates(subset=["run_id"]).iterrows():
            label_to_run[row["benchmark_label"]] = row["run_id"]

    # Map experiment log entries to catalog entries via slug matching
    exp_outcomes: dict[str, dict] = {}  # exp_id -> outcome data
    if not exp_log.empty:
        for _, row in exp_log.iterrows():
            # Extract slug from experiment_id (e.g., exp-20260309-college-blend-70)
            exp_id_str = str(row.get("experiment_id", ""))
            # Match against catalog slugs
            for slug, exp_id in SLUG_TO_EXP.items():
                if slug.replace("-", "") in exp_id_str.replace("-", ""):
                    exp_outcomes[exp_id] = {
                        "outcome": row.get("outcome", ""),
                        "run_id": row.get("run_id", ""),
                        "key_metrics": row.get("key_metrics_summary", ""),
                        "config_delta": row.get("config_delta_summary", ""),
                        "interpretation": row.get("interpretation", ""),
                    }
                    break

    # Load per-run artifacts for completed experiments
    scorecards: dict[str, list[dict]] = {}  # exp_id -> scorecard rows
    comparisons: dict[str, dict] = {}  # exp_id -> comparison data
    projection_curves: dict[str, pd.DataFrame] = {}  # exp_id -> curves
    horizon_summaries: dict[str, pd.DataFrame] = {}  # exp_id -> horizon data
    county_metrics: dict[str, pd.DataFrame] = {}  # exp_id -> county data
    state_metrics: dict[str, pd.DataFrame] = {}  # exp_id -> state data
    champion_scorecard: dict | None = None

    for exp_id, outcome_data in exp_outcomes.items():
        run_id = outcome_data.get("run_id", "")
        run_dir = _find_run_dir(run_id)
        if not run_dir:
            continue

        # Summary scorecard
        sc_path = run_dir / "summary_scorecard.json"
        if sc_path.exists():
            sc = _load_json(sc_path)
            scorecards[exp_id] = sc
            # Extract champion scorecard (same across all runs)
            if champion_scorecard is None:
                for entry in sc:
                    if entry.get("status_at_run") == "champion":
                        champion_scorecard = entry
                        break

        # Comparison to champion
        cmp_path = run_dir / "comparison_to_champion.json"
        if cmp_path.exists():
            comparisons[exp_id] = _load_json(cmp_path)

        # Projection curves
        curves_path = run_dir / "projection_curves.csv"
        if curves_path.exists():
            projection_curves[exp_id] = pd.read_csv(curves_path)

        # Horizon summary
        hz_path = run_dir / "annual_horizon_summary.csv"
        if hz_path.exists():
            horizon_summaries[exp_id] = pd.read_csv(hz_path)

        # County metrics
        cm_path = run_dir / "county_metrics.csv"
        if cm_path.exists():
            county_metrics[exp_id] = pd.read_csv(cm_path)

        # State metrics (for actuals)
        sm_path = run_dir / "state_metrics.csv"
        if sm_path.exists():
            state_metrics[exp_id] = pd.read_csv(sm_path)

    return {
        "catalog": catalog,
        "exp_outcomes": exp_outcomes,
        "scorecards": scorecards,
        "comparisons": comparisons,
        "projection_curves": projection_curves,
        "horizon_summaries": horizon_summaries,
        "county_metrics": county_metrics,
        "state_metrics": state_metrics,
        "champion_scorecard": champion_scorecard,
    }


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------
def _setup_plotly() -> str:
    """Register template, return template name."""
    tpl = get_plotly_template()
    name = "sdc_dashboard"
    pio.templates[name] = tpl
    return name


def build_spaghetti_fig(
    data: dict[str, object], template_name: str
) -> go.Figure:
    """Build the spaghetti plot: overlaid projection curves per origin year.

    Creates a Plotly figure with actual population, champion, and experiment
    projection curves.  Includes origin-year dropdown for filtering.

    Args:
        data: Dashboard data dict from :func:`load_dashboard_data`.
        template_name: Registered Plotly template name.

    Returns:
        Plotly Figure with traces and origin-year dropdown.
    """
    fig = go.Figure()

    # Collect actuals from any state_metrics file
    actuals_df = None
    for sm_df in data["state_metrics"].values():
        cols_needed = {"validation_year", "actual_state"}
        if cols_needed.issubset(sm_df.columns):
            actuals_df = (
                sm_df[["validation_year", "actual_state"]]
                .drop_duplicates()
                .sort_values("validation_year")
            )
            break

    origins = [2005, 2010, 2015, 2020]
    completed_ids = sorted(data["scorecards"].keys())

    # Build traces per origin, tracking visibility groups
    trace_meta: list[dict] = []  # origin, group for visibility control

    # --- Actuals ---
    if actuals_df is not None:
        fig.add_trace(
            go.Scatter(
                x=actuals_df["validation_year"],
                y=actuals_df["actual_state"],
                name="Actual (Census PEP)",
                mode="lines+markers",
                line={"color": "black", "width": 3},
                marker={"size": 5, "symbol": "circle"},
                legendgroup="actual",
                hovertemplate=(
                    "<b>Actual</b><br>"
                    "Year: %{x}<br>"
                    "Population: %{y:,.0f}<extra></extra>"
                ),
            )
        )
        trace_meta.append({"origin": "all", "group": "actual"})

    # --- Champion curves (from first experiment's data) ---
    champion_curves_df = None
    for exp_id in completed_ids:
        if exp_id in data["projection_curves"]:
            df = data["projection_curves"][exp_id]
            # Champion is the non-m2026r1 method — typically m2026
            champ_methods = [
                m for m in df["method"].unique() if m != "m2026r1"
            ]
            if champ_methods:
                champion_curves_df = df[df["method"] == champ_methods[0]]
            break

    if champion_curves_df is not None:
        for origin in origins:
            odf = champion_curves_df[champion_curves_df["origin_year"] == origin]
            if odf.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=odf["year"],
                    y=odf["projected_state"],
                    name="Champion (m2026)",
                    mode="lines",
                    line={"color": CHAMPION_COLOR, "width": 2, "dash": "dash"},
                    legendgroup="champion",
                    showlegend=(origin == origins[0]),
                    opacity=0.7,
                    hovertemplate=(
                        "<b>Champion (m2026)</b><br>"
                        f"Origin: {origin}<br>"
                        "Year: %{x}<br>"
                        "Projected: %{y:,.0f}<extra></extra>"
                    ),
                )
            )
            trace_meta.append({"origin": origin, "group": "champion"})

    # --- Experiment curves ---
    for exp_id in completed_ids:
        if exp_id not in data["projection_curves"]:
            continue
        df = data["projection_curves"][exp_id]
        color = EXPERIMENT_COLORS.get(exp_id, "#888888")
        # Find catalog entry name
        exp_label = exp_id
        for cat in data["catalog"]:
            if cat["exp_id"] == exp_id:
                exp_label = f"{exp_id}: {cat['name']}"
                break

        challenger_df = df[df["method"] == "m2026r1"]
        for origin in origins:
            odf = challenger_df[challenger_df["origin_year"] == origin]
            if odf.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=odf["year"],
                    y=odf["projected_state"],
                    name=exp_label,
                    mode="lines",
                    line={"color": color, "width": 2},
                    legendgroup=exp_id,
                    showlegend=(origin == origins[0]),
                    hovertemplate=(
                        f"<b>{exp_id}</b><br>"
                        f"Origin: {origin}<br>"
                        "Year: %{{x}}<br>"
                        "Projected: %{{y:,.0f}}<extra></extra>"
                    ),
                )
            )
            trace_meta.append({"origin": origin, "group": exp_id})

    # --- Origin year dropdown ---
    n_traces = len(trace_meta)
    buttons = []

    # "All Origins" button
    buttons.append(
        {
            "label": "All Origins",
            "method": "update",
            "args": [{"visible": [True] * n_traces}],
        }
    )

    for origin in origins:
        vis = []
        for tm in trace_meta:
            vis.append(tm["origin"] == "all" or tm["origin"] == origin)
        buttons.append(
            {
                "label": f"Origin {origin}",
                "method": "update",
                "args": [{"visible": vis}],
            }
        )

    fig.update_layout(
        template=template_name,
        title="State Population Projection Ensemble",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        legend={"orientation": "h", "y": -0.18, "x": 0, "xanchor": "left"},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": WHITE,
                "bordercolor": MID_GRAY,
                "font": {"size": 12},
                "pad": {"r": 10, "t": 10},
            }
        ],
        annotations=[
            {
                "text": "Origin Year:",
                "x": -0.01,
                "xref": "paper",
                "y": 1.12,
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12, "color": DARK_GRAY},
                "xanchor": "right",
            }
        ],
        height=550,
    )

    return fig


def build_delta_bar_fig(
    data: dict[str, object], template_name: str
) -> go.Figure:
    """Build grouped bar chart of MAPE deltas vs champion.

    Args:
        data: Dashboard data dict from :func:`load_dashboard_data`.
        template_name: Registered Plotly template name.

    Returns:
        Plotly Figure with grouped bars and gate threshold lines.
    """
    completed_ids = sorted(data["comparisons"].keys())
    metrics_to_show = [
        ("county_mape_overall", "Overall"),
        ("county_mape_urban_college", "College"),
        ("county_mape_rural", "Rural"),
        ("county_mape_bakken", "Bakken"),
    ]

    fig = go.Figure()

    for exp_id in completed_ids:
        cmp = data["comparisons"][exp_id]
        if not cmp.get("challengers"):
            continue
        deltas = cmp["challengers"][0].get("deltas", {})
        color = EXPERIMENT_COLORS.get(exp_id, "#888888")

        vals = [deltas.get(m, 0) for m, _ in metrics_to_show]

        fig.add_trace(
            go.Bar(
                name=exp_id,
                x=[label for _, label in metrics_to_show],
                y=vals,
                marker_color=color,
                text=[f"{v:+.3f}" for v in vals],
                textposition="outside",
                textfont={"size": 10},
                hovertemplate=(
                    f"<b>{exp_id}</b><br>"
                    "%{x}: %{y:+.4f} pp<extra></extra>"
                ),
            )
        )

    # Gate threshold reference lines
    thresholds = {
        "Overall": 0.05,
        "Rural": 0.10,
        "Bakken": 0.50,
    }
    for label, thresh in thresholds.items():
        fig.add_shape(
            type="line",
            x0=label,
            x1=label,
            y0=thresh,
            y1=thresh,
            xref="x",
            yref="y",
            line={"color": "#dc2626", "width": 2, "dash": "dot"},
        )
        fig.add_annotation(
            x=label,
            y=thresh,
            text=f"max {thresh}pp",
            showarrow=False,
            yshift=12,
            font={"size": 9, "color": "#dc2626"},
        )

    fig.add_hline(y=0, line_width=1, line_color=DARK_GRAY)

    fig.update_layout(
        template=template_name,
        title="MAPE Delta vs Champion (negative = improvement)",
        yaxis_title="Delta (percentage points)",
        barmode="group",
        legend={"orientation": "h", "y": -0.15},
        height=420,
    )

    return fig


def build_horizon_fig(
    data: dict[str, object], template_name: str
) -> go.Figure:
    """Build horizon analysis: MAPE and bias vs forecast horizon.

    Creates a two-panel subplot: top panel shows mean county MAPE by
    horizon, bottom panel shows mean signed percentage error (bias).

    Args:
        data: Dashboard data dict from :func:`load_dashboard_data`.
        template_name: Registered Plotly template name.

    Returns:
        Plotly Figure with two vertically stacked subplots.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Mean County MAPE by Horizon", "Mean Signed Bias (MPE) by Horizon"],
    )

    completed_ids = sorted(data["horizon_summaries"].keys())

    # Champion horizon data (from first experiment)
    champion_plotted = False
    for exp_id in completed_ids:
        hz_df = data["horizon_summaries"][exp_id]
        # Champion
        if not champion_plotted:
            champ = hz_df[hz_df["method"] != "m2026r1"].sort_values("horizon")
            if not champ.empty:
                fig.add_trace(
                    go.Scatter(
                        x=champ["horizon"],
                        y=champ["mean_county_mape"],
                        name="Champion (m2026)",
                        line={"color": CHAMPION_COLOR, "width": 2, "dash": "dash"},
                        legendgroup="champion",
                        hovertemplate="<b>Champion</b><br>Horizon: %{x}yr<br>MAPE: %{y:.2f}%<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=champ["horizon"],
                        y=champ["mean_county_mpe"],
                        name="Champion (m2026)",
                        line={"color": CHAMPION_COLOR, "width": 2, "dash": "dash"},
                        legendgroup="champion",
                        showlegend=False,
                        hovertemplate="<b>Champion</b><br>Horizon: %{x}yr<br>MPE: %{y:.2f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                champion_plotted = True

        # Challenger
        challenger = hz_df[hz_df["method"] == "m2026r1"].sort_values("horizon")
        if challenger.empty:
            continue
        color = EXPERIMENT_COLORS.get(exp_id, "#888888")
        fig.add_trace(
            go.Scatter(
                x=challenger["horizon"],
                y=challenger["mean_county_mape"],
                name=exp_id,
                line={"color": color, "width": 2},
                legendgroup=exp_id,
                hovertemplate=f"<b>{exp_id}</b><br>Horizon: %{{x}}yr<br>MAPE: %{{y:.2f}}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=challenger["horizon"],
                y=challenger["mean_county_mpe"],
                name=exp_id,
                line={"color": color, "width": 2},
                legendgroup=exp_id,
                showlegend=False,
                hovertemplate=f"<b>{exp_id}</b><br>Horizon: %{{x}}yr<br>MPE: %{{y:.2f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_width=1, line_color=DARK_GRAY, row=2, col=1)

    fig.update_layout(
        template=template_name,
        height=700,
        legend={"orientation": "h", "y": -0.1},
    )
    fig.update_xaxes(title_text="Forecast Horizon (years)", row=2, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="MPE (%)", row=2, col=1)

    return fig


def build_county_heatmap_fig(
    data: dict[str, object], template_name: str
) -> go.Figure:
    """Build county x experiment MAPE heatmap.

    Creates a Plotly heatmap with counties on the y-axis (grouped by
    category) and experiments on the x-axis, colored by mean MAPE.

    Args:
        data: Dashboard data dict from :func:`load_dashboard_data`.
        template_name: Registered Plotly template name.

    Returns:
        Plotly Heatmap figure with category separator lines.
    """
    completed_ids = sorted(data["county_metrics"].keys())
    if not completed_ids:
        fig = go.Figure()
        fig.add_annotation(text="No county data available", showarrow=False)
        return fig

    # Aggregate county MAPE across all origins/horizons for each experiment
    agg_frames: list[pd.DataFrame] = []
    champion_agg: pd.DataFrame | None = None

    for exp_id in completed_ids:
        cm = data["county_metrics"][exp_id]
        # Challenger
        challenger = cm[cm["method"] == "m2026r1"]
        agg = (
            challenger.groupby(["county_fips", "county_name", "category"])["pct_error"]
            .apply(lambda x: x.abs().mean())
            .reset_index()
            .rename(columns={"pct_error": "mape"})
        )
        agg["exp_id"] = exp_id
        agg_frames.append(agg)

        # Champion (take once)
        if champion_agg is None:
            champ = cm[cm["method"] != "m2026r1"]
            if not champ.empty:
                champion_agg = (
                    champ.groupby(["county_fips", "county_name", "category"])[
                        "pct_error"
                    ]
                    .apply(lambda x: x.abs().mean())
                    .reset_index()
                    .rename(columns={"pct_error": "mape"})
                )

    if not agg_frames:
        fig = go.Figure()
        fig.add_annotation(text="No county data available", showarrow=False)
        return fig

    all_agg = pd.concat(agg_frames, ignore_index=True)

    # Pivot to county x experiment matrix
    pivot = all_agg.pivot_table(
        index=["category", "county_name"],
        columns="exp_id",
        values="mape",
    )

    # Sort by category then county name
    category_order = ["Bakken", "Urban/College", "Reservation", "Rural"]
    pivot = pivot.reset_index()
    pivot["cat_order"] = pivot["category"].map(
        {c: i for i, c in enumerate(category_order)}
    )
    pivot = pivot.sort_values(["cat_order", "county_name"]).drop(columns=["cat_order"])

    # Build labels and z-matrix
    y_labels = [f"{row['county_name']} ({row['category'][:3]})" for _, row in pivot.iterrows()]
    exp_cols = [c for c in pivot.columns if c.startswith("EXP-")]
    exp_cols_sorted = sorted(exp_cols)
    z = pivot[exp_cols_sorted].values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=exp_cols_sorted,
            y=y_labels,
            colorscale=[
                [0.0, "#059669"],
                [0.3, "#ecfdf5"],
                [0.5, "#FFFFFF"],
                [0.7, "#fef2f2"],
                [1.0, "#dc2626"],
            ],
            zmin=0,
            zmax=max(30, float(z[~pd.isna(z)].max()) if z.size > 0 else 30),
            text=[[f"{v:.1f}" if not pd.isna(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="County: %{y}<br>Experiment: %{x}<br>MAPE: %{z:.2f}%<extra></extra>",
            colorbar={"title": "MAPE (%)"},
        )
    )

    # Add category separator lines
    cumulative = 0
    for cat in category_order:
        cat_count = len(pivot[pivot["category"] == cat])
        if cat_count > 0:
            cumulative += cat_count
            if cumulative < len(pivot):
                fig.add_hline(
                    y=cumulative - 0.5,
                    line_width=2,
                    line_color=NAVY,
                )

    fig.update_layout(
        template=template_name,
        title="County MAPE by Experiment",
        height=max(600, len(y_labels) * 16 + 100),
        yaxis={"autorange": "reversed", "dtick": 1, "tickfont": {"size": 9}},
        xaxis={"side": "top", "tickfont": {"size": 11}},
        margin={"l": 200},
    )

    return fig


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def _outcome_badge(outcome: str) -> str:
    """Return HTML badge for an experiment outcome."""
    colors = {
        "passed_all_gates": ("#059669", "#ecfdf5"),
        "needs_human_review": ("#d97706", "#fffbeb"),
        "failed_hard_gate": ("#dc2626", "#fef2f2"),
        "inconclusive": ("#6b7280", "#f3f4f6"),
    }
    fg, bg = colors.get(outcome, ("#94a3b8", "#f1f5f9"))
    label = outcome.replace("_", " ").title() if outcome else "Not Tested"
    return f'<span class="badge" style="background:{bg};color:{fg};border:1px solid {fg}">{label}</span>'


def _tier_badge(tier: int | None) -> str:
    tier_colors = {1: "#0563C1", 2: "#FFC000", 3: "#ED7D31"}
    color = tier_colors.get(tier, "#94a3b8")
    return f'<span class="badge" style="background:{color};color:white">Tier {tier}</span>'


def _delta_cell(value: float | None, threshold: float | None = None) -> str:
    """Format a delta value as a colored table cell."""
    if value is None:
        return '<td class="delta-cell">-</td>'
    color = "#059669" if value < 0 else "#dc2626"
    bold = ""
    if threshold is not None and value > threshold:
        bold = "font-weight:700;"
    return f'<td class="delta-cell" style="color:{color};{bold}">{value:+.4f}</td>'


def _build_tracker_tab(data: dict) -> str:
    """Build Tab 1: Experiment Tracker HTML."""
    catalog = data["catalog"]
    outcomes = data["exp_outcomes"]
    comparisons = data["comparisons"]

    # KPI counts
    total = len(catalog)
    completed = len(outcomes)
    passed = sum(
        1
        for o in outcomes.values()
        if o.get("outcome") == "passed_all_gates"
        # Also count EXP-B which is logged as failed_hard_gate but reclassified
    )
    # Check catalog results_text for reclassified entries
    for cat in catalog:
        exp_id = cat["exp_id"]
        if (
            exp_id in outcomes
            and "passed_all_gates" in cat.get("results_text", "")
            and outcomes[exp_id].get("outcome") != "passed_all_gates"
        ):
            passed = max(passed, 1)  # At least EXP-B
    blocked = sum(1 for c in catalog if not c["config_only"])

    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{total}</div>
            <div class="kpi-label">Total Experiments</div>
        </div>
        <div class="kpi-card" style="border-top-color:#059669">
            <div class="kpi-value">{completed}</div>
            <div class="kpi-label">Completed</div>
        </div>
        <div class="kpi-card" style="border-top-color:#00B050">
            <div class="kpi-value">{passed}</div>
            <div class="kpi-label">Passed All Gates</div>
        </div>
        <div class="kpi-card" style="border-top-color:#dc2626">
            <div class="kpi-value">{blocked}</div>
            <div class="kpi-label">Blocked (need code)</div>
        </div>
    </div>
    """

    # Tracker table
    rows_html = ""
    for cat in catalog:
        exp_id = cat["exp_id"]
        outcome_data = outcomes.get(exp_id, {})

        # Determine display outcome
        raw_outcome = outcome_data.get("outcome", "")
        display_outcome = raw_outcome
        # Check if catalog says reclassified
        if "passed_all_gates" in cat.get("results_text", "") and raw_outcome:
            display_outcome = "passed_all_gates"

        # Status
        if outcome_data:
            status = "Completed"
        elif not cat["config_only"]:
            status = "Blocked"
        else:
            status = "Pending"

        status_colors = {
            "Completed": ("#059669", "#ecfdf5"),
            "Blocked": ("#dc2626", "#fef2f2"),
            "Pending": ("#94a3b8", "#f1f5f9"),
        }
        sfg, sbg = status_colors[status]

        # Deltas from comparison
        deltas = {}
        if exp_id in comparisons:
            cmp = comparisons[exp_id]
            if cmp.get("challengers"):
                deltas = cmp["challengers"][0].get("deltas", {})

        runnable = "Yes" if cat["config_only"] else "No"
        runnable_color = "#059669" if cat["config_only"] else "#dc2626"

        rows_html += f"""
        <tr>
            <td><strong>{exp_id}</strong></td>
            <td>{cat['name']}</td>
            <td class="mono">{cat.get('parameter', '-')}</td>
            <td>{_tier_badge(cat['tier'])}</td>
            <td style="color:{runnable_color};font-weight:600">{runnable}</td>
            <td><span class="badge" style="background:{sbg};color:{sfg};border:1px solid {sfg}">{status}</span></td>
            <td>{_outcome_badge(display_outcome) if display_outcome else _outcome_badge('')}</td>
            {_delta_cell(deltas.get('county_mape_overall'), 0.05)}
            {_delta_cell(deltas.get('county_mape_rural'), 0.10)}
            {_delta_cell(deltas.get('county_mape_bakken'), 0.50)}
            {_delta_cell(deltas.get('county_mape_urban_college'))}
        </tr>
        """

    table_html = f"""
    <table class="data-table sortable">
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Parameter</th>
                <th>Tier</th>
                <th>Runnable</th>
                <th>Status</th>
                <th>Outcome</th>
                <th>Overall &Delta;</th>
                <th>Rural &Delta;</th>
                <th>Bakken &Delta;</th>
                <th>College &Delta;</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """

    return kpi_html + table_html


def _build_scorecard_tab(data: dict) -> str:
    """Build Tab 3: Scorecard Comparison table."""
    champion = data["champion_scorecard"]
    if not champion:
        return "<p>No champion scorecard data available.</p>"

    completed_ids = sorted(data["scorecards"].keys())

    # Build rows for each metric
    all_metrics = [
        ("state_ape_recent_short", "State APE (1-5yr)"),
        ("state_ape_recent_medium", "State APE (6-10yr)"),
        ("state_signed_bias_recent", "State Signed Bias"),
        ("county_mape_overall", "County MAPE Overall"),
        ("county_mape_urban_college", "County MAPE College"),
        ("county_mape_rural", "County MAPE Rural"),
        ("county_mape_bakken", "County MAPE Bakken"),
    ]
    all_metrics.extend(list(SENTINEL_LABELS.items()))

    # Header row
    header = "<tr><th>Metric</th><th>Champion (m2026)</th>"
    for exp_id in completed_ids:
        color = EXPERIMENT_COLORS.get(exp_id, "#888")
        header += f'<th style="border-bottom:3px solid {color}">{exp_id}</th>'
    header += "</tr>"

    rows = ""
    for metric_key, metric_label in all_metrics:
        champ_val = champion.get(metric_key)
        champ_str = f"{champ_val:.3f}" if champ_val is not None else "-"
        row = f"<tr><td><strong>{metric_label}</strong></td><td>{champ_str}</td>"

        for exp_id in completed_ids:
            # Find challenger scorecard entry
            sc = data["scorecards"].get(exp_id, [])
            challenger_entry = None
            for entry in sc:
                if entry.get("status_at_run") != "champion":
                    challenger_entry = entry
                    break

            if challenger_entry and metric_key in challenger_entry:
                val = challenger_entry[metric_key]
                if champ_val is not None and isinstance(val, (int, float)):
                    delta = val - champ_val
                    # For bias, closer to 0 is better
                    if metric_key == "state_signed_bias_recent":
                        is_better = abs(val) < abs(champ_val)
                    else:
                        is_better = delta < 0
                    color = "#059669" if is_better else "#dc2626"
                    row += (
                        f'<td style="color:{color}">'
                        f"{val:.3f} "
                        f'<small>({delta:+.3f})</small></td>'
                    )
                else:
                    row += f"<td>{val}</td>"
            else:
                row += "<td>-</td>"

        row += "</tr>"
        rows += row

    return f"""
    <table class="data-table sortable">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------
_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; font-size: 15px; }
body {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #333; background: #f8f9fa; line-height: 1.6;
}

/* Header */
header {
    background: linear-gradient(135deg, #1F3864 0%, #0563C1 100%);
    color: white; padding: 1.5rem 2rem;
}
header h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.subtitle { font-size: 0.85rem; opacity: 0.85; }

/* Nav */
nav {
    background: white; border-bottom: 1px solid #ddd;
    position: sticky; top: 0; z-index: 99;
    padding: 0 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
nav ul { list-style: none; display: flex; gap: 0; max-width: 1200px; margin: 0 auto; overflow-x: auto; }
nav a {
    display: block; padding: 0.75rem 1.2rem; color: #595959;
    text-decoration: none; font-size: 0.85rem; font-weight: 500;
    border-bottom: 3px solid transparent; white-space: nowrap;
    transition: color 0.15s, border-color 0.15s;
}
nav a:hover { color: #0563C1; }
nav a.active { color: #0563C1; border-bottom-color: #0563C1; font-weight: 600; }

/* Main */
main { max-width: 1200px; margin: 1.5rem auto; padding: 0 1.5rem; }
section { display: none; }
section.active { display: block; }
section h2 {
    font-size: 1.25rem; color: #1F3864; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid #0563C1;
}

/* KPI cards */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: white; border-radius: 6px; padding: 1.2rem;
    border-top: 4px solid #0563C1; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #1F3864; }
.kpi-label { font-size: 0.8rem; color: #595959; margin-top: 0.2rem; }

/* Tables */
.data-table {
    width: 100%; border-collapse: collapse; background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 6px;
    overflow: hidden; font-size: 0.85rem; margin-bottom: 1.5rem;
}
.data-table thead { background: #1F3864; color: white; }
.data-table th {
    padding: 0.6rem 0.8rem; text-align: left; font-weight: 600;
    font-size: 0.8rem; cursor: pointer; white-space: nowrap;
    user-select: none;
}
.data-table th:hover { background: rgba(255,255,255,0.1); }
.data-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }
.data-table tbody tr:hover { background: #f8f9fa; }
.data-table .mono { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.78rem; }

/* Badges */
.badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
    font-size: 0.72rem; font-weight: 600; white-space: nowrap;
}

/* Delta cells */
.delta-cell { font-family: 'Consolas', monospace; font-size: 0.8rem; text-align: right; }

/* Chart container */
.chart-container {
    background: white; border-radius: 6px; padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem;
}

/* Footer */
footer {
    text-align: center; padding: 1rem; font-size: 0.75rem; color: #999;
    border-top: 1px solid #eee; margin-top: 2rem;
}

/* Print */
@media print {
    nav { position: static; }
    section { display: block !important; page-break-inside: avoid; }
    .chart-container { box-shadow: none; border: 1px solid #ddd; }
}
"""

_JS = """
document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    const links = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('main > section');

    function activateTab(id) {
        sections.forEach(s => s.classList.remove('active'));
        links.forEach(l => l.classList.remove('active'));
        const target = document.getElementById(id);
        if (target) target.classList.add('active');
        const link = document.querySelector('nav a[data-tab="' + id + '"]');
        if (link) link.classList.add('active');
    }

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            activateTab(this.getAttribute('data-tab'));
        });
    });

    // Table sorting
    document.querySelectorAll('.sortable th').forEach(function(th) {
        th.addEventListener('click', function() {
            const table = th.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const idx = Array.from(th.parentNode.children).indexOf(th);
            const asc = th.dataset.sort !== 'asc';
            th.dataset.sort = asc ? 'asc' : 'desc';
            rows.sort(function(a, b) {
                let va = a.children[idx] ? a.children[idx].textContent.trim() : '';
                let vb = b.children[idx] ? b.children[idx].textContent.trim() : '';
                let na = parseFloat(va.replace(/[^0-9.+-]/g, ''));
                let nb = parseFloat(vb.replace(/[^0-9.+-]/g, ''));
                if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
                return asc ? va.localeCompare(vb) : vb.localeCompare(va);
            });
            rows.forEach(r => tbody.appendChild(r));
        });
    });
});
"""


def build_html(
    data: dict[str, object],
    use_cdn_plotly: bool = False,
) -> str:
    """Assemble the full dashboard HTML.

    Builds all Plotly figures, renders tab content, and wraps everything
    in a self-contained HTML document with embedded CSS, JS, and Plotly.

    Args:
        data: Dashboard data dict from :func:`load_dashboard_data`.
        use_cdn_plotly: If True, load Plotly.js from CDN instead of
            bundling inline (smaller file, requires internet).

    Returns:
        Complete HTML string ready to write to disk.
    """
    template_name = _setup_plotly()

    # Build Plotly figures
    spaghetti_fig = build_spaghetti_fig(data, template_name)
    delta_bar_fig = build_delta_bar_fig(data, template_name)
    horizon_fig = build_horizon_fig(data, template_name)
    county_fig = build_county_heatmap_fig(data, template_name)

    # Convert figures to HTML divs
    plotly_config = {"displayModeBar": True, "responsive": True}
    spaghetti_html = pio.to_html(
        spaghetti_fig, full_html=False, include_plotlyjs=False, config=plotly_config
    )
    delta_bar_html = pio.to_html(
        delta_bar_fig, full_html=False, include_plotlyjs=False, config=plotly_config
    )
    horizon_html = pio.to_html(
        horizon_fig, full_html=False, include_plotlyjs=False, config=plotly_config
    )
    county_html = pio.to_html(
        county_fig, full_html=False, include_plotlyjs=False, config=plotly_config
    )

    # Tab content
    tracker_content = _build_tracker_tab(data)
    scorecard_content = _build_scorecard_tab(data)

    # Plotly JS inclusion (always CDN — inline bundling is not worth the file size)
    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Champion info
    champ = data.get("champion_scorecard", {})
    champ_label = f"{champ.get('method_id', '?')} / {champ.get('config_id', '?')}" if champ else "unknown"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ND Projection Experiment Dashboard</title>
    {plotly_js}
    <style>{_CSS}</style>
</head>
<body>

<header>
    <h1>Experiment Dashboard</h1>
    <div class="subtitle">North Dakota Cohort-Component Projection &mdash; A/B Testing Overview</div>
    <div class="subtitle" style="margin-top:0.3rem;opacity:0.7">Champion: {champ_label} &bull; Generated: {now}</div>
</header>

<nav>
    <ul>
        <li><a href="#" data-tab="tracker" class="active">Experiment Tracker</a></li>
        <li><a href="#" data-tab="spaghetti">Spaghetti Plot</a></li>
        <li><a href="#" data-tab="scorecard">Scorecard Comparison</a></li>
        <li><a href="#" data-tab="horizon">Horizon Analysis</a></li>
        <li><a href="#" data-tab="county">County Detail</a></li>
    </ul>
</nav>

<main>

    <section id="tracker" class="active">
        <h2>Experiment Tracker</h2>
        <p style="margin-bottom:1rem;color:#595959;font-size:0.85rem">
            All catalog experiments with current status. Delta columns show
            change in MAPE vs champion (negative = improvement, bold red = exceeds gate threshold).
        </p>
        {tracker_content}
    </section>

    <section id="spaghetti">
        <h2>Projection Ensemble (Spaghetti Plot)</h2>
        <p style="margin-bottom:1rem;color:#595959;font-size:0.85rem">
            Overlaid state-level projection curves from all tested experiments.
            Use the origin year dropdown to filter by validation start year, or
            view all origins for the full ensemble spread.
        </p>
        <div class="chart-container">
            {spaghetti_html}
        </div>
    </section>

    <section id="scorecard">
        <h2>Scorecard Comparison</h2>
        <p style="margin-bottom:1rem;color:#595959;font-size:0.85rem">
            MAPE deltas vs champion across key county categories. Gate thresholds
            shown as dotted red lines.
        </p>
        <div class="chart-container">
            {delta_bar_html}
        </div>
        <h3 style="margin:1rem 0 0.5rem;color:#1F3864">Full Metrics Table</h3>
        <p style="margin-bottom:0.5rem;color:#595959;font-size:0.8rem">
            Absolute values with delta from champion in parentheses. Green = improvement, red = regression.
        </p>
        {scorecard_content}
    </section>

    <section id="horizon">
        <h2>Horizon Analysis</h2>
        <p style="margin-bottom:1rem;color:#595959;font-size:0.85rem">
            How projection accuracy degrades with forecast distance. Top panel shows
            mean absolute error (MAPE); bottom panel shows directional bias (MPE).
        </p>
        <div class="chart-container">
            {horizon_html}
        </div>
    </section>

    <section id="county">
        <h2>County Detail</h2>
        <p style="margin-bottom:1rem;color:#595959;font-size:0.85rem">
            Mean MAPE by county and experiment, averaged across all origin years and
            horizons. Counties grouped by category (Bakken, Urban/College, Reservation, Rural).
        </p>
        <div class="chart-container">
            {county_html}
        </div>
    </section>

</main>

<footer>
    ND Population Projections &mdash; Experiment Dashboard &bull; Generated {now}
</footer>

<script>{_JS}</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML dashboard comparing experiment results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the HTML file (default: data/analysis/experiments/experiment_dashboard.html)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to the experiment catalog markdown file.",
    )
    parser.add_argument(
        "--no-plotly-js",
        action="store_true",
        help="Use CDN Plotly.js instead of bundling (smaller file, requires internet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading experiment data...")
    data = load_dashboard_data(catalog_path=args.catalog)

    print(
        f"  Catalog: {len(data['catalog'])} experiments, "
        f"{len(data['exp_outcomes'])} completed"
    )
    print(f"  Scorecards loaded: {len(data['scorecards'])}")
    print(f"  Projection curves loaded: {len(data['projection_curves'])}")

    print("Building dashboard...")
    html = build_html(data, use_cdn_plotly=args.no_plotly_js)

    output_path = args.output or (DEFAULT_OUTPUT_DIR / "experiment_dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"Dashboard written to {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
