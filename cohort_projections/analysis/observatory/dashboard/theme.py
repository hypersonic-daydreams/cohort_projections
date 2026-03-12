"""SDC-branded theme constants and helpers for the Panel observatory dashboard.

Ports the color palette from ``scripts/exports/_report_theme.py`` and adds
Panel-specific CSS, status badge palettes, and Plotly layout helpers used by
all dashboard tabs.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# SDC Brand Colors (mirrored from _report_theme.py)
# ---------------------------------------------------------------------------
SDC_NAVY = "#1F3864"
SDC_BLUE = "#0563C1"
SDC_TEAL = "#00B0F0"
SDC_RED = "#C00000"
SDC_WHITE = "#FFFFFF"
SDC_LIGHT_GRAY = "#F2F2F2"
SDC_MID_GRAY = "#D9D9D9"
SDC_DARK_GRAY = "#595959"

# Supplementary accent colors
GROWTH_GREEN = "#00B050"
GOLD = "#FFC000"
ORANGE = "#ED7D31"
PURPLE = "#7030A0"

# ---------------------------------------------------------------------------
# Font Stack
# ---------------------------------------------------------------------------
FONT_FAMILY = "'Aptos', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ---------------------------------------------------------------------------
# Status badge colors
# ---------------------------------------------------------------------------
STATUS_COLORS: dict[str, str] = {
    "passed_all_gates": "#00B050",
    "needs_human_review": "#FFC000",
    "failed_hard_gate": "#C00000",
    "untested": "#A0A0A0",
    "champion": "#0563C1",
}

# ---------------------------------------------------------------------------
# County category colors
# ---------------------------------------------------------------------------
CATEGORY_COLORS: dict[str, str] = {
    "Rural": "#0563C1",
    "Bakken": "#ED7D31",
    "Urban/College": "#7030A0",
    "Reservation": "#00B050",
}

# ---------------------------------------------------------------------------
# Experiment line colors (distinguishable palette for up to 15 series)
# ---------------------------------------------------------------------------
EXPERIMENT_COLORS: list[str] = [
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

# ---------------------------------------------------------------------------
# Panel Dashboard CSS
# ---------------------------------------------------------------------------
DASHBOARD_CSS = """\
/* --- Observatory Dashboard Theme --- */

/* Header */
:host(.pn-header), header.pn-header {
    background-color: #1F3864 !important;
    color: #FFFFFF !important;
}

/* Sidebar */
nav.pn-sidebar, :host(.pn-sidebar) {
    background-color: #1F3864 !important;
    color: #FFFFFF !important;
}
nav.pn-sidebar .bk-btn, nav.pn-sidebar label {
    color: #FFFFFF !important;
}

/* Main content area */
:host(.pn-main), .pn-main {
    background: linear-gradient(180deg, #EDF3FB 0%, #F6F8FC 100%) !important;
}

/* Card styling */
.card-container, .bk-Card {
    background-color: #FFFFFF;
    border: 1px solid #D9E3F0;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(31, 56, 100, 0.06);
    padding: 14px;
    margin-bottom: 14px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.82em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.badge-passed   { background-color: #00B050; color: #FFFFFF; }
.badge-review   { background-color: #FFC000; color: #1F3864; }
.badge-failed   { background-color: #C00000; color: #FFFFFF; }
.badge-untested { background-color: #A0A0A0; color: #FFFFFF; }
.badge-champion { background-color: #0563C1; color: #FFFFFF; }

/* KPI card */
.kpi-card {
    text-align: center;
    padding: 16px 12px;
    min-width: 160px;
    border-radius: 10px;
    background: #FFFFFF;
    border: 1px solid #D9E3F0;
    box-shadow: 0 6px 20px rgba(31, 56, 100, 0.06);
}
.kpi-card .kpi-value {
    font-size: 2.0em;
    font-weight: 700;
    line-height: 1.1;
    color: #1F3864;
}
.kpi-card .kpi-label {
    font-size: 0.85em;
    color: #595959;
    margin-top: 4px;
}
.kpi-card .kpi-delta {
    font-size: 0.9em;
    font-weight: 600;
    margin-top: 2px;
}
.kpi-delta.positive { color: #C00000; }
.kpi-delta.negative { color: #00B050; }
.kpi-delta.neutral  { color: #595959; }

/* Section headers */
.section-header h2 {
    margin: 0;
    color: #1F3864;
    font-size: 1.25em;
    font-weight: 700;
}
.section-header .subtitle {
    margin: 2px 0 0 0;
    color: #595959;
    font-size: 0.88em;
    font-weight: 400;
}

.summary-card {
    min-width: 220px;
    padding: 16px 18px;
    border-radius: 12px;
    border: 1px solid #D9E3F0;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
    box-shadow: 0 6px 18px rgba(31, 56, 100, 0.06);
}
.summary-card .eyebrow {
    color: #5A6C84;
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.summary-card .headline {
    margin-top: 6px;
    color: #1F3864;
    font-size: 1.2em;
    font-weight: 700;
    line-height: 1.25;
}
.summary-card .detail {
    margin-top: 8px;
    color: #4F5F74;
    font-size: 0.9em;
    line-height: 1.4;
}
.summary-card.primary {
    background: linear-gradient(180deg, #F8FBFF 0%, #E9F1FC 100%);
}
.summary-card.warning {
    background: linear-gradient(180deg, #FFF9E8 0%, #FFF3CC 100%);
}
.summary-card.success {
    background: linear-gradient(180deg, #F3FBF6 0%, #E2F3E8 100%);
}

.filters-help {
    color: #5A6C84;
    font-size: 0.88em;
    margin-bottom: 6px;
}

/* Empty placeholder */
.empty-placeholder {
    text-align: center;
    padding: 40px 20px;
    color: #A0A0A0;
    font-style: italic;
    font-size: 0.95em;
}

/* Tabulator tweaks */
.tabulator .tabulator-header {
    background-color: #1F3864 !important;
    color: #FFFFFF !important;
}
.tabulator .tabulator-header .tabulator-col {
    background-color: #1F3864 !important;
    color: #FFFFFF !important;
}

@media (max-width: 700px) {
    .bk-Card {
        padding: 10px;
    }
    .kpi-card {
        min-width: calc(50% - 12px);
    }
    .summary-card {
        min-width: 100%;
    }
    .section-header h2 {
        font-size: 1.1em;
    }
}
"""

TABS_STYLESHEET = """
:host {
    overflow: visible;
}

.bk-header {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0 0 4px 0;
    scrollbar-width: thin;
}

.bk-tab {
    white-space: nowrap;
    min-width: max-content;
    padding: 10px 14px;
    border-radius: 999px 999px 0 0;
    font-weight: 600;
    color: #41566F;
}

.bk-tab.bk-active {
    color: #1F3864;
    background: linear-gradient(180deg, #FFFFFF 0%, #EDF3FB 100%);
}

@media (max-width: 700px) {
    .bk-header {
        gap: 8px;
        padding-bottom: 8px;
    }

    .bk-tab {
        font-size: 12px;
        padding: 10px 16px;
    }
}
"""

TABULATOR_STYLESHEET = """
.tabulator {
    background: #FFFFFF;
    border: 1px solid #D9E3F0;
    border-radius: 10px;
    overflow: hidden;
    font-size: 12px;
}

.tabulator .tabulator-header {
    background: linear-gradient(180deg, #34588E 0%, #1F3864 100%) !important;
}

.tabulator-row {
    background: #FFFFFF;
}

.tabulator-row:nth-child(even) {
    background: #F8FBFF;
}

.tabulator-row:hover {
    background: #EAF2FD !important;
}

.tabulator-cell {
    border-right: 1px solid #E6EEF8 !important;
}

.tabulator-footer {
    background: #F8FBFF;
    border-top: 1px solid #D9E3F0;
}
"""


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def get_plotly_layout_defaults() -> dict[str, Any]:
    """Return a dict of Plotly layout kwargs for consistent chart styling.

    Use via ``fig.update_layout(**get_plotly_layout_defaults())``.
    """
    return {
        "font": {"family": FONT_FAMILY, "size": 12, "color": SDC_DARK_GRAY},
        "title_font": {"family": FONT_FAMILY, "size": 16, "color": SDC_NAVY},
        "plot_bgcolor": SDC_WHITE,
        "paper_bgcolor": SDC_WHITE,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "xaxis": {
            "showgrid": True,
            "gridcolor": SDC_LIGHT_GRAY,
            "linecolor": SDC_MID_GRAY,
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": SDC_LIGHT_GRAY,
            "linecolor": SDC_MID_GRAY,
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": SDC_MID_GRAY,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "hoverlabel": {
            "bgcolor": SDC_WHITE,
            "font_size": 12,
            "font_family": FONT_FAMILY,
        },
        "colorway": EXPERIMENT_COLORS,
    }


def get_plotly_template() -> go.layout.Template:
    """Return an SDC-branded Plotly template for observatory charts.

    Mirrors the template from ``_report_theme.py`` but uses the dashboard
    color-way so that experiment lines get distinct colors by default.
    """
    template = go.layout.Template()
    template.layout = go.Layout(**get_plotly_layout_defaults())
    return template
