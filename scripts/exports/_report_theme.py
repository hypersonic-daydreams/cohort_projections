"""
Shared theme constants for the ND Population Projections interactive HTML report.

Defines color palettes, font stacks, Plotly layout templates, and formatting
helpers used by all report section builders.

References:
    SDC brand guide (Navy #1F3864, Blue #0563C1, Teal #00B0F0)
    ADR-037: CBO-grounded scenario methodology (scenario naming)
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# SDC Brand Colors
# ---------------------------------------------------------------------------
NAVY = "#1F3864"
BLUE = "#0563C1"
TEAL = "#00B0F0"
LIGHT_GRAY = "#F2F2F2"
MID_GRAY = "#D9D9D9"
DARK_GRAY = "#595959"
WHITE = "#FFFFFF"
RED_ACCENT = "#C00000"

# ---------------------------------------------------------------------------
# Scenario Colors
# ---------------------------------------------------------------------------
SCENARIO_COLORS = {
    "baseline": "#0563C1",
    "high_growth": "#00B050",
    "restricted_growth": "#FF0000",
}

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "high_growth": "High Growth",
    "restricted_growth": "Restricted Growth",
}

# Tier colors
TIER_COLORS = {
    "HIGH": "#0563C1",
    "MODERATE": "#FFC000",
    "LOWER": "#ED7D31",
}

# Growth / decline colors
GROWTH_COLOR = "#00B050"
DECLINE_COLOR = "#FF0000"
NEUTRAL_COLOR = "#808080"

# Sex colors for population pyramid
SEX_COLORS = {
    "Male": "#0563C1",
    "Female": "#C00000",
}

# Age structure colors
AGE_STRUCTURE_COLORS = {
    "Youth (0-14)": "#00B0F0",
    "Working Age (15-64)": "#0563C1",
    "Elderly (65+)": "#1F3864",
}

# ---------------------------------------------------------------------------
# Font Stack
# ---------------------------------------------------------------------------
FONT_FAMILY = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


# ---------------------------------------------------------------------------
# Plotly Layout Template
# ---------------------------------------------------------------------------
def get_plotly_template() -> go.layout.Template:
    """Return a consistent Plotly template for all report charts."""
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
        colorway=[BLUE, TEAL, NAVY, "#ED7D31", "#00B050", "#FFC000", "#C00000"],
    )
    return template


def register_template() -> str:
    """Register the SDC template with Plotly and return its name."""
    template_name = "sdc_report"
    pio.templates[template_name] = get_plotly_template()
    return template_name


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------
def format_population(n: float | int) -> str:
    """Format a population number with commas.  e.g., 799,358."""
    return f"{int(round(n)):,}"


def format_percent(p: float, decimals: int = 1) -> str:
    """Format a decimal ratio as a percentage string.  e.g., 0.104 -> '+10.4%'."""
    pct = p * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def format_change(n: float | int) -> str:
    """Format an absolute population change with +/- sign and commas."""
    val = int(round(n))
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,}"


def get_scenario_color(name: str) -> str:
    """Return the brand color for a scenario key."""
    return SCENARIO_COLORS.get(name, NEUTRAL_COLOR)


def get_scenario_label(name: str) -> str:
    """Return the display label for a scenario key."""
    return SCENARIO_LABELS.get(name, name.replace("_", " ").title())


def growth_color(rate: float) -> str:
    """Return green for growth, red for decline."""
    if rate > 0.001:
        return GROWTH_COLOR
    elif rate < -0.001:
        return DECLINE_COLOR
    return NEUTRAL_COLOR
