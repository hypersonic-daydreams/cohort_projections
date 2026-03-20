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

# Extended accent palette
SDC_BLUE_LIGHT = "#EBF3FE"
SDC_BLUE_DARK = "#044B8F"

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
# Dashboard layout modes
# ---------------------------------------------------------------------------
LAYOUT_STANDARD = "standard"
LAYOUT_PORTRAIT = "portrait"
LAYOUT_AUTO = "auto"


def resolve_layout_mode(
    viewport_width: int | None,
    viewport_height: int | None,
    *,
    default: str = LAYOUT_STANDARD,
) -> str:
    """Resolve a viewport into a dashboard layout mode.

    The Observatory relies on CSS media queries for live browser behaviour,
    but tests and static rendering still need a deterministic Python helper.
    """
    if viewport_width is None or viewport_height is None:
        return default
    if viewport_width <= 0 or viewport_height <= 0:
        return default
    return LAYOUT_PORTRAIT if viewport_height > viewport_width else LAYOUT_STANDARD


def layout_mode_classes(
    *base_classes: str,
    layout_mode: str = LAYOUT_AUTO,
) -> list[str]:
    """Return standard CSS classes for layout-aware dashboard sections."""
    normalized = (
        layout_mode
        if layout_mode
        in {
            LAYOUT_STANDARD,
            LAYOUT_PORTRAIT,
            LAYOUT_AUTO,
        }
        else LAYOUT_AUTO
    )
    classes = [cls for cls in base_classes if cls]
    classes.extend(["obs-layout-managed", f"obs-layout-{normalized}"])
    return classes


# ---------------------------------------------------------------------------
# Panel Dashboard CSS
# ---------------------------------------------------------------------------
DASHBOARD_CSS = """\
/* --- Observatory Dashboard Theme — Bold Editorial --- */

/* Noise texture SVG (inline, used as background overlay) */

/* Header — dramatic gradient with glow accent */
:host(.pn-header), header.pn-header {
    background:
        radial-gradient(ellipse at 30% 50%, rgba(5,99,193,0.15) 0%, transparent 60%),
        linear-gradient(135deg, #080F1A 0%, #0D1B2A 30%, #1B2D4A 70%, #1F3864 100%) !important;
    color: #FFFFFF !important;
    border-bottom: 1px solid rgba(5, 99, 193, 0.3) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.05) inset,
        0 4px 24px rgba(8, 15, 26, 0.4) !important;
}

/* Sidebar — dark with subtle texture */
nav.pn-sidebar, :host(.pn-sidebar) {
    background: linear-gradient(180deg, #0D1B2A 0%, #152238 100%) !important;
    color: #FFFFFF !important;
}
nav.pn-sidebar .bk-btn, nav.pn-sidebar label {
    color: #FFFFFF !important;
}

/* Main content — atmospheric orbs on cool blue-gray */
:host(.pn-main), .pn-main {
    background-color: #E2E7EE !important;
    background-image:
        linear-gradient(180deg, rgba(5,99,193,0.04) 0%, transparent 120px),
        radial-gradient(circle at 8% 5%, rgba(5,99,193,0.06) 0%, transparent 50%),
        radial-gradient(circle at 88% 90%, rgba(31,56,100,0.045) 0%, transparent 35%),
        radial-gradient(circle at 75% 35%, rgba(0,128,128,0.035) 0%, transparent 22%),
        radial-gradient(circle, rgba(31,56,100,0.045) 1px, transparent 1px) !important;
    background-size: auto, auto, auto, auto, 22px 22px !important;
}

/* Soft inner vignette — adds depth to the viewport edges */
:host(.pn-main)::after, .pn-main::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 50% 50%, transparent 55%, rgba(31,56,100,0.03) 100%);
    pointer-events: none;
    z-index: 0;
}

/* --- Three-Tier Elevation System --- */
.card-container, .bk-Card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #C4D0DE;
    border-radius: 10px;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8) inset,
        0 1px 3px rgba(31, 56, 100, 0.08),
        0 6px 24px rgba(31, 56, 100, 0.10),
        0 12px 48px rgba(31, 56, 100, 0.04);
    padding: 18px;
    margin-bottom: 16px;
}
.obs-elevation-1 {
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.05) !important;
    border-color: #E4E9F0 !important;
}
.obs-elevation-2 {
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.04),
        0 4px 16px rgba(31, 56, 100, 0.06) !important;
    border-color: #DDE4ED !important;
}
.obs-elevation-3 {
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 8px 32px rgba(31, 56, 100, 0.10),
        0 20px 48px rgba(31, 56, 100, 0.04) !important;
    border-color: #C4D0DE !important;
}

/* --- Typography Scale (editorial: high contrast, tight tracking) --- */
.obs-text-display { font-size: 3.2em; font-weight: 800; letter-spacing: -0.03em; line-height: 0.95; color: #0D1B2A; }
.obs-text-headline { font-size: 1.6em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; color: #0D1B2A; }
.obs-text-title { font-size: 1.15em; font-weight: 700; line-height: 1.3; color: #1F3864; }
.obs-text-body { font-size: 0.92em; font-weight: 400; line-height: 1.6; color: #3D4F63; }
.obs-text-caption { font-size: 0.78em; font-weight: 600; color: #6B7D93; letter-spacing: 0.03em; }
.obs-text-eyebrow { font-size: 0.7em; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #0563C1; }

/* Status badges — bolder, with gradient backgrounds */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.72em;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-passed   { background: linear-gradient(135deg, #00B050, #009940); color: #FFFFFF; }
.badge-review   { background: linear-gradient(135deg, #FFC000, #E6AC00); color: #1F3864; }
.badge-failed   { background: linear-gradient(135deg, #C00000, #A00000); color: #FFFFFF; }
.badge-untested { background: linear-gradient(135deg, #8090A0, #6B7D93); color: #FFFFFF; }
.badge-champion { background: linear-gradient(135deg, #0563C1, #044B8F); color: #FFFFFF; }

/* KPI card — editorial number display */
.kpi-card {
    text-align: center;
    padding: 22px 16px 18px;
    min-width: 150px;
    border-radius: 10px;
    background: #FFFFFF;
    border: 1px solid #DDE4ED;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.04),
        0 4px 16px rgba(31, 56, 100, 0.06);
    transition: box-shadow 0.25s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0563C1, #00B0F0);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.kpi-card:hover {
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 8px 28px rgba(31, 56, 100, 0.10);
    transform: translateY(-2px);
}
.kpi-card:hover::before {
    opacity: 1;
}
.kpi-card .kpi-value {
    font-size: 2.2em;
    font-weight: 800;
    line-height: 1.0;
    color: #0D1B2A;
    letter-spacing: -0.02em;
}
.kpi-card .kpi-label {
    font-size: 0.72em;
    font-weight: 700;
    color: #6B7D93;
    margin-top: 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.kpi-card .kpi-delta {
    font-size: 0.85em;
    font-weight: 800;
    margin-top: 6px;
}
.kpi-delta.positive { color: #C00000; }
.kpi-delta.negative { color: #00803C; }
.kpi-delta.neutral  { color: #6B7D93; }

/* KPI ghost state — visible accent bar + colored tint */
.kpi-card-ghost {
    position: relative;
    overflow: hidden;
    border-color: transparent !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.6) inset,
        0 2px 8px rgba(31, 56, 100, 0.08),
        0 8px 24px rgba(31, 56, 100, 0.06) !important;
}
.kpi-card-ghost .kpi-card-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 10px 10px 0 0;
    opacity: 1;
}
.kpi-card-ghost:hover {
    transform: translateY(-3px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.6) inset,
        0 4px 12px rgba(31, 56, 100, 0.10),
        0 12px 32px rgba(31, 56, 100, 0.08) !important;
}
.kpi-card-ghost:hover .kpi-card-accent {
    height: 5px;
}

/* KPI responsive grid */
.obs-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
}

/* Section headers — bold editorial with accent underline */
.section-header {
    position: relative;
    padding-bottom: 12px;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #0563C1, #00B0F0 40%, transparent 100%);
    border-radius: 2px;
}
.section-header h2 {
    margin: 0;
    color: #0D1B2A;
    font-size: 1.6em;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.section-header .subtitle {
    margin: 6px 0 0 0;
    color: #6B7D93;
    font-size: 0.88em;
    font-weight: 400;
    line-height: 1.5;
}

/* Summary card — refined with gradient border accent */
.summary-card {
    min-width: 220px;
    padding: 20px 22px;
    border-radius: 10px;
    border: 1px solid #DDE4ED;
    background: #FFFFFF;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.04),
        0 4px 16px rgba(31, 56, 100, 0.06);
    position: relative;
}
.summary-card .eyebrow {
    color: #FFFFFF;
    font-size: 0.7em;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #1F3864, #0563C1);
    margin: -20px -22px 14px -22px;
    padding: 10px 22px;
    border-radius: 10px 10px 0 0;
}
.summary-card.primary .eyebrow {
    background: linear-gradient(135deg, #0563C1, #044B8F);
}
.summary-card.warning .eyebrow {
    background: linear-gradient(135deg, #BF8F00, #E6AC00);
    color: #1F3864;
}
.summary-card.success .eyebrow {
    background: linear-gradient(135deg, #00803C, #00B050);
}
.summary-card .headline {
    margin-top: 8px;
    color: #0D1B2A;
    font-size: 1.3em;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.summary-card .detail {
    margin-top: 10px;
    color: #3D4F63;
    font-size: 0.9em;
    line-height: 1.55;
}
.summary-card.primary {
    background:
        radial-gradient(ellipse at 100% 0%, rgba(5,99,193,0.06) 0%, transparent 50%),
        linear-gradient(180deg, #FFFFFF 0%, #F0F4FA 100%);
    border-color: #B8C8DB;
    border-top: 3px solid #0563C1;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 8px 32px rgba(31, 56, 100, 0.10),
        0 0 0 1px rgba(5,99,193,0.04) inset;
}
.summary-card.warning {
    background: linear-gradient(180deg, #FFFFFF 0%, #FFF8EC 100%);
    border-color: #E8D49E;
    border-top: 3px solid #E6AC00;
}
.summary-card.success {
    background: linear-gradient(180deg, #FFFFFF 0%, #F2FAF5 100%);
    border-color: #A0D4B0;
    border-top: 3px solid #00803C;
}

.filters-help {
    color: #6B7D93;
    font-size: 0.85em;
    margin-bottom: 6px;
}

/* Empty placeholder */
.empty-placeholder {
    text-align: center;
    padding: 56px 24px;
    color: #6B7D93;
    font-style: normal;
    font-size: 0.92em;
    line-height: 1.55;
}

/* --- Tabulator (editorial: strong header, clean grid) --- */
.tabulator .tabulator-header {
    background: #0D1B2A !important;
    color: #FFFFFF !important;
    border-bottom: none !important;
}
.tabulator .tabulator-header .tabulator-col {
    background: transparent !important;
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.72em;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* --- Workflow Stepper (editorial: confident, linear) --- */
.obs-workflow-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 12px 16px;
    margin: 8px 0 4px;
}
.obs-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 4px 16px;
    font-size: 0.78em;
    font-weight: 700;
    color: #8090A0;
    white-space: nowrap;
    letter-spacing: 0.02em;
}
.obs-step .obs-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid #C4D0DE;
    background: #FFFFFF;
    font-size: 0.82em;
    font-weight: 800;
    color: #8090A0;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.obs-step .obs-step-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
}
.obs-step .obs-step-subtitle {
    font-size: 0.68em;
    font-weight: 600;
    color: #8090A0;
    opacity: 0.7;
}
.obs-step.active .obs-step-num {
    border-color: transparent;
    background: linear-gradient(135deg, #0563C1, #044B8F);
    color: #FFFFFF;
    box-shadow:
        0 3px 12px rgba(5, 99, 193, 0.35),
        0 0 20px rgba(5, 99, 193, 0.15);
    animation: obs-step-glow 2.5s ease-in-out infinite;
}
@keyframes obs-step-glow {
    0%, 100% { box-shadow: 0 3px 12px rgba(5, 99, 193, 0.35), 0 0 20px rgba(5, 99, 193, 0.15); }
    50% { box-shadow: 0 3px 16px rgba(5, 99, 193, 0.50), 0 0 30px rgba(5, 99, 193, 0.25); }
}
.obs-step.active {
    color: #0D1B2A;
}
.obs-step.active .obs-step-subtitle {
    color: #3D4F63;
}
.obs-step.completed .obs-step-num {
    border-color: transparent;
    background: linear-gradient(135deg, #00B050, #009940);
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0, 176, 80, 0.25);
}
.obs-step.completed {
    color: #00803C;
}
.obs-step-connector {
    flex: 1;
    height: 2px;
    min-width: 30px;
    max-width: 80px;
    background: #DDE4ED;
    transition: background 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 16px;
}
.obs-step-connector.completed {
    background: linear-gradient(90deg, #00B050, #00803C);
}

/* --- Progress Ring --- */
.obs-progress-ring {
    position: relative;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: conic-gradient(
        var(--ring-color, #0563C1) calc(var(--progress, 0) * 1%),
        #E8EDF4 calc(var(--progress, 0) * 1%)
    );
    transition: --progress 0.5s ease;
    flex-shrink: 0;
}
.obs-progress-ring::after {
    content: '';
    position: absolute;
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: #FFFFFF;
}
.obs-progress-ring .obs-ring-label {
    position: relative;
    z-index: 1;
    text-align: center;
    font-weight: 700;
    color: #1F3864;
    line-height: 1.2;
}
.obs-progress-ring .obs-ring-label .obs-ring-pct {
    font-size: 1.6em;
    display: block;
}
.obs-progress-ring .obs-ring-label .obs-ring-text {
    font-size: 0.72em;
    font-weight: 600;
    color: #595959;
    display: block;
}
.obs-progress-ring.running {
    animation: obs-pulse-ring 2s ease-in-out infinite;
}
@keyframes obs-pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(5, 99, 193, 0.15); }
    50% { box-shadow: 0 0 0 8px rgba(5, 99, 193, 0.05); }
}

/* --- Candidate Feed --- */
.obs-candidate-feed {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.obs-candidate-feed-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #F8FBFF;
    border: 1px solid #E6EEF8;
    font-size: 0.88em;
    animation: obs-slideIn 0.3s ease-out;
}
.obs-candidate-feed-item .obs-cf-name {
    flex: 1;
    font-weight: 600;
    color: #1F3864;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.obs-candidate-feed-item .obs-cf-metric {
    font-weight: 700;
    color: #595959;
}
.obs-candidate-feed-item .obs-cf-delta {
    font-weight: 600;
    font-size: 0.9em;
}
.obs-cf-delta.improved { color: #00B050; }
.obs-cf-delta.regressed { color: #C00000; }
@keyframes obs-slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* --- Completion Banner --- */
.obs-completion-banner {
    padding: 20px 24px;
    border-radius: 12px;
    border: 1px solid #D9E3F0;
}
.obs-completion-banner.success {
    background: linear-gradient(135deg, #F0FAF3 0%, #E2F3E8 100%);
    border-color: #B7E1C4;
}
.obs-completion-banner.mixed {
    background: linear-gradient(135deg, #FFF9E8 0%, #FFF3CC 100%);
    border-color: #FFE08A;
}
.obs-completion-banner.failed {
    background: linear-gradient(135deg, #FDF0F0 0%, #F9E0E0 100%);
    border-color: #E6B0B0;
}
.obs-completion-banner .obs-cb-icon {
    font-size: 2.0em;
    margin-bottom: 8px;
}
.obs-completion-banner .obs-cb-title {
    font-size: 1.15em;
    font-weight: 700;
    color: #1F3864;
    margin-bottom: 6px;
}
.obs-completion-banner .obs-cb-detail {
    font-size: 0.92em;
    color: #4F5F74;
    line-height: 1.5;
    margin-bottom: 14px;
}
.obs-completion-banner .obs-cb-next-steps {
    font-size: 0.9em;
    color: #334E68;
    line-height: 1.6;
}
.obs-completion-banner .obs-cb-next-steps ol {
    margin: 6px 0 0 0;
    padding-left: 20px;
}
.obs-completion-banner .obs-cb-next-steps li {
    margin-bottom: 4px;
}
.obs-completion-banner .obs-cb-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.obs-completion-banner .obs-cb-actions .obs-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: 0.88em;
    font-weight: 600;
    border: none;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.2s ease, transform 0.1s ease;
}
.obs-btn.primary {
    background: #0563C1;
    color: #FFFFFF;
}
.obs-btn.primary:hover {
    background: #044D94;
    transform: translateY(-1px);
}
.obs-btn.secondary {
    background: #E9F1FC;
    color: #1F3864;
}
.obs-btn.secondary:hover {
    background: #D4E4F7;
}

/* --- Two-Column Layout --- */
.obs-two-col {
    display: grid;
    grid-template-columns: 55fr 45fr;
    gap: 20px;
    align-items: start;
}
@media (max-width: 900px) {
    .obs-two-col {
        grid-template-columns: 1fr;
    }
}

/* --- Layout Shell (state-reactive atmospheric background) --- */
.obs-layout-shell {
    gap: 16px;
    position: relative;
}

/* Primary atmospheric orb — large, top-right, shifts color with state */
.obs-layout-shell::before {
    content: '';
    position: fixed;
    top: 40px;
    right: -120px;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(5,99,193,0.045) 0%, rgba(5,99,193,0.02) 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    transition: background 2s ease, opacity 2s ease, transform 2s ease;
    opacity: 0.7;
}

/* Secondary atmospheric orb — smaller, bottom-left, complementary */
.obs-layout-shell::after {
    content: '';
    position: fixed;
    bottom: -80px;
    left: -60px;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(31,56,100,0.04) 0%, rgba(31,56,100,0.015) 45%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    transition: background 2s ease, opacity 2s ease, transform 2s ease;
    opacity: 0.5;
}

/* --- State-specific orb treatments --- */

/* Empty/Ready — neutral, quiet, inviting */
.obs-state-empty-ready::before {
    background: radial-gradient(circle, rgba(5,99,193,0.045) 0%, rgba(5,99,193,0.015) 40%, transparent 70%);
    opacity: 0.6;
}
.obs-state-empty-ready::after {
    background: radial-gradient(circle, rgba(31,56,100,0.035) 0%, transparent 70%);
    opacity: 0.4;
}

/* Search in Progress — teal energy, slightly more vivid */
.obs-state-search-in-progress::before {
    background: radial-gradient(circle, rgba(0,176,240,0.07) 0%, rgba(0,176,240,0.025) 40%, transparent 70%);
    opacity: 1;
    transform: scale(1.05);
}
.obs-state-search-in-progress::after {
    background: radial-gradient(circle, rgba(5,99,193,0.05) 0%, rgba(5,99,193,0.02) 45%, transparent 70%);
    opacity: 0.7;
}

/* Review Ready — confident blue, present but calm */
.obs-state-review-ready::before {
    background: radial-gradient(circle, rgba(5,99,193,0.08) 0%, rgba(5,99,193,0.03) 40%, transparent 70%);
    opacity: 1;
    transform: scale(1.08);
}
.obs-state-review-ready::after {
    background: radial-gradient(circle, rgba(0,176,240,0.04) 0%, rgba(0,176,240,0.015) 45%, transparent 70%);
    opacity: 0.6;
}

/* Recommendation Ready — green forward momentum */
.obs-state-recommendation-ready::before {
    background: radial-gradient(circle, rgba(0,176,80,0.06) 0%, rgba(0,176,80,0.02) 40%, transparent 70%);
    opacity: 1;
    transform: scale(1.1);
}
.obs-state-recommendation-ready::after {
    background: radial-gradient(circle, rgba(5,99,193,0.04) 0%, rgba(5,99,193,0.015) 45%, transparent 70%);
    opacity: 0.7;
}

/* Senior Judgment Needed — warm gold caution */
.obs-state-senior-judgment-needed::before {
    background: radial-gradient(circle, rgba(230,172,0,0.055) 0%, rgba(230,172,0,0.02) 40%, transparent 70%);
    opacity: 0.9;
}
.obs-state-senior-judgment-needed::after {
    background: radial-gradient(circle, rgba(237,125,49,0.035) 0%, rgba(237,125,49,0.012) 45%, transparent 70%);
    opacity: 0.5;
}

/* Recovery Needed — faint warm red, not alarming */
.obs-state-recovery-needed::before {
    background: radial-gradient(circle, rgba(192,0,0,0.04) 0%, rgba(192,0,0,0.015) 40%, transparent 70%);
    opacity: 0.8;
}
.obs-state-recovery-needed::after {
    background: radial-gradient(circle, rgba(230,172,0,0.03) 0%, rgba(230,172,0,0.01) 45%, transparent 70%);
    opacity: 0.5;
}

/* Setup Needed — nearly invisible, just the structural shapes */
.obs-state-setup-needed::before {
    opacity: 0.3;
}
.obs-state-setup-needed::after {
    opacity: 0.2;
}

/* Hide orbs on very small screens to avoid visual clutter */
@media (max-width: 700px) {
    .obs-layout-shell::before,
    .obs-layout-shell::after {
        display: none;
    }
}

/* --- Command Center Grid --- */
.obs-command-center-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 24px;
    align-items: start;
    grid-template-areas:
        "session session"
        "brief kpis"
        "hero strip"
        "runindex champion";
}
.obs-command-center-section {
    min-width: 0;
}
.obs-cc-area-session { grid-area: session; }
.obs-cc-area-brief { grid-area: brief; }
.obs-cc-area-kpis { grid-area: kpis; }
.obs-cc-area-hero { grid-area: hero; }
.obs-cc-area-strip { grid-area: strip; }
.obs-cc-area-runindex { grid-area: runindex; }
.obs-cc-area-champion { grid-area: champion; }

/* --- Decision Brief Grid --- */
.obs-decision-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 12px;
    align-items: start;
}

/* --- Verdict Strip (editorial: bold left accent, layered shadow) --- */
.obs-verdict-strip {
    background: #FFFFFF;
    border: 1px solid #DDE4ED;
    border-radius: 10px;
    padding: 22px 24px;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 8px 32px rgba(31, 56, 100, 0.08);
    position: relative;
}
.obs-verdict-strip.safe    { border-left: 5px solid #00B050; }
.obs-verdict-strip.caution { border-left: 5px solid #E6AC00; }
.obs-verdict-strip.blocked { border-left: 5px solid #C00000; }
.obs-verdict-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.obs-verdict-badges {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.obs-verdict-pill,
.obs-safe-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82em;
    font-weight: 700;
}
.obs-verdict-pill {
    background: #E9F1FC;
    color: #1F3864;
}
.obs-safe-pill.safe {
    background: #E2F3E8;
    color: #0A6B3C;
}
.obs-safe-pill.caution {
    background: #FFF3CC;
    color: #7A5A00;
}
.obs-safe-pill.blocked {
    background: #FBE6E6;
    color: #9B1C1C;
}
.obs-verdict-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px 14px;
    margin-top: 16px;
}
.obs-verdict-item {
    min-width: 0;
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid #E4ECF7;
    background: #FFFFFF;
    transition: background 0.15s ease;
}
.obs-verdict-item:hover {
    background: #F8FBFF;
}
.obs-verdict-item.tint-blue { background: #F8FBFF; }
.obs-verdict-item.tint-green { background: #F5FCF7; }
.obs-verdict-label {
    color: #5A6C84;
    font-size: 0.76em;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.obs-verdict-value {
    color: #1F3864;
    font-size: 0.96em;
    font-weight: 600;
    line-height: 1.45;
}
.obs-verdict-value.long {
    color: #334E68;
    font-weight: 500;
}
.obs-reference-note {
    margin-top: 12px;
    color: #5A6C84;
    font-size: 0.82em;
    line-height: 1.45;
}

/* --- Projection Chips --- */
.obs-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.obs-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #E9F1FC;
    color: #1F3864;
    font-size: 0.82em;
    font-weight: 700;
}
.obs-chip.champion {
    background: #DBEAFE;
}

/* --- Recommendation Cards (editorial: numbered accent) --- */
.obs-recommendation-card {
    border: 1px solid #D0D8E3;
    border-radius: 8px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.05),
        0 4px 16px rgba(31, 56, 100, 0.06);
    padding: 18px 20px;
    border-left: 4px solid #0563C1;
    position: relative;
}
.obs-recommendation-card + .obs-recommendation-card {
    margin-top: 12px;
}
.obs-recommendation-title {
    color: #1F3864;
    font-size: 1.02em;
    font-weight: 700;
    margin-bottom: 8px;
}
.obs-recommendation-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}
.obs-recommendation-detail {
    color: #334E68;
    font-size: 0.9em;
    line-height: 1.5;
    margin-top: 6px;
}
.obs-recommendation-kicker {
    color: #5A6C84;
    font-size: 0.76em;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* --- Guided Review Cards --- */
.obs-compact-review-card .bk-Card-header,
.obs-compact-review-card .card-header {
    min-height: 0;
}

/* --- Filter Bar (contained) --- */
.obs-filter-bar {
    display: flex;
    align-items: flex-end;
    gap: 12px;
    flex-wrap: wrap;
    padding: 12px 16px;
    background: #F8FAFC;
    border-radius: 10px;
    border: 1px solid #E8EEF6;
}
.obs-filter-bar > * {
    min-width: 180px;
    flex: 1 1 180px;
}

/* --- Hero Metric (editorial: dramatic number, gradient underline) --- */
.obs-metric-hero {
    text-align: center;
    padding: 28px 24px 22px;
    border-radius: 10px;
    background: linear-gradient(180deg, #FFFFFF 0%, #F6F8FB 100%);
    border: 1px solid #C4D0DE;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 8px 32px rgba(31, 56, 100, 0.10);
    position: relative;
    overflow: hidden;
}
.obs-metric-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #0563C1, #00B0F0, #0563C1);
}
.obs-metric-hero .obs-mh-value {
    font-size: 3.2em;
    font-weight: 800;
    line-height: 1.0;
    color: #0D1B2A;
    letter-spacing: -0.03em;
}
.obs-metric-hero .obs-mh-label {
    font-size: 0.78em;
    color: #6B7D93;
    margin-top: 6px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.obs-metric-hero-empty {
    background: linear-gradient(180deg, #F4F6FA 0%, #ECEEF2 100%) !important;
    border-style: dashed !important;
    border-color: #C4D0DE !important;
    box-shadow: none !important;
}
.obs-metric-hero-empty::before {
    opacity: 0.3;
}
.obs-metric-hero .obs-mh-delta {
    font-size: 1.0em;
    font-weight: 600;
    margin-top: 6px;
}
.obs-metric-hero .obs-mh-underline {
    width: 50px;
    height: 3px;
    margin: 8px auto 0;
    border-radius: 2px;
}

/* Hero metric — animated border shimmer */
.obs-metric-hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 10px;
    padding: 2px;
    background: linear-gradient(135deg, #0563C1, #00B0F0, #00B050, #0563C1);
    background-size: 300% 300%;
    -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    animation: obs-border-shimmer 6s ease infinite;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.5s ease;
}
.obs-metric-hero:hover::after {
    opacity: 1;
}
.obs-metric-hero-empty::after {
    opacity: 0.4;
    animation: obs-border-shimmer 8s ease infinite;
}
@keyframes obs-border-shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* --- Tooltip --- */
.obs-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #E0E7F0;
    color: #5A6C84;
    font-size: 0.7em;
    font-weight: 700;
    cursor: help;
    margin-left: 6px;
    vertical-align: middle;
}
.obs-tooltip .obs-tooltip-text {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #1F3864;
    color: #FFFFFF;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 400;
    line-height: 1.4;
    white-space: normal;
    width: 240px;
    max-width: 320px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: opacity 0.2s ease, visibility 0.2s ease;
    text-transform: none;
    letter-spacing: normal;
}
.obs-tooltip:hover .obs-tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* --- Terminal Output --- */
.obs-terminal {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 0.78em;
    line-height: 1.5;
    padding: 12px 14px;
    border-radius: 8px;
    overflow: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    border: 1px solid #2a2a4a;
    box-sizing: border-box;
    width: 100%;
}

/* --- Step Badge (guided review) --- */
.obs-step-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 12px;
    background: #E9F1FC;
    color: #0563C1;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}

/* --- Next Step Bar (sticky, frosted, editorial) --- */
.obs-next-step-bar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 28px;
    margin-top: 24px;
    position: sticky;
    bottom: 0;
    z-index: 40;
    background: rgba(240, 242, 245, 0.88);
    backdrop-filter: blur(16px) saturate(1.2);
    -webkit-backdrop-filter: blur(16px) saturate(1.2);
    border: 1px solid #C4D0DE;
    border-radius: 10px 10px 0 0;
    box-shadow:
        0 -2px 8px rgba(31, 56, 100, 0.06),
        0 -8px 32px rgba(31, 56, 100, 0.08);
}
.obs-next-step-bar .obs-nsb-label {
    font-size: 0.88em;
    color: #7A8CA0;
    font-weight: 500;
}

/* --- Illustrated Empty State (editorial: atmospheric) --- */
.obs-empty-state {
    text-align: center;
    padding: 72px 32px;
    position: relative;
}
.obs-empty-state::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(5,99,193,0.06) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.obs-empty-state svg {
    margin-bottom: 24px;
    opacity: 0.7;
    filter: drop-shadow(0 4px 12px rgba(5,99,193,0.15));
}
.obs-empty-state.animate svg {
    animation: obs-gentle-float 4s ease-in-out infinite;
}
@keyframes obs-gentle-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.obs-empty-state .obs-es-message {
    color: #3D4F63;
    font-size: 0.95em;
    font-weight: 500;
    line-height: 1.6;
    max-width: 400px;
    margin: 0 auto;
}
.obs-empty-state .obs-es-action {
    display: inline-block;
    margin-top: 16px;
    color: #0563C1;
    font-size: 0.88em;
    font-weight: 700;
    text-decoration: none;
    cursor: pointer;
    padding: 8px 20px;
    border: 2px solid #0563C1;
    border-radius: 6px;
    transition: all 0.2s ease;
}
.obs-empty-state .obs-es-action:hover {
    background: #0563C1;
    color: #FFFFFF;
    text-decoration: none;
}

/* --- Card weight variants --- */
.obs-card-subtle {
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.04) !important;
    border-color: #E8EEF6 !important;
}
.obs-card-prominent {
    box-shadow: 0 8px 32px rgba(31, 56, 100, 0.10) !important;
    border-color: #C8D6EA !important;
    border-left: 3px solid #0563C1 !important;
}

/* --- CTA Card (editorial gradient with depth) --- */
.obs-cta-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 22px 28px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0563C1 0%, #1B2D4A 60%, #0D1B2A 100%);
    color: #FFFFFF;
    cursor: pointer;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease;
    border: none;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 4px 12px rgba(5, 99, 193, 0.2),
        0 12px 32px rgba(13, 27, 42, 0.15);
}
.obs-cta-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 200px;
    background: radial-gradient(circle at top right, rgba(0, 176, 240, 0.15), transparent 70%);
    pointer-events: none;
}
.obs-cta-card:hover {
    transform: translateY(-3px);
    box-shadow:
        0 6px 16px rgba(5, 99, 193, 0.3),
        0 16px 40px rgba(13, 27, 42, 0.2);
}
.obs-cta-card .obs-cta-icon {
    flex-shrink: 0;
    opacity: 0.9;
}
.obs-cta-card .obs-cta-title {
    font-size: 1.1em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.obs-cta-card .obs-cta-subtitle {
    font-size: 0.85em;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 2px;
}

/* --- Preset Button Row (card-like selectable options) --- */
.obs-preset-button-row .bk-btn {
    border: 2px solid #DDE4ED !important;
    border-radius: 10px !important;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
    color: #3D4F63 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.05);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}
.obs-preset-button-row .bk-btn:hover {
    border-color: #B8C8DB !important;
    background: linear-gradient(180deg, #FFFFFF 0%, #F0F4FA 100%) !important;
    box-shadow: 0 2px 8px rgba(5, 99, 193, 0.10) !important;
    transform: translateY(-1px);
}
.obs-preset-button-row .bk-btn-primary,
.obs-preset-button-row .bk-btn.bk-btn-primary {
    border-color: #0563C1 !important;
    background: linear-gradient(135deg, #0563C1 0%, #044B8F 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow:
        0 2px 8px rgba(5, 99, 193, 0.25),
        0 0 0 1px rgba(5, 99, 193, 0.1) inset !important;
}
.obs-preset-button-row .bk-btn-primary:hover,
.obs-preset-button-row .bk-btn.bk-btn-primary:hover {
    background: linear-gradient(135deg, #0770D4 0%, #0563C1 100%) !important;
    box-shadow:
        0 4px 16px rgba(5, 99, 193, 0.30),
        0 0 0 1px rgba(5, 99, 193, 0.15) inset !important;
    transform: translateY(-1px);
}

/* --- Inset Settings Panel --- */
.obs-inset-panel {
    background: #F4F7FB;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #E8EEF6;
}

/* --- Onboarding Card (visual welcome — warm, layered) --- */
.obs-onboarding {
    padding: 28px 32px;
    border-radius: 12px;
    background:
        radial-gradient(ellipse at 10% 80%, rgba(0,176,80,0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 20%, rgba(0,176,240,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(5,99,193,0.03) 0%, transparent 60%),
        linear-gradient(180deg, #FFFFFF 0%, #F2F5FA 100%);
    border: 1px solid #C4D0DE;
    box-shadow:
        0 1px 2px rgba(31, 56, 100, 0.06),
        0 6px 24px rgba(31, 56, 100, 0.08);
    position: relative;
    overflow: hidden;
}
.obs-onboarding::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0563C1, #00B0F0, #00B050);
    opacity: 0.6;
}
.obs-onboarding-steps {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    align-items: stretch;
}
.obs-onboarding-step {
    flex: 1;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 14px 16px;
    border-radius: 8px;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(5,99,193,0.06);
    transition: background 0.2s ease, border-color 0.2s ease;
}
.obs-onboarding-step:hover {
    background: rgba(255,255,255,0.9);
    border-color: rgba(5,99,193,0.12);
}
/* Per-step color tints */
.obs-onboarding-step:nth-child(1) {
    background: rgba(5, 99, 193, 0.06);
    border-color: rgba(5, 99, 193, 0.12);
}
.obs-onboarding-step:nth-child(1):hover {
    background: rgba(5, 99, 193, 0.10);
}
.obs-onboarding-step:nth-child(2) {
    background: rgba(0, 176, 240, 0.05);
    border-color: rgba(0, 176, 240, 0.10);
}
.obs-onboarding-step:nth-child(2):hover {
    background: rgba(0, 176, 240, 0.09);
}
.obs-onboarding-step:nth-child(3) {
    background: rgba(0, 176, 80, 0.05);
    border-color: rgba(0, 176, 80, 0.10);
}
.obs-onboarding-step:nth-child(3):hover {
    background: rgba(0, 176, 80, 0.09);
}
.obs-onboarding-num {
    flex-shrink: 0;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0563C1, #044B8F);
    color: #FFFFFF;
    font-size: 0.85em;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow:
        0 2px 8px rgba(5, 99, 193, 0.25),
        0 0 0 3px rgba(5, 99, 193, 0.08);
}
.obs-onboarding-title {
    font-size: 0.92em;
    font-weight: 800;
    color: #0D1B2A;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.obs-onboarding-desc {
    font-size: 0.82em;
    color: #5A6C84;
    line-height: 1.55;
}
.obs-onboarding-footer {
    font-size: 0.8em;
    color: #7A8CA0;
    padding-top: 14px;
    border-top: 1px solid #DDE4ED;
}
@media (max-width: 900px) {
    .obs-onboarding-steps {
        flex-direction: column;
        gap: 12px;
    }
}

/* --- Collapsible card headers (rich dark treatment) --- */
.bk-Card-header, .card-header {
    background: linear-gradient(135deg, #1B2D4A 0%, #1F3864 60%, #244272 100%) !important;
    border-bottom: 1px solid rgba(5, 99, 193, 0.3) !important;
    border-radius: 10px 10px 0 0;
    padding: 14px 20px !important;
    box-shadow: 0 2px 8px rgba(13, 27, 42, 0.15) inset;
}
.bk-Card-header .bk-Card-title,
.card-header .card-title,
.bk-Card-header .bk-panel-models-markup-HTML,
.bk-Card-header > div {
    color: #FFFFFF !important;
    font-size: 0.92em !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
/* Arrow/toggle icon in card headers */
.bk-Card-header .bk-btn,
.bk-Card-header button {
    color: rgba(255,255,255,0.8) !important;
    filter: brightness(10);
}
.bk-Card.collapsed .bk-Card-header,
.bk-Card.collapsed .card-header {
    border-radius: 10px !important;
    border-bottom: none !important;
}
/* Primary workflow card — blue accent glow */
.obs-primary-workflow-card > .bk-Card-header,
.obs-primary-workflow-card > .card-header {
    background: linear-gradient(135deg, #0563C1 0%, #1B2D4A 60%, #0D1B2A 100%) !important;
    border-bottom: 1px solid rgba(0, 176, 240, 0.3) !important;
    box-shadow:
        0 2px 8px rgba(13, 27, 42, 0.15) inset,
        0 4px 16px rgba(5, 99, 193, 0.15);
}

/* --- Card hover transitions --- */
.card-container, .bk-Card {
    transition: box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1), transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.obs-elevation-2:hover,
.card-container:hover, .bk-Card:hover {
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8) inset,
        0 1px 3px rgba(31, 56, 100, 0.08),
        0 8px 32px rgba(31, 56, 100, 0.14),
        0 16px 56px rgba(31, 56, 100, 0.06);
    transform: translateY(-2px);
}
.obs-elevation-1:hover {
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.05) !important;
    transform: none !important;
}
.obs-elevation-3:hover {
    transform: none !important;
}

/* --- Insight Card (editorial: bold accent bars, rich backgrounds) --- */
.obs-insight-card {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
}
.obs-insight-section {
    padding: 18px;
    border-radius: 8px;
    border: 1px solid #DDE4ED;
    background: #FFFFFF;
    position: relative;
    overflow: hidden;
}
.obs-insight-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
}
.obs-insight-section.gain::before { background: linear-gradient(180deg, #00B050, #009940); }
.obs-insight-section.tradeoff::before { background: linear-gradient(180deg, #E6AC00, #CC9900); }
.obs-insight-section.risk::before { background: linear-gradient(180deg, #C00000, #A00000); }
.obs-insight-section.gain { background: linear-gradient(135deg, #FFFFFF 0%, #F2FAF5 100%); }
.obs-insight-section.tradeoff { background: linear-gradient(135deg, #FFFFFF 0%, #FFFBF0 100%); }
.obs-insight-section.risk { background: linear-gradient(135deg, #FFFFFF 0%, #FDF5F5 100%); }
.obs-insight-icon {
    font-size: 1.2em;
    margin-bottom: 6px;
}
.obs-insight-label {
    font-size: 0.7em;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7D93;
    margin-bottom: 8px;
    padding-left: 8px;
}
.obs-insight-value {
    font-size: 0.92em;
    color: #3D4F63;
    line-height: 1.55;
    padding-left: 8px;
}

/* --- Checklist Card (structured) --- */
.obs-checklist-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #E8EEF6;
    background: #FFFFFF;
    margin-bottom: 6px;
    transition: background 0.15s ease;
}
.obs-checklist-item:hover { background: #F8FBFF; }
.obs-checklist-item .obs-check-icon {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72em;
    font-weight: 700;
    color: #FFFFFF;
}
.obs-checklist-item.passed .obs-check-icon { background: #00B050; }
.obs-checklist-item.failed .obs-check-icon { background: #C00000; }
.obs-checklist-item.pending .obs-check-icon { background: #A0A0A0; }
.obs-check-label {
    font-size: 0.92em;
    font-weight: 600;
    color: #1F3864;
}
.obs-check-detail {
    font-size: 0.85em;
    color: #7A8CA0;
    margin-top: 2px;
}

/* --- Inner Tabs (Experiment History sub-tabs) --- */
.obs-inner-tabs .bk-tab {
    font-size: 0.85em;
    padding: 8px 16px;
    border-radius: 8px;
    background: transparent;
    border: none;
}
.obs-inner-tabs .bk-tab.bk-active {
    background: #EBF3FE;
    color: #1F3864;
}

@media (orientation: portrait), (max-aspect-ratio: 1/1) {
    header.pn-header,
    :host(.pn-header) {
        min-height: 48px !important;
    }

    .card-container, .bk-Card {
        padding: 12px;
        margin-bottom: 12px;
    }

    .obs-workflow-stepper {
        padding: 6px 8px;
        margin: 4px 0;
    }

    .obs-step {
        padding: 4px 8px;
        font-size: 0.75em;
    }

    .obs-step .obs-step-subtitle { display: none; }

    .section-header h2 {
        font-size: 1.15em;
    }

    .section-header .subtitle {
        font-size: 0.82em;
        line-height: 1.4;
    }

    .obs-command-center-grid {
        grid-template-columns: 1fr;
        grid-template-areas:
            "session"
            "brief"
            "kpis"
            "hero"
            "strip"
            "runindex"
            "champion";
    }

    .obs-decision-grid {
        grid-template-columns: 1fr;
    }

    .obs-verdict-grid {
        grid-template-columns: 1fr;
    }

    .obs-filter-bar {
        gap: 8px;
        padding: 10px 12px;
    }

    .obs-filter-bar > * {
        flex: 1 1 280px;
        min-width: 0;
    }

    .obs-preset-button-row {
        display: grid !important;
        grid-template-columns: 1fr;
        gap: 8px;
    }

    .obs-preset-button-row .bk-btn {
        min-height: 52px;
        width: 100%;
        justify-content: flex-start;
    }

    .obs-next-step-bar {
        bottom: 8px;
        box-shadow: 0 -8px 28px rgba(31, 56, 100, 0.12);
        margin-top: 12px;
    }

    .js-plotly-plot .modebar {
        opacity: 0 !important;
        pointer-events: none !important;
        transition: opacity 0.12s ease;
    }

    .js-plotly-plot:hover .modebar,
    .js-plotly-plot:focus-within .modebar {
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    .obs-insight-card {
        grid-template-columns: 1fr;
    }
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
    .obs-workflow-stepper {
        padding: 10px 12px;
        gap: 0;
    }
    .obs-step {
        font-size: 0.78em;
        padding: 4px 8px;
    }
    .obs-step .obs-step-num {
        width: 22px;
        height: 22px;
        font-size: 0.75em;
    }
    .obs-step-connector {
        min-width: 16px;
    }
    .obs-metric-hero .obs-mh-value {
        font-size: 2.0em;
    }
}
"""

_BASE_TABS_STYLESHEET = """
:host {
    overflow: visible;
}

.bk-header {
    display: flex;
    gap: 2px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0 0 0 0;
    scrollbar-width: thin;
    border-bottom: 2px solid #DDE4ED;
}

.bk-tab {
    white-space: nowrap;
    min-width: max-content;
    padding: 12px 18px;
    border-radius: 0;
    font-weight: 700;
    font-size: 0.82em;
    color: #6B7D93;
    letter-spacing: 0.02em;
    transition: color 0.2s ease, border-color 0.2s ease;
    position: relative;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
}

.bk-tab:hover {
    color: #0D1B2A;
}

.bk-tab.bk-active {
    color: #0D1B2A;
    font-weight: 800;
    border-bottom: 3px solid #0563C1;
    background: linear-gradient(180deg, rgba(5,99,193,0.08) 0%, rgba(5,99,193,0.02) 100%);
    border-radius: 6px 6px 0 0;
}

.bk-tab.bk-active::after {
    content: none;
}

/* Tab grouping gaps */
.bk-tab:nth-child(2) { margin-left: 16px; }
.bk-tab:nth-child(7) { margin-left: 16px; opacity: 0.6; }

@media (max-width: 700px) {
    .bk-header {
        gap: 2px;
    }

    .bk-tab {
        font-size: 0.75em;
        padding: 10px 12px;
    }

    .bk-tab:nth-child(2),
    .bk-tab:nth-child(7) { margin-left: 8px; }
}
"""


def build_tabs_stylesheet(
    review_mode: bool = False,
    *,
    experience_mode: str = "guided",
) -> str:
    """Return the tabs stylesheet, optionally emphasizing guided review tabs."""
    if not review_mode or experience_mode != "guided":
        return _BASE_TABS_STYLESHEET

    return (
        _BASE_TABS_STYLESHEET
        + """

.bk-tab {
    opacity: 0.62;
}

.bk-tab:nth-child(2),
.bk-tab:nth-child(3),
.bk-tab:nth-child(4),
.bk-tab:nth-child(5),
.bk-tab:nth-child(6),
.bk-tab.bk-active {
    opacity: 1;
}

.bk-tab:nth-child(1),
.bk-tab:nth-child(7) {
    color: #7A8798;
}
"""
    )


TABS_STYLESHEET = build_tabs_stylesheet(review_mode=False)

TABULATOR_STYLESHEET = """
.tabulator {
    background: #FFFFFF;
    border: 1px solid #DDE4ED;
    border-radius: 8px;
    overflow: hidden;
    font-size: 12px;
}

.tabulator .tabulator-header {
    background: linear-gradient(180deg, #0D1B2A, #152238) !important;
    border-bottom: none !important;
}

.tabulator .tabulator-header .tabulator-col {
    background: transparent !important;
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.72em;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.tabulator-row {
    background: #FFFFFF;
    transition: background 0.15s ease;
}

.tabulator-row:nth-child(even) {
    background: #F8F9FB;
}

.tabulator-row:hover {
    background: #EDF2F8 !important;
}

.tabulator-cell {
    border-right: 1px solid #E8ECF1 !important;
}

.tabulator-footer {
    background: #F4F5F7;
    border-top: 1px solid #DDE4ED;
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
        "font": {"family": FONT_FAMILY, "size": 12, "color": "#3D4F63"},
        "title_font": {"family": FONT_FAMILY, "size": 15, "color": "#0D1B2A"},
        "plot_bgcolor": "#F8F9FB",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 64, "r": 24, "t": 48, "b": 48},
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#E8ECF1",
            "linecolor": "#C4D0DE",
            "linewidth": 1,
            "tickfont": {"size": 11, "color": "#6B7D93"},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#E8ECF1",
            "linecolor": "#C4D0DE",
            "linewidth": 1,
            "tickfont": {"size": 11, "color": "#6B7D93"},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#DDE4ED",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "hoverlabel": {
            "bgcolor": "#0D1B2A",
            "font_size": 12,
            "font_color": "#FFFFFF",
            "font_family": FONT_FAMILY,
            "bordercolor": "#0D1B2A",
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
