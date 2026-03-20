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
    background: linear-gradient(180deg, #F7F9FC 0%, #F1F4F8 100%) !important;
}

/* --- Three-Tier Elevation System --- */
.card-container, .bk-Card {
    background-color: #FFFFFF;
    border: 1px solid #E2EAF4;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(31, 56, 100, 0.06);
    padding: 16px;
    margin-bottom: 16px;
}
.obs-elevation-1 {
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.04) !important;
    border-color: #E8EEF6 !important;
}
.obs-elevation-2 {
    box-shadow: 0 4px 16px rgba(31, 56, 100, 0.06) !important;
    border-color: #E2EAF4 !important;
}
.obs-elevation-3 {
    box-shadow: 0 8px 32px rgba(31, 56, 100, 0.10) !important;
    border-color: #C8D6EA !important;
}

/* --- Typography Scale --- */
.obs-text-display { font-size: 2.8em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.0; color: #1F3864; }
.obs-text-headline { font-size: 1.5em; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2; color: #1F3864; }
.obs-text-title { font-size: 1.15em; font-weight: 700; line-height: 1.3; color: #1F3864; }
.obs-text-body { font-size: 0.92em; font-weight: 400; line-height: 1.55; color: #4F5F74; }
.obs-text-caption { font-size: 0.8em; font-weight: 500; color: #7A8CA0; letter-spacing: 0.02em; }
.obs-text-eyebrow { font-size: 0.72em; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #7A8CA0; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-passed   { background-color: #00B050; color: #FFFFFF; }
.badge-review   { background-color: #FFC000; color: #1F3864; }
.badge-failed   { background-color: #C00000; color: #FFFFFF; }
.badge-untested { background-color: #A0A0A0; color: #FFFFFF; }
.badge-champion { background-color: #0563C1; color: #FFFFFF; }

/* KPI card */
.kpi-card {
    text-align: center;
    padding: 18px 14px;
    min-width: 150px;
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid #E2EAF4;
    box-shadow: 0 4px 16px rgba(31, 56, 100, 0.06);
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.kpi-card:hover {
    box-shadow: 0 8px 24px rgba(31, 56, 100, 0.10);
    transform: translateY(-1px);
}
.kpi-card .kpi-value {
    font-size: 2.0em;
    font-weight: 800;
    line-height: 1.1;
    color: #1F3864;
    letter-spacing: -0.01em;
}
.kpi-card .kpi-label {
    font-size: 0.8em;
    font-weight: 500;
    color: #7A8CA0;
    margin-top: 6px;
    letter-spacing: 0.02em;
}
.kpi-card .kpi-delta {
    font-size: 0.88em;
    font-weight: 700;
    margin-top: 4px;
}
.kpi-delta.positive { color: #C00000; }
.kpi-delta.negative { color: #00B050; }
.kpi-delta.neutral  { color: #7A8CA0; }

/* KPI responsive grid */
.obs-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
}

/* Section headers */
.section-header h2 {
    margin: 0;
    color: #1F3864;
    font-size: 1.5em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.section-header .subtitle {
    margin: 4px 0 0 0;
    color: #7A8CA0;
    font-size: 0.88em;
    font-weight: 400;
    line-height: 1.45;
}

.summary-card {
    min-width: 220px;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #E2EAF4;
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFD 100%);
    box-shadow: 0 4px 16px rgba(31, 56, 100, 0.06);
}
.summary-card .eyebrow {
    color: #7A8CA0;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.summary-card .headline {
    margin-top: 8px;
    color: #1F3864;
    font-size: 1.25em;
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: -0.01em;
}
.summary-card .detail {
    margin-top: 8px;
    color: #4F5F74;
    font-size: 0.9em;
    line-height: 1.5;
}
.summary-card.primary {
    background: linear-gradient(180deg, #F8FBFF 0%, #EDF3FB 100%);
    border-color: #C8D6EA;
    box-shadow: 0 8px 32px rgba(31, 56, 100, 0.10);
}
.summary-card.warning {
    background: linear-gradient(180deg, #FFFBF0 0%, #FFF5D6 100%);
    border-color: #F0D88A;
}
.summary-card.success {
    background: linear-gradient(180deg, #F5FCF7 0%, #E6F6EB 100%);
    border-color: #A8DEB8;
}

.filters-help {
    color: #7A8CA0;
    font-size: 0.85em;
    margin-bottom: 6px;
}

/* Empty placeholder */
.empty-placeholder {
    text-align: center;
    padding: 48px 24px;
    color: #7A8CA0;
    font-style: normal;
    font-size: 0.92em;
    line-height: 1.5;
}

/* --- Tabulator (modernized light header) --- */
.tabulator .tabulator-header {
    background: #F4F7FB !important;
    color: #1F3864 !important;
    border-bottom: 2px solid #D9E3F0 !important;
}
.tabulator .tabulator-header .tabulator-col {
    background: transparent !important;
    color: #1F3864 !important;
    font-size: 0.82em;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* --- Workflow Stepper --- */
.obs-workflow-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 10px 16px;
    margin: 8px 0 4px;
}
.obs-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 4px 14px;
    font-size: 0.82em;
    font-weight: 600;
    color: #A0A0A0;
    white-space: nowrap;
}
.obs-step .obs-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 2px solid #D9D9D9;
    background: #FFFFFF;
    font-size: 0.82em;
    font-weight: 700;
    color: #A0A0A0;
    transition: all 0.3s ease;
}
.obs-step .obs-step-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
}
.obs-step .obs-step-subtitle {
    font-size: 0.72em;
    font-weight: 500;
    color: #A0A0A0;
    opacity: 0.8;
}
.obs-step.active .obs-step-num {
    border-color: #0563C1;
    background: #0563C1;
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(5, 99, 193, 0.25);
}
.obs-step.active {
    color: #1F3864;
}
.obs-step.active .obs-step-subtitle {
    color: #5A6C84;
}
.obs-step.completed .obs-step-num {
    border-color: #00B050;
    background: #00B050;
    color: #FFFFFF;
}
.obs-step.completed {
    color: #00B050;
}
.obs-step-connector {
    flex: 1;
    height: 2px;
    min-width: 30px;
    max-width: 80px;
    background: #E2EAF4;
    transition: background 0.3s ease;
    margin-bottom: 16px;
}
.obs-step-connector.completed {
    background: #00B050;
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

/* --- Layout Shell --- */
.obs-layout-shell {
    gap: 16px;
}

/* --- Command Center Grid --- */
.obs-command-center-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 24px;
    align-items: start;
    grid-template-areas:
        "session session"
        "launch brief"
        "launch kpis"
        "hero strip"
        "runindex champion";
}
.obs-command-center-section {
    min-width: 0;
}
.obs-cc-area-session { grid-area: session; }
.obs-cc-area-launch { grid-area: launch; }
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

/* --- Verdict Strip --- */
.obs-verdict-strip {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFD 100%);
    border: 1px solid #C8D6EA;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 8px 32px rgba(31, 56, 100, 0.10);
}
.obs-verdict-strip.safe    { border-left: 4px solid #00B050; }
.obs-verdict-strip.caution { border-left: 4px solid #FFC000; }
.obs-verdict-strip.blocked { border-left: 4px solid #C00000; }
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

/* --- Recommendation Cards --- */
.obs-recommendation-card {
    border: 1px solid #D9E3F0;
    border-radius: 12px;
    background: #FFFFFF;
    box-shadow: 0 8px 22px rgba(31, 56, 100, 0.06);
    padding: 16px 18px;
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

/* --- Hero Metric --- */
.obs-metric-hero {
    text-align: center;
    padding: 24px 20px 20px;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid #C8D6EA;
    box-shadow: 0 8px 32px rgba(31, 56, 100, 0.10);
}
.obs-metric-hero .obs-mh-value {
    font-size: 2.8em;
    font-weight: 800;
    line-height: 1.1;
    color: #1F3864;
    letter-spacing: -0.02em;
}
.obs-metric-hero .obs-mh-label {
    font-size: 0.88em;
    color: #5A6C84;
    margin-top: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
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

/* --- Next Step Bar (sticky frosted glass) --- */
.obs-next-step-bar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    padding: 14px 24px;
    margin-top: 24px;
    position: sticky;
    bottom: 0;
    z-index: 40;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid #E2EAF4;
    border-radius: 14px 14px 0 0;
    box-shadow: 0 -4px 24px rgba(31, 56, 100, 0.08);
}
.obs-next-step-bar .obs-nsb-label {
    font-size: 0.88em;
    color: #7A8CA0;
    font-weight: 500;
}

/* --- Illustrated Empty State --- */
.obs-empty-state {
    text-align: center;
    padding: 56px 32px;
}
.obs-empty-state svg {
    margin-bottom: 20px;
    opacity: 0.6;
}
.obs-empty-state.animate svg {
    animation: obs-gentle-float 3s ease-in-out infinite;
}
@keyframes obs-gentle-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.obs-empty-state .obs-es-message {
    color: #7A8CA0;
    font-size: 0.92em;
    line-height: 1.55;
    max-width: 380px;
    margin: 0 auto;
}
.obs-empty-state .obs-es-action {
    display: inline-block;
    margin-top: 14px;
    color: #0563C1;
    font-size: 0.88em;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
}
.obs-empty-state .obs-es-action:hover {
    text-decoration: underline;
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

/* --- CTA Card (gradient hero button) --- */
.obs-cta-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0563C1 0%, #1F3864 100%);
    color: #FFFFFF;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
    border: none;
    position: relative;
    overflow: hidden;
}
.obs-cta-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(5, 99, 193, 0.30);
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

/* --- Inset Settings Panel --- */
.obs-inset-panel {
    background: #F4F7FB;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #E8EEF6;
}

/* --- Card hover transitions --- */
.card-container, .bk-Card {
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.obs-elevation-2:hover,
.card-container:hover, .bk-Card:hover {
    box-shadow: 0 8px 28px rgba(31, 56, 100, 0.10);
    transform: translateY(-1px);
}
.obs-elevation-1:hover {
    box-shadow: 0 1px 3px rgba(31, 56, 100, 0.04) !important;
    transform: none !important;
}
.obs-elevation-3:hover {
    transform: none !important;
}

/* --- Insight Card (structured gain/tradeoff/risk) --- */
.obs-insight-card {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}
.obs-insight-section {
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #E4ECF7;
    background: #FFFFFF;
}
.obs-insight-section.gain { border-left: 3px solid #00B050; }
.obs-insight-section.tradeoff { border-left: 3px solid #FFC000; }
.obs-insight-section.risk { border-left: 3px solid #C00000; }
.obs-insight-icon {
    font-size: 1.2em;
    margin-bottom: 6px;
}
.obs-insight-label {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7A8CA0;
    margin-bottom: 6px;
}
.obs-insight-value {
    font-size: 0.92em;
    color: #334E68;
    line-height: 1.5;
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
            "launch"
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
    gap: 4px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0 0 6px 0;
    scrollbar-width: thin;
    border-bottom: 1px solid #E2EAF4;
}

.bk-tab {
    white-space: nowrap;
    min-width: max-content;
    padding: 10px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 0.9em;
    color: #5A6C84;
    transition: color 0.15s ease, background 0.15s ease;
    position: relative;
}

.bk-tab:hover {
    color: #1F3864;
    background: #F4F7FB;
}

.bk-tab.bk-active {
    color: #1F3864;
    background: #FFFFFF;
    font-weight: 700;
}

.bk-tab.bk-active::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 24px;
    height: 2px;
    border-radius: 1px;
    background: #0563C1;
}

/* Tab grouping gaps */
.bk-tab:nth-child(2) { margin-left: 12px; }
.bk-tab:nth-child(7) { margin-left: 12px; opacity: 0.7; }

@media (max-width: 700px) {
    .bk-header {
        gap: 4px;
        padding-bottom: 6px;
    }

    .bk-tab {
        font-size: 0.8em;
        padding: 8px 12px;
    }

    .bk-tab:nth-child(2),
    .bk-tab:nth-child(7) { margin-left: 6px; }
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
    border: 1px solid #E2EAF4;
    border-radius: 10px;
    overflow: hidden;
    font-size: 12px;
}

.tabulator .tabulator-header {
    background: #F4F7FB !important;
    border-bottom: 2px solid #D9E3F0 !important;
}

.tabulator .tabulator-header .tabulator-col {
    background: transparent !important;
    color: #1F3864 !important;
    font-size: 0.82em;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.tabulator-row {
    background: #FFFFFF;
}

.tabulator-row:nth-child(even) {
    background: #FAFBFD;
}

.tabulator-row:hover {
    background: #EBF3FE !important;
}

.tabulator-cell {
    border-right: 1px solid #EEF1F6 !important;
}

.tabulator-footer {
    background: #FAFBFD;
    border-top: 1px solid #E2EAF4;
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
        "title_font": {"family": FONT_FAMILY, "size": 15, "color": SDC_NAVY},
        "plot_bgcolor": "#FAFBFD",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 64, "r": 24, "t": 48, "b": 48},
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#EEF1F6",
            "linecolor": "#E2EAF4",
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#EEF1F6",
            "linecolor": "#E2EAF4",
            "linewidth": 1,
            "tickfont": {"size": 11},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#E2EAF4",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "hoverlabel": {
            "bgcolor": SDC_WHITE,
            "font_size": 12,
            "font_family": FONT_FAMILY,
            "bordercolor": "#E2EAF4",
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
