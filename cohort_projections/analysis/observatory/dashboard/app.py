"""Main Panel application assembly for the Projection Observatory dashboard.

Brings together all tab modules into a unified ``FastListTemplate``
application with SDC branding and a workflow stepper.
"""

from __future__ import annotations

import logging

import panel as pn

from .data_manager import DashboardDataManager
from .theme import (
    DASHBOARD_CSS,
    SDC_NAVY,
    SDC_WHITE,
    build_tabs_stylesheet,
    layout_mode_classes,
)
from .widgets import workflow_stepper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tab index constants — import these instead of hard-coding integers.
# ---------------------------------------------------------------------------
TAB_COMMAND_CENTER = 0
TAB_DECISION_BRIEF = 1
TAB_SCORECARDS = 2
TAB_PROJECTIONS = 3
TAB_HORIZON_BIAS = 4
TAB_SENSITIVITY = 5
TAB_EXPERIMENT_HISTORY = 6

_WORKFLOW_STEPS = ["Launch", "Monitor", "Review", "Decide"]
_WORKFLOW_SUBTITLES = ["Command Center", "Search Progress", "Brief \u2192 Bias", "Sensitivity"]

# Mapping from tab index to stepper step index.
_TAB_TO_STEP: dict[int, int] = {
    TAB_COMMAND_CENTER: 0,  # Launch (or Monitor when search running)
    TAB_DECISION_BRIEF: 2,  # Review
    TAB_SCORECARDS: 2,  # Review
    TAB_PROJECTIONS: 2,  # Review
    TAB_HORIZON_BIAS: 2,  # Review
    TAB_SENSITIVITY: 3,  # Decide
    TAB_EXPERIMENT_HISTORY: 2,  # Review / audit trail
}

# Review-mode tab sequence for the "Next Step" navigation.
REVIEW_TAB_SEQUENCE = [
    TAB_DECISION_BRIEF,
    TAB_SCORECARDS,
    TAB_PROJECTIONS,
    TAB_HORIZON_BIAS,
    TAB_SENSITIVITY,
]


def _has_running_search(dm: DashboardDataManager) -> bool:
    """Return whether the Observatory currently has a live search session."""
    try:
        search_id = dm.active_search_id
        if not search_id:
            return False
        session_row = dm.search_session_row(search_id)
        if session_row is None:
            return False
        return bool(session_row.get("dashboard_process_running", False))
    except Exception:  # pragma: no cover - defensive UI guard
        logger.exception("Failed to inspect autonomous-search state for stepper.")
        return False


def _resolve_stepper_state(
    dm: DashboardDataManager,
    active_tab: int,
) -> tuple[int, list[int]]:
    """Return the current stepper active index and completed steps."""
    step = _TAB_TO_STEP.get(active_tab, 0)
    completed: list[int] = []

    if dm.selection_state.review_mode and active_tab in (
        TAB_DECISION_BRIEF,
        TAB_SCORECARDS,
        TAB_PROJECTIONS,
        TAB_HORIZON_BIAS,
        TAB_SENSITIVITY,
    ):
        return 2, [0, 1]

    has_runs = len(dm.run_ids) > 0
    has_search_history = not dm.search_sessions.empty

    if active_tab == TAB_COMMAND_CENTER:
        if _has_running_search(dm):
            return 1, [0]
        if has_runs or has_search_history:
            return 0, [0, 1]
        return 0, []

    if has_runs or active_tab >= TAB_SCORECARDS:
        completed = [0, 1]
    return step, completed


def _stepper_html(active: int, completed: list[int] | None = None) -> str:
    """Return the raw HTML string for the workflow stepper at a given state."""
    pane = workflow_stepper(
        _WORKFLOW_STEPS,
        active=active,
        completed=completed,
        subtitles=_WORKFLOW_SUBTITLES,
    )
    return str(pane.object)


def create_app(dm: DashboardDataManager | None = None) -> pn.template.FastListTemplate:
    """Build and return the Observatory dashboard as a Panel template.

    Parameters
    ----------
    dm:
        Pre-built data manager.  If ``None``, one is created from the
        default observatory configuration.

    Returns
    -------
    pn.template.FastListTemplate
        Ready-to-serve Panel application.
    """
    if dm is None:
        dm = DashboardDataManager()

    # ------------------------------------------------------------------
    # Lazy-import tab builders to keep startup fast if only some tabs
    # are needed and to tolerate import-time issues in individual tabs.
    # ------------------------------------------------------------------
    from .tab_command_center import build_command_center
    from .tab_decision_brief import build_decision_brief_tab
    from .tab_experiment_history import build_experiment_history
    from .tab_horizon_bias import build_horizon_bias_tab
    from .tab_projection_ensemble import build_projection_ensemble
    from .tab_scorecard import build_scorecard_tab
    from .tab_sensitivity import build_sensitivity_tab

    # Build tabs --------------------------------------------------------
    logger.info("Building dashboard tabs …")

    tabs = pn.Tabs(
        tabs_location="above",
        dynamic=True,
    )

    # Unified header card: state summary + stepper + mode switch in one.
    header_pane = pn.pane.HTML(sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])
    mode_switch = pn.widgets.RadioButtonGroup(
        name="Experience Mode",
        options={"Guided": "guided", "Explore Directly": "direct"},
        value=dm.selection_state.experience_mode,
        button_type="default",
        sizing_mode="fixed",
    )

    tabs.extend(
        [
            ("Command Center", build_command_center(dm, tabs=tabs)),
            ("Decision Brief", build_decision_brief_tab(dm, tabs=tabs)),
            ("Scorecards", build_scorecard_tab(dm, tabs=tabs)),
            ("Projections", build_projection_ensemble(dm, tabs=tabs)),
            ("Horizon & Bias", build_horizon_bias_tab(dm, tabs=tabs)),
            ("Sensitivity", build_sensitivity_tab(dm, tabs=tabs)),
            ("Experiment History", build_experiment_history(dm)),
        ]
    )
    tabs.active = 0
    tabs.stylesheets = [
        build_tabs_stylesheet(
            dm.selection_state.review_mode,
            experience_mode=dm.selection_state.experience_mode,
        )
    ]

    def _refresh_tab_chrome() -> None:
        """Keep tab chrome aligned with guided-review emphasis."""
        tabs.stylesheets = [
            build_tabs_stylesheet(
                dm.selection_state.review_mode,
                experience_mode=dm.selection_state.experience_mode,
            )
        ]

    def _refresh_header() -> None:
        """Render the unified header card: state + stepper + mode indicator."""
        state = dm.workspace_state
        dm.selection_state.workspace_state = str(state.get("state", "") or "")
        mode_label = "Guided" if dm.selection_state.experience_mode == "guided" else "Direct"
        step, completed = _resolve_stepper_state(dm, tabs.active)
        stepper_markup = _stepper_html(active=step, completed=completed)
        header_pane.object = (
            '<div class="summary-card primary" '
            'style="max-width:none;position:relative;padding:18px 22px 12px">'
            '<div class="obs-text-eyebrow">Workspace</div>'
            f'<div class="obs-text-headline" style="margin-top:6px">'
            f"{state.get('route_title', 'Command Center')}"
            f'<span style="font-size:0.55em;font-weight:500;color:#7A8CA0;'
            f'margin-left:10px;letter-spacing:0.02em">{mode_label}</span>'
            f"</div>"
            f"{stepper_markup}"
            f'<div class="obs-text-body" style="margin-top:4px">'
            f"{state.get('summary', '')}</div>"
            "</div>"
        )

    # Stepper reactivity ------------------------------------------------
    def _on_tab_change(event: object) -> None:
        """Update the workflow stepper when the active tab changes."""
        _refresh_tab_chrome()
        _refresh_header()

    def _on_mode_change(event: object) -> None:
        """Switch between guided and direct exploration modes."""
        new_mode = str(getattr(event, "new", dm.selection_state.experience_mode) or "guided")
        dm.selection_state.experience_mode = new_mode
        if new_mode == "direct":
            dm.selection_state.review_mode = False
        else:
            tabs.active = TAB_COMMAND_CENTER
        _refresh_tab_chrome()
        _refresh_header()

    tabs.param.watch(_on_tab_change, "active")
    dm.selection_state.param.watch(lambda event: _refresh_tab_chrome(), "review_mode")
    dm.selection_state.param.watch(
        lambda event: setattr(mode_switch, "value", event.new)
        if mode_switch.value != event.new
        else None,
        "experience_mode",
    )
    mode_switch.param.watch(_on_mode_change, "value")
    _refresh_header()

    if pn.state.curdoc is not None:

        def _refresh_periodic() -> None:
            """Keep the header aligned with live search activity."""
            dm.refresh_search_sessions()
            _refresh_tab_chrome()
            _refresh_header()

        pn.state.add_periodic_callback(_refresh_periodic, period=5000, start=True)

    shell = pn.Column(
        pn.Row(mode_switch, sizing_mode="stretch_width"),
        header_pane,
        tabs,
        css_classes=layout_mode_classes("obs-layout-shell"),
        sizing_mode="stretch_width",
    )

    # Assemble template -------------------------------------------------
    template = pn.template.FastListTemplate(
        title="Projection Observatory",
        header_background=SDC_NAVY,
        header_color=SDC_WHITE,
        accent_base_color=SDC_NAVY,
        shadow=True,
        main=[shell],
        raw_css=[DASHBOARD_CSS],
        main_max_width="1600px",
    )

    logger.info("Observatory dashboard ready.")
    return template
