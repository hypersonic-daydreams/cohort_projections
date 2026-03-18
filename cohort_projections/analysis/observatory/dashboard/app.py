"""Main Panel application assembly for the Projection Observatory dashboard.

Brings together all tab modules into a unified ``FastListTemplate``
application with SDC branding and a workflow stepper.
"""

from __future__ import annotations

import logging

import panel as pn

from .data_manager import DashboardDataManager
from .theme import DASHBOARD_CSS, SDC_NAVY, SDC_WHITE, TABS_STYLESHEET
from .widgets import workflow_stepper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tab index constants — import these instead of hard-coding integers.
# ---------------------------------------------------------------------------
TAB_COMMAND_CENTER = 0
TAB_EXPERIMENT_HISTORY = 1
TAB_SCORECARDS = 2
TAB_PROJECTIONS = 3
TAB_HORIZON_BIAS = 4
TAB_SENSITIVITY = 5

_WORKFLOW_STEPS = ["Launch", "Monitor", "Review", "Decide"]

# Mapping from tab index to stepper step index.
_TAB_TO_STEP: dict[int, int] = {
    TAB_COMMAND_CENTER: 0,  # Launch (or Monitor when search running)
    TAB_EXPERIMENT_HISTORY: 2,  # Review
    TAB_SCORECARDS: 2,  # Review
    TAB_PROJECTIONS: 2,  # Review
    TAB_HORIZON_BIAS: 2,  # Review
    TAB_SENSITIVITY: 3,  # Decide
}

# Review-mode tab sequence for the "Next Step" navigation.
REVIEW_TAB_SEQUENCE = [TAB_SCORECARDS, TAB_PROJECTIONS, TAB_HORIZON_BIAS, TAB_SENSITIVITY]


def _stepper_html(active: int, completed: list[int] | None = None) -> str:
    """Return the raw HTML string for the workflow stepper at a given state."""
    pane = workflow_stepper(_WORKFLOW_STEPS, active=active, completed=completed)
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

    # Build the workflow stepper pane (reactive — updated on tab change).
    stepper_pane = pn.pane.HTML(
        _stepper_html(active=0),
        sizing_mode="stretch_width",
        stylesheets=[DASHBOARD_CSS],
    )

    tabs.extend(
        [
            ("Command Center", build_command_center(dm, tabs=tabs)),
            ("Experiment History", build_experiment_history(dm)),
            ("Scorecards", build_scorecard_tab(dm, tabs=tabs)),
            ("Projections", build_projection_ensemble(dm, tabs=tabs)),
            ("Horizon & Bias", build_horizon_bias_tab(dm, tabs=tabs)),
            ("Sensitivity", build_sensitivity_tab(dm, tabs=tabs)),
        ]
    )
    tabs.active = 0
    tabs.stylesheets = [TABS_STYLESHEET]

    # Stepper reactivity ------------------------------------------------
    def _on_tab_change(event: object) -> None:
        """Update the workflow stepper when the active tab changes."""
        active_tab = tabs.active
        step = _TAB_TO_STEP.get(active_tab, 0)

        # Determine completed steps.
        completed: list[int] = []

        # If in review mode, lock stepper to "Review".
        if dm.selection_state.review_mode and active_tab in (
            TAB_SCORECARDS,
            TAB_PROJECTIONS,
            TAB_HORIZON_BIAS,
            TAB_SENSITIVITY,
        ):
            step = 2
            completed = [0, 1]
        else:
            has_runs = len(dm.run_ids) > 0
            if has_runs or active_tab >= TAB_SCORECARDS:
                completed = [0, 1]

        stepper_pane.object = _stepper_html(active=step, completed=completed)

    tabs.param.watch(_on_tab_change, "active")

    # Assemble template -------------------------------------------------
    template = pn.template.FastListTemplate(
        title="Projection Observatory",
        header_background=SDC_NAVY,
        header_color=SDC_WHITE,
        accent_base_color=SDC_NAVY,
        shadow=True,
        main=[stepper_pane, tabs],
        raw_css=[DASHBOARD_CSS],
        main_max_width="1600px",
    )

    logger.info("Observatory dashboard ready.")
    return template
