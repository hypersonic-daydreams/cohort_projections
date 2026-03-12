"""Main Panel application assembly for the Projection Observatory dashboard.

Brings together all six tab modules into a unified ``FastListTemplate``
application with SDC branding.
"""

from __future__ import annotations

import logging

import panel as pn

from .data_manager import DashboardDataManager
from .theme import DASHBOARD_CSS, SDC_NAVY, SDC_WHITE

logger = logging.getLogger(__name__)


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
    from .tab_experiment_tracker import build_experiment_tracker
    from .tab_horizon_bias import build_horizon_bias_tab
    from .tab_projection_ensemble import build_projection_ensemble
    from .tab_scorecard import build_scorecard_tab
    from .tab_sensitivity import build_sensitivity_tab

    # Build tabs --------------------------------------------------------
    logger.info("Building dashboard tabs …")

    tabs = pn.Tabs(
        ("Command Center", build_command_center(dm)),
        ("Experiments", build_experiment_tracker(dm)),
        ("Scorecards", build_scorecard_tab(dm)),
        ("Projections", build_projection_ensemble(dm)),
        ("Horizon & Bias", build_horizon_bias_tab(dm)),
        ("Sensitivity", build_sensitivity_tab(dm)),
        tabs_location="above",
        dynamic=True,
    )

    # Assemble template -------------------------------------------------
    template = pn.template.FastListTemplate(
        title="Projection Observatory",
        header_background=SDC_NAVY,
        header_color=SDC_WHITE,
        accent_base_color=SDC_NAVY,
        shadow=True,
        main=[tabs],
        raw_css=[DASHBOARD_CSS],
        main_max_width="1600px",
    )

    logger.info("Observatory dashboard ready.")
    return template
