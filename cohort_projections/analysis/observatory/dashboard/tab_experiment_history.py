"""Merged Experiment History tab — catalog, timeline, trends, and log.

Combines the Experiment Tracker (catalog, detail panel, experiment log, grid
definitions) and the History tab (run timeline, champion history, delta heatmap,
category trends, outcome history) into a single tab with nested sub-tabs.

All builder functions are imported from the original modules, which remain
unchanged as library modules.
"""

from __future__ import annotations

import logging

import panel as pn

from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
)
from cohort_projections.analysis.observatory.dashboard.tab_experiment_tracker import (
    _build_detail_panel,
    _build_experiment_log,
    _build_grid_definitions,
    _build_variant_catalog,
)
from cohort_projections.analysis.observatory.dashboard.tab_history import (
    _build_category_trend_chart,
    _build_champion_history_chart,
    _build_history_table,
    _build_metric_delta_heatmap,
    _build_outcome_history_chart,
    _history_takeaway_text,
    _selected_statuses,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    markdown_card,
    section_header,
)

logger = logging.getLogger(__name__)


def build_experiment_history(dm: DashboardDataManager) -> pn.Column:
    """Build the merged Experiment History tab.

    Combines variant catalog browsing, run timeline, benchmark trends, and
    the experiment log into four nested sub-tabs under a single top-level
    section.

    Parameters
    ----------
    dm:
        The :class:`DashboardDataManager` providing all data access.

    Returns
    -------
    pn.Column
        A Panel Column containing the section header, takeaway card, and
        nested sub-tabs.
    """
    # -- Takeaway card (history summary) --
    takeaway = markdown_card(
        "Executive Summary",
        _history_takeaway_text(dm),
        min_width=420,
    )

    # -- Catalog sub-tab --
    catalog_layout, tabulator = _build_variant_catalog(dm)
    detail_panel = _build_detail_panel(dm, tabulator)
    catalog_tab = pn.Column(
        catalog_layout,
        detail_panel,
        sizing_mode="stretch_width",
    )

    # -- Timeline sub-tab --
    statuses = _selected_statuses(None, dm)
    timeline_tab = pn.Column(
        _build_history_table(dm, statuses),
        _build_champion_history_chart(dm, statuses),
        sizing_mode="stretch_width",
    )

    # -- Trends sub-tab --
    trends_tab = pn.Column(
        _build_metric_delta_heatmap(dm, statuses, None),
        _build_category_trend_chart(dm, statuses),
        _build_outcome_history_chart(dm, statuses),
        sizing_mode="stretch_width",
    )

    # -- Log sub-tab --
    log_tab = pn.Column(
        _build_experiment_log(dm),
        _build_grid_definitions(dm),
        sizing_mode="stretch_width",
    )

    # -- Assemble nested sub-tabs --
    sub_tabs = pn.Tabs(
        ("Catalog", catalog_tab),
        ("Timeline", timeline_tab),
        ("Trends", trends_tab),
        ("Log", log_tab),
        dynamic=True,
    )

    return pn.Column(
        section_header(
            "Experiment History",
            tooltip=(
                "Browse the variant catalog, review experiment history, "
                "and track benchmark trends over time."
            ),
        ),
        takeaway,
        sub_tabs,
        sizing_mode="stretch_width",
    )
