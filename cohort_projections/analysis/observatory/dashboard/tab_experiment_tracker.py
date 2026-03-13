"""Experiment Tracker tab — catalog browsing, detail inspection, and history.

Displays the variant catalog with filtering, a detail panel for selected
variants, the experiment log, and grid definitions from the
:class:`VariantCatalog`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
)
from cohort_projections.analysis.observatory.dashboard.theme import (
    SDC_NAVY,
    STATUS_COLORS,
    TABULATOR_STYLESHEET,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    empty_placeholder,
    section_header,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tabulator formatters for status badges
# ---------------------------------------------------------------------------

def _status_badge_html(status: object) -> str:
    """Render a small inline status pill for tables."""
    value = str(status or "untested").strip() or "untested"
    color = STATUS_COLORS.get(value, STATUS_COLORS["untested"])
    text_color = "#1F3864" if value == "needs_human_review" else "#FFFFFF"
    return (
        '<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'font-size:0.82em;font-weight:600;text-transform:uppercase;'
        f'background:{color};color:{text_color}">{value.replace("_", " ")}</span>'
    )


# ---------------------------------------------------------------------------
# Section 1: Variant Catalog Table
# ---------------------------------------------------------------------------


def _build_variant_catalog(dm: DashboardDataManager) -> tuple[pn.Column, pn.widgets.Tabulator | None]:
    """Build the variant catalog table with filter widgets.

    Returns
    -------
    tuple[pn.Column, pn.widgets.Tabulator | None]
        The layout column and the Tabulator widget (or None if no data).
    """
    if dm.catalog is None:
        return (
            pn.Column(
                section_header("Variant Catalog"),
                empty_placeholder("Variant catalog YAML not available."),
                sizing_mode="stretch_width",
            ),
            None,
        )

    variants_df = dm.catalog.list_variants()
    if variants_df.empty:
        return (
            pn.Column(
                section_header("Variant Catalog"),
                empty_placeholder("No variants defined in the catalog."),
                sizing_mode="stretch_width",
            ),
            None,
        )

    # Ensure consistent columns for display
    display_cols = [
        "variant_id", "name", "parameter", "value",
        "tier", "config_only", "tested", "resolved_status",
    ]
    display_cols = [c for c in display_cols if c in variants_df.columns]
    catalog_df = variants_df[display_cols].copy()

    # Fill empty status for display
    if "resolved_status" in catalog_df.columns:
        catalog_df["resolved_status"] = (
            catalog_df["resolved_status"].fillna("").replace("", "untested")
        )
        catalog_df["resolved_status"] = catalog_df["resolved_status"].map(_status_badge_html)

    # --- Filter widgets ---
    tier_options = ["All"]
    if "tier" in catalog_df.columns:
        tier_options += sorted(catalog_df["tier"].dropna().unique().tolist())
    tier_select = pn.widgets.Select(
        name="Tier",
        options=tier_options,
        value="All",
        width=120,
    )

    status_options = ["All", "tested", "untested"]
    status_select = pn.widgets.Select(
        name="Status",
        options=status_options,
        value="All",
        width=140,
    )

    # Tabulator widget
    tabulator = pn.widgets.Tabulator(
        catalog_df,
        sizing_mode="stretch_width",
        theme="simple",
        stylesheets=[TABULATOR_STYLESHEET],
        show_index=False,
        selectable=1,
        pagination="remote",
        page_size=15,
        header_filters=False,
        frozen_columns=["variant_id"],
        formatters={
            "resolved_status": "html",
            "tested": {"type": "tickCross"},
            "config_only": {"type": "tickCross"},
        },
    )

    # Keep a reference to the full DataFrame for filtering
    _full_df = catalog_df.copy()

    def _apply_filters(event: Any = None) -> None:
        """Apply tier and status filters to the catalog table."""
        filtered = _full_df.copy()
        tier_val = tier_select.value
        status_val = status_select.value

        if tier_val != "All" and "tier" in filtered.columns:
            filtered = filtered[filtered["tier"] == tier_val]
        if status_val != "All":
            if status_val == "tested" and "tested" in filtered.columns:
                filtered = filtered[filtered["tested"]]
            elif status_val == "untested" and "tested" in filtered.columns:
                filtered = filtered[~filtered["tested"]]

        tabulator.value = filtered.reset_index(drop=True)

    tier_select.param.watch(_apply_filters, "value")
    status_select.param.watch(_apply_filters, "value")

    filter_row = pn.Row(tier_select, status_select, sizing_mode="stretch_width")

    layout = pn.Column(
        section_header(
            "Variant Catalog",
            subtitle="All defined experiment variants — click a row for details",
        ),
        filter_row,
        tabulator,
        sizing_mode="stretch_width",
    )
    return layout, tabulator


# ---------------------------------------------------------------------------
# Section 2: Variant Detail Panel
# ---------------------------------------------------------------------------


def _build_detail_panel(
    dm: DashboardDataManager,
    tabulator: pn.widgets.Tabulator | None,
) -> pn.Column:
    """Build a reactive detail panel for the selected catalog variant.

    Parameters
    ----------
    dm:
        The data manager (provides access to the catalog).
    tabulator:
        The Tabulator widget from section 1.  If None, returns a static
        placeholder.

    Returns
    -------
    pn.Column
        A column that updates when a row is selected.
    """
    detail_pane = pn.pane.HTML(
        '<div style="padding:20px;color:#A0A0A0;font-style:italic;text-align:center">'
        "Select a variant above to view details.</div>",
        sizing_mode="stretch_width",
    )

    if tabulator is None or dm.catalog is None:
        return pn.Column(
            section_header("Variant Detail"),
            detail_pane,
            sizing_mode="stretch_width",
        )

    def _on_selection_change(event: Any) -> None:
        """Update detail panel when a row is selected in the catalog table."""
        selection = tabulator.selection
        if not selection:
            detail_pane.object = (
                '<div style="padding:20px;color:#A0A0A0;font-style:italic;text-align:center">'
                "Select a variant above to view details.</div>"
            )
            return

        row_idx = selection[0]
        current_df = tabulator.value
        if row_idx >= len(current_df):
            return

        variant_id = current_df.iloc[row_idx].get("variant_id", "")
        if not variant_id:
            return

        if dm.catalog is None:
            detail_pane.object = (
                '<div style="padding:10px;color:#C00000">'
                "Variant catalog is not available.</div>"
            )
            return

        try:
            vdef = dm.catalog.get_variant(str(variant_id))
        except KeyError:
            detail_pane.object = (
                f'<div style="padding:10px;color:#C00000">'
                f"Variant '{variant_id}' not found in catalog.</div>"
            )
            return

        _render_variant_detail(detail_pane, vdef)

    tabulator.param.watch(_on_selection_change, "selection")

    return pn.Column(
        section_header("Variant Detail"),
        detail_pane,
        sizing_mode="stretch_width",
    )


def _render_variant_detail(pane: pn.pane.HTML, vdef: dict[str, Any]) -> None:
    """Render variant details into the HTML pane.

    Parameters
    ----------
    pane:
        The target HTML pane to update.
    vdef:
        Variant definition dict from ``VariantCatalog.get_variant()``.
    """
    variant_id = vdef.get("variant_id", "?")
    name = vdef.get("name", "")
    parameter = vdef.get("parameter", "")
    value = vdef.get("value", "")
    tier = vdef.get("tier", "?")
    config_only = vdef.get("config_only", True)
    tested = vdef.get("tested", False)
    hypothesis = vdef.get("hypothesis", "No hypothesis recorded.")
    expected_improvement = vdef.get("expected_improvement", [])
    risk_areas = vdef.get("risk_areas", [])
    slug = vdef.get("slug", "")

    # Results section (if tested)
    results_html = ""
    results = vdef.get("results", {})
    if results:
        status = results.get("status", "")
        status_color = STATUS_COLORS.get(status, "#A0A0A0")
        interpretation = results.get("interpretation", "")
        key_metrics = results.get("key_metrics", {})

        metrics_rows = ""
        if isinstance(key_metrics, dict):
            for k, v in key_metrics.items():
                metrics_rows += (
                    f'<tr><td style="padding:2px 12px 2px 0;color:#595959">{k}</td>'
                    f'<td style="padding:2px 0;font-weight:600">{v}</td></tr>'
                )

        results_html = f"""
        <div style="margin-top:12px;padding-top:12px;border-top:1px solid #D9D9D9">
          <h4 style="margin:0 0 8px 0;color:{SDC_NAVY}">Results</h4>
          <p><span style="
            display:inline-block;padding:2px 10px;border-radius:12px;
            font-size:0.82em;font-weight:600;text-transform:uppercase;
            background-color:{status_color};
            color:{'#1F3864' if status == 'needs_human_review' else '#FFFFFF'}
          ">{status or 'unknown'}</span></p>
          {'<table style="width:100%;border-collapse:collapse">'
           + metrics_rows + '</table>' if metrics_rows else ''}
          {'<p style="margin-top:8px;color:#595959"><strong>Interpretation:</strong> '
           + interpretation + '</p>' if interpretation else ''}
        </div>
        """

    # Expected improvement list
    improvement_html = ""
    if expected_improvement:
        if isinstance(expected_improvement, list):
            items = "".join(f"<li>{item}</li>" for item in expected_improvement)
            improvement_html = f'<p style="margin:4px 0 0 0"><strong>Expected improvement:</strong></p><ul style="margin:2px 0">{items}</ul>'
        else:
            improvement_html = f'<p style="margin:4px 0 0 0"><strong>Expected improvement:</strong> {expected_improvement}</p>'

    # Risk areas list
    risk_html = ""
    if risk_areas:
        if isinstance(risk_areas, list):
            items = "".join(f"<li>{item}</li>" for item in risk_areas)
            risk_html = f'<p style="margin:4px 0 0 0"><strong>Risk areas:</strong></p><ul style="margin:2px 0">{items}</ul>'
        else:
            risk_html = f'<p style="margin:4px 0 0 0"><strong>Risk areas:</strong> {risk_areas}</p>'

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;padding:8px 0">
      <h3 style="margin:0 0 4px 0;color:{SDC_NAVY}">{variant_id}: {name}</h3>
      <table style="border-collapse:collapse;margin-bottom:8px">
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Parameter</td>
          <td style="padding:2px 0;font-weight:600">{parameter}</td>
        </tr>
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Value</td>
          <td style="padding:2px 0;font-weight:600">{value}</td>
        </tr>
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Tier</td>
          <td style="padding:2px 0">{tier}</td>
        </tr>
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Config-only</td>
          <td style="padding:2px 0">{'Yes' if config_only else 'No (code change required)'}</td>
        </tr>
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Slug</td>
          <td style="padding:2px 0">{slug}</td>
        </tr>
        <tr>
          <td style="padding:2px 16px 2px 0;color:#595959">Tested</td>
          <td style="padding:2px 0;font-weight:600;color:{'#00B050' if tested else '#A0A0A0'}">
            {'Yes' if tested else 'No'}
          </td>
        </tr>
      </table>
      <p style="margin:8px 0 4px 0"><strong>Hypothesis:</strong></p>
      <p style="margin:0;padding:8px 12px;background:#F2F2F2;border-radius:4px;
                border-left:3px solid {SDC_NAVY}">{hypothesis}</p>
      {improvement_html}
      {risk_html}
      {results_html}
    </div>
    """
    pane.object = html


# ---------------------------------------------------------------------------
# Section 3: Experiment Log
# ---------------------------------------------------------------------------


def _build_experiment_log(dm: DashboardDataManager) -> pn.Card:
    """Build the experiment log table."""
    log_df = dm.experiment_log

    if log_df.empty:
        return pn.Card(
            empty_placeholder("No experiment log entries found."),
            title="Experiment Log",
            sizing_mode="stretch_width",
        )

    # Select relevant columns if available
    preferred_cols = [
        "experiment_id", "run_date", "hypothesis", "outcome",
        "key_metrics_summary", "interpretation", "next_action",
    ]
    display_cols = [c for c in preferred_cols if c in log_df.columns]
    if not display_cols:
        display_cols = list(log_df.columns)

    display_df = log_df[display_cols].copy()
    if "outcome" in display_df.columns:
        display_df["status"] = display_df["outcome"].map(_status_badge_html)
        display_df = display_df.drop(columns=["outcome"])

    tabulator = pn.widgets.Tabulator(
        display_df,
        sizing_mode="stretch_width",
        theme="simple",
        stylesheets=[TABULATOR_STYLESHEET],
        show_index=False,
        pagination="remote",
        page_size=15,
        header_filters=True,
        frozen_columns=["experiment_id"],
        formatters={"status": "html"},
    )

    return pn.Card(
        tabulator,
        title="Experiment Log",
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Section 4: Grid Definitions
# ---------------------------------------------------------------------------


def _build_grid_definitions(dm: DashboardDataManager) -> pn.Card:
    """Build the grid definitions display."""
    if dm.catalog is None:
        return pn.Card(
            empty_placeholder("Variant catalog not available — no grid definitions."),
            title="Parameter Grids",
            sizing_mode="stretch_width",
        )

    grid_ids = dm.catalog.grid_ids
    if not grid_ids:
        return pn.Card(
            empty_placeholder("No parameter grids defined in the catalog."),
            title="Parameter Grids",
            sizing_mode="stretch_width",
        )

    rows: list[dict[str, Any]] = []
    for grid_id in grid_ids:
        try:
            gdef = dm.catalog.get_grid(grid_id)
        except KeyError:
            continue

        parameters = gdef.get("parameters", {})
        mode = gdef.get("mode", "cartesian")
        hypothesis = gdef.get("hypothesis", "")

        for param_name, param_values in parameters.items():
            if isinstance(param_values, list):
                values_str = ", ".join(str(v) for v in param_values)
                n_values = len(param_values)
                min_val = min(param_values) if param_values and all(isinstance(v, (int, float)) for v in param_values) else ""
                max_val = max(param_values) if param_values and all(isinstance(v, (int, float)) for v in param_values) else ""
            else:
                values_str = str(param_values)
                n_values = 1
                min_val = ""
                max_val = ""

            rows.append({
                "grid_id": grid_id,
                "parameter": param_name,
                "values": values_str,
                "n_values": n_values,
                "min": min_val,
                "max": max_val,
                "mode": mode,
                "hypothesis": hypothesis,
            })

    if not rows:
        return pn.Card(
            empty_placeholder("Grid definitions are empty."),
            title="Parameter Grids",
            sizing_mode="stretch_width",
        )

    grid_df = pd.DataFrame(rows)
    tabulator = pn.widgets.Tabulator(
        grid_df,
        sizing_mode="stretch_width",
        theme="simple",
        stylesheets=[TABULATOR_STYLESHEET],
        show_index=False,
        pagination="remote",
        page_size=20,
        header_filters=True,
        frozen_columns=["grid_id", "parameter"],
    )

    return pn.Card(
        tabulator,
        title="Parameter Grids",
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_experiment_tracker(dm: DashboardDataManager) -> pn.Column:
    """Build the Experiment Tracker tab.

    Displays the variant catalog, detail panel, experiment log, and grid
    definitions with interactive filtering and selection.

    Parameters
    ----------
    dm:
        The :class:`DashboardDataManager` providing all data access.

    Returns
    -------
    pn.Column
        A Panel Column containing the four tracker sections.
    """
    catalog_layout, tabulator = _build_variant_catalog(dm)
    detail_panel = _build_detail_panel(dm, tabulator)
    experiment_log = _build_experiment_log(dm)
    grid_defs = _build_grid_definitions(dm)

    return pn.Column(
        section_header(
            "Experiment Tracker",
            subtitle="Catalog, history, and grid definitions",
        ),
        catalog_layout,
        detail_panel,
        experiment_log,
        grid_defs,
        sizing_mode="stretch_width",
    )
