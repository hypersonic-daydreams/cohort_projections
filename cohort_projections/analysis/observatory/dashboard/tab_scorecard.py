"""Scorecard comparison tab for the Observatory dashboard.

The tab compares selected benchmark bundles against the current champion using
one comparison row per bundle. This avoids showing repeated champion rows from
every benchmark run and keeps the charts focused on actual challenger variants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from cohort_projections.analysis.observatory.comparator import (
    METRIC_COLUMNS,
    SENTINEL_COLUMNS,
)
from cohort_projections.analysis.observatory.dashboard import theme, widgets
from cohort_projections.analysis.observatory.dashboard.data_manager import (
    RUN_SELECTION_PRESETS,
)

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

logger = logging.getLogger(__name__)

_DELTA_METRICS: list[tuple[str, str]] = [
    ("county_mape_overall", "Overall"),
    ("county_mape_urban_college", "College"),
    ("county_mape_rural", "Rural"),
    ("county_mape_bakken", "Bakken"),
]
_SENTINEL_LABELS: dict[str, str] = {
    "sentinel_cass_mape": "Cass",
    "sentinel_grand_forks_mape": "Grand Forks",
    "sentinel_ward_mape": "Ward",
    "sentinel_burleigh_mape": "Burleigh",
    "sentinel_williams_mape": "Williams",
    "sentinel_mckenzie_mape": "McKenzie",
}


def _run_color(run_id: str, run_ids: list[str]) -> str:
    """Return a consistent color for a run based on its position."""
    idx = run_ids.index(run_id) if run_id in run_ids else 0
    return theme.EXPERIMENT_COLORS[idx % len(theme.EXPERIMENT_COLORS)]


def _selected_scorecard_rows(
    selected_runs: list[str],
    dm: DashboardDataManager,
    *,
    include_champion: bool = True,
) -> pd.DataFrame:
    """Return one scorecard row per selected bundle plus optional champion."""
    comparison_rows = dm.comparison_rows.copy()
    if comparison_rows.empty:
        return pd.DataFrame()

    chosen_run_ids = list(selected_runs)
    if include_champion and dm.champion_id is not None:
        chosen_run_ids = [dm.champion_id, *chosen_run_ids]

    rows = comparison_rows[comparison_rows["run_id"].isin(chosen_run_ids)].copy()
    if rows.empty:
        return rows

    metadata = dm.run_metadata.set_index("run_id", drop=False)
    rows["run"] = rows["run_id"].map(lambda run_id: dm.run_label(str(run_id), short=True))
    rows["run_long"] = rows["run_id"].map(dm.run_label)
    rows["status"] = rows["run_id"].map(
        lambda run_id: (
            str(metadata.loc[run_id, "status_label"])
            if run_id in metadata.index
            else "Unknown"
        )
    )
    rows["role"] = rows["run_id"].map(
        lambda run_id: "Champion" if run_id == dm.champion_id else "Selected Variant"
    )
    rows = rows.drop_duplicates(subset=["run_id", "method_id", "config_id"])

    sort_order: list[str] = []
    if dm.champion_id is not None:
        rows["_champion_first"] = (rows["run_id"] != dm.champion_id).astype(int)
        sort_order.append("_champion_first")
    if "county_mape_overall" in rows.columns:
        sort_order.append("county_mape_overall")
    if sort_order:
        rows = rows.sort_values(sort_order, ascending=True, na_position="last")
    return rows.drop(columns=[col for col in ["_champion_first"] if col in rows.columns])


def _build_scorecard_table(
    selected_runs: list[str],
    dm: DashboardDataManager,
) -> pn.Column:
    """Render one comparison row per selected bundle plus the champion row."""
    if not selected_runs and dm.champion_id is None:
        return pn.Column(widgets.empty_placeholder("Select at least one benchmark bundle."))

    filtered = _selected_scorecard_rows(selected_runs, dm)
    if filtered.empty:
        return pn.Column(widgets.empty_placeholder("No comparison rows found for the current selection."))

    display_cols = [
        col
        for col in [
            "run",
            "role",
            "status",
            "method_id",
            "config_id",
            "county_mape_overall",
            "county_mape_rural",
            "county_mape_bakken",
            "county_mape_urban_college",
            "state_ape_recent_short",
            "state_ape_recent_medium",
            "run_id",
        ]
        if col in filtered.columns
    ]
    display_df = filtered[display_cols].rename(
        columns={
            "method_id": "method",
            "config_id": "config",
            "run_id": "exact_run_id",
        }
    )

    return widgets.metric_table(
        display_df,
        title="Scorecard Comparison",
        highlight_cols=[
            col
            for col in [
                "county_mape_overall",
                "county_mape_rural",
                "county_mape_bakken",
                "county_mape_urban_college",
                "state_ape_recent_short",
                "state_ape_recent_medium",
            ]
            if col in display_df.columns
        ],
        page_size=0,
        frozen_columns=["run"],
    )


def _build_delta_bar_chart(
    selected_runs: list[str],
    dm: DashboardDataManager,
) -> pn.pane.Plotly:
    """Grouped bar chart of key MAPE deltas vs the global champion."""
    comparison = _selected_scorecard_rows(selected_runs, dm)
    if comparison.empty or dm.champion_id is None:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    champion_rows = comparison[comparison["run_id"] == dm.champion_id]
    if champion_rows.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")
    champion = champion_rows.iloc[0]

    fig = go.Figure()
    challengers = comparison[comparison["run_id"] != dm.champion_id]
    for _, row in challengers.iterrows():
        labels: list[str] = []
        values: list[float] = []
        for col, label in _DELTA_METRICS:
            if col in comparison.columns and pd.notna(row.get(col)) and pd.notna(champion.get(col)):
                labels.append(label)
                values.append(float(row[col]) - float(champion[col]))

        if not labels:
            continue

        run_id = str(row["run_id"])
        fig.add_trace(
            go.Bar(
                name=str(row["run"]),
                x=labels,
                y=values,
                marker_color=_run_color(run_id, dm.ordered_run_ids),
                text=[f"{value:+.3f}" for value in values],
                textposition="outside",
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:+.4f} pp<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1, line_color=theme.SDC_DARK_GRAY)
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="MAPE Delta vs Champion (negative = improvement)",
        yaxis_title="Delta (percentage points)",
        barmode="group",
        legend={"orientation": "h", "y": -0.18},
        height=420,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_sentinel_chart(
    selected_runs: list[str],
    dm: DashboardDataManager,
) -> pn.pane.Plotly:
    """Grouped sentinel-county chart for the selected challengers."""
    comparison = _selected_scorecard_rows(selected_runs, dm)
    if comparison.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    available_sentinels = [
        (col, label)
        for col, label in _SENTINEL_LABELS.items()
        if col in comparison.columns
    ]
    if not available_sentinels:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()
    for _, row in comparison.iterrows():
        run_id = str(row["run_id"])
        fig.add_trace(
            go.Bar(
                name=str(row["run"]),
                x=[label for _, label in available_sentinels],
                y=[float(row.get(col, 0)) for col, _ in available_sentinels],
                marker_color=_run_color(run_id, dm.ordered_run_ids),
                text=[f"{float(row.get(col, 0)):.2f}" for col, _ in available_sentinels],
                textposition="outside",
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.3f}%<extra></extra>",
            )
        )

    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Sentinel County MAPE",
        yaxis_title="MAPE (%)",
        barmode="group",
        legend={"orientation": "h", "y": -0.18},
        height=420,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_pareto_scatter(
    selected_runs: list[str],
    x_metric: str,
    y_metric: str,
    dm: DashboardDataManager,
) -> pn.pane.Plotly:
    """Scatter selected bundles and highlight the local Pareto frontier."""
    comparison = _selected_scorecard_rows(selected_runs, dm)
    if comparison.empty or x_metric not in comparison.columns or y_metric not in comparison.columns:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    filtered = comparison.dropna(subset=[x_metric, y_metric]).copy()
    if filtered.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    pareto_mask: list[bool] = []
    xs = filtered[x_metric].tolist()
    ys = filtered[y_metric].tolist()
    for i in range(len(filtered)):
        dominated = False
        for j in range(len(filtered)):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                dominated = True
                break
        pareto_mask.append(not dominated)

    filtered["_pareto"] = pareto_mask
    fig = go.Figure()

    others = filtered[~filtered["_pareto"]]
    if not others.empty:
        fig.add_trace(
            go.Scatter(
                x=others[x_metric],
                y=others[y_metric],
                mode="markers+text",
                marker={
                    "size": 11,
                    "color": theme.SDC_MID_GRAY,
                    "line": {"width": 1, "color": theme.SDC_DARK_GRAY},
                },
                text=others["run"],
                textposition="top center",
                name="Other selected runs",
                hovertemplate="<b>%{text}</b><br>" + f"{x_metric}: %{{x:.4f}}<br>{y_metric}: %{{y:.4f}}<extra></extra>",
            )
        )

    pareto = filtered[filtered["_pareto"]]
    if not pareto.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto[x_metric],
                y=pareto[y_metric],
                mode="markers+text",
                marker={
                    "size": 15,
                    "color": [
                        _run_color(str(run_id), dm.ordered_run_ids)
                        for run_id in pareto["run_id"]
                    ],
                    "symbol": "star",
                    "line": {"width": 1, "color": theme.SDC_NAVY},
                },
                text=pareto["run"],
                textposition="top center",
                name="Pareto-optimal",
                hovertemplate="<b>%{text}</b><br>" + f"{x_metric}: %{{x:.4f}}<br>{y_metric}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Pareto Frontier",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=470,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_county_group_table(dm: DashboardDataManager) -> pn.Column:
    """Best variant per county group and delta table."""
    best = dm.comparator.best_variant_per_group()
    if not best:
        return pn.Column(widgets.empty_placeholder("No county group data available."))

    rows = []
    for group, info in best.items():
        rows.append(
            {
                "County Group": group,
                "Best Run": dm.run_label(str(info.get("run_id", "?")), short=True),
                "Config": info.get("config_id", "?"),
            }
        )

    best_df = pd.DataFrame(rows)
    impact = dm.comparator.county_group_impact().copy()
    if not impact.empty and "run_id" in impact.columns:
        impact["run"] = impact["run_id"].map(lambda run_id: dm.run_label(str(run_id), short=True))
        cols = ["run"] + [col for col in impact.columns if col.startswith("delta_")]
        impact = impact[cols]

    if impact.empty:
        return widgets.metric_table(best_df, title="Best Variant per County Group", page_size=0)

    return pn.Column(
        widgets.metric_table(
            best_df,
            title="Best Variant per County Group",
            page_size=0,
            frozen_columns=["County Group"],
        ),
        widgets.metric_table(
            impact,
            title="County Group Impact (delta vs champion)",
            highlight_cols=[col for col in impact.columns if col.startswith("delta_")],
            page_size=0,
            frozen_columns=["run"],
        ),
    )


def build_scorecard_tab(dm: DashboardDataManager) -> pn.Column:
    """Build the scorecard comparison tab."""
    if not dm.run_ids:
        return pn.Column(
            widgets.section_header("Scorecard Comparison", "No benchmark runs found."),
            widgets.empty_placeholder("Run benchmarks first to populate this tab."),
        )

    default_runs = dm.preset_run_ids(RUN_SELECTION_PRESETS[0])
    run_selector = pn.widgets.MultiChoice(
        name="Runs",
        options=dm.run_option_map(),
        value=default_runs,
        placeholder="Search benchmark bundles...",
        delete_button=True,
        search_option_limit=12,
        max_items=max(6, len(dm.run_ids)),
        sizing_mode="stretch_width",
    )
    preset_selector = pn.widgets.Select(
        name="Comparison Preset",
        options=list(RUN_SELECTION_PRESETS),
        value=RUN_SELECTION_PRESETS[0],
        width=240,
    )

    def _apply_preset(event: object) -> None:
        del event
        run_selector.value = dm.preset_run_ids(str(preset_selector.value))

    preset_selector.param.watch(_apply_preset, "value")

    metric_options = [col for col in METRIC_COLUMNS if col in dm.comparison_rows.columns]
    metric_options += [col for col in SENTINEL_COLUMNS if col in dm.comparison_rows.columns]
    if not metric_options:
        metric_options = ["county_mape_overall"]
    pareto_x = pn.widgets.Select(
        name="X Axis Metric",
        options=metric_options,
        value="county_mape_overall" if "county_mape_overall" in metric_options else metric_options[0],
        sizing_mode="stretch_width",
    )
    pareto_y = pn.widgets.Select(
        name="Y Axis Metric",
        options=metric_options,
        value="state_ape_recent_short" if "state_ape_recent_short" in metric_options else metric_options[min(1, len(metric_options) - 1)],
        sizing_mode="stretch_width",
    )

    scorecard_table_bound = pn.panel(
        pn.bind(_build_scorecard_table, selected_runs=run_selector, dm=dm),
        loading_indicator=True,
    )
    delta_chart = pn.panel(
        pn.bind(_build_delta_bar_chart, selected_runs=run_selector, dm=dm),
        loading_indicator=True,
    )
    sentinel_chart = pn.panel(
        pn.bind(_build_sentinel_chart, selected_runs=run_selector, dm=dm),
        loading_indicator=True,
    )
    pareto_chart = pn.panel(
        pn.bind(
            _build_pareto_scatter,
            selected_runs=run_selector,
            x_metric=pareto_x,
            y_metric=pareto_y,
            dm=dm,
        ),
        loading_indicator=True,
    )

    return pn.Column(
        widgets.section_header(
            "Scorecard Comparison",
            "Focused challenger comparison against the current champion.",
        ),
        pn.Card(
            pn.pane.HTML(
                '<div class="filters-help">Presets start from readable experiment bundles instead of raw run IDs. The champion reference row stays visible in the scorecard views.</div>',
                sizing_mode="stretch_width",
                stylesheets=[theme.DASHBOARD_CSS],
            ),
            pn.FlexBox(
                preset_selector,
                flex_wrap="wrap",
                sizing_mode="stretch_width",
                styles={"gap": "10px"},
            ),
            run_selector,
            title="Comparison Selector",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            scorecard_table_bound,
            title="Side-by-Side Scorecard",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            delta_chart,
            title="MAPE Delta vs Champion",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            sentinel_chart,
            title="Sentinel County Comparison",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.FlexBox(
                pareto_x,
                pareto_y,
                flex_wrap="wrap",
                sizing_mode="stretch_width",
                styles={"gap": "10px"},
            ),
            pareto_chart,
            title="Pareto Frontier",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            _build_county_group_table(dm),
            title="County Group Impact",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
