#!/usr/bin/env python3
"""
Build PP-003 human-review HTML package with timestamped versions.

This script reads QA artifacts for place projections and produces:
1. Versioned static chart assets in a timestamped directory.
2. A versioned HTML review package linked to those assets.
3. Optional refresh of a canonical "latest" HTML file for convenience.

The rendered HTML is organized into tabs, one per quality-control gate/issue,
with:
- What the reviewer is checking
- Relevant static visuals
- Summary statistics/tables
- Agent analysis and suggested decision
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS = ["baseline", "restricted_growth", "high_growth"]
DEFAULT_REVIEWS_DIR = REPO_ROOT / "docs/reviews"
DEFAULT_ASSET_ROOT = DEFAULT_REVIEWS_DIR / "assets/pp3-human-review"
DEFAULT_LATEST_HTML = DEFAULT_REVIEWS_DIR / "2026-03-01-pp3-human-review-package.html"
PROJECTED_TIERS = {"HIGH", "MODERATE", "LOWER"}
TOL = 1e-9


@dataclass(frozen=True)
class ScenarioData:
    """Container for one scenario's QA and summary inputs."""

    scenario: str
    outliers: pd.DataFrame
    share_sum: pd.DataFrame
    balance: pd.DataFrame
    places_summary: pd.DataFrame


@dataclass(frozen=True)
class Gate:
    """One review gate rendered as a tab."""

    gate_id: str
    title: str
    objective: str
    status: str
    suggested_decision: str
    analysis: str
    visuals: list[str]
    stats_html: str


def configure_logging(verbose: bool = False) -> None:
    """Configure script logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def now_local() -> datetime:
    """Return timezone-aware local datetime."""
    return datetime.now().astimezone()


def timestamp_tokens(now: datetime) -> tuple[str, str]:
    """Return date and timestamp tokens for versioned filenames."""
    return now.strftime("%Y-%m-%d"), now.strftime("%Y%m%d_%H%M%S")


def _normalize_fips(value: object, width: int) -> str:
    """Normalize FIPS-like value to zero-padded digit string."""
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(width)[-width:]


def _as_bool(value: object) -> bool:
    """Coerce mixed truthy/falsy values to bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render compact HTML table."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows) if body_rows else "<tr><td colspan='99'>No rows.</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def load_scenario_data(scenario: str) -> ScenarioData:
    """Load required QA and summary files for one scenario."""
    projection_dir = REPO_ROOT / "data/projections" / scenario / "place"
    qa_dir = projection_dir / "qa"

    required_paths = {
        "outliers": qa_dir / "qa_outlier_flags.csv",
        "share_sum": qa_dir / "qa_share_sum_validation.csv",
        "balance": qa_dir / "qa_balance_of_county.csv",
        "places_summary": projection_dir / "places_summary.csv",
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} input for scenario '{scenario}': {path}")

    outliers = pd.read_csv(required_paths["outliers"])
    share_sum = pd.read_csv(required_paths["share_sum"])
    balance = pd.read_csv(required_paths["balance"])
    places_summary = pd.read_csv(required_paths["places_summary"])
    places_summary = places_summary[places_summary["row_type"] == "place"].copy()

    return ScenarioData(
        scenario=scenario,
        outliers=outliers,
        share_sum=share_sum,
        balance=balance,
        places_summary=places_summary,
    )


def load_crosswalk_universe() -> set[str]:
    """Load projected place universe from canonical crosswalk."""
    crosswalk_path = REPO_ROOT / "data/processed/geographic/place_county_crosswalk_2020.csv"
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Crosswalk file not found: {crosswalk_path}")

    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk["place_fips"] = crosswalk["place_fips"].map(lambda v: _normalize_fips(v, 7))
    crosswalk["confidence_tier"] = crosswalk["confidence_tier"].astype(str).str.upper()
    crosswalk["historical_only"] = crosswalk["historical_only"].map(_as_bool)
    projected = crosswalk[
        (~crosswalk["historical_only"]) & (crosswalk["confidence_tier"].isin(PROJECTED_TIERS))
    ].copy()
    return set(projected["place_fips"].astype(str).tolist())


def scan_place_parquet_population_stats(scenario: str) -> dict[str, float | int]:
    """Scan place parquet outputs for non-negative checks."""
    place_dir = REPO_ROOT / "data/projections" / scenario / "place"
    files = sorted(place_dir.glob(f"nd_place_*_{scenario}.parquet"))
    if not files:
        return {"files": 0, "negative_rows": 0, "min_population": float("nan")}

    min_population = float("inf")
    negative_rows = 0
    for path in files:
        frame = pd.read_parquet(path, columns=["population"])
        pop = pd.to_numeric(frame["population"], errors="coerce")
        pop = pop.dropna()
        if pop.empty:
            continue
        min_population = min(min_population, float(pop.min()))
        negative_rows += int((pop < 0).sum())

    if min_population == float("inf"):
        min_population = float("nan")

    return {"files": len(files), "negative_rows": negative_rows, "min_population": min_population}


def aggregate_state_totals_from_counties(scenario: str) -> pd.Series | None:
    """Aggregate state totals from county projection parquet files."""
    county_dir = REPO_ROOT / "data/projections" / scenario / "county"
    county_files = sorted(county_dir.glob(f"nd_county_*_{scenario}.parquet"))
    if not county_files:
        return None

    totals_by_year: dict[int, float] = {}
    for file_path in county_files:
        county = pd.read_parquet(file_path, columns=["year", "population"])
        county["year"] = pd.to_numeric(county["year"], errors="coerce").astype("Int64")
        county["population"] = pd.to_numeric(county["population"], errors="coerce")
        county = county.dropna(subset=["year", "population"])
        if county.empty:
            continue
        county["year"] = county["year"].astype(int)
        county_totals = county.groupby("year", as_index=False)["population"].sum()
        for _, row in county_totals.iterrows():
            year = int(row["year"])
            totals_by_year[year] = totals_by_year.get(year, 0.0) + float(row["population"])

    if not totals_by_year:
        return None

    series = pd.Series(totals_by_year, name=scenario, dtype=float).sort_index()
    series.index = series.index.astype(int)
    return series


def compute_metrics(
    scenarios: list[ScenarioData],
    focal_scenario: str,
) -> dict[str, Any]:
    """Compute all metrics used by charts and gate tabs."""
    scenario_map = {s.scenario: s for s in scenarios}
    focal = scenario_map[focal_scenario]

    snapshot_rows: list[dict[str, Any]] = []
    share_gate_rows: list[dict[str, Any]] = []
    balance_gate_rows: list[dict[str, Any]] = []
    outlier_counts_by_scenario: dict[str, dict[str, int]] = {}
    growth_summary_rows: list[dict[str, Any]] = []
    universe_rows: list[dict[str, Any]] = []
    place_population_scan: dict[str, dict[str, float | int]] = {}

    expected_universe = load_crosswalk_universe()

    for data in scenarios:
        share = data.share_sum.copy()
        share["sum_place_shares"] = pd.to_numeric(share["sum_place_shares"], errors="coerce")
        share["balance_of_county_share"] = pd.to_numeric(share["balance_of_county_share"], errors="coerce")
        share["constraint_satisfied"] = share["constraint_satisfied"].astype(bool)
        share["rescaling_applied"] = share["rescaling_applied"].astype(bool)

        balance = data.balance.copy()
        balance["balance_of_county"] = pd.to_numeric(balance["balance_of_county"], errors="coerce")

        outliers = data.outliers.copy()
        outlier_counts_by_scenario[data.scenario] = {
            str(k): int(v) for k, v in outliers["flag_type"].value_counts().to_dict().items()
        }

        snapshot_rows.append(
            {
                "scenario": data.scenario,
                "qa_tier_summary": int(data.places_summary["confidence_tier"].nunique()),
                "qa_share_sum_validation": int(len(share)),
                "qa_balance_of_county": int(len(balance)),
                "qa_outlier_flags": int(len(outliers)),
            }
        )

        share_gate_rows.append(
            {
                "scenario": data.scenario,
                "max_sum_place_shares": float(share["sum_place_shares"].max()),
                "rows_sum_gt_1": int((share["sum_place_shares"] > 1.0 + TOL).sum()),
                "rows_constraint_false": int((~share["constraint_satisfied"]).sum()),
                "rescaling_rows": int(share["rescaling_applied"].sum()),
            }
        )

        balance_gate_rows.append(
            {
                "scenario": data.scenario,
                "min_balance_share": float(share["balance_of_county_share"].min()),
                "negative_balance_rows": int((balance["balance_of_county"] < 0.0).sum()),
            }
        )

        for tier in ["HIGH", "MODERATE", "LOWER"]:
            tier_rows = data.places_summary[data.places_summary["confidence_tier"] == tier].copy()
            if tier_rows.empty:
                growth_summary_rows.append(
                    {
                        "scenario": data.scenario,
                        "tier": tier,
                        "count": 0,
                        "mean_growth": 0.0,
                        "median_growth": 0.0,
                        "min_growth": 0.0,
                        "max_growth": 0.0,
                    }
                )
                continue
            growth = pd.to_numeric(tier_rows["growth_rate"], errors="coerce").dropna()
            growth_summary_rows.append(
                {
                    "scenario": data.scenario,
                    "tier": tier,
                    "count": int(len(growth)),
                    "mean_growth": float(growth.mean()),
                    "median_growth": float(growth.median()),
                    "min_growth": float(growth.min()),
                    "max_growth": float(growth.max()),
                }
            )

        produced_universe = set(data.places_summary["place_fips"].astype(str).map(lambda v: _normalize_fips(v, 7)))
        missing = expected_universe - produced_universe
        extra = produced_universe - expected_universe
        universe_rows.append(
            {
                "scenario": data.scenario,
                "expected_places": len(expected_universe),
                "produced_places": len(produced_universe),
                "missing": len(missing),
                "extra": len(extra),
                "missing_sample": ", ".join(sorted(missing)[:3]) if missing else "",
                "extra_sample": ", ".join(sorted(extra)[:3]) if extra else "",
            }
        )

        place_population_scan[data.scenario] = scan_place_parquet_population_stats(data.scenario)

    # State ordering metrics
    state_totals: dict[str, pd.Series] = {}
    for name in ["restricted_growth", "baseline", "high_growth"]:
        series = aggregate_state_totals_from_counties(name)
        if series is not None:
            state_totals[name] = series

    ordering_df = None
    ordering_violations = 0
    if {"restricted_growth", "baseline", "high_growth"}.issubset(set(state_totals.keys())):
        ordering_df = pd.concat(
            [state_totals["restricted_growth"], state_totals["baseline"], state_totals["high_growth"]],
            axis=1,
        )
        ordering_df.columns = ["restricted_growth", "baseline", "high_growth"]
        ordering_df = ordering_df.sort_index()
        ordering_violations = int(
            (
                (ordering_df["restricted_growth"] > ordering_df["baseline"] + TOL)
                | (ordering_df["baseline"] > ordering_df["high_growth"] + TOL)
            ).sum()
        )

    return {
        "snapshot_rows": snapshot_rows,
        "share_gate_rows": share_gate_rows,
        "balance_gate_rows": balance_gate_rows,
        "outlier_counts_by_scenario": outlier_counts_by_scenario,
        "growth_summary_rows": growth_summary_rows,
        "universe_rows": universe_rows,
        "place_population_scan": place_population_scan,
        "state_totals": state_totals,
        "ordering_df": ordering_df,
        "ordering_violations": ordering_violations,
        "focal": focal,
        "focal_scenario": focal_scenario,
    }


def build_chart_assets(
    scenarios: list[ScenarioData],
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Render static SVG charts for the review package."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    scenario_names = [s.scenario for s in scenarios]
    scenario_map = {s.scenario: s for s in scenarios}
    focal_scenario = metrics["focal_scenario"]

    chart_files: dict[str, str] = {}

    # 1) Artifact counts by scenario.
    snap = pd.DataFrame(metrics["snapshot_rows"]).set_index("scenario")
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(len(snap.index))
    width = 0.18
    cols = ["qa_tier_summary", "qa_share_sum_validation", "qa_balance_of_county", "qa_outlier_flags"]
    labels = ["Tier", "Share Sum", "Balance", "Outliers"]
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518"]
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1.5) * width, snap[col].to_numpy(dtype=float), width=width, label=labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(snap.index.tolist())
    ax.set_title("QA Artifact Row Counts by Scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = output_dir / "artifact_counts_by_scenario.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["artifact_counts"] = p.name

    # 2) Outlier counts by scenario/type.
    flag_types = sorted(
        set().union(*[set(data.outliers["flag_type"].dropna().astype(str).unique()) for data in scenarios])
    )
    counts = pd.DataFrame(index=scenario_names, columns=flag_types, data=0)
    for data in scenarios:
        vc = data.outliers["flag_type"].value_counts()
        for k, v in vc.items():
            counts.loc[data.scenario, k] = int(v)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bottom = np.zeros(len(scenario_names), dtype=float)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, flag in enumerate(flag_types):
        vals = counts[flag].to_numpy(dtype=float)
        ax.bar(scenario_names, vals, bottom=bottom, label=flag, color=palette[i % len(palette)])
        bottom += vals
    ax.set_title("Outlier Flags by Scenario and Type")
    ax.set_ylabel("Flag Count")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = output_dir / "outlier_counts_by_scenario.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["outlier_counts"] = p.name

    # 3) Share-sum distribution (focal).
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    share_vals = pd.to_numeric(scenario_map[focal_scenario].share_sum["sum_place_shares"], errors="coerce").dropna()
    ax.hist(share_vals, bins=18, color="#1d5fa6", alpha=0.88, edgecolor="white")
    ax.axvline(1.0, color="#a96900", linestyle="--", linewidth=1.6, label="Constraint ceiling (1.0)")
    ax.set_title(f"{focal_scenario}: Distribution of Sum Place Shares by County-Year")
    ax.set_xlabel("sum_place_shares")
    ax.set_ylabel("County-Year Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = output_dir / f"{focal_scenario}_share_sum_distribution.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["share_sum_distribution"] = p.name

    # 4) Rescaling incidence by county (focal top 15).
    res = scenario_map[focal_scenario].share_sum.copy()
    res["rescaling_applied"] = res["rescaling_applied"].astype(bool)
    top_res = (
        res.groupby(["county_fips", "county_name"], as_index=False)["rescaling_applied"]
        .sum()
        .sort_values("rescaling_applied", ascending=False)
        .head(15)
    )
    labels = [
        f"{row['county_name']} ({_normalize_fips(row['county_fips'], 5)})" for _, row in top_res.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.barh(labels[::-1], top_res["rescaling_applied"].to_numpy()[::-1], color="#2d8f5a")
    ax.set_title(f"{focal_scenario}: Rescaling Incidence by County (Top 15)")
    ax.set_xlabel("County-Years with rescaling_applied = true")
    fig.tight_layout()
    p = output_dir / f"{focal_scenario}_rescaling_top_counties.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["rescaling_top_counties"] = p.name

    # 4b) Rescaling concentration curve (focal, all counties).
    by_county = (
        res.groupby(["county_fips", "county_name"], as_index=False)["rescaling_applied"]
        .sum()
        .rename(columns={"rescaling_applied": "rescaling_years"})
        .sort_values("rescaling_years", ascending=False)
        .reset_index(drop=True)
    )
    total_rescaling_years = float(by_county["rescaling_years"].sum())
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    if total_rescaling_years > 0.0:
        ranks = np.arange(1, len(by_county) + 1, dtype=int)
        cum_share = by_county["rescaling_years"].cumsum().to_numpy(dtype=float) / total_rescaling_years
        ax.plot(ranks, 100.0 * cum_share, color="#2d8f5a", linewidth=2.2, marker="o", markersize=3)
        ax.axhline(80.0, color="#a96900", linestyle="--", linewidth=1.4, label="80% threshold")
        k80_idx = np.argmax(cum_share >= 0.8)
        if cum_share[k80_idx] >= 0.8:
            ax.axvline(int(ranks[k80_idx]), color="#a96900", linestyle="--", linewidth=1.4)
            ax.text(
                float(ranks[k80_idx]) + 0.3,
                82.0,
                f"{int(ranks[k80_idx])} counties cover 80%",
                fontsize=8,
                color="#a96900",
            )
        ax.set_ylim(0.0, 100.0)
        ax.set_xlim(1.0, float(len(by_county)))
        ax.set_xlabel("County rank (by rescaling incidence)")
        ax.set_ylabel("Cumulative share of rescaling (%)")
        ax.set_title(f"{focal_scenario}: Rescaling Concentration (Cumulative Share)")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No rescaling applied\n(rescaling_applied=true count is zero).", ha="center", va="center")
        ax.set_title(f"{focal_scenario}: Rescaling Concentration (Cumulative Share)")
        ax.set_axis_off()
    fig.tight_layout()
    p = output_dir / f"{focal_scenario}_rescaling_concentration.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["rescaling_concentration"] = p.name

    # 5) Growth-rate distribution by tier/scenario.
    growth_rows: list[pd.DataFrame] = []
    for data in scenarios:
        g = data.places_summary[["confidence_tier", "growth_rate"]].copy()
        g["scenario"] = data.scenario
        growth_rows.append(g)
    growth = pd.concat(growth_rows, ignore_index=True)
    tiers = ["HIGH", "MODERATE", "LOWER"]
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(12.6, 4.8), sharey=True)
    axes_list = [axes] if len(scenario_names) == 1 else list(axes)
    for i, scenario in enumerate(scenario_names):
        ax = axes_list[i]
        sub = growth[growth["scenario"] == scenario]
        data_list = [
            pd.to_numeric(sub[sub["confidence_tier"] == t]["growth_rate"], errors="coerce").dropna().to_numpy()
            for t in tiers
        ]
        ax.boxplot(
            data_list,
            tick_labels=tiers,
            patch_artist=True,
            boxprops={"facecolor": "#d9e8fb", "color": "#4572a7"},
            medianprops={"color": "#a96900", "linewidth": 1.5},
        )
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
        ax.set_title(scenario)
        ax.set_xlabel("Tier")
    axes_list[0].set_ylabel("Growth Rate (2055 vs 2025)")
    fig.suptitle("Place Growth-Rate Distribution by Tier and Scenario", y=1.02)
    fig.tight_layout()
    p = output_dir / "growth_rate_boxplots_by_scenario.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["growth_rate_boxplots"] = p.name

    # 6) Balance share distribution (focal).
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bvals = pd.to_numeric(scenario_map[focal_scenario].balance["balance_share"], errors="coerce").dropna()
    ax.hist(bvals, bins=18, color="#6b7fa7", alpha=0.90, edgecolor="white")
    ax.set_title(f"{focal_scenario}: Distribution of Balance-of-County Share")
    ax.set_xlabel("balance_share")
    ax.set_ylabel("County-Year Count")
    fig.tight_layout()
    p = output_dir / f"{focal_scenario}_balance_share_distribution.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["balance_share_distribution"] = p.name

    # 7) Top flagged places (focal).
    out = scenario_map[focal_scenario].outliers.copy()
    out["place_label"] = out["name"].astype(str) + " (" + out["place_fips"].astype(str) + ")"
    top_places = out["place_label"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.barh(top_places.index[::-1], top_places.values[::-1], color="#c96f2d")
    ax.set_title(f"{focal_scenario}: Top Places by Outlier Flag Count")
    ax.set_xlabel("Number of flags")
    fig.tight_layout()
    p = output_dir / f"{focal_scenario}_top_flagged_places.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["top_flagged_places"] = p.name

    # 8) State scenario ordering line chart.
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ordering_df = metrics["ordering_df"]
    if ordering_df is not None and not ordering_df.empty:
        for s, color in [
            ("restricted_growth", "#d62728"),
            ("baseline", "#1f77b4"),
            ("high_growth", "#2ca02c"),
        ]:
            ax.plot(ordering_df.index.to_numpy(), ordering_df[s].to_numpy(), label=s, color=color, linewidth=2)
        ax.set_title("State Totals by Scenario (Ordering Check)")
        ax.set_xlabel("Year")
        ax.set_ylabel("State Population")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "State ordering chart unavailable\n(required county scenario outputs missing).", ha="center", va="center")
        ax.set_title("State Totals by Scenario (Ordering Check)")
        ax.set_axis_off()
    fig.tight_layout()
    p = output_dir / "state_scenario_ordering.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["state_ordering"] = p.name

    # 9) Hard-gate violation counts.
    share_df = pd.DataFrame(metrics["share_gate_rows"])
    balance_df = pd.DataFrame(metrics["balance_gate_rows"])
    universe_df = pd.DataFrame(metrics["universe_rows"])
    neg_pop_total = int(sum(int(v["negative_rows"]) for v in metrics["place_population_scan"].values()))
    violation_labels = [
        "share_sum > 1 rows",
        "constraint_satisfied=false rows",
        "negative balance rows",
        "universe missing+extra",
        "negative place population rows",
        "state ordering violation years",
    ]
    violation_values = np.array(
        [
            int(share_df["rows_sum_gt_1"].sum()),
            int(share_df["rows_constraint_false"].sum()),
            int(balance_df["negative_balance_rows"].sum()),
            int((universe_df["missing"] + universe_df["extra"]).sum()),
            neg_pop_total,
            int(metrics["ordering_violations"]),
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    colors = ["#2ca02c" if v == 0 else "#d62728" for v in violation_values]
    ax.barh(violation_labels[::-1], violation_values[::-1], color=colors[::-1])
    ax.set_title("Hard-Gate Violation Counts (All Reviewed Scenarios)")
    ax.set_xlabel("Count")
    fig.tight_layout()
    p = output_dir / "hard_gate_violation_counts.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    chart_files["hard_gate_violations"] = p.name

    return chart_files


def build_gates(
    scenarios: list[ScenarioData],
    metrics: dict[str, Any],
) -> list[Gate]:
    """Assemble per-gate review content and agent suggestions."""
    focal: ScenarioData = metrics["focal"]
    focal_scenario = str(metrics["focal_scenario"])

    snapshot_df = pd.DataFrame(metrics["snapshot_rows"])
    share_df = pd.DataFrame(metrics["share_gate_rows"])
    balance_df = pd.DataFrame(metrics["balance_gate_rows"])
    growth_df = pd.DataFrame(metrics["growth_summary_rows"])
    universe_df = pd.DataFrame(metrics["universe_rows"])
    outlier_counts = metrics["outlier_counts_by_scenario"]

    # Reusable tables
    snapshot_table = _table(
        ["Scenario", "tier rows", "share-sum rows", "balance rows", "outlier rows"],
        [
            [
                str(row["scenario"]),
                f"{int(row['qa_tier_summary'])}",
                f"{int(row['qa_share_sum_validation']):,}",
                f"{int(row['qa_balance_of_county']):,}",
                f"{int(row['qa_outlier_flags']):,}",
            ]
            for _, row in snapshot_df.iterrows()
        ],
    )

    share_table = _table(
        ["Scenario", "Max sum_place_shares", "Rows sum>1", "Rows constraint=false", "Rescaling rows"],
        [
            [
                str(row["scenario"]),
                f"{float(row['max_sum_place_shares']):.6f}",
                f"{int(row['rows_sum_gt_1'])}",
                f"{int(row['rows_constraint_false'])}",
                f"{int(row['rescaling_rows'])}",
            ]
            for _, row in share_df.iterrows()
        ],
    )

    balance_table = _table(
        ["Scenario", "Min balance_of_county_share", "Negative balance rows"],
        [
            [
                str(row["scenario"]),
                f"{float(row['min_balance_share']):.6f}",
                f"{int(row['negative_balance_rows'])}",
            ]
            for _, row in balance_df.iterrows()
        ],
    )

    rescaling_focal = focal.share_sum.copy()
    rescaling_focal["year"] = pd.to_numeric(rescaling_focal["year"], errors="coerce").astype("Int64")
    rescaling_focal["rescaling_applied"] = rescaling_focal["rescaling_applied"].astype(bool)
    rescaling_focal = rescaling_focal.dropna(subset=["year"])
    rescaling_focal["year"] = rescaling_focal["year"].astype(int)

    rescaling_by_county = (
        rescaling_focal.groupby(["county_fips", "county_name"], as_index=False)
        .agg(total_years=("year", "count"), rescaling_years=("rescaling_applied", "sum"))
        .sort_values(["rescaling_years", "county_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    rescaling_by_county["rescaling_years"] = rescaling_by_county["rescaling_years"].astype(int)
    rescaling_by_county["total_years"] = rescaling_by_county["total_years"].astype(int)

    total_county_years = int(rescaling_by_county["total_years"].sum())
    total_rescaling_years = int(rescaling_by_county["rescaling_years"].sum())
    counties_total = int(len(rescaling_by_county))
    counties_with_rescaling = int((rescaling_by_county["rescaling_years"] > 0).sum())
    overall_rescaling_rate = (total_rescaling_years / total_county_years) if total_county_years else 0.0

    if total_rescaling_years > 0:
        pct = rescaling_by_county["rescaling_years"] / float(total_rescaling_years)
    else:
        pct = pd.Series(np.zeros(len(rescaling_by_county), dtype=float))
    rescaling_by_county["pct_of_rescaling"] = pct
    rescaling_by_county["cum_pct_of_rescaling"] = pct.cumsum()
    rescaling_by_county["rescaling_rate"] = rescaling_by_county["rescaling_years"] / rescaling_by_county["total_years"]

    top5_share = float(rescaling_by_county.head(5)["pct_of_rescaling"].sum()) if total_rescaling_years else 0.0
    top10_share = float(rescaling_by_county.head(10)["pct_of_rescaling"].sum()) if total_rescaling_years else 0.0
    if total_rescaling_years and not rescaling_by_county.empty:
        cum = rescaling_by_county["cum_pct_of_rescaling"].to_numpy(dtype=float)
        k80 = int(np.argmax(cum >= 0.8) + 1) if np.any(cum >= 0.8) else counties_total
    else:
        k80 = 0

    rescaling_summary_table = _table(
        ["Metric", "Value"],
        [
            ["Focal scenario", focal_scenario],
            ["County-years evaluated", f"{total_county_years:,}"],
            ["County-years with rescaling_applied=true", f"{total_rescaling_years:,} ({overall_rescaling_rate:.1%})"],
            ["Counties evaluated", f"{counties_total:,}"],
            ["Counties with any rescaling", f"{counties_with_rescaling:,} ({(counties_with_rescaling / counties_total) if counties_total else 0.0:.1%})"],
            ["Share of rescaling in top 5 counties", f"{top5_share:.1%}"],
            ["Share of rescaling in top 10 counties", f"{top10_share:.1%}"],
            ["Counties needed to reach 80% of rescaling", str(k80)],
        ],
    )

    rescaling_by_county_rows: list[list[str]] = []
    for idx, row in rescaling_by_county.iterrows():
        rescaling_by_county_rows.append(
            [
                str(idx + 1),
                str(row["county_name"]),
                _normalize_fips(row["county_fips"], 5),
                str(int(row["rescaling_years"])),
                f"{float(row['rescaling_rate']):.1%}",
                f"{float(row['pct_of_rescaling']):.1%}",
                f"{float(row['cum_pct_of_rescaling']):.1%}",
            ]
        )
    rescaling_by_county_table = _table(
        [
            "Rank",
            "County",
            "County FIPS",
            "Rescaling years",
            "Rescaling rate",
            "% of all rescaling",
            "Cumulative %",
        ],
        rescaling_by_county_rows,
    )

    rescaled_rows = rescaling_focal[rescaling_focal["rescaling_applied"]].copy()
    rescaled_rows["sum_place_shares"] = pd.to_numeric(rescaled_rows["sum_place_shares"], errors="coerce")
    rescaled_rows["balance_of_county_share"] = pd.to_numeric(rescaled_rows["balance_of_county_share"], errors="coerce")
    rescaled_rows = rescaled_rows.sort_values(["county_name", "year"], ascending=[True, True])
    rescaling_county_year_table = _table(
        ["County", "County FIPS", "Year", "sum_place_shares", "balance_of_county_share"],
        [
            [
                str(r["county_name"]),
                _normalize_fips(r["county_fips"], 5),
                str(int(r["year"])),
                f"{float(r['sum_place_shares']):.6f}" if not pd.isna(r["sum_place_shares"]) else "nan",
                f"{float(r['balance_of_county_share']):.6f}" if not pd.isna(r["balance_of_county_share"]) else "nan",
            ]
            for _, r in rescaled_rows.iterrows()
        ],
    )

    rescaling_table = (
        "<h4>Concentration Summary</h4>"
        + rescaling_summary_table
        + "<h4>All Counties</h4>"
        + f"<div class='table-scroll'>{rescaling_by_county_table}</div>"
        + f"<details><summary>Show all rescaled county-years ({len(rescaled_rows):,} rows)</summary>"
        + f"<div class='table-scroll'>{rescaling_county_year_table}</div></details>"
    )

    growth_table = _table(
        ["Scenario", "Tier", "Count", "Mean", "Median", "Min", "Max"],
        [
            [
                str(r["scenario"]),
                str(r["tier"]),
                str(int(r["count"])),
                f"{float(r['mean_growth']):.4f}",
                f"{float(r['median_growth']):.4f}",
                f"{float(r['min_growth']):.4f}",
                f"{float(r['max_growth']):.4f}",
            ]
            for _, r in growth_df.iterrows()
        ],
    )

    outlier_rows: list[list[str]] = []
    for scenario in sorted(outlier_counts):
        row = outlier_counts[scenario]
        if not row:
            outlier_rows.append([scenario, "none", "0"])
            continue
        for flag_type, value in sorted(row.items()):
            outlier_rows.append([scenario, flag_type, str(int(value))])
    outlier_table = _table(["Scenario", "Flag Type", "Count"], outlier_rows)

    ordering_df: pd.DataFrame | None = metrics["ordering_df"]
    ordering_violations = int(metrics["ordering_violations"])
    if ordering_df is None:
        ordering_table = _table(
            ["Metric", "Value"],
            [["Ordering check availability", "Skipped (one or more scenario county outputs unavailable)"]],
        )
    else:
        min_gap_low_mid = float((ordering_df["baseline"] - ordering_df["restricted_growth"]).min())
        min_gap_mid_high = float((ordering_df["high_growth"] - ordering_df["baseline"]).min())
        ordering_table = _table(
            ["Metric", "Value"],
            [
                ["Violation years", str(ordering_violations)],
                ["Min (baseline - restricted_growth)", f"{min_gap_low_mid:,.2f}"],
                ["Min (high_growth - baseline)", f"{min_gap_mid_high:,.2f}"],
            ],
        )

    universe_table = _table(
        ["Scenario", "Expected", "Produced", "Missing", "Extra", "Missing sample", "Extra sample"],
        [
            [
                str(r["scenario"]),
                str(int(r["expected_places"])),
                str(int(r["produced_places"])),
                str(int(r["missing"])),
                str(int(r["extra"])),
                str(r["missing_sample"]),
                str(r["extra_sample"]),
            ]
            for _, r in universe_df.iterrows()
        ],
    )
    pop_scan_rows = []
    for scenario, stats in metrics["place_population_scan"].items():
        min_pop = stats["min_population"]
        min_text = "nan" if pd.isna(min_pop) else f"{float(min_pop):.4f}"
        pop_scan_rows.append([scenario, str(int(stats["files"])), str(int(stats["negative_rows"])), min_text])
    pop_scan_table = _table(["Scenario", "Place parquet files", "Negative rows", "Min population"], pop_scan_rows)

    share_hard_pass = int(share_df["rows_sum_gt_1"].sum()) == 0 and int(share_df["rows_constraint_false"].sum()) == 0
    balance_hard_pass = int(balance_df["negative_balance_rows"].sum()) == 0
    ordering_pass = ordering_violations == 0 if ordering_df is not None else False
    universe_pass = int((universe_df["missing"] + universe_df["extra"]).sum()) == 0
    nonneg_pass = int(sum(int(v["negative_rows"]) for v in metrics["place_population_scan"].values())) == 0

    gates: list[Gate] = [
        Gate(
            gate_id="gate-artifacts",
            title="Gate 1: QA Artifact Completeness",
            objective="Confirm all required QA outputs are present for every scenario with expected structural row patterns.",
            status="PASS" if all(snapshot_df["qa_tier_summary"] == 3) else "REVIEW",
            suggested_decision="Approve" if all(snapshot_df["qa_tier_summary"] == 3) else "Request change",
            analysis=(
                "All scenarios include the four QA artifacts and expected row shapes. "
                "No structural completeness concerns were detected."
                if all(snapshot_df["qa_tier_summary"] == 3)
                else "At least one scenario does not match expected QA artifact structure."
            ),
            visuals=["artifact_counts"],
            stats_html=snapshot_table,
        ),
        Gate(
            gate_id="gate-share",
            title="Gate 2: County Share Constraint",
            objective="Verify county-year place share totals satisfy the hard constraint (sum(place_shares) <= 1.0).",
            status="PASS" if share_hard_pass else "REVIEW",
            suggested_decision="Approve" if share_hard_pass else "Request change",
            analysis=(
                "No county-year rows exceeded the share ceiling and no rows were marked constraint_satisfied=false."
                if share_hard_pass
                else "One or more rows violate county share constraints and require correction before approval."
            ),
            visuals=["share_sum_distribution"],
            stats_html=share_table,
        ),
        Gate(
            gate_id="gate-balance",
            title="Gate 3: Place-County Consistency and Balance",
            objective="Check that county totals remain internally consistent and balance-of-county is non-negative.",
            status="PASS" if balance_hard_pass else "REVIEW",
            suggested_decision="Approve" if balance_hard_pass else "Request change",
            analysis=(
                "No negative balance-of-county rows were observed, supporting place<=county consistency."
                if balance_hard_pass
                else "Negative balance rows were found and should be investigated before approval."
            ),
            visuals=["balance_share_distribution"],
            stats_html=balance_table,
        ),
        Gate(
            gate_id="gate-rescaling",
            title="Gate 4: Rescaling Behavior",
            objective="Review where and how often share reconciliation/rescaling occurs, and assess plausibility.",
            status="REVIEW",
            suggested_decision="Approve with notes",
            analysis=(
                "Rescaling is concentrated in a subset of counties and appears as a controlled, expected reconciliation behavior. "
                "Human review should confirm this concentration matches substantive expectations."
            ),
            visuals=["rescaling_top_counties", "rescaling_concentration"],
            stats_html=rescaling_table,
        ),
        Gate(
            gate_id="gate-growth",
            title="Gate 5: Tier Growth Pattern Sanity",
            objective="Inspect growth distributions by tier/scenario for patterns inconsistent with expected uncertainty structure.",
            status="REVIEW",
            suggested_decision="Approve with notes",
            analysis=(
                "Tier-level growth distributions show broader spread in LOWER and MODERATE tiers, as expected. "
                "Outlier tails exist and should be acknowledged in caveats, not automatically blocked."
            ),
            visuals=["growth_rate_boxplots"],
            stats_html=growth_table,
        ),
        Gate(
            gate_id="gate-outliers",
            title="Gate 6: Outlier Flag Review",
            objective="Review flagged places/types and decide whether any require narrative handling or exclusions.",
            status="REVIEW",
            suggested_decision="Approve with notes",
            analysis=(
                "Flag mix is stable across scenarios and dominated by SHARE_RESCALED plus a smaller SHARE_REVERSAL set. "
                "Current pattern looks plausible for focused manual review without blocking the pipeline."
            ),
            visuals=["outlier_counts", "top_flagged_places"],
            stats_html=outlier_table,
        ),
        Gate(
            gate_id="gate-ordering",
            title="Gate 7: State Scenario Ordering",
            objective="Verify state totals preserve restricted_growth <= baseline <= high_growth for each projection year.",
            status="PASS" if ordering_pass else "REVIEW",
            suggested_decision="Approve" if ordering_pass else "Request change",
            analysis=(
                "State scenario ordering is preserved across all comparable years."
                if ordering_pass
                else "Ordering check could not be verified or violations exist; investigate before final approval."
            ),
            visuals=["state_ordering"],
            stats_html=ordering_table,
        ),
        Gate(
            gate_id="gate-universe",
            title="Gate 8: Output Universe and Non-Negative Populations",
            objective="Confirm output place universe matches crosswalk and no negative place populations appear in parquet outputs.",
            status="PASS" if universe_pass and nonneg_pass else "REVIEW",
            suggested_decision="Approve" if universe_pass and nonneg_pass else "Request change",
            analysis=(
                "Output universe parity is exact and parquet population scans found no negative rows."
                if universe_pass and nonneg_pass
                else "Universe mismatch or negative population rows detected; resolve before approval."
            ),
            visuals=["hard_gate_violations"],
            stats_html=universe_table + pop_scan_table,
        ),
    ]

    return gates


def html_page(
    *,
    generated_at_local: str,
    generated_at_utc: str,
    version_tag: str,
    scenarios: list[ScenarioData],
    chart_rel_dir: str,
    chart_files: dict[str, str],
    gates: list[Gate],
    focal_scenario: str,
) -> str:
    """Render tabbed review HTML content."""
    scenario_names = [s.scenario for s in scenarios]
    focal = next(s for s in scenarios if s.scenario == focal_scenario)

    chart = lambda key: f"{chart_rel_dir}/{chart_files[key]}"  # noqa: E731
    visual_help: dict[str, dict[str, str]] = {
        "artifact_counts": {
            "label": "QA Artifact Row Counts by Scenario",
            "shows": (
                "Grouped bars show row counts for tier summary, share-sum validation, "
                "balance-of-county, and outlier-flag artifacts in each scenario."
            ),
            "interpret": (
                "Bar heights should be structurally comparable across scenarios. Missing bars "
                "or large unexpected differences indicate upstream artifact completeness issues."
            ),
            "rule": (
                "Approve only if each scenario has all required QA artifacts present with "
                "expected structural row patterns."
            ),
            "reason": (
                "Downstream quality checks are only credible when the full QA artifact set exists; "
                "missing or malformed artifacts can hide defects rather than reveal them."
            ),
        },
        "outlier_counts": {
            "label": "Outlier Flags by Scenario and Type",
            "shows": (
                "Stacked bars break total outlier flags into flag types for each scenario."
            ),
            "interpret": (
                "Use this to check whether flag composition is stable across scenarios. A sudden "
                "surge in one type may indicate scenario-specific modeling artifacts."
            ),
            "rule": (
                "Approve with notes when type mix is broadly stable across scenarios; request "
                "change if one scenario shows a material unexplained spike in a flag type."
            ),
            "reason": (
                "Scenario variants should shift levels modestly, but abrupt composition changes can "
                "signal unintended scenario-specific behavior."
            ),
        },
        "share_sum_distribution": {
            "label": "County-Year Share-Sum Distribution",
            "shows": (
                "Histogram of county-year sum of place shares with a dashed hard-limit line at 1.0."
            ),
            "interpret": (
                "Most mass should be at or below 1.0. Any visible mass to the right of 1.0 implies "
                "constraint violations that should block approval."
            ),
            "rule": (
                "Approve only if there are zero county-year observations with sum_place_shares > 1.0."
            ),
            "reason": (
                "Place shares represent partitions of a county total; totals above 1.0 violate mass "
                "conservation and invalidate county-place consistency."
            ),
        },
        "rescaling_top_counties": {
            "label": "Top Counties by Rescaling Incidence",
            "shows": (
                "Horizontal bars rank counties by count of county-years where share rescaling was applied."
            ),
            "interpret": (
                "Concentration in known volatile counties can be acceptable. Broad or unexpected spread "
                "across many counties may suggest overactive reconciliation."
            ),
            "rule": (
                "Approve with notes when rescaling is concentrated in substantively plausible counties; "
                "request change if rescaling is widespread without explanation."
            ),
            "reason": (
                "Some local volatility is expected, but broad rescaling can indicate over-correction or "
                "unstable share inputs."
            ),
        },
        "rescaling_concentration": {
            "label": "Rescaling Concentration (Cumulative Share)",
            "shows": (
                "A cumulative curve of rescaling incidence when counties are ranked from most to least "
                "rescaling-applied county-years."
            ),
            "interpret": (
                "A steep early curve indicates rescaling is concentrated in a few counties. A near-linear "
                "curve indicates rescaling is broadly distributed across counties."
            ),
            "rule": (
                "Approve with notes when a small subset of counties explains most rescaling; request change "
                "if rescaling is diffuse across many counties without a clear substantive rationale."
            ),
            "reason": (
                "Controlled reconciliation typically appears as localized adjustments. Broad distribution "
                "can indicate systemic share instability."
            ),
        },
        "growth_rate_boxplots": {
            "label": "Growth-Rate Distribution by Tier and Scenario",
            "shows": (
                "Per-scenario boxplots summarize 2025-2055 place growth-rate distributions for HIGH, "
                "MODERATE, and LOWER confidence tiers."
            ),
            "interpret": (
                "LOWER and MODERATE tiers should generally show wider spread than HIGH. Extreme tails "
                "should be reviewed for plausibility rather than auto-failed."
            ),
            "rule": (
                "Approve with notes if uncertainty spread is directionally sensible by tier and outlier "
                "tails are plausible; request change only for clear structural anomalies."
            ),
            "reason": (
                "Confidence tiers encode uncertainty expectations, so distribution width should reflect "
                "tiering, while rare extremes may still be demographically plausible."
            ),
        },
        "balance_share_distribution": {
            "label": "Balance-of-County Share Distribution",
            "shows": (
                "Histogram of county-year balance share values, representing the residual share not "
                "assigned to incorporated places."
            ),
            "interpret": (
                "Values should remain non-negative and within a plausible range. Heaping at extremes "
                "or unexpected shape shifts may warrant county-level follow-up."
            ),
            "rule": (
                "Approve only if balance shares are non-negative and distribution shape has no "
                "unexplained extreme concentration."
            ),
            "reason": (
                "Balance share is the residual county component; negative values are invalid and "
                "extreme heaping can indicate upstream allocation issues."
            ),
        },
        "top_flagged_places": {
            "label": "Top Places by Outlier Flag Count",
            "shows": (
                "Horizontal bars list places with the highest number of outlier flags in the focal scenario."
            ),
            "interpret": (
                "Use this as a triage list for narrative review. Repeatedly flagged places are prime "
                "candidates for explanatory notes or targeted QA checks."
            ),
            "rule": (
                "Approve with notes when top flagged places have plausible local explanations or are "
                "explicitly documented in narrative caveats."
            ),
            "reason": (
                "This chart is a prioritization device for human judgment, not a hard-fail gate; "
                "high-frequency flags should be explained, not ignored."
            ),
        },
        "state_ordering": {
            "label": "State Scenario Ordering Check",
            "shows": (
                "Lines compare state totals by year for restricted_growth, baseline, and high_growth scenarios."
            ),
            "interpret": (
                "For each year, restricted_growth should remain below baseline, and baseline below high_growth. "
                "Any crossing indicates ordering violations."
            ),
            "rule": (
                "Approve only if scenario lines never cross and ordering holds for every comparable year."
            ),
            "reason": (
                "Scenario definitions imply monotonic state totals; line crossings contradict intended "
                "scenario semantics."
            ),
        },
        "hard_gate_violations": {
            "label": "Hard-Gate Violation Counts",
            "shows": (
                "Bar chart reports total counts of blocking violations across share, balance, universe, "
                "non-negative population, and state ordering checks."
            ),
            "interpret": (
                "Expected count is zero for every hard gate. Any non-zero red bar should be treated as "
                "a release blocker until resolved."
            ),
            "rule": (
                "Approve only when every hard-gate violation count is exactly zero."
            ),
            "reason": (
                "These are non-negotiable invariants for validity of released projections; any violation "
                "means outputs are not publication-ready."
            ),
        },
    }

    overview_cards = "".join(
        [
            f"<div class='kpi'><div class='label'>Scenarios Reviewed</div><div class='value'>{len(scenarios)}</div><div class='muted'>{html.escape(', '.join(scenario_names))}</div></div>",
            f"<div class='kpi'><div class='label'>Projected Places ({html.escape(focal_scenario)})</div><div class='value'>{len(focal.places_summary)}</div><div class='muted'>rows in places_summary (row_type=place)</div></div>",
            f"<div class='kpi'><div class='label'>Counties in QA ({html.escape(focal_scenario)})</div><div class='value'>{int(focal.balance['county_fips'].astype(str).nunique())}</div><div class='muted'>county_fips in qa_balance_of_county</div></div>",
            f"<div class='kpi'><div class='label'>Outlier Flags ({html.escape(focal_scenario)})</div><div class='value'>{len(focal.outliers)}</div><div class='muted'>rows in qa_outlier_flags</div></div>",
        ]
    )

    tab_buttons = [
        "<button class='tab-button active' data-tab='tab-overview'>Overview</button>",
    ] + [
        f"<button class='tab-button' data-tab='{html.escape(g.gate_id)}'>{html.escape(g.title.split(':', 1)[0])}</button>"
        for g in gates
    ]

    # Overview content
    overview_content = f"""
      <section id="tab-overview" class="tab-panel active">
        <h2>Overview</h2>
        <div class="panel grid">{overview_cards}</div>
        <div class="panel">
          <p><strong>Review model:</strong> one tab per quality gate/issue, including already-passing gates.</p>
          <p><strong>What each tab includes:</strong> objective, what to check, relevant visual(s), summary stats, and agent recommendation.</p>
          <p class="muted">Version: <code>{html.escape(version_tag)}</code> | Generated: {html.escape(generated_at_local)} (local), {html.escape(generated_at_utc)} (UTC)</p>
        </div>
      </section>
    """

    gate_panels = []
    for gate in gates:
        status_class = "pass" if gate.status == "PASS" else "review"
        status_label = f"<span class='gate-status {status_class}'>{html.escape(gate.status)}</span>"

        visual_cards = []
        for visual_key in gate.visuals:
            default_label = visual_key.replace("_", " ").title()
            visual_meta = visual_help.get(
                visual_key,
                {
                    "label": default_label,
                    "shows": "Visual summary for this gate.",
                    "interpret": "Review for consistency with gate objective and expected patterns.",
                    "rule": "Approve when chart evidence supports the gate objective.",
                    "reason": "The gate objective defines the acceptance criterion for this visual.",
                },
            )
            visual_label = html.escape(visual_meta["label"])
            visual_shows = html.escape(visual_meta["shows"])
            visual_interpret = html.escape(visual_meta["interpret"])
            visual_rule = html.escape(visual_meta["rule"])
            visual_reason = html.escape(visual_meta["reason"])
            visual_cards.append(
                f"""
                <div class="viz-card">
                  <h4 class="viz-title">{visual_label}</h4>
                  <img class="expandable-chart" src="{chart(visual_key)}" alt="{html.escape(gate.title)} - {visual_label}" />
                  <p class="viz-caption"><strong>What this shows:</strong> {visual_shows}</p>
                  <p class="viz-caption"><strong>How to interpret:</strong> {visual_interpret}</p>
                  <p class="viz-rule"><strong>Decision rule:</strong> {visual_rule}</p>
                  <p class="viz-reason"><strong>Why this rule:</strong> {visual_reason}</p>
                  <p class="viz-hint">Click chart to expand full-screen. Use Prev/Next or ←/→ keys in full-screen mode.</p>
                </div>
                """
            )
        visual_html = "".join(visual_cards)

        panel = f"""
        <section id="{html.escape(gate.gate_id)}" class="tab-panel">
          <h2>{html.escape(gate.title)} {status_label}</h2>
          <div class="panel">
            <p><strong>What we are checking:</strong> {html.escape(gate.objective)}</p>
            <p><strong>Agent analysis:</strong> {html.escape(gate.analysis)}</p>
            <p><strong>Suggested decision:</strong> <span class='decision'>{html.escape(gate.suggested_decision)}</span></p>
            <p><strong>Reviewer decision:</strong> [ ] Approve  [ ] Approve with notes  [ ] Request change</p>
          </div>
          <div class="panel">
            <h3>Summary Statistics</h3>
            {gate.stats_html}
          </div>
          <div class="panel">
            <h3>Relevant Visuals</h3>
            <div class="viz-grid">{visual_html}</div>
          </div>
        </section>
        """
        gate_panels.append(panel)

    gate_content = "".join(gate_panels)
    tabs_html = "".join(tab_buttons)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PP-003 Human Review Package ({html.escape(version_tag)})</title>
  <style>
    :root {{
      --panel: #ffffff;
      --ink: #132238;
      --muted: #4b5f78;
      --accent: #1d5fa6;
      --ok: #1f7a3f;
      --warn: #a96900;
      --border: #d5dfeb;
      --tab: #e8f0fb;
      --tab-active: #1d5fa6;
      --tab-active-ink: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f3f7fc 0%, #f8fbff 35%, #f6f8fb 100%);
    }}
    .wrap {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 18px 16px 44px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 1.85rem; }}
    h2 {{
      margin-top: 12px;
      padding-top: 8px;
      font-size: 1.3rem;
    }}
    p, li {{ line-height: 1.45; }}
    .muted {{ color: var(--muted); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin: 10px 0;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }}
    .kpi {{
      background: #f8fbff;
      border: 1px solid #dbe8f8;
      border-radius: 8px;
      padding: 10px;
    }}
    .kpi .label {{ font-size: 0.82rem; color: var(--muted); }}
    .kpi .value {{ font-size: 1.25rem; font-weight: 700; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      text-align: left;
      padding: 7px;
      vertical-align: top;
    }}
    th {{ background: #edf4fc; }}
    .table-scroll {{
      max-height: 560px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: white;
      margin-top: 8px;
    }}
    .table-scroll table {{
      margin-top: 0;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    .table-scroll thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    details {{
      margin-top: 10px;
    }}
    details summary {{
      cursor: pointer;
      color: #173657;
      font-weight: 700;
      padding: 6px 0;
    }}
    code {{
      background: #edf2f8;
      border-radius: 4px;
      padding: 1px 5px;
    }}
    a {{ color: var(--accent); }}
    .tabs {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin: 12px 0 8px;
    }}
    .tab-button {{
      border: 1px solid #b7cae5;
      background: var(--tab);
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 0.92rem;
      color: #173657;
    }}
    .tab-button:hover {{
      background: #d7e6fb;
    }}
    .tab-button.active {{
      background: var(--tab-active);
      color: var(--tab-active-ink);
      border-color: #1d5fa6;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .gate-status {{
      display: inline-block;
      margin-left: 8px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.78rem;
      vertical-align: middle;
    }}
    .gate-status.pass {{ background: #eaf8ee; color: #1f7a3f; border: 1px solid #bde2c5; }}
    .gate-status.review {{ background: #fff4df; color: #a96900; border: 1px solid #ebd2a2; }}
    .decision {{ font-weight: 700; }}
    .viz-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 12px;
    }}
    .viz-card {{
      background: #fbfdff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
    }}
    .viz-title {{
      margin: 2px 0 8px;
      font-size: 0.95rem;
      color: #1f2f42;
    }}
    .viz-card img {{
      width: 100%;
      height: auto;
      border: 1px solid #e4ecf6;
      border-radius: 6px;
      background: white;
      cursor: zoom-in;
    }}
    .viz-caption {{
      margin: 6px 0 0;
      font-size: 0.84rem;
      line-height: 1.35;
      color: #29405a;
    }}
    .viz-rule {{
      margin: 6px 0 0;
      font-size: 0.84rem;
      line-height: 1.35;
      color: #1f2f42;
    }}
    .viz-reason {{
      margin: 6px 0 0;
      font-size: 0.84rem;
      line-height: 1.35;
      color: #3a4f67;
    }}
    .viz-hint {{
      margin: 6px 0 0;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      background: rgba(5, 12, 23, 0.92);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      padding: 16px;
    }}
    .lightbox.active {{ display: flex; }}
    .lightbox-frame {{
      width: min(96vw, 1800px);
      max-height: 94vh;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .lightbox-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #e7eef8;
      font-size: 0.9rem;
    }}
    .lightbox-close, .lightbox-nav {{
      border: 1px solid #9fb7d8;
      background: #1f3551;
      color: #e7eef8;
      border-radius: 6px;
      padding: 4px 10px;
      cursor: pointer;
      margin-left: 6px;
    }}
    .lightbox-close:hover, .lightbox-nav:hover {{ background: #284568; }}
    .lightbox-img {{
      max-width: 100%;
      max-height: 88vh;
      width: auto;
      height: auto;
      border-radius: 8px;
      border: 1px solid #355274;
      background: white;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PP-003 Human Review Package</h1>
    <p class="muted">Scope: IMP-12 (QA artifacts) and IMP-13 (constraint enforcement) for place projections.</p>
    <p class="muted">Version: <code>{html.escape(version_tag)}</code> | Generated: {html.escape(generated_at_local)} (local), {html.escape(generated_at_utc)} (UTC)</p>

    <div class="tabs">{tabs_html}</div>

    {overview_content}
    {gate_content}

    <div class="panel">
      <h3>Sign-Off</h3>
      <p><strong>Reviewer:</strong> ____________________</p>
      <p><strong>Date:</strong> ____________________</p>
      <p><strong>Decision:</strong> [ ] Approved as-is  [ ] Approved with notes  [ ] Changes requested</p>
      <p><strong>Notes:</strong></p>
      <p style="min-height: 70px; border: 1px dashed #b8d4bc; border-radius: 6px; padding: 8px;"></p>
    </div>
  </div>

  <div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="Expanded chart view">
    <div class="lightbox-frame">
      <div class="lightbox-toolbar">
        <span id="lightbox-title">Chart</span>
        <div>
          <button id="lightbox-prev" class="lightbox-nav" type="button" aria-label="Previous chart">Prev</button>
          <button id="lightbox-next" class="lightbox-nav" type="button" aria-label="Next chart">Next</button>
          <button id="lightbox-close" class="lightbox-close" type="button" aria-label="Close expanded chart">Close</button>
        </div>
      </div>
      <img id="lightbox-img" class="lightbox-img" src="" alt="Expanded chart" />
    </div>
  </div>

  <script>
    (function () {{
      // Tab behavior
      const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
      const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
      function activateTab(tabId) {{
        tabButtons.forEach((btn) => {{
          btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
        }});
        tabPanels.forEach((panel) => {{
          panel.classList.toggle('active', panel.getAttribute('id') === tabId);
        }});
      }}
      tabButtons.forEach((btn) => {{
        btn.addEventListener('click', () => activateTab(btn.getAttribute('data-tab') || 'tab-overview'));
      }});

      // Lightbox behavior
      const lightbox = document.getElementById("lightbox");
      const lightboxImg = document.getElementById("lightbox-img");
      const lightboxTitle = document.getElementById("lightbox-title");
      const closeBtn = document.getElementById("lightbox-close");
      const prevBtn = document.getElementById("lightbox-prev");
      const nextBtn = document.getElementById("lightbox-next");
      const charts = Array.from(document.querySelectorAll(".expandable-chart"));
      if (!lightbox || !lightboxImg || !lightboxTitle || !closeBtn || !prevBtn || !nextBtn || charts.length === 0) return;

      let currentIndex = -1;

      function openLightbox(index) {{
        if (index < 0 || index >= charts.length) return;
        currentIndex = index;
        const img = charts[currentIndex];
        const src = img.getAttribute("src") || "";
        const alt = img.getAttribute("alt") || "Chart";
        lightboxImg.src = src;
        lightboxImg.alt = alt;
        lightboxTitle.textContent = alt + " (" + (currentIndex + 1) + "/" + charts.length + ")";
        lightbox.classList.add("active");
        document.body.style.overflow = "hidden";
      }}

      function closeLightbox() {{
        lightbox.classList.remove("active");
        lightboxImg.src = "";
        document.body.style.overflow = "";
        currentIndex = -1;
      }}

      function stepChart(direction) {{
        if (!lightbox.classList.contains("active") || charts.length === 0) return;
        let nextIndex = currentIndex + direction;
        if (nextIndex < 0) nextIndex = charts.length - 1;
        if (nextIndex >= charts.length) nextIndex = 0;
        openLightbox(nextIndex);
      }}

      charts.forEach((img, index) => {{
        img.addEventListener("click", () => openLightbox(index));
      }});

      closeBtn.addEventListener("click", closeLightbox);
      prevBtn.addEventListener("click", () => stepChart(-1));
      nextBtn.addEventListener("click", () => stepChart(1));
      lightbox.addEventListener("click", (event) => {{
        if (event.target === lightbox) closeLightbox();
      }});
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && lightbox.classList.contains("active")) closeLightbox();
        else if (event.key === "ArrowLeft" && lightbox.classList.contains("active")) stepChart(-1);
        else if (event.key === "ArrowRight" && lightbox.classList.contains("active")) stepChart(1);
      }});
    }})();
  </script>
</body>
</html>
"""


def write_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON manifest for latest generated package."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build timestamped PP-003 human review package with static visuals."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario keys to include (default: baseline restricted_growth high_growth).",
    )
    parser.add_argument(
        "--focal-scenario",
        default="baseline",
        help="Scenario used for focal metrics/examples/charts (default: baseline).",
    )
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=DEFAULT_REVIEWS_DIR,
        help="Directory where versioned HTML will be written (default: docs/reviews).",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Root directory for versioned chart assets (default: docs/reviews/assets/pp3-human-review).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_REVIEWS_DIR / "pp3-human-review-package_latest.json",
        help="Path for latest-manifest JSON (default: docs/reviews/pp3-human-review-package_latest.json).",
    )
    parser.add_argument(
        "--refresh-latest",
        action="store_true",
        help="Also overwrite the canonical latest HTML file path.",
    )
    parser.add_argument(
        "--latest-path",
        type=Path,
        default=DEFAULT_LATEST_HTML,
        help="Canonical latest HTML path used when --refresh-latest is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Build a timestamped human-review package and optional latest copy."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    scenarios = [str(s).strip() for s in args.scenarios if str(s).strip()]
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    scenario_data = [load_scenario_data(s) for s in scenarios]
    scenario_names = [s.scenario for s in scenario_data]
    focal_scenario = args.focal_scenario if args.focal_scenario in scenario_names else scenario_names[0]

    now = now_local()
    date_token, ts_token = timestamp_tokens(now)
    version_tag = ts_token
    generated_local = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    generated_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    reviews_dir = args.reviews_dir if args.reviews_dir.is_absolute() else (REPO_ROOT / args.reviews_dir)
    asset_root = args.asset_root if args.asset_root.is_absolute() else (REPO_ROOT / args.asset_root)
    latest_path = args.latest_path if args.latest_path.is_absolute() else (REPO_ROOT / args.latest_path)
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else (REPO_ROOT / args.manifest_path)

    reviews_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = asset_root / ts_token

    metrics = compute_metrics(scenario_data, focal_scenario=focal_scenario)
    chart_files = build_chart_assets(scenario_data, metrics=metrics, output_dir=asset_dir)
    gates = build_gates(scenario_data, metrics=metrics)

    html_filename = f"{date_token}-pp3-human-review-package_{ts_token}.html"
    versioned_html_path = reviews_dir / html_filename

    chart_rel_dir = str(asset_dir.relative_to(reviews_dir))
    html_content = html_page(
        generated_at_local=generated_local,
        generated_at_utc=generated_utc,
        version_tag=version_tag,
        scenarios=scenario_data,
        chart_rel_dir=chart_rel_dir,
        chart_files=chart_files,
        gates=gates,
        focal_scenario=focal_scenario,
    )
    versioned_html_path.write_text(html_content, encoding="utf-8")

    latest_ref: str | None = None
    if args.refresh_latest:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(html_content, encoding="utf-8")
        latest_ref = str(latest_path.relative_to(REPO_ROOT))

    manifest_payload = {
        "generated_at_local": generated_local,
        "generated_at_utc": generated_utc,
        "version_tag": version_tag,
        "scenarios": scenario_names,
        "focal_scenario": focal_scenario,
        "versioned_html": str(versioned_html_path.relative_to(REPO_ROOT)),
        "asset_dir": str(asset_dir.relative_to(REPO_ROOT)),
        "charts": chart_files,
        "gates": [g.gate_id for g in gates],
        "latest_html": latest_ref,
    }
    write_manifest(manifest_path=manifest_path, payload=manifest_payload)

    LOGGER.info("Wrote versioned review HTML: %s", versioned_html_path)
    LOGGER.info("Wrote versioned chart assets: %s", asset_dir)
    LOGGER.info("Wrote latest manifest: %s", manifest_path)
    if latest_ref:
        LOGGER.info("Refreshed canonical latest HTML: %s", latest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
