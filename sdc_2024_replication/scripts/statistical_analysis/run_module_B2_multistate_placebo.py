#!/usr/bin/env python3
"""
Module B2: Multi-State Placebo Analysis
========================================

Answers the critical question from ADR-020:
"Is North Dakota's international migration shift unusual compared to other states,
or is it a nationwide pattern?"

This analysis tests whether ND's observed regime shift (2010-2019 vs 2021-2024)
is exceptional or typical by comparing it to the distribution of shifts across
all 50 states.

Key Findings This Module Provides:
1. ND's percentile rank in the national distribution of regime shifts
2. Oil/Energy state hypothesis test (do oil states shift more?)
3. Bakken-specific timing hypothesis (boom timing vs. static classification)

Data Source:
- PostgreSQL: census.state_components (2010-2024)
- Covers all 50 states + DC

ADR-020 Context:
This is the key remaining ADR-020 task that ADR-021 did NOT address.
The reconciliation analysis identified this as STILL PENDING and uniquely valuable.

Usage:
    uv run python run_module_B2_multistate_placebo.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add scripts directory to path for db_config
sys.path.append(str(Path(__file__).parent.parent))
from database import db_config

# Import module components
from module_B2_multistate_placebo import (
    add_vintage_labels,
    calculate_all_state_shifts,
    compare_boom_categories,
    get_boom_category,
    get_nd_percentile,
    get_nd_rank_among_boom_states,
    get_nd_rank_among_oil_states,
    rank_states_by_shift,
    run_bakken_specific_hypothesis_test,
    run_boom_state_hypothesis_test,
    run_oil_state_hypothesis_test,
    ALL_OIL_STATES,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
ADR_RESULTS_DIR = PROJECT_ROOT / "docs" / "governance" / "adrs" / "020-reports" / "results"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
ADR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "nd_actual": "#E31A1C",  # Red for ND highlight
    "oil_state": "#8B4513",  # Brown for oil states
}


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.decisions: list[dict[str, Any]] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        evidence: str | None = None,
        reversible: bool = True,
    ) -> None:
        """Log a decision with full context."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": reversible,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "decisions": self.decisions,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str, output_dir: Path = RESULTS_DIR) -> Path:
        """Save results to JSON file."""
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


# =============================================================================
# DATA LOADING FROM POSTGRESQL
# =============================================================================


def load_state_panel_from_db(exclude_territories: bool = True) -> pd.DataFrame:
    """
    Load the 50-state population components panel data from PostgreSQL.

    Parameters
    ----------
    exclude_territories : bool
        If True, excludes DC and Puerto Rico.

    Returns
    -------
    pd.DataFrame
        State-level panel with columns: state, state_fips, year, population,
        intl_migration, and other demographic components.
    """
    import warnings

    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            year,
            state_name as state,
            state_fips,
            population,
            intl_migration,
            domestic_migration,
            net_migration,
            births,
            deaths,
            natural_change,
            pop_change
        FROM census.state_components
        WHERE state_name IS NOT NULL
        ORDER BY state_name, year
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} state-year observations from PostgreSQL")
    finally:
        conn.close()

    # Exclude territories if requested
    if exclude_territories:
        territories = ["District of Columbia", "Puerto Rico", "United States"]
        df = df[~df["state"].isin(territories)]
        print(f"After excluding territories: {len(df)} observations, "
              f"{df['state'].nunique()} states")

    # Sort by state and year
    df = df.sort_values(["state", "year"]).reset_index(drop=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for the loaded data."""
    return {
        "n_states": int(df["state"].nunique()),
        "n_observations": len(df),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "years_per_state": int(df.groupby("state").size().mode().iloc[0]),
        "states": sorted(df["state"].unique().tolist()),
        "mean_intl_migration_by_state": (
            df.groupby("state")["intl_migration"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ),
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_shift_distribution(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
    result: ModuleResult | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Create histogram of regime shifts across all states with ND highlighted.

    This is the key visualization answering: "Is ND's shift unusual?"
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get ND value
    nd_row = shift_df[shift_df["state"] == "North Dakota"]
    nd_value = nd_row[metric].iloc[0] if len(nd_row) > 0 else None

    # Get oil state values for separate coloring
    oil_states = shift_df[shift_df["state"].isin(ALL_OIL_STATES)]
    non_oil_states = shift_df[~shift_df["state"].isin(ALL_OIL_STATES)]

    # Histogram of all states
    all_values = shift_df[metric].dropna()
    bins_arr = np.linspace(all_values.min(), all_values.max(), 20)
    bins: list[float] = bins_arr.tolist()

    ax.hist(
        non_oil_states[metric].dropna(),
        bins=bins,
        color=COLORS["neutral"],
        alpha=0.6,
        label=f"Non-oil states (n={len(non_oil_states)})",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.hist(
        oil_states[metric].dropna(),
        bins=bins,
        color=COLORS["oil_state"],
        alpha=0.6,
        label=f"Oil/energy states (n={len(oil_states)})",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add vertical line for ND
    if nd_value is not None:
        ax.axvline(
            nd_value,
            color=COLORS["nd_actual"],
            linewidth=3,
            linestyle="--",
            label=f"North Dakota ({nd_value:.1f}%)",
        )

        # Add percentile annotation
        nd_pct = get_nd_percentile(shift_df, metric)
        ax.text(
            nd_value + 1,
            ax.get_ylim()[1] * 0.85,
            f"Rank: {nd_pct['rank']}/{nd_pct['n_states']}\n"
            f"Percentile: {nd_pct['percentile_from_top']:.1f}th\n"
            f"from top",
            fontsize=10,
            color=COLORS["nd_actual"],
            fontweight="bold",
            verticalalignment="top",
        )

    # Add mean line
    mean_val = all_values.mean()
    ax.axvline(
        mean_val,
        color="black",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
        label=f"National mean ({mean_val:.1f}%)",
    )

    # Labels and formatting
    metric_label = {
        "relative_shift": "Relative Shift (%)",
        "shift_magnitude": "Shift Magnitude (persons)",
        "cohens_d": "Cohen's d (effect size)",
    }.get(metric, metric)

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel("Number of States", fontsize=12)
    ax.set_title(
        "Distribution of International Migration Regime Shifts (2010-2019 vs 2021-2024)\n"
        "Comparing All 50 States",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = FIGURES_DIR / f"module_B2_shift_distribution_{metric}"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    # Also save to ADR directory
    adr_path = ADR_RESULTS_DIR.parent / "figures" / f"module_B2_shift_distribution_{metric}"
    adr_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


def plot_oil_vs_nonoil_boxplot(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
    result: ModuleResult | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Create box plot comparing oil/energy states vs non-oil states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Legacy oil state classification
    ax1 = axes[0]
    shift_df_copy = shift_df.copy()
    shift_df_copy["oil_category"] = shift_df_copy["state"].apply(
        lambda s: "Oil/Energy States" if s in ALL_OIL_STATES else "Non-Oil States"
    )

    categories = ["Oil/Energy States", "Non-Oil States"]
    data_by_cat = [
        shift_df_copy[shift_df_copy["oil_category"] == cat][metric].dropna().values
        for cat in categories
    ]

    bp1 = ax1.boxplot(
        data_by_cat,
        tick_labels=categories,
        patch_artist=True,
    )

    # Color boxes
    bp1["boxes"][0].set_facecolor(COLORS["oil_state"])
    bp1["boxes"][0].set_alpha(0.6)
    bp1["boxes"][1].set_facecolor(COLORS["neutral"])
    bp1["boxes"][1].set_alpha(0.6)

    # Add individual points
    for i, (cat, data) in enumerate(zip(categories, data_by_cat)):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax1.scatter(x, data, alpha=0.5, s=30, c=COLORS["primary"])

    # Highlight ND
    nd_val = shift_df_copy[shift_df_copy["state"] == "North Dakota"][metric].iloc[0]
    nd_cat_idx = 1 if "North Dakota" in ALL_OIL_STATES else 2
    ax1.scatter([nd_cat_idx], [nd_val], s=150, c=COLORS["nd_actual"],
                marker="*", zorder=10, label="North Dakota")

    ax1.set_ylabel("Relative Shift (%)", fontsize=11)
    ax1.set_title("Oil/Energy States vs. Non-Oil States", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right panel: Boom-timing classification
    ax2 = axes[1]

    # Add boom category (get_boom_category imported at module level)
    shift_df_copy["boom_category"] = shift_df_copy["state"].apply(get_boom_category)

    boom_categories = ["Bakken Boom", "Permian Boom", "Other Shale", "Mature Oil", "Non-Oil"]
    boom_colors = [COLORS["nd_actual"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["oil_state"], COLORS["neutral"]]

    data_by_boom = []
    for cat in boom_categories:
        cat_data = shift_df_copy[shift_df_copy["boom_category"] == cat][metric].dropna().values
        data_by_boom.append(cat_data if len(cat_data) > 0 else [np.nan])

    bp2 = ax2.boxplot(
        data_by_boom,
        tick_labels=[f"{c}\n(n={len(d)})" for c, d in zip(boom_categories, data_by_boom)],
        patch_artist=True,
    )

    # Color boxes
    for patch, color in zip(bp2["boxes"], boom_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Highlight ND
    nd_boom_idx = boom_categories.index("Bakken Boom") + 1
    ax2.scatter([nd_boom_idx], [nd_val], s=150, c=COLORS["nd_actual"],
                marker="*", zorder=10, label="North Dakota")

    ax2.set_ylabel("Relative Shift (%)", fontsize=11)
    ax2.set_title("Boom-Timing Classification\n(2008-2015 boom alignment)",
                  fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Comparing Regime Shifts by Oil/Energy State Classification",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = FIGURES_DIR / "module_B2_oil_state_comparison"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_RESULTS_DIR.parent / "figures" / "module_B2_oil_state_comparison"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


def plot_state_ranking(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
    top_n: int = 20,
    save_path: Path | None = None,
) -> None:
    """
    Create horizontal bar chart of top N states by regime shift.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Rank states
    ranked = rank_states_by_shift(shift_df, metric, ascending=False)
    top_states = ranked.head(top_n)

    # Colors based on oil state status
    colors = [
        COLORS["nd_actual"] if state == "North Dakota"
        else COLORS["oil_state"] if state in ALL_OIL_STATES
        else COLORS["primary"]
        for state in top_states["state"]
    ]

    y_pos = range(len(top_states) - 1, -1, -1)  # Reverse for top-to-bottom
    bars = ax.barh(y_pos, top_states[metric].values, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_states["state"].values)

    # Add rank numbers
    for i, (bar, rank) in enumerate(zip(bars, top_states["rank"])):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"#{rank}",
            va="center",
            fontsize=9,
        )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["nd_actual"], alpha=0.7, label="North Dakota"),
        Patch(facecolor=COLORS["oil_state"], alpha=0.7, label="Oil/Energy State"),
        Patch(facecolor=COLORS["primary"], alpha=0.7, label="Non-Oil State"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    metric_label = {
        "relative_shift": "Relative Shift (%)",
        "shift_magnitude": "Shift Magnitude (persons)",
    }.get(metric, metric)

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_title(
        f"Top {top_n} States by International Migration Regime Shift\n"
        "(2010-2019 vs 2021-2024)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"module_B2_state_ranking_{metric}"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_RESULTS_DIR.parent / "figures" / f"module_B2_state_ranking_{metric}"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_analysis() -> ModuleResult:
    """Main analysis function for multi-state placebo analysis."""
    result = ModuleResult(
        module_id="B2",
        analysis_name="multistate_placebo_analysis",
    )

    print("=" * 70)
    print("Module B2: Multi-State Placebo Analysis")
    print("ADR-020 Key Question: Is ND's shift unusual or nationwide?")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # ==========================================================================
    # 1. Load Data from PostgreSQL
    # ==========================================================================
    print("\n[1/7] Loading 50-state panel data from PostgreSQL...")

    df = load_state_panel_from_db(exclude_territories=True)
    df = add_vintage_labels(df, "year")

    data_summary = get_data_summary(df)
    result.input_files.append("census.state_components (PostgreSQL)")
    result.parameters["data_source"] = "PostgreSQL: census.state_components"
    result.parameters["data_summary"] = data_summary

    print(f"  States: {data_summary['n_states']}")
    print(f"  Year range: {data_summary['year_range']}")
    print(f"  Observations: {data_summary['n_observations']}")

    result.add_decision(
        decision_id="B2001",
        category="data_selection",
        decision="Use census.state_components from PostgreSQL for 50-state panel",
        rationale="Consistent data source with other modules; covers 2010-2024",
        alternatives=["Use CSV exports", "Use ACS data"],
        evidence=f"{data_summary['n_states']} states, {data_summary['n_observations']} obs",
    )

    # ==========================================================================
    # 2. Calculate Regime Shifts for All States
    # ==========================================================================
    print("\n[2/7] Calculating regime shifts for all 50 states...")

    shift_df = calculate_all_state_shifts(df, exclude_2020=True)
    print(f"  Calculated shifts for {len(shift_df)} states")

    result.parameters["regime_comparison"] = {
        "pre_period": "2010-2019 (Vintage 2020)",
        "post_period": "2021-2024 (Vintage 2024, excluding 2020)",
        "exclude_2020": True,
        "rationale": "2020 excluded as COVID outlier year",
    }

    # Store all state shifts for reference
    result.results["all_state_shifts"] = shift_df.to_dict("records")

    result.add_decision(
        decision_id="B2002",
        category="methodology",
        decision="Exclude 2020 from post-period comparison",
        rationale="2020 is a COVID outlier; including it would bias shift estimates downward",
        alternatives=["Include 2020", "Use COVID-adjusted values"],
        evidence="2020 had near-zero migration for many states",
    )

    # ==========================================================================
    # 3. Rank States and Get ND Percentile (KEY FINDING)
    # ==========================================================================
    print("\n[3/7] Ranking all states by shift magnitude...")

    # Analyze multiple metrics
    metrics = ["relative_shift", "shift_magnitude", "cohens_d"]
    nd_rankings: dict[str, Any] = {}

    for metric in metrics:
        nd_pct = get_nd_percentile(shift_df, metric)
        nd_rankings[metric] = nd_pct
        print(f"\n  {metric}:")
        print(f"    ND value: {nd_pct['value']:.2f}")
        print(f"    Rank: {nd_pct['rank']}/{nd_pct['n_states']}")
        print(f"    Percentile from top: {nd_pct['percentile_from_top']:.1f}th")
        print(f"    Interpretation: {nd_pct['interpretation']}")

    result.results["nd_national_ranking"] = nd_rankings

    # Get full ranking table
    ranked_df = rank_states_by_shift(shift_df, "relative_shift", ascending=False)
    result.results["full_ranking"] = ranked_df[
        ["state", "rank", "relative_shift", "shift_magnitude", "cohens_d", "p_value"]
    ].to_dict("records")

    # ==========================================================================
    # 4. Oil/Energy State Hypothesis Test
    # ==========================================================================
    print("\n[4/7] Testing oil/energy state hypothesis...")

    # Legacy oil state test
    oil_test = run_oil_state_hypothesis_test(shift_df, metric="relative_shift")
    print("\n  Legacy Oil State Test:")
    print(f"    Oil states mean: {oil_test['oil_mean']:.1f}%")
    print(f"    Non-oil states mean: {oil_test['non_oil_mean']:.1f}%")
    print(f"    Difference: {oil_test['difference']:.1f}%")
    print(f"    t-test p-value: {oil_test['t_test']['p_value']:.4f}")
    print(f"    Interpretation: {oil_test['interpretation']}")

    result.results["oil_state_hypothesis"] = oil_test

    # ND rank among oil states
    nd_oil_rank = get_nd_rank_among_oil_states(shift_df, "relative_shift")
    print("\n  ND among oil states:")
    print(f"    Rank: {nd_oil_rank['rank_among_oil']}/{nd_oil_rank['n_oil_states']}")
    print(f"    Interpretation: {nd_oil_rank['interpretation']}")

    result.results["nd_oil_state_ranking"] = nd_oil_rank

    # ==========================================================================
    # 5. Boom-Timing Hypothesis Test (Improved Classification)
    # ==========================================================================
    print("\n[5/7] Testing boom-timing hypothesis (Bakken vs Permian)...")

    # Boom state test
    boom_test = run_boom_state_hypothesis_test(shift_df, metric="relative_shift")
    print("\n  Boom State Test (timing-based):")
    print(f"    Boom states mean: {boom_test['boom_mean']:.1f}%")
    print(f"    Non-oil states mean: {boom_test['non_oil_mean']:.1f}%")
    print(f"    t-test p-value: {boom_test['t_test']['p_value']:.4f}")
    print(f"    Interpretation: {boom_test['interpretation']}")

    result.results["boom_timing_hypothesis"] = boom_test

    # Bakken-specific analysis
    bakken_test = run_bakken_specific_hypothesis_test(shift_df, metric="relative_shift")
    print("\n  Bakken vs Permian Comparison:")
    bakken_vs_permian = bakken_test["bakken_vs_permian"]
    print(f"    Bakken mean: {bakken_vs_permian['bakken_mean']:.1f}%")
    print(f"    Permian mean: {bakken_vs_permian['permian_mean']:.1f}%")
    print(f"    Interpretation: {bakken_test['interpretation']}")

    result.results["bakken_specific_analysis"] = bakken_test

    # ND rank among boom states
    nd_boom_rank = get_nd_rank_among_boom_states(shift_df, "relative_shift")
    print("\n  ND among boom states:")
    print(f"    Rank: {nd_boom_rank['rank_among_boom']}/{nd_boom_rank['n_boom_states']}")
    print(f"    Interpretation: {nd_boom_rank['interpretation']}")

    result.results["nd_boom_state_ranking"] = nd_boom_rank

    # Category comparison table
    boom_comparison = compare_boom_categories(shift_df)
    result.results["boom_category_comparison"] = boom_comparison.to_dict("records")

    result.add_decision(
        decision_id="B2003",
        category="hypothesis_testing",
        decision="Use boom-timing classification as primary energy state grouping",
        rationale="Boom timing (2008-2015 vs 2015+) better captures the Bakken "
                 "effect than static oil production levels",
        alternatives=["Static oil state classification", "Production volume classification"],
        evidence=f"Bakken mean={bakken_vs_permian['bakken_mean']:.1f}% vs "
                f"Permian mean={bakken_vs_permian['permian_mean']:.1f}%",
    )

    # ==========================================================================
    # 6. Generate Visualizations
    # ==========================================================================
    print("\n[6/7] Generating visualizations...")

    # Distribution histogram with ND highlighted
    plot_shift_distribution(shift_df, metric="relative_shift", result=result)

    # Oil vs non-oil box plot
    plot_oil_vs_nonoil_boxplot(shift_df, metric="relative_shift", result=result)

    # State ranking bar chart
    plot_state_ranking(shift_df, metric="relative_shift", top_n=25)

    # ==========================================================================
    # 7. Synthesize Key Finding
    # ==========================================================================
    print("\n[7/7] Synthesizing key findings...")

    # Determine the answer to the key question
    nd_rel_pct = nd_rankings["relative_shift"]["percentile_from_top"]
    nd_rel_rank = nd_rankings["relative_shift"]["rank"]
    n_states = nd_rankings["relative_shift"]["n_states"]

    if nd_rel_pct <= 10:
        answer = "ND is EXCEPTIONAL - in top 10% of all states"
        supports = "real_driver"
        confidence = "high"
    elif nd_rel_pct <= 25:
        answer = "ND is UNUSUAL - in top 25% of all states"
        supports = "real_driver"
        confidence = "moderate"
    elif nd_rel_pct <= 50:
        answer = "ND is ABOVE AVERAGE but not exceptional"
        supports = "mixed"
        confidence = "low"
    else:
        answer = "ND is TYPICAL or below average"
        supports = "methodology_artifact"
        confidence = "moderate"

    key_finding = {
        "question": "Is North Dakota's international migration shift unusual compared "
                   "to other states, or is it a nationwide pattern?",
        "answer": answer,
        "nd_percentile_from_top": nd_rel_pct,
        "nd_rank": nd_rel_rank,
        "n_states_analyzed": n_states,
        "supports_hypothesis": supports,
        "confidence": confidence,
        "interpretation": (
            f"North Dakota ranks #{nd_rel_rank} out of {n_states} states in "
            f"relative regime shift magnitude ({nd_rel_pct:.1f}th percentile from top). "
            f"{answer}."
        ),
        "oil_state_context": (
            f"Among oil/energy states, ND ranks "
            f"#{nd_oil_rank['rank_among_oil']}/{nd_oil_rank['n_oil_states']}. "
            f"Oil states as a group "
            f"{'show significantly' if oil_test['significant_at_05'] else 'do not show significantly'} "
            f"higher shifts than non-oil states "
            f"(p={oil_test['t_test']['p_value']:.4f})."
        ),
        "boom_timing_context": (
            f"Using boom-timing classification, Bakken boom states (ND, MT) show "
            f"{'higher' if bakken_vs_permian['difference'] > 0 else 'lower'} shifts "
            f"than Permian boom states (TX, NM), suggesting timing of the oil boom matters."
        ),
    }

    result.results["key_finding"] = key_finding

    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print(f"\nQuestion: {key_finding['question']}")
    print(f"\nAnswer: {key_finding['answer']}")
    print(f"\n{key_finding['interpretation']}")
    print(f"\n{key_finding['oil_state_context']}")
    print(f"\n{key_finding['boom_timing_context']}")

    # Diagnostics
    result.diagnostics = {
        "data_coverage": {
            "states_analyzed": n_states,
            "year_range": data_summary["year_range"],
            "pre_period_years": 10,  # 2010-2019
            "post_period_years": 4,  # 2021-2024
        },
        "nd_shift_statistics": shift_df[shift_df["state"] == "North Dakota"].iloc[0].to_dict(),
        "national_shift_statistics": {
            "mean_relative_shift": float(shift_df["relative_shift"].mean()),
            "std_relative_shift": float(shift_df["relative_shift"].std()),
            "median_relative_shift": float(shift_df["relative_shift"].median()),
        },
    }

    # Warnings
    if data_summary["year_range"][0] > 2000:
        result.warnings.append(
            f"Data starts at {data_summary['year_range'][0]}, not 2000. "
            "Cannot analyze pre-2010 vintage."
        )

    if oil_test["t_test"]["p_value"] > 0.05:
        result.warnings.append(
            "Oil state hypothesis test not significant at 5% level. "
            "Oil state classification may not be the primary driver."
        )

    # Next steps
    result.next_steps = [
        "Integrate findings with ADR-021 regime framework analysis",
        "Consider per-capita migration rates for size-normalized comparison",
        "Explore lagged effects (boom timing + adjustment period)",
        "Cross-validate with ACS foreign-born population changes",
    ]

    return result


def main() -> int:
    """Main entry point."""
    try:
        result = run_analysis()

        # Save main results
        output_file = result.save("module_B2_multistate_placebo.json")

        # Also save to ADR-020 results directory
        adr_output = result.save(
            "module_B2_multistate_placebo.json",
            output_dir=ADR_RESULTS_DIR
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nMain output: {output_file}")
        print(f"ADR output: {adr_output}")

        # Print summary table
        if "full_ranking" in result.results:
            print("\n" + "-" * 70)
            print("TOP 10 STATES BY RELATIVE SHIFT MAGNITUDE")
            print("-" * 70)
            rankings = result.results["full_ranking"][:10]
            print(f"{'Rank':<6}{'State':<25}{'Rel Shift %':<15}{'p-value':<12}")
            print("-" * 58)
            for r in rankings:
                pval_str = f"{r['p_value']:.4f}" if r['p_value'] else "N/A"
                print(f"{r['rank']:<6}{r['state']:<25}{r['relative_shift']:<15.1f}{pval_str:<12}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")

        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
