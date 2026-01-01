#!/usr/bin/env python3
"""
Module B2: Multi-State Placebo Analysis
========================================

Tests whether North Dakota's vintage transition patterns are unusual
compared to other states, supporting the "real driver" vs.
"methodology artifact" interpretation per ADR-020.

Key Questions:
1. Where does ND rank in the national distribution of regime shifts?
2. Do oil/energy states cluster with higher shifts?
3. Is ND unusual even among oil states?
4. NEW: Do boom-timing states (Bakken 2008-2015) differ from other states?

Classification update (2026-01-01):
- Original analysis used ad-hoc oil state classification (top EIA producers)
- Research found boom timing matters more than static production levels
- Now includes boom-timing classification per OIL_STATE_RESEARCH.md:
  * Bakken Boom: ND, MT (2008-2015 boom, ~40% CAGR)
  * Permian Boom: TX, NM (two-phase boom)
  * Other Shale: CO, OK, LA
  * Mature Oil: CA, AK, WY, KS (stable/declining)
  * Non-Oil: All others

Usage:
    uv run python module_B2_multistate_placebo.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Import B2 module components
from module_B2_multistate_placebo import (  # noqa: E402
    load_state_panel,
    add_vintage_labels,
    calculate_all_state_shifts,
    rank_states_by_shift,
    get_nd_percentile,
    # Legacy oil state classification
    test_oil_state_hypothesis,
    compare_oil_vs_non_oil,
    get_nd_rank_among_oil_states,
    OIL_STATES,
    SECONDARY_OIL_STATES,
    ALL_OIL_STATES,
    # Boom-timing classification (new)
    BAKKEN_BOOM_STATES,
    PERMIAN_BOOM_STATES,
    OTHER_SHALE_STATES,
    MATURE_OIL_STATES,
    ALL_BOOM_STATES,
    get_boom_category,
    test_boom_state_hypothesis,
    test_bakken_specific_hypothesis,
    get_nd_rank_among_boom_states,
    compare_boom_categories,
)


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.warnings: list[str] = []
        self.next_steps: list[str] = []

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def plot_shift_distribution(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
    output_path: Path = None,
) -> None:
    """Create histogram of shift distribution with ND highlighted."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get ND value
    nd_value = shift_df[shift_df["state"] == "North Dakota"][metric].values
    nd_value = nd_value[0] if len(nd_value) > 0 else None

    # Histogram
    values = shift_df[metric].dropna()
    ax.hist(values, bins=20, color="#3498db", alpha=0.7, edgecolor="black")

    # Highlight ND
    if nd_value is not None:
        ax.axvline(
            x=nd_value,
            color="#e74c3c",
            linewidth=3,
            linestyle="--",
            label=f"North Dakota: {nd_value:,.0f}",
        )

    # Add distribution stats
    mean_val = values.mean()
    ax.axvline(
        x=mean_val,
        color="#2ecc71",
        linewidth=2,
        linestyle="-",
        label=f"National Mean: {mean_val:,.0f}",
    )

    ax.set_xlabel(f"{metric.replace('_', ' ').title()}")
    ax.set_ylabel("Number of States")
    ax.set_title(
        f"Distribution of {metric.replace('_', ' ').title()} Across 50 States\n"
        f"(Vintage 2020 to Vintage 2024 Transition)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_oil_state_comparison(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
    output_path: Path = None,
) -> None:
    """Create box plot comparing oil vs non-oil states."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add category
    df = shift_df.copy()
    df["category"] = df["state"].apply(
        lambda s: "Primary Oil"
        if s in OIL_STATES
        else ("Secondary Oil" if s in SECONDARY_OIL_STATES else "Non-Oil")
    )

    # Get values by category
    categories = ["Primary Oil", "Secondary Oil", "Non-Oil"]
    data = [df[df["category"] == cat][metric].dropna().values for cat in categories]
    colors = ["#e74c3c", "#f39c12", "#3498db"]

    # Box plot
    bp = ax.boxplot(data, tick_labels=categories, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight ND
    nd_row = df[df["state"] == "North Dakota"]
    if len(nd_row) > 0:
        nd_value = nd_row[metric].values[0]
        nd_cat = nd_row["category"].values[0]
        cat_idx = categories.index(nd_cat) + 1
        ax.scatter(
            [cat_idx],
            [nd_value],
            s=200,
            c="#000000",
            marker="*",
            zorder=10,
            label="North Dakota",
        )

    ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax.set_title(
        "Regime Shift by Oil State Category\n"
        "(Comparing Vintage 2020 to Vintage 2024)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_state_rankings(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
    top_n: int = 20,
    output_path: Path = None,
) -> None:
    """Create bar chart of top states by shift magnitude."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Rank and get top N
    ranked = rank_states_by_shift(shift_df, metric, ascending=False)
    top = ranked.head(top_n)

    # Colors based on oil state status
    colors = [
        "#e74c3c"
        if s in OIL_STATES
        else ("#f39c12" if s in SECONDARY_OIL_STATES else "#3498db")
        for s in top["state"]
    ]

    # Horizontal bar chart
    y_pos = range(len(top))
    ax.barh(y_pos, top[metric], color=colors, alpha=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["state"])
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Primary Oil State"),
        Patch(facecolor="#f39c12", alpha=0.7, label="Secondary Oil State"),
        Patch(facecolor="#3498db", alpha=0.7, label="Non-Oil State"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_xlabel(f"{metric.replace('_', ' ').title()}")
    ax.set_title(
        f"Top {top_n} States by {metric.replace('_', ' ').title()}\n"
        f"(Vintage 2020 to Vintage 2024 Transition)"
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_boom_category_comparison(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
    output_path: Path = None,
) -> None:
    """Create box plot comparing boom-timing categories.

    Uses boom-timing classification per OIL_STATE_RESEARCH.md:
    - Bakken Boom: ND, MT
    - Permian Boom: TX, NM
    - Other Shale: CO, OK, LA
    - Mature Oil: CA, AK, WY, KS
    - Non-Oil: All others
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add boom category
    df = shift_df.copy()
    df["boom_category"] = df["state"].apply(get_boom_category)

    # Define categories in order
    categories = ["Bakken Boom", "Permian Boom", "Other Shale", "Mature Oil", "Non-Oil"]
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#95a5a6", "#3498db"]

    # Get values by category
    data = []
    for cat in categories:
        vals = df[df["boom_category"] == cat][metric].dropna().values
        data.append(vals if len(vals) > 0 else [0])

    # Box plot
    bp = ax.boxplot(data, tick_labels=categories, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight ND
    nd_row = df[df["state"] == "North Dakota"]
    if len(nd_row) > 0:
        nd_value = nd_row[metric].values[0]
        nd_cat = nd_row["boom_category"].values[0]
        if nd_cat in categories:
            cat_idx = categories.index(nd_cat) + 1
            ax.scatter(
                [cat_idx],
                [nd_value],
                s=200,
                c="#000000",
                marker="*",
                zorder=10,
                label="North Dakota",
            )

    # Add counts
    for i, cat in enumerate(categories):
        n = len(df[df["boom_category"] == cat])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center", va="top", fontsize=9)

    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(
        "Regime Shift by Boom-Timing Category\n"
        "(Classification based on 2008-2015 production growth)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def run_analysis() -> ModuleResult:
    """
    Main analysis function implementing all B2 components.

    Returns:
        ModuleResult object with all findings
    """
    result = ModuleResult(
        module_id="B2",
        analysis_name="multistate_placebo",
    )

    print("\n" + "=" * 60)
    print("1. LOADING STATE PANEL DATA")
    print("=" * 60)

    # Load data
    df = load_state_panel(exclude_territories=True)
    df = add_vintage_labels(df)
    result.input_files.append("07_data_panel.csv")

    n_states = df["state"].nunique()
    year_range = f"{df['year'].min()}-{df['year'].max()}"

    print(f"Loaded {len(df)} observations for {n_states} states")
    print(f"Year range: {year_range}")
    print(f"Vintages: {df['vintage'].unique().tolist()}")

    result.parameters = {
        "n_states": n_states,
        "year_range": year_range,
        "exclude_2020_from_post": True,
        "primary_oil_states": OIL_STATES,
        "secondary_oil_states": SECONDARY_OIL_STATES,
    }

    print("\n" + "=" * 60)
    print("2. CALCULATING STATE SHIFTS")
    print("=" * 60)

    # Calculate shifts for all states
    shift_df = calculate_all_state_shifts(df, exclude_2020=True)

    print(f"Calculated shifts for {len(shift_df)} states")
    print("\nShift magnitude summary:")
    print(f"  Mean: {shift_df['shift_magnitude'].mean():,.0f}")
    print(f"  Std: {shift_df['shift_magnitude'].std():,.0f}")
    print(
        f"  Min: {shift_df['shift_magnitude'].min():,.0f} ({shift_df.loc[shift_df['shift_magnitude'].idxmin(), 'state']})"
    )
    print(
        f"  Max: {shift_df['shift_magnitude'].max():,.0f} ({shift_df.loc[shift_df['shift_magnitude'].idxmax(), 'state']})"
    )

    # Save shift rankings
    ranked = rank_states_by_shift(shift_df, "shift_magnitude", ascending=False)
    ranked.to_csv(RESULTS_DIR / "module_B2_state_shift_rankings.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'module_B2_state_shift_rankings.csv'}")

    result.results["state_shifts"] = {
        "n_states": len(shift_df),
        "shift_magnitude_stats": {
            "mean": float(shift_df["shift_magnitude"].mean()),
            "std": float(shift_df["shift_magnitude"].std()),
            "min": float(shift_df["shift_magnitude"].min()),
            "max": float(shift_df["shift_magnitude"].max()),
        },
    }

    print("\n" + "=" * 60)
    print("3. NORTH DAKOTA'S POSITION IN NATIONAL DISTRIBUTION")
    print("=" * 60)

    # Get ND percentile
    nd_percentile = get_nd_percentile(shift_df, "shift_magnitude")

    print("\nNorth Dakota:")
    print(f"  Shift magnitude: {nd_percentile['value']:,.0f}")
    print(f"  National rank: {nd_percentile['rank']} of {nd_percentile['n_states']}")
    print(f"  Percentile from top: {nd_percentile['percentile_from_top']:.1f}%")
    print(f"  Interpretation: {nd_percentile['interpretation']}")

    result.results["nd_national_position"] = nd_percentile

    # Create distribution plot
    plot_shift_distribution(
        shift_df, "shift_magnitude", FIGURES_DIR / "module_B2_shift_distribution.png"
    )

    print("\n" + "=" * 60)
    print("4. OIL STATE HYPOTHESIS TEST")
    print("=" * 60)

    # Test oil state hypothesis
    oil_test = test_oil_state_hypothesis(shift_df, "shift_magnitude")

    print("\nOil vs Non-Oil States Comparison:")
    print(f"  Oil states mean shift: {oil_test['oil_mean']:,.0f}")
    print(f"  Non-oil states mean shift: {oil_test['non_oil_mean']:,.0f}")
    print(f"  Difference: {oil_test['difference']:,.0f}")
    print(f"  t-test p-value: {oil_test['t_test']['p_value']:.4f}")
    print(f"  Significant at 5%: {oil_test['significant_at_05']}")
    print(f"  Interpretation: {oil_test['interpretation']}")

    result.results["oil_state_hypothesis"] = oil_test

    # Compare oil categories
    oil_comparison = compare_oil_vs_non_oil(shift_df)
    oil_comparison.to_csv(
        RESULTS_DIR / "module_B2_oil_states_analysis.csv", index=False
    )
    print(f"\nSaved: {RESULTS_DIR / 'module_B2_oil_states_analysis.csv'}")

    print("\nOil State Category Comparison:")
    print(oil_comparison.to_string(index=False))

    # Create oil state comparison plot
    plot_oil_state_comparison(
        shift_df, "shift_magnitude", FIGURES_DIR / "module_B2_oil_states_comparison.png"
    )

    print("\n" + "=" * 60)
    print("5. ND RANK AMONG OIL STATES")
    print("=" * 60)

    # Rank ND among oil states
    nd_oil_rank = get_nd_rank_among_oil_states(shift_df, "shift_magnitude")

    print("\nNorth Dakota among oil states:")
    print(f"  Rank: {nd_oil_rank['rank_among_oil']} of {nd_oil_rank['n_oil_states']}")
    print(f"  Percentile among oil: {nd_oil_rank['percentile_among_oil']:.1f}%")
    print(f"  Interpretation: {nd_oil_rank['interpretation']}")

    print("\nAll oil states ranked by shift:")
    for entry in nd_oil_rank["all_oil_rankings"]:
        marker = "**" if entry["state"] == "North Dakota" else "  "
        print(
            f"  {marker}{entry['rank']}. {entry['state']}: {entry['shift_magnitude']:,.0f}"
        )

    result.results["nd_among_oil_states"] = {
        k: v for k, v in nd_oil_rank.items() if k != "all_oil_rankings"
    }
    result.results["oil_state_rankings"] = nd_oil_rank["all_oil_rankings"]

    # Create rankings plot
    plot_state_rankings(
        shift_df,
        "shift_magnitude",
        top_n=20,
        output_path=FIGURES_DIR / "module_B2_state_rankings.png",
    )

    print("\n" + "=" * 60)
    print("6. BOOM-TIMING HYPOTHESIS TEST (NEW)")
    print("=" * 60)
    print("\nUsing evidence-based boom-timing classification per OIL_STATE_RESEARCH.md")
    print(f"  Bakken Boom States: {BAKKEN_BOOM_STATES}")
    print(f"  Permian Boom States: {PERMIAN_BOOM_STATES}")
    print(f"  Other Shale States: {OTHER_SHALE_STATES}")
    print(f"  Mature Oil States: {MATURE_OIL_STATES}")

    # Test boom state hypothesis (using relative_shift to normalize for state size)
    boom_test = test_boom_state_hypothesis(shift_df, "relative_shift")

    print("\nBoom States vs Non-Oil States (relative shift %):")
    print(f"  Boom states mean: {boom_test['boom_mean']:.1f}%")
    print(f"  Non-oil states mean: {boom_test['non_oil_mean']:.1f}%")
    print(f"  Difference: {boom_test['difference']:.1f} percentage points")
    print(
        f"  t-test p-value: {boom_test['t_test']['p_value']:.4f}"
        if boom_test["t_test"]["p_value"]
        else "  t-test p-value: N/A"
    )
    print(f"  Significant at 5%: {boom_test['significant_at_05']}")
    print(f"  Interpretation: {boom_test['interpretation']}")

    result.results["boom_state_hypothesis"] = boom_test

    # Bakken-specific test
    bakken_test = test_bakken_specific_hypothesis(shift_df, "relative_shift")

    print("\nBakken vs Permian Comparison:")
    bvp = bakken_test["bakken_vs_permian"]
    print(
        f"  Bakken mean: {bvp['bakken_mean']:.1f}%"
        if bvp["bakken_mean"]
        else "  Bakken mean: N/A"
    )
    print(
        f"  Permian mean: {bvp['permian_mean']:.1f}%"
        if bvp["permian_mean"]
        else "  Permian mean: N/A"
    )
    print(
        f"  Difference: {bvp['difference']:.1f} percentage points"
        if bvp["difference"]
        else "  Difference: N/A"
    )
    print(f"  Interpretation: {bakken_test['interpretation']}")

    print("\nGroup Statistics (relative shift %):")
    for group, stats in bakken_test["group_statistics"].items():
        if stats["mean"] is not None:
            print(f"  {group}: mean={stats['mean']:.1f}%, n={stats['n']}")

    result.results["bakken_specific_hypothesis"] = bakken_test

    # ND rank among boom states
    nd_boom_rank = get_nd_rank_among_boom_states(shift_df, "relative_shift")

    print("\nNorth Dakota among boom states (relative shift):")
    print(
        f"  Rank: {nd_boom_rank['rank_among_boom']} of {nd_boom_rank['n_boom_states']}"
    )
    print(f"  Percentile among boom: {nd_boom_rank['percentile_among_boom']:.1f}%")
    print(f"  ND relative shift: {nd_boom_rank['value']:.1f}%")
    print(f"  Interpretation: {nd_boom_rank['interpretation']}")

    print("\nAll boom states ranked by relative shift:")
    for entry in nd_boom_rank["all_boom_rankings"]:
        marker = "**" if entry["state"] == "North Dakota" else "  "
        print(
            f"  {marker}{entry['rank']}. {entry['state']} ({entry['boom_category']}): {entry['relative_shift']:.1f}%"
        )

    result.results["nd_among_boom_states"] = {
        k: v for k, v in nd_boom_rank.items() if k != "all_boom_rankings"
    }
    result.results["boom_state_rankings"] = nd_boom_rank["all_boom_rankings"]

    # Save boom category comparison
    boom_comparison = compare_boom_categories(shift_df)
    boom_comparison.to_csv(
        RESULTS_DIR / "module_B2_boom_category_analysis.csv", index=False
    )
    print(f"\nSaved: {RESULTS_DIR / 'module_B2_boom_category_analysis.csv'}")

    # Create boom category comparison plot
    plot_boom_category_comparison(
        shift_df,
        "relative_shift",
        FIGURES_DIR / "module_B2_boom_category_comparison.png",
    )

    print("\n" + "=" * 60)
    print("7. KEY FINDINGS SUMMARY")
    print("=" * 60)

    # Synthesize findings
    # Legacy metrics (static oil state classification)
    nd_in_top_10 = nd_percentile["percentile_from_top"] <= 10
    oil_significant = oil_test["significant_at_05"]
    nd_top_oil = nd_oil_rank["rank_among_oil"] <= 3

    # New boom-timing metrics
    boom_significant = boom_test["significant_at_05"]
    nd_top_boom = nd_boom_rank["rank_among_boom"] <= 3

    # Revised assessment using boom-timing classification
    # This is the better test per OIL_STATE_RESEARCH.md
    if boom_significant and nd_top_boom:
        overall = (
            "STRONG SUPPORT for real driver hypothesis (boom-timing classification)"
        )
    elif nd_top_boom or boom_significant:
        overall = (
            "MODERATE SUPPORT for real driver hypothesis (boom-timing classification)"
        )
    elif nd_in_top_10 or (oil_significant and nd_top_oil):
        overall = "MODERATE SUPPORT based on legacy classification"
    else:
        overall = "WEAK SUPPORT - methodology artifact cannot be ruled out"

    print("\nLegacy Analysis (static oil state classification):")
    print(f"  1. ND in top 10% nationally: {nd_in_top_10}")
    print(f"  2. Oil states significantly different: {oil_significant}")
    print(f"  3. ND in top 3 among oil states: {nd_top_oil}")

    print("\nBoom-Timing Analysis (evidence-based classification):")
    print(f"  4. Boom states significantly different: {boom_significant}")
    print(f"  5. ND in top 3 among boom states: {nd_top_boom}")
    print(
        f"  6. ND rank among boom states: {nd_boom_rank['rank_among_boom']} of {nd_boom_rank['n_boom_states']}"
    )

    print(f"\nOverall Assessment: {overall}")

    result.results["key_findings"] = {
        # Legacy
        "nd_in_top_10_nationally": nd_in_top_10,
        "oil_states_significantly_different": oil_significant,
        "nd_top_3_among_oil": nd_top_oil,
        # Boom-timing (new)
        "boom_states_significantly_different": boom_significant,
        "nd_top_3_among_boom": nd_top_boom,
        "nd_rank_among_boom": nd_boom_rank["rank_among_boom"],
        "nd_relative_shift": nd_boom_rank["value"],
        "overall_assessment": overall,
    }

    # Diagnostics
    result.diagnostics = {
        "data_years": year_range,
        "n_states_analyzed": n_states,
        "n_oil_states_legacy": len(ALL_OIL_STATES),
        "n_boom_states": len(ALL_BOOM_STATES),
        "exclude_2020": True,
        "comparison_periods": {
            "pre": "2010-2019 (Vintage 2020)",
            "post": "2021-2024 (Vintage 2024, excluding 2020)",
        },
        "classification_methodology": "Boom-timing per OIL_STATE_RESEARCH.md",
        "boom_categories": {
            "bakken_boom": BAKKEN_BOOM_STATES,
            "permian_boom": PERMIAN_BOOM_STATES,
            "other_shale": OTHER_SHALE_STATES,
            "mature_oil": MATURE_OIL_STATES,
        },
    }

    # Warnings
    if n_states < 50:
        result.warnings.append(f"Only {n_states} states analyzed (expected 50)")

    if not boom_significant:
        result.warnings.append(
            "Boom states not significantly different from non-oil states - "
            "consider examining relative shift metric or focusing on 2009-2020 vintage transition"
        )

    # Next steps
    result.next_steps = [
        "B3: Incorporate boom-timing findings into journal article methodology section",
        "B3: Document Bakken-specific timing (2008-2015) vs Permian (2015+)",
        "Consider extending analysis to 2009-to-2020 vintage transition (captures full Bakken boom)",
        "B4: Use boom-category fixed effects in panel models",
        "B6: Create unit tests for B2 module functions",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module B2: Multi-State Placebo Analysis")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_B2_multistate_placebo.json")
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        if result.next_steps:
            print("\nNext Steps:")
            for s in result.next_steps:
                print(f"  - {s}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
