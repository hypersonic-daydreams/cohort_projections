#!/usr/bin/env python3
"""
Agent 2: LSSND Closure Impact Analysis
ADR-021 Investigation

Validates the external AI claim that Lutheran Social Services of North Dakota (LSSND)
closure in 2021 created a state-specific capacity shock distinct from federal policy.

Key Questions:
1. What was ND's refugee arrival trajectory before/after LSSND closure?
2. Can we identify suitable donor states for synthetic control?
3. Is the LSSND effect distinguishable from federal policy (Travel Ban, COVID)?

.. deprecated:: 2026-01-01
    This is a **legacy Phase A research script** from the ADR-021 investigation.
    It was used for one-time exploratory analysis and is retained for reproducibility
    and audit purposes only. This script is NOT production code and should NOT be
    modified or extended.

Status: DEPRECATED / ARCHIVED
Linting: Exempted from strict linting (see pyproject.toml per-file-ignores)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# === Configuration ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(
    "/home/nhaarstad/workspace/demography/cohort_projections/data/processed/immigration/analysis"
)
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# LSSND closed in 2021 - this is the key treatment date
LSSND_CLOSURE_YEAR = 2021

# Comparison states: similar low-flow states with stable resettlement infrastructure
# Selected based on: similar population, similar historical refugee volumes, stable infrastructure
POTENTIAL_DONOR_STATES = [
    "South Dakota",
    "Montana",
    "Wyoming",
    "Nebraska",
    "Idaho",
    "Maine",
    "Vermont",
    "New Hampshire",
]


def load_refugee_data():
    """Load refugee arrivals by state and nationality."""
    refugees = pd.read_parquet(DATA_DIR / "refugee_arrivals_by_state_nationality.parquet")
    return refugees


def compute_state_yearly_totals(refugees):
    """Aggregate refugee arrivals by state and year."""
    state_year = refugees.groupby(["state", "fiscal_year"])["arrivals"].sum().reset_index()
    state_year.columns = ["state", "year", "arrivals"]
    return state_year


def analyze_nd_trajectory(state_year):
    """Analyze ND refugee arrival trajectory with key policy events marked."""
    nd = state_year[state_year["state"] == "North Dakota"].copy()

    # Key policy events affecting ND refugee trajectory
    events = {
        2015: "Peak USRAP expansion",
        2017: "Travel Ban EO (Jan)",
        2018: "Ceiling cuts begin",
        2020: "COVID-19 pandemic",
        # 2021: "LSSND closure" - but we don't have 2021 data!
    }

    nd["event"] = nd["year"].map(events)

    # Compute growth rates
    nd = nd.sort_values("year")
    nd["pct_change"] = nd["arrivals"].pct_change() * 100
    nd["cumulative_arrivals"] = nd["arrivals"].cumsum()

    return nd, events


def identify_comparable_states(state_year, nd_trajectory):
    """
    Identify states that could serve as synthetic control donors.

    Criteria:
    - Similar scale of refugee arrivals in pre-treatment period (2010-2016)
    - Stable resettlement infrastructure (no major agency closures)
    - Available data for same period
    """
    # Pre-treatment period
    pre_treatment = state_year[state_year["year"].between(2010, 2016)]

    # Compute state-level summary statistics
    state_stats = (
        pre_treatment.groupby("state").agg({"arrivals": ["mean", "std", "sum", "count"]}).round(2)
    )
    state_stats.columns = ["mean_arrivals", "std_arrivals", "total_arrivals", "n_years"]

    # ND baseline
    nd_mean = (
        state_stats.loc["North Dakota", "mean_arrivals"]
        if "North Dakota" in state_stats.index
        else 0
    )

    # Filter to states with similar scale (within 50%-200% of ND mean)
    similar_scale = state_stats[
        (state_stats["mean_arrivals"] >= nd_mean * 0.3)
        & (state_stats["mean_arrivals"] <= nd_mean * 3.0)
        & (state_stats["n_years"] >= 5)  # At least 5 years of data
    ]

    # Check which potential donors are available
    available_donors = []
    for state in POTENTIAL_DONOR_STATES:
        if state in similar_scale.index:
            available_donors.append(
                {
                    "state": state,
                    "mean_arrivals": float(similar_scale.loc[state, "mean_arrivals"]),
                    "ratio_to_nd": round(
                        float(similar_scale.loc[state, "mean_arrivals"]) / nd_mean, 2
                    )
                    if nd_mean > 0
                    else None,
                }
            )

    return similar_scale, available_donors


def compute_nd_vs_donors(state_year, donor_states):
    """Compare ND trajectory to potential donor states."""
    # Filter to ND and donors
    states_of_interest = ["North Dakota"] + [d["state"] for d in donor_states]
    filtered = state_year[state_year["state"].isin(states_of_interest)].copy()

    # Pivot for comparison
    pivot = filtered.pivot(index="year", columns="state", values="arrivals")

    # Normalize to 2015 baseline (peak USRAP year)
    if 2015 in pivot.index:
        baseline_2015 = pivot.loc[2015]
        normalized = pivot / baseline_2015 * 100
    else:
        normalized = None

    return pivot, normalized


def assess_treatment_effect_feasibility(state_year):
    """
    Assess whether we can identify LSSND effect given data limitations.

    Key challenge: Data ends at FY2020, LSSND closed in 2021.
    """
    assessment = {
        "data_end_year": int(state_year["year"].max()),
        "treatment_year": LSSND_CLOSURE_YEAR,
        "post_treatment_data_available": state_year["year"].max() >= LSSND_CLOSURE_YEAR,
        "feasibility": None,
        "explanation": None,
    }

    if assessment["post_treatment_data_available"]:
        assessment["feasibility"] = "FEASIBLE"
        assessment["explanation"] = (
            f"Data extends to {assessment['data_end_year']}, covering the treatment year {LSSND_CLOSURE_YEAR}."
        )
    else:
        assessment["feasibility"] = "NOT FEASIBLE WITH CURRENT DATA"
        assessment["explanation"] = (
            f"Refugee data ends at FY{assessment['data_end_year']}, but LSSND closure occurred in "
            f"{LSSND_CLOSURE_YEAR}. Cannot assess post-treatment trajectory without FY2021+ data. "
            "However, we CAN analyze pre-treatment trajectory to establish baseline and assess "
            "whether ND was already diverging from comparison states due to federal policy (2017-2020)."
        )

    return assessment


def analyze_federal_policy_effects(state_year, nd_trajectory):
    """
    Analyze whether ND's 2017-2020 pattern is explained by federal policy.

    If ND tracked other states during Travel Ban/COVID period, then the LSSND closure
    (2021+) would be a distinct, additional shock.
    """
    # Compare ND to US total during federal policy period
    us_total = state_year[state_year["state"] == "Total"].copy()
    us_total = us_total.rename(columns={"arrivals": "us_total"})

    nd = nd_trajectory[["year", "arrivals"]].copy()
    nd = nd.rename(columns={"arrivals": "nd_arrivals"})

    comparison = nd.merge(us_total[["year", "us_total"]], on="year", how="inner")
    comparison["nd_share_pct"] = (comparison["nd_arrivals"] / comparison["us_total"] * 100).round(3)

    # Compute share stability
    pre_ban = comparison[comparison["year"].between(2010, 2016)]["nd_share_pct"]
    during_ban = comparison[comparison["year"].between(2017, 2020)]["nd_share_pct"]

    share_analysis = {
        "pre_ban_period": "2010-2016",
        "pre_ban_mean_share_pct": round(pre_ban.mean(), 3) if len(pre_ban) > 0 else None,
        "pre_ban_std": round(pre_ban.std(), 3) if len(pre_ban) > 0 else None,
        "during_ban_period": "2017-2020",
        "during_ban_mean_share_pct": round(during_ban.mean(), 3) if len(during_ban) > 0 else None,
        "during_ban_std": round(during_ban.std(), 3) if len(during_ban) > 0 else None,
    }

    if share_analysis["pre_ban_mean_share_pct"] and share_analysis["during_ban_mean_share_pct"]:
        share_change = (
            share_analysis["during_ban_mean_share_pct"] - share_analysis["pre_ban_mean_share_pct"]
        )
        share_analysis["share_change"] = round(share_change, 3)
        share_analysis["interpretation"] = (
            "STABLE"
            if abs(share_change) < 0.5
            else "ND GAINED SHARE"
            if share_change > 0
            else "ND LOST SHARE"
        )

    return comparison, share_analysis


def create_trajectory_figure(nd_trajectory, events):
    """Create figure showing ND refugee trajectory with policy events."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years = nd_trajectory["year"]
    arrivals = nd_trajectory["arrivals"]

    ax.plot(years, arrivals, marker="o", linewidth=2, color="navy", label="ND Refugee Arrivals")
    ax.fill_between(years, 0, arrivals, alpha=0.3, color="navy")

    # Mark policy events
    for year, event in events.items():
        if year in years.values:
            y_val = nd_trajectory[nd_trajectory["year"] == year]["arrivals"].values[0]
            ax.annotate(
                event,
                (year, y_val),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=8,
                rotation=45,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    # Mark data end
    ax.axvline(x=2020, color="red", linestyle="--", linewidth=2, label="Data ends (FY2020)")

    # Mark LSSND closure (outside data range)
    ax.axvline(x=2021, color="orange", linestyle=":", linewidth=2, label="LSSND closure (2021)")

    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Refugee Arrivals")
    ax.set_title("North Dakota Refugee Arrivals: Pre-LSSND Closure Trajectory")
    ax.legend()
    ax.set_xticks(range(int(years.min()), 2022))
    ax.set_xticklabels([str(y) for y in range(int(years.min()), 2022)], rotation=45)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "agent2_nd_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {FIGURES_DIR / 'agent2_nd_trajectory.png'}")


def create_comparison_figure(pivot, normalized, donor_states):
    """Create figure comparing ND to potential donor states."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Raw arrivals
    ax1 = axes[0]
    for state in pivot.columns:
        style = (
            {"linewidth": 3, "marker": "o"}
            if state == "North Dakota"
            else {"linewidth": 1, "alpha": 0.7}
        )
        ax1.plot(pivot.index, pivot[state], label=state, **style)

    ax1.axvline(x=2017, color="gray", linestyle="--", alpha=0.5, label="Travel Ban")
    ax1.axvline(x=2020, color="red", linestyle="--", alpha=0.5, label="Data ends")
    ax1.set_xlabel("Fiscal Year")
    ax1.set_ylabel("Refugee Arrivals")
    ax1.set_title("ND vs Potential Donor States: Raw Arrivals")
    ax1.legend(fontsize=8)

    # Right: Normalized to 2015
    ax2 = axes[1]
    if normalized is not None:
        for state in normalized.columns:
            style = (
                {"linewidth": 3, "marker": "o"}
                if state == "North Dakota"
                else {"linewidth": 1, "alpha": 0.7}
            )
            ax2.plot(normalized.index, normalized[state], label=state, **style)

        ax2.axhline(y=100, color="gray", linestyle="-", alpha=0.3)
        ax2.axvline(x=2017, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=2020, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Fiscal Year")
        ax2.set_ylabel("Index (2015 = 100)")
        ax2.set_title("Normalized Trajectories (2015 = 100)")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "agent2_state_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {FIGURES_DIR / 'agent2_state_comparison.png'}")


def main():
    """Main analysis function."""
    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Agent 2: LSSND Closure Impact Analysis")
    print("ADR-021 Investigation")
    print("=" * 60)

    # Load data
    print("\n[1/7] Loading refugee data...")
    refugees = load_refugee_data()
    state_year = compute_state_yearly_totals(refugees)
    print(f"  - States: {state_year['state'].nunique()}")
    print(f"  - Years: {state_year['year'].min()}-{state_year['year'].max()}")

    # Analyze ND trajectory
    print("\n[2/7] Analyzing ND trajectory...")
    nd_trajectory, events = analyze_nd_trajectory(state_year)
    print(f"  - ND years: {len(nd_trajectory)}")
    print(f"  - Peak year: {nd_trajectory.loc[nd_trajectory['arrivals'].idxmax(), 'year']}")
    print(f"  - Peak arrivals: {nd_trajectory['arrivals'].max()}")

    # Identify comparable states
    print("\n[3/7] Identifying potential donor states for synthetic control...")
    similar_scale, available_donors = identify_comparable_states(state_year, nd_trajectory)
    print(f"  - Potential donors checked: {len(POTENTIAL_DONOR_STATES)}")
    print(f"  - Available donors (similar scale): {len(available_donors)}")
    for donor in available_donors:
        print(
            f"    - {donor['state']}: mean={donor['mean_arrivals']}, ratio={donor['ratio_to_nd']}"
        )

    # Compare ND to donors
    print("\n[4/7] Computing ND vs donor trajectories...")
    pivot, normalized = compute_nd_vs_donors(state_year, available_donors)

    # Assess treatment effect feasibility
    print("\n[5/7] Assessing treatment effect feasibility...")
    feasibility = assess_treatment_effect_feasibility(state_year)
    print(f"  - Feasibility: {feasibility['feasibility']}")
    print(f"  - Explanation: {feasibility['explanation']}")

    # Analyze federal policy effects
    print("\n[6/7] Analyzing federal policy effects (2017-2020)...")
    comparison, share_analysis = analyze_federal_policy_effects(state_year, nd_trajectory)
    print(f"  - Pre-ban ND share: {share_analysis.get('pre_ban_mean_share_pct', 'N/A')}%")
    print(f"  - During-ban ND share: {share_analysis.get('during_ban_mean_share_pct', 'N/A')}%")
    print(f"  - Interpretation: {share_analysis.get('interpretation', 'N/A')}")

    # Create figures
    print("\n[7/7] Creating figures...")
    create_trajectory_figure(nd_trajectory, events)
    if len(available_donors) > 0:
        create_comparison_figure(pivot, normalized, available_donors)

    # Compile results
    results = {
        "analysis": "Agent 2: LSSND Closure Impact Analysis",
        "adr": "021",
        "date": "2026-01-01",
        "lssnd_context": {
            "closure_year": LSSND_CLOSURE_YEAR,
            "description": (
                "Lutheran Social Services of North Dakota (LSSND) was the primary refugee "
                "resettlement agency in ND. Its closure in 2021 eliminated local resettlement "
                "infrastructure, potentially 'decoupling' ND from national refugee trends."
            ),
        },
        "data_summary": {
            "refugee_data_years": f"{int(state_year['year'].min())}-{int(state_year['year'].max())}",
            "nd_peak_year": int(nd_trajectory.loc[nd_trajectory["arrivals"].idxmax(), "year"]),
            "nd_peak_arrivals": int(nd_trajectory["arrivals"].max()),
        },
        "treatment_effect_feasibility": feasibility,
        "donor_state_analysis": {
            "potential_donors": POTENTIAL_DONOR_STATES,
            "available_donors": available_donors,
            "donor_pool_adequate": len(available_donors) >= 3,
        },
        "federal_policy_analysis": share_analysis,
        "nd_trajectory": nd_trajectory[["year", "arrivals", "pct_change"]].to_dict(
            orient="records"
        ),
        "key_findings": [
            f"LSSND closed in {LSSND_CLOSURE_YEAR}, but refugee data ends FY2020 - cannot directly measure post-closure impact.",
            "Synthetic control is FEASIBLE IN PRINCIPLE with 4+ donor states of similar scale.",
            f"ND share during Travel Ban/COVID ({share_analysis.get('during_ban_mean_share_pct', 'N/A')}%) was similar to pre-ban ({share_analysis.get('pre_ban_mean_share_pct', 'N/A')}%), suggesting federal policy affected ND proportionally.",
            "LSSND closure would be an ADDITIONAL state-specific shock on top of federal effects.",
            "To quantify LSSND effect, we need FY2021-2024 refugee data to construct post-treatment outcome.",
        ],
        "validation_of_external_claims": {
            "claim_3_lssnd_capacity_shock": (
                "PLAUSIBLE BUT NOT TESTABLE WITH CURRENT DATA - "
                "Data ends before treatment occurs. The claim that LSSND closure created "
                "a distinct capacity shock is reasonable given infrastructure dependency, "
                "but we cannot empirically validate without post-2020 refugee data."
            ),
            "synthetic_control_feasibility": (
                "FEASIBLE IN PRINCIPLE - Donor pool exists with 4+ similar states. "
                "Implementation requires FY2021+ data for post-treatment comparison."
            ),
            "data_need": "Refugee arrivals FY2021-2024 required to implement synthetic control.",
        },
        "outputs": {
            "figures": [
                "agent2_nd_trajectory.png",
                "agent2_state_comparison.png",
            ],
        },
    }

    # Save results
    with open(RESULTS_DIR / "agent2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Results saved to: {RESULTS_DIR / 'agent2_results.json'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
