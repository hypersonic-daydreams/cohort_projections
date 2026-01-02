#!/usr/bin/env python3
"""
Agent 3: Existing Module Gap Analysis
ADR-021 Investigation

Analyzes how existing Modules 8-9 handle temporal variation and regime breaks,
and identifies gaps relative to external AI recommendations.

Key Questions:
1. How do existing modules handle temporal variation?
2. Does the current scenario framework encode regime-like structures?
3. What gaps remain after accounting for existing functionality?

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

# === Configuration ===
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Paths to existing module results
SDC_RESULTS = Path(
    "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/scripts/statistical_analysis/results"
)

# Paths to module source code for analysis
MODULE_DIR = Path(
    "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/scripts/statistical_analysis"
)


def load_module_results():
    """Load existing module results to understand current functionality."""
    results = {}

    # Module 8: Duration Analysis
    m8_path = SDC_RESULTS / "module_8_duration_analysis.json"
    if m8_path.exists():
        with open(m8_path) as f:
            results["module_8"] = json.load(f)
    else:
        results["module_8"] = None
        print(f"  WARNING: Module 8 results not found at {m8_path}")

    # Module 8: Hazard Model
    m8_hazard_path = SDC_RESULTS / "module_8_hazard_model.json"
    if m8_hazard_path.exists():
        with open(m8_hazard_path) as f:
            results["module_8_hazard"] = json.load(f)
    else:
        results["module_8_hazard"] = None

    # Module 9: Scenario Modeling
    m9_path = SDC_RESULTS / "module_9_scenario_modeling.json"
    if m9_path.exists():
        with open(m9_path) as f:
            results["module_9"] = json.load(f)
    else:
        results["module_9"] = None
        print(f"  WARNING: Module 9 results not found at {m9_path}")

    return results


def analyze_module_8_capabilities(m8_results, m8_hazard):
    """Analyze what Module 8 currently does for duration/wave analysis."""
    capabilities = {
        "module": "Module 8: Duration Analysis",
        "status": "EXISTS" if m8_results else "NOT FOUND",
        "current_functionality": [],
        "relevant_to_external_claims": [],
        "gaps": [],
    }

    if not m8_results:
        capabilities["gaps"].append("Module 8 results not available for analysis")
        return capabilities

    # Extract current functionality from results
    if "analysis_name" in m8_results:
        capabilities["current_functionality"].append(f"Analysis: {m8_results['analysis_name']}")

    if "parameters" in m8_results:
        params = m8_results["parameters"]
        capabilities["current_functionality"].extend(
            [
                f"Wave threshold: {params.get('wave_threshold_pct', 'N/A')}% above baseline",
                f"Minimum wave duration: {params.get('min_wave_duration', 'N/A')} years",
                f"Data source: {params.get('data_source', 'N/A')}",
            ]
        )

    if "results" in m8_results:
        res = m8_results["results"]
        if "wave_summary" in res:
            wave_sum = res["wave_summary"]
            capabilities["current_functionality"].extend(
                [
                    f"Waves identified: {wave_sum.get('total_waves', 'N/A')}",
                    f"Median duration: {wave_sum.get('median_duration', 'N/A')} years",
                    f"Ongoing waves: {wave_sum.get('ongoing_waves', 'N/A')}",
                ]
            )

    # Hazard model capabilities
    if m8_hazard:
        capabilities["current_functionality"].extend(
            [
                "Cox Proportional Hazards model fitted",
                f"Covariates: {', '.join(m8_hazard.get('covariates', []))}",
                f"Concordance: {m8_hazard.get('concordance', 'N/A')}",
            ]
        )

    # Assess relevance to external claims
    capabilities["relevant_to_external_claims"] = [
        {
            "claim": "Status durability / retention layer",
            "module_8_relevance": "PARTIAL - Module 8 models wave duration but not LEGAL STATUS durability",
            "gap": "Need status-specific hazard rates (refugee vs parole)",
        },
        {
            "claim": "Parole 'cliff' at years 2-4",
            "module_8_relevance": "PARTIAL - Hazard model exists but trained on REFUGEE waves only",
            "gap": "Need separate hazard curves for parole cohorts",
        },
    ]

    # Identify gaps
    capabilities["gaps"] = [
        "Wave detection is nationality-based, not legal-status-based",
        "Hazard model trained on refugee data only (no parole cohorts)",
        "No explicit 'status precarity' or 'legal cliff' modeling",
        "Survival analysis assumes all arrivals have similar retention probability",
    ]

    return capabilities


def analyze_module_9_capabilities(m9_results):
    """Analyze what Module 9 currently does for scenario modeling."""
    capabilities = {
        "module": "Module 9: Scenario Modeling",
        "status": "EXISTS" if m9_results else "NOT FOUND",
        "current_functionality": [],
        "scenario_framework": {},
        "relevant_to_external_claims": [],
        "gaps": [],
    }

    if not m9_results:
        capabilities["gaps"].append("Module 9 results not available for analysis")
        return capabilities

    # Extract current functionality
    if "parameters" in m9_results:
        params = m9_results["parameters"]
        capabilities["current_functionality"].extend(
            [
                f"Base year: {params.get('base_year', 'N/A')}",
                f"Projection horizon: {params.get('projection_horizon', 'N/A')}",
                f"Monte Carlo draws: {params.get('monte_carlo_draws', 'N/A')}",
                f"Scenarios defined: {', '.join(params.get('scenarios', []))}",
            ]
        )

        capabilities["scenario_framework"] = {
            "scenarios": params.get("scenarios", []),
            "description": "4 scenarios covering policy spectrum from zero to high growth",
        }

    # Extract scenario definitions if available
    if "decisions" in m9_results:
        for decision in m9_results["decisions"]:
            if decision.get("category") == "scenario_design":
                capabilities["scenario_framework"]["design_rationale"] = decision.get(
                    "rationale", ""
                )

    # Assess relevance to external claims
    capabilities["relevant_to_external_claims"] = [
        {
            "claim": "Policy regime framework",
            "module_9_relevance": "PARTIAL - Scenarios exist but not defined by policy REGIMES",
            "current_approach": "Scenarios use ARIMA + growth rate assumptions",
            "gap": "Scenarios not tied to policy lever mechanisms",
        },
        {
            "claim": "Mechanism-based scenarios",
            "module_9_relevance": "LOW - CBO Full uses 8% compound growth, not policy mechanisms",
            "current_approach": "Growth rates are somewhat arbitrary",
            "gap": "Need scenarios defined by ceilings, parole continuation, capacity",
        },
        {
            "claim": "Wave integration",
            "module_9_relevance": "HIGH - Module 9 DOES integrate Module 8 wave survival",
            "current_approach": "Wave registry with conditional duration prediction",
            "gap": "Wave integration is refugee-centric, not status-aware",
        },
    ]

    # Identify gaps
    capabilities["gaps"] = [
        "Scenarios defined by growth rates, not policy mechanisms",
        "'CBO Full' at 8% compound is not tied to specific policy assumptions",
        "No 'parole cliff' scenario that models attrition at years 2-4",
        "No capacity parameter (LSSND-like) in scenario framework",
        "Wave integration treats all waves uniformly (no status differentiation)",
    ]

    return capabilities


def compare_to_external_recommendations():
    """Map external recommendations to current module capabilities."""
    recommendations = [
        {
            "id": 1,
            "recommendation": "Two-component estimand (durable vs temporary)",
            "current_coverage": "NOT ADDRESSED",
            "module_changes_needed": [
                "New data pipeline to classify arrivals by status durability",
                "Modify Module 8 to track status-specific waves",
                "Modify Module 9 to project Y_t^dur and Y_t^temp separately",
            ],
            "effort": "HIGH",
        },
        {
            "id": 2,
            "recommendation": "Status-transition hazard model (Module 8b)",
            "current_coverage": "PARTIALLY ADDRESSED by Module 8",
            "module_changes_needed": [
                "Extend Module 8 with status-specific hazard rates",
                "Add parole-specific attrition curve (years 2-4 cliff)",
                "Create 'regularization event' probability parameter",
            ],
            "effort": "MEDIUM",
        },
        {
            "id": 3,
            "recommendation": "Synthetic control for LSSND",
            "current_coverage": "NOT ADDRESSED",
            "module_changes_needed": [
                "New analysis module or extend Module 7 (Causal)",
                "Construct donor pool from state-level refugee data",
                "Estimate capacity parameter from counterfactual",
            ],
            "effort": "MEDIUM-HIGH",
        },
        {
            "id": 4,
            "recommendation": "Policy regime variable R_t",
            "current_coverage": "IMPLICITLY in Module 2.1 (ARIMA structural breaks)",
            "module_changes_needed": [
                "Formalize regime definition (expansion/restriction/volatility)",
                "Create regime indicator for use across modules",
                "Add regime-conditional parameters to scenario framework",
            ],
            "effort": "LOW-MEDIUM",
        },
        {
            "id": 5,
            "recommendation": "Update data through FY2024",
            "current_coverage": "PARTIALLY - ND migration to 2024, refugees only to 2020",
            "module_changes_needed": [
                "Acquire updated refugee data (FY2021-2024)",
                "Create parole proxy dataset",
                "Re-run Modules 8-9 with extended data",
            ],
            "effort": "MEDIUM (data acquisition)",
        },
        {
            "id": 6,
            "recommendation": "Mechanism-based scenario framework",
            "current_coverage": "NOT ADDRESSED - scenarios use growth rates",
            "module_changes_needed": [
                "Redesign Module 9 scenario definitions",
                "Map scenarios to policy levers (ceilings, parole, capacity)",
                "Create 'parole cliff' and 'capacity constraint' scenarios",
            ],
            "effort": "MEDIUM-HIGH",
        },
        {
            "id": 7,
            "recommendation": "Secondary migration module",
            "current_coverage": "NOT ADDRESSED",
            "module_changes_needed": [
                "New module using ACS state-to-state migration",
                "Decompose FB growth into direct arrival vs domestic redistribution",
                "Sensitivity analysis on secondary migration share",
            ],
            "effort": "MEDIUM",
        },
        {
            "id": 8,
            "recommendation": "Policy timeline table and conceptual diagram",
            "current_coverage": "PARTIALLY in journal article text",
            "module_changes_needed": [
                "Create structured policy event table",
                "Add conceptual model diagram to Methods section",
                "Map policy events to model variables explicitly",
            ],
            "effort": "LOW",
        },
    ]

    return recommendations


def compute_gap_summary(m8_caps, m9_caps, recommendations):
    """Compute overall gap summary."""
    summary = {
        "total_recommendations": len(recommendations),
        "fully_addressed": 0,
        "partially_addressed": 0,
        "not_addressed": 0,
        "high_priority_gaps": [],
        "medium_priority_gaps": [],
        "low_priority_gaps": [],
    }

    for rec in recommendations:
        if rec["current_coverage"].startswith("NOT"):
            summary["not_addressed"] += 1
            if rec["effort"] == "HIGH" or rec["id"] in [1, 2, 6]:
                summary["high_priority_gaps"].append(rec["recommendation"])
            elif rec["effort"].startswith("MEDIUM"):
                summary["medium_priority_gaps"].append(rec["recommendation"])
            else:
                summary["low_priority_gaps"].append(rec["recommendation"])
        elif rec["current_coverage"].startswith("PARTIAL"):
            summary["partially_addressed"] += 1
            summary["medium_priority_gaps"].append(rec["recommendation"])
        else:
            summary["fully_addressed"] += 1

    # Add Module 8 and 9 specific gaps
    summary["module_8_gaps"] = m8_caps["gaps"]
    summary["module_9_gaps"] = m9_caps["gaps"]

    return summary


def main():
    """Main analysis function."""
    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Agent 3: Existing Module Gap Analysis")
    print("ADR-021 Investigation")
    print("=" * 60)

    # Load existing module results
    print("\n[1/5] Loading existing module results...")
    module_results = load_module_results()

    # Analyze Module 8
    print("\n[2/5] Analyzing Module 8 (Duration Analysis)...")
    m8_caps = analyze_module_8_capabilities(
        module_results.get("module_8"), module_results.get("module_8_hazard")
    )
    print(f"  - Status: {m8_caps['status']}")
    print(f"  - Functionality items: {len(m8_caps['current_functionality'])}")
    print(f"  - Gaps identified: {len(m8_caps['gaps'])}")

    # Analyze Module 9
    print("\n[3/5] Analyzing Module 9 (Scenario Modeling)...")
    m9_caps = analyze_module_9_capabilities(module_results.get("module_9"))
    print(f"  - Status: {m9_caps['status']}")
    print(f"  - Functionality items: {len(m9_caps['current_functionality'])}")
    print(f"  - Gaps identified: {len(m9_caps['gaps'])}")

    # Compare to external recommendations
    print("\n[4/5] Mapping external recommendations to current capabilities...")
    recommendations = compare_to_external_recommendations()

    # Compute gap summary
    print("\n[5/5] Computing gap summary...")
    gap_summary = compute_gap_summary(m8_caps, m9_caps, recommendations)
    print("\n  Gap Summary:")
    print(
        f"  - Fully addressed: {gap_summary['fully_addressed']}/{gap_summary['total_recommendations']}"
    )
    print(
        f"  - Partially addressed: {gap_summary['partially_addressed']}/{gap_summary['total_recommendations']}"
    )
    print(
        f"  - Not addressed: {gap_summary['not_addressed']}/{gap_summary['total_recommendations']}"
    )
    print("\n  High priority gaps:")
    for gap in gap_summary["high_priority_gaps"]:
        print(f"    - {gap}")

    # Compile results
    results = {
        "analysis": "Agent 3: Existing Module Gap Analysis",
        "adr": "021",
        "date": "2026-01-01",
        "module_8_analysis": m8_caps,
        "module_9_analysis": m9_caps,
        "recommendation_mapping": recommendations,
        "gap_summary": gap_summary,
        "key_findings": [
            "Module 8 (Duration) exists with wave detection and Cox PH model, but is REFUGEE-ONLY.",
            "Module 9 (Scenarios) integrates wave survival but scenarios are GROWTH-RATE-BASED, not MECHANISM-BASED.",
            "The existing framework DOES have survival/hazard machinery that could be extended for status durability.",
            "Major gaps: (1) No status differentiation, (2) No capacity parameter, (3) No mechanism-based scenarios.",
            "The 'parole cliff' concept could be implemented as an extension to Module 8 hazard model.",
            "Policy regime variable could build on Module 2.1 structural break detection.",
        ],
        "implementation_strategy": {
            "build_on_existing": [
                "Extend Module 8 hazard model with status-specific rates",
                "Add regime variable using Module 2.1 break detection",
                "Modify Module 9 scenario definitions for policy mechanisms",
            ],
            "new_modules_needed": [
                "Secondary migration analysis (ACS data)",
                "LSSND synthetic control (extend Module 7)",
            ],
            "data_acquisition_required": [
                "Refugee arrivals FY2021-2024",
                "Parole proxy dataset (OAW, U4U)",
            ],
        },
        "validation_of_external_claims": {
            "claim_methodology_gaps": (
                "PARTIALLY VALIDATED - Existing modules have relevant machinery "
                "(wave detection, hazard models, scenario framework) but lack STATUS AWARENESS "
                "and MECHANISM-BASED scenario definitions as claimed."
            ),
            "existing_strengths": (
                "Module 8-9 integration is more sophisticated than external analysis implies. "
                "The wave registry and conditional duration prediction are directly relevant "
                "to the 'status durability' concept - they just need status differentiation."
            ),
        },
    }

    # Save results
    with open(RESULTS_DIR / "agent3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Results saved to: {RESULTS_DIR / 'agent3_results.json'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
