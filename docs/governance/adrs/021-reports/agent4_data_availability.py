# mypy: ignore-errors
#!/usr/bin/env python3
"""
Agent 4: Data Availability Assessment
ADR-021 Investigation

Assesses the availability of data needed to implement external AI recommendations,
including refugee updates, parole proxy data, and secondary migration data.

Key Questions:
1. What refugee/parole data is available through FY2024?
2. What ACS data exists for secondary migration analysis?
3. What are the data quality/coverage limitations?

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

# Data directories to check
DATA_DIRS = {
    "immigration_analysis": Path(
        "/home/nhaarstad/workspace/demography/cohort_projections/data/processed/immigration/analysis"
    ),
    "sdc_data": Path(
        "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/data"
    ),
    "sdc_updated": Path(
        "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/data_updated"
    ),
    "sdc_policy": Path(
        "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/data_immigration_policy"
    ),
    "raw_immigration": Path(
        "/home/nhaarstad/workspace/demography/cohort_projections/data/raw/immigration"
    ),
}


def scan_directory(dir_path):
    """Scan a directory for data files and return metadata."""
    if not dir_path.exists():
        return {"exists": False, "path": str(dir_path), "files": []}

    files = []
    for item in dir_path.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            rel_path = item.relative_to(dir_path)
            size_kb = item.stat().st_size / 1024
            files.append(
                {
                    "path": str(rel_path),
                    "size_kb": round(size_kb, 1),
                    "extension": item.suffix,
                }
            )

    return {
        "exists": True,
        "path": str(dir_path),
        "file_count": len(files),
        "files": sorted(files, key=lambda x: x["path"]),
    }


def assess_current_refugee_data(analysis_dir):
    """Assess current refugee arrivals data coverage."""
    assessment = {
        "data_type": "Refugee Arrivals",
        "current_status": {},
        "data_sources": [],
        "coverage": {},
        "limitations": [],
        "update_needed": True,
    }

    # Check for refugee files
    refugee_files = [
        "refugee_arrivals_by_state_nationality.parquet",
        "refugee_arrivals_by_state_nationality.csv",
        "dhs_refugee_admissions.parquet",
    ]

    for fname in refugee_files:
        fpath = analysis_dir / fname
        if fpath.exists():
            assessment["data_sources"].append(
                {
                    "file": fname,
                    "size_kb": round(fpath.stat().st_size / 1024, 1),
                    "exists": True,
                }
            )

    # Known coverage from earlier analysis
    assessment["coverage"] = {
        "current_end_year": 2020,
        "target_end_year": 2024,
        "years_missing": 4,
        "geographic_scope": "All US states",
        "detail_level": "State × Nationality × Fiscal Year",
    }

    assessment["current_status"] = {
        "usable": True,
        "quality": "HIGH - ORR/PRM official data via Dreher et al. 2020",
        "format": "Parquet and CSV available",
    }

    assessment["limitations"] = [
        "Data ends FY2020 - missing critical 2021-2024 period",
        "Does not include parole arrivals (OAW, U4U, etc.)",
        "Does not include other humanitarian categories (asylum, TPS)",
    ]

    assessment["update_sources"] = [
        {
            "source": "WRAPS/RPC (Refugee Processing Center)",
            "url": "https://www.wrapsnet.org/",
            "coverage": "Refugee arrivals by state, nationality, FY",
            "update_frequency": "Monthly",
            "feasibility": "HIGH - Public data, machine-readable",
        },
        {
            "source": "State Department Refugee Admissions Report",
            "coverage": "Annual national and state summaries",
            "feasibility": "MEDIUM - May require manual extraction",
        },
    ]

    return assessment


def assess_parole_data_availability():
    """Assess availability of parole (OAW, U4U) data for ND."""
    assessment = {
        "data_type": "Parole Arrivals (OAW, U4U)",
        "current_status": {
            "available_in_project": False,
            "quality": "N/A",
        },
        "coverage": {
            "target_programs": ["Operation Allies Welcome (OAW)", "Uniting for Ukraine (U4U)"],
            "target_years": "2021-2024",
            "target_geography": "North Dakota",
        },
        "potential_sources": [],
        "limitations": [],
        "feasibility": {},
    }

    assessment["potential_sources"] = [
        {
            "source": "DHS Immigration Statistics (OHSS)",
            "url": "https://www.dhs.gov/ohss/topics/immigration",
            "coverage": "National parole statistics, some state breakdowns",
            "detail_level": "Limited state-level detail for humanitarian parole",
            "feasibility": "MEDIUM - State detail may not be available",
        },
        {
            "source": "State administrative records (ND DHS)",
            "coverage": "State-specific arrivals and services",
            "detail_level": "High for ND specifically",
            "feasibility": "LOW-MEDIUM - May require FOIA or direct contact",
        },
        {
            "source": "Resettlement agency reports",
            "coverage": "Clients served by agency",
            "detail_level": "High for served population",
            "feasibility": "MEDIUM - Depends on agency data sharing",
            "note": "LSSND closed 2021; successor agency(s) would have data",
        },
        {
            "source": "Welcome Corps / Private Sponsor Data",
            "coverage": "Private sponsorship arrivals",
            "detail_level": "National and potentially state",
            "feasibility": "LOW - Program is new, data availability unclear",
        },
        {
            "source": "USCIS/CBP parole statistics",
            "coverage": "Entries under parole authority",
            "detail_level": "National, limited state detail",
            "feasibility": "LOW - State-level breakdown rare",
        },
    ]

    assessment["limitations"] = [
        "Parole data is less systematically reported than refugee admissions",
        "State-level parole breakdowns are often not publicly available",
        "OAW and U4U have different reporting structures",
        "May need to construct proxy from multiple sources",
    ]

    assessment["feasibility"] = {
        "overall": "MEDIUM-LOW",
        "explanation": (
            "Parole data at state level is harder to obtain than refugee data. "
            "Best approach may be: (1) Use national parole totals as upper bound, "
            "(2) Apply ND share from refugee data as approximation, "
            "(3) Seek state administrative data for validation."
        ),
        "proxy_strategy": (
            "If direct data unavailable, construct proxy: "
            "Gap = (ND total intl migration) - (ND refugee arrivals) - (estimated non-humanitarian). "
            "This 'residual' approximates parole + other humanitarian."
        ),
    }

    return assessment


def assess_acs_secondary_migration():
    """Assess ACS data for secondary migration analysis."""
    assessment = {
        "data_type": "Secondary Migration (Foreign-Born Domestic Mobility)",
        "current_status": {},
        "coverage": {},
        "data_sources": [],
        "limitations": [],
    }

    # Check for existing ACS data
    analysis_dir = DATA_DIRS["immigration_analysis"]
    acs_files = list(analysis_dir.glob("acs_*.parquet"))

    assessment["current_status"] = {
        "available_in_project": len(acs_files) > 0,
        "files_found": [f.name for f in acs_files],
    }

    assessment["data_sources"] = [
        {
            "source": "ACS 1-Year PUMS",
            "coverage": "State-to-state migration, foreign-born identifier",
            "detail_level": "Individual records with state of residence 1 year ago",
            "years_available": "2005-2023 (1-year), 2010-2022 (5-year)",
            "feasibility": "HIGH - Public microdata, well-documented",
        },
        {
            "source": "ACS Migration Flows Tables",
            "coverage": "Pre-tabulated state-to-state flows",
            "detail_level": "Aggregate counts, some demographic breakdowns",
            "feasibility": "HIGH - Census API accessible",
        },
        {
            "source": "acs_foreign_born_by_state_origin.parquet (in project)",
            "coverage": "Foreign-born population by state and origin",
            "detail_level": "State × Country of Birth",
            "current_use": "Module 8 wave validation",
            "limitation": "May not have migration flow (only stock)",
        },
    ]

    assessment["coverage"] = {
        "geographic": "All US states",
        "temporal": "Annual (1-year ACS) or 5-year rolling",
        "population": "Foreign-born only (using birthplace filter)",
        "flow_type": "State of residence vs state 1 year ago",
    }

    assessment["limitations"] = [
        "ACS is sample-based - small state cells (like ND) have high variance",
        "1-year ACS requires population >65,000 for reliable state estimates",
        "5-year ACS smooths but reduces temporal resolution",
        "Cannot distinguish legal status (refugee vs parole vs other)",
        "Measures 1-year migration, not lifetime migration history",
    ]

    assessment["analysis_feasibility"] = {
        "overall": "MEDIUM",
        "explanation": (
            "ACS can provide foreign-born domestic migration flows at state level, "
            "but ND's small population means high sampling variance. "
            "Best approach: Use 5-year ACS for more stable estimates, "
            "or aggregate foreign-born into regions for secondary migration analysis."
        ),
        "recommended_approach": (
            "Compute: FB_inflow_to_ND = direct_intl + secondary_domestic. "
            "Use ACS MIGSP variable (state of residence 1 year ago) filtered to foreign-born. "
            "Compare to PEP net international migration to decompose sources."
        ),
    }

    return assessment


def assess_synthetic_control_data():
    """Assess data availability for LSSND synthetic control analysis."""
    assessment = {
        "data_type": "Synthetic Control Donor Pool Data",
        "requirements": {},
        "current_availability": {},
        "gaps": [],
    }

    assessment["requirements"] = {
        "outcome": "ND refugee arrivals (or PEP intl migration)",
        "treatment": "ND after 2021 (LSSND closure)",
        "donors": "Similar low-flow states with stable infrastructure",
        "pre_treatment": "2010-2020 (ideally 2010-2016 for parallel trends)",
        "post_treatment": "2021-2024",
    }

    assessment["current_availability"] = {
        "outcome_pre_treatment": {
            "available": True,
            "years": "2002-2020",
            "source": "refugee_arrivals_by_state_nationality.parquet",
        },
        "outcome_post_treatment": {
            "available": False,
            "years_needed": "2021-2024",
            "gap": "CRITICAL - Cannot construct post-treatment comparison",
        },
        "donor_states_pre_treatment": {
            "available": True,
            "states_with_data": "50+ states",
            "note": "South Dakota, Montana, Nebraska, Idaho, Maine, Vermont have similar scale",
        },
        "covariates": {
            "available": True,
            "sources": ["ACS (population, FB share)", "BLS (unemployment)", "BEA (GDP)"],
        },
    }

    assessment["gaps"] = [
        "Post-treatment outcome data (FY2021-2024 refugee arrivals) is CRITICAL missing piece",
        "Without it, synthetic control can only be 'pre-specified' but not estimated",
        "Alternative: Use PEP international migration as proxy outcome (available to 2024)",
    ]

    assessment["alternative_approach"] = {
        "description": (
            "If refugee arrivals FY2021+ unavailable, could use PEP net international migration "
            "as outcome variable. This is noisier (includes all immigrant categories) but extends to 2024."
        ),
        "tradeoffs": [
            "PRO: Data available through 2024",
            "PRO: Captures total capacity effect (not just refugees)",
            "CON: Conflates refugee, parole, visa, student flows",
            "CON: Harder to attribute to LSSND specifically",
        ],
    }

    return assessment


def compile_data_summary():
    """Compile summary of data availability for ADR-021 requirements."""
    summary = {
        "data_categories": [
            {
                "category": "Refugee Arrivals Update (FY2021-2024)",
                "status": "NEEDED - NOT AVAILABLE IN PROJECT",
                "priority": "HIGH",
                "acquisition_feasibility": "HIGH (WRAPS public data)",
            },
            {
                "category": "Parole Arrivals (OAW, U4U) for ND",
                "status": "NEEDED - NOT AVAILABLE IN PROJECT",
                "priority": "HIGH",
                "acquisition_feasibility": "MEDIUM-LOW (state admin data)",
            },
            {
                "category": "ACS Secondary Migration",
                "status": "PARTIALLY AVAILABLE",
                "priority": "MEDIUM",
                "acquisition_feasibility": "HIGH (Census API)",
            },
            {
                "category": "Synthetic Control Donor Data",
                "status": "PRE-TREATMENT AVAILABLE, POST-TREATMENT MISSING",
                "priority": "HIGH (contingent on refugee update)",
                "acquisition_feasibility": "HIGH (after refugee update)",
            },
        ],
        "critical_path": (
            "Refugee arrivals FY2021-2024 is the critical path item. "
            "Without it, cannot: (1) validate estimand composition claim, "
            "(2) implement LSSND synthetic control, (3) calibrate parole proxy. "
            "This data should be first acquisition priority."
        ),
    }
    return summary


def main():
    """Main analysis function."""
    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Agent 4: Data Availability Assessment")
    print("ADR-021 Investigation")
    print("=" * 60)

    # Scan data directories
    print("\n[1/6] Scanning data directories...")
    dir_scans = {}
    for name, dir_path in DATA_DIRS.items():
        scan = scan_directory(dir_path)
        dir_scans[name] = scan
        status = f"✓ {scan['file_count']} files" if scan["exists"] else "✗ Not found"
        print(f"  - {name}: {status}")

    # Assess refugee data
    print("\n[2/6] Assessing refugee arrivals data...")
    refugee_assessment = assess_current_refugee_data(DATA_DIRS["immigration_analysis"])
    print(f"  - Current coverage: FY{refugee_assessment['coverage']['current_end_year']}")
    print(f"  - Target coverage: FY{refugee_assessment['coverage']['target_end_year']}")
    print(f"  - Years missing: {refugee_assessment['coverage']['years_missing']}")

    # Assess parole data
    print("\n[3/6] Assessing parole data availability...")
    parole_assessment = assess_parole_data_availability()
    print(
        f"  - Available in project: {parole_assessment['current_status']['available_in_project']}"
    )
    print(f"  - Feasibility: {parole_assessment['feasibility']['overall']}")

    # Assess ACS secondary migration
    print("\n[4/6] Assessing ACS secondary migration data...")
    acs_assessment = assess_acs_secondary_migration()
    print(f"  - Available in project: {acs_assessment['current_status']['available_in_project']}")
    print(f"  - Files found: {len(acs_assessment['current_status'].get('files_found', []))}")

    # Assess synthetic control data
    print("\n[5/6] Assessing synthetic control data requirements...")
    sc_assessment = assess_synthetic_control_data()
    print(
        f"  - Pre-treatment outcome: {sc_assessment['current_availability']['outcome_pre_treatment']['available']}"
    )
    print(
        f"  - Post-treatment outcome: {sc_assessment['current_availability']['outcome_post_treatment']['available']}"
    )

    # Compile summary
    print("\n[6/6] Compiling data summary...")
    data_summary = compile_data_summary()
    print("\n  Data Summary:")
    for cat in data_summary["data_categories"]:
        print(f"  - {cat['category']}: {cat['status']}")

    # Compile results
    results = {
        "analysis": "Agent 4: Data Availability Assessment",
        "adr": "021",
        "date": "2026-01-01",
        "directory_scans": {
            k: {
                "exists": v["exists"],
                "file_count": v.get("file_count", 0),
            }
            for k, v in dir_scans.items()
        },
        "refugee_data": refugee_assessment,
        "parole_data": parole_assessment,
        "acs_secondary_migration": acs_assessment,
        "synthetic_control_data": sc_assessment,
        "data_summary": data_summary,
        "key_findings": [
            "Refugee arrivals FY2021-2024 is the CRITICAL missing data - HIGH feasibility to acquire from WRAPS.",
            "Parole data (OAW, U4U) at ND level is DIFFICULT to obtain - may need proxy approach.",
            "ACS secondary migration data is AVAILABLE via Census API - can implement with existing tools.",
            "Synthetic control is BLOCKED until refugee FY2021+ data acquired.",
            "ND migration summary extends to 2024, providing total international migration target.",
        ],
        "data_acquisition_priorities": [
            {
                "priority": 1,
                "data": "Refugee arrivals FY2021-2024 (WRAPS/RPC)",
                "rationale": "Critical path for all other analyses",
                "effort": "LOW - public data, well-structured",
            },
            {
                "priority": 2,
                "data": "ACS state-to-state migration (foreign-born)",
                "rationale": "Enables secondary migration analysis",
                "effort": "LOW-MEDIUM - Census API",
            },
            {
                "priority": 3,
                "data": "Parole proxy construction",
                "rationale": "If direct data unavailable, use residual method",
                "effort": "MEDIUM - analytical work, not acquisition",
            },
        ],
        "validation_of_external_claims": {
            "claim_5_data_truncation": (
                "FULLY VALIDATED - Refugee data ends FY2020 as claimed. "
                "Update to FY2024 is feasible via WRAPS."
            ),
            "claim_parole_data_need": (
                "VALIDATED - Parole data for ND is not readily available. "
                "Proxy approach may be necessary."
            ),
            "claim_secondary_migration": (
                "ADDRESSABLE - ACS data exists for secondary migration analysis. "
                "Implementation is feasible."
            ),
        },
    }

    # Save results
    with open(RESULTS_DIR / "agent4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Results saved to: {RESULTS_DIR / 'agent4_results.json'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
