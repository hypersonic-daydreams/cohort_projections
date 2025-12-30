#!/usr/bin/env python3
"""
Prepare input packages for ChatGPT 5.2 Pro methodological review sessions.

Each session gets a prompt.md file plus separate data files for attachment.
This replaces the old single-file approach that exceeded input limits.

Usage:
    python prepare_chatgpt_packages.py G1  # Prepare Session G1 package
    python prepare_chatgpt_packages.py all # Prepare all packages

Session Order (by execution):
    01_G01_estimand   - Estimand & Identification (Session 1)
    02_G04_causal     - Causal Inference (Session 2) - SCM decision
    03_G02_inference  - Inference & Forecasting (Session 3)
    04_G03_gravity    - Gravity Model (Session 4)
    05_G05_duration   - Duration Analysis (Session 5)
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "sdc_2024_replication/scripts/statistical_analysis"
DATA_ROOT = PROJECT_ROOT / "data/processed/immigration/analysis"
OUTPUT_ROOT = ANALYSIS_ROOT / "journal_article/revision_outputs"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def copy_file(src: Path, dest: Path) -> int:
    """Copy a file and return its size in bytes."""
    shutil.copy2(src, dest)
    return dest.stat().st_size


def parquet_to_csv(src: Path, dest: Path, filter_dict: dict = None) -> int:
    """Convert parquet to CSV, optionally filtering. Returns size in bytes."""
    df = pd.read_parquet(src)
    if filter_dict:
        col = filter_dict["column"]
        vals = filter_dict["values"]
        df = df[df[col].isin(vals)]
    df.to_csv(dest, index=False)
    return dest.stat().st_size


def combine_json_results(output_path: Path) -> int:
    """Combine all JSON result files into one file. Returns size in bytes."""
    results_dir = ANALYSIS_ROOT / "results"
    combined = {}
    for filepath in sorted(results_dir.glob("*.json")):
        name = filepath.stem
        with open(filepath) as f:
            combined[name] = json.load(f)

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    return output_path.stat().st_size


def prepare_g1_package() -> dict:
    """
    Session 01_G01: Estimand & Identification Strategy
    First session - defines the estimand.
    """
    session_dir = OUTPUT_ROOT / "01_G01_estimand"
    session_dir.mkdir(parents=True, exist_ok=True)

    files_created = {}

    # Create prompt.md
    prompt = """# G01: Estimand & Identification Strategy Review

## Context

You are reviewing an academic paper titled **"Forecasting International Migration to North Dakota: A Multi-Method Empirical Analysis"** that is being revised for submission to a top-tier demographic methods journal.

The paper uses a nine-module statistical analysis pipeline to examine international migration patterns to North Dakota and develop projection scenarios. A previous review identified several major issues that need to be addressed, with the top priority being: **clarifying and committing to a single forecasting target (estimand)**.

## Your Role

You are acting as a methodological consultant helping the authors strengthen the paper's identification strategy and estimand clarity. Your task is to:

1. Define the forecast target clearly
2. Map data sources to that target
3. Assess identification assumptions
4. Recommend whether to include/exclude Structural Causal Model (SCM) framework
5. Provide specific actionable recommendations

## Attached Files

Please review the following files (numbered for reference):

1. **01_critique.md** - Previous reviewer critique identifying key issues (focus on Sections 1, 2, 6 regarding estimand and identification)

2. **02_report.md** - Full statistical analysis report with findings from all nine modules

3. **03_results_combined.json** - Combined JSON results from all analysis modules (summary statistics, time series, panel, gravity, causal inference, duration analysis, scenario modeling)

4. **04_data_nd_migration.csv** - Primary migration data series (2010-2024) with ND and US international migration counts

5. **05_latex_data_methods.tex** - Data and Methods section from the LaTeX article

6. **06_latex_results.tex** - Results section from the LaTeX article

---

## Task Instructions

### Part 1: Estimand Definition

The reviewer critique states:
> "You move among at least four 'international migration' objects: (1) PEP net international migration, (2) Refugee arrivals (RPC), (3) LPR admissions (DHS), (4) Foreign-born stock (ACS). A top-tier paper needs a clean statement like: 'The forecast target is PEP net international migration to North Dakota (calendar year); the other sources are used to...'"

**Your task:**
- Propose a clear, formal estimand statement
- Define the primary forecast target variable
- Explain how each secondary data source relates to the estimand
- Address calendar year vs. fiscal year harmonization
- Identify any measurement error or conceptual issues

### Part 2: Source Mapping Matrix

Create a source mapping table that shows:
- Each data source (PEP, DHS LPR, ACS Foreign-Born, RPC Refugee)
- What it measures (flow vs. stock, net vs. gross, population type)
- Time period and geographic coverage
- Role in the analysis (primary target vs. predictor vs. validation)
- Limitations and caveats

### Part 3: Identification Assessment

For each major causal claim in the paper, assess:

1. **Travel Ban DiD (Module 7)** - Are parallel trends credible? Treatment definition? Threats?

2. **Synthetic Control for ND (Module 7)** - "A national policy shock affects all states" - is SCM appropriate? Recommend keep, modify, or drop.

3. **Gravity Model Network Effects (Module 5)** - Is the network elasticity estimate causal or associational? Language adjustments needed?

4. **Bartik Shift-Share (Module 7)** - Are shares predetermined? Instrument strength adequate? Inference procedure?

### Part 4: SCM Framework Decision

Consider whether adding a Structural Causal Model (DAG) would help or hurt given the paper's forecasting focus.

### Part 5: Specific Recommendations

Provide a prioritized list of: must-fix, should-fix, could-fix issues.

---

## OUTPUT FORMAT

Please produce three downloadable files:

### File 1: G01_recommendations.md
- Executive summary (1 paragraph)
- Prioritized recommendations list
- Specific text suggestions for the estimand statement
- Language fixes for causal claims

### File 2: G01_specifications.md
- Formal estimand definition
- Identification assumptions for each causal analysis
- SCM decision with justification

### File 3: G01_source_mapping.md
- Data source mapping table
- Calendar/fiscal year harmonization strategy

---

**Paper's Core Contribution:** Demonstrate that rigorous statistical analysis is feasible with small samples (n=15 years), and that uncertainty quantification matters more than point forecasts.

**Target Journal:** Top-tier demographic methods journal (Demography, Population Studies)
"""
    prompt_path = session_dir / "prompt.md"
    prompt_path.write_text(prompt)
    files_created["prompt.md"] = len(prompt)

    # Copy files
    files_created["01_critique.md"] = copy_file(
        ANALYSIS_ROOT
        / "journal_article/output/ChatGPT_5-2-Pro_article_draft_critique.md",
        session_dir / "01_critique.md",
    )

    files_created["02_report.md"] = copy_file(
        ANALYSIS_ROOT / "results/STATISTICAL_ANALYSIS_REPORT.md",
        session_dir / "02_report.md",
    )

    files_created["03_results_combined.json"] = combine_json_results(
        session_dir / "03_results_combined.json"
    )

    files_created["04_data_nd_migration.csv"] = copy_file(
        DATA_ROOT / "nd_migration_summary.csv",
        session_dir / "04_data_nd_migration.csv",
    )

    files_created["05_latex_data_methods.tex"] = copy_file(
        ANALYSIS_ROOT / "journal_article/sections/02_data_methods.tex",
        session_dir / "05_latex_data_methods.tex",
    )

    files_created["06_latex_results.tex"] = copy_file(
        ANALYSIS_ROOT / "journal_article/sections/03_results.tex",
        session_dir / "06_latex_results.tex",
    )

    return files_created


def prepare_g4_package() -> dict:
    """
    Session 02_G04: Causal Inference Deep Dive
    Second session - makes the CRITICAL SCM keep/drop decision.
    """
    session_dir = OUTPUT_ROOT / "02_G04_causal"
    session_dir.mkdir(parents=True, exist_ok=True)

    files_created = {}

    prompt = """# G04: Causal Inference Deep Dive - CRITICAL SCM DECISION SESSION

## Overview

This is the **CRITICAL** session for evaluating and deciding the fate of the Synthetic Control Method (SCM) analysis. The Travel Ban is a **national policy shock** affecting all states simultaneously - this creates a fundamental identification problem that must be resolved with a DEFINITIVE decision.

**Your primary task: Make the definitive keep/drop decision for SCM and specify alternative approaches.**

## Files Attached

1. **01_critique.md** - Full reviewer critique (Section 6b for SCM)
2. **02_results_causal.json** - Complete causal inference module results
3. **03_results_did.json** - Detailed DiD estimates
4. **04_results_scm.json** - Synthetic control results with donor weights
5. **05_data_event_study.csv** - Event study coefficients
6. **06_data_refugee.csv** - Refugee arrivals (filtered: ND, MN, SD, MT)
7. **07_data_panel.csv** - State-level panel data
8. **08_script_causal.py** - Full causal inference implementation

## The Core Problem

The reviewer critique identifies a **fundamental design problem**: "A national policy shock like the Travel Ban affects all states. So 'synthetic ND' from other states is not a clean counterfactual."

**The question is NOT whether SCM produces numbers - it does. The question is whether those numbers have a valid causal interpretation.**

## Your Tasks

### Task 1: DiD Assessment
- Parallel trends validity
- Standard error specification (HC3 vs clustering)
- Log(arrivals+1) functional form vs PPML

### Task 2: SCM Decision (CRITICAL)
Make a definitive recommendation: DROP, REFRAME, or KEEP as descriptive only.
Provide specific justification.

### Task 3: Bartik/Shift-Share Specification
- Base period selection
- Inference approach
- Coefficient interpretation

### Task 4: Triangulation Assessment
How do the three methods relate? What is the strongest defensible claim?

## OUTPUT FORMAT

Please produce four downloadable files:

1. **G04_scm_decision.md** - THE critical output with definitive recommendation
2. **G04_did_specification.md** - DiD specification with clustering recommendation
3. **G04_bartik_specification.md** - Complete Bartik specification
4. **G04_recommendations.md** - Triangulation assessment, priorities
"""
    prompt_path = session_dir / "prompt.md"
    prompt_path.write_text(prompt)
    files_created["prompt.md"] = len(prompt)

    # Copy files
    files_created["01_critique.md"] = copy_file(
        ANALYSIS_ROOT
        / "journal_article/output/ChatGPT_5-2-Pro_article_draft_critique.md",
        session_dir / "01_critique.md",
    )

    files_created["02_results_causal.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_7_causal_inference.json",
        session_dir / "02_results_causal.json",
    )

    files_created["03_results_did.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_7_did_estimates.json",
        session_dir / "03_results_did.json",
    )

    files_created["04_results_scm.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_7_synthetic_control.json",
        session_dir / "04_results_scm.json",
    )

    files_created["05_data_event_study.csv"] = parquet_to_csv(
        ANALYSIS_ROOT / "results/module_7_event_study.parquet",
        session_dir / "05_data_event_study.csv",
    )

    files_created["06_data_refugee.csv"] = parquet_to_csv(
        DATA_ROOT / "refugee_arrivals_by_state_nationality.parquet",
        session_dir / "06_data_refugee.csv",
        filter_dict={
            "column": "state",
            "values": ["North Dakota", "Minnesota", "South Dakota", "Montana"],
        },
    )

    files_created["07_data_panel.csv"] = copy_file(
        DATA_ROOT / "combined_components_of_change.csv",
        session_dir / "07_data_panel.csv",
    )

    files_created["08_script_causal.py"] = copy_file(
        ANALYSIS_ROOT / "module_7_causal_inference.py",
        session_dir / "08_script_causal.py",
    )

    return files_created


def prepare_g2_package() -> dict:
    """
    Session 03_G02: Small-Sample Inference & Forecasting Validation
    Third session - KPSS resolution, backtesting design, terminology audit.
    """
    session_dir = OUTPUT_ROOT / "03_G02_inference"
    session_dir.mkdir(parents=True, exist_ok=True)

    files_created = {}

    prompt = """# G02: Small-Sample Inference & Forecasting Review

## Context

You are reviewing the statistical methodology for a journal article on forecasting international migration to North Dakota. The analysis uses **n=15 annual observations (2010-2024)**, which presents significant challenges for classical statistical inference.

### Key Issues to Address

1. **KPSS Interpretation Contradiction**: Text says "fails to reject stationarity" but table shows rejection
2. **Unit Root vs Structural Break Confusion**: Major break around 2020-2021
3. **Scenario Arithmetic Issues**: 8% growth calculations, CV discrepancy
4. **Terminology Issues**: "credible intervals" vs "prediction intervals"
5. **Backtesting Gap**: No proper out-of-sample validation

## Files Attached

1. **01_critique.md** - Full critique
2. **02_results_unit_root.json** - Unit root test results
3. **03_results_breaks.json** - Structural break results
4. **04_results_arima.json** - ARIMA model results
5. **05_results_scenario.json** - Scenario modeling results
6. **06_results_summary.json** - Summary statistics (CV context)
7. **07_data_nd_migration.csv** - The actual time series (15 obs)
8. **08_script_unit_root.py** - Unit root test implementation
9. **09_script_scenario.py** - Scenario modeling implementation

## Tasks

### Task 1: KPSS Resolution
Clarify what the KPSS results actually show.

### Task 2: Break-Robust Testing
Recommend break-robust alternatives.

### Task 3: Backtesting Design
Design a feasible backtesting procedure for n=15.

### Task 4: Scenario Verification
Verify the scenario arithmetic.

### Task 5: Terminology Audit
Flag all corrections needed.

## OUTPUT FORMAT

Please produce four downloadable files:

1. **G02_kpss_resolution.md** - Correct interpretation
2. **G02_backtesting_spec.md** - Rolling-origin specification
3. **G02_terminology_corrections.csv** - Table: original_phrase, replacement_phrase, context
4. **G02_recommendations.md** - Prioritized changes
"""
    prompt_path = session_dir / "prompt.md"
    prompt_path.write_text(prompt)
    files_created["prompt.md"] = len(prompt)

    # Copy files
    files_created["01_critique.md"] = copy_file(
        ANALYSIS_ROOT
        / "journal_article/output/ChatGPT_5-2-Pro_article_draft_critique.md",
        session_dir / "01_critique.md",
    )

    files_created["02_results_unit_root.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_2_1_1_unit_root_tests.json",
        session_dir / "02_results_unit_root.json",
    )

    files_created["03_results_breaks.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_2_1_2_structural_breaks.json",
        session_dir / "03_results_breaks.json",
    )

    files_created["04_results_arima.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_2_1_arima_model.json",
        session_dir / "04_results_arima.json",
    )

    files_created["05_results_scenario.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_9_scenario_modeling.json",
        session_dir / "05_results_scenario.json",
    )

    files_created["06_results_summary.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_1_1_summary_statistics.json",
        session_dir / "06_results_summary.json",
    )

    files_created["07_data_nd_migration.csv"] = copy_file(
        DATA_ROOT / "nd_migration_summary.csv",
        session_dir / "07_data_nd_migration.csv",
    )

    files_created["08_script_unit_root.py"] = copy_file(
        ANALYSIS_ROOT / "module_2_1_1_unit_root_tests.py",
        session_dir / "08_script_unit_root.py",
    )

    files_created["09_script_scenario.py"] = copy_file(
        ANALYSIS_ROOT / "module_9_scenario_modeling.py",
        session_dir / "09_script_scenario.py",
    )

    return files_created


def prepare_g3_package() -> dict:
    """
    Session 04_G03: Gravity & Network Model Specification
    Fourth session - PPML specification, SE analysis, causal language.
    """
    session_dir = OUTPUT_ROOT / "04_G03_gravity"
    session_dir.mkdir(parents=True, exist_ok=True)

    files_created = {}

    prompt = """# G03: Gravity Model and Network Analysis Review

## Context

A prior reviewer identified several concerns with the gravity model analysis:

1. **Cross-Section Limitations**: Single FY2023 cross-section - cannot claim "causal network effect"
2. **Implausibly Small Standard Errors**: SEs of ~0.001-0.002 raise flags
3. **Missing Distance Variable**: Full specification appears to omit distance
4. **Identification Limbo**: Paper is between "prediction tool" and "causal mechanism"

## Files Attached

1. **01_critique.md** - Full critique (see Section 5 for gravity-specific issues)
2. **02_results_gravity.json** - Gravity model results (3 PPML specs)
3. **03_results_network.json** - Full gravity/network analysis
4. **04_results_effects.json** - Network effects summary
5. **05_data_lpr.csv** - DHS LPR admissions data (state x country, FY2023)
6. **06_script_gravity.py** - Gravity model implementation

## Tasks

### Task 1: PPML Specification Review
Evaluate the three specifications. What is missing?

### Task 2: Identification Analysis
What can/cannot be claimed from a single cross-section?

### Task 3: Standard Error Analysis
Why are SEs so small? Clustering recommendations?

### Task 4: Revised Specification
Provide complete revised specification.

## OUTPUT FORMAT

Please produce three downloadable files:

1. **G03_gravity_specification.md** - Revised specification with LaTeX equations
2. **G03_se_analysis.md** - Standard error diagnosis and corrections
3. **G03_recommendations.md** - Language corrections, implementation priorities
"""
    prompt_path = session_dir / "prompt.md"
    prompt_path.write_text(prompt)
    files_created["prompt.md"] = len(prompt)

    # Copy files
    files_created["01_critique.md"] = copy_file(
        ANALYSIS_ROOT
        / "journal_article/output/ChatGPT_5-2-Pro_article_draft_critique.md",
        session_dir / "01_critique.md",
    )

    files_created["02_results_gravity.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_5_gravity_model.json",
        session_dir / "02_results_gravity.json",
    )

    files_created["03_results_network.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_5_gravity_network.json",
        session_dir / "03_results_network.json",
    )

    files_created["04_results_effects.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_5_network_effects.json",
        session_dir / "04_results_effects.json",
    )

    files_created["05_data_lpr.csv"] = parquet_to_csv(
        DATA_ROOT / "dhs_lpr_by_state_country.parquet",
        session_dir / "05_data_lpr.csv",
    )

    files_created["06_script_gravity.py"] = copy_file(
        ANALYSIS_ROOT / "module_5_gravity_network.py",
        session_dir / "06_script_gravity.py",
    )

    return files_created


def prepare_g5_package() -> dict:
    """
    Session 05_G05: Duration Analysis → Forecasting Bridge
    Fifth session - connect duration analysis to operational forecasting.
    """
    session_dir = OUTPUT_ROOT / "05_G05_duration"
    session_dir.mkdir(parents=True, exist_ok=True)

    files_created = {}

    prompt = """# G05: Duration Analysis to Forecasting Bridge

## Context

The paper includes duration/survival analysis of migration waves (Module 8), which is analytically interesting but currently feels "orphaned" from the forecasting framework.

**The core problem:** The duration analysis shows that refugee arrival waves have a median duration of 3 years, with intensity and nationality region significantly affecting wave persistence. But this is currently presented as descriptive analysis rather than operationally connected to forecasting.

**Your task:** Develop the THEORETICAL and MATHEMATICAL framework for connecting duration/hazard analysis to operational forecasting.

## Critical Data Limitation

With only 15 years of data (2010-2024), there are very few complete migration waves observed. The focus must be on the THEORETICAL framework, not precise parameter estimates.

## Files Attached

1. **01_critique.md** - Original critique
2. **02_results_duration.json** - Complete duration analysis results
3. **03_results_hazard.json** - Cox PH model coefficients
4. **04_results_waves.json** - Wave duration summary
5. **05_results_scenario.json** - Scenario modeling (for integration)
6. **06_data_nd_migration.csv** - ND migration time series
7. **07_script_duration.py** - Duration analysis implementation

## Tasks

### Task 1: Hazard Interpretation for Forecasting
How to translate hazard ratios to forecasting practice?

### Task 2: Theoretical Framework
Develop framework connecting wave detection → duration prediction → flow projection → uncertainty

### Task 3: Mathematical Specifications
Provide precise specifications (survival functions, hazard, transitions)

### Task 4: Implementation Recommendations
Data structures, computational steps, validation

### Task 5: Limitations and Caveats
Honest limitations assessment

## OUTPUT FORMAT

Please produce three downloadable files:

1. **G05_forecasting_bridge.md** - Theoretical framework
2. **G05_specifications.md** - Mathematical formulas
3. **G05_recommendations.md** - Implementation guidance
"""
    prompt_path = session_dir / "prompt.md"
    prompt_path.write_text(prompt)
    files_created["prompt.md"] = len(prompt)

    # Copy files
    files_created["01_critique.md"] = copy_file(
        ANALYSIS_ROOT
        / "journal_article/output/ChatGPT_5-2-Pro_article_draft_critique.md",
        session_dir / "01_critique.md",
    )

    files_created["02_results_duration.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_8_duration_analysis.json",
        session_dir / "02_results_duration.json",
    )

    files_created["03_results_hazard.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_8_hazard_model.json",
        session_dir / "03_results_hazard.json",
    )

    files_created["04_results_waves.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_8_wave_durations.json",
        session_dir / "04_results_waves.json",
    )

    files_created["05_results_scenario.json"] = copy_file(
        ANALYSIS_ROOT / "results/module_9_scenario_modeling.json",
        session_dir / "05_results_scenario.json",
    )

    files_created["06_data_nd_migration.csv"] = copy_file(
        DATA_ROOT / "nd_migration_summary.csv",
        session_dir / "06_data_nd_migration.csv",
    )

    files_created["07_script_duration.py"] = copy_file(
        ANALYSIS_ROOT / "module_8_duration_analysis.py",
        session_dir / "07_script_duration.py",
    )

    return files_created


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python prepare_chatgpt_packages.py [G1|G2|G3|G4|G5|all]")
        print("")
        print("Session order (execution):")
        print("  G1 -> 01_G01_estimand   (Session 1: Estimand)")
        print("  G4 -> 02_G04_causal     (Session 2: Causal - SCM decision)")
        print("  G2 -> 03_G02_inference  (Session 3: Inference)")
        print("  G3 -> 04_G03_gravity    (Session 4: Gravity)")
        print("  G5 -> 05_G05_duration   (Session 5: Duration)")
        sys.exit(1)

    session = sys.argv[1].upper()

    # Mapping of sessions to preparation functions and output directories
    sessions = {
        "G1": ("01_G01_estimand", prepare_g1_package),
        "G4": ("02_G04_causal", prepare_g4_package),
        "G2": ("03_G02_inference", prepare_g2_package),
        "G3": ("04_G03_gravity", prepare_g3_package),
        "G5": ("05_G05_duration", prepare_g5_package),
    }

    if session == "ALL":
        for key, (dirname, func) in sessions.items():
            print(f"Preparing {key} -> {dirname}...")
            files = func()
            total_size = sum(files.values())
            print(f"  Created {len(files)} files ({total_size / 1024:.1f} KB total)")
        print("\nAll packages prepared.")
    elif session in sessions:
        dirname, func = sessions[session]
        print(f"Preparing {session} -> {dirname}...")
        files = func()
        total_size = sum(files.values())
        print(f"Created {len(files)} files ({total_size / 1024:.1f} KB total):")
        for name, size in sorted(files.items()):
            print(f"  {name}: {size / 1024:.1f} KB")
    else:
        print(f"Unknown session: {session}")
        print("Valid options: G1, G2, G3, G4, G5, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
