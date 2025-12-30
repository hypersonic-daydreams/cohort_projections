# Hybrid Revision Plan: ChatGPT 5.2 Pro + Codex

## For Codex: Start Here

**To find your next task:**
1. Read `REVISION_STATUS.md` in this directory
2. Find the first task with status ⬜ whose dependencies are ✅
3. Read the task instructions in the section below
4. Execute the task
5. Update `REVISION_STATUS.md` to mark the task ✅

---

## Overview

This revision uses two AI systems:
- **ChatGPT 5.2 Pro**: Methodological review (Phase 0) - completed by user manually
- **Codex**: Implementation (Phase 1) - executes based on ChatGPT recommendations

---

## Phase 0: ChatGPT Sessions (User Completes Manually)

These sessions are run by the user in ChatGPT. Each session folder contains:
- `prompt.md` - Copy into ChatGPT
- Numbered files (01_*, 02_*, etc.) - Attach to ChatGPT
- `outputs/` - Save ChatGPT's downloaded files here

| Order | Session | Folder | Purpose |
|-------|---------|--------|---------|
| 1 | G01 | `revision_outputs/01_G01_estimand/` | Define estimand, source mapping |
| 2 | G04 | `revision_outputs/02_G04_causal/` | SCM keep/drop decision, DiD fixes |
| 3 | G02 | `revision_outputs/03_G02_inference/` | KPSS resolution, backtesting spec |
| 4 | G03 | `revision_outputs/04_G03_gravity/` | Gravity model SE fixes |
| 5 | G05 | `revision_outputs/05_G05_duration/` | Duration→forecasting bridge |

---

## Phase 1: Implementation Tasks (Codex)

### Task C01: Estimand Mapping
**Depends on:** G01 ✅
**ChatGPT outputs:** `revision_outputs/01_G01_estimand/outputs/G01_*.md`

**Instructions:**
1. Read `revision_outputs/01_G01_estimand/outputs/G01_recommendations.md`
2. Read `revision_outputs/01_G01_estimand/outputs/G01_specifications.md`
3. Read `revision_outputs/01_G01_estimand/outputs/G01_source_mapping.md`

**Tasks:**
- Add "Estimand & Measurement" subsection to `sections/02_data_methods.tex`
- Add source mapping table (from G01_source_mapping.md)
- Add FY/PEP-year harmonization note
- Update variable labels throughout for consistency

**Validation:** Compile LaTeX (`pdflatex main.tex`)

---

### Task C02: Inference Methodology
**Depends on:** G02
**ChatGPT outputs:** `revision_outputs/03_G02_inference/outputs/G02_*.md`

**Instructions:**
1. Read `G02_kpss_resolution.md` - fix KPSS interpretation
2. Read `G02_terminology_corrections.csv` - apply terminology fixes
3. Read `G02_recommendations.md` - prioritized changes

**Tasks:**
- Fix KPSS interpretation in `sections/03_results.tex`
- Add break-robust unit root test to `module_02_time_series/unit_root_tests.py`
- Apply terminology corrections (credible→prediction intervals, etc.)
- Add CV explanation footnote

**Validation:** Run unit root script, compile LaTeX

---

### Task C03: Backtesting Framework
**Depends on:** G02
**ChatGPT outputs:** `revision_outputs/03_G02_inference/outputs/G02_backtesting_spec.md`

**Instructions:**
1. Read `G02_backtesting_spec.md` for protocol specification

**Tasks:**
- Create `module_02_time_series/backtesting.py`
- Implement rolling-origin cross-validation
- Add benchmark models (naive, mean, ARIMA)
- Compute MAE, RMSE, MAPE, coverage
- Save results to `results/backtesting_results.json`
- Add results table to `sections/03_results.tex`

**Validation:** Run backtesting script, verify no data leakage

---

### Task C04: Gravity Model
**Depends on:** G03
**ChatGPT outputs:** `revision_outputs/04_G03_gravity/outputs/G03_*.md`

**Instructions:**
1. Read `G03_gravity_specification.md` - correct PPML spec
2. Read `G03_se_analysis.md` - clustering recommendations
3. Read `G03_recommendations.md` - language changes

**Tasks:**
- Implement clustered SEs in `module_05_gravity/gravity_model.py`
- Re-estimate gravity model
- Save to `results/gravity_results_revised.json`
- Update "network effects" language to "diaspora associations"
- Add limitations paragraph

**Validation:** Run gravity script, compare old vs new SEs

---

### Task C05: Causal Inference
**Depends on:** G04
**ChatGPT outputs:** `revision_outputs/02_G04_causal/outputs/G04_*.md`

**Instructions:**
1. **CRITICAL:** Read `G04_scm_decision.md` first - determines SCM fate
2. Read `G04_did_specification.md` - DiD improvements
3. Read `G04_bartik_specification.md` - shift-share inference
4. Read `G04_recommendations.md` - implementation priorities

**Tasks:**
- Implement SCM decision (KEEP/DROP/REFRAME per G04_scm_decision.md)
- Add clustered SEs to DiD in `module_07_causal/did_analysis.py`
- Create event study figure `figures/event_study_travel_ban.pdf`
- Update Bartik inference approach
- Apply language changes to `sections/03_results.tex`

**Validation:** Run causal scripts, verify event study figure

---

### Task C06: Duration→Forecasting Bridge
**Depends on:** G05 ✅
**ChatGPT outputs:** `revision_outputs/05_G05_duration/outputs/G05_*.md`

**Instructions:**
1. Read `G05_forecasting_bridge.md` - theoretical framework
2. Read `G05_specifications.md` - mathematical specs
3. Read `G05_recommendations.md` - implementation guidance

**Tasks:**
- Create wave registry in `module_08_duration/wave_registry.py`
- Implement conditional duration predictor
- Integrate with `module_09_scenario/scenario_modeling.py`
- Create survival curve figure `figures/wave_survival_curves.pdf`
- Add explanation to `sections/03_results.tex`

**Validation:** Run duration scripts, verify scenario integration

---

### Task C07: Scenario Arithmetic
**Depends on:** G02
**ChatGPT outputs:** `revision_outputs/03_G02_inference/outputs/G02_recommendations.md`

**Instructions:**
1. Read scenario arithmetic issues in G02_recommendations.md

**Tasks:**
- Audit CBO Full scenario (8% growth calculation)
- Audit Pre-2020 Trend scenario (anchoring)
- Reconcile CV values (82.5% vs 0.39)
- Create `module_09_scenario/validate_scenarios.py`
- Fix any errors found
- Update `sections/03_results.tex` if needed

**Validation:** Run validation script, hand-check arithmetic

---

### Task C08: Pipeline Integration
**Depends on:** C01-C07

**Tasks:**
- Create module classification table (forecasting vs causal)
- Create pipeline diagram `figures/analysis_pipeline.pdf`
- Document module dependencies
- Add diagram to `sections/02_data_methods.tex`

**Validation:** Compile LaTeX, verify diagram renders

---

### Task C09: Figures
**Depends on:** C01-C08

**Tasks:**
- Inventory all `\includegraphics` references
- Generate any missing figures
- Standardize format (PDF + PNG)
- Review and update captions

**Validation:** Compile LaTeX, no missing figure warnings

---

### Task C10: Final Polish
**Depends on:** C01-C09

**Tasks:**
- Consolidate all terminology corrections from G01-G05
- Apply remaining search-replace fixes
- Add hedging language to causal claims
- Check consistency (variable names, data sources, time periods)
- Final proofread pass

**Validation:** Full LaTeX compile, read-through of key sections

---

## Repository Structure

```
journal_article/
├── HYBRID_REVISION_PLAN.md    # This file - main instructions
├── REVISION_STATUS.md         # Track what's complete
├── main.tex                   # Main LaTeX document
├── sections/                  # LaTeX sections to modify
├── figures/                   # Generated figures
└── revision_outputs/
    ├── 01_G01_estimand/       # ChatGPT session + outputs
    │   ├── prompt.md
    │   ├── 01_critique.md
    │   └── outputs/
    │       └── G01_*.md       # ChatGPT recommendations
    ├── 02_G04_causal/
    ├── 03_G02_inference/
    ├── 04_G03_gravity/
    ├── 05_G05_duration/
    └── C01_estimand/          # (Legacy - prompts now inline above)
```

---

## Quick Reference: File Locations

| Need | Location |
|------|----------|
| G01 recommendations | `revision_outputs/01_G01_estimand/outputs/G01_*.md` |
| G02 recommendations | `revision_outputs/03_G02_inference/outputs/G02_*.md` |
| G03 recommendations | `revision_outputs/04_G03_gravity/outputs/G03_*.md` |
| G04 recommendations | `revision_outputs/02_G04_causal/outputs/G04_*.md` |
| G05 recommendations | `revision_outputs/05_G05_duration/outputs/G05_*.md` |
| LaTeX sections | `sections/*.tex` |
| Analysis modules | `../../module_*/` (relative to journal_article) |
| Results JSON | `../../module_*/results/` |

---

*Last Updated: December 29, 2024*
