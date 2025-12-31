# Revision Status Tracker

> **For Codex:** Read HYBRID_REVISION_PLAN.md, find the first incomplete task below, and execute it.

## Phase 0: ChatGPT Sessions (Manual - User Completes)

| Session | Status | Outputs Location |
|---------|--------|------------------|
| G01 Estimand | ✅ Complete | `revision_outputs/01_G01_estimand/outputs/` |
| G04 Causal | ✅ Complete | `revision_outputs/02_G04_causal/outputs/` |
| G02 Inference | ✅ Complete | `revision_outputs/03_G02_inference/outputs/` |
| G03 Gravity | ✅ Complete | `revision_outputs/04_G03_gravity/outputs/` |
| G05 Duration | ✅ Complete | `revision_outputs/05_G05_duration/outputs/` |

---

## Phase 1: Implementation Tasks (Codex)

**Instructions:** Complete tasks in order. Mark status when done.

| Task | Status | Depends On | Description |
|------|--------|------------|-------------|
| C01 | ✅ Complete | G01 ✅ | Implement estimand mapping and source alignment |
| C02 | ✅ Complete | G02 | Implement break-robust tests, terminology changes |
| C03 | ✅ Complete | G02 | Implement rolling-origin backtesting framework |
| C04 | ✅ Complete | G03 | Re-estimate gravity model with proper SEs |
| C05 | ✅ Complete | G04 | Implement clustered DiD, event study, handle SCM |
| C06 | ✅ Complete | G05 ✅ | Implement duration → forecasting bridge |
| C07 | ✅ Complete | G02 | Verify and correct scenario arithmetic |
| C08 | ✅ Complete | C01-C07 | Create pipeline diagram, classify modules |
| C09 | ✅ Complete | C01-C08 | Generate missing figures, complete references |
| C10 | ✅ Complete | C01-C09 | Apply terminology changes, tone adjustments |

**Status Key:**
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- ⏸️ Blocked (dependency not met)

---

## Phase 2: Re-Review (User Completes)

| Step | Status | Description |
|------|--------|-------------|
| P2.1 | ✅ Complete | Compile final PDF (`pdflatex main.tex`) → `output/article_draft_v2_revised.pdf` |
| P2.2 | ✅ Complete | Verify no LaTeX errors, all figures render (55 pages, no errors) |
| P2.3a | ✅ Complete | Submit revised PDF to ChatGPT 5.2 Pro for re-review |
| P2.3b | ✅ Complete | Submit revised PDF to Gemini 3 Pro "Deep Think" for re-review |
| P2.4a | ✅ Complete | Save ChatGPT response to `revision_outputs/06_resubmission_review/outputs/` |
| P2.4b | ✅ Complete | Save Gemini response to `revision_outputs/06_resubmission_review/outputs/` |
| P2.5 | ✅ Complete | Compare reviews, assess feedback, create Phase 3 tasks |

---

## Phase 3: Post-Review Revisions (Codex Agents)

**Instructions:** See `revision_outputs/06_resubmission_review/PHASE_3_REVISION_PLAN.md` for detailed task specifications.

### Tier 1: Critical Fixes (Must Complete)

| Task | Status | Agent Group | Description |
|------|--------|-------------|-------------|
| P3.01 | ✅ Complete | A | Abstract corrections (gravity coef, DiD language) |
| P3.02 | ✅ Complete | A | DiD language softening throughout |
| P3.03 | ✅ Complete | C | Backtesting oracle benchmark clarification |
| P3.04 | ✅ Complete | B | Figure 3 scaling/labeling fix |
| P3.05 | ✅ Complete | B | Location Quotient denominator audit (fixed upstream data issue) |
| P3.06 | ✅ Complete | B | Duplicate Kaplan-Meier figure resolution |
| P3.07 | ✅ Complete | B | Residual normality statistics consistency |
| P3.08 | ✅ Complete | E | Monte Carlo double-counting clarification |
| P3.09 | ✅ Complete | A | ITS trend change coefficient caveat |

### Tier 2: Important Improvements (Should Complete)

| Task | Status | Agent Group | Description |
|------|--------|-------------|-------------|
| P3.10 | ✅ Complete | D | Wild cluster bootstrap for DiD |
| P3.11 | ✅ Complete | C | Feasible forecast benchmark addition |
| P3.12 | ✅ Complete | F | Module Outputs → Scenario Inputs mapping table |
| P3.13 | ✅ Complete | E | Gravity model specification transparency |
| P3.14 | ✅ Complete | D | Restricted pre-period DiD robustness |
| P3.15 | ✅ Complete | E | ITS ND-specific interaction term |

### Tier 3: Enhancements (Nice to Complete)

| Task | Status | Agent Group | Description |
|------|--------|-------------|-------------|
| P3.16 | ✅ Complete | F | VAR results display |
| P3.17 | ✅ Complete | F | ML module output transparency |
| P3.18 | ✅ Complete | A | Significance framing language update |
| P3.19 | ✅ Complete | E | Distance variable data collection (standalone analysis) |
| P3.20 | ✅ Complete | G | Figure quality and consistency audit |

**Agent Groups:**
- **A:** Text/Language Edits
- **B:** Numeric Consistency
- **C:** Backtesting/Forecasting
- **D:** Causal Inference Robustness
- **E:** Model Specification
- **F:** Module Integration
- **G:** Quality Assurance

---

## Next Action

**Phase 3 complete.** All 20 tasks resolved. Final PDF needs recompilation.

### Summary of Phase 3 Completion:
- **Tier 1 (Critical):** 9/9 complete
- **Tier 2 (Important):** 6/6 complete
- **Tier 3 (Enhancements):** 5/5 complete

---

## P3.05 Resolution Summary

**Issue:** Table 4 Location Quotient values were incorrect due to two upstream data bugs.

**Bug 1: Variable-depth hierarchy parsing**
- Latin American countries have 5-level hierarchies vs. 4 for others
- Fixed by detecting intermediate categories (Central America, South America, Caribbean)

**Bug 2: Year-varying variable codes**
- Census changed variable codes between ACS years (e.g., B05006_059E was India in 2019, Bhutan in 2023)
- Fixed by saving and loading year-specific label files

**Before/After Comparison (2023 national totals):**
| Country | BEFORE (Wrong) | AFTER (Correct) |
|---------|----------------|-----------------|
| India | 44,172 | 2,775,531 |
| Mexico | 24,102,255 | 10,816,311 |
| Belize | 23,269,608 | 47,227 |
| Canada | 10,816,311 | 823,584 |

**Updated LQ Top 10:**
| Country | LQ |
|---------|-----|
| Liberia | 40.83 |
| Ivory Coast | 26.19 |
| Somalia | 22.42 |
| Tanzania | 15.13 |
| DR Congo | 13.29 |
| Bhutan | 9.86 |
| Sudan | 9.23 |
| Zimbabwe | 8.21 |
| Kenya | 7.26 |
| Nepal | 6.63 |

**Files Updated:**
- `download_b05006.py` - saves year-specific label files
- `process_b05006.py` - loads year-specific labels, handles 5-level hierarchies
- Table 4 in `03_results.tex` - corrected values
- Discussion section in `04_discussion.tex` - corrected country references
- Abstract in `main.tex` - updated LQ claims

---

## Completion Log

| Task | Completed | Notes |
|------|-----------|-------|
| G01 | 2024-12-29 | ChatGPT produced recommendations, specifications, source_mapping |
| G02 | 2024-12-29 | ChatGPT produced KPSS resolution, backtesting spec, terminology corrections |
| G03 | 2024-12-29 | ChatGPT produced gravity specification, SE analysis, recommendations |
| G04 | 2024-12-29 | ChatGPT produced SCM decision, DiD spec, Bartik spec, recommendations |
| G05 | 2024-12-29 | ChatGPT produced forecasting_bridge, specifications, recommendations |
| C01 | 2025-12-30 | Added estimand/measurement subsection, source mapping table, FY vs PEP-year note; standardized net migration labels; LaTeX compiles |
| C06 | 2025-12-30 | Added wave registry + conditional duration predictor, integrated wave persistence into scenario Monte Carlo, added survival-curve figure and results bridge text |
| C02 | 2025-12-30 | Added Zivot-Andrews break-robust unit root test, corrected KPSS interpretation and appendix table, updated interval terminology and CV note; LaTeX compiles |
| C03 | 2025-12-30 | Implemented rolling-origin backtesting module, saved results JSON, added results table to Section 3; LaTeX compiles |
| C04 | 2025-12-30 | Rebuilt gravity dataset with zeros, added clustered SE inference, updated gravity tables/text with diaspora association language, saved revised results; LaTeX compiles |
| C05 | 2025-12-30 | Re-estimated DiD/ITS with clustered SEs, reframed synthetic comparator as descriptive, updated shift-share inference, generated event study figure; LaTeX text updated |
| C07 | 2025-12-30 | Validated scenario arithmetic, added validation script and appendix formulas, updated scenario descriptions and Monte Carlo summaries; scenario outputs regenerated |
| C08 | 2025-12-30 | Added pipeline overview section with module classification table, generated analysis pipeline diagram, and documented module dependencies |
| C09 | 2025-12-30 | Regenerated publication figures/captions, added pipeline diagram assets, validated citation coverage; ensured PDF+PNG figure set |
| C10 | 2025-12-30 | Completed terminology sweep, updated abstract and scenario descriptions, and aligned Monte Carlo intervals across sections |
| P3.01 | 2025-12-31 | Abstract corrections: gravity coefficient insignificance noted, DiD language softened to "policy-associated divergence" |
| P3.02 | 2025-12-31 | DiD language softening throughout article; added "upper bound" interpretation |
| P3.03 | 2025-12-31 | Backtesting: Added "Oracle (ex-post)" label, feasible competitor, MASE metric, justification for ARIMA in scenarios |
| P3.04 | 2025-12-31 | Figure 3 scaling: Verified correct (0.1-0.3% is basis points), added clarifying footnote |
| P3.06 | 2025-12-31 | Duplicate KM figure: Consolidated into single canonical Figure 8; removed duplicate |
| P3.07 | 2025-12-31 | Residual statistics: Unified Shapiro-Wilk values across text, caption, and figure annotation |
| P3.08 | 2025-12-31 | Monte Carlo clarification: Added appendix note explaining wave simulation modulates existing variance |
| P3.09 | 2025-12-31 | ITS caveat: Added cautionary note about +14,113 coefficient as short-term rebound slope |
| P3.10 | 2025-12-31 | Wild cluster bootstrap: Implemented with Rademacher weights (999 iter); full p=0.077, restricted p=0.003 |
| P3.11 | 2025-12-31 | Feasible benchmark: Added lagged Driver OLS, MASE metric to Table 6 |
| P3.12 | 2025-12-31 | Module mapping table: Added Section 2.9 Table 11 with module→scenario input mapping |
| P3.13 | 2025-12-31 | Gravity transparency: Renamed to "cross-sectional allocation model (PPML)", clarified distance exclusion |
| P3.14 | 2025-12-31 | Restricted pre-period: Estimated DiD with 2013-2017 pre-period; pre-trend test now passes (p=0.553) |
| P3.15 | 2025-12-31 | ITS ND-specific: Added ND × Post2020 interaction term; ND tracks national average |
| P3.16 | 2025-12-31 | VAR results: Added coefficient table and diagnostics to Appendix |
| P3.17 | 2025-12-31 | ML transparency: Added paragraph in Section 3.9 explaining ML role in feature selection |
| P3.20 | 2025-12-31 | Figure audit: All 12 figures verified; PDF+PNG present; numbering sequential; no broken refs |
| P3.18 | 2025-12-31 | Significance framing: Replaced "conventional levels" with cautious small-sample language in Results intro |
| P3.19 | 2025-12-31 | Distance analysis: Created standalone analysis with 110-country distance dataset; found positive distance effect (refugee patterns) |
| P3.05 | 2025-12-31 | LQ denominator fix: Fixed year-varying Census variable codes and 5-level hierarchies; corrected Table 4 (top LQ now Liberia at 40.83) |
