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
| P3.01 | ⬜ Not Started | A | Abstract corrections (gravity coef, DiD language) |
| P3.02 | ⬜ Not Started | A | DiD language softening throughout |
| P3.03 | ⬜ Not Started | C | Backtesting oracle benchmark clarification |
| P3.04 | ⬜ Not Started | B | Figure 3 scaling/labeling fix |
| P3.05 | ⬜ Not Started | B | Location Quotient denominator audit |
| P3.06 | ⬜ Not Started | B | Duplicate Kaplan-Meier figure resolution |
| P3.07 | ⬜ Not Started | B | Residual normality statistics consistency |
| P3.08 | ⬜ Not Started | E | Monte Carlo double-counting clarification |
| P3.09 | ⬜ Not Started | A | ITS trend change coefficient caveat |

### Tier 2: Important Improvements (Should Complete)

| Task | Status | Agent Group | Description |
|------|--------|-------------|-------------|
| P3.10 | ⬜ Not Started | D | Wild cluster bootstrap for DiD |
| P3.11 | ⬜ Not Started | C | Feasible forecast benchmark addition |
| P3.12 | ⬜ Not Started | F | Module Outputs → Scenario Inputs mapping table |
| P3.13 | ⬜ Not Started | E | Gravity model specification transparency |
| P3.14 | ⬜ Not Started | D | Restricted pre-period DiD robustness |
| P3.15 | ⬜ Not Started | E | ITS ND-specific interaction term |

### Tier 3: Enhancements (Nice to Complete)

| Task | Status | Agent Group | Description |
|------|--------|-------------|-------------|
| P3.16 | ⬜ Not Started | F | VAR results display |
| P3.17 | ⬜ Not Started | F | ML module output transparency |
| P3.18 | ⬜ Not Started | A | Significance framing language update |
| P3.19 | ⬜ Not Started | E | Distance variable data collection (if P3.13 Option A) |
| P3.20 | ⬜ Not Started | G | Figure quality and consistency audit |

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

**Phase 2 complete. Begin Phase 3: Post-Review Revisions.**

Recommended execution order:
1. **Wave 1 (Parallel):** Agent Groups A, B, C
2. **Wave 2 (Parallel):** Agent Groups D, E
3. **Wave 3:** Agent Group F
4. **Final:** Agent Group G

See `revision_outputs/06_resubmission_review/PHASE_3_REVISION_PLAN.md` for detailed task specifications.

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
