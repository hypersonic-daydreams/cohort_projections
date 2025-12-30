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

## Next Action

**All Phase 1 tasks complete.**

Next step: final compile and submission prep.

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
