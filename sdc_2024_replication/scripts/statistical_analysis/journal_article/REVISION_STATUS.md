# Revision Status Tracker

> **For Codex:** Read HYBRID_REVISION_PLAN.md, find the first incomplete task below, and execute it.

## Phase 0: ChatGPT Sessions (Manual - User Completes)

| Session | Status | Outputs Location |
|---------|--------|------------------|
| G01 Estimand | ✅ Complete | `revision_outputs/01_G01_estimand/outputs/` |
| G04 Causal | ⬜ Not Started | `revision_outputs/02_G04_causal/outputs/` |
| G02 Inference | ⬜ Not Started | `revision_outputs/03_G02_inference/outputs/` |
| G03 Gravity | ⬜ Not Started | `revision_outputs/04_G03_gravity/outputs/` |
| G05 Duration | ✅ Complete | `revision_outputs/05_G05_duration/outputs/` |

---

## Phase 1: Implementation Tasks (Codex)

**Instructions:** Complete tasks in order. Mark status when done.

| Task | Status | Depends On | Description |
|------|--------|------------|-------------|
| C01 | ✅ Complete | G01 ✅ | Implement estimand mapping and source alignment |
| C02 | ⬜ Not Started | G02 | Implement break-robust tests, terminology changes |
| C03 | ⬜ Not Started | G02 | Implement rolling-origin backtesting framework |
| C04 | ⬜ Not Started | G03 | Re-estimate gravity model with proper SEs |
| C05 | ⬜ Not Started | G04 | Implement clustered DiD, event study, handle SCM |
| C06 | ⬜ Not Started | G05 ✅ | Implement duration → forecasting bridge |
| C07 | ⬜ Not Started | G02 | Verify and correct scenario arithmetic |
| C08 | ⬜ Not Started | C01-C07 | Create pipeline diagram, classify modules |
| C09 | ⬜ Not Started | C01-C08 | Generate missing figures, complete references |
| C10 | ⬜ Not Started | C01-C09 | Apply terminology changes, tone adjustments |

**Status Key:**
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- ⏸️ Blocked (dependency not met)

---

## Next Action

**First available task with met dependencies:** C06 (G05 is complete)

To execute: Follow Task C06 in `HYBRID_REVISION_PLAN.md` and read:
- `revision_outputs/05_G05_duration/outputs/G05_forecasting_bridge.md`
- `revision_outputs/05_G05_duration/outputs/G05_specifications.md`
- `revision_outputs/05_G05_duration/outputs/G05_recommendations.md`

---

## Completion Log

| Task | Completed | Notes |
|------|-----------|-------|
| G01 | 2024-12-29 | ChatGPT produced recommendations, specifications, source_mapping |
| G05 | 2024-12-29 | ChatGPT produced forecasting_bridge, specifications, recommendations |
| C01 | 2025-12-30 | Added estimand/measurement subsection, source mapping table, FY vs PEP-year note; standardized net migration labels; LaTeX compiles |
