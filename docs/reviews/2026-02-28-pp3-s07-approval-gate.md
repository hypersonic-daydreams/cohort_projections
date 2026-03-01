# 2026-02-28 PP-003 Phase 1 Scoping Approval Gate (PP3-S07)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Approver** | Human (project lead) |
| **Scope** | PP3-S07 go/no-go decision to proceed from scoping to implementation for Phase 1 city/place projections |
| **Status** | Approved |
| **Related ADR** | ADR-033 |

---

## Approval Decision

**GO** — Phase 1 city/place projection scoping is complete. Implementation may proceed per PP3-S08 kickoff packet.

## Scoping Artifacts Reviewed

| Step | Artifact | Status |
|------|----------|--------|
| PP3-S01 | Scope envelope (355 places, 4 tiers, 2025-2055) | Complete |
| PP3-S02 | [Place data readiness note](2026-02-28-place-data-readiness-note.md) | Complete |
| PP3-S03 | [Place-county mapping strategy note](2026-02-28-place-county-mapping-strategy-note.md) | Complete |
| PP3-S04 | [Modeling specification](2026-02-28-pp3-s04-modeling-spec.md) | Complete — human decisions incorporated |
| PP3-S05 | [Backtesting design and metrics](2026-02-28-pp3-s05-backtesting-design.md) | Complete — human decisions incorporated |
| PP3-S06 | [Output contract](2026-02-28-pp3-s06-output-contract.md) | Complete — human decisions incorporated |

## Key Decisions Recorded

### Modeling (S04)
- **Trend model:** Logit-linear (shares bounded by construction)
- **Fitting variants:** 2x2 backtest matrix — (equal-weight vs. recency-weighted λ=0.9) × (proportional rescaling vs. cap-and-redistribute)
- **Variant selection:** One winner for all tiers, by population-weighted MedAPE
- **Balance-of-county:** Independent logit-linear trend with reconciliation

### Backtesting (S05)
- **Primary window:** Train 2000-2014, test 2015-2024
- **Secondary window:** Train 2000-2019, test 2020-2024 (included)
- **Acceptance thresholds (MedAPE):** HIGH ≤10%, MODERATE ≤15%, LOWER ≤25%
- **Exclusion policy:** Case-by-case, no numeric cap
- **EXCLUDED tier:** Informational backtest (no pass/fail)

### Outputs (S06)
- **LOWER-tier workbook:** Combined sheet with uncertainty caveat header
- **Balance-of-county:** Published as output row per county
- **Race/ethnicity:** Intentionally omitted (no future-phase commitment)
- **Key years:** Same 7 as county workbooks (2025-2055 by 5)

## Next Step

PP3-S08: Publish execution-ready implementation kickoff packet (files, tests, validation gates, ADR touchpoints).

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
