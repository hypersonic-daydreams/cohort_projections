# 2026-03-01 PP3 IMP-20 Draft: ADR-033 Status Update

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | Draft content for ADR-033 status/update closeout after IMP-19 |
| **Status** | Superseded by implemented artifact `docs/reviews/2026-03-01-pp3-imp20-results.md` |
| **Target ADR** | `docs/governance/adrs/033-city-level-projection-methodology.md` |

## Proposed ADR-033 Status Transition

- **Current**: `Deferred`
- **Proposed**: `Accepted`
- **Proposed Implemented Date**: `2026-03-01`

## Proposed Implementation Results Content

### Implementation Outcome

- PP-003 Phase 1 (IMP-01 through IMP-19) executed end-to-end.
- Production model selection finalized in IMP-09: **Variant B-II** (`wls` + `cap_and_redistribute`), winner score `3.0757840903894844`.
- IMP-19 end-to-end validation passed across all active scenarios (`baseline`, `restricted_growth`, `high_growth`) with zero hard-constraint violations.

### Backtest Acceptance Metrics (Primary Window, Winner B-II)

- `HIGH` tier: `tier_medape=3.002588`, `tier_p90_mape=4.314164`, `abs_tier_mean_me=1.106081` (PASS)
- `MODERATE` tier: `tier_medape=1.835121`, `tier_p90_mape=17.781376`, `abs_tier_mean_me=4.280532` (PASS)
- `LOWER` tier: `tier_medape=4.248954`, `tier_p90_mape=11.049137`, `abs_tier_mean_me=0.775915` (PASS)
- Primary acceptance gate flag: `all_scored_tiers_pass_primary=true`

### Prediction Interval Snapshot (Winner B-II, Primary Window)

- `HIGH`: PI80 ±`4.377292%`, PI90 ±`5.597436%`
- `MODERATE`: PI80 ±`11.321926%`, PI90 ±`23.437520%`
- `LOWER`: PI80 ±`8.337710%`, PI90 ±`13.039024%`

### Structural-Break / Exclusion Disposition

- Human narrative review completed in IMP-10.
- Structural-break interpretations (Horace, Williston) accepted.
- **No structural-break exclusions applied** for production output.

### IMP-19 Validation Snapshot

- Per scenario output counts: `90` place files + `46` balance rows.
- Hard constraints: pass (`share sum`, `place<=county`, `non-negative`, scenario ordering).
- QA artifacts: full five-file set present per scenario.

## Proposed ADR Revision-History Entry

- `2026-03-01`: Status moved from `Deferred` to `Accepted`; PP-003 IMP-01 through IMP-19 implemented with winner variant B-II and end-to-end validation pass documented in `docs/reviews/pp3-end-to-end-validation.md`.

## Human Approval Required

Changing ADR status is governance-significant and should be approved by the project owner/reviewer before editing ADR-033 status fields.
