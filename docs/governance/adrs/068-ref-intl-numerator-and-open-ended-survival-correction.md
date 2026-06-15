# ADR-068: Correct the CBO International-Migration Numerator and the Open-Ended 90+ Survival

**Status:** Accepted
**Date:** 2026-06-15
**Supersedes (numbers):** the 2026-06-13 locked run (`m2026r1` / `cfg-20260611-production-lock`)
**Related:** ADR-050 (additive migration adjustment), ADR-065 (CBO-adjusted baseline), ADR-066 (Vintage 2025 base), ADR-067 (divergence investigation), ADR-002/ADR-053 (mortality)
**Evidence:** [`docs/reviews/2026-06-15-ref-intl-migration-sum-vs-average.md`](../../reviews/2026-06-15-ref-intl-migration-sum-vs-average.md); [`docs/reviews/2026-06-15-ref-intl-sensitivity/`](../../reviews/2026-06-15-ref-intl-sensitivity/) (incl. the GPT-5.5 Pro xhigh review)

## Context

A finality re-review of the PUB-2026 public baseline — prompted by a request to re-validate the CBO migration assumptions, escalated through a multi-agent verification workflow and an independent **GPT-5.5 Pro** audit — identified two production-affecting errors in the locked run. Both were confirmed against the live repo data and code.

1. **CBO international-migration numerator was a 3-year SUM, applied per year.** `reference_intl_migration: 10051` was labeled "annual average (PEP 2023-2025)" but is the **sum** of ND international net migration for 2023+2024+2025 (3,158 + 4,083 + 2,810 = 10,051). The additive-reduction formula (ADR-050, `core/migration.py`) consumes it as a per-year flow (`annual_reduction = ref_intl × (1 − f)`), so the 2025 step removed 10,051 × 0.80 = 8,041 persons statewide — more than ND's entire realized 2025 net migration (+3,322) and ~3× realized 2025 international migration (+2,810). The true annual average is **3,350.33**. The near-term suppression — and hence the published −1.50% trough — was ~3× too deep.

2. **Open-ended 90+ survival used the 85+ group rate as terminal retention.** The ND CDC baseline is grouped, and the survival build expanded the 85+ group's single-year rate flat across ages 85-100. The cohort engine holds the open-ended `max_age` (90) group at the age-90 rate every year, so the 90+ pool retained ~0.885 (Male)/0.914 (Female) annually — far too high for an open group containing the 95- and 100-year-olds. The demographically correct open-interval survivorship ratio is `T₉₁/T₉₀` ≈ **0.778 (M)/0.806 (F)** from the CDC 2023 life table. This overstated the oldest-old by ~6,400 persons at 2055.

A third finding — **production mortality is race-flat** (the operative NP2023 survival has no race dimension and is broadcast identically across all six races, despite the methodology claiming race-specific survival) — was confirmed and dispositioned separately (D3).

## Decision

- **D1 — Correct the numerator (accepted).** Set `reference_intl_migration` to the true annual average **3,350.33** in the baseline scenario and the deprecated `restricted_growth` alias. This matches the validated on-disk sensitivity run, whose control (numerator unchanged) reproduced the locked trajectory to the person (max abs diff 0.0000), cleanly isolating the effect.
- **D2 — Correct the open-ended 90+ survival (accepted).** Override the `max_age` survival in the operative table with the open-interval ratio `T₉₁/T₉₀` (CDC 2023 life table, all-race "Total"), preserving the NP2023 improvement trajectory shape. Implemented as `apply_open_ended_survival_correction` in `data/process/mortality_improvement.py`. Ages above `max_age` are unused by the engine (population is capped at `max_age`) and unchanged.
- **D3 — Caveat race-flat mortality for the 2026 vintage; defer the race-specific rebuild (accepted).** The operative survival is applied race-flat. Its state-total effect is modest (dominant white-NH ≈ total), but it over-states AIAN survival in reservation counties (Benson, Sioux, Rolette). For the 2026 release, disclose this as a documented limitation (methodology §4 and §10) rather than rebuild the survival pipeline to carry race; schedule the race-specific rebuild for the next vintage.

## Consequences

Corrected production run (config + 90+ build change; stages 01c → 02 rerun; migration rate prep 01a/01b unaffected):

| Year | Prior locked | Corrected |
|------|-----------:|----------:|
| 2025 | 799,358 | 799,358 |
| trough | 787,382 (2028, −1.50%) | **797,298 (2027, −0.26%)** |
| 2030 | 792,478 | 804,657 |
| 2050 | 872,730 | 877,818 |
| 2055 | 889,017 (+11.22%) | **886,585 (+10.91%)** |

- The signature near-term decline is essentially erased (−1.50% → −0.26%), consistent with ND's observed 2025 growth (+0.75%, net migration +3,322).
- The horizon is **robust**: the two fixes nearly offset (ref_intl ≈ +15,700 @2055; the 90+ correction ≈ −18,100 via no-longer-suppressed old-age deaths), so 2055 barely moves.
- 90+ population at 2055: 9,971 (was ~13,707).
- Divergent-county magnitudes shift: Williams ≈ unchanged (near +51%); Ward and Grand Forks decline somewhat deeper as the 90+ correction raises old-age deaths. Precise GQ-inclusive county figures and all public artifacts are refreshed in the publication step.
- `reference_population` (799,358) is unchanged and correct.

## Implementation Results

- **Config:** `reference_intl_migration: 3350.33` (baseline + restricted alias); fertility provenance comment corrected; `sex_ratio_male: 0.51` added; `aggregation_tolerance` tightened; place projections disabled (out of scope for the 2026 release).
- **Code:** `apply_open_ended_survival_correction` in `mortality_improvement.py`; stale "2025-2045"/"20-year" horizon docstrings corrected across the convergence/mortality modules.
- **Tests** updated to the corrected values and out-of-scope place default; **2,105 passed, 5 skipped**.
- Committed `4c38f6d` (code/config/tests); documentation + run metadata in a follow-up commit. Data (regenerated survival + corrected baseline) syncs via rclone.
- **Corroboration:** four assessments (multi-agent workflow, codex GPT-5.5, GPT-5.5 Pro xhigh, the arithmetic) agreed on D1; the 90+ and race-flat findings were verified directly against the operative survival parquet and the CDC life table.

**Deferred (publication / next vintage):** regenerate the public artifacts (workbook, CSV, PDF copy, marketing `.docx`, pyramid explorer) and re-run the release QA gates against the corrected run; rebuild the survival pipeline to be race-specific (D3) for the next vintage; refresh the SDC-2024 comparison and the divergent-county narratives with corrected county figures.
