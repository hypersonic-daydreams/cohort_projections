# ADR-068: Correct the CBO International-Migration Numerator and the Open-Ended 90+ Survival

**Status:** Accepted (amended 2026-06-16 — operative survival-horizon coverage fix; see **Amendment** below)
**Date:** 2026-06-15 (amended 2026-06-16)
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

| Year | Prior locked | Corrected (full-horizon, amended 2026-06-16) |
|------|-----------:|----------:|
| 2025 | 799,358 | 799,358 |
| trough | 787,382 (2028, −1.50%) | **797,298 (2027, −0.26%)** |
| 2030 | 792,478 | 804,657 |
| 2050 | 872,730 | 883,225 |
| 2055 | 889,017 (+11.22%) | **898,907 (+12.45%)** |

(Corrected column reflects the **2026-06-16 amendment** — the operative survival table now spans the full 2025–2055 horizon. The intermediate 2026-06-15 figures, 877,818 @2050 / 886,585 @2055 / 90+ 9,971, were biased by a survival-table truncation; see **Amendment** below.)

- The signature near-term decline is essentially erased (−1.50% → −0.26%), consistent with ND's observed 2025 growth (+0.75%, net migration +3,322). Years 2025–2046 are unaffected by the amendment.
- 90+ population at 2055: **8,172** (was ~13,707 with the uncorrected 85+-plateau survival; the intermediate 2026-06-15 figure of 9,971 reflected the truncated survival table that let the correction lapse for 2047–2055).
- Divergent-county magnitudes shift: Williams ≈ unchanged (near +51%); Ward and Grand Forks decline somewhat deeper than the prior locked run as the 90+ correction raises old-age deaths. Precise GQ-inclusive county figures and all public artifacts are refreshed in the publication step.
- `reference_population` (799,358) is unchanged and correct.

## Implementation Results

- **Config:** `reference_intl_migration: 3350.33` (baseline + restricted alias); fertility provenance comment corrected; `sex_ratio_male: 0.51` added; `aggregation_tolerance` tightened; place projections disabled (out of scope for the 2026 release).
- **Code:** `apply_open_ended_survival_correction` in `mortality_improvement.py`; stale "2025-2045"/"20-year" horizon docstrings corrected across the convergence/mortality modules.
- **Tests** updated to the corrected values and out-of-scope place default; **2,105 passed, 5 skipped**.
- Committed `4c38f6d` (code/config/tests); documentation + run metadata in a follow-up commit. Data (regenerated survival + corrected baseline) syncs via rclone.
- **Corroboration:** four assessments (multi-agent workflow, codex GPT-5.5, GPT-5.5 Pro xhigh, the arithmetic) agreed on D1; the 90+ and race-flat findings were verified directly against the operative survival parquet and the CDC life table.

**Deferred (publication / next vintage):** regenerate the public artifacts (workbook, CSV, PDF copy, marketing `.docx`, pyramid explorer) and re-run the release QA gates against the corrected run; rebuild the survival pipeline to be race-specific (D3) for the next vintage; refresh the SDC-2024 comparison and the divergent-county narratives with corrected county figures.

## Amendment (2026-06-16): Operative Survival-Table Horizon Coverage

**Finding (publication-prep QA).** While clearing the round-2 review punch list, verification found that the operative survival table on disk (`data/processed/mortality/nd_adjusted_survival_projections.parquet`) spanned only **2025–2045**, not the full **2025–2055** projection horizon. The engine (`core/cohort_component.py::_get_survival_rates`) **silently falls back to the static-base survival table** (`survival_rates.parquet`, race-specific, with `improvement_factor` re-applied from a 2023 anchor) for any calendar year absent from the operative table. Consequently, for the projection steps **2047→2055** the 2026-06-15 production run used the static base — **not** the NP2023 operative table and **not** the ADR-068 open-interval 90+ correction, both of which the methodology (§4.5, §4.6, §10.7) documents as spanning the horizon. The correction therefore lapsed for the final ~9 years.

**Impact (isolated by a controlled re-run + diff).** Regenerating the survival table to the full horizon (01c with the committed config; the 2025–2045 overlap is byte-identical, max abs survival diff 0.0 — only 2046–2055 rows are added) and re-running stage 02 yields:

| Year | 2026-06-15 (survival truncated) | 2026-06-16 (full-horizon, final) | Δ |
|------|-----------:|----------:|----:|
| 2025–2046 | — | identical | 0 |
| 2050 | 877,818 | **883,225** | +5,407 |
| 2055 | 886,585 (+10.91%) | **898,907 (+12.45%)** | +12,322 |
| 90+ @2055 | 9,971 | **8,172** | −1,799 |

The 2025–2046 trajectory (including the 797,298 @2027 trough) is unchanged. The horizon **total rises** (+12,322 @2055) because the NP2023 operative table carries higher survival than the static-base fallback at working/old ages; the **90+ pool falls** (−1,799) because the open-interval correction now actually applies through 2055. State = Σ counties remains exact (residual ~1e-10), 53 unique county FIPS, base 799,358.

**Root cause (a test clobbering production data).** The on-disk operative table was 2025–2045 because the test `tests/test_data/test_mortality_improvement.py::test_full_pipeline_produces_valid_output` called `run_mortality_improvement_pipeline(config)` with `projection_horizon: 20` and **no output directory** — and the pipeline hard-coded its output to the *production* path `data/processed/mortality/nd_adjusted_survival_projections.parquet`. So **every `pytest` run silently overwrote the production survival table with a truncated 2025–2045 horizon.** Running the suite after `01c` (e.g. during the ADR-068 verification, "2,105 passed") left production at 2025–2045; stage 02 then ran with the truncated table and the 90+ correction lapsed for 2047–2055. Two integration tests had even hard-coded the truncated horizon as their expectation, hiding the regression.

**Fix + guard.** (1) Add an injectable `output_dir` to `run_mortality_improvement_pipeline` (production default) and point the test at `tmp_path` so tests can never overwrite production; bump the stale horizon default 20→30. (2) Add a coverage guard in `scripts/pipeline/02_run_projections.py::load_demographic_rates` that **hard-fails** (raises `RuntimeError`) for production/public runs when the operative survival table does not span `base_year..end_year`, so the silent static-base fallback can never again pass unnoticed; an explicit `allow_static_survival=True` opt-out covers tests, experiments, and constant-mortality scenarios. Backed by a hard release-QA assertion. (This landed initially as a loud warning and was hardened to a hard fail in the PR #25 review pass — see the recurrence-hardening note below.) (3) Make the two horizon-asserting tests derive the expected years from config / the actual table instead of hard-coding 2045. (4) Regenerate the survival table to the full horizon and re-run stage 02 (`--counties --state`). The full suite (2,260 passed, 5 skipped) now leaves production survival at 2025–2055 (verified post-run). The corrected full-horizon run (2026-06-16) supersedes the 2026-06-15 figures everywhere they appear as *current/public-facing* numbers. The independent round-2 GPT-5.5 Pro review (GO) predated this amendment; a re-review against the corrected run is a deferred decision.

**Recurrence hardening (PR #25 GO-WITH-FIXES, 2026-06-16).** A PR-framed GPT-5.5 Pro re-review of the corrected product confirmed the root cause and the corrected numbers (898,907 / 8,172) as sound, and recommended making the anti-regression guards *hard* — a warning is exactly what let the original truncation through. Three code-only guards were added, none of which changes any projection number (verified: production survival still spans 2025–2055; public CSV state-2055 still 898,907; both guards proven silent on the real production data):

- **M1** — the survival-coverage guard in `load_demographic_rates` now **hard-fails by default** (raises `RuntimeError`) on a missing or horizon-incomplete operative survival table, with an `allow_static_survival=True` opt-out for tests/experiments/constant-mortality scenarios. Implemented as a function parameter, not a config flag, so the locked config sha (`a6e0bfbc2d70be85`) is unchanged.
- **M2** — `aggregate_county_results_to_state` now enforces exactly the configured county set: a filename-horizon match rejects a *complete* stale-horizon county set (e.g. a leftover `2025_2045` run) that duplicate-detection alone would have summed, plus one-file-per-FIPS and expected-set equality (53 for `mode: all`).
- **M3** — `run_mortality_improvement_pipeline` refuses to default to the production write path when `output_dir` is omitted under pytest (`PYTEST_CURRENT_TEST` set) — the precise mistake that truncated production.

\+7 regression tests prove each guard fires on its bad state; full suite **2,267 passed, 5 skipped**. Details and the originating review: [`docs/reviews/2026-06-16-pr25-survival-horizon-review/`](../../reviews/2026-06-16-pr25-survival-horizon-review/) (verdict + hardening handoff).
