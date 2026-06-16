# Session Summary — Survival-Horizon Truncation Found & Fixed (PUB-2026 / ADR-068)

**Date:** 2026-06-16
**Author:** Claude Code (round-2 remediation session)
**Branch / PR:** `docs/ref-intl-sum-vs-average-finding` / PR #25 (the ADR-068 correction PR)
**Anchor commit:** `e719518`
**Status:** Remediation complete; production numbers changed; PR #25 pushed; PR-framed re-review **done**
(GO-WITH-FIXES) → recurrence-hardening punch list handed off (see
[`2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md`](2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md)).

This memo is the human-readable **index** for the session. The durable records are the
[ADR-068 amendment](../governance/adrs/068-ref-intl-numerator-and-open-ended-survival-correction.md)
(governance), the [corrected-run QA verification](2026-06-16-corrected-run-qa-verification.md)
(verification), and commit `e719518` (the change). This memo ties them together and records the
decisions.

## TL;DR

Starting from the round-2 remediation handoff (whose premise was "numbers are final, no model rerun"),
the survival-coverage QA gate failed and exposed a **real numeric defect**. The published "final"
figures were biased for 2047–2055. Corrected, full-horizon production numbers:

| | Published (buggy) | **Corrected (final)** |
|---|---:|---:|
| 2055 | 886,585 (+10.91%) | **898,907 (+12.45%)** |
| 90+ @2055 | 9,971 | **8,172** |
| 2050 | 877,818 | 883,225 |
| 2025 base / 2027 trough | 799,358 / 797,298 | **unchanged** |

Years 2025–2046 are identical; only 2047–2055 moved.

## The arc (how we got here)

1. **Handoff said "final."** The 2026-06-15 ADR-068 run (ref_intl 10,051→3,350.33; 90+ → T₉₁/T₉₀) had
   passed an independent GPT-5.5 Pro round-2 review (GO). The handoff scoped a text/labeling/QA +
   artifact-regeneration pass — explicitly *no model rerun*.
2. **A QA gate failed.** Verifying §D's "survival table spans 2025–2055", the operative table on disk
   spanned only **2025–2045**. The engine (`core/cohort_component.py::_get_survival_rates`) silently
   falls back to the uncorrected static-base survival for any missing year, so for the projection
   steps 2047–2055 the run used neither the NP2023 trajectory nor the ADR-068 90+ correction.
3. **User authorized a verification rerun.** Regenerating survival to the full horizon (2025–2045
   overlap byte-identical, max diff 0.0) and re-running stage 02 produced a **nonzero diff that begins
   exactly at 2047** — airtight proof the table-coverage gap, not anything else, was responsible.
4. **Root cause: a test was clobbering production data.**
   `tests/test_data/test_mortality_improvement.py::test_full_pipeline_produces_valid_output` called the
   mortality pipeline with `projection_horizon=20` and no output dir; the pipeline hard-coded its
   output to the **production** survival path. So **every `pytest` run overwrote the production survival
   table with a truncated 2025–2045 horizon.** Two integration tests even hard-coded 2045 as their
   expectation, hiding the regression.
5. **Adopted + propagated + regenerated** (user decision), with a guard so it cannot recur silently.

## Decisions on record

- **Adopt the corrected full-horizon numbers** (898,907 / 8,172) as production. *(user)*
- **Re-review: DONE** — a PR-framed GPT-5.5 Pro review (live+background, `xhigh`, 599k input, $23.33)
  returned **GO-WITH-FIXES**: it confirmed the root cause, the corrected numbers (898,907 / 8,172), and
  the GQ-inclusive summary recompute as sound, and raised three recurrence-hardening fixes (make the
  coverage guard hard-fail; assert exactly 53 current county files; block the mortality pipeline from
  defaulting to a production write under pytest) + minors. These are handed off in
  [`2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md`](2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md).
  No numeric change required.
- **PR strategy:** everything stays in **PR #25** — it already carries the whole ADR-068 model
  correction (incl. the engine fix `4c38f6d`, which is not on master). Splitting the survival-horizon
  fix into a separate PR was rejected (it would fragment one ADR-068 correction and stack on #25
  anyway). Separating the unrelated GPT-5.5 Pro **API-reference guide** (`docs/guides/gpt-5.5-pro-api-reference.md`)
  into its own PR was also considered and **declined** — not worth the git-history cleanup. (Retitling
  #25 away from `docs:` remains an optional nicety.)
- **Push: DONE** — `e719518` + `fcb0432` pushed to PR #25 (origin at `fcb0432`), so the PR canonically
  reflects the reviewed state. *(user authorized)*

## What changed (commit `e719518`)

**Root-cause + prevention**
- `cohort_projections/data/process/mortality_improvement.py` — injectable `output_dir` (production
  default); horizon default 20→30; N5 metadata (open-age source + T₉₁/T₉₀ targets).
- `tests/test_data/test_mortality_improvement.py` — pass `output_dir=tmp_path` (never touch production).
- `scripts/pipeline/02_run_projections.py` — **survival-horizon coverage guard** (warns when the
  operative table doesn't span `base_year..end_year`).
- `tests/test_integration/test_census_method_validation.py` — two horizon assertions now derive from
  config / the actual table instead of hard-coding 2045.

**Round-2 review remediation** (handoff §A–§D)
- §A: methodology §4.3 (90+ T₉₁/T₉₀), §3.1 fertility pooling-label, new §4.6 mortality provenance,
  §10.1 household-basis components, stale numbers purged / dated memos banner-superseded.
- §B: county `_summary.csv` GQ-inclusive (B1); aggregation glob + 53-FIPS assertion (B2); exact
  residual verified (B3).
- §C: methodology §10.8 disclosures + N3 race-flat migration (§10.7).
- §D: all public artifacts regenerated (workbook/CSV/charts/pyramids/6 marketing docx) → 898,907.
- ADR-068 amended; config sha256(16) `cca42fb42be76680` → `a6e0bfbc2d70be85` (comment-only delta).

## Verification

- `2,260 passed, 5 skipped`; **production survival stays 2025–2055 after a full pytest run** (the clobber
  is fixed); state = Σcounty exact (~1e-10); 53 unique county FIPS; base 799,358; survival spans
  2025–2055; coverage guard silent. ruff + mypy clean on all changed files.

## Open / next

- **Recurrence-hardening punch list (GO-WITH-FIXES)** — ✅ **DONE 2026-06-16.** All 3 majors
  (hard-fail coverage guard with `allow_static_survival` opt-out; exactly-53 current county files +
  filename-horizon match; block production write under pytest) and 3 minors implemented, +7 regression
  tests, full suite 2,267 passed; no projection number moved. Majors committed `c1e921a`; minors + doc
  wording in the follow-up commit. See
  [`2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md`](2026-06-16-pr25-survival-horizon-review/HANDOFF-pr-review-hardening.md)
  (marked implemented) and the ADR-068 amendment's recurrence-hardening note.
- Optional: retitle #25 off `docs:`; a confirming re-review of the (small) hardening diff.
- `./scripts/bisync.sh` to sync the regenerated survival/projection data (gitignored) to the other
  machine; the committed `marketing-ready/` artifacts travel with git.
- Marketing layout / publication (SDC/marketing-owned).
