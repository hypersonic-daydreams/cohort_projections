# PUB-2026 Release-Readiness Status Memo

| Field | Value |
|-------|-------|
| **Date / time** | 2026-06-17 12:06 CDT |
| **Author** | Claude Code (Opus 4.8) |
| **Audience** | Project owner (PUB-2026), pre-marketing-handoff |
| **Repo state at time of writing** | branch `fix/adr-068-recurrence-guards-and-doc-resync`, HEAD `0c8504f` |
| **Purpose** | Independent assessment of how close the 2026 public projections are to release-ready, what genuinely remains in the repo, and a defined stopping criterion. |
| **Companion** | [release-readiness-checklist.md](../plans/2026-public-projection-release-handoff/release-readiness-checklist.md) |

---

## Context

Over 2026-06-15 / 06-16, three errors were found and corrected in the locked production
run (ADR-068): a ~3× `reference_intl_migration` numerator error, an open-ended 90+
survival error, and a survival-table horizon truncation (the operative table spanned only
2025–2045, so the engine silently fell back to an uncorrected static base for 2047–2055).
All three are corrected, artifacts regenerated, and recurrence guards added. The owner,
having found multiple late-stage errors in a sprawling project, asked for an independent
read of release readiness and an objective checklist with a clear "done" line.

This memo records the findings of that assessment. Findings were **verified against the
working tree**, not taken from the status docs alone.

---

## Bottom line

The **model and the published numbers are locked and internally consistent.** The genuinely
open repo-side work is small: merge the open PR, and make a decision on the one known-stale
documentation item (F4 forward decomposition). Everything else remaining is marketing's
layout/publication responsibility or an explicitly-deferred next-vintage limitation.

---

## What was verified (facts, not doc claims)

1. **Config is the corrected, locked one.** `config/projection_config.yaml` hashes to
   `a6e0bfbc2d70be85` (the locked sha recorded in `build_public_draft_package.py`) and
   carries `reference_intl_migration: 3350.33` — the corrected value, not the ~10051 sum.

2. **Every published artifact reconciles to the corrected run.** The public CSV, Excel
   workbook, the six marketing `.docx`, and both prose files (`draft-public-pdf-copy.md`,
   `methodology_comparison_sdc_2024.md`) carry **799,358 (2025) → 797,298 trough @2027
   (−0.26%) → 898,907 @2055 (+12.45%)**, with 90+ @2055 = 8,172 (internal age-detail; the
   public CSV exposes 85+ = 22,493). Stale headline figures (886,585 / 889,017 / 787,382 /
   9,971) survive only inside explicitly-labeled "superseded" banners and gitignored working
   folders — none as a live public figure.

3. **All six objective release-QA gates PASS.** Recurrence guards from PR #25/#26 are merged;
   the `_check_prose_sync` tripwire passes; test suite reported at 2,274 passed / 5 skipped.

4. **Recurrence is now guarded.** Survival-coverage guard hard-fails for production/public
   runs; exactly-53-county-file assertion; production-write block under pytest;
   migration-horizon guard; 90+ value-pin tests.

## Open items found

### Hard gates (owner-side, block "done")

- **PR #27 is still OPEN.** `fix/adr-068-recurrence-guards-and-doc-resync` carries the
  recurrence guards + doc re-sync and is not yet merged to `master`. Until merged, the
  hardening that prevents these bugs from recurring is not on the mainline.

- **F4-RESYNC is "decision pending — not yet executed."** The F4 forward-decomposition
  figures (the "conservatism stack": CBO-migration ~−23,000, fertility ~−13,000, GQ f=0.75
  @2050) are hand-maintained across ~7 locations and went stale after ADR-068 — the
  CBO-migration lever used the old `reference_intl_migration` denominator. **These stale
  figures DO appear in the public-facing `methodology_comparison_sdc_2024.md` (lines
  437–446), but are explicitly caveated** with a pointer to
  `docs/plans/f4-decomposition-reproducibility.md`. This is therefore a real
  publish-or-defer decision, not a silent leak: either re-run the CBO-migration lever and
  replace the figures, or consciously accept the caveated stale figures for public release.

### Marketing-side (not data defects)

Contact block + live download URLs (placeholders in place); final rendered PDF layout and
chart accessibility; removal of pre-publication watermarks; publication/distribution.

### Explicitly deferred — NOT this release

ADR-068 D3 (race-flat mortality — disclosed in methodology §10, rebuild is next-vintage);
ADR-022 (Proposed, infra); housing-unit method prose update. None block 2026.

---

## Why this assessment can still be trusted given the recent error history

The three corrected bugs all **passed the test suite** — they were data/config truths the
tests did not pin. The accompanying checklist therefore includes an **independent
reconciliation pass** that verifies facts (config sha, a from-scratch reproduction of
898,907@2055, survival-horizon coverage, a stale-number grep) rather than relying solely on
green tests. That pass is the owner's confidence mechanism, and the new guards convert two
of the three failure modes into hard errors.

## Definition of done (stopping criterion)

Repo-side work is complete, and handoff to marketing is warranted, when **all** of:

1. PR #27 merged to `master`.
2. F4-RESYNC either executed (figures re-run + re-synced) **or** consciously deferred with a
   recorded decision that the caveated stale figures are acceptable for public release.
3. The independent reconciliation pass completed — including a from-scratch reproduction of
   799,358 / 797,298@2027 / 898,907@2055 / 90+ = 8,172.
4. Data bisynced.

When those four are true, the remaining open items belong to marketing.

---

*Generated by Claude Code on 2026-06-17. Findings verified against the working tree at
commit `0c8504f`. Numbers reflect the ADR-068-corrected full-horizon locked run
(config sha `a6e0bfbc2d70be85`).*
