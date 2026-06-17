# PUB-2026 Release-Readiness Checklist

| Field | Value |
|-------|-------|
| **Created** | 2026-06-17 12:06 CDT by Claude Code (Opus 4.8) |
| **Companion memo** | [2026-06-17-pub-2026-release-readiness-status.md](../../reviews/2026-06-17-pub-2026-release-readiness-status.md) |
| **Scope** | Final repo-side gate before marketing handoff. Complements the gate-by-gate [release-qa-checklist.md](./release-qa-checklist.md) (already executed and PASSED); this checklist tracks only what remains. |
| **Locked run** | config sha `a6e0bfbc2d70be85`; 799,358 (2025) → 797,298 trough @2027 → **898,907 @2055**; 90+ @2055 = 8,172. |

> Work top to bottom. You are **done** (and may hand off to marketing) when Sections 1–3
> are fully checked. Section 4 is marketing's. Section 5 is explicitly out of scope for 2026.

---

## 1. Hard gates — must close before handoff

- [ ] **Merge PR #27** (`fix/adr-068-recurrence-guards-and-doc-resync`) to `master`. Carries
      the recurrence guards (migration-horizon + 90+ pin) and the doc re-sync.
- [ ] **Resolve F4-RESYNC** — pick ONE and record it:
  - [ ] **Re-run + re-sync:** re-run the CBO-migration lever against the ADR-068-corrected
        baseline (see `docs/plans/f4-decomposition-reproducibility.md`) and replace the stale
        figures in all ~7 locations, including `methodology_comparison_sdc_2024.md` lines
        ~437–446, **OR**
  - [ ] **Conscious defer:** record a decision that the caveated stale F4 figures
        (CBO-mig ~−23,000 / fertility ~−13,000 / GQ f=0.75) are acceptable for public release,
        given they are flagged with a pointer to the re-run procedure.

## 2. Independent reconciliation pass — verify facts, not just green tests

> These exist because the three ADR-068 bugs all passed the suite. Run them fresh.

- [ ] `pytest` clean — expect ~2,274 passed, 5 skipped (alert if skips > 5).
- [ ] Confirm `config/projection_config.yaml` still hashes to `a6e0bfbc2d70be85` and
      `reference_intl_migration` = 3350.33.
- [ ] **From-scratch reproduction** from the locked config (not a cached artifact) reproduces
      799,358 / 797,298@2027 / **898,907@2055** / 90+ = 8,172.
- [ ] Survival table on disk spans the **full 2025–2055** horizon (the truncation bug); the
      hard-fail guard does not trip on the production run.
- [ ] Grep public CSV / workbook / `.docx` / both prose files for `886,585`, `889,017`,
      `787,382`, `9,971`, `13,707` — every hit sits inside an explicit "superseded" banner.
- [ ] State = Σ counties = 0 residual; region = Σ member counties; exactly 53 county files.

## 3. Finalize

- [ ] Update `DEVELOPMENT_TRACKER.md`: PUB-2026 repo-side work closed; F4-RESYNC outcome recorded.
- [ ] `./scripts/bisync.sh` (data synced before handoff).
- [ ] Confirm the marketing-ready package (`marketing-ready/`) reflects the post-PR-#27 state.

---

## 4. Marketing's responsibilities — NOT data defects, do not block on these

- [ ] Contact block + live download URLs (PLACEHOLDER markers already in `draft-public-pdf-copy.md`).
- [ ] Final rendered PDF layout; chart accessibility (color-independent encodings, legible
      labels/legends, logical reading order).
- [ ] Remove pre-publication watermarks from chart PNGs and workbook headers.
- [ ] Publish PDF, make Excel/CSV download links live, distribute.

## 5. Explicitly deferred — out of scope for the 2026 release

- [ ] *(next vintage)* ADR-068 D3: rebuild survival pipeline to be race-specific (currently
      race-flat; disclosed in methodology §10).
- [ ] *(infra)* ADR-022 unified documentation / reproducibility (Proposed).
- [ ] *(next doc pass)* Housing-unit method (ADR-060) methodology prose update.

---

*Maintained alongside the 2026-06-17 status memo. If the locked run changes, every figure in
this checklist and the public artifacts must be re-verified.*
