# PUB-2026 Release QA — Corrected Full-Horizon Run Verification

**Date:** 2026-06-16
**Reviewer:** Claude Code (round-2 remediation, ADR-068 survival-horizon amendment)
**Run under QA:** corrected full-horizon production run — `m2026r1` / `cfg-20260611-production-lock`
(functional config unchanged; ADR-068 deltas are comment-only + the survival build), config
`projection_config.yaml` sha256(16) `a6e0bfbc2d70be85`.
**Supersedes:** [`2026-06-13-release-qa-signoff.md`](2026-06-13-release-qa-signoff.md) (run against the
since-superseded 2026-06-13 locked run).

## Why a fresh verification

While clearing the round-2 review punch list, the §D survival-coverage gate ("survival table spans
2025–2055") failed: the operative survival table on disk spanned only 2025–2045, so the engine
silently fell back to the uncorrected static-base survival for the projection steps 2047–2055
(ADR-068 Amendment). A controlled re-run with the survival table regenerated to the full horizon
isolated the effect (2025–2046 identical; divergence begins exactly at 2047). The corrected
full-horizon numbers are now the production set, and a coverage guard
(`02_run_projections.py::load_demographic_rates`) prevents the silent fallback from recurring.

## Authoritative numbers (corrected full-horizon run)

| Quantity | Value |
|----------|-------|
| 2025 base | 799,358 (= Σ 53 county `POPESTIMATE2025`) |
| Near-term trough | 797,298 in 2027 (−0.26%) |
| 2050 | 883,225 (+10.49%) |
| 2055 | **898,907 (+12.45%)** |
| 90+ population @2055 | **8,172** |
| Prior figures (superseded) | 889,017 @2055 (2026-06-13 locked); 886,585 / 90+ 9,971 (2026-06-15, survival truncated) |

## Objective gates — re-verified 2026-06-16

| Gate | Check | Result |
|------|-------|--------|
| 1 — Method & Data Lock | config sha256(16) `a6e0bfbc2d70be85`; operative survival spans 2025–2055 (31 yrs); base 799,358 | **PASS** |
| 1b — Plausibility / aggregation | state = Σ county (max abs diff 1.16e-10); Σ region = state (2.33e-10); 53 unique county FIPS, one file each; trough = CBO lever, near-flat | **PASS** |
| 2 — Public Download QA | consolidated CSV exactly **1,922 rows** (1 scenario × 62 geographies × 31 years); levels = {state, region, county}; **no place rows**; baseline only; 18 columns; workbook has all required sheets incl. README, State Key Years, State Age-Sex Detail, Data Dictionary | **PASS** |
| Survival coverage | operative `nd_adjusted_survival_projections.parquet` spans 2025–2055; age-90 survival ≈ 0.778 M / 0.806 F @2025, improving; guard silent | **PASS** |
| Components labeling (A4) | public workbook README states totals include GQ held constant and any components are household-basis; methodology §10.1 carries the household-basis caveat | **PASS** |

## Artifacts regenerated against the corrected run (2026-06-16)

- Public consolidated workbook + CSV (`build_public_draft_package.py`) — CSV state 2055 = 898,907.01;
  no stale 886,585. Pyramid + key charts + 2024 SDC reference PDF refreshed.
- Detail workbooks (`build_detail_workbooks.py`); pyramid explorer (`build_pyramid_explorer.py`).
- Six marketing `.docx` (`build_marketing_docx.py`) from the updated markdown + regenerated CSV —
  layout figures show 899,000 @2055 (+12.5%), trough ~797,000 @2027, refreshed region/county tables.

## Out of scope (marketing / editorial)

Gates 3–6 (rendered-PDF content/layout, language polish on the rendered PDF, delivery & publication)
remain marketing/SDC actions on the rendered artifacts. The objective data/content gates above are
the engineering deliverable.

## Note on re-review

The independent round-2 GPT-5.5 Pro review (verdict GO) predates this survival-horizon amendment; a
re-review against the corrected full-horizon run is a **deferred** decision (user: "decide later").
