# F4 Forward Decomposition — Reproducibility, Staleness, and Re-run Procedure

| Attribute | Value |
|-----------|-------|
| Status | OPEN — decision pending (process improvement) + outstanding re-run |
| Opened | 2026-06-16 (during PR #27, ADR-068 doc re-sync) |
| Owner | TBD |
| Related | ADR-067 (F4), ADR-065 defensibility memo, ADR-068, `methodology_comparison_sdc_2024.md` §4.2 |

## 1. Why this doc exists

The **ADR-067 F4 forward decomposition** — the four "conservatism-stack" levers that explain
why the public baseline sits below the SDC path — is hand-maintained, appears in ~7 places across
3 documents, requires runs that partly clobber production data, and **silently goes stale on every
re-lock of the baseline.** The ADR-068 correction is the proof: §4.2 of the methodology-comparison
doc had to be caveated in PR #27 because its CBO-migration figure (−23k @2050) is now wrong and
nobody had a one-command way to regenerate it.

This doc (a) records the exact re-run procedure, (b) diagnoses the brittleness, and (c) lays out
improvement options so we can decide on a structural fix rather than repeatedly redo the manual dance.

## 2. What F4 is

A **one-factor forward decomposition**: take the locked production baseline as the reference, then
turn each of four levers off/on **one at a time**, and record the change in the state total at 2050
and 2055 (plus Ward and Grand Forks 2055 for the county narrative). Source of the canonical table:
ADR-067 §F4; restated in the ADR-065 defensibility memo §2.

Original table (reference = pre-ADR-068 baseline, 876,479 @2055):

| Lever | Direction | Δ State 2050 | Δ State 2055 | Config toggle | Pipeline stage |
|---|---|---|---|---|---|
| CBO front-loaded migration ramp | remove → higher | +22,948 | +23,262 | `scenarios.baseline.migration.schedule` 2025–2029 → all `1.00` (zero reduction) | **projection (02) only** |
| −5% fertility adjustment | remove → higher | +13,032 | +16,231 | `scenarios.baseline.fertility` `"-5_percent"` → `"constant"` | **projection (02) only** |
| Convergence hold extension (ADR-061 D3) | apply → higher | +9,080 | +11,677 | `…migration.interpolation.convergence_schedule.medium_hold_years` `10` → `15` | **upstream — Phase 2 convergence regen** |
| GQ correction fraction 1.0 → 0.75 | apply → higher | +3,048 | +3,619 | `rates.…gq_correction.fraction` `0.75` (vs `1.0`) | **upstream — Phase 1 residual-migration regen** |

Config line references (current `config/projection_config.yaml`): fertility `:265`; migration schedule
`:267–272`; `reference_intl_migration` `:272`; `medium_hold_years` `:250`; GQ `fraction` `:202`.
Toggle implementation: fertility `-5%` is applied at projection time in
`cohort_projections/core/fertility.py:238` (`*0.95`); the CBO migration reduction in
`cohort_projections/core/migration.py:220`. The D3 and GQ levers feed **upstream** rate parquets
(`data/processed/migration/convergence_rates_by_year.parquet` and `residual_migration_rates.parquet`),
which the projection stage merely consumes.

Original run outputs (not committed; data): `data/projections/sensitivity_20260611/{ref,cbo_off,fert_off,d3_hold15,gq075_fwd}/`.

## 3. What ADR-068 actually changed (so we know what must be re-run)

Only **one** lever moves with the ADR-068 correction:

- **CBO migration ramp** — the reduction is `reduction_rate = ref_intl·(1−factor)/ref_pop`, so it
  scales ~linearly with `reference_intl_migration`, which dropped 10,051 → 3,350.33 (×0.333). The
  ~+23k @2050 effect should fall to **roughly +7.6k @2050** (`22,948 × 0.333 ≈ 7,649`, estimate —
  the true value needs the run, because of horizon compounding/interaction).
- **−5% fertility, GQ fraction, D3 hold** are independent of both the `ref_intl` numerator and the
  90+ survival fix. They shift only marginally from the corrected baseline's modestly higher base
  (898,907 vs 876,479). The 90+ survival correction does not touch any of these levers.

So the figure that is materially wrong today is the CBO-migration component; the others are
approximately carried-forward-correct.

## 4. The re-run procedure (against the corrected baseline, config sha `a6e0bfbc2d70be85`)

Reference run already exists on disk: `data/projections/baseline/{state,county}/…`.

**Safe / fast (projection-stage, isolated `output_dir`, ~2 min each, no production data touched)** —
this is exactly the mechanism the refintl sensitivity bundle used (`pipeline.projection.output_dir`
pointed at a bundle subdir):
1. `cbo_off` — copy locked config, set the migration schedule 2025–2029 all to `1.00`, set
   `pipeline.projection.output_dir` to a bundle subdir, run the projection stage.
2. `fert_off` — copy locked config, set `fertility: "constant"`, isolated `output_dir`, run.

**Unsafe / slow (upstream regen writes to fixed `data/processed/` paths — needs backup/restore)**:
3. `d3_hold15` — set `medium_hold_years: 15`, regenerate Phase 2 convergence, run projection.
4. `gq100` (or framed `gq075` vs `1.0`) — set GQ `fraction: 1.0`, regenerate Phase 1 residual
   migration **and** Phase 2 convergence, run projection.

For each variant, diff state totals at 2050/2055 and Ward/GF 2055 against the reference.

## 5. Why this is brittle (the actual problem)

1. **No single source of truth.** The F4 numbers are hand-typed into ≥7 locations: ADR-067 §F4
   table + attribution + Decision (Ward +3,660 / GF 52%/41%), the ADR-065 defensibility-memo table,
   methodology-comparison §3.2 Ward & Grand Forks narratives, and methodology-comparison §4.2. A
   re-lock has to chase all of them; PR #27 already left the §3.2 Ward "~9,250 decline" F4 figure
   stale (corrected decline is ~8,250) precisely because it is a separate hand-typed copy.
2. **They go stale silently on every re-lock.** There is no guard, gate, or generated artifact that
   notices when the locked config sha changes. ADR-068 is the case study.
3. **No committed harness.** `sensitivity_20260611/` holds the *outputs* but not the *recipe*; the
   toggle/stage mapping in §2–4 above had to be reconstructed by archaeology this session.
4. **Two of four levers clobber production data.** The Phase 1/Phase 2 stages write to fixed
   `data/processed/` paths with no output-isolation, so reproducing `d3_hold15`/`gq100` safely needs
   a manual backup/restore — error-prone, and risky to run near a publication freeze.
5. **The projection-stage custom-config invocation is itself undocumented** (how the bundle pointed
   stage 02 at a custom config + `output_dir`).

## 6. Improvement options (decide here)

- **A. Committed decomposition harness** — `scripts/analysis/run_f4_decomposition.py`: deep-copy the
  locked config, apply each lever's toggle, isolate all outputs under a bundle dir, run the needed
  stages, and emit one `f4_decomposition.csv` (state 2050/2055 + Ward/GF 2055 deltas) as the single
  source of truth. Mirrors the existing `build_comparison.py` pattern but for the 4-lever sweep.
- **B. Output-path parameterization for upstream stages** (the structural fix for §5.4) — let the
  Phase 1 residual-migration and Phase 2 convergence stages accept an output-root override so
  sensitivity variants write to a bundle instead of `data/processed/`. Removes the backup/restore
  risk and makes `d3_hold15`/`gq100` as safe as the projection-stage variants.
- **C. Single-source the numbers** — have the methodology doc + ADRs reference the generated
  `f4_decomposition.csv` (or an auto-inserted table snippet), and add a staleness tripwire / release-QA
  gate that fails when the table's recorded config sha ≠ the locked config sha — mirroring the
  `_check_prose_sync()` tripwire already added for the public PDF in `build_marketing_docx.py`.
- **D. Checklist hook** — add "re-run F4 decomposition + re-sync the dependent doc figures" to the
  re-lock / release-QA checklist so it can't be forgotten on the next correction.

Minimum to unblock publication-quality docs **without** the structural fix: do §4's safe path
(`cbo_off` + `fert_off`) to get the corrected CBO-migration figure, carry D3/GQ forward with a note,
and update §4.2 + the ADR-067 F4 table + the §3.2 Ward/GF F4 figures. The §4.2 caveat added in PR #27
already points readers at this gap in the meantime.

## 7. Outstanding work tracker

- [ ] Decide improvement scope (A–D above).
- [ ] Re-run the CBO-migration lever (at minimum) against `a6e0bfbc2d70be85`; update §4.2.
- [ ] Re-sync all F4 figure copies (§5.1 list), incl. the §3.2 Ward "~9,250 → ~8,250" decline.
- [ ] If B is adopted, the full 4-lever table can be regenerated cleanly and the §4.2 caveat removed.
