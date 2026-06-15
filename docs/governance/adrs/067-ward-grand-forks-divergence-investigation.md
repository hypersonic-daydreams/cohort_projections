# ADR-067: Ward & Grand Forks Divergence Investigation (Corrective Review Before Release)

## Status
Accepted — investigation complete; corrective actions taken in the 2026-06-11
Tier-3 config lock; Ward/GF public narrative to be finalized against final-run
numbers in the Stage-4 sanity review

## Date
2026-06-11

## Scope
Tier-3-ordered corrective investigation into whether the m2026/hybrid method stack
artificially suppresses Ward County (−13.6% 2025–2055 vs SDC 2024 +23%) and Grand
Forks County (−4.7% vs SDC growth) before the PUB-2026 public release. Decision 1
of the finality remediation decision gates (2026-06-11): the user directed a
corrective investigation rather than immediate acceptance of the divergence.

## Context

The 2026-06-10 finality rigor review flagged the Ward sign flip as "the single
number most likely to draw public challenge" and required a written disposition
(release QA Gate 1b). Observed Census PEP components show Ward net migration
negative every year 2020–2025 (−129, −947, −1,044, −951, −483, −392; cumulative
≈ −3,950), while Grand Forks turned positive 2023–2025 (+684, +376, +560) on
international migration — the component the ADR-065 CBO assumption deliberately
reduces. The open question was whether, beyond these observed and assumed
drivers, the method stack (ADR-055 Phase 2 GQ correction × ADR-049/061 college
smoothing × convergence schedule) adds an artificial downward push.

Note: the rigor review carried −8.7% for Grand Forks from ADR-052-era (February)
outputs; the current (2026-05-27) baseline value is −4.7%, with the decline
concentrated after 2045 where long-run convergence dominates.

## Investigation Findings

### F1: Walk-forward evidence-integrity defect (all prior bundles affected)

`walk_forward_validation.py::load_migration_rates_raw()` assumes
`data/processed/migration/residual_migration_rates.parquet` contains raw rates,
then applies each method's boom dampening, male dampening, and college smoothing
on top. But pipeline stage 01a has baked the *production config's* versions of
those same adjustments into that file since the ADR-049 era. Consequences:

- All 15 pre-2026-06-11 benchmark bundles ran on the **2026-02-26 build**
  (champion-vintage: 4 college counties, ages 15–24) — verified by byte-identical
  champion sentinel metrics across the 2026-03-09 and 2026-05-27 bundles, and by
  Drive-trash recovery of the build metadata (see F2).
- Methods stacked their adjustments on the baked ones: the champion was
  effectively double-smoothed (≈0.25 county weight in the 4 baked counties) and
  Bakken boom periods double-dampened (≈0.25/0.16 effective factors).
- Measured distortion (champion, contaminated vs clean inputs): state signed
  bias **−1.23 vs +0.44**; state APE recent-medium **2.56 vs 0.38**. The
  "systematic under-projection bias" that motivated parts of ADR-061 was
  therefore substantially a measurement artifact of the harness, not a property
  of the m2026 method.

Harness remediation — **implemented 2026-06-13** (PUB-2026 finality remediation, F1
harness fix). Interim clean evidence (the F3 matrix below) was produced by temporarily
rebuilding the canonical file with adjustments disabled (GQ correction and ADR-045
reservation recalibration retained as data corrections), running the matrix, then
restoring the production build.

The permanent fix removes the file dependency entirely rather than persisting a raw
artifact. `load_migration_rates_raw()` now **recomputes** the raw base in-process from
the harness's own population snapshots and survival source, at the default GQ fraction —
it delegates to the existing `recompute_migration_with_gq_override(snapshots,
_DEFAULT_GQ_CORRECTION_FRACTION)`. Rationale for recompute-not-file: the production
pipeline (`01a`) and the harness use **different** survival sources
(`survival_rates_sdc_2024_by_age_group.csv` vs `…_full.csv` collapsed) and **different**
population loaders (`assemble_period_populations` vs `load_population_snapshot`), so any
01a-emitted file would not match the harness's own forward-projection inputs and would
re-introduce a (subtler) default-vs-override inconsistency. Recompute guarantees the
default rate base is byte-identical to the GQ-override path (verified: max abs diff 0.0)
and consistent with the harness's projection survival source. Verified: the recomputed
base differs from the adjustment-baked production file in 8,543/9,540 rows (e.g. Cass
20–24 raw +0.116/+0.125 vs baked +0.028/+0.015), confirming the adjustments are no longer
double-applied. The production `residual_migration_rates.parquet` is byte-unchanged
(sha256 `11686d1b…`); the engine change in `residual_migration.py` is documentation-only.
Regression tests added in `tests/test_analysis/test_upstream_param_injection.py`
(`TestLoadMigrationRatesRaw`): recompute-at-default-fraction, snapshot-autoload, and a
guard that the loader never reads a rates parquet.

**Known limitation (documented, not fixed):** the harness has no per-method ADR-045
reservation-recalibration knob, so the recomputed raw base — like the pre-existing
GQ-override path — omits PEP recalibration for the three reservation counties (Benson,
Sioux, Rolette). The one-off F3 matrix retained PEP as a data correction; the permanent
recompute does not. This affects only those three small counties (never decision
sentinels — the disposition sentinels were Grand Forks, Cass, Ward, Williams), so it does
not alter any ADR-061 conclusion. Adding a PEP data-correction step to the recompute is a
possible future harness enhancement.

### F2: Pipeline data-loss event (~2026-06-01) and production provenance

A bisync conflict event around 2026-06-01 removed the canonical residual
migration and convergence outputs from `data/processed/migration/` on both
machines (conflict copies recovered from Google Drive trash with `..path1` /
`..path2` suffixes). Forensic comparison of recovered metadata:

| Build | College config baked | Used by |
|---|---|---|
| 2026-02-26 09:03 | 4 counties, ages 15–24 | 2026-02-26 three-scenario run; all 2026-03-09 and 2026-05-27 benchmark bundles (sandbox-pinned) |
| 2026-05-27 14:41 | 12 counties, ages 15–29 | 2026-05-27 production baseline run (14:44) |
| 2026-06-11 regeneration | 12 counties, ages 15–29 | identical to the 2026-05-27 build except timestamp |

This **verifies the 2026-05-27 public baseline is the true config hybrid**
(rates consistent with `projection_config.yaml` at run time), resolving the
rigor review's open question about hybrid provenance, and confirms today's
regeneration restored the exact production input state.

**F2 addendum — silent mortality-file regression.** The same 2026-06-01 sync
event also *replaced* `data/processed/mortality/nd_adjusted_survival_projections.parquet`
with a stale 21-year (2025–2045) copy from the other machine — a silent
overwrite, not a loud deletion. Any rerun on that state completes without
error but freezes mortality improvement after 2045 (first divergent
population year 2047; −16,719 state total by 2055). The file was regenerated
via `01c_compute_mortality_improvement.py` (2025–2055, 31 years). With the
full regenerated pipeline state (01a + 01b + 01c), a reference engine rerun
**reproduces the 2026-05-27 production baseline exactly** (state totals and
Ward/GF trajectories to the person at 2025/2040/2050/2055) — the production
run is reproducible and its input state is verified. QA follow-up: Gate 1
gains an input-coverage check (mortality-improvement years must span the
projection horizon; rates-metadata adjustments must match the production
config).

### F3: Clean raw-base evidence matrix (2026-06-11 bundles)

Five challenger variants vs the true champion, all on raw rates (walk-forward
origins 2005–2020; sentinel county MAPE %, state APE %, signed bias pp):

| Variant | Ward | Grand Forks | Cass | Williams | County overall | Urban/college | State APE short/med | Bias |
|---|---|---|---|---|---|---|---|---|
| Champion m2026 | 13.93 | 11.12 | 9.35 | 22.48 | 8.72 | 10.15 | 1.01 / 0.38 | +0.44 |
| m2026r1 college-fix (D1+D4, blend 0.5) | 13.24 | 7.39 | 7.65 | 23.38 | 8.64 | 8.76 | 1.04 / 1.46 | +0.97 |
| blend 0.7 | 12.20 | 4.65 | 4.33 | 22.98 | 8.48 | 6.72 | 1.42 / 2.64 | +1.64 |
| blend 1.0 (no smoothing) | 10.23 | 4.22 | 2.74 | 22.48 | 8.35 | 5.31 | 2.09 / 4.84 | +2.81 |
| Williams-out (fix minus 38105) | 13.24 | 7.39 | 7.65 | 22.48 | 8.62 | 8.76 | 1.11 / 1.62 | +1.08 |
| GQ fraction 0.75 | 13.22 | 7.39 | 7.78 | 23.45 | 8.54 | 8.81 | 1.00 / 1.32 | +0.89 |

Mechanism answers:

1. **GQ Phase-2 correction (ADR-055) is NOT a Ward/GF suppressor — refuted.**
   Reducing the fraction to 0.75 leaves Ward/GF sentinels unchanged
   (13.22/7.39 vs 13.24/7.39). Mechanism: their institutional GQ (Minot AFB,
   UND dorms) is approximately level across snapshot years, so the subtraction
   cancels out of the period *differences* that drive residual rates. Only
   counties with GQ *changes* move (Cass: NDSU dorm growth). The fraction's
   true effect is mild and broad: best state APE-short (1.00), improved rural
   (7.30) and overall (8.54) MAPE. An earlier same-day run that suggested a
   ~5pp Grand Forks improvement from GQ 0.75 was a contamination artifact
   (mixed rate bases) and is superseded by this matrix.
2. **College smoothing changes are calibration, not artifact.** The D1+D4
   extension lowers Ward/GF forward paths *and improves their backtest
   accuracy* (GF 11.12→7.39, Ward 13.93→13.24) — the counties were being
   over-projected, and the divergence from SDC tracks observed 2020–2025 data.
   However, every step toward county-pure rates trades state-level accuracy
   for county-level accuracy (state APE-medium 0.38 → 1.46 → 2.64 → 4.84;
   bias drifts over-projection-ward to +2.81 at blend 1.0). The blend/extension
   choice is therefore a genuine state-vs-county accuracy trade-off — a Tier-3
   config decision (ADR-061 disposition), not a defect.
3. **Williams County inclusion (D4) is refuted by its own evidence.** Removing
   Williams restores its error to champion level (23.38 → 22.48) at no material
   cost elsewhere — consistent with ADR-061's flagged double-dampening risk and
   its below-threshold enrollment ratio (WSC 1.5% vs 2.5% threshold).

### F4: Forward decomposition (completed 2026-06-11)

One-factor forward runs against the verified production state (reference =
2026-05-27 baseline, reproduced exactly). Effects of turning each factor
off/on, state totals and county 2055 values:

| One-factor variant | State 2050 | State 2055 | Ward 2055 | Grand Forks 2055 |
|---|---|---|---|---|
| Reference (production baseline) | 862,723 | 876,479 | 58,985 | 70,974 |
| CBO migration adjustment off | +22,948 | +23,262 | +1,583 | +1,827 |
| −5% fertility adjustment off | +13,032 | +16,231 | +1,100 | +1,585 |
| ADR-061 D3 convergence hold on (10→15 yr) | +9,080 | +11,677 | +674 | +1,456 |
| GQ correction fraction 0.75 | +3,048 | +3,619 | +304 | +271 |

Attribution:

- **Ward** declines 9,248 from 2025 to 2055. All four levers combined move it
  only ≈ +3,660 — the majority of the decline survives every method variant.
  Ward's trajectory is dominated by the observed 2020–2025 out-migration
  signal in the rates, consistent with F3: signal, not artifact.
- **Grand Forks** declines 3,527. The CBO assumption alone accounts for ≈52%
  (+1,827) and the convergence schedule ≈41% (+1,456) — GF's decline is
  substantially the disclosed federal-immigration assumption plus the
  long-run convergence stance, not county rate calibration. The public
  narrative should attribute it accordingly.
- **State level**: the CBO adjustment is the dominant conservative lever
  (−23k at 2050), fertility −13k, D3 convergence −9k, GQ fraction −3k. These
  magnitudes feed the ADR-065 defensibility memo (stacked-conservatism
  stance) and the ADR-061 D3 disposition.

## Decision

**Decided 2026-06-11 (Tier-3), executed in the config lock** (decision record:
`docs/reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md`, Approved):

1. **No artificial suppression found.** Ward's decline is dominated by the
   observed 2020–2025 out-migration signal (F4: all four method/assumption
   levers combined move Ward 2055 by only +3,660 against a 9,248 projected
   decline). Grand Forks' decline is substantially the disclosed ADR-065 CBO
   assumption (≈52%) plus the long-run convergence stance (≈41%) — assumption
   and stance, not rate artifact. The college-smoothing changes *improve*
   backtest accuracy in exactly these counties.
2. **Corrective actions taken** (via ADR-061 disposition): Williams removed
   from the college smoothing list; GQ correction fraction calibrated to 0.75;
   blend factor retained at 0.5; D3 rejected for this release. Locked profile:
   `m2026r1` / `cfg-20260611-production-lock`, promoted to `county_champion`.
3. **Ward/GF disposition: accepted divergence, strengthened by this
   investigation.** The public narrative (Stage-4/5) cites: observed 2020–2025
   PEP out-migration for Ward; the disclosed federal-immigration assumption
   for Grand Forks; GQ anchors (Minot AFB, MISU, UND) held constant by design;
   and this ADR's completed mechanism checks. Final wording against final-run
   numbers in the Stage-4 sanity review (Gate 1b).
4. **Harness contract fix — implemented 2026-06-13.** `load_migration_rates_raw()`
   now recomputes the raw base in-process (delegating to
   `recompute_migration_with_gq_override` at the default GQ fraction) instead of reading
   the adjustment-baked production file; default-vs-override consistency is now exact
   (max abs diff 0.0) and regression-tested. See the F1 section for the recompute-not-file
   rationale and the documented reservation-county (PEP) limitation. The QA input-coverage
   gate was added to `release-qa-checklist.md` (Gate 1).

## Consequences

### Positive
- The public Ward/GF narrative can cite a completed adversarial investigation
  rather than an assumption of correctness.
- The benchmark evidence base is now clean; all future promotion decisions
  (SOP-003) can rely on the raw-base matrix.
- The 2026-05-27 production run's provenance is verified.

### Negative
- All pre-2026-06-11 benchmark bundle results are quantitatively unreliable
  (directionally informative at best); prior pending experiment dispositions
  (EXP-A/B/E/I/J/K and the 2026-05-27 search bundles) must be re-read against
  the raw-base matrix.
- Walk-forward harness required a contract fix before further benchmark campaigns;
  this was implemented 2026-06-13 (F1 section, Decision item 4). Future campaigns run on
  the recomputed raw base. Reservation-county (PEP) recalibration is omitted from the
  harness base — a documented, immaterial-to-disposition limitation.

## References

- Raw-base bundles (2026-06-11): `rawbase-college-fix-v1`, `rawbase-blend-70`,
  `rawbase-blend-100`, `rawbase-williams-out`, `rawbase-gq-fraction-075` in
  `data/analysis/benchmark_history/`
- Recovered build metadata: Drive trash `residual_migration_metadata.json..path1`
  (2026-05-27) / `..path2` (2026-02-26)
- `docs/reviews/2026-06-10-pub-2026-finality-rigor-review.md` (blocking item W)
- `docs/plans/2026-public-projection-release-handoff/finality-remediation-plan.md`
  (Stage 4.2 decision gate)
- ADR-040, ADR-045, ADR-049, ADR-052, ADR-055, ADR-061, ADR-065
- `docs/REFERENCE-cross-machine-db-state.md` (workspace root; bisync conflict
  class of failures)

## Revision History

- **2026-06-11**: Initial version — investigation findings F1–F3 recorded; F4
  and final Decision pending.
- **2026-06-13**: F4 forward decomposition completed; final Decision recorded; Status
  Accepted. F1 harness contract fix implemented (recompute-in-process; default==override
  exact; regression tests added; production rates file byte-unchanged). Reservation-county
  PEP limitation documented.
