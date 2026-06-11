# ADR-067: Ward & Grand Forks Divergence Investigation (Corrective Review Before Release)

## Status
Proposed — investigation in progress; forward decomposition pending (slots marked below)

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

Harness remediation (raw artifact contract for walk-forward) is tracked as a
follow-up ADR; interim clean evidence was produced by temporarily rebuilding the
canonical file with adjustments disabled (GQ correction and ADR-045 reservation
recalibration retained as data corrections), running the matrix below, then
restoring the production build.

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

### F4: Forward decomposition (PENDING — runs in progress 2026-06-11)

One-factor forward runs (state totals 2040/2050/2055 and Ward/GF trajectories)
for: CBO adjustment off; −5% fertility off; convergence medium hold 10→15 years
(ADR-061 D3); GQ fraction 0.75; plus a reference rerun doubling as a
reproducibility check of the 2026-05-27 production run. Results to be recorded
here, including the share of Ward/GF's decline attributable to the disclosed
CBO assumption vs rate calibration vs convergence schedule.

## Decision

Pending completion of F4 and the ADR-061 disposition (Decision 2 of the
remediation gates). Preliminary direction supported by F1–F3:

1. No method artifact was found that artificially suppresses Ward or Grand
   Forks beyond (a) evidence-supported rate calibration that backtests
   *better* in exactly these counties, and (b) the deliberate, disclosed
   ADR-065 CBO assumption (dominant for Grand Forks; F4 quantifies).
2. Corrective actions that DO follow from the investigation land in the
   ADR-061 config lock: remove Williams from the college list; disposition the
   GQ fraction and blend factor on the clean matrix; decide D3 on F4 evidence.
3. The Ward/GF public disposition is then an accepted-divergence rationale
   strengthened by this investigation's documented mechanism checks.

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
- Walk-forward harness requires a contract fix (follow-up ADR) before further
  benchmark campaigns.

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
