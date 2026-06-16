# ⚠️ reference_intl_migration Sensitivity — Evidence Bundle (NOT the locked release)

> **⚠️ SUPERSEDED NUMBERS — read first (updated 2026-06-16).** This bundle documents the
> *sensitivity what-if* that first surfaced the `reference_intl_migration` sum-vs-average error.
> The finding has since been **resolved**: **ADR-068** (accepted) corrected the numerator in a
> production rerun that also fixed the open-ended 90+ survival **and** a survival-horizon
> truncation. The **final** baseline is **898,907 @2055 (+12.45%)**, 90+ pool **8,172** — see
> [`../2026-06-16-corrected-run-qa-verification.md`](../2026-06-16-corrected-run-qa-verification.md)
> and the [ADR-068 amendment](../../governance/adrs/068-ref-intl-numerator-and-open-ended-survival-correction.md).
> The sensitivity figures below (e.g. 904,692 @2055) and the `round2/evidence_survival_*` CSVs
> (which stop at **2045**, pre-amendment) are **pre-correction artifacts of the investigation**,
> kept only for provenance — do not cite them as current.

**Durable, git-tracked evidence** for the finding in
[`../2026-06-15-ref-intl-migration-sum-vs-average.md`](../2026-06-15-ref-intl-migration-sum-vs-average.md)
(GitHub PR #25). It measures what the locked North Dakota baseline does when the CBO migration
adjustment's `reference_intl_migration` numerator is corrected from the 3-year **sum** (10,051) to
the true annual **average** (3,350.33).

**This is a what-if sensitivity. It is NOT the official projection and changes no published number.**
The locked public numbers remain `data/projections/baseline/` (`m2026r1` / `cfg-20260611-production-lock`,
2026-06-13). Whether 10,051 is an error to fix, and whether to do a corrected *production* rerun, is an
undecided human call.

## What was run

| | |
|---|---|
| Base configuration | The locked production config, copied verbatim (config sha256 `bf897444b5a4fec7`). |
| The ONLY substantive change | `reference_intl_migration: 10051 → 3350.33`. Everything else byte-identical. |
| Entry point | `scripts/pipeline/02_run_projections.py --counties --state --scenarios baseline` (the locked run's entry point), reusing the locked on-disk rate files. |
| Control run | Same as locked but numerator UNCHANGED — a reproducibility check. It matched the published locked trajectory to the person (max abs diff **0.0000** across all 31 years), so every difference below is attributable solely to the numerator. |

## Result — the near-term decline essentially disappears

State trajectory, GQ-inclusive (bottom-up, ADR-054):

| Year | Locked (published) | Corrected (3,350.33) | Δ | Corrected vs 2025 |
|------|-----------------:|-------------------:|----:|----:|
| 2025 | 799,358 | 799,358 | +0 | +0.00% |
| **trough** | **787,382 (2028)** | **797,911 (2026)** | — | **−0.18%** (was −1.50%) |
| 2028 | 787,382 | 799,352 | +11,970 | −0.00% |
| 2030 | 792,478 | 806,509 | +14,031 | +0.89% |
| 2040 | 836,767 | 851,565 | +14,798 | +6.53% |
| 2050 | 872,730 | 888,151 | +15,421 | +11.11% |
| 2055 | 889,017 | 904,692 | +15,675 | +13.18% |

- The visible **−1.50% trough at 2028 becomes a −0.18% blip at 2026** (~1,447 persons), then steady
  growth — an essentially flat near term rather than a decline.
- The horizon endpoint rises **+15,675 to 904,692 @2055 (+13.18%**, was +11.22%).
- County effects are broad and size-proportional (top upward revisions @2055, household basis):
  Cass +4,645, Burleigh +2,078, Grand Forks +1,218, Williams +1,148, Ward +1,057, Morton +801.
  No scenario-ordering changes; the growth-concentration / divergent-county shape is unchanged.

## Files in this folder

| File | What it is |
|------|------------|
| `comparison_state_trajectory.csv` | State totals by year: published-locked vs control vs corrected, with deltas and %-vs-2025. |
| `comparison_county_2055.csv` | Per-county published-locked vs corrected at 2055 (household basis), with deltas and growth %. |
| `corrected_state_summary_full.csv` | The corrected run's full state summary (age/sex/race, GQ-inclusive), by year. |
| `config_control_locked.yaml` | Locked config, numerator UNCHANGED (10,051), output redirected. Drives the control run. |
| `config_corrected_refintl.yaml` | Locked config, numerator CORRECTED (3,350.33). Drives the corrected run. |
| `build_comparison.py` | Builds the two comparison CSVs from the run outputs. |

## Regenerate from scratch

The two **configs here are the durable provenance**; the heavy per-county run outputs are regenerable
(they live in the working folder `data/projections/sensitivity_refintl_corrected_20260615/`, which is
laptop-local — `data/projections/` is excluded from both git and the rclone bisync). To reproduce:

```bash
# from repo root, with the venv active
python scripts/pipeline/02_run_projections.py --counties --state --scenarios baseline \
  --config docs/reviews/2026-06-15-ref-intl-sensitivity/config_control_locked.yaml      # control
python scripts/pipeline/02_run_projections.py --counties --state --scenarios baseline \
  --config docs/reviews/2026-06-15-ref-intl-sensitivity/config_corrected_refintl.yaml   # corrected
# then run build_comparison.py from the run output directory to rebuild the comparison CSVs
```

(The configs' `pipeline.projection.output_dir` points at the working folder; adjust if regenerating elsewhere.)

## Validation

- Control run reproduced the published locked trajectory exactly (max abs diff 0.0000).
- ADR-054 reconciliation: corrected state − Σcounties = −30,463.87 at every year, exactly the
  held-constant group-quarters population (ADR-055) — per-county summaries are household-basis, so the
  county deltas above are GQ-neutral and directly comparable.

**This still decides nothing.** It quantifies the effect of the correction. See the finding doc / PR #25.
