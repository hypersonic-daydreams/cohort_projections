# ⚠️ SENSITIVITY / WHAT-IF RUN — NOT THE OFFICIAL LOCKED RELEASE

> **This folder is laptop-local working output** (`data/projections/` is excluded from both git and the
> rclone bisync). The **durable, git-tracked, hand-off copy** of the results + provenance lives at
> [`docs/reviews/2026-06-15-ref-intl-sensitivity/`](../../../docs/reviews/2026-06-15-ref-intl-sensitivity/)
> (committed on PR #25). The heavy per-county run outputs here are regenerable from the configs there.

**Do not publish, hand to marketing, or cite as the projection.** This directory holds an
exploratory **sensitivity analysis**, not the production projection. The official locked
public numbers live in `data/projections/baseline/` (run `m2026r1` /
`cfg-20260611-production-lock`, 2026-06-13). Nothing in this folder supersedes them unless
and until a corrective decision (ADR-068) is explicitly made.

| | |
|---|---|
| **Purpose** | Measure the effect of correcting the CBO migration adjustment's `reference_intl_migration` numerator from the 3-year **sum** (10,051) to the true annual **average** (3,350.33). |
| **Created** | 2026-06-15 |
| **Created by** | Claude Code, at the request of N. Haarstad, following the finding below. |
| **Finding it tests** | [`docs/reviews/2026-06-15-ref-intl-migration-sum-vs-average.md`](../../../docs/reviews/2026-06-15-ref-intl-migration-sum-vs-average.md) · GitHub PR #25 |
| **Base configuration** | The locked production config (`cfg-20260611-production-lock`, config sha256 `bf897444b5a4fec7`), copied verbatim. |
| **The ONLY change** | `reference_intl_migration: 10051 → 3350.33` (and the projection `output_dir`, to isolate these outputs). Everything else — f-schedule, fertility, mortality, convergence, base population, college/Williams/GQ settings — is byte-identical to the locked run. |
| **Entry point** | `scripts/pipeline/02_run_projections.py --counties --state --scenarios baseline` — the exact entry point used for the locked run. Reuses the locked on-disk rate files (the numerator change is applied in-engine at projection time, so stages 01a/01b/01c are not recomputed). |

## Contents

| Path | What it is |
|------|------------|
| `config_control_locked.yaml` | Locked config, numerator UNCHANGED (10,051), output redirected here. Drives the control run. |
| `config_corrected_refintl.yaml` | Locked config, numerator CORRECTED (3,350.33), output redirected here. Drives the corrected run. |
| `reference_control/` | **Control run** (numerator 10,051). Should reproduce the published locked trajectory — proves the corrected-vs-control delta is attributable solely to the numerator, not environment drift. |
| `corrected_refintl/` | **Corrected run** (numerator 3,350.33). The what-if result. |
| `run_control.log`, `run_corrected.log` | Full run logs. |
| `comparison_state_trajectory.csv` | State totals by year: published-locked vs control vs corrected (appended on completion). |
| `comparison_county_2055.csv` | Per-county published-locked vs corrected at 2055 and at the trough year (appended on completion). |

## How to read the result

The headline question is the near-term trough: the locked run dips to **787,382 in 2028 (−1.50%)**.
The corrected run measures what that trough becomes when the numerator is the true annual average.
First-order analysis (in the finding doc) estimated **~798.5k–800k** at 2028 — i.e. essentially flat —
but this run replaces that estimate with the model's actual second-order-correct output.

**This still does not decide anything.** It quantifies the effect. Whether 10,051 is an error to
correct, and whether to do a corrected *production* rerun + republish, remains a human decision.

---

## Results (verified 2026-06-15)

**Reproducibility control passed:** the control run (numerator unchanged at 10,051) reproduced the
published locked trajectory to the person — max absolute difference across all 31 years = **0.0000**.
So every difference below is attributable **solely** to the numerator change.

**The near-term decline essentially disappears.** State trajectory, GQ-inclusive (bottom-up, ADR-054):

| Year | Locked (published) | Corrected (3,350.33) | Δ | Corrected vs 2025 |
|------|-----------------:|-------------------:|----:|----:|
| 2025 | 799,358 | 799,358 | +0 | +0.00% |
| **trough** | **787,382 (2028)** | **797,911 (2026)** | — | **−0.18%** (was −1.50%) |
| 2028 | 787,382 | 799,352 | +11,970 | −0.00% |
| 2030 | 792,478 | 806,509 | +14,031 | +0.89% |
| 2040 | 836,767 | 851,565 | +14,798 | +6.53% |
| 2050 | 872,730 | 888,151 | +15,421 | +11.11% |
| 2055 | 889,017 | 904,692 | +15,675 | +13.18% |

- The visible **−1.50% trough at 2028 becomes a −0.18% blip at 2026** (a ~1,447-person dip), then steady
  growth — i.e. an essentially flat near term, not a decline. This confirms the finding doc's first-order
  estimate (~798.5k–800k @2028; actual 799,352).
- The horizon endpoint rises **+15,675 to 904,692 @2055 (+13.18%** vs 2025, was +11.22%).
- County effect is broad and proportional to size (top upward revisions @2055, household basis):
  Cass +4,645, Burleigh +2,078, Grand Forks +1,218, Williams +1,148, Ward +1,057, Morton +801.
  No ordering changes; the growth-concentration and divergent-county story is unchanged in shape.

**Validation notes:** control == published (diff 0.0000); ADR-054 reconciliation = constant GQ
(−30,463.87 at every year, the held-constant group-quarters offset; per-county summaries are
household-basis, so county deltas are GQ-neutral and directly comparable).

**Reproduce:** `python build_comparison.py` (regenerates `comparison_state_trajectory.csv` and
`comparison_county_2055.csv` from the two runs).

**Still not a decision.** This quantifies the effect of the correction. Whether 10,051 is an error to
fix and whether to do a corrected *production* rerun + republish remains a human call (see the finding
doc and PR #25).
