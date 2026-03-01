# 2026-03-01 PP3 Gate 4 (Rescaling) Review Note

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 human review package Gate 4: rescaling behavior (IMP-12/IMP-13 outputs) |
| **Primary Artifact** | `docs/reviews/2026-03-01-pp3-human-review-package.html` |
| **Data Sources** | `data/projections/{scenario}/place/qa/qa_share_sum_validation.csv`, `data/projections/{scenario}/place/qa/qa_outlier_flags.csv` |

## Question

Is rescaling controlled and concentrated in expected locations, or does it indicate diffuse/systemic instability that should block the pipeline?

## Key Observations (Baseline; pattern identical across scenarios)

- **County-years evaluated:** `1,426` (`46` counties × `31` projection years).
- **Rescaling applied:** `530 / 1,426` county-years (`37.2%`) have `rescaling_applied=true`.
- **Counties impacted:** `21 / 46` counties (`45.7%`) have *any* rescaling (at least one year).
- **Strong concentration:** `16` counties are rescaled in **all 31 years**; one county is rescaled in `30/31` years; only `4` counties show a single rescaled year.
- **Where it concentrates (always-rescaled counties):** Burleigh, Cass, Dickey, Grand Forks, Hettinger, LaMoure, Morton, Mountrail, Pembina, Richland, Sargent, Stark, Traill, Walsh, Ward, Williams. (Rolette is `30/31`.)
- **Complexity linkage:** The always-rescaled counties generally have more projected places (e.g., Cass has the most projected places). Across counties, projected-place count vs. rescaling-years shows a strong positive association (corr ~ `0.67`), consistent with a reconciliation step that becomes more active as the number of independently-modeled shares increases.

## Interpretation

This pattern looks **structural and controlled**, not chaotic:

- The rescaling footprint is **stable across scenarios** (`baseline`, `restricted_growth`, `high_growth`) and consistent year-over-year within the same counties.
- The affected counties largely align with counties that have **multiple projected places**, where independent share projections are more likely to require reconciliation.
- There is no evidence (from the hard-gate checks) of constraint violations; share sums remain within the required bounds.

One important nuance: `rescaling_applied` is set when *any* place share differs from its raw value by more than a very small tolerance (`1e-9`). This is a sensitive indicator for “reconciliation occurred”, not “reconciliation was large”.

## Additional Evidence From Outlier Flags

- `SHARE_RESCALED` outlier flags (baseline) encode `raw_sum_place_shares` values slightly above 1.0, with observed overage on the order of ~`0.2%` to `1.1%` (max `raw_sum_place_shares≈1.011`).
- This supports the view that when the ceiling binds, it is typically a **small correction**, not a large instability.

## Decision Recommendation

**Do not block** on Gate 4.

- Suggested disposition: **Approve with notes** (document that reconciliation/rescaling is expected in the identified subset of counties and appears stable).

## Follow-Up Recommended (Not Blocking)

Add visibility into **reconciliation magnitude**, not just its occurrence:

- Surface per county-year reconciliation magnitude metrics (e.g., `total_before_adjustment`, `reconciliation_adjustment`, and threshold-flag) into a QA artifact and the Gate 4 review tab.
- Add a distribution visual and “top county-years by adjustment” table so review can confirm that reconciliation is *small* in magnitude in addition to being concentrated in location.

## Implementation Follow-Up Status

Implemented on 2026-03-01 as `IMP-13A`:

- New QA artifact: `qa_reconciliation_magnitude.csv` (county-year reconciliation magnitude fields).
- Human-review package Gate 4 now includes reconciliation-magnitude distribution plus top-adjustment county-year table.
- Verification details are recorded in `docs/reviews/2026-03-01-pp3-imp13a-results.md`.
