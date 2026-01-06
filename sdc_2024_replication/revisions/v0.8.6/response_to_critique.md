---
title: "Response to Critique: v0.8.6 Revision"
date_created: 2026-01-04
source_critique: "sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md"
progress_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
archived_tracker: "sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md"
target_revision: "v0.8.6"
status: "draft"
description: "Working response document for addressing the v0.8.5 critique; tracks decisions, scope, and implementation status."
---

# Response to Critique: v0.8.6 Revision

## Overview
This document records how each critique item is evaluated and addressed for v0.8.6. It is paired with the progress tracker and should stay concise.

## Scope Decision Summary
- In-scope items (v0.8.6): Refugee series extension (FY2021+), month-aware FY→PEP alignment, SIV/Amerasian parallel series, LPR panel expansion, long-run PEP regime markers, ACS moved-from-abroad proxy.
- Deferred items: 1980s-era Census components integration; full state-space fusion model.
- Approval required (Tier 3 methodology changes): Approved 2026-01-04 (data extension items) and 2026-01-06 (Travel Ban supplemental extension; ADR-027) (see Approval Log).

## Status Update Memo
- 2026-01-04: [v0.8.6 status update memo](./status_update_2026-01-04.md)
- 2026-01-06: [v0.8.6 status update memo](./status_update_2026-01-06.md)

## Critique Response Matrix
| Critique Item | Decision | Implementation Notes | Status |
|---|---|---|---|
| Extend refugee arrivals beyond FY2020 | Approved | RPC archive PDFs integrated with explicit missing-state handling (ADR-025); month-level and PEP-year variants exported. | Complete |
| Replace FY-to-PEP-year approximation with month-aware alignment | Approved | FY month → PEP-year crosswalk exported; PEP-year aligned series exported for refugees (and SIV/Amerasian). | Complete |
| Add SIV/Amerasian humanitarian series | Approved | Parallel series exported (monthly, FY, PEP-year) to avoid forcing early merging decisions. | Complete |
| Use long-run Census components series for regime modeling | Approved | 2000–2024 components exported with regime markers. | Complete |
| Integrate multi-year LPR series | Approved | LPR state totals extended to FY2000–FY2023 and saved to processed outputs. | Complete |
| Add ACS moved-from-abroad proxy | Approved | ACS moved-from-abroad proxy exported for 2010–2023 with validation note. | Complete |
| Improve Travel Ban DiD/event-study timing/robustness | Approved | Preserve FY2002--FY2019 as primary causal estimand; add FY2024 descriptive regime-dynamics supplement + appendix figure; enforce nationality-unit hygiene (exclude pseudo-nationalities) (ADR-027). | Complete |
| Update wave duration analyses (right-censoring) | Pending approval (Tier 3) | Re-estimate wave identification and survival/hazard models with FY2021–FY2024 extensions; reassess whether apparent terminations are pauses; update censoring assumptions and Monte Carlo inputs. | Not started |
| Fusion modeling / uncertainty envelope updates | Pending approval (Tier 3) | Specify a fusion model (ARIMAX/state-space) that treats admin series as correlated signals of net migration; redesign uncertainty aggregation to avoid double-counting correlated variance. | Not started |

## Data and Documentation Notes
- New data sources added:
  - RPC archive PDFs for FY2021–FY2024 refugee arrivals (partial state coverage post-2020).
  - DHS/OIS LPR yearbooks (including FY2007–FY2013 ZIP tables; FY2012 PDF fallback).
  - ACS moved-from-abroad proxy series (B07007).
- Known data limitations remaining:
  - Post-2020 refugee state omissions remain missing (not imputed); state-panel analyses must drop incomplete states (ADR-025).
  - Amerasian/SIV forecasting integration: keep USRAP refugees as strict exposure; treat Amerasian/SIV as a separate durable series linked to scenario capacity with a default post-2024 sunset (ADR-026).
  - Measurement error is nontrivial (ACS margins of error; LPR reporting lags/revisions); treat these as noisy signals rather than ground truth.
  - Correlated-signal risk: refugee/SIV, LPR, and ACS proxies overlap conceptually; uncertainty modeling must avoid stacking independent-error assumptions.

## Methodological Pitfalls (Short Notes)
- **Time-base mismatch (FY vs PEP-year vs calendar year)**: Refugee/SIV/Amerasian monthly series are aligned to PEP-year using the exported crosswalk (`data/processed/immigration/analysis/refugee_fy_month_to_pep_year_crosswalk.csv`); calendar-year PEP components remain calendar-year.
- **Net vs gross flows**: DHS LPR and refugee series are gross admissions/arrivals; Census PEP components are net migration components and should not be interpreted as gross flows.
- **Vintage revisions**: Census PEP components include vintage markers and should be treated as a sensitivity dimension when mixing pre/post-2020 segments.
- **Measurement error (ACS MOE; LPR reporting lags)**: Treat ACS stock and moved-from-abroad proxies as noisy, sometimes sparse measures for small states; avoid over-interpreting small-year changes. Treat DHS yearbook tables as subject to reporting lags and documentation changes; prefer stable aggregates and version inputs.
- **Double counting / correlated signals**: When combining multiple immigration indicators, do not assume independent errors. Avoid “adding” uncertainty from multiple sources that partially measure the same underlying process; if a fusion model is used, propagate uncertainty through a single latent process (or use joint covariance-aware simulation).
- **Partial series risk**: Post-2020 refugee state panels have missing-state omissions; downstream state-panel analyses must explicitly drop incomplete states and avoid implicit zero-fill (ADR-025).

## Approval Log (Tier 3)
| Date | Decision | Scope | Approved By |
|---|---|---|---|
| 2026-01-04 | Approved items (refugee extension, month-aware alignment, SIV/Amerasian series, LPR expansion, long-run PEP regime markers, ACS moved-from-abroad proxy) | v0.8.6 critique implementation | User |
| 2026-01-06 | Approved Travel Ban DiD refinements and supplemental FY2024 regime-dynamics extension (ADR-027) | v0.8.6 critique implementation | User |

## Open Decisions (Require User Direction)
- **Amerasian/SIV handling**: Approved. Keep USRAP refugees as strict exposure; treat Amerasian/SIV as a separate durable series linked to scenario capacity with a default post-2024 sunset (ADR-026). A merged “humanitarian” sensitivity variant is optional, but not the default.
- **Pre-2000 Census components**: Expand long-run series into the 1980s/1990s (if feasible) or remain at 2000–2024. Recommend: remain at 2000–2024 unless a specific inferential need is identified (unit-root/break power gains may be offset by comparability and sourcing effort).
