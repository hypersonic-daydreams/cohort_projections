# PUB-2026 Final Production Run Metadata

Records the locked-config production run used for the public release (QA Gate 1:
"Final production run metadata is recorded" / "Final data source notes point to the exact run").

> **⚠️ SUPERSEDED BY THE ADR-068 CORRECTED RUN (2026-06-15).** The 2026-06-13 locked run recorded below contained two confirmed errors (CBO migration numerator = 3-year sum; open-ended 90+ survival). The corrected production run replaces it:
>
> | Field | Corrected value |
> |---|---|
> | Run date | 2026-06-15 |
> | Decision record | ADR-068 |
> | `config/projection_config.yaml` sha256 (16) | `cca42fb42be76680` (was `bf897444b5a4fec7`) |
> | Key changes | `reference_intl_migration: 10051 → 3350.33`; open-ended 90+ survival corrected (`apply_open_ended_survival_correction` in `mortality_improvement.py`) |
> | Pipeline stages rerun | `01c` (survival) → `02 --counties --state` (01a/01b migration prep unaffected) |
> | State 2025 | 799,358 |
> | State trough | 797,298 (2027, −0.26%; was 787,382 / 2028 / −1.50%) |
> | State 2055 | 886,585 (+10.91%; was 889,017 / +11.22%) |
> | 90+ population @2055 | 9,971 (was ~13,707) |
>
> The trajectory table below is the prior (superseded) locked run, retained for the historical record. Public artifacts (workbook/CSV/PDF/marketing/pyramid) and the release QA gates are re-executed against the corrected run as part of publication.

## Provenance

| Field | Value |
|-------|-------|
| Run date | 2026-06-13 |
| Method / config (alias `county_champion`) | `m2026r1` / `cfg-20260611-production-lock` |
| Git commit at run | `12fa6f9` (engine components instrumentation; tree clean) |
| `config/projection_config.yaml` sha256 (16) | `bf897444b5a4fec7` |
| Locked profile sha256 (16) | `7e52c3376a4217b8` |
| Scenario | `baseline` (CBO-Adjusted), the only active/public path (ADR-065) |
| Base population | Census PEP Vintage 2025, state total 799,358 (ADR-066) |
| Geographies run | 53 counties + bottom-up state (ADR-054). Places out of scope (state/region/county release); see note below. |
| Pipeline stages | `01a` (rates) → `01b --all-variants` (convergence) → `01c` (mortality) → `02 --counties --state` (projection) |

## Locked config (vs the 2026-05-27 provisional baseline)

- College-age smoothing list: **11 counties** (Williams 38105 removed per ADR-067; was 12).
- GQ correction fraction: **0.75** (calibrated via EXP-C; was 1.0).
- Convergence medium hold: 10 years (ADR-061 D3 rejected for this release).
- Blend factor: 0.5 (EXP-B 0.7 rejected on state-accuracy grounds).

## Locked state trajectory

| Year | Population | vs 2025 |
|------|-----------:|--------:|
| 2025 | 799,358 | +0.00% |
| 2028 | 787,382 | −1.50% (trough — intended CBO front-loaded migration ramp, ADR-065 memo) |
| 2030 | 792,478 | −0.86% |
| 2040 | 836,767 | +4.68% |
| 2050 | 872,730 | +9.18% |
| 2055 | 889,017 | +11.22% |

Components of change persisted per county and aggregated to state (PUB-2026 Stage 3.1):
`data/projections/baseline/state/nd_state_38_projection_2025_2055_baseline_components.parquet`.

## Change vs provisional (2026-05-27) and attribution

Locked is **+10,007 @2050 / +12,538 @2055** above the provisional baseline. Attribution at 2055:

| Driver | State Δ @2055 | Note |
|--------|--------------:|------|
| Williams removed from college smoothing | **+8,978** | Dominant. Reverts Williams to the champion's original (unsmoothed) treatment; WSC is 1.5% of pop, so its 15–29 migration is oil-economic, not college-enrollment — smoothing it toward the state average was suppressing real dynamics (ADR-067). Backtest-justified (Williams MAPE 23.38→22.48). |
| GQ fraction 1.0 → 0.75 | ≈ +3,600 | Broad, mild; best recent-origin state APE (EXP-C). |
| Interaction / other counties | remainder | Small. |

**Williams plausibility (for Stage-4 sanity review):** Williams grows 41,767 → 63,295 (+51.5%) by
2055 — the highest-growth county. This is NOT runaway oil extrapolation: projected net migration
settles to ≈ +330/yr, **below** recent observed (+506 to +1,028 in 2023–2025). The growth is
driven primarily by natural increase from Williams' young (oil-era) age structure (births ≈ 520/yr
vs deaths ≈ 187/yr) plus modest in-migration. Williams is the "removed-and-grew" counterpart to
Ward/Grand Forks "kept-and-declined" under the same college-smoothing mechanism ADR-067 examined.
Requires explicit treatment in the public narrative alongside Ward.

## Known out-of-scope issue

`02 --all` failed at the place loader: `data/raw/geographic/nd_places.csv` holds raw Census columns
(`SUMLEV,STATE,COUNTY,PLACE,...`) where the place loader expects normalized columns
(`state_fips,place_fips,place_name,county_fips`). Places are **excluded** from the public release
(state/region/county only), so the production run used `--counties --state`. This is a pre-existing
data-prep issue (untracked file), not caused by the locked config; flagged for separate follow-up
if place projections are needed internally.
