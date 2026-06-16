# Round-2 independent review — corrected production package

## Executive verdict

**GO to proceed to the publication/regeneration + QA-gate step.** I do **not** see a remaining numeric/methodology defect requiring another production rerun before public artifacts are rebuilt.

The two ADR-068 headline fixes are implemented in code/config, not merely relabeled:

- `reference_intl_migration` is now the annual mean **3,350.33**, with the engine still consuming it as an annual flow.
- age-90 open-ended survival is now overwritten to the open-interval `T91/T90` level, with the NP2023 improvement trajectory preserved.

However, this is **not yet a “publish tomorrow without further edits” package**. Before public release, QA must fix/verify public-facing text and output labeling around fertility provenance, mortality provenance, stale old numbers, GQ/household-basis components, and county summary usage.

---

## 1. Fix verification

| Round-1 fixed item | Verification | Verdict |
|---|---|---|
| **CBO `reference_intl_migration` 10,051 → 3,350.33** | `config/projection_config.yaml`, `scenarios.baseline.migration.reference_intl_migration: 3350.33`; same value in `restricted_growth`. Evidence file `evidence_pep_state_intl_2023_2025.csv` gives 2023=3,158, 2024=4,083, 2025=2,810; sum = 10,051, mean = 3,350.33. `cohort_projections/core/migration.py::apply_migration_scenario()` still computes `annual_reduction = ref_intl * (1.0 - factor)` and `reduction_rate = annual_reduction / ref_pop`, so the corrected annual numerator is used correctly. | **Fixed.** First-year nominal decrement is now about `3,350.33 × 0.80 / 799,358 = 0.00335`, not `0.01006`. |
| **90+ open-ended survival plateau corrected** | `cohort_projections/data/process/mortality_improvement.py::apply_open_ended_survival_correction()` reads `Tx` at age 90 and 91 from `cdc_lifetables_2023_combined.csv`, sets `target = T91 / T90`, and rescales the age-90 trajectory. The pipeline calls it in `run_mortality_improvement_pipeline()` immediately before saving `nd_adjusted_survival_projections.parquet`. Evidence: `evidence_survival_85plus.csv` shows 2025 age-90 survival **Male 0.7775714493**, **Female 0.8058096419**, while ages 85–89 remain at the old 85+ plateau around **Male 0.8849821852**, **Female 0.9143115174**. | **Fixed.** Code and evidence show the operative engine age-90 row is no longer the 85+ plateau. |
| **NP2023 improvement trajectory preserved** | Evidence shows age-90 survival increases over time: Male 0.777571 in 2025 → 0.786836 in 2045; Female 0.805810 in 2025 → 0.815905 in 2045. | **Fixed as far as supplied evidence shows.** The text excerpt supplied to me shows evidence through 2045; final QA should assert the actual production survival table spans 2025–2055. |
| **`sex_ratio_male` explicit config key** | `config/projection_config.yaml`, `rates.fertility.sex_ratio_male: 0.51`. `cohort_projections/core/fertility.py::calculate_births()` reads `config['rates']['fertility']['sex_ratio_male']`, with 0.51 only as fallback. | **Fixed.** |
| **Aggregation tolerance 0.01 → 0.000001** | `config/projection_config.yaml`, `geography.hierarchy.aggregation_tolerance: 0.000001`. | **Config fixed, but enforcement remains weak in code.** `multi_geography.validate_aggregation()` still has a function default `tolerance=0.01`, and the main projection validation path mainly logs diagnostics. Since state is built bottom-up, this is not a numeric blocker, but release QA should run an explicit exact residual check. |
| **Place projections disabled/out of scope** | `config/projection_config.yaml`: `place_projections.enabled: false`; `output.aggregation_levels` contains only `state` and `county`. | **Production config fixed.** Caveat: `scripts/pipeline/02_run_projections.py --all` still expands to `["state","county","place"]`; production should continue using `--counties --state` or the runner should honor `place_projections.enabled=false`. |

### Race-flat mortality caveat

Confirmed. `evidence_survival_table_FULL.csv` header is:

`year,age,sex,survival_rate,source`

There is **no `race` column**. The runner expands survival across all six race categories in `scripts/pipeline/02_run_projections.py::_build_survival_rates_by_year()`. Methodology §4.2 and §10.7 disclose this. This is acceptable for 2026 **only if race-output users see the limitation prominently**.

---

## 2. Number consistency

### Base population

Verified from `evidence_county_base_pop_2025.csv`: the 53 county populations sum to the stated base:

- `# SUM_OF_53_COUNTIES,799358`

The loader source is also correct: `base_population_loader.py::_load_vintage_2025_county_populations()` reads `POPESTIMATE2025` from `co-est2025-alldata.parquet` and enforces `len(result) == 53`.

### Offset arithmetic

The offset story is internally consistent:

- Old locked set: **889,017.40 @2055** from `comparison_state_trajectory.csv`.
- ref-intl-only sensitivity: **904,692.25 @2055**.
- Corrected production: **886,585.25 @2055** per ADR-068/final metadata.

Therefore:

- migration-numerator effect alone: `904,692.25 − 889,017.40 = +15,674.85`, matching the stated **+15.7k**.
- 90+ production effect relative to ref-intl-only: `904,692.25 − 886,585.25 = 18,107.00`, matching the stated **−18.1k**.
- net production vs old locked: `886,585.25 − 889,017.40 = −2,432.15`.

No arithmetic inconsistency.

### State = sum of counties

The code path supports exact bottom-up state aggregation: `scripts/pipeline/02_run_projections.py::aggregate_county_results_to_state()` concatenates county parquet files and groups by `year, age, sex, race`, summing `population`.

The prompt reports corrected-run residual **0.0** at 2025 and 2055. I do not have the actual corrected production county/state parquet outputs in the text to independently re-sum, so final QA should rerun the exact check. I see no evidence of a reconciliation gap.

### 90+ population @2055

ADR-068/final metadata report **90+ population @2055 = 9,971**. The survival-rate evidence supports the old-age correction, but the supplied text does not include a 2055 age-by-age production population file from which I can independently re-sum the 90+ population. QA should include a direct assertion on the regenerated public outputs.

---

## 3. Still-open findings: severity and publication verdict

| Finding | Publication severity | Verdict |
|---|---:|---|
| **Small-county `blend_threshold` config key not wired to build** | **Major governance/reproducibility issue; not a numeric blocker for this run.** | Acceptable with disclosure for this vintage. Blending is implemented in `scripts/data/build_race_distribution_from_census.py` with `DEFAULT_BLEND_THRESHOLD = 5000` and `alpha = min(county_total / 5000, 1.0)`, but it is a CLI default, not read from `config.projection_config.yaml`. Disclose that the processed county distribution file is prebuilt with a 5,000 threshold; wire the config key before the next vintage. |
| **Fertility provenance mismatch: “pooled 2020–2023” vs evidence table `year=2023` only** | **Major; must fix before public release.** | Do not publish methodology claiming pooled 2020–2023 unless metadata proves that the `year=2023` column is merely a label for a 2020–2023 pooled estimate. If the table is 2023-only, change the text. No rerun required if the numbers are intentionally final, but provenance must be truthful. |
| **Flat 5-year → single-year ASFR expansion** | **Minor to moderate.** | Acceptable with disclosure. `02_run_projections.py::_transform_fertility_rates()` flat-copies each 5-year ASFR to every single age in the band. This preserves group-level births/TFR, but it is not Sprague or graduated interpolation. Public method text should say “constant within 5-year maternal age band.” |
| **Newborns not exposed to infant mortality in birth year** | **Minor.** | Acceptable with disclosure. Infant survival is around 0.998–0.999, so the effect is likely on the order of only tens of persons per year, not a headline issue. |
| **College smoothing updates `migration_rate` but not `net_migration` diagnostics** | **Minor.** | Acceptable if residual-rate diagnostic files are not published as final components. Projection uses `migration_rate`; diagnostics using `net_migration` after smoothing are stale. Prefer recompute or drop `net_migration` in post-smoothing outputs. |
| **Contradictory mortality provenance docs** | **Major; must fix before public release.** | Config says `life_table_year: 2023`; methodology §4.2 says NVSR 74-12 / 2022; `mortality_improvement.py` docstring/metadata refer to ND CDC 2020 / SDC 2024 baseline; ADR-068 uses CDC 2023 `Tx` for the 90+ correction. These can coexist only if clearly sequenced. Public methodology must distinguish static base, NP2023 time-varying operative table, and 2023 open-age correction. |
| **GQ hold-constant makes components household-basis** | **Major labeling issue; can be acceptable with disclosure.** | Any public components table must say births/deaths/net migration are household-basis and exclude implied GQ turnover. Otherwise users will compare projected deaths (~5k) with PEP total deaths (~7k) and infer a mortality error. |
| **CBO decrement uniform across age/sex/race and applied after rate cap** | **Major methodological limitation, not a blocker.** | Acceptable with disclosure. Current code subtracts the same per-capita decrement from all cells after convergence rates have already been capped. First-year corrected decrement is ~0.00335, so it can push a general capped `−0.080` rate to about `−0.08335`. Prefer immigrant-age-profile allocation in next vintage. |

---

## 4. New issues surfaced in round 2

### N1. County summary CSVs may be household-only after GQ re-addition

In `multi_geography.py::run_single_geography_projection()` the code:

1. runs the engine on household population,
2. adds GQ back to `projection_results`,
3. then calls `projection_engine.get_projection_summary()`.

But `get_projection_summary()` returns summaries accumulated inside the engine before GQ was re-added. Thus county parquet outputs are GQ-inclusive, but county `_summary.csv` files may remain household-only.

**Severity:** Major if public exporters consume county summary CSVs.  
**Required before release:** either recompute county summaries from the GQ-inclusive `projection_results`, or ensure public exporters ignore county summary CSVs and build from parquet/detail outputs.

### N2. Aggregation code should enforce exactly 53 unique current county files

`aggregate_county_results_to_state()` reads all files matching `nd_county_*_projection_*.parquet`. It does not visibly enforce exactly 53 unique county FIPS for the current run. A stale duplicate or partial rerun could be aggregated.

**Severity:** Minor for the corrected run as reported; major for QA robustness.  
**Fix:** release QA should assert 53 unique county FIPS, one file per county, current config/run metadata, and exact state residual.

### N3. Migration is also race-flat

The residual migration pipeline is county × age-group × sex. `02_run_projections.py::expand_5yr_migration_to_engine_format()` expands each age/sex rate uniformly to all six races. This is separate from race-flat mortality.

**Severity:** Major limitation for race/ethnicity outputs; not a state-total blocker.  
**Disclosure needed:** race projections use race-specific fertility and base composition, but migration and operative mortality are race-flat.

### N4. 90+ methodology text still conflicts with ADR-068 implementation

Methodology §4.3 still states an open-age formula involving `T91 / (T90 + L90/2)` and “typical values 0.60–0.70.” ADR-068 and code implement `T91/T90`, yielding 2025 values ~0.778 male / ~0.806 female.

**Severity:** Major documentation issue.  
**Fix before release:** make §4.3 match the operative ADR-068 formula and values.

### N5. Mortality-improvement metadata omits the open-age life table source

`mortality_improvement.py` metadata lists Census NP2023 and ND CDC baseline files, but the ADR-068 open-ended correction also depends on `data/raw/mortality/cdc_lifetables_2023_combined.csv`.

**Severity:** Minor.  
**Fix:** include that file and target age-90 values in metadata for auditability.

---

## 5. End-to-end audit notes

- **Base population/vintage:** Sound. PEP Vintage 2025 county totals, `POPESTIMATE2025`, 53 counties, state 799,358.
- **Fertility:** Numeric rates are plausible; `evidence_fertility_TFR.csv` shows total TFR 1.8625 before CBO cut and 1.7694 after −5%. Main problem is provenance/expansion documentation.
- **Mortality:** ADR-068 90+ fix is implemented. Race-flat operative survival is disclosed. Public docs still need provenance cleanup.
- **Migration residual/special adjustments:** Code implements GQ fraction 0.75, Bakken dampening, male dampening, reservation PEP recalibration, and college smoothing as described. Main remaining issue is stale `net_migration` diagnostics after smoothing.
- **CBO adjustment:** Corrected numerator implemented; order remains after convergence cap. Uniform decrement is a disclosed simplification.
- **GQ:** Mechanically sound for projections; components and summaries need careful labeling/recomputation.
- **State aggregation:** Correct bottom-up design. Final QA must verify exact residual on actual regenerated public files.

---

## 6. Final recommendation

**Proceed to publication artifact regeneration and QA gates using corrected set (3):**

- 2025: **799,358**
- trough: **797,298 in 2027**
- 2055: **886,585**
- 90+ @2055: **9,971**

I do **not** recommend another model rerun based on the supplied package.

Before public posting, the minimal release blockers to clear are:

1. purge/replace stale public text showing 889,017, 787,382, 876,479, 882,146, or old troughs;
2. fix fertility and mortality provenance language;
3. correct/label GQ household-basis components and county summaries;
4. run exact QA: 53 unique county files, state=sum counties, base=799,358, 2055=886,585.25, 90+=9,971, and survival table coverage through 2055.