## Executive verdict

**No-go on publishing the locked numbers as-is.** The open finding is confirmed from the supplied config and code: `reference_intl_migration: 10051` is labeled and used as an **annual** flow, but the supplied evidence shows it is the **2023–2025 three-year sum**. The locked-config sensitivity is the right evidentiary test because it changes only that numerator and its control reproduces the locked trajectory exactly. Its effect is material to the headline: **2028 trough −1.50% becomes a −0.18% 2026 blip; 2055 rises from 889,017 to 904,692**.

Several other issues are not necessarily fatal to the state-total projection, but are serious for a public government release—especially mortality documentation/code mismatch, unverifiable fertility-rate provenance, and stale/contradictory public-facing text. The CBO numerator issue alone is a release blocker.

---

## 1. Per-link findings table

| Chain link | What I evaluated | Evidence quoted / file | Checks out? | Finding / risk |
|---|---|---|---|---|
| **Base population & vintage** | Base year, horizon, geography | `config/projection_config.yaml`: `base_year: 2025`, `projection_horizon: 30`, `counties.mode: "all"`; final metadata: base population `799,358` | **Yes** | Base-year/horizon setup is internally consistent with locked trajectory 2025–2055. |
| **Base population & vintage** | County totals from PEP Vintage 2025 | `base_population_loader.py`: `VINTAGE_2025_COUNTY_POPEST_PATH = "parquet/2020-2025/county/co-est2025-alldata.parquet"`; `_load_vintage_2025_county_populations()` uses `POPESTIMATE2025` and requires `len(result) == 53` | **Yes, as code** | Correct source/field for 2025 county totals. I cannot verify the actual parquet contents because the data file is not included, but locked total matches 799,358. |
| **Base population structure** | Vintage-2024 age/sex/race structure scaled to Vintage-2025 totals | `docs/methodology.md §2.1`: detailed allocation uses `SC-EST2024-ALLDATA6` and `cc-est2024`, scaled to `POPESTIMATE2025`; `config`: `single_year_distribution: data/raw/population/nd_age_sex_race_distribution_single_year.csv`; `county_distributions.path: data/processed/county_age_sex_race_distributions.parquet` | **Methodologically acceptable but risky** | This is explicitly disclosed in ADR-066/methodology. It is not an arithmetic error, but public docs must keep the caveat: **2025 totals, 2024 characteristics**. Refresh when V2025 characteristics are available. |
| **Base population structure** | Small-county blending below 5,000 | `config`: `blend_threshold: 5000`; `docs/methodology.md §2.4.1` gives blending formula. In `base_population_loader.py`, `load_county_age_sex_race_distribution()` reads the county distribution and normalizes; it does **not** apply `blend_threshold`. | **Not verified / code-doc gap** | Either the prebuilt parquet already contains blended distributions or the documented blend is not implemented in the loader. The package does not include the county-distribution build code, so I cannot verify. For release, provide metadata proving the file is blended, or implement blending where the config says it occurs. |
| **Base population / Sprague** | County five-year to single-year expansion | `config`: `county_race_interpolation: "sprague"`; `base_population_loader.py::_expand_county_with_sprague()` calls `sprague_graduate(prop_vector, clamp_negatives=True)` and uses terminal survival factor `0.7` for 90+ | **Partly verified** | The loader uses a Sprague path and terminal-tail logic matching the documented `s=0.7`. But `sprague_graduate()` itself is not supplied, so I cannot verify the coefficient matrix or group-total preservation implementation. |
| **Base population / GQ separation** | Base household/GQ split | `config`: `group_quarters.enabled: true`, `method: "hold_constant"`, `gq_data_path: data/processed/gq_county_age_sex_2025.parquet`; `base_population_loader.py::_separate_gq_from_base_population()` subtracts GQ, caps at cell total, stores `gq_population` | **Mechanically sound** | Code does what methodology says. Risk remains in the GQ source/allocation assumptions: county GQ race is proportional to county race; sex/age allocation source is not fully verifiable from supplied files. |
| **Fertility** | Baseline −5% fertility adjustment | `config`: `scenarios.baseline.fertility: "-5_percent"`; `fertility.py::apply_fertility_scenario()` multiplies `fertility_rate * 0.95` | **Yes** | The flat CBO fertility cut is implemented exactly as documented. |
| **Fertility** | Fertility data source and 5-year→single-year expansion | `docs/methodology.md §3.1`: CDC WONDER ND pooled `2020–2023`, five-year age groups expanded to single-year. `config` comment says “2024 rates used directly (no multi-year averaging)”. `fertility_rates.py::create_fertility_rate_table()` expects already single-year `age` 15–49; no expansion routine is shown. | **Not verified / contradiction** | I cannot verify the actual fertility rates, TFR, fallback cells, or expansion because the processed input file is not included. There is a public-code/doc conflict: methodology says 2020–2023 pooled; config comment says 2024 direct; supplied code does not show five-year expansion. |
| **Fertility** | Birth calculation and sex ratio | `fertility.py::calculate_births()`: `sex_ratio_at_birth = birth_config.get("sex_ratio_male", 0.51)`; config has no `sex_ratio_male` key | **Calculates as documented, but not truly configured** | The 0.51 male share is effectively hard-coded as a default. Minor, but for release put `sex_ratio_male: 0.51` in config or stop calling it configured. |
| **Fertility / engine timing** | Births exposure to mortality/migration | `cohort_component.py::project_single_year()`: births are calculated from `population[sex=="Female"]` before migration and concatenated after migration; comment: “Typically births don't experience migration in birth year, so skip” | **Approximation** | Newborns do not experience infant mortality during the birth interval. Impact is small, but public methodology should acknowledge this simplification or apply a birth-survival ratio. |
| **Mortality** | Static mortality improvement path | `mortality.py::apply_mortality_improvement()`: converts survival to death rate, multiplies by `(1 - improvement_factor) ** years_elapsed`, caps survival at 1.0 | **Yes, if static tables are used** | Static path is coherent. Engine disables this when year-specific tables are supplied, avoiding double improvement. |
| **Mortality** | Production “two-track” / race-specific survival | Final metadata says pipeline ran `01c` mortality. Supplied `mortality_improvement.py` builds `ND_survival[age, sex, year]` from Census NP2023 `GROUP == 0` and ND age/sex baseline; output columns are `[year, age, sex, survival_rate, source]`—**no race**. | **Major code-doc mismatch** | Methodology claims age/sex/race-specific survival preserving race differentials. The supplied production improvement code is race-flat. The runner that expands this to engine inputs is not supplied, so I cannot verify final handling. If production replicates these rates across races, race-specific mortality is overridden for all years. |
| **Mortality** | 90+ open-ended survival | `docs/methodology.md §4.3` expects open 90+ survival typically `0.60–0.70`. `survival_rates.py::calculate_survival_rates_from_life_table()` uses `t_91 = t_90 - l_90` and denominator `t_90 + l_90/2`; `mortality_improvement.py` has no open-age 90+ formula. | **Major unresolved risk** | I cannot verify the actual `~0.885` plateau because the survival parquet is not supplied. But the supplied production improvement path appears to use closed-age survival ratios, not an open 90+ aggregate. The static processor also uses `lx` where docs call for `Lx`. This needs verification before publishing age-detail outputs. |
| **Mortality provenance** | Life table vintage/source consistency | `config`: `life_table_year: 2023`; `docs/methodology.md §4.2`: ND state tables `NVSR 74-12, 2022`; `mortality_improvement.py` docstring: ND CDC baseline / Census NP2023 | **Contradictory** | Public documentation must reconcile whether production mortality is CDC 2023, ND 2022, SDC/CDC 2020, or NP2023-adjusted. |
| **Migration residual** | Residual formula and annualization | `residual_migration.py::compute_residual_migration_rates()`: `expected = p_start * s_rate`; `migration = p_end - expected`; `rate_period = migration / expected`; `_annualize_period_migration_rate()` uses `(1 + period_rate) ** (1/period_length) - 1`; 2020–2024 survival exponent `period_length / 5.0` | **Yes** | Core residual calculation matches methodology. Actual historical snapshots/rates are not included, so I cannot independently recompute outputs. |
| **Migration residual** | 85+ and 0–4 treatment | `residual_migration.py`: 85+ expected is `80-84 survivors + 85+ survivors`; 0–4 birth cohort output rate `0.0` | **Yes** | Matches methodology. |
| **Migration residual / GQ correction** | Household-only residual rates | `config`: `gq_correction.enabled: true`, `fraction: 0.75`; `subtract_gq_from_populations()` subtracts `fraction * gq_population`, floors at zero | **Mechanically yes** | Fraction 0.75 matches locked config. Actual historical GQ file and backward-constant construction are not supplied, so I cannot verify GQ amounts. Add bounds validation for `fraction`. |
| **Migration residual / Bakken** | Oil dampening | `config`: oil counties include `38105, 38053, 38061, 38025, 38089, 38007`; factors `2005-2010: 0.50`, `2010-2015: 0.40`, `2015-2020: 0.50`; `apply_period_dampening()` multiplies `migration_rate` and `net_migration` | **Yes** | Code matches config and documentation. |
| **Migration residual / male dampening** | Male boom dampening order and compounding | `run_residual_migration_pipeline()`: calls `apply_period_dampening()` then `apply_male_migration_dampening()`; male factor `0.80` | **Yes** | Oil × male factors compound by design, e.g. `0.40 * 0.80 = 0.32` for oil-county males in 2010–2015. Not an error, but it is a strong assumption. |
| **Migration residual / reservation** | PEP recalibration | `config`: counties `["38005","38085","38079"]`; `apply_pep_recalibration()` sums PEP `netmig` for `year > start_year` and `<= end_year`; same-sign scaling or Rogers-Castro fallback | **Yes, as code** | Method matches ADR-045. PEP file not supplied, so I cannot verify period totals. Validation caveat: ADR-067 says permanent walk-forward harness omits PEP recalibration for reservation counties, so backtest evidence for these counties is limited. |
| **Migration residual / college smoothing** | College-age smoothing | `config`: 11 counties, ages `15-19`, `20-24`, `25-29`, `blend_factor: 0.5`; `apply_college_age_adjustment()` blends county `migration_rate` with unweighted statewide age/sex mean | **Mostly yes** | Projection rates are smoothed as documented. But the function does **not** update `net_migration`, so saved diagnostics/metadata using `net_migration` are inconsistent after smoothing. |
| **Convergence interpolation** | Window mapping | `config`: recent `[2023,2025]`, medium `[2014,2025]`, longterm `[2000,2025]`; `_map_config_window_to_periods()` includes any overlapping historical period | **Yes** | Produces documented windows: recent = 2020–2024; medium = 2010–2015, 2015–2020, 2020–2024; long = all five. |
| **Convergence interpolation** | 5-10-5 plus 30-year long-term hold | `config`: `recent_to_medium_years: 5`, `medium_hold_years: 10`, `medium_to_longterm_years: 5`; `projection_years = config.project.projection_horizon`; `calculate_age_specific_convergence()` clamps phase-3 `t` to 1.0 for years >20 | **Yes** | Code matches methodology for 30-year horizon. Minor stale risk: module docstring still says 2025–2045 / 20-year output. |
| **Convergence / cap** | Age-aware cap | `config`: college ages include `25-29`, `college_cap: 0.15`, `general_cap: 0.08`; `_apply_rate_cap()` caps after interpolation | **Yes pre-CBO** | Cap is applied before in-engine CBO decrement. Therefore final CBO-adjusted rates can exceed the nominal cap on the negative side. This is not necessarily wrong, but it must be documented. |
| **CBO adjustment** | Baseline scenario is CBO-adjusted and active | `config`: `scenarios.baseline.active: true`; `fertility: "-5_percent"`; `migration.type: "additive_reduction"`; pipeline scenarios `["baseline"]` | **Yes** | Public baseline-only setup matches ADR-065. |
| **CBO adjustment — central finding** | Reference international migration numerator | `config`: `reference_intl_migration: 10051  # Annual international migration (PEP 2023-2025 avg)`; `docs/methodology.md §6.1`: `M_intl = 10,051: average annual`; finding doc shows PEP values `3,158 + 4,083 + 2,810 = 10,051`, mean `3,350.33` | **No — error** | `10,051` is the three-year sum, not the annual average. |
| **CBO adjustment — central finding** | Per-year application | `migration.py::apply_migration_scenario()`: `annual_reduction = ref_intl * (1.0 - factor)` and `reduction_rate = annual_reduction / ref_pop`; log says `annual_reduction=... persons` | **No — wrong magnitude** | Because `ref_intl` is used as persons/year, the current first-step nominal reduction is `10051 * 0.80 = 8,041` instead of `3350.33 * 0.80 = 2,680`. |
| **CBO adjustment** | Corrected sensitivity | Sensitivity README: only change `10051 → 3350.33`; control run max abs diff `0.0000`; corrected trajectory 2055 `904,692`; trough `797,911` in 2026 | **Yes — right test** | This is the correct non-destructive basis for disposition. |
| **Special populations / college** | Williams removed from smoothing | `config`: Williams commented out; ADR-067 says WSC 1.5% below threshold and removal improved Williams MAPE | **Yes** | Methodology and config align. Risk: public narrative must explain Williams growth as oil/young-age-structure, not college artifact. |
| **Special populations / Bakken** | Period-specific oil dampening + male dampening | See migration rows above | **Yes** | Sound as implemented; assumptions remain judgmental and should stay in limitations. |
| **Special populations / GQ** | Hold-constant GQ | `config`: `method: "hold_constant"`; base loader stores GQ; final sanity says GQ constant and components reconcile | **Yes with caveat** | Holding GQ constant is defensible but material. Public components of change are household-basis; deaths look low unless labeled. |
| **Special populations / reservation** | PEP-anchored recalibration | See reservation row above | **Yes as code** | Production includes it; validation evidence is weaker because harness omits it in permanent raw-base recompute. |
| **Projection engine** | Order of operations | `cohort_component.py::project_single_year()`: get year-specific rates → apply scenario adjustments → survival → births from starting female population → migration on survived population → add births | **Mostly yes** | Matches methodology on broad order. Approximation: births are not exposed to infant mortality in birth year; fertility uses start-year female population, not average exposure. |
| **Projection engine** | Scenario adjustment location | `project_single_year()`: scenario fertility and migration applied after rate lookup; ADR-065 requires engine-only scenario adjustment | **Yes** | Sound; prevents double application. |
| **Projection engine** | CBO decrement denominator/application basis | `reference_population: 799358`; decrement applied to post-survival household population, not GQ | **Approximate** | Nominal statewide reduction uses total-pop denominator but is applied to household survived population. This slightly reduces realized persons removed versus nominal. Not central, but document. |
| **Bottom-up state aggregation** | State = county sum | `docs/reviews/2026-06-13-locked-run-sanity-check.md`: max abs diff `0.0000`; ADR-054 | **Yes** | Strong. I did not receive aggregation code, but validation evidence is direct. |
| **Bottom-up state aggregation** | Failure tolerance | `config`: `aggregation_tolerance: 0.01`; final run says exact | **Risk** | A 1% tolerance is too loose for release QA, even if final diff is zero. Public production should fail if any county is missing or state ≠ county sum beyond rounding. |
| **Outputs** | Locked state trajectory | Final metadata: 2025 `799,358`; trough 2028 `787,382`; 2055 `889,017` | **Yes as locked output** | Structurally validated, but not publishable as-is because the CBO numerator is wrong or undispositioned. |
| **Outputs / components** | Components of change | Locked sanity: projected 2026 deaths `5,084` vs observed total deaths ~`7,130`; explanation: household basis due to held-constant GQ | **Needs labeling** | Any public components table must say household-basis or include GQ turnover note. Otherwise users will see a mortality error. |
| **Outputs / scope** | Place outputs | `config`: place projections enabled and output levels include `place`; final metadata says `02 --all` failed at place loader and public run used `--counties --state`; places out of scope | **Scope contradiction** | Public methodology/config should not imply place outputs are included if release is state/county only. |

---

## 2. Prioritized issue list

| Priority | Severity | Issue | Why it matters | Recommended fix |
|---:|---|---|---|---|
| 1 | **Blocker** | `reference_intl_migration = 10051` is a three-year sum but used as annual | Changes headline near-term result and 2055 endpoint: locked 2055 `889,017`; corrected sensitivity `904,692`; trough shifts from `−1.50%` to `−0.18%` | Create ADR-068. Set baseline and restricted alias to `3350.333333` or change formula to divide the three-year sum by 3. Rerun projections, state aggregation, exports, and release QA. Update final metadata and all public text. |
| 2 | **Blocker** | Publishing while the finding is open would knowingly publish mislabeled assumptions | Public config/methodology say “annual average”; locked numbers rely on a 3× numerator | Do not publish locked numbers until disposition. If intentionally retaining 10,051, relabel it as a deliberate three-year-sum severity multiplier and justify publicly. I do **not** recommend that path. |
| 3 | **Major** | Mortality production path appears race-flat despite race-specific methodology claims | Public race/ethnicity projections depend on differential mortality. Supplied `mortality_improvement.py` outputs age/sex only from all-race NP2023 `GROUP=0` | Verify the actual engine survival inputs. If race-flat, either rebuild year-specific survival by age/sex/race using race-specific qx + ND adjustment or remove race-specific mortality claims and caveat race outputs. |
| 4 | **Major** | 90+ open-age survival not verified; supplied code has formula/path risks | A closed-age-90 survival ratio treated as open 90+ can materially overstate oldest-old survival | Inspect final survival table. Validate age-90 survival by sex/race/year. Compute proper open 90+ survival using life-table `T_x/L_x` or aggregate 90+ exposure. Fix static `survival_rates.py` formula using `L90`, not `lx`. |
| 5 | **Major** | Fertility data provenance and five-year→single-year expansion are not auditable from supplied code | Methodology claims CDC WONDER 2020–2023 pooled ND rates and expansion; config says 2024 direct; code expects already single-year rates | Include the exact processed fertility file and metadata. Add/point to expansion code. Report TFR before/after 0.95 and expected annual births. Put `sex_ratio_male: 0.51` in config. |
| 6 | **Major** | Small-county age/sex/race blending is documented/configured but not implemented in supplied loader | Could affect small rural/race cells and reservation/rural age structures | Prove `county_age_sex_race_distributions.parquet` is already blended, or implement `blend_threshold` in the loader/build pipeline. Add metadata per county showing blend alpha. |
| 7 | **Major** | CBO decrement is uniform across all ages/races/counties and applied after rate cap | It removes international migration from elderly/children proportionally, not using immigrant age profile; final rates can exceed cap | At minimum disclose. Prefer age/sex/race distribution of CBO decrement using recent international-migrant profile. Decide whether cap should apply pre- or post-CBO and document. |
| 8 | **Major** | GQ hold-constant components are easy to misread | Projected deaths are household-only; public users will compare to PEP total deaths and see a large gap | Label components as household-basis, or add an accounting line for implied GQ turnover/deaths/entrants. Keep GQ caveat prominent. |
| 9 | **Minor/Major** | College smoothing adjusts `migration_rate` but not `net_migration` diagnostics | Projection uses rates, but saved metadata/diagnostics can misstate migration totals | In `apply_college_age_adjustment()`, recompute diagnostic `net_migration` from smoothed rate and denominator or remove it from post-smoothing diagnostics. |
| 10 | **Minor** | Stale/contradictory public text | Examples: 10,051 “annual average”; fertility “2024 direct” vs 2020–2023 pooled; mortality 2023/2022/2020; convergence docstring 20-year; place outputs enabled but out of scope | Do a publication text freeze after corrected rerun. Grep for stale values (`876,479`, `882,146`, `10051`, “annual average”, “2024 rates used directly”, “2025–2045”). |
| 11 | **Minor** | Aggregation tolerance too loose for a public lock | `aggregation_tolerance: 0.01` could allow thousands of persons, although final diff is zero | For release QA, require exact county coverage and state=sum counties within rounding tolerance, e.g. <1 person or <0.0001%. |
| 12 | **Minor** | Births are not exposed to infant mortality in birth year | Small effect, but cohort-component convention usually applies birth survival | Either document simplification or apply infant birth-survival ratio to newborn cohort. |

---

## 3. Verdict on the open finding

### (a) Is the sum-vs-average diagnosis correct from config + `migration.py`?

**Yes. Confirmed.**

- `config/projection_config.yaml` labels `reference_intl_migration: 10051` as “Annual international migration (PEP 2023-2025 avg)”.
- The review package shows the PEP values: `2023 = 3,158`, `2024 = 4,083`, `2025 = 2,810`; sum `10,051`; mean `3,350.33`.
- `cohort_projections/core/migration.py` uses the value as an annual flow:  
  `annual_reduction = ref_intl * (1.0 - factor)`  
  `reduction_rate = annual_reduction / ref_pop`.

Therefore the first-year nominal CBO reduction is currently:

- Locked: `10,051 * 0.80 = 8,041` persons/year.
- Correct annual-average basis: `3,350.33 * 0.80 = 2,680` persons/year.

That is a roughly **3× over-suppression**.

### (b) Is the corrected run the right basis?

**Yes.**

The sensitivity changes only the numerator (`10051 → 3350.33`) while preserving the locked config and locked rate files. The control run reproduces the locked trajectory to the person (`max abs diff 0.0000`). That isolates the effect. The corrected result is therefore the right direct basis for disposition:

- Locked trough: `787,382` in 2028, `−1.50%`.
- Corrected trough: `797,911` in 2026, `−0.18%`.
- Locked 2055: `889,017`.
- Corrected 2055: `904,692`, up `+15,675`.

If the stated intent is “PEP 2023–2025 annual average,” the corrected numerator is the appropriate fix. A separate policy decision could choose a different annual reference, such as 2025-only `2,810` or 2024 `4,083`, but that would be a new assumption—not a correction of the stated average.

### (c) Should the locked numbers be published while this is undispositioned?

**No.**

This is a public-release blocker because:

1. The locked headline dip is materially dependent on the erroneous numerator.
2. The config and methodology labels are false as written.
3. The corrected sensitivity changes both the narrative shape and the endpoint.
4. Publishing now would require knowingly releasing numbers whose load-bearing assumption is under active challenge.

---

## 4. Go / no-go recommendation

**No-go for publishing the locked numbers as-is.**

The locked run is structurally coherent in many respects—state aggregation validates, components reconcile on a household basis, and the projection engine is largely consistent with the documented order of operations. But the CBO numerator error is large enough to change the public story from “North Dakota declines 1.5% by 2028” to “North Dakota is essentially flat near-term and then grows.”

Minimum conditions before public release:

1. Disposition the CBO numerator in ADR-068.
2. If correcting, rerun the locked production path and regenerate exports/figures/metadata.
3. Update all references to `10,051` and all “annual average” language.
4. Resolve or explicitly caveat the mortality race-specific/open-90+ issues.
5. Fix public components labeling for household-basis deaths.
6. Run final QA again against the corrected trajectory.

My recommended disposition is **correct-and-rerun**.