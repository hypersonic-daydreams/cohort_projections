# Methodology Comparison: ND State Data Center 2024 vs. Our Project

*Updated 2026-06-16 against the **ADR-068-corrected** locked production run (config sha `a6e0bfbc2d70be85`).
This re-syncs the figures from the superseded 2026-06-13 pre-correction run (`m2026r1` / commit `12fa6f9`),
which carried the `reference_intl_migration` (~3x) and open-ended 90+ survival errors that ADR-068 fixed;
the corrected near-term trajectory is a shallow, near-flat trough rather than a deeper dip. It also
supersedes the March 2026 three-scenario comparison: the public release is a single Baseline
(CBO-Adjusted) path (ADR-065), so the figures below are the corrected locked baseline, not the former
unadjusted trend-continuation baseline. The December 2025 IRS-migration version remains obsolete.*

## Summary

Both the North Dakota State Data Center (SDC) 2024 projections and our project use the
**cohort-component method** -- the demographic gold standard. Since December 2025, our model
has been substantially rebuilt: migration was switched from IRS county-to-county flows to
**residual migration from Census PEP** (the same conceptual method the SDC uses), race/ethnicity
detail was added, group quarters were separated, and dozens of targeted adjustments were
implemented. The public projection is now a single **Baseline (CBO-Adjusted)** path (ADR-065),
which layers CBO's January 2026 current-policy outlook -- a front-loaded reduction in net migration
plus a -5% fertility adjustment -- onto the cohort-component engine. The two sets of projections
still tell a broadly **similar long-run story of continued state growth**, but they now differ both
in magnitude and in near-term **trajectory shape**.

| Dimension | SDC 2024 | Our Project (ADR-068-corrected baseline, 2026-06-16) |
|-----------|----------|--------------------------|
| State 2025 | 796,989 | 799,358 |
| State 2045 | 925,101 | 866,590 |
| State 2050 | 957,194 | 883,225 |
| State 2055 | (horizon ends 2050) | 898,907 |
| Direction | Growth (+20.1% by 2050) | Growth (+10.5% by 2050; +12.5% by 2055), after a shallow 2025-2027 dip |
| Public path | Single projection | Single public baseline (CBO-Adjusted); High/Restricted retired to inactive internal sensitivities |

Two differences from the March comparison matter:

1. **The gap is modest and horizon-growing.** The corrected baseline gap is **~59,000 at 2045**
   and **~74,000 at 2050** (vs. ~60,000 at 2045 in March). The CBO migration adjustment lowers the
   corrected baseline below the former unadjusted baseline, so the gap to the SDC's undampened-growth
   path is larger. It remains far smaller than the ~170,000 December 2025 divergence, when the two
   models pointed in opposite directions.
2. **The near-term shape diverges.** The corrected baseline edges down from 799,358 (2025) to a shallow
   trough of **797,298 at 2027** (-0.26%) before recovering -- the intended CBO front-loaded migration
   ramp (f(2025)=0.20 rising to 0.91 by 2029; ADR-065), now near-flat after the ADR-068
   `reference_intl_migration` correction. The SDC path grows monotonically from 2020. Both paths still
   rise over the full horizon, so the long-run direction agrees even though the first few years do not.

---

## 1. Where the Methodologies Align

### 1.1 Core Method: Cohort-Component Projection

Both use the same fundamental demographic framework:

- Age cohorts forward through time
- Apply survival rates (mortality)
- Add births (fertility applied to reproductive-age females)
- Add/subtract net migration

This is the industry-standard method used by the Census Bureau, UN Population Division, and
state demographic offices worldwide.

### 1.2 Base Population Source

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Base year | Census 2020 | Census PEP Vintage 2025 (base year 2025) |
| Geography | State + 53 counties | State + 53 counties + 406+ places |

Both anchor to Census counts, though we use the most recent PEP estimates as our launch point
while the SDC starts from the 2020 decennial.

### 1.3 Survival Rates from CDC Life Tables

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Source | CDC life tables for ND, 2020 | ND-adjusted CDC life tables (ADR-053) |
| Application | By age and sex | By age, sex, and race |
| ND e0 (Male) | 74.2 | 75.37 |
| ND e0 (Female) | 80.0 | 80.76 |

Both use official CDC/NCHS life tables as the mortality foundation. Our values are slightly
higher due to use of a different vintage and ND-specific adjustment methodology.

### 1.4 Fertility Data Sources

Both incorporate:

- ND DHHS Vital Statistics (state-specific rates)
- CDC NVSS (national rates for blending/comparison)

The SDC uses 2018-2022 blended county rates; we use ND-adjusted CDC WONDER rates
(ADR-053) with a TFR of 1.863 (vs. national 1.621, ratio 1.15x).

### 1.5 Migration Estimation: Residual Method

Both now calculate migration as a residual:

```
Net Migration = Actual Population Change - Natural Increase (Births - Deaths)
```

This is the most important alignment change since the December 2025 comparison, where
our model used IRS county-to-county flows. We now use Census PEP component data for
2000-2024 across five historical periods (ADR-036).

### 1.6 Multi-Year Averaging for Stability

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Fertility | 2018-2022 (blended) | Constant (baseline), +/-5% (scenarios) |
| Migration | 4 five-year periods (2000-2020) | 5 five-year periods (2000-2024), BEBR averaging (ADR-036) |

Both recognize that single-period rates are too volatile for projections.

### 1.7 Bakken / Oil-County Migration Dampening

Both models explicitly dampen oil-boom-era migration for Bakken counties:

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Counties affected | "Bakken region counties" | Williams, McKenzie, Mountrail, Dunn, Stark |
| Dampening level | ~60% of 2000-2020 historical | 40-50% period-specific factors (ADR-040/051) |
| Rationale | Boom "unlikely to occur again" | Oil-boom migration was structurally atypical |

Both models agree that the 2005-2015 Bakken oil boom produced migration patterns that
should not be projected forward at face value.

---

## 2. Where the Methodologies Diverge

### 2.1 Race/Ethnicity Detail

**This remains the most significant structural divergence.**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Race categories | **None** -- total population only | **6 categories** (White NH, Black NH, AIAN NH, Asian/PI NH, Two+, Hispanic) (ADR-047) |
| Cohorts projected | ~182 (91 ages x 2 sexes) | **1,092** (91 ages x 2 sexes x 6 races) |
| Fertility by race | No | Yes -- 6 categories (ADR-053) |
| Mortality by race | No | Yes -- 6 categories (ADR-053) |

**Implications:**

- Our project can project how North Dakota's demographic composition will change over time
- Our project can show differential growth by race/ethnicity (e.g., growing Hispanic
  and Asian/PI populations, AIAN trends on reservations)
- SDC cannot address these questions with their current framework
- Our approach requires substantially more data but provides richer policy-relevant outputs

### 2.2 Group Quarters Separation

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| GQ treatment | Combined with household population | **Separated** (ADR-055) |
| GQ projection | Implicit in totals | Held constant at 2025 levels (30,884 persons) |
| Rate computation | On total population | On household-only population (Phase 2) |

Our model explicitly separates group quarters (military barracks, college dorms, correctional
facilities) from the household population before computing migration residuals. This prevents
large GQ changes (e.g., NDSU dorm expansions in Cass County, +2,929 persons 2020-2024) from
distorting migration signals for the general population.

### 2.3 Migration Adjustments

Both models apply adjustments to raw migration rates, but our model has a more extensive
and documented set:

| Adjustment | SDC 2024 | Our Project |
|------------|----------|-------------|
| Bakken dampening | ~60% of historical | 40-50% period-specific (ADR-040/051) |
| Male migration dampening | "Further reduced" (undocumented factor) | 0.80 factor for 2005-2015 periods |
| College-age smoothing | "Additional adjustments" | 50/50 blend with statewide for ages 15-29 in 11 college counties (ADR-049, extended to 25-29 by ADR-061 D1; Williams removed by ADR-067) |
| Reservation recalibration | Not documented | PEP-anchored for Benson, Sioux, Rolette (ADR-045) |
| Age-aware rate caps | Not documented | +/-15% ages 15-24, +/-8% all others (ADR-043) |
| Convergence schedule | Not documented | 5-10-5 schedule: 5yr recent-to-medium, 10yr hold, 5yr to long-term |
| Ward County floor | Not applicable | High-growth scenario prevents decline (ADR-052) |
| BEBR high-growth increment | Not applicable | Additive migration boost for high scenario (ADR-046) |
| Manual adjustments | Yes -- per "Adjustments" sheets | **None** -- fully algorithmic |

The SDC applies expert-judgment manual adjustments via spreadsheet worksheets. Our model
uses exclusively algorithmic adjustments, each documented in an Architecture Decision Record.

### 2.4 Mortality Improvement

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Improvement | None visible (constant rates) | Lee-Carter style: 0.5%/yr (all scenarios) |
| Over 30 years | No improvement | ~14% cumulative reduction in age-specific mortality |

The SDC holds survival rates constant at 2020 levels. Our model incorporates gradual
mortality improvement, consistent with long-term historical trends and the approach used
by the Social Security Administration and Census Bureau.

### 2.5 Scenario Structure and the Public Baseline

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Public path | Single projection | **Single public baseline (CBO-Adjusted)** (ADR-065) |
| Fertility | Fixed | -5% CBO adjustment on the public baseline |
| Migration | Fixed (dampened) | Convergence-based residual migration **plus** the CBO additive reduction schedule (ADR-050/065), front-loaded 2025-2029 |
| Mortality | Fixed | 0.5%/yr improvement |

The public release carries **one** path. After ADR-065, the model's former unadjusted
trend-continuation baseline, High Growth, and Restricted Growth variants were retired to
**inactive internal sensitivities**: the engine can still produce them, but they are not
maintained at the locked vintage and are **not** part of the public package. Their on-disk
outputs are stale (2026-02-26, pre-ADR-066) and must not be cited as current planning bounds.

The public **Baseline (CBO-Adjusted)** is therefore the comparison path throughout this
document:

- **Base:** Census PEP Vintage 2025 (799,358; ADR-066), single-year age/sex/race.
- **Fertility:** -5% relative to the held-constant ND-adjusted rates (CBO January 2026 outlook).
- **Migration:** residual migration with BEBR multi-period averaging and 5-10-5 convergence,
  then the ADR-050 additive CBO reduction applied on a front-loaded schedule (most aggressive in
  2025-2026, relaxing through 2029). This schedule is what produces the shallow 2025-2027 dip.
- **Mortality:** 0.5%/yr Lee-Carter-style improvement.

### 2.6 Base Population Processing

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Age detail | 5-year groups (18 groups) | Single-year of age via Sprague interpolation (91 cohorts, 0-90+) (ADR-048) |
| Base year | 2020 | 2025 (PEP Vintage 2025) |
| Projection step | 5-year intervals | Annual |

Our model projects in single-year steps and single-year-of-age cohorts, avoiding the
aggregation artifacts that can arise from 5-year groupings.

### 2.7 Geographic Granularity

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Levels | State, 8 regions, 53 counties | State, 53 counties, **406+ places** |
| State derivation | Independent state projection | State = sum of counties (ADR-054) |
| Place projections | No | Yes (threshold: 500+ population) |

Our state total is derived bottom-up as the sum of 53 county projections, ensuring
internal consistency. The SDC appears to project state and counties independently.

### 2.8 Validation and Quality Assurance

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Automated tests | Not documented | **1,570 tests** (unit, integration, validation) |
| Backtesting | Not documented | Rolling-origin backtests, 34 folds (ADR-057) |
| Cross-validation | Not documented | Housing-unit method cross-validation (ADR-060) |
| Documentation | PDF report | ADRs, SOPs, methodology docs with full provenance |

---

## 3. Results Comparison

### 3.1 State-Level Projections

#### SDC 2024

| Year | Population | Change from 2020 |
|------|-----------|-------------------|
| 2020 | 779,094 | -- |
| 2025 | 796,989 | +2.3% |
| 2030 | 831,543 | +6.7% |
| 2035 | 865,397 | +11.1% |
| 2040 | 890,424 | +14.3% |
| 2045 | 925,101 | +18.7% |
| 2050 | 957,194 | +22.9% |

#### Our Project (locked public Baseline, CBO-Adjusted, from 2025 base)

| Year | Baseline | Change from 2025 |
|------|----------|-------------------|
| 2025 | 799,358 | -- |
| 2027 | 797,298 | -0.26% (trough) |
| 2030 | 804,657 | +0.66% |
| 2035 | 826,051 | +3.34% |
| 2040 | 848,259 | +6.12% |
| 2045 | 866,590 | +8.41% |
| 2050 | 883,225 | +10.49% |
| 2055 | 898,907 | +12.45% |

The corrected baseline edges down for the first two years (the CBO front-loaded migration ramp),
bottoms at a shallow 797,298 in 2027 (-0.26%), then grows steadily to 898,907 by 2055. High Growth and
Restricted Growth are retired internal sensitivities (§2.5) and are not shown.

#### Direct Comparison (Baseline vs. SDC, Overlapping Years)

| Year | SDC 2024 | Our Baseline | Difference | Diff % |
|------|----------|--------------|------------|--------|
| 2025 | 796,989 | 799,358 | +2,369 | +0.3% |
| 2030 | 831,543 | 804,657 | -26,886 | -3.2% |
| 2035 | 865,397 | 826,051 | -39,346 | -4.5% |
| 2040 | 890,424 | 848,259 | -42,165 | -4.7% |
| 2045 | 925,101 | 866,590 | -58,511 | -6.3% |
| 2050 | 957,194 | 883,225 | -73,969 | -7.7% |

Both models agree on the long-run direction: **North Dakota grows**. But the gap is larger and
horizon-growing relative to the March comparison, for two reasons. First, the CBO migration
adjustment (ADR-065) lowers the public baseline below the former unadjusted baseline that the
March numbers reflected, widening the distance to the SDC's undampened-growth path. Second, the
2030 gap is exaggerated by the near-term dip: our baseline is at its CBO-suppressed trough while
the SDC is already several years into monotonic growth. By 2050 the gap is ~74,000 (-7.7%), still
far smaller than the ~170,000 December 2025 divergence when the two models pointed in opposite
directions.

### 3.2 Key County Comparisons (2025-2050)

#### Cass County (Fargo)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 194,767 | 201,794 | +7,027 |
| 2030 | 211,322 | 212,145 | +823 |
| 2035 | 227,406 | 225,856 | -1,550 |
| 2040 | 239,681 | 239,079 | -602 |
| 2045 | 255,799 | 249,864 | -5,935 |
| 2050 | 272,878 | 259,227 | -13,651 |

Both project strong Cass County growth (our corrected baseline +33.2% to 268,723 by 2055), but the
SDC projects faster acceleration. Our model starts higher (201,794 vs. 194,767 in 2025, reflecting
PEP Vintage 2025 data) but grows more conservatively due to GQ-corrected migration (ADR-055 Phase 2
removed NDSU dorm growth from the migration signal) and the CBO migration adjustment.

#### Burleigh County (Bismarck)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 100,657 | 103,251 | +2,594 |
| 2030 | 108,057 | 104,830 | -3,227 |
| 2035 | 114,646 | 107,995 | -6,651 |
| 2040 | 117,739 | 111,227 | -6,512 |
| 2045 | 123,366 | 114,149 | -9,217 |
| 2050 | 128,663 | 117,019 | -11,644 |

Both project continued Burleigh County growth (our corrected baseline +15.9% to 119,664 by 2055).
The SDC projects steeper growth after 2030; our near-term figures are also held down by the CBO
migration ramp, which is most aggressive in 2025-2029.

#### Grand Forks County

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 74,966 | 74,501 | -465 |
| 2030 | 77,443 | 73,783 | -3,660 |
| 2035 | 79,159 | 74,101 | -5,058 |
| 2040 | 80,561 | 74,533 | -6,028 |
| 2045 | 81,238 | 73,889 | -7,349 |
| 2050 | 81,582 | 72,794 | -8,788 |

The SDC projects modest growth while our corrected baseline is roughly flat-to-slightly-declining
(-3.3% over the full horizon, to 72,011 by 2055). The locked-config Grand Forks trajectory is
**less severe** than the older pre-lock figures (which fell to 67,501 by 2050): under the 25-29
smoothing extension (ADR-061 D1), the Grand Forks backtest sentinel improved sharply (MAPE 11.12 ->
7.39 on the clean raw-base matrix). The remaining decline is **assumption-driven, not a county-rate
artifact**: ADR-067 F4 attributes ~52% of it to the disclosed CBO international-migration assumption
(Grand Forks' 2023-2025 growth was on international migration, exactly the component CBO reduces) and
~41% to the long-run convergence stance. See `docs/reviews/2026-06-13-divergent-counties-methods-and-framing.md`.

#### Ward County (Minot)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 72,066 | 68,233 | -3,833 |
| 2030 | 74,545 | 65,820 | -8,725 |
| 2035 | 77,503 | 64,993 | -12,510 |
| 2040 | 79,852 | 64,096 | -15,756 |
| 2045 | 82,831 | 62,949 | -19,882 |
| 2050 | 85,975 | 61,540 | -24,435 |

**Ward County is the largest proportional divergence:** the SDC projects ~+23% growth while our
corrected baseline declines -12.1% (to 59,986 by 2055). This is **driven by observed data, not a
method artifact**: Ward's net migration was negative in every year 2020-2025 (cumulative ~-3,950),
and ADR-067 F4 found that all method/assumption levers combined move Ward only ~+3,660 at 2055
against a ~9,250 projected decline -- the majority of the decline survives every variant. The
institutional anchors (Minot AFB, MISU) are held constant (GQ), so the decline is in the
household/working-age population. The SDC's +23% predates the post-2020 reversal and the
Vintage-2025 base. The retired High Growth sensitivity carried a Ward County floor (ADR-052) that
prevented decline, but it is not part of the public baseline.

#### Williams County (Williston)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 43,807 | 41,767 | -2,040 |
| 2030 | 46,170 | 42,537 | -3,633 |
| 2035 | 48,635 | 45,935 | -2,700 |
| 2040 | 50,953 | 49,900 | -1,053 |
| 2045 | 53,957 | 54,478 | +521 |
| 2050 | 56,047 | 59,452 | +3,405 |

Both models project Williams growth; our corrected baseline is initially below the SDC but
overtakes it by 2050 and reaches 64,234 (+53.8%) by 2055. This is the county most changed by the
locked config: ADR-067 **removed** Williams from the college-age smoothing list (Williston State
College is only 1.5% of population, below ADR-061's own 2.5% threshold, so its young-adult
migration is oil-economic, not enrollment-driven). Removal improved Williams' backtest accuracy
(county MAPE 23.38 -> 22.48 on the clean raw-base matrix) and accounts for ~+8,978 of the locked
baseline's gain at 2055. Crucially, the growth is **conservative on migration**: projected net
in-migration settles near +388/yr, roughly half the recent observed +780/yr (2023-2025); the rest
is natural increase from a very young age structure. Williams is consistent with peer oil counties
McKenzie (+79%) and Billings (+50%). See `docs/reviews/2026-06-13-divergent-counties-methods-and-framing.md`.

---

## 4. Root Cause Analysis: Why the Projections Diverge

### 4.1 The Gap is Much Smaller Than Before, But Horizon-Growing

In the December 2025 comparison, the projections diverged by ~170,000 people by 2045 and
pointed in opposite directions (SDC: growth, ours: decline). The corrected gap is ~59,000 by 2045
and ~74,000 by 2050, with both models pointing in the same long-run direction: growth. The gap
is larger than the March comparison's ~60,000 mainly because the public baseline now carries the
CBO migration adjustment (ADR-065), which lowers it below the former unadjusted baseline that the
March numbers reflected.

**The near-term shape now differs too.** The corrected baseline dips to a shallow 2027 trough (797,298)
before recovering, while the SDC grows monotonically. The dip is entirely the CBO front-loaded
migration ramp: the Stage-1 sensitivity decomposition (ADR-065 defensibility memo) showed that with
the CBO migration adjustment removed, our trajectory rises monotonically with no dip. So the 2030
gap (-3.2%) overstates the structural disagreement -- it compares our CBO-suppressed trough year
against an SDC path already several years into growth.

**What changed since December 2025:**

1. **Migration method**: We switched from IRS county-to-county flows (which showed net
   out-migration for 2019-2022) to Census PEP residual migration (which shows positive
   net migration across the full 2000-2024 period after averaging). This single change
   accounts for the majority of the convergence between the two models.

2. **Base year update**: Moving from 2020 to 2025 as the base year, using PEP Vintage 2025,
   gives us a higher starting population and incorporates recent growth.

3. **Multi-period averaging**: Using five periods (2000-2024) with BEBR averaging (ADR-036)
   incorporates the positive migration signal from the early-to-mid 2000s and 2020s.

### 4.2 The Remaining ~59,000-74,000 Gap

A useful way to size the components is the ADR-067 F4 forward decomposition of the baseline
against the former unadjusted baseline at 2050: the CBO migration adjustment removes roughly
-23,000, the -5% CBO fertility adjustment roughly -13,000, the GQ-fraction calibration (f=0.75)
adds roughly +3,000, and the rejected convergence hold (ADR-061 D3) would have added a further
+9,000 had it been deployed. (**These component magnitudes are from the pre-correction 2026-06-13
F4 decomposition and have not been re-run against the ADR-068-corrected baseline; the CBO-migration
component in particular shrinks materially after the `reference_intl_migration` fix -- consistent
with the corrected baseline sitting higher and the smaller ~59,000-74,000 gap above -- so read them
as indicative of the *ordering* of effects, not exact post-correction magnitudes; see
`docs/plans/f4-decomposition-reproducibility.md` for the re-run procedure and the planned fix.**) The CBO
adjustments are thus the single largest reason the corrected baseline sits below the SDC's
undampened-growth path. Beyond those, the remaining divergence is
driven by:

#### Migration Rate Magnitude

Even though both models use residual migration, they produce different net migration rates:

- **SDC**: Projects net in-migration of ~4,000-6,000/year after dampening
- **Our model**: Projects more moderate net in-migration that declines over time due to
  convergence interpolation

Our convergence schedule (5-10-5) gradually moves rates from recent observed levels toward
long-term equilibrium, producing a decelerating growth trajectory. The SDC appears to hold
dampened rates constant throughout the projection horizon.

#### Mortality Improvement vs. Constant Mortality

Our 0.5%/year mortality improvement slightly increases our projected population relative
to a no-improvement scenario. However, the SDC's constant mortality (no improvement) is
offset by their higher migration assumptions, so this factor partially narrows the gap
rather than widening it.

#### GQ Correction Effect

Our GQ separation (ADR-055 Phase 2) removes institutional population changes from the
migration signal. This makes projections more conservative for counties with recent GQ
growth (especially Cass, Grand Forks) and slightly changes the statewide total.

The Phase 2 GQ correction reduced the state baseline by approximately 1.5 percentage points
compared to a model without GQ separation.

#### Base Year Difference

The SDC starts from 2020 (779,094) while we start from 2025 (799,358). The SDC must
project through 2020-2025 to reach their 2025 estimate of 796,989 -- already 2,369 below
our PEP-anchored starting point. This initial difference persists and can compound.

### 4.3 County-Level Divergence Patterns

The divergence is not uniform across counties:

- **Urban growth counties** (Cass, Burleigh): Both models agree on growth; SDC is more
  optimistic in the later years
- **College counties** (Grand Forks, Ward): Largest divergence; our college-age smoothing
  and GQ correction produce more conservative trajectories
- **Oil counties** (Williams, McKenzie): Both dampen; our model eventually projects faster
  growth due to convergence dynamics and recent PEP data
- **Rural decline counties**: Generally similar -- both project continued decline in most
  rural counties

---

## 5. Which Projection is More Realistic?

### Arguments for the SDC's More Optimistic View

- **Track record**: The SDC has decades of experience projecting ND population
- **Expert judgment**: Their manual adjustments incorporate local knowledge difficult to
  encode algorithmically
- **Economic momentum**: North Dakota's energy sector and economic development initiatives
  could sustain stronger in-migration than trend data alone suggests
- **Historical precedent**: The 2020-2024 period showed population growth after brief
  COVID-related disruption
- **Urban attraction**: Fargo, Bismarck, and other cities continue to attract workers from
  the broader region

### Arguments for Our More Moderate View

- **Newer data**: We incorporate PEP data through 2024, two years beyond the SDC's 2022
  cutoff for fertility and four years beyond their 2020 migration endpoint
- **Algorithmic consistency**: No manual adjustments means the methodology is fully
  reproducible and auditable
- **GQ correction**: Separating institutional populations prevents misleading migration
  signals (e.g., NDSU dorm construction appearing as Cass County in-migration)
- **Convergence**: The 5-10-5 schedule reflects the demographic principle that extreme
  rates tend to regress toward means over long horizons
- **Mortality improvement**: Including gradual mortality improvement is consistent with
  long-run historical trends and actuarial practice
- **Validation**: 1,570 automated tests, rolling-origin backtests, and housing-unit
  cross-validation provide quantitative evidence for model performance

### The Likely Reality

**Both projections point to growth; they differ on pace and on the federal-policy assumption.**
With the public release now baseline-only, the honest framing is a comparison of two paths rather
than a four-point scenario envelope:

- **SDC 2024**: ~925K by 2045 (~957K by 2050) -- an undampened-growth path that does not encode the
  CBO current-policy migration reduction.
- **Our locked Baseline (CBO-Adjusted)**: ~855K by 2045 (~873K by 2050, ~889K by 2055) -- the same
  cohort-component engine plus CBO's January 2026 outlook, which front-loads a migration reduction
  and trims fertility -5%.

Much of the ~70-84K gap is therefore an **assumption** difference (the CBO adjustments remove
roughly -36,000 by 2050; §4.2), not a structural methodological disagreement. A reader who rejects
the CBO current-policy assumption should read our baseline as a lower-migration path and the SDC as
closer to a no-policy-change path. The model can still produce higher/lower sensitivity variants
internally, but they are not maintained at the locked vintage and are not published.

### What Would Need to Happen for Each Path

**For the SDC's higher trajectory to materialize:** sustained net in-migration of ~4,000-6,000/year
for 20+ years, continued strong energy employment, and -- critically -- federal immigration running
above the CBO current-policy path (91% of recent ND net migration has been international).

**For our Baseline to materialize:** the CBO front-loaded migration reduction holds in 2025-2029,
then net migration recovers to a moderate, gradually-converging positive level; natural increase
stays positive but narrows as the population ages.

### Validated Strengths and Acknowledged Weaknesses

The clean raw-base walk-forward matrix (2026-06-11, ADR-067) gives an honest profile of where the
locked method is strong and where it is not. This matters because **neither model's long-horizon
state magnitude can be directly validated**, and the CBO adjustments postdate the entire historical
record (they cannot be backtested at all; methodology.md §10.4).

**Where the method is strong (backtest-validated):**

- **Recent-origin state accuracy.** State APE is ~1.0% at short horizons and ~1.5% at medium
  horizons from recent launch years -- the model tracks the state total well when launched from
  recent data.
- **Low signed bias.** Recent-origin signed bias is +0.97 (locked config) -- no large systematic
  over- or under-projection at the state level in backtests.
- **College-county accuracy.** The 25-29 smoothing extension materially improved the counties under
  the most public scrutiny: Grand Forks sentinel MAPE 11.12 -> 7.39, Cass 9.35 -> 7.65, Ward 13.93 ->
  13.24.

**Where the method is weak (acknowledged):**

- **Long-horizon state magnitude.** Error grows with horizon, and the long-run state total is exactly
  where the model diverges most from the SDC (-8.8% by 2050) and where no out-of-sample check exists.
  Treat the 2050+ state level as the least certain output.
- **Oil/Bakken counties.** Boom-bust volatility makes these intrinsically hard: Williams' backtest
  MAPE is ~22%, the highest of the sentinels, even after the ADR-067 smoothing-removal improvement.
- **Forward CBO assumptions.** The -5% fertility and additive migration reduction are unvalidatable
  policy assumptions; they drive the near-term dip and the lower long-run total.

---

## 6. SDC Source File Analysis

This section documents detailed findings from analysis of the SDC's actual source files
and working spreadsheets, providing insight into their specific methodological choices and
calculations.

### Data Sources Used

#### Base Population

- **Census 2020**: Used as the authoritative base population
- **Census 2010**: Used for historical comparison and migration rate calculation
- **Population Estimates Program (PEP)**: Used cc-est2019-agesex-38 for interim estimates

#### Fertility Data

- **Source**: North Dakota Vital Statistics, Department of Health
- **Time Period**: 2016-2022 (with emphasis on 2018-2022 for rate calculation)
- **Data File**: "2018-2022 ND Res Birth for Kevin Iverson.xlsx" -- prepared September 15, 2023 by Vital Records
- **Categories**: Births by county of residence, age group of mother (Under 20, 20-25, 25-29, 30-34, 35-39, 40-44, 45+)
- **Female Population**: "Average Female Count 2018 to 2022.xlsx" -- average of 2018 and 2022 female populations by age group by county
- **National Reference**: NVSS Report (nvsr72-01.pdf) -- National Vital Statistics Reports Volume 71, Number 1, dated January 31, 2023

**Fertility Rate Calculation Method:**

1. Average female population by age group (10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49) for each county from 2018-2022
2. Sum births by mother's age group for 2018-2022 period
3. Calculate 5-year fertility rate = (Total births in age group) / (Average female population in age group)
4. Rates "smoothed to reduce anomalies" -- blended with state and national rates for stability

**Key Fertility Data Points:**

- State total: ~189,000 to ~197,000 females of childbearing age (10-49)
- Birth data includes suppression ("NR") for small cells for privacy
- 2018-2022 total births by year: 10,630 (2018), 10,447 (2019), 10,051 (2020), 10,111 (2021), 9,557 (2022)
- Sex ratio at birth: ~104.8 males per 100 females (51.2% male, 48.8% female)

#### Survival/Mortality Data

- **Source**: CDC Life Tables for North Dakota, 2020
- **Publication**: NVSS Report nvsr71-02
- **Files**: ND1.xlsx (Total), ND2.xlsx (Males), ND3.xlsx (Females), ND4.xlsx (Standard Errors)

**Life Table Structure (Standard life table columns):**

| Column | Description |
|--------|-------------|
| qx | Probability of dying between ages x and x+1 |
| lx | Number surviving to age x (from radix of 100,000) |
| dx | Number dying between ages x and x+1 |
| Lx | Person-years lived between ages x and x+1 |
| Tx | Total person-years lived above age x |
| ex | Expectation of life at age x |

**5-Year Survival Rates (from Projections_Base_2023.xlsx):**

| Age Group | Male | Female |
|-----------|------|--------|
| Under 5 | 0.9915 | 0.9946 |
| 5-9 | 0.9994 | 0.9994 |
| 10-14 | 0.9987 | 0.9994 |
| 15-19 | 0.9982 | 0.9983 |
| 20-24 | 0.9949 | 0.9980 |
| 25-29 | 0.9927 | 0.9972 |
| 30-34 | 0.9897 | 0.9961 |
| 35-39 | 0.9876 | 0.9950 |
| 40-44 | 0.9860 | 0.9936 |
| 45-49 | 0.9808 | 0.9914 |
| 50-54 | 0.9776 | 0.9878 |
| 55-59 | 0.9652 | 0.9817 |
| 60-64 | 0.9521 | 0.9725 |
| 65-69 | 0.9335 | 0.9593 |
| 70-74 | 0.8972 | 0.9405 |
| 75-79 | 0.8355 | 0.9092 |
| 80-84 | 0.7353 | 0.8525 |
| 85+ | 0.5428 (calculated) | 0.6998 (calculated) |

**Life Expectancy at Birth (2020):**

- Total: 76.9 years
- Males: 74.2 years
- Females: 80.0 years

**85+ Survival Rate Calculation (from spreadsheet):**

The SDC calculated the 85+ open-ended age group survival rate using:
```
85+ Survival Rate = (85+ survivors at t+5) / (90+ survivors at t+5) = 305,579 / 121,647 = 0.398 (approximate)
```
Note: They used 0.5428 for males and 0.6998 for females in actual projections.

#### Migration Data

- **Source Files**:
  - "Mig Rate 2000-2020_final.xlsx" -- Final averaged migration rates
  - "Mig Rates 2000-2020.xlsx" -- Working calculations
- **Time Periods Analyzed**:
  - 2000-2005
  - 2005-2010
  - 2010-2015
  - 2015-2020
- **Method**: Residual method (calculated as difference between actual and expected population)

### Migration Rate Calculation Methodology

The SDC calculated migration rates using the following detailed process:

#### Step 1: Calculate Expected Population (No Migration)

For each 5-year period, calculate what the population would be with zero migration:
```
Expected_Pop[t+5] = Pop[t] * Survival_Rate + Births_to_Cohort
```

#### Step 2: Calculate Migration Residual

```
Migration_Rate = (Actual_Pop[t+5] - Expected_Pop[t+5]) / Pop[t]
```

#### Step 3: Average Across Four Periods

The final migration rates used are averages of the four 5-year periods (2000-2005, 2005-2010, 2010-2015, 2015-2020).

**Example Migration Rates by Age Group (State Level, Averaged 2000-2020):**

| Age Group | Male Rate | Female Rate |
|-----------|-----------|-------------|
| Under 5 | +0.108 | -0.003 |
| 5-9 | +0.049 | +0.011 |
| 10-14 | +0.054 | +0.017 |
| 15-19 | +0.173 | +0.081 |
| 20-24 | +0.328 | +0.117 |
| 25-29 | -0.119 | -0.243 |
| 30-34 | +0.059 | +0.083 |
| 35-39 | +0.039 | -0.022 |
| 40-44 | +0.036 | +0.004 |
| 45-49 | +0.032 | -0.001 |
| 50-54 | +0.042 | +0.005 |
| 55-59 | +0.024 | -0.008 |
| 60-64 | -0.007 | -0.011 |
| 65-69 | -0.049 | -0.023 |
| 70-74 | -0.087 | -0.033 |
| 75-79 | -0.100 | -0.085 |
| 80-84 | -0.160 | -0.132 |
| 85+ | -0.148 | -0.089 |

**Key Migration Patterns Identified:**

- **Strong in-migration**: Ages 20-24 (college/workforce entry), males especially
- **Out-migration**: Ages 25-29 for females, 65+ for both sexes
- **Net male migration**: 0.034 (males), -0.019 (females) -- significant gender imbalance

#### Migration Rate Adjustments (The "60% Dampening")

From the methodology writeup, the SDC made several critical adjustments:

1. **Bakken Dampening**: "Given the significant in-migration that North Dakota experienced from 2010 to 2020, the rates were typically reduced to about 60 percent of what was found" because the Bakken Oil Boom "is unlikely to occur again"

2. **College-Age Adjustment**: "Counties with significant college age populations typically required additional adjustments as the algorithm tends to not capture the in- and out-migration of college age residents as well as it should"

3. **Male Migration Adjustment**: "The rate of male migration was further reduced than female migration as the pattern found from 2000 to 2020 when in-migration was dominated by males is unlikely to continue into the future and would have resulted in unrealistic sex ratio in future years"

4. **Bakken Region Counties**: "The rate of migration in counties in the Bakken region that experienced significant growth during the last decade also were adjusted to a lower rate"

### Projection Workbook Structure

The main projection workbook (Projections_Base_2023.xlsx) contains 45 sheets organized as follows:

| Sheet Category | Purpose |
|----------------|---------|
| Notes | Process documentation |
| 5-Year Survival Rate By Sex | Survival rates from life tables |
| Census 2010, Census 2020 | Base population data |
| Senthetic_2015_2 | Interpolated 2015 estimates |
| Mig_Rate | Averaged migration rates by age/sex/county |
| Fer 2020-2025, Fer_2025-30, etc. | Fertility calculations per period |
| Nat_Grow 2020-2025, etc. | Natural growth (births - deaths) |
| Adjustments 2020-2025, etc. | Manual adjustment factors |
| 2020-2025 Migration, etc. | Migration applied |
| 2025 Pro, 2030 Pro, etc. | Final projections by period |

**Projection Process (from Notes sheet):**

1. Start with Census base (adjusted in 5-year increments)
2. Apply fertility rate by age of mother to get ages 0-4 population
3. Apply survival rate by age group and sex by county
4. Apply migration rate by age, sex, and county
5. Apply manual adjustments for "unexpected patterns of natural growth"
6. Output next 5-year projection

### Comparison to Our Methodology

#### Where SDC Source Files Confirm Alignment

| Aspect | SDC Source Files | Our Approach |
|--------|------------------|--------------|
| Cohort-component structure | 18 age groups x 2 sexes x 53 counties = 1,908 cells | Same structure, plus 6 race categories = 11,448 cells |
| CDC life tables | ND-specific, 2020 | ND-adjusted CDC, configurable year |
| 5-year survival rates | Calculated from single-year qx | Same approach |
| Fertility by mother's age | 7 age groups (10-14 through 45-49) | Same age groups |
| Migration method | Residual from Census | Residual from Census PEP (aligned since Feb 2026) |

#### Where SDC Source Files Reveal Key Differences

| Aspect | SDC Approach (from source files) | Our Approach |
|--------|----------------------------------|--------------|
| **Migration time period** | 2000-2020 (4 periods averaged) | 2000-2024 (5 periods, BEBR averaged) |
| **Migration adjustment** | 60% dampening + manual | Algorithmic: oil dampening, male dampening, college smoothing, rate caps, convergence |
| **GQ treatment** | Combined in totals | Separated, held constant, rates computed on HH-only |
| **Manual adjustments** | Yes -- spreadsheet "Adjustments" sheets | No manual adjustments |
| **Mortality improvement** | None visible (constant rates) | Lee-Carter style 0.5%/year |
| **Race/ethnicity** | None | 6 categories |
| **Age resolution** | 5-year groups | Single-year of age |
| **Projection step** | 5-year | Annual |

### Key Formulas Extracted

**Natural Growth Calculation:**
```
Natural_Growth[county, age, sex] = Population[t] * Survival_Rate[age, sex] + Births[county, age_mother]
```

**Migration Application:**
```
Population[t+5] = Natural_Growth * (1 + Migration_Rate[county, age, sex])
```

**85+ Survival (Open-ended):**
```
85+_Survivors[t+5] = 85+_Pop[t] * 0.5428 (males) or 0.6998 (females)
```

### Data Quality Observations

From examining the source files:

1. **Small cell suppression**: Birth data uses "NR" (Not Reported) for privacy in small counties
2. **Rounding**: Some intermediate calculations appear rounded
3. **Interpolation**: 2015 population was "synthetic" (interpolated between 2010 and 2020)
4. **Date stamps**: Files dated December 2023 through January 2024, final methodology dated March 7, 2024

---

## 7. Summary Table

| Dimension | SDC 2024 | Our Project (March 2026) | Assessment |
|-----------|----------|--------------------------|------------|
| Core method | Cohort-component | Cohort-component | **Aligned** |
| Base year | Census 2020 | PEP Vintage 2025 | **Divergent** -- Ours more recent |
| Age resolution | 5-year groups | Single-year of age (ADR-048) | **Divergent** -- Ours finer |
| Projection step | 5-year | Annual | **Divergent** -- Ours finer |
| Race/ethnicity | None | 6 categories (ADR-047) | **Divergent** -- Ours more detailed |
| Geography | State + counties | State + counties + 406+ places | **Divergent** -- Ours more granular |
| State derivation | Independent | Bottom-up sum of counties (ADR-054) | **Divergent** -- Ours internally consistent |
| Migration method | Residual (Census) | Residual (Census PEP) | **Aligned** (since Feb 2026) |
| Migration periods | 2000-2020 (4 periods) | 2000-2024 (5 periods) | **Partially aligned** |
| Bakken dampening | ~60% | 40-50% period-specific (ADR-040/051) | **Aligned** in concept |
| Male migration dampening | Yes (undocumented factor) | 0.80 for 2005-2015 | **Aligned** in concept |
| College-age adjustment | Yes (undocumented) | 50/50 statewide blend, ages 15-29, 11 counties (ADR-049/061 D1; ADR-067) | **Aligned** in concept |
| GQ separation | None | Separated + held constant (ADR-055) | **Divergent** -- Ours more rigorous |
| Reservation calibration | Not documented | PEP-anchored (ADR-045) | **Divergent** |
| Migration convergence | Not documented | 5-10-5 schedule | **Divergent** |
| Age-aware rate caps | Not documented | +/-15% (15-24), +/-8% (others) (ADR-043) | **Divergent** |
| Fertility source | ND DHHS + NVSS blended | ND-adjusted CDC WONDER (ADR-053) | **Partially aligned** |
| Fertility by race | No | Yes (6 categories) | **Divergent** |
| Mortality improvement | None (constant) | 0.5%/yr Lee-Carter style | **Divergent** |
| Mortality by race | No | Yes (6 categories) | **Divergent** |
| Manual adjustments | Yes | None (fully algorithmic) | **Divergent** |
| Public path | 1 (single projection) | 1 public Baseline (CBO-Adjusted); High/Restricted retired to inactive sensitivities | **Aligned** (both single public path) |
| Forward policy adjustment | None | CBO Jan-2026 current-policy: front-loaded migration reduction + -5% fertility (ADR-065) | **Divergent** -- largest single source of the gap |
| Automated tests | Not documented | 1,570 tests | **Divergent** |
| Backtesting | Not documented | Rolling-origin, 34 folds (ADR-057) | **Divergent** |
| Cross-validation | Not documented | Housing-unit method (ADR-060) | **Divergent** |
| Documentation | PDF report | ADRs + SOPs + metadata | **Divergent** |
| **State 2045** | **925,101** | **866,590 (Baseline)** | **-6.3% gap** |
| **State 2050** | **957,194** | **883,225 (Baseline)** | **-7.7% gap** |
| **State 2055** | **(horizon ends 2050)** | **898,907 (Baseline, +12.5%)** | **n/a** |
| **Near-term shape** | **Monotonic growth** | **shallow 2025-2027 dip (CBO ramp), then growth** | **Divergent (near-term)** |
| **Direction (long-run)** | **Growth** | **Growth** | **Aligned** |

---

## 8. Recommendations

### 8.1 Present the Public Baseline, with the SDC as External Comparison

The public release is baseline-only (ADR-065), so the four-point scenario envelope used in the
March comparison no longer applies (High/Restricted are retired internal sensitivities at a stale
vintage). For stakeholder communication, present:

- **Public planning path**: Our locked Baseline (CBO-Adjusted) -- ~873K by 2050, ~889K by 2055.
- **External comparison point**: SDC 2024 -- ~957K by 2050 -- as a higher path that does not encode
  the CBO current-policy migration reduction.

The honest message is that most of the difference is a single, disclosed federal-policy assumption
(§4.2), not a methodological disagreement, and that long-horizon state magnitude is the least
certain output for either model (§5, Validated Strengths and Acknowledged Weaknesses). Avoid
presenting the retired High/Restricted figures as a current uncertainty band.

### 8.2 Investigate County-Level Divergences

The largest divergences (Grand Forks, Ward) merit further investigation:

- Compare both models against actual Census 2025 data as it becomes available
- Evaluate whether our GQ correction and college-age smoothing are over-dampening migration
  in these counties, or whether the SDC's manual adjustments are over-compensating
- Consider whether Ward County's Minot AFB population dynamics warrant a specialized
  treatment similar to our reservation county calibration (ADR-045)

### 8.3 Leverage Our Unique Capabilities

Our model offers capabilities the SDC does not:

- **Race/ethnicity projections**: Critical for workforce planning, healthcare, education
- **Place-level projections**: City and town population forecasts for local planning
- **Scenario analysis**: Quantified uncertainty bounds for infrastructure investment decisions
- **Annual resolution**: Useful for year-by-year budget planning
- **Automated validation**: Quantifiable confidence in model performance

### 8.4 Monitor Actual Data for Model Calibration

As new Census PEP vintages become available:

- Compare both projections against actuals to assess which migration assumptions are
  proving more accurate
- If net migration consistently exceeds our Baseline assumption, consider adjusting
  the convergence schedule or dampening factors
- If net migration falls below the SDC's assumption, their projection will increasingly
  overshoot

### 8.5 Document the Convergence

The dramatic convergence between the two models -- from a 170K gap in December 2025 to a
60K gap in March 2026 -- is itself a significant finding. It demonstrates that the primary
driver of the previous divergence was the migration data source (IRS flows vs. Census
residuals), not fundamental methodological disagreements. Both models, when given similar
migration inputs, produce broadly similar trajectories, with differences attributable to
specific adjustment choices rather than structural incompatibilities.

### 8.6 Consider Future Harmonization

Given the alignment on core methodology, there may be opportunities for constructive
dialogue with the SDC:

- Share our GQ correction approach, which could improve their college-county projections
- Discuss their manual adjustment rationale, which could inform our algorithmic design
- Compare detailed age-specific migration rates to understand where the remaining
  differences arise
- Explore whether a joint or reconciled projection product would serve stakeholders better
  than two competing sets

---

*Last updated: 2026-06-16 (ADR-068-corrected locked production run, config sha `a6e0bfbc2d70be85`; re-synced from the superseded 2026-06-13 pre-correction figures).*
*Supersedes: March 2026 three-scenario comparison (pre-ADR-065/066). December 2025 IRS-migration version remains obsolete.*
*Companion references: `docs/reviews/2026-06-13-locked-run-sanity-check.md`, `docs/reviews/2026-06-13-divergent-counties-methods-and-framing.md`, `docs/governance/adrs/067-ward-grand-forks-divergence-investigation.md`.*
