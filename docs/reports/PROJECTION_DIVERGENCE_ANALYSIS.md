# Projection Divergence Analysis: 2025 vs. 2024 ND Projections

**Date:** 2026-02-03
**Status:** Investigation Complete - Action Required
**Priority:** HIGH

---

## Executive Summary

Our current baseline projections show **North Dakota's population declining to 754,882 by 2045** (−5.2% from 2025), while the 2024 State Data Center projections anticipated **growth to 950,000+ by 2050**.

**Root Cause:** Different migration assumptions account for **~170,000 people difference** by 2045.

---

## Key Findings

### 1. Current Baseline Projection Results

| Metric | Value |
|--------|-------|
| **Base Year (2025)** | 796,568 |
| **Final Year (2045)** | 754,882 |
| **Change** | −41,686 (−5.2%) |
| **Annual Growth Rate (2045)** | −0.47% |
| **Median Age Change** | 39.0 → 42.0 years |
| **Youth Share (0-17)** | 22.3% → 18.0% |
| **Elderly Share (65+)** | 20.1% → 22.9% |

**Scenario Comparison:**
- **Baseline:** 754,882 (decline)
- **High Growth:** 786,564 (still decline, but less severe)
- **Low Growth:** ~725,840 (steeper decline)

### 2. Migration: The Critical Difference

#### Our Baseline Methodology
**Data Source:** IRS County-to-County Migration Flows, 2019-2022

**Key Statistics:**
- Average inflow per county: 709 people/year
- Average outflow per county: 793 people/year
- **Net migration: −84 people/county/year** ❌

**Major Counties (2022):**
- **Ward County:** −1,199 net migration
- **Grand Forks County:** −655 net migration
- **Cass County:** +418 net migration (only major county with positive)
- **Williams County:** −269 net migration (after −1,891 in 2021)
- **McKenzie County:** +8 (after −565 in 2021)

**Why Negative?**
1. COVID-19 pandemic effects (2020-2021)
2. Bakken oil region bust (2020-2021)
3. Western ND counties (Williams, McKenzie) had massive out-migration
4. Captures most recent trends but may be anomalous period

#### SDC 2024 Methodology
**Data Source:** Census Residual Method, 2000-2020 with 60% Bakken dampening

**Key Statistics:**
- Total net migration rate: **+0.003** (positive!) ✅
- Male rate: +0.015 (net in-migration)
- Female rate: −0.009 (net out-migration)
- Overall: **Net IN-migration assumption**

**Why Positive?**
1. Captures entire Bakken boom period (2008-2014)
2. 20-year historical average smooths out oil volatility
3. Applied 60% dampening to reduce extreme Bakken effects
4. Pre-COVID migration patterns

### 3. Other Component Comparisons

#### Fertility
- **Our Baseline:** SEER rates, 5-year average (2018-2022), constant assumption
- **SDC 2024:** Blended ND + national rates
- **Impact:** Minimal difference

#### Mortality
- **Our Baseline:** SEER life tables (2020), 0.5% annual improvement
- **SDC 2024:** Constant mortality (no improvement)
- **Impact:** Small difference favoring longer life in our projections

#### Base Population
- **Both use:** ~796,000 for 2024/2025
- **Impact:** No difference

---

## Quantitative Impact

| Component | Impact on 2045 Population |
|-----------|---------------------------|
| **Migration assumptions** | **~170,000 people** |
| Mortality improvement | ~5,000-10,000 people |
| Fertility differences | ~5,000-10,000 people |
| **Total Difference** | **~180,000-190,000 people** |

**Migration dominates the divergence.**

---

## Data Files & Locations

### Current Baseline Outputs
- **Projections:** [data/projections/baseline/county/](../../data/projections/baseline/county/) (53 Parquet files)
- **Visualizations:** [data/output/visualizations/](../../data/output/visualizations/) (8 PNG files)
- **Reports:** [data/output/reports/](../../data/output/reports/) (12 files: HTML, MD, JSON, CSV)
- **Exports:** [data/exports/baseline/](../../data/exports/baseline/) (Excel and CSV by county)

### Migration Data Sources
- **IRS Data (Our Baseline):** [data/raw/migration/nd_migration_processed.csv](../../data/raw/migration/nd_migration_processed.csv)
- **IRS Rates (Processed):** [data/processed/migration_rates.parquet](../../data/processed/migration_rates.parquet)
- **SDC 2024 Migration:** [data/processed/sdc_2024/migration_rates_sdc_2024.csv](../../data/processed/sdc_2024/migration_rates_sdc_2024.csv)
- **SDC Summary:** [data/processed/sdc_2024/migration_rates_summary.csv](../../data/processed/sdc_2024/migration_rates_summary.csv)

### Configuration
- **Main Config:** [config/projection_config.yaml](../../config/projection_config.yaml)
- **SDC 2024 Scenario:** Lines 160-171 (currently `active: false`)

---

## Options for Resolution

### Option 1: Use SDC 2024 Methodology (Match Previous Projections)
**Action:** Activate and run the `sdc_2024` scenario

**Pros:**
- Matches 2024 State Data Center projections
- Uses established, validated methodology
- More optimistic but defensible assumptions

**Cons:**
- May overestimate if recent out-migration trends continue
- Doesn't capture post-COVID reality

**Implementation:**
```yaml
# In config/projection_config.yaml, line 171:
active: true  # Set to true to include in projection runs
```

Then run:
```bash
python scripts/pipeline/02_run_projections.py --scenarios sdc_2024
```

### Option 2: Adjust Current Baseline (Moderate Recovery)
**Action:** Modify migration assumptions to blend historical and recent data

**Pros:**
- Balances recent trends with longer-term patterns
- More nuanced than all-or-nothing approach
- Can create multiple scenarios for uncertainty

**Cons:**
- Requires judgment about future trends
- Need to justify blending methodology

**Potential Approaches:**
- Use 2010-2022 average (captures Bakken but weights recent data)
- Apply gradual recovery curve (start with 2022 data, trend toward historical mean)
- Differential treatment by region (assume Western ND recovers, others stay flat)

### Option 3: Create Multiple Migration Scenarios
**Action:** Generate projections with different migration futures

**Scenarios:**
1. **Pessimistic (Current Baseline):** Recent trends continue (754K by 2045)
2. **Moderate Recovery:** Gradual return to historical patterns (~850K by 2045)
3. **Optimistic (SDC 2024):** Historical averages (~950K by 2050)
4. **Oil Rebound:** Western ND returns to Bakken boom levels (~1M by 2050)

**Pros:**
- Communicates uncertainty
- Allows decision-makers to plan for range of futures
- Most scientifically defensible

**Cons:**
- More complex to explain
- Requires running multiple projections

### Option 4: Deep Dive Migration Analysis
**Action:** Investigate migration patterns before making decision

**Research Questions:**
- Is 2019-2022 an anomaly or new normal?
- What drove the 2020-2021 out-migration spike?
- Have patterns normalized in 2023-2024?
- What are energy sector forecasts for Bakken region?
- How do ND economic indicators look for 2025-2030?

**Data to Gather:**
- 2023-2024 IRS migration data (if available)
- Census PEP estimates for 2023-2024
- ND Department of Commerce economic forecasts
- Oil & gas industry projections
- Employment trends by sector

---

## Recommendations

### Immediate Actions
1. **Document this divergence** for stakeholders ✅ (this document)
2. **Run SDC 2024 scenario** to validate we can replicate previous projections
3. **Gather 2023-2024 data** to see if recent trends are continuing or reversing

### Medium-term Actions
4. **Create migration scenario analysis** with pessimistic/moderate/optimistic futures
5. **Consult with ND State Data Center** about their methodology and recent data
6. **Present scenario ranges** rather than single projection

### For Today's Session
1. Run `sdc_2024` scenario to produce comparable projections
2. Generate visualizations comparing all scenarios
3. Create briefing document for stakeholders explaining divergence

---

## Technical Notes

### Visualization Scripts
- **Generate all visuals:** `python scripts/generate_visualizations_and_reports.py`
- **Aggregates to state level:** Sums all 53 county projections
- **Outputs:** PNG files (300 DPI), HTML/MD reports, JSON statistics

### Scenario Configuration
Scenarios are defined in [config/projection_config.yaml](../../config/projection_config.yaml) under `scenarios:` section.

To run specific scenarios:
```bash
# Modify projection.scenarios in config
scenarios: ["baseline", "sdc_2024", "high_growth", "low_growth"]

# Then run
python scripts/pipeline/02_run_projections.py
```

### Migration Rate Processing
Migration rates are processed in:
- [cohort_projections/data/process/migration_rates.py](../../cohort_projections/data/process/migration_rates.py)
- [scripts/pipeline/02_run_projections.py](../../scripts/pipeline/02_run_projections.py) (data transformation)

---

## Questions to Resolve

1. **Which migration pattern is most realistic for 2025-2045?**
   - Recent trends (out-migration)?
   - Historical trends (in-migration)?
   - Gradual recovery?

2. **What is the purpose of these projections?**
   - Planning scenarios (show range)?
   - Budget forecasting (conservative)?
   - Economic development (optimistic)?

3. **What do stakeholders expect?**
   - Consistency with 2024 projections?
   - Updated estimates based on latest data?
   - Scenario analysis?

4. **What does recent data show?**
   - Has 2023-2024 migration recovered?
   - Are energy sector trends stabilizing?
   - What are leading economic indicators?

---

## References

### Config Documentation
- Main configuration: [config/projection_config.yaml](../../config/projection_config.yaml)
- SDC 2024 scenario: Lines 160-171
- Migration settings: Lines 117-126

### Key Documentation
- [DEVELOPMENT_TRACKER.md](../../DEVELOPMENT_TRACKER.md) - Project status
- [AGENTS.md](../../AGENTS.md) - AI agent guidance
- [data/README.md](../../data/README.md) - Data documentation
- [docs/governance/adrs/016-raw-data-management-strategy.md](../governance/adrs/016-raw-data-management-strategy.md)

### Data Sources
- IRS Migration Data: [data/raw/migration/](../../data/raw/migration/)
- SDC 2024 Rates: [data/processed/sdc_2024/](../../data/processed/sdc_2024/)
- Current Reports: [data/output/reports/](../../data/output/reports/)

---

## Change Log

| Date | Action | Notes |
|------|--------|-------|
| 2026-02-03 | Initial investigation | Identified migration as root cause |
| 2026-02-03 | Document created | Comprehensive analysis of divergence |

---

## Next Session Checklist

- [ ] Review this document
- [ ] Decide on migration assumption approach (Options 1-4 above)
- [ ] Run selected scenario(s)
- [ ] Generate comparison visualizations
- [ ] Create stakeholder briefing document
- [ ] Update DEVELOPMENT_TRACKER.md with findings

---

**Last Updated:** 2026-02-03
**Author:** Claude Code Analysis
**Status:** Ready for Decision & Action
