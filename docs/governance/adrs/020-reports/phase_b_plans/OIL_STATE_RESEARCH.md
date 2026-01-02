# Oil State Definition Research Report

## Document Information

| Field | Value |
|-------|-------|
| Document ID | OIL_STATE_RESEARCH |
| Purpose | Evidence-based definition of "oil states" for B2 multi-state placebo analysis |
| Created | 2026-01-01 |
| Status | Research Complete |

---

## Executive Summary

This report examines how to define "oil states" for the Module B2 multi-state placebo analysis in the North Dakota international migration study. The current ad-hoc definition (TX, ND, NM, OK, CO, AK as primary; WY, LA, KS, CA as secondary) lacks empirical justification. After reviewing academic literature, EIA production data, and BLS employment data, we recommend a **hybrid approach** that considers both production levels and boom timing, with oil price controls as an optional enhancement.

**Key Findings:**
1. The current B2 analysis shows ND ranks 43rd out of 51 states in migration shift magnitude, with no significant difference between oil and non-oil states (p=0.71)
2. Academic literature emphasizes boom timing and production growth over static production levels
3. The Bakken boom (2008-2015) had different timing than the Permian Basin boom (accelerating 2015-present)
4. Oil prices are a critical confounding factor that should be incorporated

---

## 1. Literature Summary

### 1.1 How Researchers Define "Oil States" or "Resource Boom States"

Academic research on resource booms rarely uses static "oil state" classifications. Instead, researchers typically define exposure based on:

#### 1.1.1 Production-Based Definitions

- **EIA Rankings**: The U.S. Energy Information Administration provides annual crude oil production by state, which forms the basis for most official rankings
- **Top Producer Thresholds**: Some studies use "top 5" or "top 10" producers as a cutoff
- **Production Share**: States accounting for significant percentages of national production

**Key Source**: [EIA State Rankings](https://www.eia.gov/state/rankings/) provides comprehensive production data

#### 1.1.2 Growth-Based (Boom) Definitions

Riley Wilson's influential paper "Moving to Economic Opportunity: The Migration Response to the Fracking Boom" (Journal of Human Resources, 2022) uses:
- **County-level shale intensity** rather than state-level classification
- **Timing of production increases** specific to each geographic area
- Focus on areas where fracking technology enabled new production

**Key Finding**: Wilson found that North Dakota fracking counties saw migration increases nearly **twice as large** as other fracking areas, suggesting ND-specific factors beyond just "oil state" status.

**Source**: [Wilson (2022) SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2814147)

#### 1.1.3 Employment-Based Definitions

Research by the IMF and Baker Institute uses:
- **Oil and gas employment share of total employment**
- **Changes in oil and gas employment** during boom periods
- NAICS 211 (Oil and Gas Extraction) employment data

**Key Finding**: Studies find that "a large fraction of new jobs are filled by workers who reside outside of the county," supporting the migration-boom connection.

**Source**: [Employment Impacts of Upstream Oil and Gas Investment](https://www.imf.org/external/pubs/ft/wp/2015/wp1528.pdf)

### 1.2 Critical Insight from Literature

The literature suggests that **boom timing** is more important than static production levels for understanding migration effects. The question "Is this an oil state?" matters less than:
- "When did this state experience its boom?"
- "How rapidly did production grow?"
- "What was the labor market response?"

---

## 2. Data Source Recommendations

### 2.1 EIA Crude Oil Production Data

**Primary Source**: [EIA Crude Oil Production](https://www.eia.gov/dnav/pet/pet_crd_crpdn_adc_mbbl_m.htm)

| Feature | Details |
|---------|---------|
| Coverage | All 50 states |
| Time series | 1981-present (monthly/annual) |
| Metric | Thousand barrels per day |
| Format | Downloadable CSV/Excel |

**Recommended Use**:
- Calculate average production by state for relevant periods (2000-2009, 2010-2019, 2020-2024)
- Calculate production growth rates to identify boom states
- Match production timing to vintage methodology periods

### 2.2 BLS Oil and Gas Employment Data

**Primary Source**: [BLS NAICS 211](https://www.bls.gov/iag/tgs/iag211.htm) and [FRED State-Level Data](https://fred.stlouisfed.org/series/IPUBN211W201000000)

| Feature | Details |
|---------|---------|
| Coverage | States with significant oil/gas employment |
| Time series | 1988-present |
| Metric | Employment count |
| NAICS Code | 211 (Oil and Gas Extraction) |

**Recommended Use**:
- Calculate oil/gas employment as share of total state employment
- Identify states where oil/gas is economically significant
- Track employment changes during boom periods

### 2.3 Oil Price Data

**Primary Source**: [FRED WTI Crude Prices](https://fred.stlouisfed.org/series/DCOILWTICO)

| Feature | Details |
|---------|---------|
| Coverage | Daily, monthly, annual averages |
| Time series | 1986-present |
| Metric | Dollars per barrel (WTI Cushing, OK) |

**Key Historical Prices**:
| Period | Average WTI | Context |
|--------|-------------|---------|
| 2005-2007 | $56-72/bbl | Pre-boom |
| 2008 (peak) | $147/bbl (July) | Boom driver |
| 2009 | $62/bbl | Post-crisis |
| 2011-2014 | $85-105/bbl | Sustained boom |
| 2015-2016 | $30-50/bbl | Price crash |
| 2017-2019 | $50-65/bbl | Recovery |

---

## 3. Proposed Classification Schemes

### 3.1 Option A: Production-Based (Top N Producers)

**Definition**: States ranked in the top N for crude oil production during the study period.

#### Current EIA Rankings (2024 data):
| Rank | State | Production (million bbl/year) |
|------|-------|------------------------------|
| 1 | Texas | 2,070+ |
| 2 | New Mexico | 745 |
| 3 | North Dakota | ~400 |
| 4 | Colorado | 172 |
| 5 | Alaska | 154 |
| 6 | Oklahoma | 145 |
| 7 | Wyoming | 107 |
| 8 | California | 104 |
| 9 | Louisiana | ~100 |
| 10 | Kansas | ~30 |

**Recommended Cutoff**: Top 6 states (TX, NM, ND, CO, AK, OK) as "primary oil states"

**Pros**:
- Simple, transparent, defensible
- Based on authoritative EIA data
- Current B2 implementation largely follows this

**Cons**:
- Static classification doesn't capture boom dynamics
- Doesn't distinguish mature producers (CA, AK) from boom states (ND, NM)
- Texas dominates, potentially driving results

### 3.2 Option B: Employment-Based (Oil/Gas Share of Employment)

**Definition**: States where oil/gas extraction (NAICS 211) exceeds a threshold percentage of total employment.

**Potential Thresholds**:
| Threshold | Rationale |
|-----------|-----------|
| >1% of employment | Economically significant |
| >2% of employment | Major economic driver |
| >5% of employment | Dominant industry |

**States likely to qualify at >1%**:
Wyoming, North Dakota, Oklahoma, Texas, Alaska, New Mexico, Louisiana

**Pros**:
- Captures economic significance, not just production
- Better reflects labor market conditions relevant to migration
- Aligns with literature on employment-driven migration

**Cons**:
- Requires additional data acquisition
- May exclude states with high production but diversified economies (CA, CO)
- Employment data may lag production changes

### 3.3 Option C: Boom-Timing-Based (Production Growth 2008-2015)

**Definition**: States with significant production growth during the Bakken boom period.

**Key Timing**:
| Boom | Geography | Primary Period | Secondary Period |
|------|-----------|----------------|------------------|
| Bakken | ND, MT | 2008-2015 | Continued 2016+ |
| Eagle Ford | TX (South) | 2010-2015 | Decline 2016+ |
| Permian | TX (West), NM | 2010-2014 (Phase 1) | 2015-present (Phase 2) |
| Niobrara | CO | 2010-2015 | Moderate growth |

**Recommended Metric**: Compound Annual Growth Rate (CAGR) of oil production 2008-2015

**States with highest production growth 2008-2015**:
1. North Dakota (~40% CAGR)
2. Texas (~15% CAGR in shale regions)
3. Colorado (~10% CAGR)
4. New Mexico (~8% CAGR, accelerating after 2015)
5. Oklahoma (~5% CAGR)

**"Boom States" Definition**: CAGR > 10% during 2008-2015

**Pros**:
- Directly captures the phenomenon of interest (boom dynamics)
- Distinguishes ND's exceptional growth from mature producers
- Aligns with migration literature (Wilson 2022)

**Cons**:
- Requires historical production data processing
- Different boom timings complicate analysis (Permian accelerated later)
- May have small sample size if threshold too high

### 3.4 Option D: Hybrid Approach (Recommended)

**Definition**: Combine production rank with boom timing.

**Classification Matrix**:
| Category | Criteria | States |
|----------|----------|--------|
| **Boom Oil States** | Top 10 producer AND CAGR >5% (2008-2015) | ND, TX, NM, CO, OK |
| **Mature Oil States** | Top 10 producer AND CAGR <5% | CA, AK, LA, WY, KS |
| **Non-Oil States** | Not in top 15 producers | All others |

**Recommended Primary Analysis Groups**:
1. **Bakken Boom**: ND, MT (specific shale formation)
2. **Permian Boom**: TX, NM (Western TX, SE NM specifically)
3. **Other Shale**: CO, OK, LA (various formations)
4. **Mature Oil**: CA, AK, WY, KS (declining or stable production)
5. **Non-Oil**: All remaining states

**Pros**:
- Captures both production level and boom dynamics
- Creates meaningful comparison groups
- Allows testing of specific hypotheses (Bakken vs. Permian timing)

**Cons**:
- More complex to implement
- Smaller group sizes reduce statistical power
- Requires production growth calculations

---

## 4. Oil Price Consideration

### 4.1 Why Oil Prices Matter

The migration-oil relationship is mediated by economic conditions, which are strongly affected by oil prices:

| Price Environment | Expected Migration Effect |
|-------------------|---------------------------|
| High prices (>$80/bbl) | Strong incentive for oil production; high wages attract migrants |
| Medium prices ($50-80/bbl) | Moderate production; normal labor market conditions |
| Low prices (<$50/bbl) | Reduced drilling; layoffs; out-migration |

### 4.2 Key Price Periods (2008-2019)

| Period | WTI Price Range | Bakken Context |
|--------|-----------------|----------------|
| 2008-2009 | Peak $147, crash to $32 | Boom beginning, then pause |
| 2010-2014 | $80-110 sustained | Peak drilling activity |
| 2014-2016 | Crash from $100 to $30 | Drilling decline, layoffs |
| 2017-2019 | $45-65 recovery | Moderate activity |

### 4.3 Should We Control for Oil Prices?

**Arguments For**:
- Oil prices are a key driver of migration to oil regions (Wilson 2022)
- The 2014-2016 crash significantly affected oil state economies
- Controlling for prices isolates the "oil state" effect from price effects

**Arguments Against**:
- The analysis already compares vintages, which reflect different price regimes
- Adding price controls increases complexity
- Oil prices affect all oil states similarly (common shock)

**Recommendation**: **Include oil prices as a secondary analysis**

**Implementation Options**:
1. **Stratified Analysis**: Compare shifts during high-price (2010-2014) vs. low-price (2015-2016) periods
2. **Price-Weighted Metric**: Weight production data by contemporaneous oil prices
3. **Panel Control Variable**: Include average annual WTI price in panel models (Module B4)

---

## 5. Implications of Current B2 Results

### 5.1 Summary of Current Findings

The current B2 module uses the ad-hoc oil state definition and found:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ND national rank | 43 of 51 | Bottom quartile |
| ND percentile | 15.7th from top | Not exceptional |
| Oil state mean shift | 29,873 | Below average |
| Non-oil state mean shift | 40,804 | Above average |
| t-test p-value | 0.71 | No significant difference |

**Current Conclusion**: "WEAK SUPPORT - methodology artifact cannot be ruled out"

### 5.2 Why Current Results May Be Misleading

1. **Scale dominance**: Texas and California dominate the "oil state" mean due to huge populations
2. **Wrong comparison period**: Analysis compares 2010-2019 to 2020-2024, but the Bakken boom was 2008-2015
3. **Heterogeneous boom timing**: Lumping Bakken states (2008-2015 boom) with Permian states (2015+ boom) may wash out effects
4. **Shift metric vs. level metric**: Using absolute shift magnitude disadvantages small states like ND

### 5.3 What Better Classification Might Reveal

With a boom-timing-based classification:
- Compare ND to other **2008-2015 boom states** (not all oil states)
- Use **relative shift** (% change) instead of absolute magnitude
- Focus on the **2009-to-2020 vintage transition** (captures Bakken boom)

---

## 6. Implementation Recommendations

### 6.1 Recommended Changes to B2 Module

#### 6.1.1 Update Oil State Definition

Replace current hard-coded lists with empirically-derived classification:

```python
# Current (ad-hoc)
OIL_STATES = ["Texas", "North Dakota", "New Mexico", "Oklahoma", "Colorado", "Alaska"]
SECONDARY_OIL_STATES = ["Wyoming", "Louisiana", "Kansas", "California"]

# Recommended (hybrid)
BAKKEN_BOOM_STATES = ["North Dakota", "Montana"]
PERMIAN_BOOM_STATES = ["Texas", "New Mexico"]
OTHER_SHALE_STATES = ["Colorado", "Oklahoma", "Louisiana"]
MATURE_OIL_STATES = ["California", "Alaska", "Wyoming", "Kansas"]
ALL_BOOM_STATES = BAKKEN_BOOM_STATES + PERMIAN_BOOM_STATES + OTHER_SHALE_STATES
```

#### 6.1.2 Add Production Growth Data

Create new data file with state-level production growth rates:

| File | Purpose |
|------|---------|
| `data/external/eia_state_oil_production.csv` | EIA production by state, 2000-2024 |
| `data/external/state_production_growth.csv` | Calculated CAGR by state and period |

#### 6.1.3 New Analysis Functions

```python
def classify_states_by_boom(production_df: pd.DataFrame, boom_period: tuple) -> dict:
    """Classify states by production growth during boom period."""
    pass

def test_bakken_specific_hypothesis(shift_df: pd.DataFrame) -> dict:
    """Test whether Bakken states have different shifts than Permian states."""
    pass

def get_nd_rank_among_boom_states(shift_df: pd.DataFrame) -> dict:
    """Rank ND specifically among 2008-2015 boom states."""
    pass
```

#### 6.1.4 Add Oil Price Analysis (Optional)

```python
def load_wti_prices() -> pd.DataFrame:
    """Load WTI price data from FRED."""
    pass

def stratify_by_price_period(shift_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
    """Compare shifts during high-price vs low-price periods."""
    pass
```

### 6.2 Data Acquisition Steps

1. **Download EIA production data**:
   - URL: https://www.eia.gov/dnav/pet/pet_crd_crpdn_adc_mbbl_m.htm
   - Format: CSV, annual production by state
   - Years: 2000-2024

2. **Download WTI price data**:
   - URL: https://fred.stlouisfed.org/series/DCOILWTICO
   - Format: CSV, daily/monthly
   - Years: 2000-2024

3. **Calculate growth metrics**:
   - CAGR by state for 2008-2015 (Bakken boom)
   - CAGR by state for 2015-2019 (Permian acceleration)

### 6.3 Recommended Analysis Plan

| Analysis | Purpose | Priority |
|----------|---------|----------|
| **A1**: Re-run B2 with boom-state classification | Test boom-timing hypothesis | HIGH |
| **A2**: Compare ND to other Bakken states only | ND-specific test | HIGH |
| **A3**: Use relative shift instead of absolute | Normalize for state size | HIGH |
| **A4**: Focus on 2009-to-2020 vintage transition | Capture Bakken boom period | MEDIUM |
| **A5**: Stratify by oil price periods | Control for price effects | LOW |
| **A6**: Add employment-based classification | Alternative definition | LOW |

---

## 7. Summary Recommendations

### 7.1 Classification Recommendation

**Adopt Option D (Hybrid Approach)** with the following groups:
1. Bakken Boom States (ND, MT)
2. Permian Boom States (TX, NM)
3. Other Shale States (CO, OK, LA)
4. Mature Oil States (CA, AK, WY, KS)
5. Non-Oil States (all others)

### 7.2 Oil Price Recommendation

**Include oil prices as a secondary analysis**, stratifying by price period rather than adding as a continuous control.

### 7.3 Next Steps

1. **Download EIA production data** and calculate growth rates
2. **Update oil_state_hypothesis.py** with new classification
3. **Re-run B2 analysis** with boom-state groupings
4. **Focus on relative shift metric** for cross-state comparisons
5. **Document methodology changes** in module docstrings

### 7.4 Expected Impact on Findings

With improved classification:
- ND may rank higher among **boom states** specifically
- Bakken vs. Permian timing differences may be revealed
- The "methodology artifact" conclusion may need revision

---

## References

### Academic Literature
- Wilson, Riley (2022). "Moving to Economic Opportunity: The Migration Response to the Fracking Boom." Journal of Human Resources 57(3): 918-955. [SSRN Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2814147)
- Feyrer, James, et al. (2017). "The Local Economic and Welfare Consequences of Hydraulic Fracturing." American Economic Journal: Applied Economics.

### EIA Data Sources
- [Crude Oil Production by State](https://www.eia.gov/dnav/pet/pet_crd_crpdn_adc_mbbl_m.htm)
- [State Energy Rankings](https://www.eia.gov/state/rankings/)
- [Drilling Productivity Report](https://www.eia.gov/petroleum/drilling/)

### BLS Data Sources
- [Oil and Gas Extraction (NAICS 211)](https://www.bls.gov/iag/tgs/iag211.htm)
- [FRED State Employment Data](https://fred.stlouisfed.org/series/TX10211000M175FRBDAL)

### Oil Price Data
- [FRED WTI Crude Prices](https://fred.stlouisfed.org/series/DCOILWTICO)
- [Macrotrends Oil Price History](https://www.macrotrends.net/1369/crude-oil-price-history-chart)

---

## 8. Implementation Results (2026-01-01)

The recommendations from this research were implemented in Module B2. Here are the key findings:

### 8.1 Changes Implemented

1. **Updated oil_state_hypothesis.py** with boom-timing classification:
   - Added `BAKKEN_BOOM_STATES`, `PERMIAN_BOOM_STATES`, `OTHER_SHALE_STATES`, `MATURE_OIL_STATES`
   - Created `get_boom_category()` function
   - Added `test_boom_state_hypothesis()`, `test_bakken_specific_hypothesis()`, `get_nd_rank_among_boom_states()`

2. **Updated module_B2_multistate_placebo.py** with new analysis section:
   - Section 6: Boom-Timing Hypothesis Test
   - New visualization: `module_B2_boom_category_comparison.png`
   - New output: `module_B2_boom_category_analysis.csv`

### 8.2 Key Findings from Updated Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ND rank among boom states | 4 of 7 | Lower half |
| ND relative shift | 138.3% | Above average |
| Boom vs Non-Oil t-test p | 0.65 | Not significant |
| Bakken vs Permian difference | -96% pts | Permian higher |

### 8.3 Unexpected Finding: Montana

Montana (the other Bakken boom state) has a **negative** relative shift (-27.2%), which is unexpected. This weakens the "Bakken boom timing" hypothesis because:

1. If Bakken boom timing explained ND's patterns, MT should show similar patterns
2. MT's negative shift suggests state-specific factors beyond oil production
3. The Bakken formation spans both states, so geology alone doesn't explain the difference

### 8.4 Revised Interpretation

The boom-timing classification did NOT strengthen the case for the real driver hypothesis:

1. **Boom states not significantly different** from non-oil states (p=0.65)
2. **Permian states have higher shifts** than Bakken states (151.5% vs 55.6%)
3. **ND is not exceptional** even among boom states (rank 4 of 7)
4. **Montana contradicts** the Bakken boom hypothesis

### 8.5 Implication for ADR-020

The multi-state placebo analysis provides **WEAK SUPPORT** for the real driver hypothesis. The methodology artifact explanation cannot be ruled out. This suggests:

1. The vintage transition patterns may be driven by Census methodology changes
2. ND-specific factors (if any) are not captured by oil state classifications
3. Future analysis should focus on county-level data to isolate Bakken-specific effects

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-01-01 |
| **Author** | Claude Code (research compilation) |
| **Status** | Complete - Implementation Verified |
