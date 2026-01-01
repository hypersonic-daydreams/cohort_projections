# Agent B2: Multi-State Placebo Analysis

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B2 |
| Scope | Multi-State Placebo Analysis for 50 States |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Existing Data Infrastructure

**Critical Finding**: The project already has substantial multi-state data available.

| Dataset | Location | Coverage | Content |
|---------|----------|----------|---------|
| `combined_components_of_change.csv` | `data/processed/immigration/analysis/` | 50 states + DC + PR, 2010-2024 (n=795 rows) | `intl_migration`, population, births, deaths |
| `NST-EST2009-ALLDATA.csv` | `data/raw/immigration/census_population_estimates/` | 50 states + regions, 2000-2009 | `INTERNATIONALMIG2000`-`INTERNATIONALMIG2009` |
| `NST-EST2020-ALLDATA.csv` | Same directory | 50 states, 2010-2019 | International migration components |
| `NST-EST2024-ALLDATA.csv` | Same directory | 50 states, 2020-2024 | International migration components |

**Implication**: We have data for all 50 states across all three vintage periods (2000-2009, 2010-2019, 2020-2024). No Census API calls needed.

### 1.2 Census API Infrastructure

The project has a complete `CensusDataFetcher` class at `cohort_projections/data/fetch/census_api.py`:

| Feature | Status | Notes |
|---------|--------|-------|
| PEP State-level API | Implemented | `fetch_pep_state_data()` |
| PEP County-level API | Implemented | `fetch_pep_county_data()` |
| Retry Logic | Implemented | `max_retries=3`, `retry_delay=5s` |
| Caching | Implemented | Parquet files |

**Note**: Census API is available as fallback but not needed since data is already downloaded.

### 1.3 Available Statistical Packages

| Package | Version | Purpose for B2 |
|---------|---------|----------------|
| `pandas` | >=2.0 | Data manipulation, state comparisons |
| `numpy` | >=1.24 | Numerical operations |
| `scipy` | >=1.10 | Statistical tests (t-test, Levene, etc.) |
| `statsmodels` | >=0.14 | OLS, robust SE |
| `matplotlib` | >=3.7 | Visualization |
| `seaborn` | >=0.12 | Distribution plots, heatmaps |

### 1.4 Target Data Structure

Reference from `agent2_nd_migration_data.csv`:

```csv
year,intl_migration,vintage,vintage_period
2000,258,2009,Vintage 2009 (2000-2009)
...
2010,468,2020,Vintage 2020 (2010-2019)
...
2020,30,2024,Vintage 2024 (2020-2024)
```

This is the target output format for all 50 states.

---

## 2. Data Acquisition Plan

### 2.1 Data Sources Strategy

**Primary Approach**: Use existing downloaded Census PEP files (no API needed).

### 2.2 Data Processing Pipeline

#### Step 1: Extract 2000-2009 Data (Vintage 2009)

Source: `data/raw/immigration/census_population_estimates/NST-EST2009-ALLDATA.csv`

Extract columns: `NAME`, `STATE`, `INTERNATIONALMIG2000`-`INTERNATIONALMIG2009`

#### Step 2: Extract 2010-2019 Data (Vintage 2020)

Source: `NST-EST2020-ALLDATA.csv`

#### Step 3: Extract 2020-2024 Data (Vintage 2024)

Source: `NST-EST2024-ALLDATA.csv` OR `combined_components_of_change.csv`

#### Step 4: Combine into State-Level Panel

Output format:
```csv
year,state,state_fips,intl_migration,vintage,vintage_period
2000,Alabama,1,XXX,2009,Vintage 2009 (2000-2009)
...
```

### 2.3 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year (2000-2024) |
| `state` | str | State name |
| `state_fips` | int | FIPS code |
| `intl_migration` | int | International migration count |
| `vintage` | int | PEP vintage year (2009, 2020, 2024) |
| `vintage_period` | str | Human-readable period label |
| `population` | int | Total state population (for normalization) |

---

## 3. Files Inventory

### 3.1 Files to Modify

| File | Modification | Reason |
|------|--------------|--------|
| `sdc_2024_replication/scripts/statistical_analysis/SUBAGENT_COORDINATION.md` | Add B2 documentation | Document new module |

### 3.2 New Files to Create

| File | Purpose |
|------|---------|
| **Data Processing** | |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/` | New module directory |
| `module_B2_multistate_placebo/__init__.py` | Module initialization |
| `module_B2_multistate_placebo/data_loader.py` | Load and combine PEP data across vintages |
| `module_B2_multistate_placebo/regime_shift_calculator.py` | Compute shift statistics per state |
| `module_B2_multistate_placebo/oil_state_hypothesis.py` | Test oil/energy state grouping |
| **Main Analysis Script** | |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo.py` | Primary analysis runner |
| **Output Files** | |
| `results/module_B2_multistate_placebo.json` | Primary JSON results |
| `results/module_B2_state_shift_rankings.csv` | State rankings by shift magnitude |
| `results/module_B2_oil_states_analysis.csv` | Oil state subset analysis |
| **Figures** | |
| `figures/module_B2_shift_distribution.png` | Histogram of state shifts |
| `figures/module_B2_nd_in_distribution.png` | ND highlighted in distribution |
| `figures/module_B2_oil_states_comparison.png` | Oil states vs. others |
| `figures/module_B2_choropleth_shift.png` | Geographic map of shifts |

### 3.3 Data Files to Generate

| File | Content |
|------|---------|
| `data/processed/immigration/analysis/all_states_migration_panel.csv` | 50-state panel with vintage labels (n=1,250 rows) |
| `data/processed/immigration/analysis/state_regime_shift_summary.csv` | One row per state with shift statistics |

---

## 4. Analysis Plan

### 4.1 Regime Shift Statistic Definition

**Primary Metric**: Mean Difference Between Regimes

```python
shift_magnitude = mean(intl_migration[2010-2019]) - mean(intl_migration[2000-2009])
```

**Normalized Metric**: Relative Shift (handles scale differences)

```python
relative_shift = shift_magnitude / mean(intl_migration[2000-2009])
```

**Per-capita Metric**:

```python
per_capita_shift = (mean(rate_2010s) - mean(rate_2000s))
# where rate = intl_migration / population * 1000
```

### 4.2 Statistical Tests for Each State

For each of 50 states:

1. **Two-sample t-test**: 2000s mean vs. 2010s mean
2. **Welch's t-test**: Allowing unequal variances
3. **Mann-Whitney U**: Non-parametric alternative
4. **Effect size** (Cohen's d): Standardized magnitude

### 4.3 Distribution Analysis

1. **Compute shift statistics** for all 50 states
2. **Rank states** by shift magnitude
3. **Calculate percentile** of ND's shift in national distribution
4. **Identify outliers**: States with shifts > 2 standard deviations

### 4.4 Oil/Energy State Hypothesis Test

**Oil State Definition**:
- Primary oil states: ND, TX, OK, WY, AK
- Secondary oil states: MT, NM, CO, LA, KS
- Control: Non-oil states

**Analysis**:
1. Compare mean shift between oil states and non-oil states
2. Test statistical significance of difference
3. Within-oil-state comparison (is ND unusual even among oil states?)

### 4.5 Placebo Logic

**Key Question**: "If everyone jumps similarly, that screams 'methodology.' If ND (and oil-adjacent states) are outliers, that supports a real driver story."

**Decision Framework**:

| Scenario | Interpretation |
|----------|----------------|
| ND shift is in 50th percentile of all states | Methodology effect dominates |
| ND shift is in 90th+ percentile | Real regional driver (supports oil hypothesis) |
| Oil states cluster in upper quartile | Corroborates oil-driven explanation |
| ND is outlier even among oil states | ND-specific factors (Bakken) |

### 4.6 Visualization Plan

1. **Histogram**: Distribution of shifts across 50 states, ND highlighted
2. **Box plot**: Oil states vs. non-oil states
3. **Choropleth map**: Shift magnitude by state (requires geopandas)
4. **Time series panel**: Selected comparison states (TX, OK, WY) vs. ND
5. **Scatter plot**: Pre-2010 level vs. shift magnitude

---

## 5. Code Structure

### 5.1 Module Architecture

```
sdc_2024_replication/scripts/statistical_analysis/
├── module_B2_multistate_placebo/           # New module directory
│   ├── __init__.py
│   ├── data_loader.py                      # Load/combine vintage data
│   ├── regime_shift_calculator.py          # Compute shift stats
│   └── oil_state_hypothesis.py             # Oil state analysis
├── module_B2_multistate_placebo.py         # Main runner
├── results/
│   ├── module_B2_multistate_placebo.json
│   ├── module_B2_state_shift_rankings.csv
│   └── module_B2_oil_states_analysis.csv
└── figures/
    └── module_B2_*.png
```

### 5.2 Key Functions (Pseudocode)

```python
# data_loader.py
def load_vintage_2009_data() -> pd.DataFrame:
    """Load 2000-2009 international migration for all states."""
    df = pd.read_csv(NST_EST2009_PATH)
    # Reshape wide to long
    # Add vintage labels
    return df_long

def combine_all_vintages() -> pd.DataFrame:
    """Create unified 50-state panel (2000-2024)."""
    v2009 = load_vintage_2009_data()
    v2020 = load_vintage_2020_data()
    v2024 = load_vintage_2024_data()
    return pd.concat([v2009, v2020, v2024])

# regime_shift_calculator.py
def calculate_state_shift(df: pd.DataFrame, state: str) -> dict:
    """Calculate regime shift statistics for one state."""
    pre = df[(df['state'] == state) & (df['year'] < 2010)]['intl_migration']
    post = df[(df['state'] == state) & (df['year'] >= 2010) & (df['year'] < 2020)]['intl_migration']
    return {
        'state': state,
        'mean_2000s': pre.mean(),
        'mean_2010s': post.mean(),
        'shift_magnitude': post.mean() - pre.mean(),
        'relative_shift': (post.mean() - pre.mean()) / pre.mean(),
        't_statistic': ttest_ind(post, pre).statistic,
        'p_value': ttest_ind(post, pre).pvalue,
    }

# oil_state_hypothesis.py
OIL_STATES = ['North Dakota', 'Texas', 'Oklahoma', 'Wyoming', 'Alaska',
              'Montana', 'New Mexico', 'Colorado', 'Louisiana', 'Kansas']

def test_oil_state_hypothesis(shift_df: pd.DataFrame) -> dict:
    """Test whether oil states have systematically different shifts."""
    oil = shift_df[shift_df['state'].isin(OIL_STATES)]['relative_shift']
    non_oil = shift_df[~shift_df['state'].isin(OIL_STATES)]['relative_shift']

    return {
        'oil_mean': oil.mean(),
        'non_oil_mean': non_oil.mean(),
        'difference': oil.mean() - non_oil.mean(),
        't_test': ttest_ind(oil, non_oil),
    }
```

---

## 6. Dependencies

### 6.1 Dependencies on Other Agents

| Agent | What B2 Needs | Type |
|-------|---------------|------|
| **B1 (Statistical Modeling)** | Regime definition (2000-2009, 2010-2019, 2020-2024 boundaries) | Specification |
| **Phase A (Complete)** | Understanding of vintage methodology differences | Context |

### 6.2 What Other Agents Need from B2

| Agent | What They Need | Deliverable |
|-------|----------------|-------------|
| **B3 (Journal Article)** | ND percentile in national distribution | Text for methodology section |
| **B3 (Journal Article)** | Oil state comparison results | Support for "real driver" claim |
| **B4 (Panel/Bayesian)** | 50-state panel dataset | Input data for panel models |
| **B6 (Testing)** | Test specifications for B2 functions | Unit test targets |

---

## 7. Risks and Blockers

### 7.1 Data Availability Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing states in 2009 vintage file | LOW | HIGH | Verify all 50 states before analysis |
| Inconsistent column names across vintages | MEDIUM | MEDIUM | Manual inspection + mapping dictionary |
| 2020 COVID year anomaly | HIGH | MEDIUM | Exclude 2020 from regime mean calculations |

### 7.2 Statistical Challenges

| Challenge | Risk Level | Mitigation |
|-----------|------------|------------|
| Scale differences across states (CA vs. WY) | HIGH | Use relative/per-capita metrics |
| Non-normal shift distributions | MEDIUM | Use non-parametric tests (Mann-Whitney) |
| Multiple comparison problem (50 states) | MEDIUM | Apply Bonferroni or FDR correction |
| Small sample per state (n=10 per regime) | HIGH | Report confidence intervals |

### 7.3 Technical Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| geopandas installation for choropleth | MEDIUM | Make map optional; provide fallback table |
| Memory usage for full panel | LOW | 1,250 rows is trivial |

---

## 8. Estimated Complexity

| Component | Complexity | Justification |
|-----------|------------|---------------|
| Data Loading | **LOW** | Files already downloaded; standard pandas |
| Data Transformation | **MEDIUM** | Wide-to-long reshaping; vintage merging |
| Shift Calculations | **LOW** | Simple means, t-tests |
| Oil State Analysis | **LOW** | Subsetting and comparison |
| Distribution Visualizations | **MEDIUM** | Multiple plots; highlighting ND |
| Choropleth Map | **MEDIUM** | Requires geopandas setup |
| **Overall** | **MEDIUM** | Straightforward statistics; complexity is in data wrangling |

---

## 9. Key Decisions

1. **2020 excluded from regime means**: Yes, treat as COVID outlier per B1

2. **Primary shift metric**: Relative shift (handles scale), with absolute as secondary

3. **Oil state definition**: Use petroleum production ranking from EIA as criterion

---

## Summary

This plan provides:

1. **Discovery that existing data is sufficient** - No Census API calls needed
2. **Clear data processing pipeline** from three PEP vintage files to unified panel
3. **Complete analysis methodology** for placebo test and oil state hypothesis
4. **File inventory** of all new files to create
5. **Dependency mapping** with B1, B3, B4

**Key insight**: The external reviewer's question "How unusual is ND's shift in the national distribution?" can be answered definitively with data already in the project.

**Decision Required**: Approve this plan to proceed with B2 implementation.
