# Design: Migration Rate Cap for Convergence Rates

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Related** | [Finding 2: Oil County Growth](finding-2-oil-county-growth.md), [Finding 3: Reservation Declines](finding-3-reservation-county-declines.md) |
| **Status** | Analysis complete -- ready for implementation decision |

---

## 1. Executive Summary

Individual age-sex cells in the convergence rate data carry extreme migration rates driven by small-population statistical noise and oil-boom residuals. A rate cap applied during convergence interpolation can clip the most implausible outliers without distorting legitimate growth patterns.

**Recommended cap:** Age-aware asymmetric at **[-15%, +15%] for ages 15-24** and **[-8%, +8%] for all other ages**. This clips 2.4% of all cells (1,372 of 57,240) across all year offsets, reduces Billings County's 30-year growth from +55% to +38%, preserves Cass/Fargo at +20.5% unchanged, and only marginally affects McKenzie (+45.4% to +44.9%).

**Key finding:** A rate cap alone is insufficient to address oil-county growth implausibility (Finding 2-C). McKenzie's +45% 30-year growth is driven by many moderately positive cells (25-29 at 8.5%, 5-9 at 3.9%, 30-34 at 3.7%), not a single extreme outlier. The rate cap is a necessary guard-rail against statistical noise, but boom dampening (ADR-040) remains the primary tool for oil-county growth control.

---

## 2. Data Sources

| Source | Path | Description |
|--------|------|-------------|
| Convergence rates | `data/processed/migration/convergence_rates_by_year.parquet` | 57,240 rows: 30 year_offsets x 53 counties x 18 age-sex cells |
| Residual rates | `data/processed/migration/residual_migration_rates.parquet` | 9,540 rows: 5 periods x 53 counties x 18 age-sex cells |
| County population | `data/raw/population/nd_county_population.csv` | 2025 population estimates |
| Convergence code | `cohort_projections/data/process/convergence_interpolation.py` | Pipeline implementation |

---

## 3. Distribution Analysis

### 3.1 Overall Distribution by Year Offset

The convergence schedule uses a 5-10-5 structure: rates interpolate from recent to medium (years 1-5), hold at medium (years 6-15), then interpolate from medium to long-term (years 16-20). Years 21-30 hold at long-term.

| Statistic | YO=1 | YO=5 | YO=10 | YO=20 | YO=30 |
|-----------|-----:|-----:|------:|------:|------:|
| Mean | -0.90% | -0.57% | -0.57% | -0.74% | -0.74% |
| Median | -0.69% | -0.24% | -0.24% | -0.25% | -0.25% |
| Std Dev | 4.70% | 3.27% | 3.27% | 3.20% | 3.20% |
| Min | -23.59% | -15.58% | -15.58% | -16.09% | -16.09% |
| P1 | -13.19% | -9.73% | -9.73% | -11.01% | -11.01% |
| P5 | -8.25% | -6.45% | -6.45% | -6.91% | -6.91% |
| P25 | -3.20% | -2.12% | -2.12% | -1.90% | -1.90% |
| P75 | +1.14% | +1.05% | +1.05% | +0.84% | +0.84% |
| P95 | +6.22% | +4.18% | +4.18% | +3.56% | +3.56% |
| P99 | +14.47% | +8.45% | +8.45% | +6.73% | +6.73% |
| P99.5 | +16.68% | +10.93% | +10.93% | +9.68% | +9.68% |
| P99.9 | +23.43% | +14.37% | +14.37% | +13.09% | +13.09% |
| Max | +29.76% | +16.04% | +16.04% | +15.00% | +15.00% |

**Key observation:** Year offset 1 has the widest spread (driven by the "recent" window, which is a single period 2020-2024). The medium and long-term averages attenuate extremes but still carry rates as high as +16% and as low as -16%.

### 3.2 Cells Exceeding Positive Thresholds

| Threshold | YO=1 | YO=5 | YO=10 | YO=20 | YO=30 |
|-----------|-----:|-----:|------:|------:|------:|
| > +2% | 367 (19.2%) | 306 (16.0%) | 306 (16.0%) | 227 (11.9%) | 227 (11.9%) |
| > +3% | 261 (13.7%) | 168 (8.8%) | 168 (8.8%) | 133 (7.0%) | 133 (7.0%) |
| > +5% | 141 (7.4%) | 70 (3.7%) | 70 (3.7%) | 45 (2.4%) | 45 (2.4%) |
| > +8% | 63 (3.3%) | 23 (1.2%) | 23 (1.2%) | 13 (0.7%) | 13 (0.7%) |
| > +10% | 39 (2.0%) | 14 (0.7%) | 14 (0.7%) | 9 (0.5%) | 9 (0.5%) |

### 3.3 Cells Exceeding Negative Thresholds

| Threshold | YO=1 | YO=5 | YO=10 | YO=20 | YO=30 |
|-----------|-----:|-----:|------:|------:|------:|
| < -2% | 682 (35.7%) | 495 (25.9%) | 495 (25.9%) | 451 (23.6%) | 451 (23.6%) |
| < -3% | 502 (26.3%) | 317 (16.6%) | 317 (16.6%) | 303 (15.9%) | 303 (15.9%) |
| < -5% | 272 (14.3%) | 162 (8.5%) | 162 (8.5%) | 173 (9.1%) | 173 (9.1%) |
| < -8% | 108 (5.7%) | 51 (2.7%) | 51 (2.7%) | 62 (3.3%) | 62 (3.3%) |
| < -10% | 54 (2.8%) | 17 (0.9%) | 17 (0.9%) | 33 (1.7%) | 33 (1.7%) |

---

## 4. Outlier Identification

### 4.1 Top Positive Outliers at Medium Hold (YO=10)

The 14 cells above +10% at YO=10 fall into two distinct categories:

**College-town university enrollment (7 cells, all in counties > 50K pop):**

| County | Pop 2025 | Age | Sex | Rate | 30-yr Compound |
|--------|-------:|-----|-----|-----:|---------------:|
| Grand Forks | 74,501 | 20-24 | Male | +14.0% | +4,931% |
| Cass | 201,794 | 20-24 | Female | +13.2% | +4,010% |
| Grand Forks | 74,501 | 20-24 | Female | +12.9% | +3,733% |
| Cass | 201,794 | 20-24 | Male | +12.4% | +3,190% |
| Ward | 68,233 | 20-24 | Male | +11.0% | +2,157% |
| Grand Forks | 74,501 | 15-19 | Female | +10.9% | +2,153% |
| Grand Forks | 74,501 | 15-19 | Male | +10.9% | +2,142% |

These rates reflect genuine university enrollment dynamics (NDSU, UND, Minot State). A 13% rate for Cass 20-24 implies ~2,600 net in-migrants per year into a cohort of ~20,000, consistent with NDSU's incoming class size of ~3,000/year.

**Small-county statistical noise (7 cells, all in counties < 5K pop):**

| County | Pop 2025 | Age | Sex | Rate | 30-yr Compound |
|--------|-------:|-----|-----|-----:|---------------:|
| Logan | 1,859 | 25-29 | Female | +16.0% | +8,580% |
| Billings | 1,071 | 30-34 | Female | +14.6% | +5,811% |
| Billings | 1,071 | 40-44 | Female | +14.4% | +5,492% |
| Hettinger | 2,492 | 25-29 | Female | +12.4% | +3,271% |
| Grant | 2,206 | 35-39 | Male | +10.8% | +2,045% |
| Sheridan | 1,296 | 25-29 | Female | +10.2% | +1,744% |
| Golden Valley | 1,808 | 25-29 | Female | +10.1% | +1,675% |

These rates are driven by tiny base populations (often 5-30 people in a cell). A single family moving to Logan County can produce a +16% migration rate for that age-sex cell. These are not economically meaningful trends.

### 4.2 County Concentration of High Positive Rates (>5%, YO=10)

| Tier | Counties | Cells > +5% | Max Rate | Character |
|------|----------|-------------|----------|-----------|
| Large (>50K) | 4 counties | 10 cells | +14.0% | University enrollment |
| Medium (5-50K) | 19 counties | 10 cells | +8.5% | Mixed (oil, regional centers) |
| Small (<5K) | 30 counties | 50 cells | +16.0% | Statistical noise |

**71% of cells above +5% are in small counties.** The most affected age groups are 25-29 (31 cells) and 20-24 (9 cells).

### 4.3 County Concentration of High Negative Rates (<-5%, YO=10)

| Tier | Counties | Cells < -5% | Min Rate | Character |
|------|----------|-------------|----------|-----------|
| Large (>50K) | 3 counties | 7 cells | -12.0% | College-exit (25-29) |
| Medium (5-50K) | 19 counties | 29 cells | -14.4% | Elderly exit (85+) |
| Small (<5K) | 47 counties | 126 cells | -15.6% | Elderly exit + young adult exit |

**78% of negative outlier cells are in small counties.** The most affected age groups are 85+ (69 cells) and 20-24 (53 cells), reflecting elderly migration to care facilities and young-adult exit from rural areas.

### 4.4 Oil County Detail: McKenzie (38053) at YO=10

| Age | Sex | Rate | Notes |
|-----|-----|-----:|-------|
| 25-29 | Female | +8.45% | Boom-era residual; highest rate |
| 25-29 | Male | +6.26% | Boom-era residual |
| 5-9 | Male | +3.90% | Family in-migration |
| 5-9 | Female | +3.69% | Family in-migration |
| 30-34 | Female | +3.65% | Working-age in-migration |
| 10-14 | Male | +3.61% | Family in-migration |
| 30-34 | Male | +3.60% | Working-age in-migration |
| ... (18 more cells) | | | Range: +3.3% to -3.9% |

McKenzie's county-level mean rate is **+1.26%** (compounding to +45% over 30 years). This is driven by a broad pattern of positive rates across many working-age cells, not a single extreme outlier. **No rate cap below 8% can meaningfully reduce McKenzie's 30-year growth without also clipping Cass/Fargo's legitimate rates.**

---

## 5. Cass County (Fargo) Analysis

### 5.1 Rate Profile at YO=10

| Age | Sex | Rate |
|-----|-----|-----:|
| 20-24 | Female | +13.19% |
| 20-24 | Male | +12.35% |
| 15-19 | Female | +6.58% |
| 15-19 | Male | +6.30% |
| 5-9 | Male | +0.70% |
| (remaining 31 cells) | | -0.59% to +0.35% |

Cass's high rates are concentrated in exactly 4 cells (15-19 and 20-24, both sexes), all reflecting NDSU enrollment. The county mean rate is **+0.62%** (compounding to +20.5% over 30 years), which is a plausible growth trajectory for Fargo.

### 5.2 Cap Sensitivity

| Cap Threshold | Cass Cells Clipped at YO=10 | Cass 30-yr Growth |
|---------------|---:|---:|
| No cap | 0 | +20.5% |
| 15% | 0 | +20.5% |
| 12% | 2 (20-24 M, F) | +19.0% |
| 10% | 2 (20-24 M, F) | +15.1% |
| 8% | 2 (20-24 M, F) | +11.4% |
| 6% | 4 (20-24 + 15-19 M, F) | +6.9% |
| 5% | 4 (20-24 + 15-19 M, F) | +3.4% |

**Any cap below 13.2% clips Cass's college-age rates.** A cap at 6% or below would reduce Fargo's 30-year growth to unrealistically low levels.

---

## 6. Reservation County Analysis

### 6.1 Benson County (38005, pop 5,759)

Most extreme rates at YO=10: 85+ Female at -14.4%, 85+ Male at -8.6%, 20-24 Male at -7.8%.
County mean rate: **-2.89%** (compounding to -58.5% over 30 years).

### 6.2 Rolette County (38079, pop 11,688)

Most extreme rates at YO=10: 85+ Male at -12.4%, 85+ Female at -10.6%, 80-84 Male at -8.3%.
County mean rate: **-3.06%** (compounding to -60.7% over 30 years).

### 6.3 Sioux County (38085, pop 3,667)

Most extreme rates at YO=10: 70-74 Female at -10.1%, 85+ Female at -10.1%, 75-79 Female at -6.9%.
County mean rate: **-2.86%** (compounding to -58.1% over 30 years).

### 6.4 Cap Sensitivity for Reservation Counties

| Cap | Benson 30yr | Rolette 30yr | Sioux 30yr |
|-----|---:|---:|---:|
| No cap | -58.5% | -60.7% | -58.1% |
| -15% | -58.5% | -60.7% | -58.1% |
| -10% | -57.5% | -59.6% | -57.9% |
| -8% | -56.0% | -58.1% | -56.6% |
| -6% | -52.4% | -54.7% | -54.7% |

The cap has only modest effect on reservation counties because their declines are driven by consistently negative rates across many cells (mean of -2.9% to -3.1%), not by a few extreme outliers.

---

## 7. Residual Migration Rates (Pre-Convergence Context)

The residual migration rates span 5 periods (2000-2005 through 2020-2024) and have a wider range than the convergence rates:

| Statistic | Value |
|-----------|------:|
| Min | -39.05% |
| Max | +37.57% |
| P1 | -13.82% |
| P99 | +10.79% |
| Mean | -0.74% |
| Std Dev | 4.31% |

The most extreme residual rates (e.g., Slope 85+ Male at +37.6% in 2020-2024, with a base population of 7) confirm that small-cell noise is the primary driver of outliers. The convergence averaging already reduces these by a factor of 2-5x, but insufficient attenuation remains in the medium-term window.

---

## 8. Cap Option Evaluation

### 8.1 Option A: Symmetric Cap

Clip all rates to [-X%, +X%].

| Cap | Cells Clipped (YO=10) | % | Problem |
|-----|---:|---:|-----------|
| 5% | 232 | 12.2% | Clips Cass 15-19 and 20-24; Fargo growth drops to +3.4% |
| 6% | 157 | 8.2% | Clips Cass 20-24; Fargo growth drops to +6.9% |
| 8% | 74 | 3.9% | Clips Cass 20-24; Fargo growth drops to +11.4% |
| 10% | 31 | 1.6% | Clips Cass 20-24 and Grand Forks 15-19; Fargo drops to +15.1% |
| 12% | 14 | 0.7% | Clips only extreme small-county noise; minimal impact |
| 15% | 2 | 0.1% | Only clips Logan and Billings; negligible impact |

**Verdict:** Any symmetric cap that meaningfully clips oil-county rates also clips legitimate college-town dynamics. **Not recommended** as sole approach.

### 8.2 Option B: Asymmetric Cap (Different Pos/Neg Limits)

| Pos Cap | Neg Cap | Cells Clipped | % |
|---------|---------|---:|---:|
| +8% | -10% | 40 | 2.1% |
| +8% | -12% | 29 | 1.5% |
| +10% | -10% | 31 | 1.6% |
| +10% | -12% | 20 | 1.0% |

**Verdict:** Addresses the positive/negative asymmetry but still cannot distinguish college-town 20-24 from oil-county 25-29 rates.

### 8.3 Option C: Relative Cap (N times State Average)

Cap each cell at N times the state average rate for that age-sex combination.

| Multiplier | Cells Clipped | % |
|------------|---:|---:|
| 2x | 770 | 40.4% |
| 3x | 543 | 28.5% |
| 5x | 369 | 19.3% |
| 10x | 243 | 12.7% |

**Verdict:** Too aggressive at any reasonable multiplier. The state average is often near zero (many counties are declining), so even 10x the state average clips 12.7% of cells. **Not recommended.**

### 8.4 Option D: Population-Tiered Cap

Apply different cap levels based on county population size.

| Small (<5K) | Medium (5-50K) | Large (50K+) | Cells Clipped | % |
|-------------|----------------|--------------|---:|---:|
| 5% | 8% | 15% | 164 | 8.6% |
| 6% | 10% | 15% | 107 | 5.6% |
| 8% | 10% | 15% | 54 | 2.8% |

**Verdict:** Population-based thresholds add implementation complexity. The age-aware approach (Option E below) achieves similar targeting with simpler logic and clearer demographic justification.

### 8.5 Option E: Age-Aware Cap (Recommended)

Clip rates to **[-15%, +15%] for ages 15-24** and **[-8%, +8%] for all other ages**.

| Year Offset | College Clipped | Other >+8% | Other <-8% | Total | % |
|-------------|---:|---:|---:|---:|---:|
| 1 | 4 | 49 | 86 | 139 | 7.3% |
| 5 | 0 | 15 | 32 | 47 | 2.5% |
| 10 | 0 | 15 | 32 | 47 | 2.5% |
| 20 | 4 | 6 | 21 | 31 | 1.6% |
| 30 | 4 | 6 | 21 | 31 | 1.6% |
| **All** | **56** | **399** | **917** | **1,372** | **2.4%** |

---

## 9. Recommended Cap: Age-Aware [-15%/+15%] for 15-24, [-8%/+8%] for Others

### 9.1 Rationale

1. **College-town rates (15-24) are genuine.** University enrollment produces migration rates of 10-14% in Cass, Grand Forks, and Ward. These reflect real institutional dynamics (NDSU, UND, Minot State) that have been stable for decades. The 15% upper limit provides headroom without allowing implausible rates.

2. **Non-college-age rates above 8% are almost exclusively noise.** Outside the 15-24 age group, rates above 8% at the medium hold (YO=10) are found only in:
   - 7 small counties (<5K pop) with tiny cell populations
   - Billings County (pop 1,071) with 3 oil-boom residual cells
   - McKenzie County with 1 cell marginally above (8.45%)

3. **The 8% threshold sits at P99 of the medium-term distribution.** For non-college ages at YO=10, P99 = 8.13% and P99.5 = 9.97%. The 8% cap clips ~1% of non-college cells, targeting only the statistical tail.

4. **Negative rates below -8% are predominantly 85+ and 20-24 cells.** The 85+ negative rates reflect small-cell noise (elderly populations of 5-50 people) and should not compound for 30 years. The 20-24 negative rates reflect young-adult exit from rural areas, which at -13% would imply near-complete depopulation of the age cell.

### 9.2 Impact on Key Counties

| County | Pop | Base 30yr | Capped 30yr | Delta |
|--------|----:|----------:|------------:|------:|
| Cass (Fargo) | 201,794 | +20.5% | **+20.5%** | 0.0% |
| Burleigh (Bismarck) | 103,251 | +19.3% | **+19.3%** | 0.0% |
| Grand Forks | 74,501 | -12.3% | **-7.6%** | +4.7% |
| Ward (Minot) | 68,233 | -20.8% | **-20.8%** | 0.0% |
| Williams | 41,767 | +10.8% | **+10.8%** | 0.0% |
| McKenzie | 15,192 | +45.4% | **+44.9%** | -0.5% |
| Rolette | 11,688 | -60.7% | **-58.1%** | +2.6% |
| Benson | 5,759 | -58.5% | **-56.0%** | +2.6% |
| Sioux | 3,667 | -58.1% | **-56.6%** | +1.6% |
| Billings | 1,071 | +54.6% | **+38.0%** | -16.6% |
| Logan | 1,859 | -7.7% | **-13.7%** | -6.0% |

**Cass/Fargo is completely unaffected.** Billings sees the largest reduction (-16.6 percentage points). Grand Forks benefits from capping extreme 25-29 exit rates (which were -12% for females leaving after graduation). Reservation counties see modest reductions in decline rates (2-3 percentage points) from capping extreme 85+ exit rates.

### 9.3 Cells with Change > 1 Percentage Point

| Year Offset | Cells Changed > 1pp | % of 1,908 |
|-------------|---:|---:|
| 1 | 99 | 5.2% |
| 5 | 31 | 1.6% |
| 10 | 31 | 1.6% |
| 20 | 18 | 0.9% |
| 30 | 18 | 0.9% |

---

## 10. Implementation

### 10.1 Location

The cap should be applied in `cohort_projections/data/process/convergence_interpolation.py`, specifically in the `calculate_age_specific_convergence()` function, after computing the interpolated rate for each year but before storing it in the results dict.

This is the correct insertion point because:
1. It catches all three convergence phases (recent-to-medium, medium hold, medium-to-long-term)
2. It operates on the final rate that will be used by the projection engine
3. It does not modify the underlying window averages, preserving data lineage

### 10.2 Configuration

Add to `config/projection_config.yaml` under `rates.migration.interpolation`:

```yaml
rates:
  migration:
    interpolation:
      rate_cap:
        college_ages: ["15-19", "20-24"]
        college_cap: 0.15        # +/-15% for ages 15-24
        general_cap: 0.08        # +/-8% for all other ages
        enabled: true
```

### 10.3 Code Change (Pseudocode)

```python
# In calculate_age_specific_convergence(), after line 209:
if rate_cap_config and rate_cap_config.get("enabled", False):
    college_ages = rate_cap_config.get("college_ages", ["15-19", "20-24"])
    college_cap = rate_cap_config.get("college_cap", 0.15)
    general_cap = rate_cap_config.get("general_cap", 0.08)

    college_mask = merged["age_group"].isin(college_ages)
    rate = rate.clip(-general_cap, general_cap)
    rate[college_mask] = rate_before_cap[college_mask].clip(-college_cap, college_cap)
```

### 10.4 Relationship to Boom Dampening

The rate cap and boom dampening (ADR-040) are complementary:

| Mechanism | Target | Scope |
|-----------|--------|-------|
| Boom dampening | Reduce oil-boom period rates before averaging | Specific counties (dampening list) |
| Rate cap | Clip extreme individual cells after averaging | All counties, automatic |

Both are needed. The rate cap handles statistical noise that dampening cannot address (e.g., Logan County 25-29 Female at 16% has nothing to do with oil). Dampening handles the broad pattern of elevated working-age rates in oil counties that the cap cannot address (McKenzie's many cells at 3-8%).

---

## 11. Limitations and Caveats

1. **McKenzie remains at +45% even with the cap.** This is a Finding 2 issue that requires stronger boom dampening, not a rate cap.

2. **Reservation county declines remain at -56% to -58%.** The cap only modestly reduces their extreme 85+ rates. The fundamental driver is persistent negative migration across all working-age cells, which is a structural demographic pattern, not statistical noise.

3. **The cap is symmetric within each age tier.** A -8% cap on 85+ rates may clip genuine elderly out-migration patterns. However, these rates are the most statistically noisy (smallest populations) and the most likely to reflect measurement error.

4. **College-age dynamics may shift.** If NDSU or UND enrollment patterns change, the 15% cap could become too restrictive or too lenient. This should be reviewed if base data is updated.

5. **The cap does not address the fertility multiplier.** High working-age in-migration rates produce additional births through the fertility component, amplifying growth beyond what migration rates alone suggest. This interaction is not addressed by a rate cap.
