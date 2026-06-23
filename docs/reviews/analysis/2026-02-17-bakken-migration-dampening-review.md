# Bakken Oil Boom Migration Dampening Review

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-17 |
| **Reviewer** | Claude Code (Opus 4.6) |
| **Scope** | Migration dampening for oil-impacted counties; convergence window interaction |
| **Status** | Findings documented; corrective action needed |

---

## 1. Summary

The baseline projection produces implausible growth for Bakken-area counties:

| County | FIPS | 2025 | 2055 | 30-Year % Change |
|--------|------|-----:|-----:|-----------------:|
| McKenzie | 38053 | 15,192 | 32,556 | **+114.3%** |
| Williams | 38105 | 41,767 | 72,445 | **+73.5%** |
| Stark | 38089 | 34,013 | 42,070 | +23.7% |
| Mountrail | 38061 | 9,395 | 9,573 | +1.9% |
| Dunn | 38025 | 4,058 | 3,766 | -7.2% |
| **State total** | | **799,358** | **1,020,775** | **+27.7%** |

McKenzie at +114% and Williams at +74% are far above the state average (+28%) and are not supported by recent migration trends. The most recent period (2020-2024) shows **negative** net migration for both counties (McKenzie: -893, Williams: -3,033).

---

## 2. Current Dampening Mechanisms

### 2a. Period dampening (config lines 129-141)

A 0.60 factor is applied to oil county migration rates in the **residual migration** step, but only for configured boom periods:

| Period | Dampened? | Factor |
|--------|-----------|--------|
| 2000-2005 | No | 1.00 |
| 2005-2010 | **Yes** | 0.60 |
| 2010-2015 | **Yes** | 0.60 |
| 2015-2020 | No | 1.00 |
| 2020-2024 | No | 1.00 |

Counties affected: Williams (38105), McKenzie (38053), Mountrail (38061), Dunn (38025), Stark (38089).

### 2b. Male dampening (config lines 162-167)

An additional 0.80 factor on male migration rates in boom periods (all counties, not just oil counties). Combined male reduction in boom periods: 0.60 x 0.80 = 0.48.

---

## 3. Root Cause: Convergence Window Includes Undampened Boom-Era Migration

### The convergence window-to-period mapping

The convergence interpolation (config lines 171-179) uses three windows:

| Window | Config Range | Maps to Periods | Description |
|--------|-------------|-----------------|-------------|
| Recent | [2023, 2025] | (2020, 2024) | 1 period |
| Medium | [2014, 2025] | (2010, 2015), (2015, 2020), (2020, 2024) | 3 periods |
| Long-term | [2000, 2025] | All 5 periods | 5 periods |

The convergence schedule then applies these averaged rates:

| Phase | Years | Rate Used |
|-------|-------|-----------|
| Recent -> Medium | 1-5 | Linear ramp from recent to medium |
| Medium hold | 6-15 | Hold at medium |
| Medium -> Long-term | 16-20 | Linear ramp from medium to long-term |
| Long-term hold | 21-30 | Hold at long-term |

### The problem: 2015-2020 is undampened but still boom-elevated

McKenzie County residual migration rates by period (mean across all age-sex cells):

| Period | Mean Rate | Dampened? | In Medium Window? |
|--------|----------:|-----------|-------------------|
| 2000-2005 | -0.00497 | No | No |
| 2005-2010 | +0.00282 | Yes (0.60x) | No |
| 2010-2015 | +0.04003 | Yes (0.60x) | **Yes** |
| 2015-2020 | +0.02679 | **No** | **Yes** |
| 2020-2024 | -0.01827 | No | **Yes** |

Even after 0.60x dampening on the 2010-2015 period, the **undampened** 2015-2020 period (+0.027) pulls the medium-window average strongly positive (+0.016). This medium rate is then held for 10 straight years (years 6-15 of the projection).

### Window averages for oil counties vs. comparison counties

| County | Recent | Medium | Long-term | Pattern |
|--------|-------:|-------:|----------:|---------|
| McKenzie | -0.01827 | **+0.01612** | +0.00928 | recent << medium |
| Williams | -0.02283 | **+0.00605** | +0.00007 | recent << medium |
| Cass (Fargo) | +0.00234 | +0.00624 | +0.00543 | normal |
| Burleigh (Bismarck) | +0.00040 | +0.00590 | +0.00346 | normal |

For McKenzie and Williams, the medium rate is **strongly positive** while the recent rate is **strongly negative**. The convergence schedule ramps from the negative recent rate toward the positive medium rate over years 1-5, then holds the positive medium rate for 10 years. This is the engine of the implausible growth.

### Compounding effects

1. **Working-age magnification**: The boom disproportionately attracted 20-34 year-olds. McKenzie's 25-29 convergence rate at medium is ~8-10% annually. These in-migrants are in peak childbearing years.

2. **Fertility multiplier**: Each year's in-migrants produce children, compounding population growth beyond just migration.

3. **Small denominator**: McKenzie has only ~15,000 people. Even modest absolute migration flows produce large percentage rates that compound over 30 years.

---

## 4. Why 2015-2020 Should Be Considered Boom-Adjacent

The 2015-2020 period is not dampened because it is not configured as a boom period, but it should be reconsidered:

- **2015-2020 McKenzie migration: +2,612 net people** — This is *higher* in absolute terms than the dampened 2010-2015 period (+2,048 after dampening).
- The Bakken boom created infrastructure (housing, services, roads) that continued attracting population through 2019 even as drilling slowed.
- Population momentum from boom arrivals: workers who came 2011-2015 brought families, enrolled children, established businesses.
- The COVID disruption in 2020 artificially truncates this period, but 2015-2019 was still elevated.

---

## 5. Options for Corrective Action

### Option A: Extend boom dampening to 2015-2020

**Change**: Add `[2015, 2020]` to `boom_periods` in config.

**Effect**: The 0.60x factor would apply to the 2015-2020 period for oil counties. The medium window average would drop from approximately +0.016 to approximately +0.006 for McKenzie.

**Pros**: Simple config change; preserves existing methodology; addresses the specific problem.

**Cons**: Somewhat arbitrary — is the boom really over in 2020 but not 2019?

### Option B: Narrow the medium window to exclude boom years

**Change**: Move `medium_period` from `[2014, 2025]` to `[2020, 2025]` (or `[2018, 2025]`).

**Effect**: Medium window would only include the post-boom period (2020-2024), which shows negative migration for oil counties. The convergence would ramp from recent (negative) toward medium (also negative), then converge to long-term (slightly positive).

**Pros**: Clean separation of boom era from projection basis.

**Cons**: Reduces medium window to only 1 period, losing multi-period stability. Also affects all 53 counties, not just the 5 oil counties.

### Option C: Apply dampening in the convergence step, not just residual

**Change**: After computing medium window averages, apply a secondary dampening factor to oil county medium-window rates.

**Effect**: Direct reduction of the problematic medium-window rate for oil counties.

**Pros**: Surgically targets the problem without changing the window or period dampening.

**Cons**: Adds complexity; two dampening layers could be confusing to document.

### Option D: Apply a migration rate cap

**Change**: No county's projected migration rate can exceed N times the state average (e.g., 1.5x or 2x).

**Effect**: Would cap McKenzie's outlier rates while leaving normal counties untouched.

**Pros**: Principle-based; prevents any single county from dominating.

**Cons**: Hard to calibrate N; could inadvertently cap legitimate high-growth counties (e.g., Cass/Fargo).

---

## 6. Related Files

| File | Relevance |
|------|-----------|
| `config/projection_config.yaml` (lines 129-179) | Dampening config, convergence windows |
| `cohort_projections/data/process/residual_migration.py` (lines 388-442) | `apply_period_dampening()` |
| `cohort_projections/data/process/convergence_interpolation.py` (lines 130-220) | `calculate_age_specific_convergence()` |
| `scripts/pipeline/01_compute_residual_migration.py` | Runs dampening as part of residual computation |
| `scripts/pipeline/01b_compute_convergence.py` | Computes window averages and convergence schedule |
| `cohort_projections/data/process/pep_regime_analysis.py` | Regime classification (boom/bust/recovery) |
