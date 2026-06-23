# Projection Accuracy Analysis: Problems, Findings, and Recommendations

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-04 |
| **Author** | Claude Code (Opus 4.6) |
| **Type** | Comprehensive accuracy review |
| **Scope** | Sensitivity analysis, walk-forward backtests, SDC comparison, QC reports |
| **Status** | Active reference |
| **Related ADRs** | ADR-036, ADR-049, ADR-055, ADR-057 |
| **Related Reviews** | [`2026-03-04-college-fix-research-implications.md`](2026-03-04-college-fix-research-implications.md) |
| **Related Scripts** | `scripts/analysis/sensitivity_analysis.py`, `scripts/analysis/walk_forward_validation.py`, `scripts/exports/build_sensitivity_report.py`, `scripts/exports/build_sdc_comparison_report.py`, `scripts/exports/build_qc_report.py` |
| **Data Outputs** | `data/analysis/walk_forward/`, `data/exports/sensitivity/` |

---

## Overall Verdict

The 2026 model (m2026) is structurally sound and well-validated, but it has a **systematic under-projection bias** that grows with horizon length. At short horizons (1-5 years), accuracy is excellent. At medium horizons (5-10 years from recent origins), it's acceptable. But the model is **too conservative** — it consistently underestimates North Dakota's growth, and in some areas (urban/college counties), it's actually *less accurate* than the simpler SDC 2024 approach it aims to replace.

---

## The Five Core Problems

### 1. The Convergence Schedule Is Too Aggressive

This is the single most actionable finding. The 5-10-5 convergence schedule (5 years recent rates -> 10 years hold -> 5 years converge to long-run average) causes migration rates to decay toward historical equilibrium too quickly. The backtesting shows:

- **3.44 pp state error swing** — the largest m2026-specific parameter sensitivity
- The `all_longterm` convergence variant produces **-6.86% state error** from a 2015 origin (vs -3.26% for the blended baseline)
- The SDC method, which holds dampened rates *constant* with no convergence, outperforms m2026 at the state level at every horizon beyond 4 years
- ND has consistently grown faster than long-run equilibrium would predict — the mean-reversion assumption appears wrong for this state

**Recommendation:** Extend the hold period (e.g., 5-15-5 or 5-20-5), or reduce the convergence target. The backtesting evidence suggests ND's growth is structural, not cyclical, and converging to long-run averages prematurely suppresses it.

### 2. GQ Correction (ADR-055 Phase 2) Overcorrects Urban Counties

The GQ separation was theoretically sound, but Phase 2 (removing GQ from population snapshots before computing residual migration) removed genuine migration signal along with institutional noise:

- **37,084-person impact** on the 2050 projection — the third-largest sensitivity factor
- Urban/college counties (Cass, Grand Forks, Ward, Burleigh) are where the SDC **clearly outperforms** m2026:
  - Cass MAPE: SDC 10.1% vs m2026 13.3%
  - Ward MAPE: SDC 12.9% vs m2026 15.2%
  - Grand Forks MAPE: SDC 13.7% vs m2026 15.5%
  - Burleigh MAPE: SDC 5.8% vs m2026 7.4%
- The Phase 2 delta from Phase 1: State baseline **-1.5pp**, Cass **-1.9pp**, Grand Forks **-2.9pp**

**Recommendation:** Consider reverting to Phase 1 only (hold GQ constant but don't subtract from migration denominators), or calibrate the GQ correction factor so it removes a fraction rather than 100% of GQ population from the migration calculation.

### 3. College-Age Smoothing May Be Over-Dampening

College-age smoothing (ADR-049) reduces 2050 state population by **19,137** and is the dominant sensitivity parameter for Cass County. Combined with the GQ correction, it creates a double-dampening effect on university towns:

- Ward County: SDC projects growth to 85,975; m2026 projects **decline** to 59,420 — a 26,555-person gap
- Grand Forks: SDC projects 81,582; m2026 projects 67,501 — a 14,081-person gap
- Both Ward and Grand Forks are flagged as declining under baseline despite being the 3rd and 4th largest counties

**Recommendation:** Review whether the 50% statewide blending factor for college-age smoothing is too aggressive, especially when combined with GQ correction. These two features may be correcting for the same underlying issue.

### 4. Fertility Assumptions Deserve More Scrutiny

Fertility is the **#1 driver of state-level accuracy** in both frameworks:

- **4.07 pp state error swing** in backtesting — larger than any other parameter
- **72,348-person range** in the 2050 forward projection
- A 25% increase in fertility closes most of the 9-year under-projection gap
- This is a strong signal that ND birth rates were meaningfully under-projected relative to the 2015-2024 trajectory

**Recommendation:** Validate current fertility assumptions against actual 2020-2024 vital statistics. If TFR has been higher than projected, adjust upward. Consider whether the model uses national fertility trends vs ND-specific trends.

### 5. Small/Rural County Projections Have Unacceptable Uncertainty

- Slope County: projection range is **597%** of baseline (555 to 5,629 people)
- Nelson: **119%**, Billings: **92%**, Renville: **87%**
- Migration window weighting is the dominant sensitivity factor for **42 of 53 counties**
- 38 of 53 counties (72%) decline under baseline; median county growth is -7.1%

These projections are effectively noise for the smallest counties. This isn't necessarily fixable — it's a fundamental limitation of projecting small populations — but it needs to be communicated clearly.

**Recommendation:** Publish explicit uncertainty bands with all projections (the prediction interval infrastructure already exists). Consider floor constraints or Bayesian shrinkage toward regional averages for counties under a population threshold.

---

## How Far From Acceptable Error?

**What's acceptable** depends on horizon:

| Horizon | Target State APE | Current m2026 | Gap |
|---------|-----------------|---------------|-----|
| 5 yr | <3% | 6.6%* | ~4 pp |
| 10 yr | <5% | 17.1%* | ~12 pp |
| 15 yr | <8% | 25.3%* | ~17 pp |

*\*Averaged across all origins including pre-Bakken boom — unfairly penalizes both methods*

**From recent origins only (2015, 2020)** — the relevant benchmark for 2026 projections:

| Horizon | m2026 State APE | Assessment |
|---------|----------------|------------|
| 4 yr (2020->2024) | **0.32%** | Excellent |
| 9 yr (2015->2024) | **3.26%** | Good |

The model is already performing well from recent origins. The long-horizon errors are dominated by the Bakken boom — a structural break that is now fully incorporated in the training data. **The 2026 projection, launching from 2025 with complete boom-and-bust history, is in the best-case position of all tested origins.**

---

## What We Learn From Backtesting

1. **Neither method can predict structural breaks** (oil booms, pandemics). The `both_struggle = True` flag on all 2005/2010-origin folds confirms this is a fundamental limitation, not a model deficiency.

2. **Errors are serially correlated** (lag-1 autocorrelation 0.53-0.72). Errors compound over time and are not independent. Prediction intervals based on i.i.d. assumptions will be too narrow. The empirical percentile-based intervals already computed are the correct approach.

3. **Errors are non-normal** (Shapiro-Wilk p ~ 0, positive kurtosis ~9). Fat tails mean extreme outcomes are more likely than a Gaussian model would suggest.

4. **m2026 reduces systematic bias** (28% of variance vs SDC's 41%) but introduces more residual unexplained variance. The model is less biased but noisier.

5. **The 2026 model wins where it matters most** — from recent origins projecting forward. The SDC's advantage at long horizons comes from being *less wrong* about an unpredictable event, not from superior methodology.

---

## Prioritized Action Items

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| **1** | Relax convergence schedule (extend hold period or raise floor) | Reduce state-level under-projection by 2-4 pp |
| **2** | Recalibrate or revert GQ Phase 2 correction | Improve urban/college county accuracy by 2-3 pp MAPE |
| **3** | Audit fertility rate assumptions against 2020-2024 actuals | Potentially the largest single accuracy improvement |
| **4** | Review college-age smoothing interaction with GQ correction | Eliminate double-dampening on university counties |
| **5** | Publish uncertainty bands with all projections | Manage expectations, especially for small counties |

The model is well-built and well-tested. The primary deficiency is not structural — it's that several conservative design choices (convergence, GQ correction, college smoothing) compound to produce a systematic growth-dampening effect that the historical data doesn't support. Relaxing these constraints, guided by the backtesting evidence, should bring the projections into a tighter range.

---

## Supporting Data Summary

### Sensitivity Analysis: Parameter Impact Rankings

**Backtesting sensitivity (state % error swing, averaged across origins):**

| Method | Parameter | State Error Swing | County MAPE Swing |
|--------|-----------|-------------------|-------------------|
| M2026 | fertility_rate | 4.07 pp | 0.44 pp |
| SDC 2024 | fertility_rate | 4.05 pp | 0.40 pp |
| M2026 | convergence_schedule | 3.44 pp | 1.48 pp |
| M2026 | migration_rate | 0.19 pp | 2.91 pp |
| SDC 2024 | migration_rate | 0.73 pp | 1.87 pp |
| M2026 | survival_rate | 0.38 pp | 0.01 pp |
| SDC 2024 | sdc_bakken_dampening | 0.32 pp | 0.02 pp |
| M2026 | mortality_improvement | 0.16 pp | 0.02 pp |

**Forward projection sensitivity (2050 state population range):**

| Parameter | Min 2050 Pop | Max 2050 Pop | Range |
|-----------|-------------|-------------|-------|
| Fertility | 752,170 | 824,518 | **72,348** |
| Migration window weighting | 762,898 | 825,168 | **62,270** |
| GQ correction | 788,310 | 825,394 | **37,084** |
| Rate caps | 788,310 | 809,036 | **20,726** |
| College-age smoothing | 769,173 | 788,310 | **19,137** |
| Bakken dampening | 783,605 | 798,856 | 15,251 |
| Mortality improvement | 788,310 | 801,846 | 13,536 |
| Convergence schedule | 785,006 | 791,643 | 6,637 |

### Walk-Forward Validation: Accuracy by Horizon

**Annual-granularity validation (averaged across origins):**

| Horizon | SDC State APE | m2026 State APE | SDC County MAPE | m2026 County MAPE |
|---------|--------------|-----------------|-----------------|-------------------|
| 1 yr | 0.99% | 1.04% | 1.15% | 1.13% |
| 5 yr | 6.58% | 6.62% | 6.69% | 6.58% |
| 10 yr | 15.24% | 17.10% | 13.09% | 12.55% |
| 15 yr | 22.29% | 25.27% | 18.40% | 17.70% |
| 19 yr | 25.54% | 29.06% | 22.09% | 21.51% |

### Bias Analysis by County Category

- **Bakken**: Mean signed error reaches -37% at horizon 19. Both methods fail identically — neither captures oil-boom migration surges from pre-boom data.
- **Rural**: Consistent negative bias growing from -0.3% at h=1 to -15% to -18% at h=19. m2026 is 2-3 pp less biased than SDC.
- **Urban/College**: m2026 is *more* biased than SDC at long horizons (-34% vs -28% at h=19). The GQ correction and college smoothing overcorrect.
- **Reservation**: Highly volatile and regime-dependent. Short horizons near-zero; mid-horizons show +18-24% over-projection from 2015 origin.

### Decomposition Waterfall (SDC Base -> M2026 Baseline, 2050)

| Step | Feature | 2050 Pop | Delta |
|------|---------|----------|-------|
| 0 | SDC Base | 791,792 | -- |
| 1 | + BEBR Averaging | 788,310 | -3,482 |
| 2 | + Convergence (5-10-5) | 790,452 | +2,142 |
| 3 | + Mortality Improvement (0.5%/yr) | 797,462 | +7,010 |
| 4 | + Rate Caps | 817,705 | +20,243 |
| 5 | + College Smoothing (50%) | 809,824 | -7,881 |

### Error Decomposition

| Metric | SDC 2024 | M2026 |
|--------|----------|-------|
| Mean signed pct error | -7.05% | -6.16% |
| Horizon slope (error growth/yr) | -1.158%/yr | -1.081%/yr |
| Bias fraction of total variance | 41% | 28% |
| Residual fraction | 44% | 50% |

### County Report Card Highlights

| County | Category | m2026 MAPE | SDC MAPE | Grade |
|--------|----------|-----------|----------|-------|
| Mercer | Rural | 2.70% | 2.61% | A |
| Traill | Rural | 2.85% | 3.05% | A |
| Burleigh | Urban | 7.40% | 5.84% | C / B |
| Cass | Urban | 13.33% | 10.05% | C |
| Ward | Urban | 15.19% | 12.94% | D / C |
| Grand Forks | Urban | 15.52% | 13.71% | D / C |
| Williams | Bakken | 23.58% | 22.92% | D |
| McKenzie | Bakken | 29.65% | 29.50% | D |
