# Phase 3 Revision Plan

## Response to Dual AI Review (December 30, 2025)

**Source Reviews:**
- `outputs/ChatGPT_5-2-Pro_revision_critique.md`
- `outputs/Gemini_3_Pro_DeepThink_revision_critique.md`

**Overall Assessment:**
- ChatGPT 5.2 Pro: "Publishable with revisions" (not yet minor)
- Gemini 3 Pro: "Publishable with Minor Revisions"

---

## Reviewer Consensus Summary

### Fully Resolved (Both Agree)
1. ✅ Estimand/forecasting target
7. ✅ Scenario arithmetic consistency
8. ✅ Duration analysis connection
10. ✅ ACS measurement error
11. ✅ Tone adjustments

### Partially Resolved (Need Additional Work)
2. ⚠️ Module narrative integration (ChatGPT: needs VAR/ML results; Gemini: resolved)
3. ⚠️ Small-sample inference (ChatGPT: needs wild bootstrap; Gemini: resolved)
4. ⚠️ Forecast backtesting (Both: "oracle" benchmark issue)
5. ⚠️ Gravity model specification (ChatGPT: distance ambiguity; Gemini: abstract misreports)
6. ⚠️ Causal inference/DiD (Both: parallel trends failure → language change)
9. ⚠️ References/figures (ChatGPT: new numeric inconsistencies introduced)

---

## Phase 3 Task Structure

Tasks are organized into three tiers based on criticality and reviewer consensus.

### Tier 1: Critical Fixes (Both Reviewers Agree)
Must fix before resubmission. Failure to address will likely result in rejection.

### Tier 2: Important Improvements (One Reviewer Strongly Recommends)
Should fix to strengthen the paper. May not block acceptance but will invite criticism.

### Tier 3: Enhancements (Would Strengthen Paper)
Nice to have. Demonstrates thoroughness and may preempt future reviewer concerns.

---

## Tier 1: Critical Fixes

### P3.01: Abstract Corrections
**Priority:** 🔴 Critical
**Reviewers:** Both
**Type:** Text edit

**Issues:**
1. Gravity coefficient (0.14) is not statistically significant with controls (Gemini)
2. "75% reduction" implies clean causal effect when parallel trends fail (Both)

**Tasks:**
- [ ] Revise gravity sentence: "Gravity model estimation reveals that diaspora associations lose statistical significance when controlling for population mass"
- [ ] Revise DiD sentence: Change "75% reduction" to "policy-associated divergence of approximately 75%" with caveat about pre-existing trends
- [ ] Add qualifier that DiD estimate is likely an upper bound

**Files to modify:**
- `sections/01_introduction.tex` (abstract)

**Validation:** Read abstract aloud; verify no overclaiming

---

### P3.02: DiD Language Softening Throughout
**Priority:** 🔴 Critical
**Reviewers:** Both
**Type:** Text edit + minor analysis

**Issues:**
- Pre-trend test rejects parallel trends (p<0.05)
- Figure 8 shows divergence beginning ~10 years before treatment
- Cannot claim "causal effect" with failed parallel trends

**Tasks:**
- [ ] Search and replace "causal effect" → "associational divergence" or "policy-associated divergence" in DiD context
- [ ] Add explicit acknowledgment in Section 3.7.1 that estimate conflates policy with pre-existing trends
- [ ] Add sentence stating estimate is "likely an upper bound"
- [ ] Update Discussion section to reflect weaker causal claims
- [ ] Consider: Add restricted pre-period robustness check (years where trends parallel)

**Files to modify:**
- `sections/03_results.tex`
- `sections/04_discussion.tex`
- `sections/01_introduction.tex`

**Validation:** Grep for "causal" in DiD context; verify all hedged

---

### P3.03: Backtesting Oracle Benchmark Clarification
**Priority:** 🔴 Critical
**Reviewers:** Both
**Type:** Text edit + analysis extension

**Issues:**
- Driver OLS uses contemporaneous US migration (infeasible for real-time forecasting)
- Table 6 mixes feasible and infeasible models without clear labeling
- Paper argues Driver/VAR superior but uses ARIMA for scenarios without explanation

**Tasks:**
- [ ] Add "Oracle (ex-post)" label to Driver OLS row in Table 6
- [ ] Add footnote explaining Driver OLS requires future knowledge
- [ ] Add explicit justification for ARIMA in scenarios: "Although Driver OLS performs best historically, it relies on oracle knowledge of future national flows. Therefore, we use the conservative ARIMA baseline for long-term projection."
- [ ] **Enhancement:** Add feasible competitor (lagged Driver or VAR-based forecast)
- [ ] **Enhancement:** Add MASE metric alongside/instead of MAPE (robust to 2020 near-zero)

**Files to modify:**
- `sections/03_results.tex` (Table 6 and surrounding text)
- Potentially: `module_02_time_series/backtesting.py` (if adding feasible competitor)

**Validation:** Table clearly distinguishes feasible vs oracle benchmarks

---

### P3.04: Figure 3 Scaling/Labeling Fix
**Priority:** 🔴 Critical
**Reviewers:** ChatGPT
**Type:** Figure regeneration

**Issues:**
- Table 3 reports ND share as 0.10-0.30% (mean 0.173)
- Figure 3 Panel B shows values 10-30 with axis labeled "ND share of US int'l migration (%)"
- ×100 scaling mismatch (percent vs basis points)

**Tasks:**
- [ ] Investigate source data to determine correct scaling
- [ ] Either rescale figure data OR relabel axis (e.g., "basis points" or "×0.01%")
- [ ] Ensure Table 3 and Figure 3 use consistent units
- [ ] Regenerate figure with corrected labels/scaling
- [ ] Update figure caption if needed

**Files to modify:**
- Figure generation script
- `figures/fig_03_*.pdf`
- `sections/03_results.tex` (if caption needs update)

**Validation:** Table 3 values match Figure 3 visual; axis label matches scale

---

### P3.05: Location Quotient Denominator Audit
**Priority:** 🔴 Critical
**Reviewers:** ChatGPT
**Type:** Data audit + text clarification

**Issues:**
- Table 4 "US Share (%)" values seem implausible
- India's share appears far too small if denominator is total US foreign-born
- Kenya's 5.77% is eyebrow-raising
- LQs >15 are highlighted in abstract—must be correct

**Tasks:**
- [ ] Audit LQ calculation in source code
- [ ] Verify denominators: ND share = (origin-born in ND) / (total foreign-born in ND); US share = (origin-born in US) / (total foreign-born in US)
- [ ] Check for percent vs proportion confusion
- [ ] Add explicit denominator statement to Methods or table footnote
- [ ] Add sanity check in text: "India-born share of total U.S. foreign-born is X% in ACS 2023"
- [ ] Regenerate Table 4 if calculation was incorrect

**Files to modify:**
- LQ calculation script (audit)
- `sections/02_data_methods.tex` or `sections/03_results.tex`
- Table 4 if values incorrect

**Validation:** LQ formula explicitly stated; denominators documented; sanity check passes

---

### P3.06: Duplicate Kaplan-Meier Figure Resolution
**Priority:** 🔴 Critical
**Reviewers:** ChatGPT
**Type:** Figure reorganization

**Issues:**
- KM intensity-quartile plot appears as both Figure 2 (p. 27) and Figure 9 (p. 35)
- Different captions but same/similar content
- Reader confusion: "why am I seeing this twice?"

**Tasks:**
- [ ] Identify which figure appearances are necessary
- [ ] Keep one canonical KM figure in Results
- [ ] If second needed, make genuinely different (e.g., regional strata vs intensity strata)
- [ ] Renumber figures if removing duplicate
- [ ] Update all figure references in text

**Files to modify:**
- `main.tex` or section files (figure placement)
- Figure numbering throughout

**Validation:** Each figure appears exactly once; no duplicates

---

### P3.07: Residual Normality Statistics Consistency
**Priority:** 🔴 Critical
**Reviewers:** ChatGPT
**Type:** Data reconciliation

**Issues:**
- Text reports Shapiro-Wilk W=0.966, p=0.820
- Figure 12 caption repeats W=0.966, p=0.820
- Figure 12 plot annotation shows W=0.939, p=0.411
- Versioning error from different runs

**Tasks:**
- [ ] Rerun ARIMA residual diagnostics once
- [ ] Record single authoritative W and p values
- [ ] Update Section 3.3 text
- [ ] Update Figure 12 caption
- [ ] Regenerate Figure 12 with correct annotation
- [ ] Verify all three locations match

**Files to modify:**
- Diagnostic script (rerun)
- `sections/03_results.tex`
- `figures/fig_12_residuals.pdf` (or equivalent)
- Figure caption file

**Validation:** Grep for Shapiro-Wilk; all instances show same values

---

### P3.08: Monte Carlo Double-Counting Clarification
**Priority:** 🔴 Critical
**Reviewers:** Gemini
**Type:** Text clarification

**Issues:**
- ARIMA baseline trained on Net migration (includes refugee variance)
- Wave Duration simulation (Module 8) adds refugee wave draws on top
- Risk of double-counting refugee volatility

**Tasks:**
- [ ] Review Monte Carlo implementation to understand actual approach
- [ ] Add clarification in Appendix B.5 explaining either:
  - (a) ARIMA was trained on refugee-stripped series, OR
  - (b) Wave simulation modulates existing variance rather than adding, OR
  - (c) Explicit acknowledgment of conservative (inflated) uncertainty bounds
- [ ] If actual double-counting exists, consider fixing methodology

**Files to modify:**
- `sections/06_appendix.tex` (Appendix B.5)
- Potentially: scenario modeling code if fix needed

**Validation:** Clear statement about how double-counting is avoided/handled

---

### P3.09: ITS Trend Change Coefficient Caveat
**Priority:** 🟡 High
**Reviewers:** Gemini
**Type:** Text edit

**Issues:**
- COVID-19 "Trend Change" coefficient is +14,113 in Table 9
- For mean migration ~1,800, this implies explosive growth
- Mathematical artifact of fitting linear trend to short steep recovery (2021-2024)

**Tasks:**
- [ ] Add cautionary note in text near Table 9 interpretation
- [ ] Explain coefficient represents short-term "rebound slope"
- [ ] State explicitly: should not be extrapolated as long-run trend
- [ ] Consider: Add ND-specific interaction term (ND × Post2020) to distinguish from national average

**Files to modify:**
- `sections/03_results.tex` (ITS results discussion)

**Validation:** Reader warned not to extrapolate explosive coefficient

---

## Tier 2: Important Improvements

### P3.10: Wild Cluster Bootstrap for DiD
**Priority:** 🟡 High
**Reviewers:** ChatGPT
**Type:** Analysis extension

**Issues:**
- Only 7 treated nationalities in Travel Ban DiD
- Clustering with few treated clusters yields fragile inference
- Conventional α=0.05 framing invites skepticism

**Tasks:**
- [ ] Implement wild cluster bootstrap for DiD ATT
- [ ] Or: implement randomization/permutation inference
- [ ] Report bootstrap/permutation p-value alongside conventional
- [ ] Add brief methods note explaining small-sample robustness approach
- [ ] Update Results table with robust inference

**Files to modify:**
- DiD analysis script
- `sections/02_data_methods.tex` (methods note)
- `sections/03_results.tex` (results table)

**Validation:** Small-sample robust inference reported

---

### P3.11: Feasible Forecast Benchmark Addition
**Priority:** 🟡 High
**Reviewers:** ChatGPT
**Type:** Analysis extension

**Issues:**
- Table 6 only has oracle (Driver OLS) as strong performer
- No feasible competitor that outperforms naive RW
- Weakens claim that multi-method approach adds value

**Tasks:**
- [ ] Implement one or more feasible alternatives:
  - VAR-based forecast (ND using lagged/forecasted US)
  - Driver model with lagged US migration (t-1)
  - ETS/ARIMA with drift
- [ ] Add to backtesting framework
- [ ] Include in Table 6 with clear "Feasible" label
- [ ] Add MASE and/or sMAPE metrics (robust to 2020 near-zero)

**Files to modify:**
- `module_02_time_series/backtesting.py`
- Results JSON files
- `sections/03_results.tex` (Table 6)

**Validation:** At least one feasible model shown; MASE/sMAPE reported

---

### P3.12: Module Outputs → Scenario Inputs Mapping Table
**Priority:** 🟡 High
**Reviewers:** ChatGPT
**Type:** Content addition

**Issues:**
- VAR described but results not shown in detail
- ML methods described but no corresponding output shown
- Modules feel like "vibes-based anthology" without clear mapping

**Tasks:**
- [ ] Create one-page mapping table showing:
  - Module 2 (time series): baseline drift/volatility priors
  - Module 7 (policy): bounds for restrictive/permissive multipliers
  - Module 8 (duration): wave persistence draw parameters
  - Module 5/6 (gravity/ML): composition/allocative parameters OR explicit note they are context-only (move to appendix)
- [ ] Add table to Methods or Results
- [ ] If VAR/ML don't feed scenarios, move detailed description to appendix

**Files to modify:**
- `sections/02_data_methods.tex` or `sections/03_results.tex`
- Potentially: move ML section to appendix

**Validation:** Clear pipeline from each module to scenario engine

---

### P3.13: Gravity Model Specification Transparency
**Priority:** 🟡 High
**Reviewers:** ChatGPT
**Type:** Text clarification + potential analysis

**Issues:**
- Equation (10) lists distance but Table 8 doesn't show it
- "Gravity" label implies distance but specification unclear
- FY2023 cross-section only—can't lean on panel variation

**Tasks:**
- [ ] Option A: Include distance variable and report coefficient
  - Source: Calculate origin country centroid → state centroid distance
  - Add to gravity regression
  - Report in table
- [ ] Option B: Rename model and clarify exclusion
  - Rename to "Cross-sectional allocation model (PPML)"
  - Add explicit note explaining distance exclusion rationale
  - Remove or caveat Equation (10) distance term
- [ ] Either way: Add transparency about what's in/out

**Files to modify:**
- Gravity analysis script (if adding distance)
- `sections/02_data_methods.tex`
- `sections/03_results.tex`
- Tables 8/13/16 (if adding distance)

**Validation:** Specification matches label; all terms in equation appear in table or explicitly excluded

---

### P3.14: Restricted Pre-Period DiD Robustness
**Priority:** 🟡 High
**Reviewers:** ChatGPT
**Type:** Analysis extension

**Issues:**
- Joint pre-trend test rejects parallel trends
- But early years show more divergence than later years
- Restricting to "plausibly parallel" years could strengthen inference

**Tasks:**
- [ ] Identify years where pre-trends look parallel (visual inspection of Figure 8)
- [ ] Re-estimate DiD using restricted pre-period
- [ ] Or: Re-weight controls matching on pre-trend slopes
- [ ] Report sensitivity results
- [ ] Note if findings robust to restriction

**Files to modify:**
- DiD analysis script
- `sections/03_results.tex` (robustness discussion)
- Potentially: appendix table

**Validation:** Robustness check addresses pre-trend concern

---

### P3.15: ITS ND-Specific Interaction Term
**Priority:** 🟡 Medium
**Reviewers:** ChatGPT
**Type:** Analysis extension

**Issues:**
- Current ITS estimates average state-level COVID disruption
- Paper's RQ is about ND specifically
- ND-specific coefficient would strengthen relevance

**Tasks:**
- [ ] Add ND × Post2020 and ND × trend_change interaction terms
- [ ] Estimate whether ND differs from national average
- [ ] If significant: report ND-specific effect
- [ ] If not significant: note ND tracks national average
- [ ] Or: clearly state current model estimates average state disruption, not ND-specific

**Files to modify:**
- ITS analysis script
- `sections/03_results.tex`

**Validation:** ND-specific vs national effect is clear

---

## Tier 3: Enhancements

### P3.16: VAR Results Display
**Priority:** 🟢 Medium
**Reviewers:** ChatGPT
**Type:** Content addition

**Issues:**
- VAR described in Methods
- VAR mentioned as dominating model averaging weight
- But VAR results, diagnostics, backtest not shown

**Tasks:**
- [ ] Add VAR coefficient table to appendix
- [ ] Add VAR diagnostics (stability, residual tests)
- [ ] Include VAR in backtesting table (even if infeasible forecast)
- [ ] Or: reduce VAR description in Methods to match output prominence

**Files to modify:**
- `sections/06_appendix.tex`
- `sections/03_results.tex` (brief reference to appendix)

**Validation:** VAR claims supported by shown results

---

### P3.17: ML Module Output Transparency
**Priority:** 🟢 Medium
**Reviewers:** ChatGPT
**Type:** Content addition or removal

**Issues:**
- Elastic Net, Random Forest, K-means described in detail
- Results barely presented
- "Promised chapter that never arrives"

**Tasks:**
- [ ] Option A: Add ML results section
  - Feature importance plots
  - Cross-validation performance
  - Comparison to linear methods
- [ ] Option B: Move ML description to appendix
  - Brief mention in Methods
  - Full description in appendix
  - Note it provides context but not primary contribution
- [ ] Option C: Remove ML section entirely if not contributing

**Files to modify:**
- `sections/02_data_methods.tex`
- `sections/03_results.tex`
- `sections/06_appendix.tex`

**Validation:** ML description matches output prominence

---

### P3.18: Significance Framing Language Update
**Priority:** 🟢 Low
**Reviewers:** ChatGPT
**Type:** Text edit

**Issues:**
- "Reported at conventional levels" (α=0.05) invites skepticism in small-n settings
- Better to acknowledge uncertainty explicitly

**Tasks:**
- [ ] Remove or soften "conventional significance level" language
- [ ] Replace with: "We report point estimates and standard errors; readers should interpret significance cautiously given small samples"
- [ ] Or: Adopt Bayesian framing where appropriate

**Files to modify:**
- `sections/03_results.tex` (intro paragraph)

**Validation:** No overconfident significance framing

---

### P3.19: Distance Variable Data Collection
**Priority:** 🟢 Low (if P3.13 chooses Option A)
**Reviewers:** ChatGPT
**Type:** Data enhancement

**Issues:**
- Gravity model could include distance if data available
- Would make "gravity" label more accurate
- Standard in migration literature

**Tasks:**
- [ ] Source origin country centroids (lat/lon)
- [ ] Source US state centroids (lat/lon)
- [ ] Calculate great-circle distance matrix
- [ ] Merge with gravity dataset
- [ ] Re-estimate with log(distance) covariate

**Files to modify:**
- Data processing scripts
- Gravity analysis script
- Tables and results

**Validation:** Distance coefficient reported in gravity table

---

### P3.20: Figure Quality and Consistency Audit
**Priority:** 🟢 Low
**Reviewers:** Implicit (publication standards)
**Type:** Quality assurance

**Tasks:**
- [ ] Verify all figures have consistent styling (fonts, colors, sizes)
- [ ] Ensure all figures have both PDF and PNG versions
- [ ] Check figure numbering is sequential with no gaps
- [ ] Verify all figure references in text resolve correctly
- [ ] Check captions are complete and accurate

**Files to modify:**
- All figure files
- Figure captions

**Validation:** Visual consistency across all figures

---

## Execution Strategy

### Agent Assignment Philosophy

Each task should be assigned to a dedicated Codex agent that can focus deeply on implementation. Tasks are grouped by similarity to allow agents to build context efficiently.

### Recommended Agent Groupings

**Agent Group A: Text/Language Edits (P3.01, P3.02, P3.09, P3.18)**
- Focus: Abstract, Results, Discussion language corrections
- Skills: LaTeX editing, search/replace, academic writing
- Dependencies: None

**Agent Group B: Numeric Consistency (P3.04, P3.05, P3.06, P3.07)**
- Focus: Figure/table data validation and reconciliation
- Skills: Data audit, figure regeneration, cross-referencing
- Dependencies: Access to source data and scripts

**Agent Group C: Backtesting/Forecasting (P3.03, P3.11)**
- Focus: Table 6 improvements, feasible benchmark addition
- Skills: Time series analysis, Python coding
- Dependencies: Backtesting module code

**Agent Group D: Causal Inference Robustness (P3.10, P3.14)**
- Focus: DiD wild bootstrap, restricted pre-period check
- Skills: Econometrics, bootstrap implementation
- Dependencies: DiD analysis code, panel data

**Agent Group E: Model Specification (P3.08, P3.13, P3.15)**
- Focus: Monte Carlo clarification, gravity transparency, ITS enhancement
- Skills: Methods writing, potential reanalysis
- Dependencies: Scenario and gravity modules

**Agent Group F: Module Integration (P3.12, P3.16, P3.17)**
- Focus: Pipeline mapping table, VAR/ML results
- Skills: Technical writing, appendix organization
- Dependencies: All module outputs

**Agent Group G: Quality Assurance (P3.20)**
- Focus: Final figure/reference audit
- Skills: LaTeX compilation, visual inspection
- Dependencies: All other tasks complete

### Execution Order

1. **Parallel Wave 1:** Groups A, B, C (independent)
2. **Parallel Wave 2:** Groups D, E (may depend on Wave 1 findings)
3. **Parallel Wave 3:** Group F (depends on earlier analysis decisions)
4. **Final Wave:** Group G (depends on all others)

---

## Success Criteria

### Minimum Bar (Must Achieve)
- [ ] All Tier 1 tasks complete
- [ ] No numeric inconsistencies remain
- [ ] DiD claims appropriately hedged
- [ ] Backtesting table clearly labels oracle vs feasible
- [ ] Abstract accurately reflects results

### Target Bar (Should Achieve)
- [ ] All Tier 1 and Tier 2 tasks complete
- [ ] Wild cluster bootstrap reported for DiD
- [ ] At least one feasible forecast benchmark added
- [ ] Module → Scenario mapping clear

### Stretch Bar (Nice to Achieve)
- [ ] All tiers complete
- [ ] Distance included in gravity model
- [ ] VAR/ML results fully documented
- [ ] Figure quality audit passed

---

## Files Created by This Plan

- `PHASE_3_REVISION_PLAN.md` (this file)
- Task tracking will be added to `REVISION_STATUS.md`

---

*Created: December 30, 2025*
*Status: Ready for Agent Assignment*
