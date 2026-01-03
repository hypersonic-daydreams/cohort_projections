# ADR-020 / ADR-021 Reconciliation Report

## Document Information

| Field | Value |
|-------|-------|
| Created | 2026-01-02 |
| Author | Claude Code (Opus 4.5) |
| Purpose | Identify overlap, completion status, and conflicts between ADR-020 and ADR-021 |

---

## Executive Summary

**ADR-020** (Extended Time Series - Option C Hybrid) and **ADR-021** (Immigration Status Durability) both address policy-regime methodology for North Dakota immigration forecasting. Significant implementation occurred under ADR-021 that directly fulfills or supersedes planned ADR-020 Phase B tasks.

**Key Findings:**

| Category | Count | Notes |
|----------|-------|-------|
| **NOW COMPLETE** (via ADR-021) | 6 tasks | Core regime framework, synthetic control, scenarios |
| **PARTIALLY COMPLETE** | 4 tasks | Module structure exists; may need minor extensions |
| **STILL PENDING** | 5 tasks | Multi-state placebo, journal article, Bayesian, tests |
| **CONFLICTS** | 2 areas | Naming conventions, module organization |

**Recommendation:** Merge ADR-020 Phase B into ADR-021 framework, adopting ADR-021's completed modules as the authoritative implementation.

---

## Task-by-Task Analysis

### 1. NOW COMPLETE (ADR-020 Tasks Fulfilled by ADR-021)

#### 1.1 Regime-Aware Statistical Framework (B1)

| ADR-020 Task | ADR-021 Deliverable | Status |
|--------------|---------------------|--------|
| Create `module_regime_aware/` directory | **COMPLETE** | Directory exists with 6 submodules |
| `regime_definitions.py` | **COMPLETE** | `/sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/regime_definitions.py` |
| `vintage_dummies.py` | **COMPLETE** | Same directory |
| `piecewise_trends.py` | **COMPLETE** | Same directory |
| `covid_intervention.py` | **COMPLETE** | Same directory |
| `robust_inference.py` | **COMPLETE** | Same directory |
| `sensitivity_suite.py` | **COMPLETE** | Same directory |

**Additionally, ADR-021 created:**
- `module_regime_framework.py` - Comprehensive PolicyRegime enum and 25 policy events with Federal Register citations
- JSON export: `docs/governance/adrs/021-reports/results/rec4_regime_framework.json`

**ADR-021 Enhancement:** The ADR-021 implementation is more comprehensive than ADR-020's original plan:
- Three named regimes: EXPANSION (2010-2016), RESTRICTION (2017-2020), VOLATILITY (2021-2024)
- Policy event mapping with primary source citations (Presidential Determinations, Federal Register)
- Regime-specific parameters: mean migration, refugee share, trend coefficient, volatility ratio
- Chow test structural break evidence at 2017, 2020, 2021

#### 1.2 Synthetic Control Analysis (B1/B2 overlap)

| ADR-020 Task | ADR-021 Deliverable | Status |
|--------------|---------------------|--------|
| Synthetic control for ND | **COMPLETE** | `module_7b_lssnd_synthetic_control.py` |
| Placebo tests | **COMPLETE** | In-space placebo and leave-one-out analysis |
| Capacity parameter derivation | **COMPLETE** | 67.2% capacity multiplier post-LSSND closure |

**Key Results (from ADR-021):**
- **ATT (Average Treatment Effect):** -147 refugees/year fewer than expected post-LSSND
- **Capacity Multiplier:** 67.2% (ND receives 67% of what national share predicts)
- **Robustness:** Capacity range 58.5% - 64.9% across share calculation windows (SD=2.7%)

**Files:**
- `/sdc_2024_replication/scripts/statistical_analysis/module_7b_lssnd_synthetic_control.py`
- `/docs/governance/adrs/021-reports/results/rec3_lssnd_synthetic_control.json`
- `/docs/governance/adrs/021-reports/figures/rec3_lssnd_synthetic_control.png`

#### 1.3 Policy-Lever Scenario Framework (B1 extension)

| ADR-020 Task | ADR-021 Deliverable | Status |
|--------------|---------------------|--------|
| Sensitivity analysis specifications | **SUPERSEDED** | ADR-021 created mechanism-based scenarios |

**ADR-021 created `module_9b_policy_scenarios.py` with 5 named scenarios:**
1. **Durable-Growth:** High ceiling (150K) + full capacity recovery + 72.5% regularization
2. **Status-Quo:** Current policy trajectory continuation
3. **Welcome-Corps:** Private sponsorship growth with 110% capacity recovery
4. **Parole-Cliff:** Near-term high, then attrition years 2-4
5. **Restriction:** Low ceiling (15-30K) + 67.2% capacity + 29.9% regularization

**2045 Projection Results:**
| Scenario | Total | Durable | Temporary | 80% CI |
|----------|-------|---------|-----------|--------|
| Durable-Growth | 33,700 | 18,217 | 15,493 | [28,526, 38,771] |
| Status-Quo | 32,592 | 11,451 | 21,158 | [27,502, 37,587] |
| Parole-Cliff | 524 | 398 | 125 | [469, 577] |
| Restriction | 153 | 103 | 50 | [137, 168] |

#### 1.4 Status Durability Analysis (ADR-021 unique)

**Not in ADR-020 scope, but provides value for B1 regime-aware modeling:**

- `module_8b_status_durability.py` - Cox PH model with status interaction terms
- Key finding: Parole status has 11.29x hazard ratio vs refugee (p < 0.0001)
- 5-year survival: Refugee 95.7%, Parole 41.5%, Other ~85%
- Regularization probability parameter: 50.3% [29.9%, 72.5%]

#### 1.5 Two-Component Estimand Decomposition (ADR-021 unique)

**Not in ADR-020 scope, but implements regime-aware projection:**

- `module_10_two_component_estimand.py` - Y_t = Y_t^dur + Y_t^temp decomposition
- Historical mean (2010-2024): Y^dur = 7,465, Y^temp = 2,781, Durable share = 78.9%
- Integrates with Rec #6 scenarios for forward projections

#### 1.6 Secondary Migration Analysis (B2 partial)

| ADR-020 Task | ADR-021 Deliverable | Status |
|--------------|---------------------|--------|
| State-level analysis context | **PARTIAL** | ND-specific secondary migration analyzed |

- `module_secondary_migration.py` - Residual method decomposition
- Pre-COVID (2015-2019): Net positive secondary migration (~+544/year)
- Post-COVID (2020-2023): Net negative secondary migration (~-731/year)
- ND share of US migration: 0.17% (below population share of 0.23%)

---

### 2. PARTIALLY COMPLETE (ADR-020 Tasks with Overlap)

#### 2.1 Module B1 Main Script

| ADR-020 Planned | Current State | Gap |
|-----------------|---------------|-----|
| `module_B1_regime_aware_models.py` | File exists, uses module_regime_aware/* | May need sensitivity suite integration |

**Action Required:** Verify `module_B1_regime_aware_models.py` produces the sensitivity table specified in ADR-020 Agent B1 Plan (Section 3.5).

#### 2.2 50-State Panel Data

| ADR-020 Planned | ADR-021 State | Gap |
|-----------------|---------------|-----|
| Generate `all_states_migration_panel.csv` | Data exists in PostgreSQL + CSV | May need vintage labels |
| 50-state x 25-year with vintages | `combined_components_of_change.csv` has 2010-2024 | Missing 2000-2009 in panel format |

**Action Required:**
1. Extend `combined_components_of_change.csv` to include 2000-2009 data
2. Add vintage/regime labels per ADR-021 framework

#### 2.3 Journal Article Policy Documentation (B3 partial)

| ADR-020 Planned | ADR-021 Deliverable | Gap |
|-----------------|---------------------|-----|
| Data Comparability subsection | Policy timeline and conceptual diagrams created | Different focus |

**ADR-021 Created:**
- `journal_article/policy_timeline_table.md` - 28 policy events documented
- `journal_article/conceptual_diagram.md` - 5 Mermaid diagrams
- `journal_article/conceptual_diagram.tex` - 4 TikZ figures for LaTeX

**Gap:** ADR-020 B3 focused on "Data Comparability" subsection about PEP vintage methodology. ADR-021 focused on policy mechanism documentation. Both are needed.

#### 2.4 Robust Standard Errors Implementation

| ADR-020 Planned | Current State | Gap |
|-----------------|---------------|-----|
| HC0/HC1/HC2/HC3/HAC comparison | `robust_inference.py` exists | May need documentation |

**File exists:** `/sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/robust_inference.py`

---

### 3. STILL PENDING (ADR-020 Tasks Not Addressed by ADR-021)

#### 3.1 Multi-State Placebo Analysis (B2)

| Task | Status | Notes |
|------|--------|-------|
| Generate 50-state placebo distribution | **PENDING** | ADR-021 focused on ND-specific analysis |
| Calculate ND percentile in national distribution | **PENDING** | Key "real vs artifact" test |
| Oil state hypothesis test | **PENDING** | Compare ND shift to TX, OK, WY, AK |
| Geographic visualization (choropleth) | **PENDING** | Optional |

**Critical for ADR-020:** This is the key robustness test - "If everyone jumps similarly, that screams methodology. If ND is an outlier, that supports a real driver story."

**Files Still Needed:**
- `module_B2_multistate_placebo.py`
- `module_B2_multistate_placebo/` directory with data loading, shift calculator, oil state hypothesis

#### 3.2 Journal Article Data Comparability Section (B3)

| Task | Status | Notes |
|------|--------|-------|
| "Data Comparability and Vintage Structure" subsection | **PENDING** | ~150 lines LaTeX |
| PEP vintage system explanation | **PENDING** | Define "vintage" concept |
| Census Bureau warning block quote | **PENDING** | Cross-vintage combination caveat |
| Methodology differences matrix table | **PENDING** | Vintage 2009 vs 2020 vs 2024 |
| "Spliced multi-instrument series" framing | **PENDING** | Consistent terminology |

**Current State:** ADR-021 focused on policy timeline (useful for different section). The vintage methodology disclosure is distinct and still needed.

#### 3.3 Bayesian VAR / Panel Extensions (B4)

| Task | Status | Notes |
|------|--------|-------|
| Minnesota prior implementation | **PENDING** | PyMC dependency question unresolved |
| Bayesian VAR estimation | **PENDING** | Requires PyMC installation |
| Panel VAR extension | **PENDING** | Leverages B2 50-state panel |
| Model comparison framework | **PENDING** | Classical vs Bayesian |

**Files Exist (but incomplete):**
- `module_B4_bayesian_panel_var.py` - Main script exists but Bayesian portion incomplete

**Blocker:** PyMC dependency decision not made

#### 3.4 Test Infrastructure (B6)

| Task | Status | Notes |
|------|--------|-------|
| `tests/test_statistical/` directory | **PENDING** | No statistical module tests |
| B1 unit tests | **PENDING** | For regime_aware functions |
| B2 unit tests | **PENDING** | For multistate placebo functions |
| B3 validation tests | **PENDING** | LaTeX compilation, numeric claims |
| Shared fixtures (`conftest.py`) | **PENDING** | Sample data for testing |

**ADR-021 modules were created without corresponding test coverage.**

#### 3.5 ADR-020 Decision Record Update (B5)

| Task | Status | Notes |
|------|--------|-------|
| ADR-020 "Accepted" status | **PENDING** | Decision documentation |
| Option C rationale | **PENDING** | Formal acceptance |
| Cross-reference updates | **PENDING** | Link to ADR-021 work |

---

### 4. CONFLICTS / INCONSISTENCIES

#### 4.1 Module Naming Conventions

| ADR-020 Convention | ADR-021 Convention | Conflict |
|--------------------|--------------------|----------|
| `module_B1_regime_aware_models.py` | `module_regime_framework.py` | Two regime modules with different naming |
| `module_B2_multistate_placebo.py` | `module_7b_lssnd_synthetic_control.py` | Different numbering schemes |
| `module_regime_aware/` directory | Standalone `module_regime_framework.py` | Parallel implementations |

**Resolution Recommendation:**
- Keep ADR-021's `module_regime_framework.py` as the authoritative regime definitions
- Rename ADR-020's planned `module_B1_regime_aware_models.py` to integrate with ADR-021 framework
- Use numeric scheme (7b, 8b, 9b, 10) from ADR-021 for consistency

#### 4.2 Regime Boundary Definition

| Source | Expansion End | Restriction End | Notes |
|--------|---------------|-----------------|-------|
| ADR-020 Planning | Implicit (2010-2019 vs 2020+) | Not defined | Focused on vintage transitions |
| ADR-021 Implementation | 2016 | 2020 | Based on Chow tests + policy events |

**Resolution:** Adopt ADR-021's three-regime framework (2010-2016, 2017-2020, 2021-2024) based on structural break analysis.

#### 4.3 Primary Analytical Focus

| ADR-020 Focus | ADR-021 Focus | Tension |
|---------------|---------------|---------|
| Extended time series (2000-2024) | Current status durability | Different temporal emphasis |
| Vintage methodology differences | Policy mechanism effects | Different causal questions |
| Multi-state placebo | ND-specific status composition | Different geographic scope |

**Resolution:** These are complementary, not conflicting. ADR-021 provides the ND-specific policy framework; ADR-020's multi-state analysis validates whether ND's patterns are unique.

---

## Consolidated Task List

### Immediate Actions (Integrate ADR-021 with ADR-020 Phase B)

1. **Update PHASE_METADATA.md** - Mark B1 core tasks as complete via ADR-021
2. **Create ADR cross-reference** - Link ADR-020 to ADR-021 deliverables
3. **Resolve naming convention** - Adopt ADR-021 module numbering

### Remaining ADR-020 Phase B Tasks (Priority Order)

| Priority | Task | Agent | Est. Effort |
|----------|------|-------|-------------|
| **HIGH** | Multi-state placebo analysis | B2 | 4-6 hours |
| **HIGH** | Data Comparability subsection | B3 | 2-3 hours |
| **MEDIUM** | Test infrastructure | B6 | 4-6 hours |
| **MEDIUM** | Extend panel to 2000-2009 | B2 | 2-3 hours |
| **LOW** | Bayesian VAR (optional) | B4 | 6-8 hours |
| **LOW** | ADR-020 formal acceptance | B5 | 1-2 hours |

### Dependencies for Remaining Work

```
Multi-State Placebo (B2)
        ↓
Data Comparability (B3) + Test Infrastructure (B6)
        ↓
    ADR-020 Acceptance (B5)
        ↓
 Bayesian VAR (B4) [Optional]
```

---

## Recommendations

### 1. Adopt ADR-021 as Primary Implementation

ADR-021's completed modules should be the authoritative implementation for:
- Policy regime framework (`module_regime_framework.py`)
- Synthetic control analysis (`module_7b_lssnd_synthetic_control.py`)
- Status durability (`module_8b_status_durability.py`)
- Policy scenarios (`module_9b_policy_scenarios.py`)
- Two-component estimand (`module_10_two_component_estimand.py`)

### 2. Complete ADR-020 Multi-State Analysis

The "50-state placebo" analysis from ADR-020 Agent B2 is still needed and provides unique value:
- Answers: "Is ND's shift unusual in the national distribution?"
- Tests: Oil state hypothesis
- Validates: Whether observed patterns reflect methodology or real drivers

### 3. Merge Documentation Efforts

Combine:
- ADR-021's policy timeline and mechanism diagrams (for policy discussion)
- ADR-020's Data Comparability section (for methodology disclosure)

### 4. Establish Test Coverage for ADR-021 Modules

All ADR-021 modules were created without corresponding tests. Priority:
1. Unit tests for `module_regime_framework.py` functions
2. Integration tests for synthetic control pipeline
3. Validation tests for scenario output consistency

---

## Files Reference

### ADR-021 Completed Modules

| Module | Path |
|--------|------|
| Regime Framework | `/sdc_2024_replication/scripts/statistical_analysis/module_regime_framework.py` |
| LSSND Synthetic Control | `/sdc_2024_replication/scripts/statistical_analysis/module_7b_lssnd_synthetic_control.py` |
| Status Durability | `/sdc_2024_replication/scripts/statistical_analysis/module_8b_status_durability.py` |
| Policy Scenarios | `/sdc_2024_replication/scripts/statistical_analysis/module_9b_policy_scenarios.py` |
| Two-Component Estimand | `/sdc_2024_replication/scripts/statistical_analysis/module_10_two_component_estimand.py` |
| Secondary Migration | `/sdc_2024_replication/scripts/statistical_analysis/module_secondary_migration.py` |

### ADR-021 Results JSON

| Result | Path |
|--------|------|
| Regime Framework | `/docs/governance/adrs/021-reports/results/rec4_regime_framework.json` |
| Synthetic Control | `/docs/governance/adrs/021-reports/results/rec3_lssnd_synthetic_control.json` |
| Secondary Migration | `/docs/governance/adrs/021-reports/results/rec7_secondary_migration.json` |
| Presentation Materials | `/docs/governance/adrs/021-reports/results/rec8_presentation.json` |

### ADR-020 Existing Infrastructure (Pre-ADR-021)

| Component | Path |
|-----------|------|
| Regime-aware submodules | `/sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/` |
| B1 Main Script | `/sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py` |
| B4 Bayesian Script | `/sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel_var.py` |

---

## Conclusion

ADR-021's implementation addressed the core statistical infrastructure needs of ADR-020 Phase B, but with a different focus (status durability and policy mechanisms vs. vintage methodology). The remaining ADR-020 tasks - particularly the multi-state placebo analysis and Data Comparability documentation - provide distinct value and should be completed.

**Recommended Action:** Proceed with ADR-020 Phase B completion, leveraging ADR-021's modules as the foundational regime-aware framework, and focusing remaining effort on:
1. Multi-state placebo analysis (unique to ADR-020)
2. Journal article Data Comparability section (unique to ADR-020)
3. Test coverage for all new modules (gap in both ADRs)

---

*Generated: 2026-01-02*
*Report Author: Claude Code (Opus 4.5)*
