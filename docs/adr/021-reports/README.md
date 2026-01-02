# ADR-021 Reports: Immigration Status Durability and Policy-Regime Methodology

## Overview

This investigation evaluates external AI recommendations (ChatGPT 5.2 Pro, informed by Gemini Deep Research) for enhancing the cohort projection methodology to account for immigration status durability, policy regimes, and North Dakota-specific reception capacity effects.

**Core Question**: Do post-2021 changes in immigration composition (parole vs refugee) and ND infrastructure (LSSND closure) require fundamental methodology changes, or can existing modules be adapted?

## Investigation Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | Complete | Intake and triage of external analysis |
| Phase A | In Progress | Exploratory analysis to validate claims |
| Phase B | Not Started | Implementation planning and execution |

## Phase A Validation Questions

The external analysis makes 8 recommendations. Before implementing, we need to validate:

### Q1: Estimand Composition (Agent 1)
**Claim**: Post-2021 ND international migration is increasingly parole-driven, not refugee-driven.
- What is the empirical split between refugee and parole arrivals in ND 2021-2024?
- Can we construct a parole proxy dataset from available sources?
- How does this compare to pre-2021 composition?

### Q2: LSSND Closure Impact (Agent 2)
**Claim**: LSSND closure (2021) created a state-specific capacity shock distinct from federal policy.
- What was ND's refugee arrival trajectory before/after LSSND closure?
- Can we identify suitable donor states for synthetic control?
- Is the effect distinguishable from federal policy (Travel Ban recovery, COVID)?

### Q3: Existing Module Coverage (Agent 3)
**Claim**: Current methodology doesn't handle status durability or policy regimes.
- How do existing modules (esp. Module 8-9) currently handle temporal variation?
- Does the current scenario framework already encode regime-like structures?
- What gaps remain after accounting for existing functionality?

### Q4: Data Availability (Agent 4)
**Claim**: Data exists to extend analysis through FY2024 and add parole tracking.
- What refugee/parole data is available through FY2024?
- What ACS data exists for secondary migration analysis?
- What are the data quality/coverage limitations?

## Agent Coordination

| Agent | Responsibility | Script | Status |
|-------|----------------|--------|--------|
| Agent 1 | Estimand composition analysis | `agent1_estimand_composition.py` | Complete |
| Agent 2 | LSSND impact analysis | `agent2_lssnd_impact.py` | Complete |
| Agent 3 | Existing module gap analysis | `agent3_module_gap_analysis.py` | Complete |
| Agent 4 | Data availability assessment | `agent4_data_availability.py` | Complete |

## Key Findings

### Agent 1: Estimand Composition (VALIDATED with caveats)

- **Data truncation confirmed**: Refugee data ends FY2020, exactly as external analysis claimed
- **2022-2024 surge is dramatic**: 1.9x-3.0x the 2015-2019 baseline
- **Composition is UNKNOWN**: Cannot empirically validate parole vs refugee split without updated data
- **Regime statistics**:
  - Expansion (2010-2016): mean 1,289 intl migrants, 92% refugee share
  - Restriction (2017-2020): mean 1,197 intl migrants, refugee share varies widely
  - Volatility (2021-2024): mean 3,284 intl migrants, 0% known refugee (data gap)

### Agent 2: LSSND Impact (PLAUSIBLE but NOT TESTABLE)

- **LSSND closed 2021**, but refugee data ends FY2020 - cannot measure post-closure impact
- **Synthetic control is FEASIBLE in principle**: 6 suitable donor states identified (SD, NE, ID, ME, VT, NH)
- **ND peaked at 1,164 refugee arrivals in 2014** (pre-Travel Ban era)
- **Data acquisition is the blocker**: Need FY2021-2024 refugee data to implement

### Agent 3: Module Gap Analysis (PARTIAL coverage)

- **Existing capabilities**: Module 8 has wave detection + Cox PH; Module 9 has 4 scenarios + MC simulation
- **Gap summary**: 1/8 fully addressed, 3/8 partially addressed, 4/8 not addressed
- **High priority gaps**:
  1. Two-component estimand (durable vs temporary status)
  2. Mechanism-based scenario framework
- **Build-on-existing opportunity**: Wave registry machinery could be extended for status-specific hazards

### Agent 4: Data Availability (CRITICAL PATH identified)

| Data Need | Status | Feasibility |
|-----------|--------|-------------|
| Refugee FY2021-2024 | **MISSING - CRITICAL** | HIGH (WRAPS public) |
| Parole (OAW, U4U) for ND | MISSING | MEDIUM-LOW |
| ACS secondary migration | Partially available | HIGH |
| Synthetic control donors | Pre-treatment only | HIGH (after refugee update) |

**Critical path**: Refugee arrivals FY2021-2024 blocks all other analyses

## Directory Structure

```
021-reports/
├── README.md                      # This file
├── external_analysis_intake.md    # Phase 0 intake document
├── PLANNING_SYNTHESIS.md          # Phase B planning (created after Phase A)
├── phase_b_plans/                 # Individual agent plans (Phase B)
├── agent1_estimand_composition.py # Phase A script
├── agent2_lssnd_impact.py         # Phase A script
├── agent3_module_gap_analysis.py  # Phase A script
├── agent4_data_availability.py    # Phase A script
├── figures/                       # Generated visualizations
├── data/                          # Intermediate data outputs
└── results/                       # Final analysis results (JSON)
```

## Phase A Conclusions

### External Claims Validation Summary

| Claim | Validation Status | Evidence |
|-------|------------------|----------|
| Estimand composition shift | **PARTIALLY VALIDATED** | 2022-2024 surge confirmed, but composition unknown without data |
| Status durability matters | **PLAUSIBLE** | Cannot test without parole-specific data |
| LSSND capacity shock | **PLAUSIBLE, NOT TESTABLE** | Data ends before treatment year |
| Policy regimes | **VALIDATED** | Clear regime differences in historical data |
| Data truncation | **FULLY VALIDATED** | Refugee data ends FY2020 as claimed |

### Recommended Next Steps

1. **Priority 1: Data Acquisition**
   - Acquire refugee arrivals FY2021-2024 from WRAPS (high feasibility, blocks everything else)
   - Investigate parole proxy options (ND DHS, resettlement agency records)

2. **Priority 2: After Data Acquired**
   - Re-run Agent 1 to validate estimand composition claim
   - Implement LSSND synthetic control (Agent 2 design ready)
   - Extend Module 8 with status-specific hazard rates

3. **Priority 3: Phase B Planning**
   - Create PLANNING_SYNTHESIS.md for implementation
   - Define module specifications for status durability layer
   - Design mechanism-based scenario framework

### Decision Point

**Before proceeding to Phase B, the following must be resolved:**

- [ ] Refugee arrivals FY2021-2024 acquired and integrated
- [ ] Decision on parole data approach (direct acquisition vs proxy)
- [ ] Human review of Phase A findings and approval to proceed

## Related Documents

- [ADR-021](../021-immigration-status-durability-methodology.md) - Parent ADR
- [external_analysis_intake.md](./external_analysis_intake.md) - Full external analysis and triage
- [SOP-001](../../sops/SOP-001-external-ai-analysis-integration.md) - Process guidance

## Output Artifacts

- `results/agent1_results.json` - Estimand composition analysis
- `results/agent2_results.json` - LSSND impact analysis
- `results/agent3_results.json` - Module gap analysis
- `results/agent4_results.json` - Data availability assessment
- `figures/agent1_composition_by_year.png` - Migration composition chart
- `figures/agent1_surge_analysis.png` - 2022-2024 surge analysis
- `figures/agent2_nd_trajectory.png` - ND refugee trajectory
- `figures/agent2_state_comparison.png` - Donor state comparison

---

*Last Updated: 2026-01-01*
*Phase A Status: Complete*
