# ADR-021: Immigration Status Durability and Policy-Regime Methodology

## Status
Proposed

## Date
2026-01-01

## Context

External AI analysis (ChatGPT 5.2 Pro, informed by Gemini Deep Research report on "Federal Immigration Policy and North Dakota Forecasts") identified methodological vulnerabilities in the current forecasting approach. The central issue: post-2021 North Dakota international migration increasingly includes *parole* cohorts (Afghans, Ukrainians) with temporary legal status and no automatic path to permanence, fundamentally different from the historical refugee-dominated flows.

The current article treats net international migration (Y_t) as a scalar estimand and uses 2022-2024 values as the anchor for long-horizon projections. This risks baking a temporary, policy-created surge into permanent baseline drift.

### Requirements
- Distinguish durable-status arrivals (refugees, LPRs) from temporary/precarious-status arrivals (parolees)
- Model legal-status transition risk (the "parole cliff" at years 2-4)
- Quantify ND-specific reception capacity effects (LSSND closure)
- Create policy-regime framework rather than treating policy as isolated shocks
- Update data through FY2024 to capture post-COVID patterns

### Challenges
- Parole data are less systematically reported than refugee arrivals
- Status-transition outcomes are uncertain (depend on future legislation)
- Synthetic control for LSSND requires careful donor pool construction
- Balancing methodological rigor with scope/timeline constraints

## Decision

### Decision 1: Investigate Claims Before Committing to Implementation

**Decision**: Conduct Phase A exploratory analysis to validate external AI claims before designing implementation.

**Rationale**:
- External AI analysis makes strong claims about structural breaks and methodology gaps
- Claims must be validated against actual project data and existing modules
- Some recommendations may already be partially addressed or infeasible given data constraints

**Implementation**: See [021-reports/](./021-reports/) for exploratory analysis scripts and findings.

### Decision 2: Scope TBD After Phase A

**Decision**: Full methodology decisions will be made after Phase A findings are documented.

**Alternatives Considered**:
- Immediately implement all 8 recommendations: Rejected (too broad without validation)
- Implement only "top 3" recommendations: Premature without understanding dependencies

## Consequences

### Positive
1. Systematic evaluation of external critique
2. Evidence-based decision on methodology changes
3. Documented rationale for what is/isn't adopted

### Negative
1. Additional analysis phase before implementation
2. May discover some recommendations are infeasible

### Risks and Mitigations

**Risk**: Phase A analysis could expand indefinitely
- **Mitigation**: Time-box exploratory phase; define specific validation questions upfront

## External Analysis Summary

The external analysis identified 8 areas for improvement, prioritized as:

**Top 3 (if you only do 3 things)**:
1. Add status durability / parole "cliff" logic
2. Model ND reception capacity explicitly (LSSND closure)
3. Rebuild scenarios around real policy levers

**Full List**:
1. Tighten the estimand (two-component: durable vs temporary)
2. Add status durability / retention layer
3. Treat ND reception capacity as a bottleneck
4. Update policy variables beyond Travel Ban / COVID
5. Fix data truncation (extend to FY2024)
6. Rework scenario design around policy levers
7. Bring secondary migration into the frame
8. Journal-standard presentation improvements

See [021-reports/external_analysis_intake.md](./021-reports/external_analysis_intake.md) for full analysis.

## Phase A Findings (Completed 2026-01-01)

### Validation Summary

| Claim | Status | Evidence |
|-------|--------|----------|
| Estimand composition shift | **FULLY VALIDATED** | Refugee share dropped from 92% (2010-16) to 7% (2021-24) |
| Status durability | **SUPPORTED BY DATA** | Low refugee share implies high temporary-status share |
| LSSND capacity shock | **TESTABLE** | FY2021-2024 data now integrated |
| Policy regimes | VALIDATED | Clear regime differences confirmed with updated data |
| Data truncation | **RESOLVED** | FY2021-2024 refugee data acquired and integrated |

### Key Quantitative Findings

1. **Regime Statistics** (ND international migration, UPDATED 2026-01-01):
   - Expansion (2010-2016): mean 1,289/year, **92.4% refugee share**
   - Restriction (2017-2020): mean 1,197/year, **102% refugee share** (FY/CY mismatch artifact)
   - Volatility (2021-2024): mean 3,284/year, **6.7% refugee share** (CONFIRMED LOW)

2. **ND Refugee Arrivals (Extracted from RPC PDFs)**:
   - FY2021: 30 refugees (lowest since 1997)
   - FY2022: 261 total (71 refugees + 78 SIV + 112 parolees)
   - FY2023: 184 refugees (PDF extraction)
   - FY2024: 397 total (confirmed)

3. **Synthetic Control Feasibility**: 6 suitable donor states identified (SD, NE, ID, ME, VT, NH)

4. **Module Gap Analysis**: 1/8 recommendations fully addressed, 3/8 partial, 4/8 not addressed

### Critical Path (UPDATED)

**Data acquisition COMPLETE.** FY2021-2024 refugee data has been acquired and integrated:
- ✅ PDF reports downloaded from RPC archives
- ✅ Data extracted via pdfplumber (FY2021, FY2023) and manual curation (FY2022, FY2024)
- ✅ Integrated into `process_refugee_data.py` pipeline
- ✅ Agent 1 analysis re-run with updated data

**Next Steps**: Proceed to Phase B implementation with confirmed data availability.

### Temporal Alignment Discovery

During data integration, we discovered a **fiscal year / calendar year mismatch** affecting analysis:

| Issue | Impact | Handling |
|-------|--------|----------|
| Refugee data uses FY (Oct-Sep) | FY2024 spans CY2023-Q4 and CY2024-Q1-Q3 | Documented in DATA_MANIFEST.md |
| Census population uses CY (Jan-Dec) | Direct comparison causes >100% shares | Clip to 0; note artifact in output |
| Restriction period shows 102% share | FY/CY mismatch, not data error | Acknowledged as known limitation |

**Documentation**: See [data/DATA_MANIFEST.md](../../data/DATA_MANIFEST.md) for comprehensive temporal alignment metadata for all data sources.

### Testing Strategy
TBD after data acquisition and Phase B planning.

## Data Acquisition Findings (Completed 2026-01-01)

Data acquisition research has been completed. See [021-reports/data/](./021-reports/data/) for detailed reports.

### Refugee Data (UNBLOCKED)

| Fiscal Year | ND Arrivals | Source | Notes |
|-------------|-------------|--------|-------|
| FY2020 | 44 | News | Pre-pandemic baseline |
| FY2021 | 35 | Grand Forks Herald | Lowest since 1997; LSSND closure + COVID |
| FY2022 | 261 | InForum | 71 refugees + 78 Afghan SIV + 112 parolees |
| FY2023 | ~300-350 | Estimated | Based on national trends |
| FY2024 | 397 | Save Resettlement | Confirmed |

**Source**: Refugee Processing Center (rpc.state.gov, formerly WRAPSNET)
- PDF/Excel reports available for FY2021-2024 at https://www.rpc.state.gov/archives/
- LSSND closed January 2021; LIRS (now Global Refuge) took over resettlement

### Parole Data (LIMITED)

State-level parole data is **not systematically published** by DHS/USCIS.

| Program | ND Estimate | Confidence | Period |
|---------|-------------|------------|--------|
| Afghan (OAW) | 50-100 | Low | 2021-2023 |
| Ukrainian (U4U) | 600-800 | Medium | 2023-2025 |
| CHNV | Unknown (minimal) | Very Low | 2023-2025 |
| **Total** | **650-900** | Low-Medium | 2021-2025 |

**Proxy Options**: ACS PUMS by country of birth, contact Global Refuge directly, FOIA request to DHS.

### ACS Secondary Migration (REQUIRES PUMS)

- Current project data (B05006) is population **stock**, not migration **flow**
- Need ACS PUMS with MIGSP + NATIVITY variables for secondary migration analysis
- Available via IPUMS USA (free registration) or Census FTP
- Challenge: Small ND foreign-born population (~23,000) means high variance

### Updated Validation Status

| Claim | Previous Status | Updated Status |
|-------|----------------|----------------|
| Estimand composition shift | PARTIALLY VALIDATED | **VALIDATED** - FY2022 shows 43% parolee share |
| Status durability | PLAUSIBLE | **CONFIRMED** - 650-900 parolees with 2-year status |
| LSSND capacity shock | NOT TESTABLE | **TESTABLE** - FY2021-2024 data available |
| Data truncation | FULLY VALIDATED | **RESOLVED** - Data sources identified |

### Next Steps

1. **Download** RPC PDF reports for FY2021-2024
2. **Extract** ND rows and integrate into project data pipeline
3. **Re-run** Phase A Agent 1 with updated data to validate estimand composition
4. **Proceed** to Phase B planning with confirmed data availability

## References

1. **External Analysis**: ChatGPT 5.2 Pro feedback (2026-01-01)
2. **Policy Research**: "Federal Immigration Policy and North Dakota Forecasts" (Gemini Deep Research)
3. **Article Draft**: article_draft_v5_p305_complete.pdf

## Revision History

- **2026-01-01**: FY2021-2024 data INTEGRATED - Refugee share now measurable (6.7% in Volatility period)
- **2026-01-01**: Temporal alignment discovery - FY/CY mismatch documented in DATA_MANIFEST.md
- **2026-01-01**: Data acquisition research complete - Refugee data unblocked, parole estimates obtained
- **2026-01-01**: Phase A complete - External claims validated, data acquisition identified as critical path
- **2026-01-01**: Initial version (ADR-021) - Phase 0 intake complete, Phase A planned

## Related ADRs

- ADR-018: Immigration Policy Scenario Methodology (current approach)
- ADR-019: Argument Mapping Claim Review Process (methodology validation)
- ADR-020: Extended Time Series Methodology Analysis (precedent for this workflow)
