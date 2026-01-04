# ADR-024: Immigration Data Extension and Fusion Strategy

## Status
Proposed

## Date
2026-01-04

## Context

The v0.8.5 critique identified several limitations that can be addressed by extending existing data sources and tightening time-base alignment. The current analysis relies on a truncated refugee arrivals panel (full state×nationality through FY2020; ND-only partial 2021–2024), a limited LPR panel, and a short PEP international migration time series (2010–2024). These constraints limit causal timing analyses, wave duration modeling, and forecasting, especially in the post-2020 regime shift.

The repository already contains raw RPC archives for FY2021–FY2024 (PDFs) and DHS LPR files (2007–2023), but not all are integrated into the processed datasets or models. There is also a combined Census PEP components series (2000–2024) that can support regime-aware modeling if used explicitly.

### Requirements
- Extend refugee arrivals beyond FY2020 using authoritative sources (RPC archives).
- Align fiscal-year data with PEP demographic years using month-level mapping where available.
- Expand LPR panel coverage and integrate into modeling as a covariate.
- Preserve reproducibility with clear raw/processed data manifests and provenance notes.
- Avoid conflating net vs. gross flows and document measurement error and revisions.

### Challenges
- FY2021–FY2024 RPC state×nationality data are in PDFs and require extraction.
- Monthly arrivals are subject to reconciliation; time alignment must be explicit.
- LPR and ACS series are measured differently than PEP net migration.
- Census vintages revise prior years; regime boundaries must be explicit.
- Data fusion risks double counting or overconfident uncertainty intervals.

## Decision

### Decision 1: Extend RPC Refugee Arrivals and Adopt Month-Aware Alignment

**Decision**: Build a complete FY2002+ refugee arrivals panel through FY2024 (and later if available) and map monthly fiscal-year arrivals to PEP demographic years using a Jul–Jun crosswalk when monthly data are available.

**Rationale**:
- Removes the FY2020 truncation and improves post-COVID analysis.
- Corrects timing alignment for policy interventions (e.g., Travel Ban, COVID).
- Uses authoritative RPC sources already staged in raw archives.

**Implementation**:
- Extract FY2021–FY2024 state×nationality arrivals from RPC archive PDFs.
- Store raw PDFs in `data/raw/immigration/refugee_arrivals/` and document in its manifest.
- Produce a monthly-to-PEP-year crosswalk and persist in processed metadata.
- Regenerate `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet`.

**Alternatives Considered**:
- Continue FY-level approximation (0.75/0.25 split): rejected due to avoidable timing error.
- Keep ND-only manual series: rejected due to partial coverage and panel inconsistency.

### Decision 2: Integrate SIV/Amerasian as Parallel Humanitarian Series

**Decision**: Add a parallel humanitarian inflow series (SIV/Amerasian where available) with clear labeling and separation from refugee-only measures.

**Rationale**:
- Post-2021 inflows include meaningful SIV/parole components.
- Improves interpretation of post-2020 regime shifts without redefining the estimand.

**Implementation**:
- Identify authoritative SIV/Amerasian sources and coverage.
- Store series in `data/processed/immigration/analysis/` with clear metadata.
- Use as optional covariate/diagnostic signal in modeling and narrative sections.

**Alternatives Considered**:
- Fold SIV into refugee series: rejected to preserve definitional clarity.
- Ignore SIV/Amerasian: rejected due to post-2021 relevance.

### Decision 3: Expand LPR Multi-Year Panels and Normalize Identifiers

**Decision**: Extend LPR processing to cover the full available DHS/OIS window and produce normalized state×year and origin×year panels for modeling inputs.

**Rationale**:
- Current LPR country/state cross-sections are limited; multi-year panels support dynamic modeling.
- LPR flows offer a distinct administrative signal to complement PEP net migration.

**Implementation**:
- Process raw DHS LPR XLSX files in `data/raw/immigration/dhs_lpr/`.
- Update processed outputs in `data/processed/immigration/analysis/`.
- Normalize state identifiers and document any aggregation assumptions.

**Alternatives Considered**:
- Use FY2023-only cross-section: rejected due to limited inference value.
- Use only national totals: rejected due to loss of state-level signal.

### Decision 4: Use Long-Run PEP Components with Regime Markers

**Decision**: Treat 2000–2024 PEP components as a long-run context series with explicit regime boundaries and vintage sensitivity checks.

**Rationale**:
- Supports regime-aware modeling without redefining the primary inference window.
- Makes vintage changes explicit rather than implicit splicing.

**Implementation**:
- Use `data/processed/immigration/state_migration_components_2000_2024.csv`.
- Add regime flags (pre/post-2010, pre/post-2020) for variance shifts.
- Conduct sensitivity checks across vintages where overlap exists.

**Alternatives Considered**:
- Restrict to 2010–2024 only: rejected due to limited historical context.
- Treat vintages as fully comparable without controls: rejected due to known revisions.

## Consequences

### Positive
1. Extends refugee series and reduces post-2020 blind spots.
2. Improves alignment of policy timing and year definitions.
3. Adds administrative signals (LPR, SIV) to strengthen forecasting inputs.
4. Makes regime shifts and vintage revisions explicit and testable.

### Negative
1. Requires PDF extraction and additional data cleaning steps.
2. Adds modeling complexity and raises interpretability burden.
3. Increased maintenance for raw/processed data manifests.

### Risks and Mitigations

**Risk**: Misinterpreting partial series (ND-only) as national panels.
- **Mitigation**: Add explicit metadata flags and validation checks in processing.

**Risk**: Double counting uncertainty when combining correlated series.
- **Mitigation**: Document variance sources and avoid stacking uncorrelated error terms.

**Risk**: Measurement mismatch across sources (net vs gross).
- **Mitigation**: Maintain estimand clarity and label all signals as covariates.

## Alternatives Considered

### Alternative 1: Keep Existing Series and Only Update Narrative
**Description**: Do not change data inputs; only improve caveats in the manuscript.
**Pros**: Low effort; no pipeline changes.
**Cons**: Leaves core limitations unresolved; weaker reviewer response.
**Why Rejected**: Critique explicitly requests addressing limitations via data.

### Alternative 2: Replace PEP Estimand with Alternative Flow Measures
**Description**: Shift primary estimand to administrative flows (LPR/refugee).
**Pros**: Longer series, clearer administrative counts.
**Cons**: Changes estimand and breaks comparability with PEP.
**Why Rejected**: Out of scope; conflicts with current research framing.

## Implementation Notes

### Key Functions/Classes
- `process_refugee_data.py`: Extend to include FY2021–FY2024 extraction.
- `process_dhs_lpr_data.py`: Expand to full year coverage and normalized outputs.
- `combine_census_vintages.py`: Verify regime markers and vintage sensitivity.

### Configuration Integration
No changes to `config/projection_config.yaml` expected initially; analysis scripts should reference processed datasets via consistent paths.

### Testing Strategy
- Add unit tests for new data ingestion steps (year ranges, missing values).
- Add validation checks for time-base crosswalk integrity.
- Confirm downstream statistical modules read updated inputs without schema drift.

## References
1. RPC Archives: https://www.rpc.state.gov/archives/
2. DHS Yearbook of Immigration Statistics (LPR tables): https://ohss.dhs.gov/topics/immigration/yearbook
3. Census PEP Components: https://www.census.gov/programs-surveys/popest.html
4. v0.8.5 critique intake: `sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md`
5. Progress tracker: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`

## Revision History

- **2026-01-04**: Initial draft (ADR-024) - Proposed data extension and fusion strategy.

## Related ADRs

- ADR-016: Raw Data Management Strategy (data provenance and manifests)
- ADR-018: Immigration Policy Scenario Methodology (policy scenario framing)
- ADR-020: Extended Time Series Methodology Analysis (vintage handling)
- ADR-021: Immigration Status Durability Methodology (status composition modeling)
