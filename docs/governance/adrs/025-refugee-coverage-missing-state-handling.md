# ADR-025: Post-2020 Refugee Coverage and Missing-State Handling

## Status
Accepted

## Date
2026-01-04

## Context

The RPC/WRAPS PDF exports for FY2021–FY2024 omit some states entirely. These
omissions are not distinguishable from true zeros using available sources.
Downstream analyses now require post-2020 data, so we must handle incomplete
state coverage explicitly to avoid implicit zero-filling or undercounted totals.

### Requirements
- Preserve raw sources and avoid inventing missing state-year values.
- Require post-2020 data in state-panel analyses while accounting for gaps.
- Avoid undercounting national totals in FY2021–FY2024.

### Challenges
- State omissions are not labeled as zero in source PDFs.
- Official national totals exist, but state-level allocations do not.

## Decision

### Decision 1: Treat omitted states as missing/unknown

**Decision**: Do not zero-fill missing state-years in FY2021–FY2024 RPC panels.

**Rationale**:
- Source documents omit states entirely; zero cannot be justified.
- Missingness is ambiguous and should remain explicit for downstream handling.

**Implementation**:
- Leave omitted state-years absent in processed outputs.
- Flag or drop missing states in analyses that require post-2020 coverage.

**Alternatives Considered**:
- Zero-fill omitted states: rejected due to unsupported assumption.
- Impute from external sources: rejected pending vetted data.

### Decision 2: Require post-2020 coverage in state-panel analyses

**Decision**: For analyses that include FY2021+, drop states that lack complete
post-2020 coverage and record the exclusions.

**Rationale**:
- Ensures state-panel analyses do not treat missing as zero.
- Preserves interpretability of post-2020 results.

**Implementation**:
- Synthetic control and Bartik instrument builders drop states missing any
  post-2020 totals and log warnings.

### Decision 3: Use official national totals for FY2021–FY2024

**Decision**: Replace summed state totals with official national totals for
FY2021–FY2024 in national-level series.

**Rationale**:
- Partial state coverage would undercount national totals.

**Implementation**:
- Append official totals from RPC/DHS for FY2021–FY2024 in national series loaders.

## Consequences

### Positive
1. Missing state-years remain explicit and defensible.
2. State-panel analyses do not silently impute zeros post-2020.
3. National totals avoid undercounting for FY2021–FY2024.

### Negative
1. Dropping states reduces donor pools in some analyses.
2. Post-2020 state-level comparability is constrained by coverage gaps.

### Risks and Mitigations

**Risk**: Reduced statistical power in state-panel methods.
- **Mitigation**: Log dropped states and include sensitivity summaries.

## Alternatives Considered

### Alternative 1: Zero-fill omitted states

**Description**: Assume missing states had zero arrivals.

**Pros**:
- Full state panel.

**Cons**:
- Unjustified assumption; likely biased downward.

**Why Rejected**: Source does not indicate zero.

### Alternative 2: External imputation

**Description**: Estimate missing states from third-party data.

**Pros**:
- Potentially closer to true values.

**Cons**:
- Requires new vetted data source and approval.

**Why Rejected**: Not approved yet.

## Implementation Notes

### Key Functions/Classes
- `module_7b_lssnd_synthetic_control.py`: drop states missing post-2020.
- `module_7_causal_inference.py`: drop states missing post-2020 in Bartik.
- `module_8_duration_analysis.py`: drop states missing post-2020 in wave analysis.
- `module_9b_policy_scenarios.py`: official national totals FY2021–FY2024.
- `module_10_two_component_estimand.py`: official national totals FY2021–FY2024.

### Configuration Integration
No changes to `projection_config.yaml`.

### Testing Strategy
- Validate that post-2020 missing states are excluded (not zero-filled) in
  state-panel outputs.
- Verify national totals use official FY2021–FY2024 values.

## References

1. `data/raw/immigration/refugee_arrivals/` (FY2021–FY2024 RPC PDFs)
2. `data/DATA_MANIFEST.md`
3. `data/raw/immigration/refugee_arrivals/MANIFEST.md`

## Revision History

- **2026-01-04**: Initial version (ADR-025) - Missing-state handling and national totals.

## Related ADRs

- ADR-016: Raw Data Management Strategy
- ADR-024: Immigration Data Extension and Fusion Strategy
