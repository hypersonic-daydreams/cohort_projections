# ADR-042: Baseline Projection Presentation Requirements

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Presentation framing and mandatory caveats for baseline population projections

**Motivated by**: [Finding 6 investigation](../../reviews/2026-02-18-sanity-check-investigations/finding-6-baseline-1m-plausibility.md) from the 2026-02-18 projection output sanity check

## Context

### Problem: Baseline Projection Risks Being Misinterpreted as a Forecast

The baseline scenario projects North Dakota reaching approximately 1 million residents by 2055, representing a 30-year compound annual growth rate (CAGR) of 0.77%. While conditionally plausible as a trend-continuation exercise, this trajectory carries significant concentration risks and historical-precedent concerns that require explicit framing to prevent misinterpretation.

### Key Risk Factors

1. **International migration dependency**: 91% of recent net migration (2023-2025) is international. Any change in federal immigration policy, refugee resettlement patterns, or global migration flows would fundamentally alter the trajectory. The restricted_growth scenario models one such policy change.

2. **Geographic concentration**: 89% of projected growth is concentrated in just 3 of 53 counties (Cass, Burleigh, Williams). The baseline does not represent broad-based statewide growth.

3. **Historical precedent**: The 0.77%/yr CAGR exceeds every historical decade of North Dakota growth except the 2010-2015 oil boom period. Sustained growth at this rate for 30 years would be unprecedented in state history.

4. **Single-scenario risk**: If the baseline is presented alone, stakeholders may interpret it as the "expected" or "most likely" outcome. Population projections are scenario exercises, not predictions.

### Requirements

- Establish mandatory presentation requirements for baseline projections
- Ensure the restricted_growth scenario is always presented alongside baseline
- Require explicit caveats about migration dependency and geographic concentration
- Prevent misuse of projections as forecasts in policy documents

## Decision

### Decision 1: Baseline Must Be Labeled as "Trend Continuation", Never "Forecast"

**Decision**: All publications, exports, presentations, and data products must label the baseline as a "trend-continuation scenario" or "scenario projection." The terms "forecast," "expected outcome," "prediction," or "most likely" must never be used in connection with any scenario.

**Rationale**:
- Cohort-component projections extend observed demographic rates into the future. They do not model structural breaks, policy changes, or economic shocks.
- The distinction between "projection" and "forecast" is standard practice in demography (Keyfitz 1972, NRC 2000) and is followed by the Census Bureau, UN Population Division, and state demographic centers.

### Decision 2: Baseline Must Always Be Paired with Restricted Growth

**Decision**: The baseline scenario must never be published or presented in isolation. It must always appear alongside the restricted_growth scenario to show the plausible range of outcomes.

**Rationale**:
- The restricted_growth scenario (CBO-grounded policy adjustment) models a plausible downside case where federal immigration enforcement reduces international migration by 60-80% in the near term.
- Together, baseline and restricted_growth bracket the likely range: baseline represents trend continuation and restricted_growth represents policy-adjusted reduction.
- Presenting only the baseline would give a misleadingly narrow view of future population.

**Implementation**:
- Export workbooks already include both active scenarios side-by-side
- Visualization scripts should generate comparison charts by default
- Any single-scenario extract must include a note referencing the companion scenario

### Decision 3: Mandatory Caveats for All Baseline Publications

**Decision**: All publications using baseline projections must include the following caveats, adapted to the medium:

1. **International migration dependency**: "91% of recent net migration to North Dakota is international. The baseline assumes continuation of recent immigration levels; actual migration will depend on federal policy, global conditions, and economic factors."

2. **Geographic concentration**: "89% of projected growth is concentrated in Cass, Burleigh, and Williams counties. Statewide totals do not reflect uniform growth across all regions."

3. **Historical context**: "The baseline growth rate of 0.77%/yr exceeds all historical decades except the 2010-2015 oil boom. Sustained growth at this rate for 30 years would be unprecedented."

4. **Scenario framing**: "This is a trend-continuation scenario, not a forecast. It shows what would happen if recent demographic patterns persist unchanged. See the restricted growth scenario for an alternative trajectory."

**Rationale**:
- These four caveats address the four key risk factors identified in the Finding 6 investigation
- They provide stakeholders with the context needed to interpret the numbers responsibly

## Consequences

### Positive
1. **Prevents misinterpretation**: Stakeholders will understand the conditional nature of projections
2. **Highlights concentration risks**: Policy makers will know that statewide numbers mask county-level variation
3. **Encourages scenario thinking**: Pairing baseline with restricted_growth normalizes range-based planning
4. **Professional credibility**: Aligns with best practices from Census Bureau, NCHS, and peer state demographers

### Negative
1. **Communication complexity**: Presenting two scenarios with caveats is harder than presenting one number
2. **Caveat fatigue**: If caveats are too long or too frequent, stakeholders may stop reading them
3. **No enforcement mechanism**: This ADR establishes requirements but cannot prevent downstream users from stripping caveats when citing numbers

### Risks and Mitigations

**Risk**: Downstream users quote the 1M headline number without caveats
- **Mitigation**: Include caveats directly in data exports (methodology tabs, footnotes in CSV headers) so they travel with the data

**Risk**: Caveats become boilerplate and lose their communicative value
- **Mitigation**: Keep caveats concise and specific (with actual numbers like "91%" and "3 counties") rather than generic disclaimers

## Implementation Notes

### Key Files
- `scripts/exports/_methodology.py`: Shared methodology text used in export workbooks — update to include baseline framing caveats
- `scripts/exports/build_detail_workbooks.py`: Detail workbook builder — verify caveats appear in methodology tab
- `scripts/exports/build_provisional_workbook.py`: Provisional workbook builder — verify caveats appear

### Configuration Integration
No configuration changes are needed. This ADR governs presentation and documentation practices, not projection engine behavior.

### Testing Strategy
1. **Export review**: Verify that methodology text in exported workbooks includes the four mandatory caveats
2. **Visualization review**: Confirm that baseline charts include restricted_growth comparison or reference
3. **Documentation review**: Check that README files and user-facing documentation follow the framing requirements

## References

1. **Finding 6 Investigation Report**: [finding-6-baseline-1m-plausibility.md](../../reviews/2026-02-18-sanity-check-investigations/finding-6-baseline-1m-plausibility.md) — Detailed analysis of baseline growth trajectory plausibility
2. **2026-02-18 Sanity Check**: [Projection Output Sanity Check](../../reviews/2026-02-18-projection-output-sanity-check.md) — Parent review that identified baseline framing as a concern
3. **Census PEP Components (2023-2025)**: Source for the 91% international migration share statistic
4. **Keyfitz, N. (1972)**: "On Future Population" — Foundational work on the distinction between projections and forecasts

## Revision History

- **2026-02-18**: Initial version (ADR-042) — Establish baseline projection presentation requirements

## Related ADRs

- **ADR-037: CBO-Grounded Scenario Methodology** — Defines the restricted_growth scenario that must accompany baseline
- **ADR-039: International-Only Migration Factor** — Documents the domestic/international migration decomposition underlying the 91% international share
- **ADR-038: Multi-Workbook Export Format** — Export format where baseline framing caveats must appear

## Related Reviews

- [Projection Output Sanity Check](../../reviews/2026-02-18-projection-output-sanity-check.md): Parent review identifying the baseline framing concern
- [Finding 6: Baseline 1M Plausibility](../../reviews/2026-02-18-sanity-check-investigations/finding-6-baseline-1m-plausibility.md): Deep investigation of the baseline trajectory
