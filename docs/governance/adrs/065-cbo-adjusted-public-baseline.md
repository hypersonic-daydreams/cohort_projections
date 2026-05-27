# ADR-065: CBO-Adjusted Public Baseline

## Status
Accepted

## Date
2026-05-27

## Context

ADR-037 created three active scenarios: an unadjusted recent-trend baseline, a
CBO policy-adjusted restricted-growth scenario, and a high-growth sensitivity.
That design was useful during model development because it isolated the effect
of CBO's immigration revision from the unadjusted trend-continuation path.

During PUB-2026 release planning, the public framing was reconsidered. The
public report is intended to present an official central projection, similar in
shape to the 2024 State Data Center release. If CBO's current demographic
outlook is the best available external benchmark for national immigration and
fertility assumptions, then the public baseline should incorporate those
assumptions directly rather than treating them as a restricted-growth side case.

CBO's January 2026 demographic outlook states that its population projections
underlie CBO's baseline budget and economic forecast, reflect laws and policies
in place as of September 30, 2025, and reduce net immigration projections for
2025-2029 because of policy changes and updated data. For a public central
projection, those assumptions are better interpreted as current-policy baseline
inputs than as a low-growth alternative.

### Requirements

- Make `baseline` the CBO-adjusted public projection path.
- Avoid publishing duplicate `baseline` and `restricted_growth` paths with the
  same assumptions.
- Preserve the former unadjusted recent-trend path for internal sensitivity
  analysis and reproducibility.
- Keep the ADR-050 additive migration formula because it fixes signed
  net-migration behavior.
- Update public-release documentation so marketing and data users receive a
  baseline-led, baseline-only package unless a sensitivity appendix is later
  approved.

## Decision

### Decision 1: Baseline Uses CBO-Adjusted Assumptions

**Decision**: Redefine the production `baseline` scenario as the CBO-adjusted
current-policy baseline.

The baseline now uses:

- Fertility: `-5_percent`, grounded in CBO's lower fertility revision.
- Mortality: `improving`, unchanged from the prior baseline.
- Migration: `additive_reduction`, using the ADR-050 CBO schedule and reference
  values:
  - 2025: `0.20`
  - 2026: `0.37`
  - 2027: `0.55`
  - 2028: `0.78`
  - 2029: `0.91`
  - 2030+: `1.00`
  - `reference_intl_migration: 10051`
  - `reference_population: 799358`

**Rationale**:

- CBO's 2026 demographic outlook is a current-policy baseline, not only a
  downside scenario.
- Publishing an unadjusted recent-trend path as "baseline" would overstate the
  centrality of elevated recent immigration after CBO has incorporated newer
  policy and demographic information.
- The ADR-050 additive formula remains the correct implementation because it
  always reduces net migration regardless of whether a county's baseline
  migration rate is positive or negative.

### Decision 2: Public PUB-2026 Release Is Baseline-Only

**Decision**: The PUB-2026 public PDF, public Excel workbook, and public CSV use
the CBO-adjusted `baseline` as the public projection series. `restricted_growth`
and `high_growth` are not public default scenarios for this release.

**Rationale**:

- A single public baseline is consistent with common state projection products
  and the 2024 North Dakota release pattern.
- Once CBO-adjusted assumptions are in the baseline, the former
  `restricted_growth` scenario no longer adds meaningful public information.
- The high-growth path remains useful for internal capacity stress testing but
  has a weaker public-facing interpretation because it combines elevated
  immigration framing with BEBR optimistic historical migration periods.

### Decision 3: Retain Sensitivity Paths Inactive

**Decision**: Retain non-public sensitivity scenarios in configuration:

- `recent_trend_continuation`: former unadjusted baseline, inactive.
- `restricted_growth`: deprecated compatibility alias for the CBO-adjusted
  baseline, inactive.
- `high_growth`: internal elevated-growth sensitivity, inactive.

**Rationale**:

- This preserves reproducibility and allows explicit sensitivity runs without
  making those scenarios part of default public production.
- Keeping `restricted_growth` as an inactive compatibility key avoids breaking
  older scripts or review artifacts that refer to the name.

### Decision 4: Apply Scenario Adjustments Only In The Engine

**Decision**: The projection runner prepares/copies rate tables, but fertility
and migration scenario adjustments are applied inside
`CohortComponentProjection.project_single_year()` after the year-specific rate
table is selected.

**Rationale**:

- Applying adjustments in both the runner and the engine can double-apply CBO
  fertility or migration assumptions.
- Centralizing scenario application in the engine keeps time-varying convergence
  rates and additive CBO reductions in one consistent path.

## Consequences

### Positive

1. The public baseline now reflects the best available CBO current-policy
   immigration and fertility outlook.
2. PUB-2026 public communication becomes simpler: one baseline projection with
   clear caveats, not a three-scenario range.
3. The former unadjusted trend path remains available for internal diagnostics.
4. The projection runner no longer risks double-applying scenario multipliers.

### Negative

1. Results change materially relative to prior March 2026 baseline drafts.
2. Existing draft marketing workbooks, PNGs, and rounded-number documents are
   stale until production is rerun.
3. Older review documents that discuss `restricted_growth <= baseline <=
   high_growth` are historical, not current public-release guidance.

### Risks and Mitigations

**Risk**: Users compare new baseline outputs to old draft baseline values and
interpret the difference as an error.
- **Mitigation**: Label all March 2026 draft numbers as stale after ADR-065 and
  record the final production run metadata in the public workbook and CSV.

**Risk**: Downstream users treat the baseline as guaranteed because alternatives
are less visible.
- **Mitigation**: Keep public text clear that the baseline is a projection based
  on stated assumptions, not a guaranteed outcome.

## Alternatives Considered

### Alternative 1: Keep Baseline Recent-Trend And Publish Restricted Beside It

**Description**: Preserve ADR-042's baseline-plus-restricted public framing.

**Why Rejected**: Once CBO-adjusted assumptions are judged appropriate for the
central path, keeping them as a separate restricted scenario makes the public
baseline less defensible and the report harder to read.

### Alternative 2: Publish Three Scenarios

**Description**: Keep baseline, restricted, and high growth in the public PDF and
downloads.

**Why Rejected**: The public product is meant to be a central official
projection. The high-growth path is still useful internally, but it should not
carry equal public weight in PUB-2026.

## Implementation Notes

### Configuration Integration

- `config/projection_config.yaml`: `baseline` now uses `fertility:
  "-5_percent"` and the `additive_reduction` migration schedule.
- `recent_trend_continuation` preserves the former `baseline` settings as an
  inactive sensitivity.
- `restricted_growth` and `high_growth` are inactive by default.

### Testing Strategy

- Add a production-config test asserting that `baseline` uses the CBO-adjusted
  fertility and additive migration settings.
- Add a production-config test asserting that only `baseline` is active by
  default for PUB-2026 production.
- Update projection-runner tests so scenario preparation passes rates through
  without pre-applying adjustments that the engine will apply.

### Documentation

- [x] Update `docs/methodology.md`.
- [x] Update PUB-2026 public-release handoff docs.
- [x] Add back-references to ADR-037 and ADR-042.

## Implementation Results

Implemented on 2026-05-27.

- `config/projection_config.yaml` now makes `baseline` the only active
  scenario by default and assigns it the CBO-adjusted fertility and additive
  migration settings.
- `recent_trend_continuation` preserves the former unadjusted baseline as an
  inactive internal sensitivity.
- `restricted_growth` is inactive and retained as a deprecated compatibility
  alias; `high_growth` is inactive and internal.
- `scripts/pipeline/02_run_projections.py` now passes fertility and migration
  rates through during scenario preparation so the engine applies scenario
  adjustments once, after selecting the year-specific rate table.
- Focused verification passed:
  - `uv run ruff check scripts/pipeline/02_run_projections.py scripts/exports/_methodology.py scripts/exports/_report_theme.py scripts/exports/build_public_draft_package.py tests/test_config/test_projection_scenarios.py tests/test_integration/test_pep_pipeline.py`
  - `uv run pytest tests/test_config/test_projection_scenarios.py tests/test_integration/test_pep_pipeline.py -q` (`19 passed`)
  - `uv run python scripts/pipeline/02_run_projections.py --all --dry-run`
    resolved the active scenario list to `baseline` only.

## References

1. **ADR-037**: CBO-Grounded Scenario Methodology.
2. **ADR-050**: Restricted Growth Additive Migration Adjustment.
3. **ADR-042**: Baseline Projection Presentation Requirements.
4. **CBO January 2026 Demographic Outlook**: https://www.cbo.gov/publication/61879

## Revision History

- **2026-05-27**: Accepted and implemented for PUB-2026 baseline redefinition.

## Related ADRs

- **ADR-037**: Amended by this ADR; CBO adjustment moves from restricted-growth
  side scenario into the public baseline.
- **ADR-042**: Amended by this ADR; baseline-only public release is permitted
  because the baseline now includes CBO-adjusted assumptions.
- **ADR-050**: Formula reused for the CBO-adjusted baseline.
