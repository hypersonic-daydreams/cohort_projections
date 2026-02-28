# ADR-056: Testing Strategy Maturation

## Status
Accepted

## Date
2026-02-28

## Supersedes
[ADR-011](011-testing-strategy.md) (Testing Strategy, 2025-12-18)

## Context

### The Gap Between ADR-011 and Reality

ADR-011 (December 2025) described a "pragmatic, low-coverage" testing approach appropriate for an early-stage research codebase. It framed the testing pyramid with built-in validation at the base, example scripts as integration tests, and "selective" unit tests for critical functions only. It set a coverage target of 50-60% and explicitly marked a comprehensive unit test suite and CI pipeline as "To Be Implemented."

By February 2026, the project had organically outgrown that framing:

- **1,263 tests** across 10 test directories, up from a handful of example scripts.
- **The pyramid inverted**: the project now has more unit tests than integration tests, the opposite of what ADR-011 described.
- **Pre-commit hooks** run ~1,100 tests on every commit touching production code, providing a de facto regression gate.
- **Coverage is highly uneven**: core engine modules (cohort_component, mortality, migration) sit at 87-99%, while data processing modules are at 36-54% and utilities at 18-55%.
- **ADR-011 was never updated** with Implementation Results, leaving a documented strategy that no longer describes the actual practice.

The testing approach that works is not the one ADR-011 described. This ADR supersedes ADR-011 to acknowledge the evolved state and establish forward-looking strategy.

### What Works Well

Three patterns emerged as particularly effective:

1. **Invariant tests** that check relationships (scenario ordering, demographic bounds, state = sum of counties) rather than specific numbers. These survive data vintage changes without maintenance.
2. **Real-data integration tests** (the `test_end_to_end.py` pattern) that run against actual Cass County data. These are the strongest guard against tests that pass on synthetic data but fail on reality.
3. **Pre-commit as regression gate**: running the full suite before every commit catches regressions immediately, without CI infrastructure.

### What Needs Attention

- **GQ separation functions** (ADR-055) have 0% test coverage despite being a critical pipeline component.
- **Pipeline orchestrators** and **base population loaders** have minimal coverage.
- **No coverage ratchet**: coverage can silently regress when new untested code is added.

## Decision

### 1. Supersede ADR-011's "Low-Coverage" Framing

ADR-011's characterization of the project as having "low coverage" with testing as a secondary concern is no longer accurate. The suite of 1,263 tests with pre-commit enforcement represents a mature testing practice. This ADR replaces ADR-011 as the authoritative testing strategy.

### 2. Coverage-Guided Development

New production code in `core/` and `data/process/` must have tests before merge. This is not a numeric coverage target (which encourages gaming) but a practice: run `pytest --cov`, check that new lines are exercised, and write tests for any untested logic paths. The goal is to prevent coverage from regressing, not to chase a number.

### 3. Invariant Tests as the Durable Backbone

Tests that check structural relationships are more valuable than tests that check specific numeric outputs:

- **Scenario ordering**: restricted <= baseline <= high growth, for every county and year.
- **Demographic bounds**: no negative populations, fertility rates within plausible ranges, survival rates between 0 and 1.
- **Aggregation consistency**: state total = sum of county totals (ADR-054).
- **Additive reduction**: restricted scenario subtracts from baseline, never adds (ADR-050).

These tests do not go stale when data vintages change and do not require maintenance when rate assumptions are updated. Prefer writing invariant tests over point-value assertions.

### 4. Real-Data Integration Tests as the Reality Anchor

The `test_end_to_end.py` pattern -- running the projection engine against actual Cass County data and checking that outputs are demographically plausible -- is the strongest guard against a test suite that diverges from reality. Maintain at least one real-data integration test per major pipeline path (projection engine, rate computation, base population loading).

Synthetic-data unit tests verify logic; real-data integration tests verify that the logic produces sensible results on actual inputs. Both are necessary.

### 5. Periodic Coverage Review at Publication Milestones

Aligned with the PP-002 non-regression cadence, perform a coverage review at each publication milestone:

- Check that overall coverage has not regressed from the previous milestone.
- Review the skip/xfail list for tests that should be re-enabled.
- Verify that functions introduced by recent ADRs have test coverage.
- Document any intentional coverage gaps with rationale.

### 6. Address Critical Coverage Gaps

The following areas are identified as highest-priority coverage gaps, in order:

1. **GQ separation functions** (ADR-055): 0% coverage on a component that modifies base populations and migration rate denominators.
2. **Pipeline orchestrators**: the scripts that chain loading, processing, and projection have minimal automated testing.
3. **Base population loaders**: the entry point for all projection data has limited coverage.

These gaps should be addressed incrementally as these modules are next touched, not as a standalone testing sprint.

### 7. CI/CD Remains Deferred

The project is single-developer with effective pre-commit hooks that run the full test suite on every commit. A CI/CD pipeline would add value (branch protection, coverage badges, automated reporting) but is not the highest priority given current constraints. This decision should be revisited if the project gains additional contributors.

## Consequences

### Positive

1. **Documented reality**: the testing strategy now describes what the project actually does, not an aspirational plan from three months ago.
2. **Clear expectations for new code**: coverage-guided development prevents the gap from widening.
3. **Durable tests**: prioritizing invariant tests reduces maintenance burden across data vintage updates.
4. **Targeted gap closure**: identifying specific coverage gaps (GQ, pipeline, loaders) focuses effort where it matters most.

### Negative

1. **No numeric coverage target**: some stakeholders may prefer a concrete threshold. The trade-off is that numeric targets encourage low-value tests that inflate coverage without improving confidence.
2. **No CI/CD**: pre-commit hooks are developer-side only. A force-push or `--no-verify` bypasses all checks. Mitigated by the single-developer context and project norms (CLAUDE.md explicitly prohibits `--no-verify`).
3. **Incremental gap closure is slower**: addressing coverage gaps only when modules are touched means some gaps persist until the next relevant change.

### Risks and Mitigations

**Risk**: Coverage silently regresses as new untested code accumulates.
- **Mitigation**: Coverage review at publication milestones (Decision 5).
- **Mitigation**: Coverage-guided development norm for `core/` and `data/process/` (Decision 2).

**Risk**: Invariant tests pass but mask subtle numeric errors.
- **Mitigation**: Real-data integration tests (Decision 4) catch errors that invariant tests miss.
- **Mitigation**: Manual review of projection outputs at publication milestones (unchanged from ADR-011).

**Risk**: GQ separation remains untested and a regression goes unnoticed.
- **Mitigation**: Identified as priority #1 coverage gap (Decision 6).

## Implementation Notes

Detailed guidance for working with the test suite is maintained in dedicated reference documents:

- **[docs/guides/test-suite-reference.md](../../guides/test-suite-reference.md)**: what each test directory and module covers.
- **[docs/guides/test-maintenance-practices.md](../../guides/test-maintenance-practices.md)**: how to keep tests healthy across data vintage changes.
- **[docs/guides/testing-workflow.md](../../guides/testing-workflow.md)**: how to run tests, interpret coverage, and debug failures.

## Related ADRs

- [ADR-011](011-testing-strategy.md) -- superseded by this ADR.
- [ADR-055](055-group-quarters-separation.md) -- GQ separation, identified as highest-priority coverage gap.
- [ADR-054](054-state-county-aggregation-reconciliation.md) -- state-county aggregation, invariant: state = sum of counties.
- [ADR-050](050-restricted-growth-additive-migration-adjustment.md) -- additive reduction, invariant: restricted <= baseline.
- [ADR-036](036-migration-averaging-methodology.md) -- BEBR averaging, scenario generation tested.
