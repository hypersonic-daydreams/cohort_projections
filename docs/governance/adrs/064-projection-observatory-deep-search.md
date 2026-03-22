# ADR-064: Projection Observatory Deep Search Workflow

## Status
Accepted

## Date
2026-03-21

## Implemented
2026-03-21

## Context

PP-007 through PP-009 made the Projection Observatory operational, but the
primary unattended workflow still revolved around a preset-first
`search-auto` experience layered on top of lower-level planning and run
commands. The system could execute isolated worktree searches, yet several
operational gaps remained:

- the default dashboard launch flow exposed more launch mechanics than the
  target user experience required,
- CPU allocation logic was duplicated across layers instead of being governed
  by one control-plane policy,
- parallel benchmark launches could collide on `run_id`,
- orchestration defects could be mistaken for analytical model evidence,
- session artifacts emphasized raw tables more than a guided decision surface,
- there was no first-class abstraction for scope-specific search objectives and
  guardrails,
- optional AI synthesis needed a safe boundary so narrative help could exist
  without weakening deterministic review discipline.

The owner goal for this phase was a hands-off process where the main user path
is: open the Observatory, choose CPU cores, click `Begin Deep Search`, then
verify the resulting recommendation logic rather than manually assemble it.

### Requirements

- Make deep search the primary Observatory workflow for unattended variant
  discovery.
- Keep deterministic ranking and decision policy authoritative.
- Preserve all drill-down tables and visuals while making the review path more
  guided.
- Centralize CPU budgeting, batch sizing, stopping logic, and operational
  quarantine behavior.
- Allow advisory AI synthesis only when explicitly enabled and when its claims
  can be checked against deterministic evidence.
- Keep all code-backed experimentation sandbox-only with no automatic
  promotion, alias mutation, or live-checkout edits.

### Challenges

- The new workflow had to remain compatible with the existing CLI, dashboard,
  benchmark history, and experiment-log surfaces.
- Search had to become more adaptive without becoming unconstrained or opaque.
- AI narrative support had to remain useful without turning into an
  ungrounded recommendation engine.
- Documentation had to be updated across guides, config reference, and
  architecture records because this crossed both configuration and algorithmic
  boundaries.

## Decision

### Decision 1: Promote Deep Search to the Canonical Observatory Workflow

**Decision**: Introduce `deep-search` as the canonical end-to-end Observatory
search command and redesign the dashboard home path around a CPU-budget-driven
`Begin Deep Search` action. Keep `search-auto` as a backward-compatible alias
to the same controller.

**Rationale**:
- The target user interaction is guided, not preset-driven.
- `deep-search` describes the actual workflow better than the older
  `search-auto` label.
- Retaining `search-auto` avoids breaking scripts and existing operator habits.

**Implementation**:
- `scripts/analysis/observatory.py` now exposes `deep-search` and routes
  `search-auto` to the same handler.
- The dashboard `Command Center` now uses `CPU cores -> Begin Deep Search` as
  the primary launch path, with expert controls collapsed by default.

**Alternatives Considered**:
- Keep `search-auto` as the only entrypoint and merely rename UI labels:
  rejected because the control-plane changes are substantial enough that the
  canonical language should match the actual architecture.
- Remove `search-auto` immediately: rejected because backward compatibility is
  useful for scripts, operators, and historical documentation.

### Decision 2: Introduce Search Packs and a Shared Deep-Search Policy

**Decision**: Add `DeepSearchPolicy` and `SearchPack` abstractions to govern
CPU allocation, batch sizing, stopping behavior, frontier ordering, seeded
candidate selection, and optional code mutators. Ship `cf001` as the first
pack.

**Rationale**:
- Search packs make scope-specific objectives explicit instead of relying on
  generic “best metric” heuristics.
- The shared CPU allocator prevents dashboard/CLI/controller drift.
- One policy surface makes adaptive stopping and operational limits
  understandable and testable.

**Implementation**:
- `cohort_projections/analysis/observatory/deep_search.py`
- `config/observatory_search_policy.yaml`
- `config/observatory_search_packs/cf001.yaml`

**Alternatives Considered**:
- Keep heuristic ranking in the controller: rejected because it hides search
  intent and makes future scopes harder to package safely.
- Store search logic entirely in code: rejected because pack-level YAML makes
  objective order, seeds, and guardrails auditable without a code edit.

### Decision 3: Treat Orchestration Failures as Operational Blockers, Not Model Evidence

**Decision**: Quarantine incomplete bundles, duplicate `run_id`s, missing
runtime/reproducibility metadata, and other orchestration defects as
`operational_blocker` outcomes with append-only journaling and session-level
quarantine artifacts.

**Rationale**:
- Operational defects and analytical evidence are different kinds of facts and
  should not share one decision path.
- Explicit quarantine artifacts make it easier to audit why a candidate did not
  contribute to the frontier.
- This improves trust in unattended sessions because bundle health is actively
  surfaced, not passively assumed.

**Implementation**:
- Collision-resistant benchmark `run_id` generation
- `search_journal.jsonl`
- session `quarantine/` and `code_candidates/validation_report.json`
- controller stop logic for repeated or excessive operational blockers

**Alternatives Considered**:
- Mark such failures as generic `failed` results: rejected because that blurs
  model quality with controller/runtime faults.
- Drop broken candidates silently: rejected because it destroys reproducibility
  and operator trust.

### Decision 4: Keep AI Synthesis Optional, Advisory, and Claim-Checked

**Decision**: Add a provider-agnostic AI synthesis layer that operates on a
structured evidence payload, but only when explicitly enabled. Every AI summary
is checked against deterministic claims; contradictions suppress the AI output
and fall back to the deterministic Deep Search Brief.

**Rationale**:
- The Observatory benefits from narrative synthesis, tradeoff framing, and
  review questions.
- The decision spine must remain deterministic under SOP-003 governance.
- Claim checking provides a hard boundary between “helpful summary” and
  “untrustworthy invented reasoning.”

**Implementation**:
- `cohort_projections/analysis/observatory/ai_synthesis.py`
- `observatory.ai_synthesis` config block in `config/observatory_config.yaml`
- Deep Search Brief fields for deterministic summary, AI summary, and
  validation metadata

**Alternatives Considered**:
- No AI support at all: rejected because guided narrative support is useful for
  analyst reasoning and was an explicit owner goal.
- Let AI choose the winner: rejected because it conflicts with the project’s
  deterministic review and promotion boundaries.

## Consequences

### Positive
1. The primary user path is simpler and closer to the intended “open and run”
   Observatory experience.
2. Search intent is now explicit through search packs, objective order, and
   controller defaults.
3. Operational faults are visible, quarantined, and excluded from analytical
   conclusions.
4. Session artifacts now support guided review through frontier and brief files
   rather than requiring the user to start from raw tables.
5. Optional AI narrative support is available without weakening deterministic
   decision authority.

### Negative
1. The Observatory now has more configuration surface area:
   `observatory_config.yaml`, `observatory_search_policy.yaml`, and search-pack
   YAMLs all matter.
2. The controller is more complex because it now handles replanning, stop
   policies, journaling, and AI-brief generation.
3. The older “preset-first” dashboard mental model is no longer accurate and
   required documentation cleanup.

### Risks and Mitigations

**Risk**: Search packs drift away from actual controller behavior.
- **Mitigation**: Search packs are validated on load and covered by unit tests.

**Risk**: AI summaries are treated as authoritative.
- **Mitigation**: AI synthesis is disabled by default, claim-checked, and
  explicitly documented as advisory only.

**Risk**: Code-backed deep-search candidates become too permissive.
- **Mitigation**: Patch/file limits, protected paths, allowlisted roots,
  `py_compile`, targeted tests, benchmark preflight, and sandbox-only artifact
  harvesting remain enforced before any candidate contributes evidence.

## Implementation Notes

### Key Functions/Classes
- `resolve_parallelism()`: Shared CPU allocator for CLI, dashboard, and
  controller use.
- `DeepSearchPolicy`: Resolved controller defaults for one session.
- `SearchPack`: Search-domain definition including objectives, seeds, and
  optional code mutators.
- `AutonomousSearchController.run_to_completion()`: Canonical deep-search loop.
- `build_evidence_payload()` / `synthesize_observatory_summary()`: Optional AI
  synthesis interface and deterministic claim checker.

### Configuration Integration

- `config/observatory_config.yaml`
  - `default_cpu_budget`
  - `ai_synthesis`
- `config/observatory_search_policy.yaml`
  - `search_pack_root`
  - `search.deep_search.*`
- `config/observatory_search_packs/cf001.yaml`
  - first enabled pack for the CF-001 search domain

### Testing Strategy

- Unit coverage for:
  - collision-resistant `run_id` generation,
  - shared CPU allocation,
  - search-pack loading and policy defaults,
  - operational-blocker stop logic,
  - brief artifact generation,
  - AI synthesis claim checking.
- Dashboard/CLI coverage for:
  - deep-search parser/handler behavior,
  - session artifact discovery,
  - latest-finished-session brief fallback.

### Documentation
- [x] Update Observatory operator guides and start-here guide.
- [x] Update configuration reference for deep-search policy, search packs, and
  AI synthesis.
- [x] Update `DEVELOPMENT_TRACKER.md`.
- [x] No `docs/methodology.md` update required because this ADR does not change
  projection formulas, rates, data sources, or projection logic.

## References

1. `docs/guides/observatory-start-here.md` -- current Observatory entry point.
2. `docs/guides/observatory-search-loop.md` -- bounded queue operator guide.
3. `docs/guides/observatory-autonomous-search.md` -- deep-search operating
   guide.
4. `docs/guides/configuration-reference.md` -- observatory config and
   deep-search policy reference.
5. [SOP-001](../sops/SOP-001-external-ai-analysis-integration.md) -- governance
   boundary for carrying AI analysis into formal project decisions.
6. [SOP-003](../sops/SOP-003-method-benchmarking-versioning-promotion.md) --
   review/promotion boundary that deep search must not bypass.

## Implementation Results (2026-03-21)

Implemented in the main Observatory stack on 2026-03-21.

### Files Added

- `cohort_projections/analysis/observatory/deep_search.py`
- `cohort_projections/analysis/observatory/ai_synthesis.py`
- `config/observatory_search_packs/cf001.yaml`
- `tests/test_analysis/test_observatory_ai_synthesis.py`

### Major Files Updated

- `scripts/analysis/observatory.py`
- `cohort_projections/analysis/observatory/search_controller.py`
- `cohort_projections/analysis/observatory/dashboard/tab_command_center.py`
- `cohort_projections/analysis/observatory/dashboard/data_manager.py`
- `cohort_projections/analysis/benchmarking.py`
- `config/observatory_config.yaml`
- `config/observatory_search_policy.yaml`

### Validation

- `ruff check ...` across the modified Observatory, benchmarking, and test
  surface passed.
- `pytest tests/test_analysis/test_observatory_search.py tests/test_analysis/test_observatory_dashboard.py tests/test_analysis/test_observatory_cli.py tests/test_analysis/test_observatory_ai_synthesis.py tests/test_analysis/test_benchmarking.py -q`
  -> `168 passed`
- `pytest tests/test_analysis/test_observatory_recommender.py -q`
  -> `44 passed`

### Notable Deviations from Earlier Observatory Behavior

- `deep-search` is now the canonical entrypoint and dashboard language.
- The dashboard no longer uses the older preset-first `Start Exploring` primary
  flow.
- Session artifacts now include a deterministic Deep Search Brief and
  append-only search journal.
- AI synthesis exists only as optional advisory scaffolding; no live provider
  implementation was made mandatory.

## Revision History

- **2026-03-21**: Initial version (ADR-064) documenting the Projection
  Observatory deep-search architecture and operator boundary changes.

## Related ADRs

- [ADR-005](005-configuration-management-strategy.md): Configuration
  Management Strategy
- [ADR-056](056-testing-strategy-maturation.md): Testing Strategy Maturation
- [ADR-061](061-college-fix-model-revision.md): College Fix Model Revision
- [ADR-063](063-evaluation-framework.md): Evaluation Framework Architecture
