# ADR-028: Monte Carlo Simulation Rigor and Parallelization

## Status
Accepted

## Date
2026-01-06

## Context

The v0.8.6 revision cycle relies on Monte Carlo simulation to quantify uncertainty for:
- Module 9 (scenario projection prediction intervals)
- Module 9b (policy-lever scenario uncertainty bounds)

Earlier runs were tuned for limited compute and used low simulation counts (e.g., 1,000 draws), which can yield visibly noisy tail quantiles and unstable interval endpoints. Additionally, the existing implementations were primarily single-process, underutilizing modern multi-core CPUs.

### Requirements
- Increase Monte Carlo draw counts for smoother, more stable prediction intervals.
- Use full CPU parallelism when available.
- Preserve reproducibility (deterministic results for a given seed), independent of the number of worker processes.
- Avoid introducing heavy GPU-only dependencies into the core pipeline.

### Challenges
- Python loops over many draws can be slow if not parallelized.
- Parallelization must avoid nondeterminism from scheduling order.
- Some simulation components (e.g., wave-duration adjustments) must remain compatible with multiprocessing.

## Decision

### Decision 1: Increase Monte Carlo draw counts via explicit CLI controls

**Decision**: Add CLI flags to control simulation counts (and a `--rigorous` preset) rather than hard-coding low defaults.

**Rationale**:
- Makes the rigor level explicit and reproducible.
- Avoids silently changing results for existing workflows unless the user opts in.

**Implementation**:
- Module 9: `--n-draws`, `--seed`, `--n-jobs`, `--chunk-size`, `--rigorous`
- Module 9b: `--n-simulations`, `--seed`, `--n-jobs`, `--chunk-size`, `--rigorous`

**Alternatives Considered**:
- Change defaults globally to a high draw count: rejected because it can unexpectedly increase runtime for quick exploratory runs.
- Store draw count only in config: rejected for now to avoid config churn during v0.8.6 unless needed.

### Decision 2: Deterministic multi-process chunking for parallel execution

**Decision**: Parallelize Monte Carlo by splitting draws into fixed-size chunks and running chunks in `ProcessPoolExecutor`, with deterministic per-chunk seeds derived from a base seed.

**Rationale**:
- Uses multi-core CPUs effectively.
- Deterministic chunk seeds ensure that results do not depend on worker count or scheduling order.

**Implementation**:
- Module 9: `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`
  - Chunked simulation implemented in `_simulate_module9_monte_carlo_chunk()`
- Module 9b: `sdc_2024_replication/scripts/statistical_analysis/module_9b_policy_scenarios.py`
  - Chunked simulation implemented in `_simulate_module9b_policy_chunk()`

**Alternatives Considered**:
- GPU acceleration (CuPy/JAX): rejected as the default path because it adds large dependencies and complicates reproducibility; the simulation state space (≈ 21 projection years) is small enough that CPU parallelization is sufficient.
- Thread-based parallelism: rejected due to the GIL and reliance on Python-level loops.

## Consequences

### Positive
1. More stable tail quantiles and prediction intervals for scenario outputs.
2. Material runtime improvements on multi-core machines.
3. Reproducible results for a given seed, independent of worker count.

### Negative
1. Slightly more complex CLI surface area.
2. Parallel execution increases total CPU utilization; users should be aware when running on shared systems.

### Risks and Mitigations

**Risk**: Differences from prior low-draw outputs may change reported interval endpoints.
- **Mitigation**: Treat this as a rigor improvement; ensure manuscript text and figures are regenerated from the updated outputs and record the draw counts in results metadata.

## Implementation Notes

### Key Functions/Classes
- `module_9_scenario_modeling.monte_carlo_simulation()`: now supports `seed`, `n_jobs`, `chunk_size`
- `module_9b_policy_scenarios.project_scenario()`: now supports `seed`, `n_jobs`, `chunk_size`
- `_simulate_module9_monte_carlo_chunk()`, `_simulate_module9b_policy_chunk()`: multiprocessing-safe chunk executors

### Testing Strategy
- Execute `pytest tests/ -q` after changes.
- Regenerate scenario outputs and verify LaTeX-referenced figures and tables are consistent with updated draw counts.

## Related ADRs
- ADR-018: Immigration policy scenario methodology
- ADR-021: Immigration status durability methodology (wave-duration integration)

## Revision History
- 2026-01-06: Initial version (ADR-028) - Increase Monte Carlo rigor and parallelization.
