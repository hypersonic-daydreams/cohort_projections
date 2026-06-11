# Compute Runtime Reference: Workload Profile and Machine Selection

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-06-11 |
| **Basis** | Measured runtimes from `data/analysis/benchmark_history/*/execution_log.json` (13 bundles, Mar–May 2026) and output-file timestamps of the 2026-05-27 production run |
| **Occasion** | PUB-2026 finality remediation: assessed whether Stage 1/3 compute justified moving to a more capable desktop. Verdict: no. |

This reference records how long this project's heavy compute steps actually take, where the
parallelism ceiling is, and the decision rule for whether a faster machine helps. Consult it
before moving work between machines for performance reasons, or when a new workload makes the
runtime profile feel out of date.

---

## Measured workload profile

| Workload | Wall time | Basis |
|----------|-----------|-------|
| One walk-forward benchmark bundle (`run_benchmark_suite.py`) | **3.2–5.2 min** (May 2026 batch: ~3.4 min) | 13 bundles, `duration_seconds` in `execution_log.json`, range 191–312 s |
| Full production forward run, baseline scenario (`run_all_projections.py`) | **~12.5 min** (445 output files) | mtime span of `data/projections/baseline/` files from the 2026-05-27 run (14:44:12 → 14:56:36) |
| Naive-method comparison (ADR-063 runners over existing walk-forward results) | seconds–minutes | post-processing only, no projection compute |
| Full test suite (`pytest`) | ~5 min | `test-suite-reference.md` |

Rule of thumb: a multi-variant disposition benchmark (5–7 bundles) is **~25–35 min**; a
sensitivity decomposition of 4–5 one-factor forward runs is **~1 hour**. A "compute-heavy"
stage in plan documents is typically **≤2 hours of actual CPU time** — the multi-day estimates
are orchestration, analysis, and writing.

## Where the parallelism ceiling is

- Walk-forward validation parallelizes over **origin years only**:
  `ORIGIN_YEARS = [2005, 2010, 2015, 2020]` in `scripts/analysis/walk_forward_validation.py`,
  dispatched via `ProcessPoolExecutor` with `max_workers = min(len(ORIGIN_YEARS), cpu_count)`.
  **Maximum 4 worker processes per bundle, regardless of core count.**
- Benchmark sweeps run bundles **sequentially** (confirmed by bundle timestamp spacing in
  `data/analysis/benchmark_history/`).
- Consequence: any machine with ≥4 strong cores saturates a single bundle. More cores only help
  if multiple bundles/forward runs are launched concurrently, or if `ORIGIN_YEARS` grows.

## Machine inventory (as of 2026-06-11)

| | Work laptop (HP ZBook Fury) | Desktop |
|---|---|---|
| CPU | Intel i7-13850HX, 20 cores (8P+12E) / 28 threads | AMD Ryzen 9 7900X, 12 cores / 24 threads, sustains ~5.2 GHz all-core under 70 °C |
| RAM | 64 GB physical; **31 GB visible to WSL2** | 96 GB DDR5 |
| GPU | (integrated/mobile) | NVIDIA RTX 3070 8 GB |
| Storage | NVMe (836 GB free on WSL2 root, 2026-06-11) | NVMe PCIe 4 |

Desktop specs are user-reported; laptop specs measured via `lscpu`/`free` inside WSL2.

## Decision rule

**Stay on the laptop** for the standard workloads above. Measured reasoning (2026-06-11):

- At 4-wide parallelism, the laptop keeps 4 P-cores in their boost envelope without thermal
  throttling — the desktop's sustained-all-core advantage barely engages. Realistic gain is
  ~15–30% per bundle, i.e. **10–20 minutes saved across an entire benchmark stage**.
- RAM is not a constraint: all recorded runs fit comfortably inside the 31 GB WSL2 allocation.
- The GPU is irrelevant: the pipeline is pandas/numpy on CPU with no CUDA path. (PyMC appears
  only in research/test code, not the production path.)
- Switching costs exceed the savings: bisync round trip + git sync + `uv sync` on the target,
  then bisync the artifacts *back* for decisions/QA. Cross-machine artifact drift is a
  documented workspace failure mode (`docs/REFERENCE-cross-machine-db-state.md` at the
  workspace root; 2026-05-12 CSEA case study).

**Move to the desktop** only for sustained-throughput workloads where sequential bundles stack
into hours, e.g.:

- An Observatory deep-search / `search-auto` session exploring dozens of variants overnight.
- Bulk regeneration loops (all scenarios × geospatial × housing-unit, repeatedly).

Even then the win is sustained clock speed, not RAM or GPU. If a switch happens: run
`./scripts/bisync.sh` before and after (workspace rule), and keep each benchmark→decision→QA
chain on one machine.

## Provenance hygiene for benchmark runs

Several March 2026 bundles record `"git_dirty": true` in `execution_log.json`. Run benchmark
bundles and production runs **from a clean commit** so the recorded `git_commit` actually
identifies the code that produced the numbers — release QA (Gate 1) keys on run metadata and
config identity.

## What would invalidate this reference

- `ORIGIN_YEARS` grows beyond ~8, or sweeps gain concurrent bundle execution → core count
  starts mattering; re-measure.
- A workload adds GPU acceleration or genuinely memory-bound computation.
- County/age/race dimensionality expands materially (e.g., 50-state scope).
- Hardware changes on either machine.

If any of these happen, re-measure with the same two probes used here: `duration_seconds`
across `execution_log.json` files, and the mtime span of a production run's output tree.
