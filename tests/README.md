# tests/

Status: current · Last Updated: 2026-06-23

Orientation README for the test suite. The suite is organized so that each `test_*`
subdirectory mirrors one layer of the `cohort_projections/` library (plus an
out-of-package `unit/` tier for the sibling SDC replication repo). Tests run under
`pytest` configured in `pyproject.toml` (`[tool.pytest.ini_options]`), with coverage,
short tracebacks, and `pytest-xdist` parallelism (`-n auto --dist loadfile`) enabled by
default. Shared fixtures live in `conftest.py`; SDC-repo path resolution lives in
`_sdc_paths.py`.

## How the subdirectories map to the system

The eleven test subdirectories line up with the library packages under
`cohort_projections/` and the script/tooling layers:

| Test dir | Covers |
|----------|--------|
| `test_core/` | `cohort_projections/core/` — the cohort-component engine (cohort_component, fertility, mortality, migration, time-varying engine). |
| `test_data/` | `cohort_projections/data/` — base population loading, residual/PEP/BEBR migration rates, convergence interpolation, GQ separation, mortality improvement, housing-unit, multi-county allocation, pipeline orchestrators. The largest tier (~27 modules). |
| `test_geographic/` | `cohort_projections/geographic/` — geography reference loading and multi-geography (county/place) projection wiring. |
| `test_output/` | `cohort_projections/output/` — writers, workbooks, reports, visualizations, geospatial exports. |
| `test_config/` | `cohort_projections/config/` and `config/` YAML — race mappings, projection scenarios, place projection config. |
| `test_utils/` | `cohort_projections/utils/` — config loader, demographic utils, BigQuery client, reproducibility. |
| `test_statistical/` | Statistical/experimental models — Bayesian VAR, covariate anchor, regime-aware, multistate placebo. Has its own `conftest.py`. |
| `test_analysis/` | `cohort_projections/analysis/` — benchmarking, evaluation framework (nested `test_evaluation/`), and the Observatory (catalog, search, recommender, store, dashboard, CLI). The runtime contracts here guard the `scripts/analysis/` CLI runners. |
| `test_integration/` | End-to-end and cross-module pipeline tests (PEP pipeline, place pipeline stage, housing-unit pipeline, geospatial pipeline, export, census method validation). |
| `test_output/` + `test_tools/` | `test_tools/` covers repo tooling such as the citation audit. |
| `unit/` | **Out of package.** Tests for the sibling `sdc_2024_replication` journal-article code (DHS/LPR panels, duration analysis, causal inference modules, SDC data loader, journal versioning). These resolve the external repo via `tests/_sdc_paths.py` / `cohort_projections.utils.sdc_paths` and **skip automatically** when `sdc_2024_replication` is not synced locally. |

## How to run

```bash
pytest                                  # full suite (parallel, with coverage)
pytest tests/unit/                      # one tier only
pytest -m "not slow"                    # skip slow-marked tests
pytest -k "not test_residual_computation_single_period"  # skip the slow Excel read
```

Registered markers (`pyproject.toml`): `slow`, `serial`, `data_fetch`, `integration`.
Most tests gate themselves with `pytest.mark.skipif` on optional dependencies (matplotlib,
openpyxl, lifelines, geopandas/shapely, PyMC) or on the presence of real data files, rather
than being statically marked. The repo also ships `scripts/testing/run_relevant_tests.sh`
(modes: `relevant`, `fast`, `full`) used by pre-commit to run a change-scoped subset.

## Known pre-existing skips

These skips are expected on a healthy checkout and should not be treated as regressions
(baseline ~5 skips on `master`):

- **One slow PyMC test** in `test_statistical/test_bayesian_var.py` (`reason="PyMC MCMC tests are slow…"`).
- **Four upstream-bug skips** in the same file (`reason="Upstream bug in model_comparison.py sigma_u.tolist()"`).
- **`unit/` SDC tests** skip wholesale when `sdc_2024_replication` is not present.
- **Optional-dependency skips** (matplotlib, openpyxl, lifelines, geopandas) fire when those
  extras are not installed; install `--extra dev`/`--extra dashboard` to exercise them.
- `test_data/test_residual_migration.py::test_residual_computation_single_period` reads a large
  Excel file and can time out (>60s); exclude it with the `-k` filter above for quick runs.

For deeper guidance see `docs/guides/test-suite-reference.md`,
`docs/guides/test-maintenance-practices.md`, and `docs/guides/testing-workflow.md`.
