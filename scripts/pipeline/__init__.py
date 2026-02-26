"""
Pipeline orchestration scripts for North Dakota Population Projection System.

This package contains the stage scripts used by
`scripts/pipeline/run_complete_pipeline.sh`:

1. `00_prepare_processed_data.py`
2. `01_process_demographic_data.py`
3. `01a_compute_residual_migration.py`
4. `01b_compute_convergence.py`
5. `01c_compute_mortality_improvement.py`
6. `02_run_projections.py`
7. `03_export_results.py`

Stages can be run independently or orchestrated via the shell runner.
"""

__version__ = "0.1.0"
