"""
Population Projection Evaluation Framework.

Implements the Evaluation Blueprint with five modules:
  1. Forecast Accuracy (forecast_accuracy.py)
  2. Structural Realism (structural_realism.py)
  3. Sensitivity & Robustness (sensitivity.py)
  4. Benchmark Comparison (benchmark_comparison.py)
  5. Reporting & Visualization (visualization.py)

Supporting layers:
  - data_structures.py: Canonical schemas (dataclasses) for runs, results, diagnostics
  - metrics.py: Core metric computation functions
  - scorecard.py: Multi-axis model scorecard
  - runner.py: Orchestrator that runs all modules
"""

from .data_structures import (
    ComponentRecord,
    DiagnosticRecord,
    ExperimentRegistryEntry,
    ProjectionResultRecord,
    RunIdentity,
)
from .metrics import (
    mae,
    mape,
    mean_signed_error,
    mean_signed_percentage_error,
    median_absolute_percentage_error,
    rmse,
    wape,
)

__all__ = [
    "ComponentRecord",
    "DiagnosticRecord",
    "ExperimentRegistryEntry",
    "ProjectionResultRecord",
    "RunIdentity",
    "mae",
    "mape",
    "mean_signed_error",
    "mean_signed_percentage_error",
    "median_absolute_percentage_error",
    "rmse",
    "wape",
]
