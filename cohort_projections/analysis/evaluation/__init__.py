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
    ScorecardEntry,
)
from .benchmark_runners import (
    average_growth,
    build_component_swap,
    carry_forward,
    linear_trend,
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
from .schemas import (
    HorizonBands,
    METRIC_REGISTRY,
    PROJECTION_RESULT_COLUMNS,
)
from .utils import (
    make_diagnostic_record,
    resolve_county_group,
    validate_dataframe,
)

__all__ = [
    "ComponentRecord",
    "DiagnosticRecord",
    "ExperimentRegistryEntry",
    "HorizonBands",
    "METRIC_REGISTRY",
    "PROJECTION_RESULT_COLUMNS",
    "ProjectionResultRecord",
    "RunIdentity",
    "ScorecardEntry",
    "average_growth",
    "build_component_swap",
    "carry_forward",
    "linear_trend",
    "mae",
    "make_diagnostic_record",
    "mape",
    "mean_signed_error",
    "mean_signed_percentage_error",
    "median_absolute_percentage_error",
    "resolve_county_group",
    "rmse",
    "validate_dataframe",
    "wape",
]
