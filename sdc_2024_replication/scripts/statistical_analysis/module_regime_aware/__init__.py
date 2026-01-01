"""
Regime-Aware Statistical Modeling Module
=========================================

This module provides tools for analyzing time series with known methodology
regime changes (vintage transitions) per ADR-020 Option C (Hybrid Approach).

Components:
- regime_definitions: Vintage boundary constants and regime labels
- vintage_dummies: Create vintage indicator variables
- piecewise_trends: Segment-specific trend estimation
- covid_intervention: 2020 COVID shock modeling
- robust_inference: Heteroskedasticity-robust and regime-specific variance
- sensitivity_suite: Coordinated sensitivity analysis runner

Usage:
    from module_regime_aware import (
        create_vintage_dummies,
        estimate_piecewise_trend,
        create_covid_intervention,
        estimate_regime_variances,
        run_sensitivity_suite,
    )
"""

from .regime_definitions import (
    VINTAGE_2009_YEARS,
    VINTAGE_2020_YEARS,
    VINTAGE_2024_YEARS,
    VINTAGE_BOUNDARIES,
    get_vintage_for_year,
)
from .vintage_dummies import create_vintage_dummies
from .piecewise_trends import estimate_piecewise_trend
from .covid_intervention import (
    create_covid_intervention,
    estimate_covid_effect,
    calculate_counterfactual_2020,
)
from .robust_inference import (
    estimate_regime_variances,
    estimate_with_robust_se,
    estimate_wls_by_regime,
)
from .sensitivity_suite import (
    run_sensitivity_suite,
    SensitivityResult,
    create_sensitivity_table,
)

__all__ = [
    # Regime definitions
    "VINTAGE_2009_YEARS",
    "VINTAGE_2020_YEARS",
    "VINTAGE_2024_YEARS",
    "VINTAGE_BOUNDARIES",
    "get_vintage_for_year",
    # Vintage dummies
    "create_vintage_dummies",
    # Piecewise trends
    "estimate_piecewise_trend",
    # COVID intervention
    "create_covid_intervention",
    "estimate_covid_effect",
    "calculate_counterfactual_2020",
    # Robust inference
    "estimate_regime_variances",
    "estimate_with_robust_se",
    "estimate_wls_by_regime",
    # Sensitivity suite
    "run_sensitivity_suite",
    "SensitivityResult",
    "create_sensitivity_table",
]
