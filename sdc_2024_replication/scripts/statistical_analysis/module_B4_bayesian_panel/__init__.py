"""
Bayesian and Panel VAR Module
=============================

This module implements Bayesian VAR methods to address small-n limitations
in North Dakota migration analysis, per ADR-020 Phase B4.

Components:
- minnesota_prior: Minnesota/Litterman prior construction for VAR shrinkage
- bayesian_var: BVAR estimation using PyMC (with conjugate fallback)
- panel_var: Panel VAR with entity and time fixed effects
- model_comparison: Compare classical vs Bayesian estimates

Key Motivation:
Short time series (n=15-25) limit the reliability of classical VAR.
Bayesian methods with informative priors can stabilize estimation
and provide better uncertainty quantification.

Usage:
    from module_B4_bayesian_panel import (
        construct_minnesota_prior,
        estimate_bayesian_var,
        estimate_panel_var,
        compare_var_models,
    )
"""

from .minnesota_prior import (
    construct_minnesota_prior,
    MinnesotaPrior,
)
from .bayesian_var import (
    estimate_bayesian_var,
    BayesianVARResult,
    PYMC_AVAILABLE,
)
from .panel_var import (
    estimate_panel_var,
    PanelVARResult,
)
from .model_comparison import (
    compare_var_models,
    ModelComparisonResult,
)

__all__ = [
    # Minnesota prior
    "construct_minnesota_prior",
    "MinnesotaPrior",
    # Bayesian VAR
    "estimate_bayesian_var",
    "BayesianVARResult",
    "PYMC_AVAILABLE",
    # Panel VAR
    "estimate_panel_var",
    "PanelVARResult",
    # Model comparison
    "compare_var_models",
    "ModelComparisonResult",
]
