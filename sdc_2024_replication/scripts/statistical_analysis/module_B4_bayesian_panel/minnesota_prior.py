"""
Minnesota/Litterman Prior Construction
======================================

Implements the Minnesota (Litterman) prior for Bayesian VAR models.
This prior imposes shrinkage toward a random walk, which is sensible
for economic time series and helps stabilize estimation with small samples.

References:
- Litterman, R. (1986). "Forecasting with Bayesian Vector Autoregressions"
- Doan, T., Litterman, R., & Sims, C. (1984). "Forecasting and Conditional
  Projection Using Realistic Prior Distributions"
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MinnesotaPrior:
    """
    Container for Minnesota prior specification.

    Attributes
    ----------
    prior_mean : np.ndarray
        Prior mean for VAR coefficients (shape: n_vars * n_lags, n_vars)
    prior_var : np.ndarray
        Prior variance for VAR coefficients (same shape as prior_mean)
    prior_cov : np.ndarray
        Full prior covariance matrix for vectorized coefficients
    sigma_estimates : np.ndarray
        AR(1) residual variances used in scaling
    hyperparameters : dict
        Dictionary of hyperparameters used
    n_vars : int
        Number of endogenous variables
    n_lags : int
        Number of lags in the VAR
    """

    prior_mean: np.ndarray
    prior_var: np.ndarray
    prior_cov: np.ndarray
    sigma_estimates: np.ndarray
    hyperparameters: dict
    n_vars: int
    n_lags: int
    include_constant: bool = True
    variable_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prior_mean": self.prior_mean.tolist(),
            "prior_var": self.prior_var.tolist(),
            "sigma_estimates": self.sigma_estimates.tolist(),
            "hyperparameters": self.hyperparameters,
            "n_vars": self.n_vars,
            "n_lags": self.n_lags,
            "include_constant": self.include_constant,
            "variable_names": self.variable_names,
        }


def estimate_ar1_variances(data: pd.DataFrame, var_cols: list[str]) -> np.ndarray:
    """
    Estimate residual variances from AR(1) models for each variable.

    These are used to scale the Minnesota prior variances so that
    variables with different scales receive appropriate shrinkage.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for variables to estimate

    Returns
    -------
    np.ndarray
        Array of residual variances from AR(1) models
    """
    sigmas = []
    for col in var_cols:
        y = data[col].values
        # Simple AR(1) regression: y_t = a + b*y_{t-1} + e_t
        y_lag = y[:-1]
        y_curr = y[1:]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta = np.linalg.lstsq(X, y_curr, rcond=None)[0]
            residuals = y_curr - X @ beta
            sigma_sq = np.var(residuals, ddof=2)  # ddof=2 for OLS with constant
        except np.linalg.LinAlgError:
            # Fallback to unconditional variance
            sigma_sq = np.var(y, ddof=1)
        sigmas.append(max(sigma_sq, 1e-8))  # Ensure positive
    return np.array(sigmas)


def construct_minnesota_prior(
    n_vars: int,
    n_lags: int,
    sigma_estimates: np.ndarray,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    lambda4: float = 1e5,
    include_constant: bool = True,
    variable_names: Optional[list[str]] = None,
) -> MinnesotaPrior:
    """
    Construct Minnesota/Litterman prior matrices for BVAR.

    Prior specification for VAR(p) coefficient A_k[i,j]:
        A_k[i,j] ~ N(m_ij, v_ij)

    where:
        m_ii = 1 for k=1, 0 otherwise (random walk prior for own lags)
        m_ij = 0 for i != j

        v_ij = (lambda1 / k^lambda3) * (sigma_i / sigma_j)  if i = j
             = (lambda1 * lambda2 / k^lambda3) * (sigma_i / sigma_j)  if i != j

    Parameters
    ----------
    n_vars : int
        Number of endogenous variables
    n_lags : int
        Number of lags in VAR
    sigma_estimates : np.ndarray
        AR(1) residual standard deviations for scaling (length n_vars)
    lambda1 : float
        Overall tightness (smaller = more shrinkage). Default 0.1.
    lambda2 : float
        Cross-variable tightness (smaller = more shrinkage to zero for
        cross-variable effects). Default 0.5.
    lambda3 : float
        Lag decay (larger = faster decay of variance with lag). Default 1.0.
    lambda4 : float
        Variance for constant term (large = diffuse). Default 1e5.
    include_constant : bool
        Whether to include constant in prior specification
    variable_names : list[str], optional
        Names of variables for documentation

    Returns
    -------
    MinnesotaPrior
        Container with prior mean and variance matrices
    """
    if len(sigma_estimates) != n_vars:
        raise ValueError(
            f"sigma_estimates length ({len(sigma_estimates)}) must match n_vars ({n_vars})"
        )

    # Ensure sigma_estimates are standard deviations (take sqrt if variances passed)
    sigmas = (
        np.sqrt(sigma_estimates) if np.all(sigma_estimates > 1) else sigma_estimates
    )

    # Total number of coefficients per equation
    n_coef = n_vars * n_lags + (1 if include_constant else 0)

    # Initialize prior mean and variance matrices
    # Shape: (n_coef, n_vars) - each column is coefficients for one equation
    prior_mean = np.zeros((n_coef, n_vars))
    prior_var = np.zeros((n_coef, n_vars))

    # Fill in for each equation (i = dependent variable)
    for i in range(n_vars):
        coef_idx = 0

        # Constant term (if included, it's first)
        if include_constant:
            prior_mean[coef_idx, i] = 0.0  # Prior mean for constant
            prior_var[coef_idx, i] = lambda4  # Very diffuse
            coef_idx += 1

        # Lagged variables
        for lag in range(1, n_lags + 1):
            for j in range(n_vars):  # j = explanatory variable
                # Prior mean
                if i == j and lag == 1:
                    # Random walk prior: own first lag has mean 1
                    prior_mean[coef_idx, i] = 1.0
                else:
                    prior_mean[coef_idx, i] = 0.0

                # Prior variance
                # Scale by ratio of residual standard deviations
                scale = sigmas[i] / sigmas[j]

                if i == j:
                    # Own lags: lambda1 / lag^lambda3
                    prior_var[coef_idx, i] = (lambda1 / (lag**lambda3)) ** 2 * scale**2
                else:
                    # Cross lags: lambda1 * lambda2 / lag^lambda3
                    prior_var[coef_idx, i] = (
                        lambda1 * lambda2 / (lag**lambda3)
                    ) ** 2 * scale**2

                coef_idx += 1

    # Build full covariance matrix (diagonal, assuming independence across equations)
    # This is used for vectorized estimation
    prior_cov = np.diag(prior_var.T.flatten())

    return MinnesotaPrior(
        prior_mean=prior_mean,
        prior_var=prior_var,
        prior_cov=prior_cov,
        sigma_estimates=sigma_estimates,
        hyperparameters={
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4,
        },
        n_vars=n_vars,
        n_lags=n_lags,
        include_constant=include_constant,
        variable_names=variable_names or [f"var_{i}" for i in range(n_vars)],
    )


def construct_minnesota_prior_from_data(
    data: pd.DataFrame,
    var_cols: list[str],
    n_lags: int = 1,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    include_constant: bool = True,
) -> MinnesotaPrior:
    """
    Convenience function to construct Minnesota prior directly from data.

    Estimates AR(1) variances from the data and constructs the prior.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    n_lags : int
        Number of lags in VAR
    lambda1, lambda2, lambda3 : float
        Minnesota prior hyperparameters
    include_constant : bool
        Whether to include constant

    Returns
    -------
    MinnesotaPrior
        Prior specification
    """
    sigma_sq = estimate_ar1_variances(data, var_cols)
    return construct_minnesota_prior(
        n_vars=len(var_cols),
        n_lags=n_lags,
        sigma_estimates=sigma_sq,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        include_constant=include_constant,
        variable_names=var_cols,
    )


def summarize_prior(prior: MinnesotaPrior) -> dict:
    """
    Create a summary of the Minnesota prior for reporting.

    Parameters
    ----------
    prior : MinnesotaPrior
        Prior specification

    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        "n_vars": prior.n_vars,
        "n_lags": prior.n_lags,
        "hyperparameters": prior.hyperparameters,
        "variable_names": prior.variable_names,
        "prior_means": {},
        "prior_stds": {},
    }

    for i, var in enumerate(prior.variable_names):
        # Extract first own-lag coefficient
        if prior.include_constant:
            own_lag_idx = 1 + i  # Skip constant, then position of own first lag
        else:
            own_lag_idx = i

        summary["prior_means"][var] = {
            "own_lag1": float(prior.prior_mean[own_lag_idx, i]),
            "other_lag1": 0.0,  # Always 0 for others
        }
        summary["prior_stds"][var] = {
            "own_lag1": float(np.sqrt(prior.prior_var[own_lag_idx, i])),
        }

    # Overall shrinkage intensity
    summary["shrinkage_intensity"] = {
        "overall": prior.hyperparameters["lambda1"],
        "cross_variable": prior.hyperparameters["lambda1"]
        * prior.hyperparameters["lambda2"],
        "interpretation": (
            f"Own-lag coefficients shrunk toward 1 with SD={prior.hyperparameters['lambda1']:.2f}. "
            f"Cross-variable coefficients shrunk toward 0 with SD={prior.hyperparameters['lambda1']*prior.hyperparameters['lambda2']:.3f}."
        ),
    }

    return summary
