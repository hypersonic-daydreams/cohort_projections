"""
Bayesian VAR Estimation
=======================

Implements Bayesian VAR with Minnesota prior using PyMC for MCMC
estimation, with a conjugate prior fallback when PyMC is unavailable.

The Bayesian approach addresses small-n limitations by:
1. Using informative priors to stabilize coefficient estimates
2. Providing full posterior distributions for uncertainty quantification
3. Enabling proper uncertainty propagation in forecasts
"""

from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

from .minnesota_prior import (
    MinnesotaPrior,
    construct_minnesota_prior_from_data,
)

# Check if PyMC is available
try:
    import pymc as pm
    import arviz as az

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


@dataclass
class BayesianVARResult:
    """
    Container for Bayesian VAR estimation results.

    Attributes
    ----------
    method : str
        Estimation method used ('pymc_mcmc' or 'conjugate_analytical')
    coefficients : dict
        Posterior mean coefficients for each equation
    coef_std : dict
        Posterior standard deviations for coefficients
    credible_intervals : dict
        90% credible intervals for coefficients
    posterior_samples : np.ndarray, optional
        MCMC samples (if PyMC used)
    trace : Any, optional
        PyMC trace object (if PyMC used)
    sigma_posterior : np.ndarray
        Posterior mean of error covariance matrix
    prior : MinnesotaPrior
        Prior specification used
    n_obs : int
        Number of observations used
    n_lags : int
        Number of lags in VAR
    variable_names : list[str]
        Names of VAR variables
    diagnostics : dict
        MCMC diagnostics (Rhat, ESS, etc.)
    """

    method: str
    coefficients: dict
    coef_std: dict
    credible_intervals: dict
    posterior_samples: Optional[np.ndarray]
    trace: Optional[Any]
    sigma_posterior: np.ndarray
    prior: MinnesotaPrior
    n_obs: int
    n_lags: int
    variable_names: list[str]
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "coefficients": self.coefficients,
            "coef_std": self.coef_std,
            "credible_intervals": self.credible_intervals,
            "sigma_posterior": self.sigma_posterior.tolist(),
            "prior": self.prior.to_dict(),
            "n_obs": self.n_obs,
            "n_lags": self.n_lags,
            "variable_names": self.variable_names,
            "diagnostics": self.diagnostics,
        }


def prepare_var_data(
    data: pd.DataFrame,
    var_cols: list[str],
    n_lags: int,
    include_constant: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Prepare data matrices for VAR estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    n_lags : int
        Number of lags
    include_constant : bool
        Whether to include constant

    Returns
    -------
    Y : np.ndarray
        Dependent variable matrix (T-p, n_vars)
    X : np.ndarray
        Regressor matrix (T-p, n_vars*n_lags + constant)
    n_obs : int
        Number of observations after lag truncation
    """
    Y_full = data[var_cols].values
    T, n_vars = Y_full.shape

    # Create lagged regressors
    Y = Y_full[n_lags:]  # Dependent variables
    n_obs = len(Y)

    X_parts = []
    if include_constant:
        X_parts.append(np.ones((n_obs, 1)))

    for lag in range(1, n_lags + 1):
        X_parts.append(Y_full[n_lags - lag : T - lag])

    X = np.hstack(X_parts)

    return Y, X, n_obs


def estimate_bayesian_var_pymc(
    data: pd.DataFrame,
    var_cols: list[str],
    prior: MinnesotaPrior,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> BayesianVARResult:
    """
    Estimate Bayesian VAR using PyMC MCMC.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    prior : MinnesotaPrior
        Minnesota prior specification
    n_samples : int
        Number of posterior samples per chain
    n_tune : int
        Number of tuning samples
    n_chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance probability for NUTS
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    BayesianVARResult
        Estimation results with posterior summaries
    """
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is not available. Use estimate_bayesian_var_conjugate instead."
        )

    n_vars = len(var_cols)
    n_lags = prior.n_lags

    # Prepare data
    Y, X, n_obs = prepare_var_data(data, var_cols, n_lags, prior.include_constant)
    n_coef = X.shape[1]

    # Flatten prior mean and std for PyMC
    prior_mean_flat = prior.prior_mean.T.flatten()  # (n_vars * n_coef,)
    prior_std_flat = np.sqrt(prior.prior_var.T.flatten())

    with pm.Model():
        # Coefficients with Minnesota prior
        # A is shape (n_coef, n_vars)
        A_flat = pm.Normal(
            "A_flat",
            mu=prior_mean_flat,
            sigma=prior_std_flat,
            shape=n_coef * n_vars,
        )
        A = pm.Deterministic("A", A_flat.reshape((n_coef, n_vars)))

        # Error covariance - LKJ prior for correlation, HalfNormal for scales
        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov",
            n=n_vars,
            eta=2.0,  # Weakly informative
            sd_dist=pm.HalfNormal.dist(sigma=1000),
            compute_corr=True,
        )
        pm.Deterministic("cov", pm.math.dot(chol, chol.T))

        # Predicted values
        Y_hat = pm.math.dot(X, A)

        # Likelihood
        pm.MvNormal("Y_obs", mu=Y_hat, chol=chol, observed=Y)

        # Sample
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract posterior summaries
    A_samples = trace.posterior["A"].values  # (chains, samples, n_coef, n_vars)
    A_mean = A_samples.mean(axis=(0, 1))
    A_std = A_samples.std(axis=(0, 1))
    A_q05 = np.percentile(A_samples, 5, axis=(0, 1))
    A_q95 = np.percentile(A_samples, 95, axis=(0, 1))

    cov_samples = trace.posterior["cov"].values
    cov_mean = cov_samples.mean(axis=(0, 1))

    # Organize coefficients by equation
    coefficients = {}
    coef_std = {}
    credible_intervals = {}

    for i, var in enumerate(var_cols):
        eq_coefs = {}
        eq_std = {}
        eq_ci = {}

        coef_idx = 0
        if prior.include_constant:
            eq_coefs["const"] = float(A_mean[coef_idx, i])
            eq_std["const"] = float(A_std[coef_idx, i])
            eq_ci["const"] = (float(A_q05[coef_idx, i]), float(A_q95[coef_idx, i]))
            coef_idx += 1

        for lag in range(1, n_lags + 1):
            for j, var_j in enumerate(var_cols):
                name = f"L{lag}.{var_j}"
                eq_coefs[name] = float(A_mean[coef_idx, i])
                eq_std[name] = float(A_std[coef_idx, i])
                eq_ci[name] = (float(A_q05[coef_idx, i]), float(A_q95[coef_idx, i]))
                coef_idx += 1

        coefficients[var] = eq_coefs
        coef_std[var] = eq_std
        credible_intervals[var] = eq_ci

    # MCMC diagnostics
    diagnostics = {
        "n_samples": n_samples,
        "n_tune": n_tune,
        "n_chains": n_chains,
    }

    # Add convergence diagnostics
    summary = az.summary(trace, var_names=["A_flat"])
    diagnostics["rhat_max"] = float(summary["r_hat"].max())
    diagnostics["rhat_mean"] = float(summary["r_hat"].mean())
    diagnostics["ess_bulk_min"] = float(summary["ess_bulk"].min())
    diagnostics["ess_tail_min"] = float(summary["ess_tail"].min())
    diagnostics["converged"] = diagnostics["rhat_max"] < 1.05

    return BayesianVARResult(
        method="pymc_mcmc",
        coefficients=coefficients,
        coef_std=coef_std,
        credible_intervals=credible_intervals,
        posterior_samples=A_samples.reshape(-1, n_coef, n_vars),
        trace=trace,
        sigma_posterior=cov_mean,
        prior=prior,
        n_obs=n_obs,
        n_lags=n_lags,
        variable_names=var_cols,
        diagnostics=diagnostics,
    )


def estimate_bayesian_var_conjugate(
    data: pd.DataFrame,
    var_cols: list[str],
    prior: MinnesotaPrior,
) -> BayesianVARResult:
    """
    Estimate Bayesian VAR using conjugate Normal-Inverse-Wishart prior.

    This is an analytical solution that doesn't require MCMC, useful
    as a fallback when PyMC is unavailable or for quick estimation.

    The posterior is also Normal-Inverse-Wishart:
        A | Sigma, Y ~ N(A_post, Sigma x V_post)
        Sigma | Y ~ IW(S_post, nu_post)

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    prior : MinnesotaPrior
        Minnesota prior specification

    Returns
    -------
    BayesianVARResult
        Estimation results with posterior means and standard deviations
    """
    n_vars = len(var_cols)
    n_lags = prior.n_lags

    # Prepare data
    Y, X, n_obs = prepare_var_data(data, var_cols, n_lags, prior.include_constant)
    n_coef = X.shape[1]

    # Prior parameters
    # Prior mean for coefficients (A_0)
    A_0 = prior.prior_mean  # (n_coef, n_vars)

    # For conjugate prior, we estimate each equation separately
    # This is the equation-by-equation approach with Minnesota prior

    # Prior for Sigma: use scale from AR(1) estimates
    nu_0 = n_vars + 2  # Weakly informative
    S_0 = np.diag(prior.sigma_estimates)  # Prior scale matrix

    # Data sufficient statistics
    XtX = X.T @ X
    XtY = X.T @ Y

    # Estimate each equation with equation-specific prior
    A_post = np.zeros((n_coef, n_vars))
    A_std = np.zeros((n_coef, n_vars))

    for i in range(n_vars):
        # Prior precision for this equation (diagonal)
        prior_var_i = prior.prior_var[:, i]
        V_0_inv_i = np.diag(1.0 / prior_var_i)

        # Posterior precision
        V_post_inv_i = V_0_inv_i + XtX

        try:
            V_post_i = np.linalg.inv(V_post_inv_i)
        except np.linalg.LinAlgError:
            V_post_i = np.linalg.inv(V_post_inv_i + 1e-6 * np.eye(n_coef))

        # Posterior mean
        A_post[:, i] = V_post_i @ (V_0_inv_i @ A_0[:, i] + XtY[:, i])

        # Posterior std (approximate, ignoring Sigma uncertainty)
        # Residual variance estimate
        resid_i = Y[:, i] - X @ A_post[:, i]
        sigma_sq_i = np.sum(resid_i**2) / (n_obs - n_coef)
        A_std[:, i] = np.sqrt(sigma_sq_i * np.diag(V_post_i))

    # Posterior for Sigma
    nu_post = nu_0 + n_obs
    resid = Y - X @ A_post
    S_post = S_0 + resid.T @ resid

    # Posterior mean of Sigma
    Sigma_post_mean = S_post / (nu_post - n_vars - 1)

    # A_std was already computed equation-by-equation above

    # 90% credible intervals (assuming approximate normality)
    z_95 = 1.645  # For 90% CI
    A_q05 = A_post - z_95 * A_std
    A_q95 = A_post + z_95 * A_std

    # Organize coefficients by equation
    coefficients = {}
    coef_std = {}
    credible_intervals = {}

    for i, var in enumerate(var_cols):
        eq_coefs = {}
        eq_std = {}
        eq_ci = {}

        coef_idx = 0
        if prior.include_constant:
            eq_coefs["const"] = float(A_post[coef_idx, i])
            eq_std["const"] = float(A_std[coef_idx, i])
            eq_ci["const"] = (float(A_q05[coef_idx, i]), float(A_q95[coef_idx, i]))
            coef_idx += 1

        for lag in range(1, n_lags + 1):
            for j, var_j in enumerate(var_cols):
                name = f"L{lag}.{var_j}"
                eq_coefs[name] = float(A_post[coef_idx, i])
                eq_std[name] = float(A_std[coef_idx, i])
                eq_ci[name] = (float(A_q05[coef_idx, i]), float(A_q95[coef_idx, i]))
                coef_idx += 1

        coefficients[var] = eq_coefs
        coef_std[var] = eq_std
        credible_intervals[var] = eq_ci

    # Diagnostics
    diagnostics = {
        "method": "conjugate_analytical",
        "posterior_df": nu_post,
        "note": "Analytical posterior; no MCMC convergence diagnostics",
    }

    return BayesianVARResult(
        method="conjugate_analytical",
        coefficients=coefficients,
        coef_std=coef_std,
        credible_intervals=credible_intervals,
        posterior_samples=None,
        trace=None,
        sigma_posterior=Sigma_post_mean,
        prior=prior,
        n_obs=n_obs,
        n_lags=n_lags,
        variable_names=var_cols,
        diagnostics=diagnostics,
    )


def estimate_bayesian_var(
    data: pd.DataFrame,
    var_cols: list[str],
    n_lags: int = 1,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    use_pymc: bool = True,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 2,
    random_seed: int = 42,
) -> BayesianVARResult:
    """
    Main entry point for Bayesian VAR estimation.

    Automatically constructs Minnesota prior from data and estimates
    using PyMC (if available) or conjugate analytical posterior.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    n_lags : int
        Number of lags in VAR
    lambda1 : float
        Overall shrinkage hyperparameter (default 0.1)
    lambda2 : float
        Cross-variable shrinkage (default 0.5)
    lambda3 : float
        Lag decay (default 1.0)
    use_pymc : bool
        Whether to use PyMC for MCMC (falls back to conjugate if unavailable)
    n_samples : int
        Number of MCMC samples (if using PyMC)
    n_tune : int
        Number of tuning samples (if using PyMC)
    n_chains : int
        Number of MCMC chains (if using PyMC)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    BayesianVARResult
        Estimation results
    """
    # Construct prior from data
    prior = construct_minnesota_prior_from_data(
        data=data,
        var_cols=var_cols,
        n_lags=n_lags,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        include_constant=True,
    )

    # Choose estimation method
    if use_pymc and PYMC_AVAILABLE:
        return estimate_bayesian_var_pymc(
            data=data,
            var_cols=var_cols,
            prior=prior,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            random_seed=random_seed,
        )
    else:
        if use_pymc and not PYMC_AVAILABLE:
            import warnings

            warnings.warn(
                "PyMC not available. Using conjugate analytical posterior instead."
            )
        return estimate_bayesian_var_conjugate(
            data=data,
            var_cols=var_cols,
            prior=prior,
        )


def forecast_bayesian_var(
    result: BayesianVARResult,
    data: pd.DataFrame,
    var_cols: list[str],
    h: int = 5,
    n_sim: int = 1000,
) -> dict:
    """
    Generate forecasts with posterior predictive intervals.

    Parameters
    ----------
    result : BayesianVARResult
        Fitted BVAR result
    data : pd.DataFrame
        Original data (for initial conditions)
    var_cols : list[str]
        Column names for VAR variables
    h : int
        Forecast horizon
    n_sim : int
        Number of simulation draws

    Returns
    -------
    dict
        Forecasts with point estimates and credible intervals
    """
    n_vars = len(var_cols)
    n_lags = result.n_lags

    # Get last observations for initial conditions
    Y_init = data[var_cols].values[-n_lags:]

    # Point forecast using posterior mean coefficients
    # Reconstruct coefficient matrix
    n_coef = n_vars * n_lags + 1  # With constant
    A_mean = np.zeros((n_coef, n_vars))

    for i, var in enumerate(var_cols):
        coef_idx = 0
        A_mean[coef_idx, i] = result.coefficients[var]["const"]
        coef_idx += 1
        for lag in range(1, n_lags + 1):
            for j, var_j in enumerate(var_cols):
                A_mean[coef_idx, i] = result.coefficients[var][f"L{lag}.{var_j}"]
                coef_idx += 1

    # Generate forecasts
    forecasts = np.zeros((h, n_vars))
    Y_history = list(Y_init)

    for t in range(h):
        # Build regressor vector
        x = [1.0]  # Constant
        for lag in range(1, n_lags + 1):
            x.extend(Y_history[-lag])
        x = np.array(x)

        # Point forecast
        y_hat = x @ A_mean
        forecasts[t] = y_hat
        Y_history.append(y_hat)

    # Simulation-based credible intervals
    if result.posterior_samples is not None:
        # Use MCMC samples
        n_samples = min(n_sim, result.posterior_samples.shape[0])
        sample_idx = np.random.choice(
            result.posterior_samples.shape[0], size=n_samples, replace=False
        )

        sim_forecasts = np.zeros((n_samples, h, n_vars))

        for s, idx in enumerate(sample_idx):
            A_s = result.posterior_samples[idx]
            Y_hist = list(Y_init)

            for t in range(h):
                x = [1.0]
                for lag in range(1, n_lags + 1):
                    x.extend(Y_hist[-lag])
                x = np.array(x)
                y_hat = x @ A_s

                # Add error term
                eps = np.random.multivariate_normal(
                    np.zeros(n_vars), result.sigma_posterior
                )
                y_sim = y_hat + eps
                sim_forecasts[s, t] = y_sim
                Y_hist.append(y_sim)

        forecast_q05 = np.percentile(sim_forecasts, 5, axis=0)
        forecast_q50 = np.percentile(sim_forecasts, 50, axis=0)
        forecast_q95 = np.percentile(sim_forecasts, 95, axis=0)
    else:
        # Approximate intervals using error covariance
        # This is a rough approximation without full posterior samples
        forecast_std = np.sqrt(np.diag(result.sigma_posterior)) * np.sqrt(
            np.arange(1, h + 1)[:, None]
        )

        forecast_q05 = forecasts - 1.645 * forecast_std
        forecast_q50 = forecasts
        forecast_q95 = forecasts + 1.645 * forecast_std

    # Organize output
    output = {
        "horizon": list(range(1, h + 1)),
        "point_forecast": {},
        "credible_interval_90": {},
    }

    for i, var in enumerate(var_cols):
        output["point_forecast"][var] = forecasts[:, i].tolist()
        output["credible_interval_90"][var] = {
            "lower": forecast_q05[:, i].tolist(),
            "median": forecast_q50[:, i].tolist(),
            "upper": forecast_q95[:, i].tolist(),
        }

    return output
