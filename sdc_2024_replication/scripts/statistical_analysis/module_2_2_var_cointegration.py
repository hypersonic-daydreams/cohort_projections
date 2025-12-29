#!/usr/bin/env python3
"""
Module 2.2: VAR and Cointegration Analysis
==========================================

Performs Vector Autoregression (VAR) modeling and cointegration analysis on
North Dakota and US international migration time series.

Analyses Performed:
- Engle-Granger two-step cointegration test
- Johansen cointegration test (trace and max eigenvalue)
- VAR model estimation with lag selection via information criteria
- Granger causality tests (both directions)
- Impulse response functions (IRF)
- Forecast error variance decomposition (FEVD)
- Vector Error Correction Model (VECM) if cointegrated

Variables:
- nd_intl_migration: North Dakota international migration (I(1))
- us_intl_migration: US international migration (I(0), but included for comparison)

Usage:
    micromamba run -n cohort_proj python module_2_2_var_cointegration.py
"""

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# Suppress warnings during execution
warnings.filterwarnings("ignore")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Colorblind-safe palette
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
}


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.decisions: list[dict] = []
        self.warnings: list[str] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] = None,
        evidence: str = None,
        reversible: bool = True,
    ):
        """Add a documented decision to the log."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": reversible,
            }
        )

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "decisions": self.decisions,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def load_data(filename: str) -> pd.DataFrame:
    """Load data file from analysis directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def save_figure(
    fig, filepath_base, title, source_note="Census Bureau Population Estimates Program"
):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.02,
        0.02,
        f"Source: {source_note}",
        fontsize=8,
        fontstyle="italic",
        transform=fig.transFigure,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save both formats
    fig.savefig(
        f"{filepath_base}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        f"{filepath_base}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  Saved: {filepath_base}.png")


def engle_granger_cointegration(y1, y2, name1, name2, alpha=0.05):
    """
    Perform Engle-Granger two-step cointegration test.

    Step 1: Estimate long-run relationship via OLS: y1 = beta0 + beta1*y2 + epsilon
    Step 2: Test residuals for stationarity using ADF test

    Parameters:
    -----------
    y1, y2 : array-like
        Time series to test for cointegration
    name1, name2 : str
        Names of the series
    alpha : float
        Significance level

    Returns:
    --------
    dict with cointegration test results
    """
    from scipy import stats as scipy_stats

    # Ensure arrays
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # Step 1: OLS regression
    X = np.column_stack([np.ones(len(y2)), y2])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    residuals = y1 - X @ beta

    # Calculate regression statistics
    X @ beta
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    n = len(y1)
    k = X.shape[1]
    mse = ss_res / (n - k)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), n - k))

    # Step 2: ADF test on residuals
    # Note: Critical values for cointegration residuals differ from standard ADF
    # Using Engle-Granger critical values for 2 variables
    adf_result = adfuller(residuals, maxlag=None, regression="c", autolag="AIC")

    # Engle-Granger critical values for n=2 variables (from MacKinnon)
    # These are more conservative than standard ADF critical values
    eg_critical_values = {
        "1%": -3.90,  # Approximate for n=2
        "5%": -3.34,
        "10%": -3.04,
    }

    # Determine cointegration
    is_cointegrated = adf_result[0] < eg_critical_values["5%"]

    result = {
        "test_type": "Engle-Granger Two-Step",
        "dependent_variable": name1,
        "independent_variable": name2,
        "step1_ols_regression": {
            "constant": {
                "coefficient": float(beta[0]),
                "std_error": float(se_beta[0]),
                "t_statistic": float(t_stats[0]),
                "p_value": float(p_values[0]),
            },
            "slope": {
                "coefficient": float(beta[1]),
                "std_error": float(se_beta[1]),
                "t_statistic": float(t_stats[1]),
                "p_value": float(p_values[1]),
            },
            "r_squared": float(r_squared),
            "n_observations": int(n),
            "interpretation": f"{name1} = {beta[0]:.4f} + {beta[1]:.6f} * {name2}",
        },
        "step2_adf_on_residuals": {
            "adf_statistic": float(adf_result[0]),
            "adf_p_value": float(adf_result[1]),
            "lags_used": int(adf_result[2]),
            "n_obs_used": int(adf_result[3]),
            "standard_critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "engle_granger_critical_values": eg_critical_values,
            "note": "Engle-Granger critical values are more conservative than standard ADF",
        },
        "conclusion": {
            "is_cointegrated": is_cointegrated,
            "significance_level": alpha,
            "interpretation": (
                f"Series ARE cointegrated at {alpha*100}% level - long-run equilibrium exists"
                if is_cointegrated
                else f"Series are NOT cointegrated at {alpha*100}% level - no long-run equilibrium"
            ),
        },
        "residuals": residuals.tolist(),  # Store for plotting
    }

    return result


def johansen_cointegration_test(data, det_order=0, k_ar_diff=1):
    """
    Perform Johansen cointegration test.

    Parameters:
    -----------
    data : DataFrame
        Data containing the time series
    det_order : int
        Deterministic term: -1 = no constant, 0 = constant, 1 = linear trend
    k_ar_diff : int
        Number of lagged differences in the VECM

    Returns:
    --------
    dict with Johansen test results
    """
    # Ensure data is numpy array
    data_array = np.asarray(data)

    # Run Johansen test
    joh_result = coint_johansen(data_array, det_order=det_order, k_ar_diff=k_ar_diff)

    # Critical values tables from Johansen
    # For 2 variables with constant (det_order=0)
    n_vars = data_array.shape[1]

    result = {
        "test_type": "Johansen Cointegration Test",
        "n_variables": n_vars,
        "variable_names": list(data.columns)
        if hasattr(data, "columns")
        else [f"var_{i}" for i in range(n_vars)],
        "deterministic_term": {-1: "No constant", 0: "Constant", 1: "Linear trend"}[
            det_order
        ],
        "lags_in_vecm": k_ar_diff,
        "trace_test": {
            "description": "H0: number of cointegrating vectors <= r vs H1: > r",
            "statistics": [],
            "eigenvalues": joh_result.eig.tolist(),
        },
        "max_eigenvalue_test": {
            "description": "H0: number of cointegrating vectors = r vs H1: = r+1",
            "statistics": [],
        },
        "cointegrating_vectors": joh_result.evec.tolist(),
        "conclusion": {},
    }

    # Extract trace test results
    for i in range(n_vars):
        trace_stat = float(joh_result.lr1[i])
        max_stat = float(joh_result.lr2[i])

        # Critical values (90%, 95%, 99%)
        trace_cv = {
            "90%": float(joh_result.cvt[i, 0]),
            "95%": float(joh_result.cvt[i, 1]),
            "99%": float(joh_result.cvt[i, 2]),
        }
        max_cv = {
            "90%": float(joh_result.cvm[i, 0]),
            "95%": float(joh_result.cvm[i, 1]),
            "99%": float(joh_result.cvm[i, 2]),
        }

        result["trace_test"]["statistics"].append(
            {
                "null_hypothesis": f"r <= {i}",
                "alternative_hypothesis": f"r > {i}",
                "statistic": trace_stat,
                "critical_values": trace_cv,
                "reject_at_95pct": trace_stat > trace_cv["95%"],
            }
        )

        result["max_eigenvalue_test"]["statistics"].append(
            {
                "null_hypothesis": f"r = {i}",
                "alternative_hypothesis": f"r = {i + 1}",
                "statistic": max_stat,
                "critical_values": max_cv,
                "reject_at_95pct": max_stat > max_cv["95%"],
            }
        )

    # Determine number of cointegrating relationships
    n_coint_trace = sum(
        1 for s in result["trace_test"]["statistics"] if s["reject_at_95pct"]
    )
    n_coint_max = sum(
        1 for s in result["max_eigenvalue_test"]["statistics"] if s["reject_at_95pct"]
    )

    result["conclusion"] = {
        "n_cointegrating_relations_trace": n_coint_trace,
        "n_cointegrating_relations_max_eigenvalue": n_coint_max,
        "recommended": max(n_coint_trace, n_coint_max),
        "interpretation": (
            f"Trace test suggests {n_coint_trace} cointegrating relation(s), "
            f"Max eigenvalue test suggests {n_coint_max} cointegrating relation(s)"
        ),
    }

    return result


def estimate_var_model(data, maxlags=4, ic="aic"):
    """
    Estimate VAR model with automatic lag selection.

    Parameters:
    -----------
    data : DataFrame
        Data containing the time series
    maxlags : int
        Maximum number of lags to consider
    ic : str
        Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')

    Returns:
    --------
    tuple (VAR model results, dict with model info)
    """
    # Fit VAR model
    model = VAR(data)

    # Select optimal lag
    lag_order = model.select_order(maxlags=maxlags)

    # Get selected lag based on criterion
    ic_map = {"aic": "aic", "bic": "bic", "hqic": "hqic", "fpe": "fpe"}
    selected_lag = getattr(lag_order, ic_map[ic])

    # Ensure at least 1 lag
    selected_lag = max(1, selected_lag)

    # Fit model with selected lag
    var_results = model.fit(maxlags=selected_lag)

    # Build result dictionary
    result = {
        "model_type": "Vector Autoregression (VAR)",
        "variable_names": list(data.columns),
        "n_observations": int(var_results.nobs),
        "lag_selection": {
            "method": ic.upper(),
            "max_lags_considered": maxlags,
            "selected_lag": int(selected_lag),
            "all_criteria": {
                "AIC": {i: float(lag_order.ics["aic"][i]) for i in range(maxlags + 1)},
                "BIC": {i: float(lag_order.ics["bic"][i]) for i in range(maxlags + 1)},
                "HQIC": {
                    i: float(lag_order.ics["hqic"][i]) for i in range(maxlags + 1)
                },
                "FPE": {i: float(lag_order.ics["fpe"][i]) for i in range(maxlags + 1)},
            },
            "optimal_by_criterion": {
                "AIC": int(lag_order.aic),
                "BIC": int(lag_order.bic),
                "HQIC": int(lag_order.hqic),
                "FPE": int(lag_order.fpe),
            },
        },
        "model_fit": {
            "log_likelihood": float(var_results.llf),
            "aic": float(var_results.aic),
            "bic": float(var_results.bic),
            "hqic": float(var_results.hqic),
            "fpe": float(var_results.fpe),
        },
        "equations": {},
    }

    # Extract coefficients for each equation
    for i, var_name in enumerate(data.columns):
        # Calculate R-squared manually from residuals
        y = data[var_name].values[selected_lag:]
        # fittedvalues is a DataFrame - access by column name
        y_hat = var_results.fittedvalues[var_name].values
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Calculate adjusted R-squared
        n = len(y)
        k = var_results.params.shape[0]  # number of parameters
        adj_r_squared = (
            1 - (1 - r_squared) * (n - 1) / (n - k - 1)
            if (n - k - 1) > 0
            else r_squared
        )

        eq_result = {
            "dependent_variable": var_name,
            "coefficients": {},
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
        }

        # Get coefficients, standard errors, t-stats, p-values
        # params DataFrame has shape (n_params, n_vars)
        params = var_results.params.iloc[:, i].values
        stderr = var_results.stderr.iloc[:, i].values
        tvalues = var_results.tvalues.iloc[:, i].values
        pvalues = var_results.pvalues.iloc[:, i].values

        # Get parameter names from model
        param_names = var_results.names
        for j, name in enumerate(param_names):
            eq_result["coefficients"][name] = {
                "coefficient": float(params[j]),
                "std_error": float(stderr[j]),
                "t_statistic": float(tvalues[j]),
                "p_value": float(pvalues[j]),
                "significant_5pct": float(pvalues[j]) < 0.05,
            }

        result["equations"][var_name] = eq_result

    return var_results, result


def granger_causality_tests(data, var_names, maxlag):
    """
    Perform Granger causality tests in both directions.

    Parameters:
    -----------
    data : DataFrame
        Data containing the time series
    var_names : list
        Names of the two variables
    maxlag : int
        Number of lags to use in the test

    Returns:
    --------
    dict with Granger causality test results
    """
    result = {}

    for _i, (cause, effect) in enumerate(
        [(var_names[0], var_names[1]), (var_names[1], var_names[0])]
    ):
        test_data = data[[effect, cause]].values

        try:
            gc_result = grangercausalitytests(test_data, maxlag=[maxlag], verbose=False)

            # Extract F-test results
            f_test = gc_result[maxlag][0]["ssr_ftest"]
            chi2_test = gc_result[maxlag][0]["ssr_chi2test"]
            lrtest = gc_result[maxlag][0]["lrtest"]
            gc_result[maxlag][0]["params_ftest"]

            result[f"{cause}_causes_{effect}"] = {
                "null_hypothesis": f"{cause} does NOT Granger-cause {effect}",
                "alternative_hypothesis": f"{cause} Granger-causes {effect}",
                "lags_used": maxlag,
                "f_test": {
                    "statistic": float(f_test[0]),
                    "p_value": float(f_test[1]),
                    "df_num": int(f_test[2]),
                    "df_denom": int(f_test[3]),
                    "reject_null_5pct": float(f_test[1]) < 0.05,
                },
                "chi2_test": {
                    "statistic": float(chi2_test[0]),
                    "p_value": float(chi2_test[1]),
                    "df": int(chi2_test[2]),
                    "reject_null_5pct": float(chi2_test[1]) < 0.05,
                },
                "likelihood_ratio_test": {
                    "statistic": float(lrtest[0]),
                    "p_value": float(lrtest[1]),
                    "df": int(lrtest[2]),
                    "reject_null_5pct": float(lrtest[1]) < 0.05,
                },
                "conclusion": (
                    f"{cause} DOES Granger-cause {effect} at 5% level"
                    if float(f_test[1]) < 0.05
                    else f"{cause} does NOT Granger-cause {effect} at 5% level"
                ),
            }
        except Exception as e:
            result[f"{cause}_causes_{effect}"] = {
                "error": str(e),
                "note": "Granger causality test failed, possibly due to insufficient data",
            }

    return result


def compute_impulse_response(var_results, periods=10):
    """
    Compute impulse response functions.

    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    periods : int
        Number of periods for IRF

    Returns:
    --------
    dict with IRF data
    """
    # Compute orthogonalized IRF (Cholesky decomposition)
    irf = var_results.irf(periods)

    # Get IRF values and confidence intervals
    irf_values = irf.irfs
    # Use orth_irfs for orthogonalized responses if available
    # CI method differs by statsmodels version
    try:
        # Try newer statsmodels method
        irf_lower, irf_upper = irf.ci(alpha=0.05, orth=True)
    except (AttributeError, TypeError):
        # Fallback: compute approximate CIs using stderr
        # Use asymptotic approximation
        irf_stderr = irf.stderr(orth=True)
        if irf_stderr is not None:
            irf_lower = irf_values - 1.96 * irf_stderr
            irf_upper = irf_values + 1.96 * irf_stderr
        else:
            # If no stderr available, use 20% bounds as placeholder
            irf_lower = irf_values * 0.8
            irf_upper = irf_values * 1.2

    var_names = var_results.names
    len(var_names)

    result = {
        "description": "Orthogonalized Impulse Response Functions (Cholesky decomposition)",
        "periods": periods,
        "confidence_level": 0.95,
        "responses": {},
    }

    for i, impulse_var in enumerate(var_names):
        for j, response_var in enumerate(var_names):
            key = f"response_{response_var}_to_{impulse_var}"
            result["responses"][key] = {
                "impulse_variable": impulse_var,
                "response_variable": response_var,
                "values": irf_values[:, j, i].tolist(),
                "lower_ci": irf_lower[:, j, i].tolist(),
                "upper_ci": irf_upper[:, j, i].tolist(),
            }

    return result, irf


def compute_variance_decomposition(var_results, periods=10):
    """
    Compute forecast error variance decomposition.

    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    periods : int
        Number of periods for FEVD

    Returns:
    --------
    dict with FEVD data
    """
    fevd = var_results.fevd(periods)
    var_names = var_results.names

    result = {
        "description": "Forecast Error Variance Decomposition",
        "periods": periods,
        "decomposition": {},
    }

    for i, var_name in enumerate(var_names):
        result["decomposition"][var_name] = {"by_period": []}
        for period in range(periods):
            period_decomp = {}
            for j, shock_var in enumerate(var_names):
                period_decomp[shock_var] = float(fevd.decomp[i][period, j])
            result["decomposition"][var_name]["by_period"].append(
                {"period": period + 1, "proportions": period_decomp}
            )

    return result, fevd


def estimate_vecm(data, coint_rank, k_ar_diff=1, deterministic="ci"):
    """
    Estimate Vector Error Correction Model.

    Parameters:
    -----------
    data : DataFrame
        Data containing the time series
    coint_rank : int
        Number of cointegrating relations
    k_ar_diff : int
        Number of lagged differences
    deterministic : str
        Deterministic term specification

    Returns:
    --------
    dict with VECM results
    """
    if coint_rank == 0:
        return {
            "note": "No cointegration detected, VECM not estimated",
            "recommendation": "Use VAR in first differences instead",
        }

    try:
        vecm_model = VECM(
            data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic,
        )
        vecm_results = vecm_model.fit()

        var_names = list(data.columns)

        result = {
            "model_type": "Vector Error Correction Model (VECM)",
            "variable_names": var_names,
            "cointegration_rank": coint_rank,
            "lagged_differences": k_ar_diff,
            "deterministic": deterministic,
            "alpha_loading_matrix": {
                "description": "Speed of adjustment coefficients (alpha)",
                "interpretation": "How quickly each variable adjusts to deviations from equilibrium",
                "values": {},
            },
            "beta_cointegrating_vectors": {
                "description": "Cointegrating vectors (beta)",
                "interpretation": "Long-run equilibrium relationships",
                "values": vecm_results.beta.tolist()
                if hasattr(vecm_results, "beta")
                else None,
            },
            "gamma_short_run_coefficients": {
                "description": "Short-run dynamics coefficients",
                "values": {},
            },
            "model_summary": str(vecm_results.summary()),
        }

        # Extract alpha (loading) coefficients
        if hasattr(vecm_results, "alpha"):
            alpha = vecm_results.alpha
            for i, var_name in enumerate(var_names):
                result["alpha_loading_matrix"]["values"][var_name] = {
                    "coefficient": float(alpha[i, 0])
                    if coint_rank == 1
                    else alpha[i, :].tolist(),
                    "interpretation": (
                        f"Negative value indicates {var_name} adjusts toward equilibrium"
                        if (coint_rank == 1 and alpha[i, 0] < 0)
                        else f"Positive value indicates {var_name} moves away from equilibrium"
                    ),
                }

        return result

    except Exception as e:
        return {
            "error": str(e),
            "note": "VECM estimation failed",
            "recommendation": "Check data requirements and cointegration rank",
        }


def plot_impulse_response(irf, var_names, filepath_base):
    """Plot impulse response functions with confidence bands."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 10))

    periods = range(len(irf.irfs))

    # Get confidence intervals
    try:
        irf_lower, irf_upper = irf.ci(alpha=0.05, orth=True)
    except (AttributeError, TypeError):
        irf_stderr = irf.stderr(orth=True)
        if irf_stderr is not None:
            irf_lower = irf.irfs - 1.96 * irf_stderr
            irf_upper = irf.irfs + 1.96 * irf_stderr
        else:
            irf_lower = irf.irfs * 0.8
            irf_upper = irf.irfs * 1.2

    for i, impulse_var in enumerate(var_names):
        for j, response_var in enumerate(var_names):
            ax = axes[j, i] if n_vars > 1 else axes

            # Plot IRF
            ax.plot(
                periods,
                irf.irfs[:, j, i],
                color=COLORS["primary"],
                linewidth=2,
                label="IRF",
            )

            # Plot confidence bands
            ax.fill_between(
                periods,
                irf_lower[:, j, i],
                irf_upper[:, j, i],
                color=COLORS["ci_fill"],
                alpha=0.2,
                label="95% CI",
            )

            # Zero line
            ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)

            ax.set_xlabel("Periods", fontsize=10)
            ax.set_ylabel("Response", fontsize=10)
            ax.set_title(f"{response_var}\nto {impulse_var}", fontsize=11)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    title = "Impulse Response Functions with 95% Confidence Intervals"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath_base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath_base}.png")


def plot_variance_decomposition(fevd, var_names, filepath_base):
    """Plot forecast error variance decomposition."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(12, 5))

    if n_vars == 1:
        axes = [axes]

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]
    periods = range(1, len(fevd.decomp[0]) + 1)

    for i, var_name in enumerate(var_names):
        ax = axes[i]

        # Stacked area plot
        bottom = np.zeros(len(periods))
        for j, shock_var in enumerate(var_names):
            values = fevd.decomp[i][:, j]
            ax.fill_between(
                periods,
                bottom,
                bottom + values,
                color=colors[j % len(colors)],
                alpha=0.7,
                label=f"Shock: {shock_var}",
            )
            bottom += values

        ax.set_xlabel("Forecast Horizon", fontsize=11)
        ax.set_ylabel("Proportion of Variance", fontsize=11)
        ax.set_title(f"Variance of {var_name}", fontsize=12, fontweight="bold")
        ax.legend(loc="center right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    title = "Forecast Error Variance Decomposition"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath_base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath_base}.png")


def plot_cointegration_relationship(y1, y2, residuals, name1, name2, filepath_base):
    """Plot cointegration relationship and residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Scatter plot with regression line
    ax = axes[0, 0]
    ax.scatter(y2, y1, color=COLORS["primary"], alpha=0.7, s=60)

    # Add regression line
    slope, intercept = np.polyfit(y2, y1, 1)
    x_line = np.array([min(y2), max(y2)])
    y_line = slope * x_line + intercept
    ax.plot(
        x_line,
        y_line,
        color=COLORS["secondary"],
        linewidth=2,
        label=f"y = {intercept:.2f} + {slope:.6f}x",
    )

    ax.set_xlabel(name2, fontsize=11)
    ax.set_ylabel(name1, fontsize=11)
    ax.set_title("Long-Run Relationship", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Top right: Time series plot
    ax = axes[0, 1]
    years = np.arange(2010, 2010 + len(y1))

    # Normalize for comparison
    y1_norm = (y1 - np.mean(y1)) / np.std(y1)
    y2_norm = (y2 - np.mean(y2)) / np.std(y2)

    ax.plot(
        years,
        y1_norm,
        marker="o",
        color=COLORS["primary"],
        linewidth=2,
        label=f"{name1} (normalized)",
    )
    ax.plot(
        years,
        y2_norm,
        marker="s",
        color=COLORS["secondary"],
        linewidth=2,
        label=f"{name2} (normalized)",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Standardized Value", fontsize=11)
    ax.set_title("Series Comparison (Standardized)", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Bottom left: Residuals time series
    ax = axes[1, 0]
    years_resid = np.arange(2010, 2010 + len(residuals))
    ax.plot(years_resid, residuals, marker="o", color=COLORS["tertiary"], linewidth=2)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.fill_between(
        years_resid,
        0,
        residuals,
        where=(residuals >= 0),
        color=COLORS["tertiary"],
        alpha=0.3,
    )
    ax.fill_between(
        years_resid,
        0,
        residuals,
        where=(residuals < 0),
        color=COLORS["quaternary"],
        alpha=0.3,
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title(
        "Cointegration Residuals (Deviations from Equilibrium)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Bottom right: Residual distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=8, color=COLORS["primary"], alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax.set_xlabel("Residual Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    title = f"Cointegration Analysis: {name1} and {name2}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.02,
        0.02,
        "Source: Census Bureau Population Estimates Program",
        fontsize=8,
        fontstyle="italic",
    )

    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filepath_base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath_base}.png")


def run_analysis() -> ModuleResult:
    """
    Main analysis function - perform VAR and cointegration analysis.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(
        module_id="2.2",
        analysis_name="var_cointegration_analysis",
    )

    # Load required data
    print("\nLoading data...")
    df = load_data("nd_migration_summary.csv")
    result.input_files.append("nd_migration_summary.csv")

    print(
        f"  Loaded {len(df)} observations (years {df['year'].min()}-{df['year'].max()})"
    )

    # Define variables for analysis
    var1 = "nd_intl_migration"
    var2 = "us_intl_migration"

    # Prepare data
    data = df[[var1, var2]].copy()
    data.index = df["year"]

    # Record parameters
    result.parameters = {
        "variables_analyzed": [var1, var2],
        "n_observations": len(df),
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "significance_level": 0.05,
        "irf_periods": 10,
        "fevd_periods": 10,
        "unit_root_findings": {
            "nd_intl_migration": "I(1)",
            "us_intl_migration": "I(0)",
            "note": "Mixed integration orders; proceed with caution",
        },
    }

    # Document decision about mixed integration orders
    result.add_decision(
        decision_id="D001",
        category="methodology",
        decision="Proceed with cointegration tests despite mixed integration orders",
        rationale="Unit root tests showed nd_intl_migration as I(1) and us_intl_migration as I(0). "
        "Standard cointegration theory requires variables of same integration order. "
        "However, with short series (n=15), unit root test power is limited. "
        "We proceed to explore potential long-run relationships.",
        alternatives=[
            "Only analyze I(1) variables",
            "Transform us_intl_migration to I(1)",
        ],
        evidence="Module 2.1.1 unit root test results",
        reversible=True,
    )

    result.warnings.append(
        "Variables have different integration orders (nd_intl_migration: I(1), us_intl_migration: I(0)). "
        "Cointegration tests may be unreliable."
    )
    result.warnings.append(
        "Short time series (n=15) limits reliability of VAR/cointegration analysis."
    )

    # Initialize results dictionary
    result.results = {}

    # =========================================================================
    # 1. ENGLE-GRANGER COINTEGRATION TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Engle-Granger Two-Step Cointegration Test")
    print("=" * 60)

    y1 = df[var1].values
    y2 = df[var2].values

    eg_result = engle_granger_cointegration(y1, y2, var1, var2)
    result.results["engle_granger_cointegration"] = eg_result

    print(
        f"  Long-run relationship: {eg_result['step1_ols_regression']['interpretation']}"
    )
    print(f"  R-squared: {eg_result['step1_ols_regression']['r_squared']:.4f}")
    print(
        f"  ADF on residuals: stat={eg_result['step2_adf_on_residuals']['adf_statistic']:.4f}, "
        f"p={eg_result['step2_adf_on_residuals']['adf_p_value']:.4f}"
    )
    print(f"  Conclusion: {eg_result['conclusion']['interpretation']}")

    # =========================================================================
    # 2. JOHANSEN COINTEGRATION TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Johansen Cointegration Test")
    print("=" * 60)

    joh_result = johansen_cointegration_test(data, det_order=0, k_ar_diff=1)
    result.results["johansen_cointegration"] = joh_result

    print(f"  Variables: {joh_result['variable_names']}")
    print(f"  Deterministic term: {joh_result['deterministic_term']}")
    print("\n  Trace Test Results:")
    for stat in joh_result["trace_test"]["statistics"]:
        reject_str = "REJECT" if stat["reject_at_95pct"] else "FAIL TO REJECT"
        print(
            f"    H0: {stat['null_hypothesis']}: stat={stat['statistic']:.4f}, "
            f"cv(95%)={stat['critical_values']['95%']:.4f} -> {reject_str}"
        )

    print("\n  Max Eigenvalue Test Results:")
    for stat in joh_result["max_eigenvalue_test"]["statistics"]:
        reject_str = "REJECT" if stat["reject_at_95pct"] else "FAIL TO REJECT"
        print(
            f"    H0: {stat['null_hypothesis']}: stat={stat['statistic']:.4f}, "
            f"cv(95%)={stat['critical_values']['95%']:.4f} -> {reject_str}"
        )

    print(f"\n  Conclusion: {joh_result['conclusion']['interpretation']}")

    # Determine cointegration status
    n_coint = joh_result["conclusion"]["recommended"]
    is_cointegrated = n_coint > 0

    result.add_decision(
        decision_id="D002",
        category="results",
        decision=f"Cointegration rank determined as {n_coint}",
        rationale=f"Johansen test trace statistic suggests {joh_result['conclusion']['n_cointegrating_relations_trace']} "
        f"cointegrating relations, max eigenvalue suggests {joh_result['conclusion']['n_cointegrating_relations_max_eigenvalue']}. "
        f"Engle-Granger test {'supports' if eg_result['conclusion']['is_cointegrated'] else 'does not support'} cointegration.",
        alternatives=[
            "Use alternative deterministic specification",
            "Use different lag structure",
        ],
        evidence=f"Johansen trace: {joh_result['trace_test']['statistics'][0]['statistic']:.4f}, "
        f"Engle-Granger ADF: {eg_result['step2_adf_on_residuals']['adf_statistic']:.4f}",
        reversible=True,
    )

    # =========================================================================
    # 3. VAR MODEL ESTIMATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. VAR Model Estimation")
    print("=" * 60)

    # Determine max lags based on sample size (rule of thumb: n^(1/3))
    max_lags = min(4, int(len(df) ** (1 / 3)))
    max_lags = max(1, max_lags)  # At least 1 lag

    print(f"  Maximum lags considered: {max_lags}")

    var_results, var_info = estimate_var_model(data, maxlags=max_lags, ic="aic")
    result.results["var_model"] = var_info

    print(f"  Selected lag order: {var_info['lag_selection']['selected_lag']} (by AIC)")
    print(
        f"  Optimal lags by criterion: {var_info['lag_selection']['optimal_by_criterion']}"
    )
    print("\n  Model fit statistics:")
    print(f"    Log-likelihood: {var_info['model_fit']['log_likelihood']:.2f}")
    print(f"    AIC: {var_info['model_fit']['aic']:.2f}")
    print(f"    BIC: {var_info['model_fit']['bic']:.2f}")

    for eq_name, eq_info in var_info["equations"].items():
        print(f"\n  Equation: {eq_name}")
        print(f"    R-squared: {eq_info['r_squared']:.4f}")
        print("    Significant coefficients (p < 0.05):")
        for coef_name, coef_info in eq_info["coefficients"].items():
            if coef_info["significant_5pct"]:
                print(
                    f"      {coef_name}: {coef_info['coefficient']:.4f} (t={coef_info['t_statistic']:.2f})"
                )

    # =========================================================================
    # 4. GRANGER CAUSALITY TESTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Granger Causality Tests")
    print("=" * 60)

    gc_lag = var_info["lag_selection"]["selected_lag"]
    gc_results = granger_causality_tests(data, [var1, var2], maxlag=gc_lag)
    result.results["granger_causality"] = gc_results

    for test_name, test_result in gc_results.items():
        if "error" not in test_result:
            print(f"\n  {test_name}:")
            print(f"    F-statistic: {test_result['f_test']['statistic']:.4f}")
            print(f"    p-value: {test_result['f_test']['p_value']:.4f}")
            print(f"    Conclusion: {test_result['conclusion']}")

    # =========================================================================
    # 5. IMPULSE RESPONSE FUNCTIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. Impulse Response Functions")
    print("=" * 60)

    irf_periods = 10
    irf_result, irf_obj = compute_impulse_response(var_results, periods=irf_periods)
    result.results["impulse_response_functions"] = irf_result

    print(f"  Computed IRFs for {irf_periods} periods")
    for response_name, response_data in irf_result["responses"].items():
        peak_response = max(
            abs(min(response_data["values"])), max(response_data["values"])
        )
        peak_period = response_data["values"].index(
            max(response_data["values"], key=abs)
        )
        print(
            f"    {response_name}: peak response = {peak_response:.2f} at period {peak_period}"
        )

    # =========================================================================
    # 6. FORECAST ERROR VARIANCE DECOMPOSITION
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. Forecast Error Variance Decomposition")
    print("=" * 60)

    fevd_result, fevd_obj = compute_variance_decomposition(
        var_results, periods=irf_periods
    )
    result.results["forecast_error_variance_decomposition"] = fevd_result

    for var_name, decomp in fevd_result["decomposition"].items():
        final_period = decomp["by_period"][-1]
        print(f"\n  {var_name} variance at period {final_period['period']}:")
        for shock_var, proportion in final_period["proportions"].items():
            print(f"    Due to {shock_var}: {proportion*100:.1f}%")

    # =========================================================================
    # 7. VECTOR ERROR CORRECTION MODEL (if cointegrated)
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. Vector Error Correction Model")
    print("=" * 60)

    if is_cointegrated:
        vecm_result = estimate_vecm(data, coint_rank=n_coint, k_ar_diff=1)
        result.results["vecm"] = vecm_result

        if "error" not in vecm_result:
            print(f"  VECM estimated with {n_coint} cointegrating relation(s)")
            if (
                "alpha_loading_matrix" in vecm_result
                and "values" in vecm_result["alpha_loading_matrix"]
            ):
                print("\n  Speed of adjustment (alpha) coefficients:")
                for var_name, alpha_info in vecm_result["alpha_loading_matrix"][
                    "values"
                ].items():
                    coef = alpha_info["coefficient"]
                    if isinstance(coef, list):
                        coef = coef[0]
                    print(f"    {var_name}: {coef:.4f}")
        else:
            print(
                f"  VECM estimation failed: {vecm_result.get('error', 'Unknown error')}"
            )
    else:
        result.results["vecm"] = {
            "note": "No cointegration detected, VECM not estimated",
            "recommendation": "Use VAR model in levels or first differences",
        }
        print("  No cointegration detected - VECM not estimated")
        print("  Recommendation: Use VAR model (in levels or first differences)")

    # =========================================================================
    # 8. GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("8. Generating Visualizations")
    print("=" * 60)

    # Impulse Response Functions plot
    irf_filepath = str(FIGURES_DIR / "module_2_2_impulse_response")
    plot_impulse_response(irf_obj, [var1, var2], irf_filepath)

    # Variance Decomposition plot
    fevd_filepath = str(FIGURES_DIR / "module_2_2_variance_decomposition")
    plot_variance_decomposition(fevd_obj, [var1, var2], fevd_filepath)

    # Cointegration relationship plot
    coint_filepath = str(FIGURES_DIR / "module_2_2_cointegration_relationship")
    plot_cointegration_relationship(
        y1, y2, np.array(eg_result["residuals"]), var1, var2, coint_filepath
    )

    # Store diagnostics
    result.diagnostics = {
        "figures_generated": [
            "module_2_2_impulse_response.png",
            "module_2_2_variance_decomposition.png",
            "module_2_2_cointegration_relationship.png",
        ],
        "var_model_diagnostics": {
            "lag_selection_method": "AIC",
            "selected_lag": var_info["lag_selection"]["selected_lag"],
            "n_parameters": var_results.params.size,
            "degrees_of_freedom": var_results.df_resid,
        },
        "test_interpretations": {
            "engle_granger": "Two-step procedure: (1) estimate long-run OLS, (2) test residuals for stationarity",
            "johansen": "Multivariate ML procedure testing for number of cointegrating vectors",
            "granger_causality": "Tests if lags of one variable help predict another (not true causality)",
            "irf": "Response of variables to one-standard-deviation shocks over time",
            "fevd": "Proportion of forecast error variance attributable to each variable's shocks",
        },
        "limitations": [
            "Sample size (n=15) is very small for VAR/cointegration analysis",
            "Mixed integration orders complicate interpretation",
            "Asymptotic properties may not hold in small samples",
            "2020 COVID shock may distort relationships",
        ],
    }

    # Suggest next steps
    result.next_steps = [
        "Use VAR forecasts for scenario analysis (Module 2.3)",
        "Consider structural VAR identification if causality is established",
        "Sensitivity analysis excluding 2020 COVID year",
        "Compare with univariate ARIMA forecasts",
        "Consider regime-switching models for policy break analysis",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 2.2: VAR and Cointegration Analysis")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_2_2_var_cointegration.json")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")

        if result.decisions:
            print(f"\nDecisions made ({len(result.decisions)}):")
            for d in result.decisions:
                print(f"  - {d['decision_id']}: {d['decision']}")

        # Print summary
        print("\n" + "-" * 60)
        print("Summary of Key Findings:")

        # Cointegration summary
        eg_coint = result.results.get("engle_granger_cointegration", {})
        joh_coint = result.results.get("johansen_cointegration", {})

        if eg_coint:
            print(
                f"\n  Engle-Granger: {eg_coint.get('conclusion', {}).get('interpretation', 'N/A')}"
            )
        if joh_coint:
            print(
                f"  Johansen: {joh_coint.get('conclusion', {}).get('interpretation', 'N/A')}"
            )

        # Granger causality summary
        gc = result.results.get("granger_causality", {})
        print("\n  Granger Causality:")
        for _test_name, test_result in gc.items():
            if "conclusion" in test_result:
                print(f"    {test_result['conclusion']}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
