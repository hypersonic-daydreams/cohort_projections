"""
Model Comparison: Classical vs Bayesian VAR
==========================================

Compares classical MLE-based VAR with Bayesian VAR (Minnesota prior)
to assess whether Bayesian methods add value for this small-n problem.

Comparison criteria:
1. Forecast accuracy (RMSE, MAE)
2. Uncertainty calibration (prediction interval coverage)
3. Coefficient stability
4. Robustness of conclusions
"""

from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


@dataclass
class ModelComparisonResult:
    """
    Container for model comparison results.

    Attributes
    ----------
    classical_results : dict
        Summary of classical VAR results
    bayesian_results : dict
        Summary of Bayesian VAR results
    panel_results : dict, optional
        Summary of Panel VAR results
    forecast_comparison : dict
        Forecast accuracy metrics
    uncertainty_comparison : dict
        Uncertainty quantification comparison
    coefficient_comparison : dict
        Coefficient estimate comparison
    recommendation : dict
        Summary recommendation
    """

    classical_results: dict
    bayesian_results: dict
    panel_results: Optional[dict]
    forecast_comparison: dict
    uncertainty_comparison: dict
    coefficient_comparison: dict
    recommendation: dict
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "classical_results": self.classical_results,
            "bayesian_results": self.bayesian_results,
            "panel_results": self.panel_results,
            "forecast_comparison": self.forecast_comparison,
            "uncertainty_comparison": self.uncertainty_comparison,
            "coefficient_comparison": self.coefficient_comparison,
            "recommendation": self.recommendation,
            "diagnostics": self.diagnostics,
        }


def estimate_classical_var(
    data: pd.DataFrame,
    var_cols: list[str],
    n_lags: int = 1,
) -> dict:
    """
    Estimate classical VAR using statsmodels.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    n_lags : int
        Number of lags

    Returns
    -------
    dict
        Classical VAR results summary
    """
    model = VAR(data[var_cols])
    results = model.fit(maxlags=n_lags)

    # Extract coefficients
    coefficients = {}
    coef_std = {}

    for i, var in enumerate(var_cols):
        eq_coefs = {}
        eq_std = {}

        # Constant
        eq_coefs["const"] = float(results.params.iloc[0, i])
        eq_std["const"] = float(results.stderr.iloc[0, i])

        # Lag coefficients
        for lag in range(1, n_lags + 1):
            for j, var_j in enumerate(var_cols):
                idx = 1 + (lag - 1) * len(var_cols) + j
                name = f"L{lag}.{var_j}"
                eq_coefs[name] = float(results.params.iloc[idx, i])
                eq_std[name] = float(results.stderr.iloc[idx, i])

        coefficients[var] = eq_coefs
        coef_std[var] = eq_std

    # Confidence intervals (95%)
    conf_int = {}
    for i, var in enumerate(var_cols):
        var_ci = {}
        for j, param_name in enumerate(results.params.index):
            coef = results.params.iloc[j, i]
            se = results.stderr.iloc[j, i]
            var_ci[param_name] = (float(coef - 1.96 * se), float(coef + 1.96 * se))
        conf_int[var] = var_ci

    return {
        "method": "classical_mle",
        "coefficients": coefficients,
        "coef_std": coef_std,
        "confidence_intervals_95": conf_int,
        "aic": float(results.aic),
        "bic": float(results.bic),
        "n_obs": int(results.nobs),
        "n_lags": n_lags,
        "sigma": results.sigma_u.values.tolist()
        if hasattr(results.sigma_u, "values")
        else results.sigma_u.tolist(),
        "results_object": results,
    }


def leave_one_out_cv(
    data: pd.DataFrame,
    var_cols: list[str],
    target_var: str,
    n_lags: int,
    estimation_fn: callable,
) -> dict:
    """
    Perform leave-one-out cross-validation for forecast evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    target_var : str
        Target variable for forecast evaluation
    n_lags : int
        Number of lags
    estimation_fn : callable
        Function that takes data and returns fitted model

    Returns
    -------
    dict
        LOO-CV results
    """
    n_obs = len(data)
    errors = []

    # Leave-one-out for each observation (excluding first n_lags)
    for i in range(n_lags, n_obs - 1):
        # Train on all except observation i
        train_idx = list(range(i)) + list(range(i + 1, n_obs))
        train_data = data.iloc[train_idx].reset_index(drop=True)

        # Get actual value at i
        actual = data.iloc[i][target_var]

        try:
            # Fit model on training data
            model_results = estimation_fn(train_data, var_cols, n_lags)

            # Get coefficient for one-step forecast
            if (
                "coefficients" in model_results
                and target_var in model_results["coefficients"]
            ):
                coefs = model_results["coefficients"][target_var]
                # Simple one-step forecast using previous values
                forecast = coefs.get("const", 0)
                for lag in range(1, n_lags + 1):
                    lag_val = data.iloc[i - lag][target_var]
                    forecast += coefs.get(f"L{lag}.{target_var}", 0) * lag_val

                error = actual - forecast
                errors.append(error)
        except Exception:
            continue

    if len(errors) > 0:
        errors = np.array(errors)
        return {
            "n_folds": len(errors),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    else:
        return {
            "n_folds": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "error": "LOO-CV failed",
        }


def expanding_window_forecast(
    data: pd.DataFrame,
    var_cols: list[str],
    target_var: str,
    n_lags: int,
    estimation_fn: callable,
    min_train: int = 10,
) -> dict:
    """
    Evaluate forecasts using expanding window.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    target_var : str
        Target variable
    n_lags : int
        Number of lags
    estimation_fn : callable
        Estimation function
    min_train : int
        Minimum training observations

    Returns
    -------
    dict
        Expanding window forecast results
    """
    n_obs = len(data)
    actuals = []
    forecasts = []
    forecast_se = []

    for t in range(min_train, n_obs):
        train_data = data.iloc[:t].copy()
        actual = data.iloc[t][target_var]

        try:
            model_results = estimation_fn(train_data, var_cols, n_lags)

            if (
                "coefficients" in model_results
                and target_var in model_results["coefficients"]
            ):
                coefs = model_results["coefficients"][target_var]
                coef_std = model_results.get("coef_std", {}).get(target_var, {})

                # One-step forecast
                forecast = coefs.get("const", 0)
                for lag in range(1, n_lags + 1):
                    lag_val = data.iloc[t - lag][target_var]
                    forecast += coefs.get(f"L{lag}.{target_var}", 0) * lag_val

                actuals.append(actual)
                forecasts.append(forecast)

                # Approximate forecast SE
                se_sum = sum(coef_std.get(k, 0) ** 2 for k in coefs.keys())
                forecast_se.append(np.sqrt(se_sum))
        except Exception:
            continue

    if len(actuals) > 0:
        actuals = np.array(actuals)
        forecasts = np.array(forecasts)
        errors = actuals - forecasts

        return {
            "n_forecasts": len(actuals),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "mape": float(np.mean(np.abs(errors / (actuals + 1e-8))) * 100),
            "mean_error": float(np.mean(errors)),
            "forecast_se": forecast_se,
        }
    else:
        return {"error": "Expanding window forecast failed"}


def compute_prediction_interval_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """
    Compute prediction interval coverage probability.

    Parameters
    ----------
    actuals : np.ndarray
        Actual values
    lower : np.ndarray
        Lower bounds of prediction intervals
    upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    dict
        Coverage statistics
    """
    covered = (actuals >= lower) & (actuals <= upper)

    return {
        "coverage": float(np.mean(covered)),
        "n_covered": int(np.sum(covered)),
        "n_total": len(actuals),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def compare_coefficients(
    classical_coefs: dict,
    bayesian_coefs: dict,
    var_cols: list[str],
) -> dict:
    """
    Compare coefficient estimates between classical and Bayesian models.

    Parameters
    ----------
    classical_coefs : dict
        Classical VAR coefficients
    bayesian_coefs : dict
        Bayesian VAR coefficients
    var_cols : list[str]
        Variable names

    Returns
    -------
    dict
        Coefficient comparison
    """
    comparison = {}

    for var in var_cols:
        if var not in classical_coefs or var not in bayesian_coefs:
            continue

        var_comp = {}
        class_eq = classical_coefs[var]
        bayes_eq = bayesian_coefs[var]

        for coef_name in class_eq:
            if coef_name in bayes_eq:
                class_val = class_eq[coef_name]
                bayes_val = bayes_eq[coef_name]

                diff = bayes_val - class_val
                pct_diff = (diff / (abs(class_val) + 1e-8)) * 100

                var_comp[coef_name] = {
                    "classical": float(class_val),
                    "bayesian": float(bayes_val),
                    "difference": float(diff),
                    "pct_difference": float(pct_diff),
                    "same_sign": np.sign(class_val) == np.sign(bayes_val),
                }

        comparison[var] = var_comp

    # Summary statistics
    all_diffs = []
    all_pct_diffs = []
    sign_agreements = []

    for var in comparison:
        for coef in comparison[var]:
            all_diffs.append(comparison[var][coef]["difference"])
            all_pct_diffs.append(comparison[var][coef]["pct_difference"])
            sign_agreements.append(comparison[var][coef]["same_sign"])

    comparison["summary"] = {
        "mean_absolute_difference": float(np.mean(np.abs(all_diffs))),
        "mean_pct_difference": float(np.mean(np.abs(all_pct_diffs))),
        "sign_agreement_rate": float(np.mean(sign_agreements)),
        "interpretation": (
            "Bayesian shrinkage moves coefficients toward prior (random walk). "
            f"Average shrinkage: {np.mean(np.abs(all_pct_diffs)):.1f}%."
        ),
    }

    return comparison


def assess_uncertainty_calibration(
    classical_se: dict,
    bayesian_se: dict,
    var_cols: list[str],
) -> dict:
    """
    Compare uncertainty quantification between methods.

    Parameters
    ----------
    classical_se : dict
        Classical VAR standard errors
    bayesian_se : dict
        Bayesian VAR posterior standard deviations
    var_cols : list[str]
        Variable names

    Returns
    -------
    dict
        Uncertainty comparison
    """
    comparison = {}

    for var in var_cols:
        if var not in classical_se or var not in bayesian_se:
            continue

        var_comp = {}
        class_eq = classical_se[var]
        bayes_eq = bayesian_se[var]

        for coef_name in class_eq:
            if coef_name in bayes_eq:
                class_se = class_eq[coef_name]
                bayes_se = bayes_eq[coef_name]

                var_comp[coef_name] = {
                    "classical_se": float(class_se),
                    "bayesian_posterior_sd": float(bayes_se),
                    "ratio": float(bayes_se / (class_se + 1e-8)),
                    "bayesian_tighter": bayes_se < class_se,
                }

        comparison[var] = var_comp

    # Summary
    all_ratios = []
    tighter_count = 0

    for var in comparison:
        if var == "summary":
            continue
        for coef in comparison[var]:
            all_ratios.append(comparison[var][coef]["ratio"])
            if comparison[var][coef]["bayesian_tighter"]:
                tighter_count += 1

    comparison["summary"] = {
        "mean_se_ratio": float(np.mean(all_ratios)) if all_ratios else np.nan,
        "bayesian_tighter_pct": float(tighter_count / len(all_ratios) * 100)
        if all_ratios
        else 0,
        "interpretation": (
            "Ratio < 1 indicates Bayesian posterior is tighter (more confident). "
            "Prior information reduces uncertainty when prior is informative."
        ),
    }

    return comparison


def generate_recommendation(
    forecast_comparison: dict,
    coefficient_comparison: dict,
    uncertainty_comparison: dict,
    n_obs: int,
) -> dict:
    """
    Generate recommendation on whether Bayesian VAR adds value.

    Parameters
    ----------
    forecast_comparison : dict
        Forecast comparison results
    coefficient_comparison : dict
        Coefficient comparison
    uncertainty_comparison : dict
        Uncertainty comparison
    n_obs : int
        Number of observations

    Returns
    -------
    dict
        Recommendation
    """
    points_for_bayesian = 0
    points_for_classical = 0
    reasons = []

    # Check forecast accuracy
    if "rmse_improvement_pct" in forecast_comparison:
        if forecast_comparison["rmse_improvement_pct"] > 10:
            points_for_bayesian += 2
            reasons.append(
                f"Bayesian improves RMSE by {forecast_comparison['rmse_improvement_pct']:.1f}%"
            )
        elif forecast_comparison["rmse_improvement_pct"] < -10:
            points_for_classical += 2
            reasons.append(
                f"Classical has better RMSE by {-forecast_comparison['rmse_improvement_pct']:.1f}%"
            )
        else:
            reasons.append("Forecast accuracy similar between methods")

    # Check coefficient stability (sign agreement)
    if "summary" in coefficient_comparison:
        sign_rate = coefficient_comparison["summary"]["sign_agreement_rate"]
        if sign_rate >= 0.9:
            reasons.append("Coefficients have same signs - conclusions robust")
        else:
            reasons.append(
                f"WARNING: Sign disagreement in {(1-sign_rate)*100:.0f}% of coefficients"
            )

    # Small sample consideration
    if n_obs < 20:
        points_for_bayesian += 1
        reasons.append(f"Small sample (n={n_obs}) favors Bayesian regularization")

    # Determine recommendation
    if points_for_bayesian > points_for_classical:
        recommendation = "BAYESIAN_PREFERRED"
        summary = (
            "Bayesian VAR with Minnesota prior recommended for this small-n problem."
        )
    elif points_for_classical > points_for_bayesian:
        recommendation = "CLASSICAL_PREFERRED"
        summary = "Classical VAR sufficient; Bayesian adds complexity without benefit."
    else:
        recommendation = "EITHER_ACCEPTABLE"
        summary = "Both methods give similar results; choice depends on preference."

    return {
        "recommendation": recommendation,
        "summary": summary,
        "points_bayesian": points_for_bayesian,
        "points_classical": points_for_classical,
        "reasons": reasons,
        "caveat": (
            "This comparison is based on single dataset. Cross-validation "
            "and sensitivity analysis recommended before final decision."
        ),
    }


def compare_var_models(
    data: pd.DataFrame,
    var_cols: list[str],
    bayesian_result: Any,
    n_lags: int = 1,
    panel_result: Optional[Any] = None,
) -> ModelComparisonResult:
    """
    Main entry point for comparing classical and Bayesian VAR models.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var_cols : list[str]
        Column names for VAR variables
    bayesian_result : BayesianVARResult
        Results from Bayesian VAR estimation
    n_lags : int
        Number of lags
    panel_result : PanelVARResult, optional
        Results from Panel VAR estimation

    Returns
    -------
    ModelComparisonResult
        Comprehensive comparison results
    """
    # Estimate classical VAR
    classical = estimate_classical_var(data, var_cols, n_lags)

    # Extract Bayesian results
    bayesian = {
        "method": bayesian_result.method,
        "coefficients": bayesian_result.coefficients,
        "coef_std": bayesian_result.coef_std,
        "credible_intervals_90": bayesian_result.credible_intervals,
        "n_obs": bayesian_result.n_obs,
        "n_lags": bayesian_result.n_lags,
        "diagnostics": bayesian_result.diagnostics,
    }

    # Panel VAR summary (if provided)
    panel = None
    if panel_result is not None:
        panel = panel_result.to_dict()

    # Coefficient comparison
    coef_comp = compare_coefficients(
        classical["coefficients"],
        bayesian["coefficients"],
        var_cols,
    )

    # Uncertainty comparison
    uncert_comp = assess_uncertainty_calibration(
        classical["coef_std"],
        bayesian["coef_std"],
        var_cols,
    )

    # Forecast comparison (using expanding window)
    def classical_fit(d, cols, lags):
        return estimate_classical_var(d, cols, lags)

    def bayesian_fit(d, cols, lags):
        # Use conjugate prior for CV (faster)
        from .bayesian_var import estimate_bayesian_var

        result = estimate_bayesian_var(d, cols, lags, use_pymc=False)
        return {
            "coefficients": result.coefficients,
            "coef_std": result.coef_std,
        }

    target_var = var_cols[0]

    classical_forecast = expanding_window_forecast(
        data, var_cols, target_var, n_lags, classical_fit, min_train=max(10, n_lags + 5)
    )
    bayesian_forecast = expanding_window_forecast(
        data, var_cols, target_var, n_lags, bayesian_fit, min_train=max(10, n_lags + 5)
    )

    forecast_comp = {
        "classical": classical_forecast,
        "bayesian": bayesian_forecast,
    }

    if "rmse" in classical_forecast and "rmse" in bayesian_forecast:
        if not np.isnan(classical_forecast["rmse"]) and not np.isnan(
            bayesian_forecast["rmse"]
        ):
            improvement = (
                (classical_forecast["rmse"] - bayesian_forecast["rmse"])
                / classical_forecast["rmse"]
                * 100
            )
            forecast_comp["rmse_improvement_pct"] = float(improvement)
            forecast_comp["bayesian_better_rmse"] = (
                bayesian_forecast["rmse"] < classical_forecast["rmse"]
            )

    # Generate recommendation
    recommendation = generate_recommendation(
        forecast_comp,
        coef_comp,
        uncert_comp,
        len(data),
    )

    # Additional diagnostics
    diagnostics = {
        "data_points": len(data),
        "variables": var_cols,
        "lags": n_lags,
        "classical_aic": classical["aic"],
        "classical_bic": classical["bic"],
    }

    return ModelComparisonResult(
        classical_results=classical,
        bayesian_results=bayesian,
        panel_results=panel,
        forecast_comparison=forecast_comp,
        uncertainty_comparison=uncert_comp,
        coefficient_comparison=coef_comp,
        recommendation=recommendation,
        diagnostics=diagnostics,
    )
