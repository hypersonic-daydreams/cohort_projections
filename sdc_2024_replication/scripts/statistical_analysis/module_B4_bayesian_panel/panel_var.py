"""
Panel VAR Estimation
====================

Implements Panel VAR methods that leverage multi-state data to overcome
small-n limitations in single-state time series analysis.

Key approach: Use 50-state panel (50 entities x 15 periods = 750 obs)
to estimate more stable VAR coefficients, then extract ND-specific effects.

Methods:
1. Pooled Panel VAR with entity fixed effects
2. Mean Group Estimator (heterogeneous slopes)
3. ND-specific coefficient extraction
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PanelVARResult:
    """
    Container for Panel VAR estimation results.

    Attributes
    ----------
    method : str
        Estimation method ('panel_fe', 'mean_group', 'pooled')
    coefficients : dict
        Estimated coefficients
    coef_std : dict
        Standard errors of coefficients
    nd_coefficients : dict
        ND-specific coefficient estimates (if available)
    entity_effects : dict
        Entity (state) fixed effects
    time_effects : dict
        Time fixed effects
    n_entities : int
        Number of entities in panel
    n_periods : int
        Number of time periods
    n_obs : int
        Total number of observations used
    r_squared : float
        R-squared from panel regression
    variable_names : list[str]
        Names of VAR variables
    diagnostics : dict
        Additional diagnostics
    """

    method: str
    coefficients: dict
    coef_std: dict
    nd_coefficients: Optional[dict]
    entity_effects: dict
    time_effects: dict
    n_entities: int
    n_periods: int
    n_obs: int
    r_squared: float
    variable_names: list[str]
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "coefficients": self.coefficients,
            "coef_std": self.coef_std,
            "nd_coefficients": self.nd_coefficients,
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects,
            "n_entities": self.n_entities,
            "n_periods": self.n_periods,
            "n_obs": self.n_obs,
            "r_squared": self.r_squared,
            "variable_names": self.variable_names,
            "diagnostics": self.diagnostics,
        }


def prepare_panel_var_data(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    target_var: str,
    n_lags: int = 1,
) -> pd.DataFrame:
    """
    Prepare panel data for VAR estimation by creating lagged variables.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with entity and time columns
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time period
    target_var : str
        Column name for target variable
    n_lags : int
        Number of lags to create

    Returns
    -------
    pd.DataFrame
        Panel data with lagged variables
    """
    df = df.sort_values([entity_col, time_col]).copy()

    # Create lagged variables within each entity
    for lag in range(1, n_lags + 1):
        df[f"{target_var}_L{lag}"] = df.groupby(entity_col)[target_var].shift(lag)

    # Drop rows with missing lags
    df = df.dropna()

    return df


def estimate_panel_var_fe(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    target_var: str,
    n_lags: int = 1,
    nd_interaction: bool = True,
    focal_entity: str = "North Dakota",
) -> PanelVARResult:
    """
    Estimate Panel VAR with two-way fixed effects.

    Model:
        y_{it} = alpha_i + gamma_t + sum_k(beta_k * y_{i,t-k}) + epsilon_{it}

    With optional ND interaction:
        + sum_k(delta_k * D_ND * y_{i,t-k})

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time period
    target_var : str
        Column name for target variable
    n_lags : int
        Number of lags in VAR
    nd_interaction : bool
        Whether to include ND-specific slope interactions
    focal_entity : str
        Name of focal entity for interaction terms

    Returns
    -------
    PanelVARResult
        Estimation results
    """
    # Prepare data with lags
    panel_df = prepare_panel_var_data(df, entity_col, time_col, target_var, n_lags)

    # Create entity and time dummies
    entities = panel_df[entity_col].unique()
    times = panel_df[time_col].unique()
    n_entities = len(entities)
    n_periods = len(times) - n_lags  # Effective periods after lag truncation

    # Build design matrix
    # Dependent variable
    y = panel_df[target_var].values

    # Lagged dependent variables
    X_lags = []
    lag_names = []
    for lag in range(1, n_lags + 1):
        X_lags.append(panel_df[f"{target_var}_L{lag}"].values)
        lag_names.append(f"L{lag}.{target_var}")

    X_lags = np.column_stack(X_lags)

    # ND interaction terms
    if nd_interaction:
        nd_dummy = (panel_df[entity_col] == focal_entity).astype(float).values
        X_nd_interact = X_lags * nd_dummy[:, np.newaxis]
        nd_interact_names = [f"ND_x_{name}" for name in lag_names]
    else:
        X_nd_interact = np.empty((len(y), 0))
        nd_interact_names = []

    # Entity dummies (LSDV approach with reference category)
    entity_dummies = pd.get_dummies(
        panel_df[entity_col], drop_first=True, prefix="entity"
    ).values

    # Time dummies
    time_dummies = pd.get_dummies(
        panel_df[time_col], drop_first=True, prefix="time"
    ).values

    # Full design matrix: [lags, nd_interactions, entity_fe, time_fe, constant]
    X = np.column_stack(
        [
            np.ones(len(y)),
            X_lags,
            X_nd_interact,
            entity_dummies,
            time_dummies,
        ]
    )

    # OLS estimation with heteroskedasticity-robust standard errors
    n_obs = len(y)
    n_params = X.shape[1]

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        # Regularized inverse if singular
        XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(n_params))
        beta = XtX_inv @ X.T @ y

    # Residuals and fit statistics
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # HC1 robust standard errors
    resid_sq = residuals**2
    meat = X.T @ np.diag(resid_sq) @ X
    bread = XtX_inv
    robust_cov = bread @ meat @ bread * (n_obs / (n_obs - n_params))
    se_robust = np.sqrt(np.diag(robust_cov))

    # Organize coefficients
    coefficients = {}
    coef_std = {}

    # Main lag coefficients
    for i, name in enumerate(lag_names):
        idx = 1 + i  # Skip constant
        coefficients[name] = float(beta[idx])
        coef_std[name] = float(se_robust[idx])

    # ND-specific coefficients
    nd_coefficients = None
    if nd_interaction:
        nd_coefficients = {}
        for i, name in enumerate(nd_interact_names):
            idx = 1 + len(lag_names) + i
            nd_coefficients[name] = float(beta[idx])
            nd_coefficients[f"{name}_se"] = float(se_robust[idx])

        # Total ND effect = main + interaction
        nd_coefficients["total_ND_effect"] = {}
        for i, base_name in enumerate(lag_names):
            main_coef = coefficients[base_name]
            interact_coef = nd_coefficients[f"ND_x_{base_name}"]
            nd_coefficients["total_ND_effect"][base_name] = main_coef + interact_coef

    # Entity fixed effects
    entity_effects = {"reference": list(entities)[0]}
    fe_start = 1 + len(lag_names) + len(nd_interact_names)
    for i, entity in enumerate(entities[1:]):
        entity_effects[entity] = float(beta[fe_start + i])

    # Time fixed effects
    time_start = fe_start + len(entities) - 1
    time_effects = {"reference": times[0]}
    for i, t in enumerate(times[1:]):
        if time_start + i < len(beta):
            time_effects[int(t)] = float(beta[time_start + i])

    # Diagnostics
    diagnostics = {
        "dof": n_obs - n_params,
        "rmse": float(np.sqrt(ss_res / (n_obs - n_params))),
        "adjusted_r_squared": 1
        - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params - 1),
        "focal_entity": focal_entity,
        "nd_in_sample": focal_entity in entities,
    }

    # F-test for ND interaction significance
    if nd_interaction and len(nd_interact_names) > 0:
        # Wald test for joint significance of ND interactions
        nd_indices = [1 + len(lag_names) + i for i in range(len(nd_interact_names))]
        R = np.zeros((len(nd_indices), n_params))
        for i, idx in enumerate(nd_indices):
            R[i, idx] = 1

        r = np.zeros(len(nd_indices))
        Rb_r = R @ beta - r

        try:
            wald_stat = Rb_r.T @ np.linalg.inv(R @ robust_cov @ R.T) @ Rb_r
            df1 = len(nd_indices)
            df2 = n_obs - n_params
            f_stat = wald_stat / df1
            f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
            diagnostics["nd_interaction_f_test"] = {
                "f_statistic": float(f_stat),
                "df1": df1,
                "df2": df2,
                "p_value": float(f_pvalue),
                "significant_at_05": f_pvalue < 0.05,
            }
        except np.linalg.LinAlgError:
            diagnostics["nd_interaction_f_test"] = {
                "error": "Singular matrix in F-test"
            }

    return PanelVARResult(
        method="panel_fe_twoway",
        coefficients=coefficients,
        coef_std=coef_std,
        nd_coefficients=nd_coefficients,
        entity_effects=entity_effects,
        time_effects=time_effects,
        n_entities=n_entities,
        n_periods=n_periods,
        n_obs=n_obs,
        r_squared=r_squared,
        variable_names=[target_var],
        diagnostics=diagnostics,
    )


def estimate_mean_group(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    target_var: str,
    n_lags: int = 1,
    focal_entity: str = "North Dakota",
) -> PanelVARResult:
    """
    Estimate Mean Group estimator allowing for heterogeneous slopes.

    This method:
    1. Estimates separate AR(n_lags) model for each entity
    2. Averages coefficients across entities
    3. Reports ND-specific coefficients separately

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time period
    target_var : str
        Column name for target variable
    n_lags : int
        Number of lags in VAR
    focal_entity : str
        Name of focal entity

    Returns
    -------
    PanelVARResult
        Mean group estimation results
    """
    # Prepare data with lags
    panel_df = prepare_panel_var_data(df, entity_col, time_col, target_var, n_lags)

    entities = panel_df[entity_col].unique()
    n_entities = len(entities)

    # Estimate model for each entity
    entity_coefs = {}
    entity_se = {}

    for entity in entities:
        entity_data = panel_df[panel_df[entity_col] == entity]

        if len(entity_data) < n_lags + 3:  # Need minimum observations
            continue

        y = entity_data[target_var].values
        X_parts = [np.ones(len(y))]
        for lag in range(1, n_lags + 1):
            X_parts.append(entity_data[f"{target_var}_L{lag}"].values)
        X = np.column_stack(X_parts)

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            sigma_sq = np.sum(resid**2) / (len(y) - X.shape[1])
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(sigma_sq * np.diag(XtX_inv))

            entity_coefs[entity] = {"const": beta[0]}
            entity_se[entity] = {"const": se[0]}
            for lag in range(1, n_lags + 1):
                entity_coefs[entity][f"L{lag}"] = beta[lag]
                entity_se[entity][f"L{lag}"] = se[lag]
        except (np.linalg.LinAlgError, ValueError):
            continue

    # Mean group estimator
    mg_coefs = {"const": []}
    mg_se = {"const": []}
    for lag in range(1, n_lags + 1):
        mg_coefs[f"L{lag}"] = []
        mg_se[f"L{lag}"] = []

    for entity in entity_coefs:
        for key in mg_coefs:
            if key in entity_coefs[entity]:
                mg_coefs[key].append(entity_coefs[entity][key])

    # Calculate mean and standard error of mean
    coefficients = {}
    coef_std = {}
    for key in mg_coefs:
        if len(mg_coefs[key]) > 0:
            mean_val = np.mean(mg_coefs[key])
            # SE of mean = SD / sqrt(n)
            se_val = np.std(mg_coefs[key], ddof=1) / np.sqrt(len(mg_coefs[key]))
            coefficients[key] = float(mean_val)
            coef_std[key] = float(se_val)

    # ND-specific coefficients
    nd_coefficients = None
    if focal_entity in entity_coefs:
        nd_coefficients = {
            f"ND_{k}": float(v) for k, v in entity_coefs[focal_entity].items()
        }
        nd_coefficients["ND_se"] = {
            f"ND_{k}": float(v) for k, v in entity_se[focal_entity].items()
        }
        # Add comparison to mean
        nd_coefficients["deviation_from_mean"] = {
            k: float(entity_coefs[focal_entity][k] - coefficients.get(k, 0))
            for k in entity_coefs[focal_entity]
        }

    # Entity effects are just the constants
    entity_effects = {
        entity: float(entity_coefs[entity]["const"]) for entity in entity_coefs
    }

    # Calculate overall R-squared using MG predictions
    y_all = panel_df[target_var].values
    y_pred = []
    for _, row in panel_df.iterrows():
        entity = row[entity_col]
        if entity in entity_coefs:
            pred = entity_coefs[entity]["const"]
            for lag in range(1, n_lags + 1):
                pred += (
                    entity_coefs[entity].get(f"L{lag}", 0) * row[f"{target_var}_L{lag}"]
                )
            y_pred.append(pred)
        else:
            y_pred.append(np.nan)

    y_pred = np.array(y_pred)
    valid = ~np.isnan(y_pred)
    ss_res = np.sum((y_all[valid] - y_pred[valid]) ** 2)
    ss_tot = np.sum((y_all[valid] - np.mean(y_all[valid])) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    diagnostics = {
        "n_entities_estimated": len(entity_coefs),
        "n_entities_failed": n_entities - len(entity_coefs),
        "focal_entity": focal_entity,
        "nd_in_sample": focal_entity in entity_coefs,
        "cross_section_variation": {
            k: float(np.std(mg_coefs[k], ddof=1)) if len(mg_coefs[k]) > 1 else 0.0
            for k in mg_coefs
        },
    }

    return PanelVARResult(
        method="mean_group",
        coefficients=coefficients,
        coef_std=coef_std,
        nd_coefficients=nd_coefficients,
        entity_effects=entity_effects,
        time_effects={},  # MG doesn't have time effects
        n_entities=len(entity_coefs),
        n_periods=len(panel_df[time_col].unique()) - n_lags,
        n_obs=len(panel_df),
        r_squared=r_squared,
        variable_names=[target_var],
        diagnostics=diagnostics,
    )


def estimate_panel_var(
    df: pd.DataFrame,
    entity_col: str = "state",
    time_col: str = "year",
    target_var: str = "intl_migration",
    n_lags: int = 1,
    method: str = "panel_fe",
    focal_entity: str = "North Dakota",
    nd_interaction: bool = True,
) -> PanelVARResult:
    """
    Main entry point for Panel VAR estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time period
    target_var : str
        Column name for target variable
    n_lags : int
        Number of lags
    method : str
        Estimation method: 'panel_fe' or 'mean_group'
    focal_entity : str
        Name of focal entity (e.g., "North Dakota")
    nd_interaction : bool
        Whether to include focal entity interaction terms

    Returns
    -------
    PanelVARResult
        Estimation results
    """
    if method == "panel_fe":
        return estimate_panel_var_fe(
            df=df,
            entity_col=entity_col,
            time_col=time_col,
            target_var=target_var,
            n_lags=n_lags,
            nd_interaction=nd_interaction,
            focal_entity=focal_entity,
        )
    elif method == "mean_group":
        return estimate_mean_group(
            df=df,
            entity_col=entity_col,
            time_col=time_col,
            target_var=target_var,
            n_lags=n_lags,
            focal_entity=focal_entity,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'panel_fe' or 'mean_group'.")


def compare_nd_to_distribution(
    result: PanelVARResult,
    df: pd.DataFrame,
    entity_col: str = "state",
    time_col: str = "year",
    target_var: str = "intl_migration",
) -> dict:
    """
    Compare ND's estimated coefficients to the cross-sectional distribution.

    Parameters
    ----------
    result : PanelVARResult
        Panel VAR result (should be from mean_group method)
    df : pd.DataFrame
        Original panel data
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time period
    target_var : str
        Column name for target variable

    Returns
    -------
    dict
        Comparison statistics including percentile rank
    """
    if result.method != "mean_group":
        # Re-estimate with mean group to get entity-level coefficients
        mg_result = estimate_mean_group(
            df, entity_col, time_col, target_var, n_lags=1, focal_entity="North Dakota"
        )
    else:
        mg_result = result

    comparison = {
        "focal_entity": "North Dakota",
        "coefficients": {},
    }

    # Get all entity coefficients from entity_effects
    all_entity_coefs = list(mg_result.entity_effects.values())

    if mg_result.nd_coefficients and "ND_L1" in mg_result.nd_coefficients:
        nd_coef = mg_result.nd_coefficients["ND_L1"]

        # Calculate percentile rank
        percentile = stats.percentileofscore(all_entity_coefs, nd_coef)

        comparison["coefficients"]["L1"] = {
            "nd_value": float(nd_coef),
            "mean_all_states": float(np.mean(all_entity_coefs)),
            "std_all_states": float(np.std(all_entity_coefs)),
            "nd_percentile": float(percentile),
            "z_score": float(
                (nd_coef - np.mean(all_entity_coefs)) / np.std(all_entity_coefs)
            ),
            "interpretation": (
                f"ND's AR(1) coefficient is at the {percentile:.1f}th percentile "
                f"among all states."
            ),
        }

    return comparison
