#!/usr/bin/env python3
"""
Module 6: Machine Learning Agent - Elastic Net, Random Forest, and State Clustering
====================================================================================

Implements machine learning techniques for immigration analysis:
1. Elastic Net regularized regression to predict ND share using state characteristics
2. Random Forest feature importance (sample permitting)
3. State clustering by immigration profile - grouping states with similar patterns
4. Quality metrics: silhouette, Calinski-Harabasz, Davies-Bouldin

Usage:
    micromamba run -n cohort_proj python module_6_machine_learning.py
"""

import json
import sys
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Scikit-learn imports
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
}

CATEGORICAL = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#999999",
]

# Cluster colors for visualization
CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.warnings: list[str] = []
        self.decisions: list[dict] = []
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
        """Log a decision with full context."""
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
            "warnings": self.warnings,
            "decisions": self.decisions,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def setup_figure(figsize=(10, 8)):
    """Standard figure setup for all visualizations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    return fig, ax


def save_figure(fig, filepath_base, title, source_note):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold")
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
    print(f"Figure saved: {filepath_base}.png/pdf")


def load_data(result: ModuleResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load panel data and components of change data."""
    # Load panel data from Module 3.1
    panel_path = RESULTS_DIR / "module_3_1_panel_data.parquet"
    if panel_path.exists():
        df_panel = pd.read_parquet(panel_path)
        result.input_files.append("module_3_1_panel_data.parquet")
    else:
        # Load raw components of change
        df_panel = pd.read_csv(DATA_DIR / "combined_components_of_change.csv")
        result.input_files.append("combined_components_of_change.csv")

    # Load components of change for all states
    coc_path = DATA_DIR / "combined_components_of_change.csv"
    df_coc = pd.read_csv(coc_path)
    result.input_files.append("combined_components_of_change.csv")

    print(f"Loaded panel data: {df_panel.shape[0]} rows, {df_panel.shape[1]} columns")
    print(
        f"Loaded components of change: {df_coc.shape[0]} rows, {df_coc.shape[1]} columns"
    )

    return df_panel, df_coc


def load_previous_results(result: ModuleResult) -> dict:
    """Load results from Module 1.1 descriptive statistics."""
    prev_results = {}

    # Look for module 1.1 results
    for f in RESULTS_DIR.glob("module_1_1_*.json"):
        with open(f) as fp:
            prev_results[f.stem] = json.load(fp)
        result.input_files.append(f.name)

    print(f"Loaded {len(prev_results)} previous result files")
    return prev_results


def prepare_state_features(df_coc: pd.DataFrame, result: ModuleResult) -> pd.DataFrame:
    """
    Prepare state-level features for machine learning.

    Creates aggregated features from components of change data.
    """
    print("\n--- Preparing State Features ---")

    # Calculate US totals for share calculations
    us_totals = (
        df_coc.groupby("year")
        .agg(
            {
                "population": "sum",
                "intl_migration": "sum",
                "domestic_migration": "sum",
                "births": "sum",
                "deaths": "sum",
                "natural_change": "sum",
                "net_migration": "sum",
            }
        )
        .add_prefix("us_")
    )

    # Merge US totals back
    df = df_coc.merge(us_totals, on="year", how="left")

    # Calculate state share of US totals
    df["intl_migration_share"] = df["intl_migration"] / df["us_intl_migration"]
    df["domestic_migration_share"] = df["domestic_migration"].abs() / df[
        "us_domestic_migration"
    ].abs().replace(0, np.nan)
    df["population_share"] = df["population"] / df["us_population"]

    # Calculate rates per 1000 population
    df["birth_rate"] = df["births"] / df["population"] * 1000
    df["death_rate"] = df["deaths"] / df["population"] * 1000
    df["intl_mig_rate"] = df["intl_migration"] / df["population"] * 1000
    df["dom_mig_rate"] = df["domestic_migration"] / df["population"] * 1000
    df["growth_rate"] = df["pop_change"] / df["population"] * 1000

    # Aggregate to state level (mean across years, with std for variability)
    state_agg = df.groupby("state").agg(
        {
            # Mean values
            "population": ["mean", "std", "min", "max"],
            "intl_migration": ["mean", "std"],
            "domestic_migration": ["mean", "std"],
            "births": ["mean", "std"],
            "deaths": ["mean", "std"],
            "natural_change": ["mean", "std"],
            "net_migration": ["mean", "std"],
            "intl_migration_share": ["mean", "std"],
            "population_share": ["mean"],
            "birth_rate": ["mean", "std"],
            "death_rate": ["mean", "std"],
            "intl_mig_rate": ["mean", "std"],
            "dom_mig_rate": ["mean", "std"],
            "growth_rate": ["mean", "std"],
            "year": "count",  # Number of years
        }
    )

    # Flatten column names
    state_agg.columns = ["_".join(col).strip() for col in state_agg.columns.values]
    state_agg = state_agg.reset_index()

    # Rename columns for clarity
    state_agg = state_agg.rename(
        columns={
            "year_count": "n_years",
            "population_mean": "avg_population",
            "population_std": "population_volatility",
            "population_min": "min_population",
            "population_max": "max_population",
            "intl_migration_mean": "avg_intl_migration",
            "intl_migration_std": "intl_migration_volatility",
            "domestic_migration_mean": "avg_domestic_migration",
            "domestic_migration_std": "domestic_migration_volatility",
            "births_mean": "avg_births",
            "births_std": "births_volatility",
            "deaths_mean": "avg_deaths",
            "deaths_std": "deaths_volatility",
            "natural_change_mean": "avg_natural_change",
            "natural_change_std": "natural_change_volatility",
            "net_migration_mean": "avg_net_migration",
            "net_migration_std": "net_migration_volatility",
            "intl_migration_share_mean": "avg_intl_share",
            "intl_migration_share_std": "intl_share_volatility",
            "population_share_mean": "avg_pop_share",
            "birth_rate_mean": "avg_birth_rate",
            "birth_rate_std": "birth_rate_volatility",
            "death_rate_mean": "avg_death_rate",
            "death_rate_std": "death_rate_volatility",
            "intl_mig_rate_mean": "avg_intl_mig_rate",
            "intl_mig_rate_std": "intl_mig_rate_volatility",
            "dom_mig_rate_mean": "avg_dom_mig_rate",
            "dom_mig_rate_std": "dom_mig_rate_volatility",
            "growth_rate_mean": "avg_growth_rate",
            "growth_rate_std": "growth_rate_volatility",
        }
    )

    # Calculate additional derived features
    state_agg["pop_growth_percent"] = (
        (state_agg["max_population"] - state_agg["min_population"])
        / state_agg["min_population"]
        * 100
    )

    # Log transform for skewed variables
    state_agg["log_avg_population"] = np.log1p(state_agg["avg_population"])
    state_agg["log_avg_intl_migration"] = np.log1p(
        state_agg["avg_intl_migration"].clip(lower=0)
    )

    # Ratio features
    state_agg["intl_to_dom_ratio"] = state_agg["avg_intl_migration"] / (
        state_agg["avg_domestic_migration"].abs() + 1
    )
    state_agg["mig_to_natural_ratio"] = state_agg["avg_net_migration"] / (
        state_agg["avg_natural_change"].abs() + 1
    )

    result.add_decision(
        decision_id="D001",
        category="feature_engineering",
        decision="Created 30+ state-level aggregated features from panel data",
        rationale="Aggregating across years captures stable state characteristics for clustering",
        alternatives=["Use year-specific features", "Time-weighted aggregation"],
        evidence=f"Generated {len(state_agg.columns)} features for {len(state_agg)} states",
    )

    print(
        f"Created state features: {len(state_agg)} states, {len(state_agg.columns)} features"
    )
    return state_agg


def elastic_net_analysis(
    df_panel: pd.DataFrame, df_state: pd.DataFrame, result: ModuleResult
) -> dict:
    """
    Elastic Net regularized regression to predict ND share using state characteristics.

    Uses cross-validation to select optimal alpha and l1_ratio.
    """
    print("\n--- Elastic Net Regression Analysis ---")

    # Prepare data: predict intl_migration_share from state characteristics
    # First, get state-year level data
    df = df_panel.copy()

    # Calculate US totals for share
    us_totals = df.groupby("year")["intl_migration"].sum().reset_index()
    us_totals.columns = ["year", "us_intl_migration"]
    df = df.merge(us_totals, on="year")
    df["intl_share"] = df["intl_migration"] / df["us_intl_migration"]

    # Features for prediction

    # Add derived features
    df["birth_rate"] = df["births"] / df["population"] * 1000
    df["death_rate"] = df["deaths"] / df["population"] * 1000
    df["dom_mig_rate"] = df["domestic_migration"] / df["population"] * 1000
    df["log_population"] = np.log1p(df["population"])

    feature_cols_extended = [
        "log_population",
        "birth_rate",
        "death_rate",
        "dom_mig_rate",
        "natural_change",
    ]

    # Remove rows with missing values
    df_model = df.dropna(subset=feature_cols_extended + ["intl_share"]).copy()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model[feature_cols_extended])
    y = df_model["intl_share"].values

    # ElasticNetCV with multiple l1_ratios
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    alphas = np.logspace(-6, 1, 50)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    elastic_net_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv,
        random_state=42,
        max_iter=10000,
        tol=1e-4,
    )

    elastic_net_cv.fit(X, y)

    # Best parameters
    best_alpha = elastic_net_cv.alpha_
    best_l1_ratio = elastic_net_cv.l1_ratio_

    print(f"Optimal alpha: {best_alpha:.6f}")
    print(f"Optimal l1_ratio: {best_l1_ratio:.3f}")

    # Refit with best parameters for coefficient extraction
    elastic_net = ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42, max_iter=10000
    )
    elastic_net.fit(X, y)

    # Coefficients
    coef_dict = {}
    for i, feat in enumerate(feature_cols_extended):
        coef_dict[feat] = {
            "coefficient": float(elastic_net.coef_[i]),
            "abs_coefficient": float(abs(elastic_net.coef_[i])),
            "standardized": True,  # Features were scaled
        }

    # Sort by absolute value
    sorted_coefs = sorted(
        coef_dict.items(), key=lambda x: x[1]["abs_coefficient"], reverse=True
    )

    # Cross-validation scores
    cv_scores = cross_val_score(
        ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42),
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    # Predictions
    y_pred = elastic_net.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Count non-zero coefficients (sparsity)
    n_nonzero = np.sum(elastic_net.coef_ != 0)

    result.add_decision(
        decision_id="D002",
        category="model_selection",
        decision=f"Selected Elastic Net with alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.2f}",
        rationale="ElasticNetCV performs 5-fold cross-validation across alpha and l1_ratio grid",
        alternatives=["Pure Lasso (l1_ratio=1)", "Ridge (l1_ratio=0)", "Manual tuning"],
        evidence=f"CV R2 = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})",
    )

    elastic_results = {
        "model_type": "Elastic Net Regularized Regression",
        "dependent_variable": "intl_migration_share (state share of US international migration)",
        "n_observations": int(len(df_model)),
        "n_features": len(feature_cols_extended),
        "optimal_parameters": {
            "alpha": float(best_alpha),
            "l1_ratio": float(best_l1_ratio),
            "interpretation": {
                "alpha": "Regularization strength",
                "l1_ratio": f"{best_l1_ratio:.0%} L1 (Lasso) vs {1-best_l1_ratio:.0%} L2 (Ridge)",
            },
        },
        "coefficients": dict(sorted_coefs),
        "coefficient_ranking": [feat for feat, _ in sorted_coefs],
        "intercept": float(elastic_net.intercept_),
        "fit_statistics": {
            "r_squared": float(r2),
            "rmse": float(rmse),
            "n_nonzero_coefficients": int(n_nonzero),
            "sparsity_ratio": float(1 - n_nonzero / len(feature_cols_extended)),
        },
        "cross_validation": {
            "n_folds": 5,
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "cv_r2_scores": [float(s) for s in cv_scores],
        },
        "feature_importance_spss_style": {
            "table_title": "Elastic Net Regression Coefficients",
            "method": "Standardized coefficients (features scaled to mean=0, std=1)",
            "columns": ["Variable", "B (Standardized)", "Abs(B)", "Non-zero"],
            "rows": [
                {
                    "variable": feat,
                    "b_standardized": f"{data['coefficient']:.6f}",
                    "abs_b": f"{data['abs_coefficient']:.6f}",
                    "nonzero": "Yes" if data["coefficient"] != 0 else "No",
                }
                for feat, data in sorted_coefs
            ],
        },
    }

    return elastic_results, elastic_net, scaler, feature_cols_extended


def random_forest_importance(df_panel: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Random Forest feature importance analysis.

    Uses permutation importance and built-in impurity-based importance.
    """
    print("\n--- Random Forest Feature Importance ---")

    # Prepare data
    df = df_panel.copy()

    # Calculate US totals
    us_totals = df.groupby("year")["intl_migration"].sum().reset_index()
    us_totals.columns = ["year", "us_intl_migration"]
    df = df.merge(us_totals, on="year")
    df["intl_share"] = df["intl_migration"] / df["us_intl_migration"]

    # Features
    df["birth_rate"] = df["births"] / df["population"] * 1000
    df["death_rate"] = df["deaths"] / df["population"] * 1000
    df["dom_mig_rate"] = df["domestic_migration"] / df["population"] * 1000
    df["log_population"] = np.log1p(df["population"])
    df["natural_rate"] = df["natural_change"] / df["population"] * 1000

    feature_cols = [
        "log_population",
        "birth_rate",
        "death_rate",
        "dom_mig_rate",
        "natural_rate",
    ]

    # Remove missing
    df_model = df.dropna(subset=feature_cols + ["intl_share"]).copy()

    X = df_model[feature_cols].values
    y = df_model["intl_share"].values

    # Check sample size
    n_samples = len(X)
    if n_samples < 50:
        result.warnings.append(
            f"Small sample size ({n_samples}) may limit Random Forest reliability"
        )

    # Fit Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X, y)

    # Feature importance (impurity-based)
    importance_impurity = rf.feature_importances_

    # Permutation importance (more reliable)
    from sklearn.inspection import permutation_importance

    perm_importance = permutation_importance(
        rf, X, y, n_repeats=30, random_state=42, n_jobs=-1
    )

    # Build importance table
    importance_table = {}
    for i, feat in enumerate(feature_cols):
        importance_table[feat] = {
            "impurity_importance": float(importance_impurity[i]),
            "permutation_importance_mean": float(perm_importance.importances_mean[i]),
            "permutation_importance_std": float(perm_importance.importances_std[i]),
            "relative_importance_pct": float(
                importance_impurity[i] / importance_impurity.sum() * 100
            ),
        }

    # Sort by permutation importance
    sorted_importance = sorted(
        importance_table.items(),
        key=lambda x: x[1]["permutation_importance_mean"],
        reverse=True,
    )

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")

    # Predictions
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    result.add_decision(
        decision_id="D003",
        category="model_selection",
        decision="Used Random Forest with 500 trees, max_depth=10 for feature importance",
        rationale="RF provides non-linear importance measures; permutation importance is more reliable than impurity-based",
        alternatives=["Gradient Boosting", "XGBoost", "SHAP values"],
        evidence=f"OOB Score = {rf.oob_score_:.4f}, CV R2 = {cv_scores.mean():.4f}",
    )

    rf_results = {
        "model_type": "Random Forest Regressor",
        "dependent_variable": "intl_migration_share",
        "n_observations": int(n_samples),
        "n_features": len(feature_cols),
        "model_parameters": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "fit_statistics": {
            "r_squared": float(r2),
            "rmse": float(rmse),
            "oob_score": float(rf.oob_score_),
        },
        "cross_validation": {
            "n_folds": 5,
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
        },
        "feature_importance": dict(sorted_importance),
        "importance_ranking": [feat for feat, _ in sorted_importance],
        "feature_importance_spss_style": {
            "table_title": "Random Forest Feature Importance",
            "method": "Permutation importance (30 repeats)",
            "columns": [
                "Variable",
                "Impurity Importance",
                "Permutation Importance",
                "Std Dev",
                "Relative %",
            ],
            "rows": [
                {
                    "variable": feat,
                    "impurity": f"{data['impurity_importance']:.4f}",
                    "permutation": f"{data['permutation_importance_mean']:.4f}",
                    "std": f"{data['permutation_importance_std']:.4f}",
                    "relative_pct": f"{data['relative_importance_pct']:.1f}%",
                }
                for feat, data in sorted_importance
            ],
        },
    }

    return rf_results, rf, feature_cols


def state_clustering_analysis(
    df_state: pd.DataFrame, result: ModuleResult
) -> tuple[dict, pd.DataFrame]:
    """
    Cluster states by immigration profile using hierarchical and k-means clustering.

    Includes optimal K analysis using elbow and silhouette methods.
    """
    print("\n--- State Clustering Analysis ---")

    # Select clustering features
    cluster_features = [
        "avg_intl_share",
        "avg_intl_mig_rate",
        "avg_dom_mig_rate",
        "avg_birth_rate",
        "avg_death_rate",
        "avg_growth_rate",
        "log_avg_population",
        "intl_share_volatility",
    ]

    # Filter to available features
    available_features = [f for f in cluster_features if f in df_state.columns]

    if len(available_features) < 3:
        result.warnings.append(
            f"Only {len(available_features)} clustering features available"
        )
        # Use what we have
        available_features = [
            c
            for c in df_state.columns
            if df_state[c].dtype in ["float64", "int64"]
            and c not in ["state_fips", "n_years"]
        ][:8]

    print(f"Clustering features: {available_features}")

    # Prepare data
    df_cluster = df_state[["state"] + available_features].dropna().copy()
    states = df_cluster["state"].values
    X = df_cluster[available_features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_states = len(states)
    print(f"Clustering {n_states} states")

    # Optimal K analysis
    k_range = range(2, min(11, n_states - 1))
    inertias = []
    silhouettes = []
    calinski_scores = []
    davies_bouldin_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

    # Find optimal K (maximize silhouette)
    optimal_k_silhouette = list(k_range)[np.argmax(silhouettes)]
    optimal_k_calinski = list(k_range)[np.argmax(calinski_scores)]
    optimal_k_db = list(k_range)[np.argmin(davies_bouldin_scores)]

    # Elbow method: find "elbow" point
    # Use second derivative to find point of maximum curvature
    inertia_diff = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff)
    optimal_k_elbow = (
        list(k_range)[np.argmin(inertia_diff2) + 1] if len(inertia_diff2) > 0 else 3
    )

    # Choose optimal K (use silhouette as primary)
    optimal_k = optimal_k_silhouette
    print(f"Optimal K by silhouette: {optimal_k_silhouette}")
    print(f"Optimal K by Calinski-Harabasz: {optimal_k_calinski}")
    print(f"Optimal K by Davies-Bouldin: {optimal_k_db}")
    print(f"Optimal K by elbow: {optimal_k_elbow}")

    result.add_decision(
        decision_id="D004",
        category="clustering",
        decision=f"Selected K={optimal_k} clusters based on silhouette score",
        rationale="Silhouette score balances cluster cohesion and separation",
        alternatives=[
            f"Elbow method suggests K={optimal_k_elbow}",
            f"Calinski-Harabasz suggests K={optimal_k_calinski}",
        ],
        evidence=f"Silhouette score at K={optimal_k}: {max(silhouettes):.4f}",
    )

    # Final K-Means clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X_scaled)

    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    hierarchical_labels = (
        fcluster(linkage_matrix, t=optimal_k, criterion="maxclust") - 1
    )

    # Add cluster labels to dataframe
    df_cluster["kmeans_cluster"] = kmeans_labels
    df_cluster["hierarchical_cluster"] = hierarchical_labels

    # Cluster quality metrics (final K)
    quality_metrics = {
        "kmeans": {
            "silhouette_score": float(silhouette_score(X_scaled, kmeans_labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(X_scaled, kmeans_labels)
            ),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, kmeans_labels)),
            "inertia": float(kmeans_final.inertia_),
        },
        "hierarchical": {
            "silhouette_score": float(silhouette_score(X_scaled, hierarchical_labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(X_scaled, hierarchical_labels)
            ),
            "davies_bouldin": float(
                davies_bouldin_score(X_scaled, hierarchical_labels)
            ),
        },
    }

    # Cluster sizes
    kmeans_sizes = pd.Series(kmeans_labels).value_counts().sort_index().to_dict()
    hier_sizes = pd.Series(hierarchical_labels).value_counts().sort_index().to_dict()

    # Cluster centroids (in original scale)
    centroids_scaled = kmeans_final.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    centroid_df = pd.DataFrame(centroids_original, columns=available_features)
    centroid_df["cluster"] = range(optimal_k)

    # States by cluster
    states_by_cluster = {}
    for c in range(optimal_k):
        states_by_cluster[f"cluster_{c}"] = df_cluster[
            df_cluster["kmeans_cluster"] == c
        ]["state"].tolist()

    # Find ND's cluster
    nd_idx = df_cluster[df_cluster["state"] == "North Dakota"].index
    if len(nd_idx) > 0:
        nd_cluster = int(df_cluster.loc[nd_idx[0], "kmeans_cluster"])
        nd_peers = [
            s for s in states_by_cluster[f"cluster_{nd_cluster}"] if s != "North Dakota"
        ]
    else:
        nd_cluster = None
        nd_peers = []

    # Cluster characteristics
    cluster_profiles = {}
    for c in range(optimal_k):
        cluster_data = df_cluster[df_cluster["kmeans_cluster"] == c][available_features]
        cluster_profiles[f"cluster_{c}"] = {
            feat: {
                "mean": float(cluster_data[feat].mean()),
                "std": float(cluster_data[feat].std()),
                "min": float(cluster_data[feat].min()),
                "max": float(cluster_data[feat].max()),
            }
            for feat in available_features
        }

    # PCA for visualization (2D projection)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_cluster["pca_1"] = X_pca[:, 0]
    df_cluster["pca_2"] = X_pca[:, 1]

    explained_variance = pca.explained_variance_ratio_

    cluster_results = {
        "method": {
            "primary": "K-Means Clustering",
            "secondary": "Hierarchical Agglomerative Clustering (Ward linkage)",
        },
        "n_states": int(n_states),
        "n_features": len(available_features),
        "features_used": available_features,
        "optimal_k_analysis": {
            "k_range_tested": list(k_range),
            "inertias": [float(i) for i in inertias],
            "silhouette_scores": [float(s) for s in silhouettes],
            "calinski_harabasz_scores": [float(c) for c in calinski_scores],
            "davies_bouldin_scores": [float(d) for d in davies_bouldin_scores],
            "optimal_k_by_method": {
                "silhouette": int(optimal_k_silhouette),
                "calinski_harabasz": int(optimal_k_calinski),
                "davies_bouldin": int(optimal_k_db),
                "elbow": int(optimal_k_elbow),
            },
            "selected_k": int(optimal_k),
            "selection_criterion": "Maximum silhouette score",
        },
        "quality_metrics": quality_metrics,
        "cluster_sizes": {
            "kmeans": {str(k): int(v) for k, v in kmeans_sizes.items()},
            "hierarchical": {str(k): int(v) for k, v in hier_sizes.items()},
        },
        "states_by_cluster": states_by_cluster,
        "cluster_centroids": centroid_df.to_dict("records"),
        "cluster_profiles": cluster_profiles,
        "north_dakota": {
            "cluster": nd_cluster,
            "peer_states": nd_peers,
            "n_peers": len(nd_peers),
        },
        "pca_projection": {
            "explained_variance_ratio": [float(v) for v in explained_variance],
            "total_variance_explained": float(sum(explained_variance)),
        },
        "spss_style_output": {
            "cluster_method": "K-Means with Ward Hierarchical Comparison",
            "distance_metric": "Euclidean (on standardized features)",
            "linkage_method": "Ward (minimum variance)",
            "final_cluster_centers": centroid_df.to_dict("records"),
            "cluster_membership": {str(k): v for k, v in states_by_cluster.items()},
        },
    }

    return cluster_results, df_cluster, linkage_matrix, X_pca, pca


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    states: np.ndarray,
    n_clusters: int,
    result: ModuleResult,
):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Color threshold for n_clusters
    max_d = 0.7 * max(linkage_matrix[:, 2])

    dendrogram(
        linkage_matrix,
        labels=states,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=max_d,
        above_threshold_color=COLORS["neutral"],
    )

    ax.axhline(y=max_d, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax.text(
        ax.get_xlim()[1] * 0.02,
        max_d * 1.05,
        f"Cut for {n_clusters} clusters",
        fontsize=10,
        color=COLORS["secondary"],
    )

    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Ward Distance", fontsize=12)
    ax.set_title("State Immigration Profile Dendrogram", fontsize=12)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_6_cluster_dendrogram"),
        "Hierarchical Clustering of States by Immigration Profile",
        "Census Bureau Components of Change (2010-2024)",
    )


def plot_elbow(k_range, inertias, silhouettes, optimal_k, result: ModuleResult):
    """Plot elbow curve and silhouette scores for optimal K selection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Elbow plot
    ax1 = axes[0]
    ax1.plot(
        list(k_range),
        inertias,
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        markersize=8,
    )
    ax1.axvline(optimal_k, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax1.text(
        optimal_k + 0.1,
        ax1.get_ylim()[1] * 0.9,
        f"Optimal K={optimal_k}",
        fontsize=10,
        color=COLORS["secondary"],
    )

    ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2 = axes[1]
    ax2.plot(
        list(k_range),
        silhouettes,
        "o-",
        color=COLORS["tertiary"],
        linewidth=2,
        markersize=8,
    )
    ax2.axvline(optimal_k, color=COLORS["secondary"], linestyle="--", linewidth=2)

    # Mark maximum
    max_idx = np.argmax(silhouettes)
    ax2.scatter(
        [list(k_range)[max_idx]],
        [silhouettes[max_idx]],
        s=200,
        color=COLORS["secondary"],
        zorder=5,
        edgecolor="black",
        linewidth=2,
    )
    ax2.annotate(
        f"Max = {silhouettes[max_idx]:.3f}",
        (list(k_range)[max_idx], silhouettes[max_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
    )

    ax2.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Analysis", fontsize=12)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_6_elbow_plot"),
        "Optimal Number of Clusters Analysis",
        "Census Bureau Components of Change (2010-2024)",
    )


def plot_silhouette_by_cluster(
    X_scaled: np.ndarray, labels: np.ndarray, n_clusters: int, result: ModuleResult
):
    """Plot silhouette scores by cluster."""
    from sklearn.metrics import silhouette_samples

    fig, ax = plt.subplots(figsize=(10, 8))

    silhouette_vals = silhouette_samples(X_scaled, labels)
    y_lower = 10

    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        cluster_size = len(cluster_silhouette_vals)
        y_upper = y_lower + cluster_size

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i), fontsize=12)
        y_lower = y_upper + 10

    avg_silhouette = silhouette_vals.mean()
    ax.axvline(avg_silhouette, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax.text(
        avg_silhouette + 0.02,
        ax.get_ylim()[1] * 0.95,
        f"Avg = {avg_silhouette:.3f}",
        fontsize=10,
        color=COLORS["secondary"],
    )

    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title(f"Silhouette Plot for K={n_clusters} Clusters", fontsize=12)
    ax.set_yticks([])

    save_figure(
        fig,
        str(FIGURES_DIR / "module_6_silhouette_plot"),
        f"Silhouette Analysis by Cluster (K={n_clusters})",
        "Census Bureau Components of Change (2010-2024)",
    )


def plot_pca_biplot(
    X_pca: np.ndarray,
    labels: np.ndarray,
    states: np.ndarray,
    pca: PCA,
    feature_names: list,
    n_clusters: int,
    nd_cluster: int,
    result: ModuleResult,
):
    """Plot PCA biplot with cluster coloring and feature loadings."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot with cluster colors
    for c in range(n_clusters):
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            s=100,
            alpha=0.7,
            label=f"Cluster {c}",
            edgecolor="white",
            linewidth=1,
        )

    # Label states
    for i, state in enumerate(states):
        # Abbreviate state names for clarity
        abbrev = state[:2] if len(state) > 2 else state
        if state == "North Dakota":
            ax.annotate(
                "ND",
                (X_pca[i, 0], X_pca[i, 1]),
                fontsize=10,
                fontweight="bold",
                color=COLORS["secondary"],
                xytext=(5, 5),
                textcoords="offset points",
            )
            ax.scatter(
                [X_pca[i, 0]],
                [X_pca[i, 1]],
                s=200,
                color=COLORS["secondary"],
                marker="*",
                zorder=10,
                edgecolor="black",
            )
        else:
            ax.annotate(
                abbrev,
                (X_pca[i, 0], X_pca[i, 1]),
                fontsize=7,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # Add feature loading arrows (biplot)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    scale_factor = 3  # Scale for visibility

    for i, feat in enumerate(feature_names):
        ax.arrow(
            0,
            0,
            loadings[i, 0] * scale_factor,
            loadings[i, 1] * scale_factor,
            head_width=0.1,
            head_length=0.05,
            fc=COLORS["neutral"],
            ec=COLORS["neutral"],
            alpha=0.8,
        )
        # Shorten feature names
        feat_short = feat.replace("avg_", "").replace("_", " ")[:12]
        ax.annotate(
            feat_short,
            (loadings[i, 0] * scale_factor * 1.1, loadings[i, 1] * scale_factor * 1.1),
            fontsize=9,
            color=COLORS["neutral"],
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    variance_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)", fontsize=12)
    ax.set_title("State Clustering: PCA Biplot", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_6_pca_biplot"),
        f"PCA Biplot: State Immigration Profiles (K={n_clusters} Clusters)",
        "Census Bureau Components of Change (2010-2024)",
    )


def plot_feature_importance(
    rf_results: dict, elastic_results: dict, result: ModuleResult
):
    """Plot feature importance comparison between Elastic Net and Random Forest."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Elastic Net coefficients
    ax1 = axes[0]
    en_features = list(elastic_results["coefficients"].keys())
    en_coefs = [
        elastic_results["coefficients"][f]["abs_coefficient"] for f in en_features
    ]

    y_pos = np.arange(len(en_features))
    colors = [
        COLORS["primary"] if c > 0 else COLORS["secondary"]
        for c in [
            elastic_results["coefficients"][f]["coefficient"] for f in en_features
        ]
    ]

    ax1.barh(y_pos, en_coefs, color=colors, alpha=0.7, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f.replace("_", " ").title() for f in en_features], fontsize=10)
    ax1.set_xlabel("Absolute Coefficient (Standardized)", fontsize=12)
    ax1.set_title("Elastic Net Coefficients", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="x")

    # Random Forest importance
    ax2 = axes[1]
    rf_features = list(rf_results["feature_importance"].keys())
    rf_importance = [
        rf_results["feature_importance"][f]["permutation_importance_mean"]
        for f in rf_features
    ]
    rf_std = [
        rf_results["feature_importance"][f]["permutation_importance_std"]
        for f in rf_features
    ]

    y_pos = np.arange(len(rf_features))
    ax2.barh(
        y_pos,
        rf_importance,
        xerr=rf_std,
        color=COLORS["tertiary"],
        alpha=0.7,
        edgecolor="white",
        capsize=3,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f.replace("_", " ").title() for f in rf_features], fontsize=10)
    ax2.set_xlabel("Permutation Importance", fontsize=12)
    ax2.set_title("Random Forest Importance", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")

    save_figure(
        fig,
        str(FIGURES_DIR / "module_6_feature_importance"),
        "Feature Importance: Elastic Net vs Random Forest",
        "Census Bureau Components of Change (2010-2024)",
    )


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 6."""
    result = ModuleResult(
        module_id="6", analysis_name="machine_learning_elastic_net_rf_clustering"
    )

    print("Loading data...")
    df_panel, df_coc = load_data(result)

    print("\nLoading previous results...")
    load_previous_results(result)

    print("\nPreparing state features...")
    df_state = prepare_state_features(df_coc, result)

    # Record parameters
    result.parameters = {
        "panel_observations": int(len(df_panel)),
        "n_states": int(df_state["state"].nunique()),
        "years_covered": sorted(df_coc["year"].unique().tolist()),
        "models": {
            "elastic_net": {
                "cv_folds": 5,
                "l1_ratios_tested": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                "alpha_range": "logspace(-6, 1, 50)",
            },
            "random_forest": {
                "n_estimators": 500,
                "max_depth": 10,
                "permutation_repeats": 30,
            },
            "clustering": {
                "methods": ["K-Means", "Hierarchical (Ward)"],
                "k_range": "2-10",
                "optimization_criterion": "Silhouette score",
            },
        },
    }

    # Elastic Net analysis
    print("\n" + "=" * 60)
    print("ELASTIC NET REGRESSION")
    print("=" * 60)
    elastic_results, elastic_model, scaler, feature_cols = elastic_net_analysis(
        df_panel, df_state, result
    )

    # Save Elastic Net results
    en_output = RESULTS_DIR / "module_6_elastic_net.json"
    with open(en_output, "w") as f:
        json.dump(elastic_results, f, indent=2, default=str)
    print(f"Elastic Net results saved: {en_output}")

    # Random Forest analysis
    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)
    rf_results, rf_model, rf_features = random_forest_importance(df_panel, result)

    # Save RF results
    rf_output = RESULTS_DIR / "module_6_feature_importance.json"
    with open(rf_output, "w") as f:
        json.dump(rf_results, f, indent=2, default=str)
    print(f"Random Forest results saved: {rf_output}")

    # State clustering
    print("\n" + "=" * 60)
    print("STATE CLUSTERING ANALYSIS")
    print("=" * 60)
    cluster_results, df_clusters, linkage_mat, X_pca, pca = state_clustering_analysis(
        df_state, result
    )

    # Save cluster results to JSON
    cluster_json_output = RESULTS_DIR / "module_6_state_clusters.json"
    with open(cluster_json_output, "w") as f:
        json.dump(cluster_results, f, indent=2, default=str)
    print(f"Clustering results saved: {cluster_json_output}")

    # Save cluster assignments to parquet
    cluster_parquet_output = RESULTS_DIR / "module_6_state_clusters.parquet"
    df_clusters.to_parquet(cluster_parquet_output, index=False)
    print(f"Cluster assignments saved: {cluster_parquet_output}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Get clustering parameters
    optimal_k = cluster_results["optimal_k_analysis"]["selected_k"]
    k_range = cluster_results["optimal_k_analysis"]["k_range_tested"]
    inertias = cluster_results["optimal_k_analysis"]["inertias"]
    silhouettes = cluster_results["optimal_k_analysis"]["silhouette_scores"]

    # Dendrogram
    plot_dendrogram(linkage_mat, df_clusters["state"].values, optimal_k, result)

    # Elbow and silhouette plots
    plot_elbow(k_range, inertias, silhouettes, optimal_k, result)

    # Silhouette by cluster
    features_used = cluster_results["features_used"]
    X_scaled = StandardScaler().fit_transform(df_clusters[features_used])
    plot_silhouette_by_cluster(
        X_scaled, df_clusters["kmeans_cluster"].values, optimal_k, result
    )

    # PCA biplot
    nd_cluster = cluster_results["north_dakota"]["cluster"]
    plot_pca_biplot(
        X_pca,
        df_clusters["kmeans_cluster"].values,
        df_clusters["state"].values,
        pca,
        features_used,
        optimal_k,
        nd_cluster,
        result,
    )

    # Feature importance comparison
    plot_feature_importance(rf_results, elastic_results, result)

    # Compile main results
    result.results = {
        "elastic_net": {
            "optimal_alpha": elastic_results["optimal_parameters"]["alpha"],
            "optimal_l1_ratio": elastic_results["optimal_parameters"]["l1_ratio"],
            "r_squared": elastic_results["fit_statistics"]["r_squared"],
            "cv_r2": elastic_results["cross_validation"]["cv_r2_mean"],
            "top_features": elastic_results["coefficient_ranking"][:3],
            "n_nonzero_coefficients": elastic_results["fit_statistics"][
                "n_nonzero_coefficients"
            ],
        },
        "random_forest": {
            "oob_score": rf_results["fit_statistics"]["oob_score"],
            "cv_r2": rf_results["cross_validation"]["cv_r2_mean"],
            "top_features_by_importance": rf_results["importance_ranking"][:3],
        },
        "clustering": {
            "optimal_k": optimal_k,
            "silhouette_score": cluster_results["quality_metrics"]["kmeans"][
                "silhouette_score"
            ],
            "calinski_harabasz": cluster_results["quality_metrics"]["kmeans"][
                "calinski_harabasz"
            ],
            "davies_bouldin": cluster_results["quality_metrics"]["kmeans"][
                "davies_bouldin"
            ],
            "cluster_sizes": cluster_results["cluster_sizes"]["kmeans"],
            "nd_cluster": nd_cluster,
            "nd_peer_states": cluster_results["north_dakota"]["peer_states"],
        },
    }

    # Diagnostics
    result.diagnostics = {
        "elastic_net": {
            "cv_scores": elastic_results["cross_validation"]["cv_r2_scores"],
            "sparsity": elastic_results["fit_statistics"]["sparsity_ratio"],
        },
        "random_forest": {
            "n_estimators": rf_results["model_parameters"]["n_estimators"],
            "max_depth": rf_results["model_parameters"]["max_depth"],
        },
        "clustering": {
            "pca_variance_explained": cluster_results["pca_projection"][
                "total_variance_explained"
            ],
            "optimal_k_by_method": cluster_results["optimal_k_analysis"][
                "optimal_k_by_method"
            ],
        },
    }

    # Next steps
    result.next_steps = [
        "Use cluster membership for stratified analysis in Module 7",
        "Compare ND with peer states identified in same cluster",
        "Apply feature importance insights to projection model refinement",
        "Consider ensemble of Elastic Net and RF for prediction",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 6: Machine Learning - Elastic Net, Random Forest, Clustering")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_6_machine_learning.json")

        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\nKey Results:")
        print(
            f"  Elastic Net: alpha={result.results['elastic_net']['optimal_alpha']:.4f}, "
            f"l1_ratio={result.results['elastic_net']['optimal_l1_ratio']:.2f}, "
            f"CV R2={result.results['elastic_net']['cv_r2']:.4f}"
        )
        print(
            f"  Random Forest: OOB={result.results['random_forest']['oob_score']:.4f}, "
            f"CV R2={result.results['random_forest']['cv_r2']:.4f}"
        )
        print(
            f"  Clustering: K={result.results['clustering']['optimal_k']}, "
            f"Silhouette={result.results['clustering']['silhouette_score']:.4f}"
        )
        print(f"  ND Cluster: {result.results['clustering']['nd_cluster']}")
        print(f"  ND Peers: {result.results['clustering']['nd_peer_states']}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision']}")

        print("\nOutput files generated:")
        print("  - module_6_elastic_net.json")
        print("  - module_6_feature_importance.json")
        print("  - module_6_state_clusters.json")
        print("  - module_6_state_clusters.parquet")
        print("  - module_6_cluster_dendrogram.png/pdf")
        print("  - module_6_elbow_plot.png/pdf")
        print("  - module_6_silhouette_plot.png/pdf")
        print("  - module_6_pca_biplot.png/pdf")
        print("  - module_6_feature_importance.png/pdf")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
