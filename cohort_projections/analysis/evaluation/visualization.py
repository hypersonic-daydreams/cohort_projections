"""Reporting and diagnostic visualization for projection evaluation.

Module 5 of the Evaluation Blueprint.  Every public function accepts
DataFrames (or scalars) and returns a ``matplotlib.figure.Figure``.
The caller is responsible for saving or displaying figures.

Requires matplotlib; seaborn is optional for enhanced styling.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Guarded matplotlib / seaborn imports
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = object  # type: ignore[assignment,misc]
    warnings.warn(
        "matplotlib not available - evaluation visualizations disabled",
        stacklevel=2,
    )

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_DEFAULT_FIGSIZE = (10, 6)
_DPI = 150


def _apply_style() -> None:
    """Apply consistent plot styling."""
    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid", palette="colorblind")
    elif MATPLOTLIB_AVAILABLE:
        plt.style.use("seaborn-v0_8-whitegrid")


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for evaluation visualizations")


# ---------------------------------------------------------------------------
# 1. Horizon profile
# ---------------------------------------------------------------------------


def plot_horizon_profile(
    diagnostics_df: pd.DataFrame,
    metric: str,
    methods: list[str] | None = None,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Error metric vs. forecast horizon, one line per model.

    Args:
        diagnostics_df: DataFrame with columns ``run_id``, ``model_name``,
            ``horizon``, ``metric_name``, ``value``.
        metric: Metric to plot (e.g. ``"mape"``).
        methods: Subset of ``model_name`` values to include.  ``None`` = all.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = diagnostics_df.loc[diagnostics_df["metric_name"] == metric].copy()
    if methods is not None:
        df = df.loc[df["model_name"].isin(methods)]

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    for name, grp in df.groupby("model_name"):
        agg = grp.groupby("horizon")["value"].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=str(name))

    ax.set_xlabel("Forecast Horizon (years)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Forecast Horizon")
    ax.legend(title="Model")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. County-horizon heatmap
# ---------------------------------------------------------------------------


def plot_county_horizon_heatmap(
    diagnostics_df: pd.DataFrame,
    metric: str,
    *,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Heatmap of *metric* with counties on y-axis, horizons on x-axis.

    Args:
        diagnostics_df: DataFrame with ``geography``, ``horizon``,
            ``metric_name``, ``value``.
        metric: Metric to visualise.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = diagnostics_df.loc[diagnostics_df["metric_name"] == metric].copy()
    pivot = df.pivot_table(
        index="geography", columns="horizon", values="value", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    if SEABORN_AVAILABLE:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax)

    ax.set_title(f"{metric.upper()} by County and Horizon")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Geography")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Bias map
# ---------------------------------------------------------------------------


def plot_bias_map(
    diagnostics_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Horizontal bar chart of signed error by county.

    Positive bars indicate overprojection; negative bars indicate
    underprojection.

    Args:
        diagnostics_df: DataFrame with ``geography``,
            ``metric_name`` (must contain ``"mean_signed_percentage_error"``),
            ``value``.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = diagnostics_df.loc[
        diagnostics_df["metric_name"] == "mean_signed_percentage_error"
    ].copy()
    agg = df.groupby("geography")["value"].mean().sort_values()

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in agg.values]
    ax.barh(agg.index.astype(str), agg.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Signed Percentage Error")
    ax.set_title("Projection Bias by County (positive = overprojection)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Component blame
# ---------------------------------------------------------------------------


def plot_component_blame(
    component_diagnostics_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Stacked bar chart showing component contributions to total error.

    Args:
        component_diagnostics_df: DataFrame with ``component``,
            ``horizon``, ``projected_component_value``,
            ``actual_component_value``.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = component_diagnostics_df.copy()
    df["error"] = df["projected_component_value"] - df["actual_component_value"]
    pivot = df.pivot_table(
        index="horizon", columns="component", values="error", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Mean Error (projected - actual)")
    ax.set_title("Component Contributions to Projection Error")
    ax.legend(title="Component")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Age divergence
# ---------------------------------------------------------------------------


def plot_age_divergence(
    diagnostics_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Jensen-Shannon divergence of age distribution over forecast horizons.

    Args:
        diagnostics_df: DataFrame with ``metric_name`` (containing ``"jsd"``
            rows), ``horizon``, ``geography``, ``value``.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = diagnostics_df.loc[diagnostics_df["metric_name"] == "jsd"].copy()
    agg = df.groupby("horizon")["value"].agg(["mean", "min", "max"]).sort_index()

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    ax.plot(agg.index, agg["mean"], marker="o", label="Mean JSD")
    ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.2, label="Min-Max range")
    ax.set_xlabel("Forecast Horizon (years)")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("Age-Distribution Divergence over Horizon")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Parameter response
# ---------------------------------------------------------------------------


def plot_parameter_response(
    sweep_df: pd.DataFrame,
    param_name: str,
    metric: str,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Metric value vs. parameter setting from a sensitivity sweep.

    Args:
        sweep_df: DataFrame with a column matching *param_name* and a
            column matching *metric*.
        param_name: Name of the swept parameter.
        metric: Metric column name.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = sweep_df.sort_values(param_name)

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    ax.plot(df[param_name], df[metric], marker="s", linewidth=2)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. {param_name}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Stability scatter
# ---------------------------------------------------------------------------


def plot_stability_scatter(
    diagnostics_df: pd.DataFrame,
    *,
    near_term_max_horizon: int = 5,
    long_term_min_horizon: int = 10,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Figure:
    """Near-term vs. long-term accuracy scatter, marker size = robustness.

    Each point represents one model/run.

    Args:
        diagnostics_df: DataFrame with ``run_id``, ``model_name``,
            ``metric_name`` (must contain ``"mape"``), ``horizon``, ``value``.
        near_term_max_horizon: Maximum horizon considered near-term.
        long_term_min_horizon: Minimum horizon considered long-term.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    _require_matplotlib()
    _apply_style()

    df = diagnostics_df.loc[diagnostics_df["metric_name"] == "mape"].copy()

    near = df.loc[df["horizon"] <= near_term_max_horizon]
    far = df.loc[df["horizon"] >= long_term_min_horizon]

    near_agg = near.groupby("model_name")["value"].mean()
    far_agg = far.groupby("model_name")["value"].mean()
    # Robustness: inverse of standard deviation across all horizons
    std_agg = df.groupby("model_name")["value"].std().replace(0, np.nan)
    robustness = (1.0 / std_agg).fillna(1.0)
    robustness = robustness / robustness.max() * 200 + 50  # scale for marker size

    models = near_agg.index.intersection(far_agg.index)

    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    for model in models:
        ax.scatter(
            near_agg[model],
            far_agg[model],
            s=robustness.get(model, 100),
            label=str(model),
            alpha=0.7,
        )
    ax.set_xlabel("Near-term MAPE (%)")
    ax.set_ylabel("Long-term MAPE (%)")
    ax.set_title("Near-term vs. Long-term Accuracy (size = robustness)")
    ax.legend(title="Model")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def save_evaluation_report(
    output_dir: str | Path,
    diagnostics_df: pd.DataFrame,
    figures: dict[str, Figure],
) -> Path:
    """Save all figures and a summary CSV to *output_dir*.

    Args:
        output_dir: Directory to write outputs into (created if needed).
        diagnostics_df: Diagnostics table to save as CSV.
        figures: Mapping of ``{name: Figure}`` to save as PNG files.

    Returns:
        Path to the output directory.
    """
    _require_matplotlib()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    diagnostics_df.to_csv(out / "diagnostics_summary.csv", index=False)
    logger.info("Saved diagnostics summary to %s", out / "diagnostics_summary.csv")

    for name, fig in figures.items():
        filepath = out / f"{name}.png"
        fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure %s", filepath)

    return out
