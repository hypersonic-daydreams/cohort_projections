"""Core metric computation functions for projection evaluation.

All functions accept numpy arrays or pandas Series and return scalar floats.
They are designed to be composed into higher-level evaluation routines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Absolute-error family
# ---------------------------------------------------------------------------

def mae(projected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(projected - actual)))


def rmse(projected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((projected - actual) ** 2)))


# ---------------------------------------------------------------------------
# Percentage-error family
# ---------------------------------------------------------------------------

def mape(projected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    """Mean Absolute Percentage Error (×100, in percentage points)."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((projected[mask] - actual[mask]) / actual[mask])) * 100)


def median_absolute_percentage_error(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> float:
    """Median Absolute Percentage Error (×100, in percentage points)."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs((projected[mask] - actual[mask]) / actual[mask])) * 100)


def wape(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> float:
    """Weighted Absolute Percentage Error (×100).

    If *weights* is ``None``, uses ``|actual|`` as weights (population-weighted).
    """
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    if weights is None:
        weights = np.abs(actual)
    else:
        weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    if total_weight == 0:
        return float("nan")
    return float(np.sum(weights * np.abs(projected - actual) / np.where(actual == 0, 1, np.abs(actual))) / total_weight * 100)


# ---------------------------------------------------------------------------
# Signed-error / bias family
# ---------------------------------------------------------------------------

def mean_signed_error(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> float:
    """Mean Signed Error (positive = overprojection)."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    return float(np.mean(projected - actual))


def mean_signed_percentage_error(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> float:
    """Mean Signed Percentage Error (×100, positive = overprojection)."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean((projected[mask] - actual[mask]) / actual[mask]) * 100)


# ---------------------------------------------------------------------------
# Rank / direction tests
# ---------------------------------------------------------------------------

def spearman_rank_correlation(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> float:
    """Spearman rank correlation of projected vs actual growth."""
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    if len(projected) < 3:
        return float("nan")
    corr, _ = scipy_stats.spearmanr(projected, actual)
    return float(corr)


def directional_accuracy(
    projected_growth: np.ndarray | pd.Series,
    actual_growth: np.ndarray | pd.Series,
) -> float:
    """Fraction of geographies where projected growth direction matches actual."""
    pg = np.asarray(projected_growth, dtype=float)
    ag = np.asarray(actual_growth, dtype=float)
    if len(pg) == 0:
        return float("nan")
    return float(np.mean(np.sign(pg) == np.sign(ag)))


def decile_capture(
    projected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    quantile: float = 0.1,
    tail: str = "bottom",
) -> float:
    """Fraction of actual top/bottom-decile counties captured by projected ranking.

    Parameters
    ----------
    quantile : float
        Quantile threshold (0.1 = decile).
    tail : str
        ``"bottom"`` for lowest-growth, ``"top"`` for highest-growth.
    """
    projected, actual = np.asarray(projected, dtype=float), np.asarray(actual, dtype=float)
    n = len(actual)
    k = max(1, int(n * quantile))
    if tail == "bottom":
        actual_set = set(np.argsort(actual)[:k])
        projected_set = set(np.argsort(projected)[:k])
    else:
        actual_set = set(np.argsort(actual)[-k:])
        projected_set = set(np.argsort(projected)[-k:])
    return float(len(actual_set & projected_set) / k)


# ---------------------------------------------------------------------------
# Distributional divergence
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(
    p: np.ndarray | pd.Series,
    q: np.ndarray | pd.Series,
) -> float:
    """Jensen-Shannon divergence between two distributions.

    Inputs are normalised internally to sum to 1.  A small epsilon is added
    to avoid log(0).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def kullback_leibler_divergence(
    p: np.ndarray | pd.Series,
    q: np.ndarray | pd.Series,
) -> float:
    """KL divergence KL(p || q).

    A small epsilon is added to avoid log(0).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))
